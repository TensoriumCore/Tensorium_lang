#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Sema/CallSupport.hpp"
#include "tensorium/Sema/tensor_type_checker.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tensorium {

static std::unique_ptr<IndexedVar>
makeVar(const std::string &name, const std::vector<std::string> &indices,
        const std::unordered_map<std::string, const FieldDecl *> &fields,
        TensorTypeChecker &checker) {
  auto it = fields.find(name);
  if (it == fields.end())
    throw std::runtime_error("Metric field not found: " + name);
  const FieldDecl *fd = it->second;

  auto v = std::make_unique<IndexedVar>(name, IndexedVarKind::Field);
  v->tensorKind = fd->kind;
  v->up = fd->up;
  v->down = fd->down;
  v->tensorIndexNames = indices;

  for (size_t i = 0; i < indices.size(); ++i) {
    bool isUp = static_cast<int>(i) < fd->up;
    v->tensorIndexIsUp.push_back(isUp);
    v->tensorIndexOffsets.push_back(0);
  }
  checker.infer(v.get());
  return v;
}

static std::unique_ptr<IndexedCall> makeDeriv(const std::string &idx,
                                              std::unique_ptr<IndexedExpr> arg,
                                              TensorTypeChecker &checker) {
  auto call = std::make_unique<IndexedCall>();
  call->callee = "d_" + idx;
  call->args.push_back(std::move(arg));
  checker.infer(call.get());
  return call;
}

static std::unique_ptr<IndexedCall>
makeContract(std::unique_ptr<IndexedExpr> arg, TensorTypeChecker &checker) {
  auto call = std::make_unique<IndexedCall>();
  call->callee = "contract";
  call->args.push_back(std::move(arg));
  call->isExtern = false;
  checker.infer(call.get());
  return call;
}

static bool isSpatialIndexName(const std::string &idx) {
  return idx.size() == 1 && core::isSpatialIndexChar(idx[0]);
}

static void collectExprIndexNames(const IndexedExpr *e,
                                  std::unordered_set<std::string> &out) {
  if (!e)
    return;

  if (auto *v = dynamic_cast<const IndexedVar *>(e)) {
    for (const auto &idx : v->tensorIndexNames)
      if (!idx.empty())
        out.insert(idx);
    return;
  }

  if (auto *b = dynamic_cast<const IndexedBinary *>(e)) {
    collectExprIndexNames(b->lhs.get(), out);
    collectExprIndexNames(b->rhs.get(), out);
    return;
  }

  if (auto *c = dynamic_cast<const IndexedCall *>(e)) {
    if (c->callee.size() == 3 && c->callee[0] == 'd' && c->callee[1] == '_') {
      std::string idx(1, c->callee[2]);
      if (isSpatialIndexName(idx))
        out.insert(idx);
    }
    if (c->callee.size() == 7 &&
        (c->callee.rfind("nabla_", 0) == 0 || c->callee.rfind("nabla^", 0) == 0)) {
      std::string idx(1, c->callee[6]);
      if (isSpatialIndexName(idx))
        out.insert(idx);
    }
    for (const auto &arg : c->args)
      collectExprIndexNames(arg.get(), out);
  }
}

static std::string chooseFreshSpatialIndex(
    const std::unordered_set<std::string> &used, const std::string &context) {
  for (const char c : {'i', 'j', 'k', 'l', 'm', 'n'}) {
    std::string idx(1, c);
    if (!used.count(idx))
      return idx;
  }
  throw std::runtime_error("nabla expansion ran out of fresh spatial indices in " +
                           context);
}

static std::pair<std::string, std::string> resolveMetricNames(
    const std::unordered_map<std::string, const FieldDecl *> &fields) {
  std::string metricName;
  std::string inverseMetricName;
  int metricCount = 0;
  int inverseCount = 0;

  for (const auto &kv : fields) {
    if (kv.second->isMetric) {
      metricName = kv.first;
      ++metricCount;
    }
    if (kv.second->isInverseMetric) {
      inverseMetricName = kv.first;
      ++inverseCount;
    }
  }

  if (metricCount > 1) {
    throw std::runtime_error(
        "nabla expansion requires a unique metric field (multiple declared)");
  }
  if (inverseCount > 1) {
    throw std::runtime_error(
        "nabla expansion requires a unique inverse_metric field (multiple declared)");
  }
  return {metricName, inverseMetricName};
}

static std::string resolveConnectionFieldName(
    const std::unordered_map<std::string, const FieldDecl *> &fields) {
  auto isConnection = [](const FieldDecl *fd) {
    return fd && fd->up == 1 && fd->down == 2;
  };

  auto pickNamed = [&](const std::string &name) -> std::string {
    auto it = fields.find(name);
    if (it == fields.end())
      return {};
    if (!isConnection(it->second))
      return {};
    return it->first;
  };

  if (auto preferred = pickNamed("Christoffel"); !preferred.empty())
    return preferred;
  if (auto preferred = pickNamed("Gamma"); !preferred.empty())
    return preferred;

  std::string candidate;
  int count = 0;
  for (const auto &kv : fields) {
    if (!isConnection(kv.second))
      continue;
    candidate = kv.first;
    ++count;
  }

  if (count > 1) {
    throw std::runtime_error(
        "nabla expansion requires a unique connection tensor field "
        "(mixed_tensor(up=1,down=2))");
  }
  return candidate;
}

static std::unique_ptr<IndexedVar>
cloneTensorWithReplacedIndex(const IndexedVar &src, size_t slot,
                             const std::string &replacement,
                             TensorTypeChecker &checker) {
  auto out = std::make_unique<IndexedVar>(src.name, src.kind);
  out->tensorKind = src.tensorKind;
  out->up = src.up;
  out->down = src.down;
  out->tensorIndexNames = src.tensorIndexNames;
  out->tensorIndexOffsets = src.tensorIndexOffsets;
  if (out->tensorIndexOffsets.size() != out->tensorIndexNames.size())
    out->tensorIndexOffsets.assign(out->tensorIndexNames.size(), 0);
  if (slot >= out->tensorIndexNames.size())
    throw std::runtime_error("internal error: invalid nabla index slot");
  out->tensorIndexNames[slot] = replacement;
  out->tensorIndexIsUp = src.tensorIndexIsUp;
  if (out->tensorIndexIsUp.size() != out->tensorIndexNames.size()) {
    out->tensorIndexIsUp.clear();
    for (size_t i = 0; i < out->tensorIndexNames.size(); ++i)
      out->tensorIndexIsUp.push_back(i < static_cast<size_t>(src.up));
  }
  checker.infer(out.get());
  return out;
}

static std::unique_ptr<IndexedExpr>
makeBinaryChecked(char op, std::unique_ptr<IndexedExpr> lhs,
                  std::unique_ptr<IndexedExpr> rhs, TensorTypeChecker &checker) {
  auto out = std::make_unique<IndexedBinary>(op, std::move(lhs), std::move(rhs));
  checker.infer(out.get());
  return out;
}

struct ChristoffelTerms {
  std::unique_ptr<IndexedExpr> a;
  std::unique_ptr<IndexedExpr> b;
  std::unique_ptr<IndexedExpr> c;
};

static ChristoffelTerms buildChristoffelTerms(
    const std::string &up, const std::string &d1, const std::string &d2,
    const std::string &g_name, const std::string &invg_name,
    const std::unordered_map<std::string, const FieldDecl *> &fields,
    TensorTypeChecker &checker,
    const std::unordered_set<std::string> *reservedIndices = nullptr) {
  std::unordered_set<std::string> used;
  used.insert(up);
  used.insert(d1);
  used.insert(d2);
  if (reservedIndices) {
    for (const auto &idx : *reservedIndices) {
      if (!idx.empty())
        used.insert(idx);
    }
  }
  std::string dum = chooseFreshSpatialIndex(used, "Christoffel");

  auto termA = makeDeriv(
      d1, makeVar(g_name, {d2, dum}, fields, checker), checker);
  auto termB = makeDeriv(
      d2, makeVar(g_name, {d1, dum}, fields, checker), checker);
  auto termC = makeDeriv(
      dum, makeVar(g_name, {d1, d2}, fields, checker), checker);

  auto prodA = makeContract(
      makeBinaryChecked('*',
                        makeVar(invg_name, {up, dum}, fields, checker),
                        std::move(termA), checker),
      checker);
  auto prodB = makeContract(
      makeBinaryChecked('*',
                        makeVar(invg_name, {up, dum}, fields, checker),
                        std::move(termB), checker),
      checker);
  auto prodC = makeContract(
      makeBinaryChecked('*',
                        makeVar(invg_name, {up, dum}, fields, checker),
                        std::move(termC), checker),
      checker);

  ChristoffelTerms out;
  out.a = std::move(prodA);
  out.b = std::move(prodB);
  out.c = std::move(prodC);
  return out;
}

static std::unique_ptr<IndexedExpr> expandCovariantNablaForTensor(
    const IndexedVar &tensorArg, std::unique_ptr<IndexedExpr> tensorArgExpr,
    const std::string &derivIdx, const std::string &g_name,
    const std::string &invg_name,
    const std::unordered_map<std::string, const FieldDecl *> &fields,
    TensorTypeChecker &checker,
    const std::unordered_set<std::string> *reservedIndices = nullptr) {
  if (tensorArg.tensorIndexNames.empty()) {
    throw std::runtime_error(
        "nabla on non-scalar tensor requires explicit indices");
  }

  std::unique_ptr<IndexedExpr> result =
      makeDeriv(derivIdx, std::move(tensorArgExpr), checker);

  std::unordered_set<std::string> baseUsed;
  for (const auto &idx : tensorArg.tensorIndexNames) {
    if (!idx.empty())
      baseUsed.insert(idx);
  }
  baseUsed.insert(derivIdx);
  if (reservedIndices) {
    for (const auto &idx : *reservedIndices) {
      if (!idx.empty())
        baseUsed.insert(idx);
    }
  }

  for (size_t slot = 0; slot < tensorArg.tensorIndexNames.size(); ++slot) {
    const std::string &slotIdx = tensorArg.tensorIndexNames[slot];
    if (slotIdx.empty()) {
      throw std::runtime_error(
          "nabla on non-scalar tensor requires explicit indices");
    }
    if (!isSpatialIndexName(slotIdx)) {
      throw std::runtime_error("Invalid tensor index '" + slotIdx +
                               "' in nabla expansion");
    }

    std::string dummy = chooseFreshSpatialIndex(baseUsed, "nabla expansion");
    bool slotIsUp = false;
    if (slot < tensorArg.tensorIndexIsUp.size()) {
      slotIsUp = tensorArg.tensorIndexIsUp[slot];
    } else {
      slotIsUp = slot < static_cast<size_t>(tensorArg.up);
    }

    ChristoffelTerms gammaTerms;
    if (slotIsUp) {
      gammaTerms = buildChristoffelTerms(slotIdx, derivIdx, dummy, g_name,
                                         invg_name, fields, checker,
                                         &baseUsed);
    } else {
      gammaTerms = buildChristoffelTerms(dummy, derivIdx, slotIdx, g_name,
                                         invg_name, fields, checker,
                                         &baseUsed);
    }

    auto repA = cloneTensorWithReplacedIndex(tensorArg, slot, dummy, checker);
    auto repB = cloneTensorWithReplacedIndex(tensorArg, slot, dummy, checker);
    auto repC = cloneTensorWithReplacedIndex(tensorArg, slot, dummy, checker);

    auto corrA = makeContract(
        makeBinaryChecked('*', std::move(gammaTerms.a), std::move(repA), checker),
        checker);
    auto corrB = makeContract(
        makeBinaryChecked('*', std::move(gammaTerms.b), std::move(repB), checker),
        checker);
    auto corrC = makeContract(
        makeBinaryChecked('*', std::move(gammaTerms.c), std::move(repC), checker),
        checker);

    auto halfA =
        makeBinaryChecked('*', std::make_unique<IndexedNumber>(0.5),
                          std::move(corrA), checker);
    auto halfB =
        makeBinaryChecked('*', std::make_unique<IndexedNumber>(0.5),
                          std::move(corrB), checker);
    auto halfC =
        makeBinaryChecked('*', std::make_unique<IndexedNumber>(0.5),
                          std::move(corrC), checker);

    auto sumAB = makeBinaryChecked('+', std::move(halfA), std::move(halfB), checker);
    auto correction = makeBinaryChecked('-', std::move(sumAB), std::move(halfC),
                                        checker);

    if (slotIsUp) {
      result = makeBinaryChecked('+', std::move(result), std::move(correction),
                                 checker);
    } else {
      result = makeBinaryChecked('-', std::move(result), std::move(correction),
                                 checker);
    }
  }

  checker.infer(result.get());
  return result;
}

static std::unique_ptr<IndexedExpr> expandCovariantNablaWithConnectionForTensor(
    const IndexedVar &tensorArg, std::unique_ptr<IndexedExpr> tensorArgExpr,
    const std::string &derivIdx, const std::string &connectionName,
    const std::unordered_map<std::string, const FieldDecl *> &fields,
    TensorTypeChecker &checker,
    const std::unordered_set<std::string> *reservedIndices = nullptr) {
  if (tensorArg.tensorIndexNames.empty()) {
    throw std::runtime_error(
        "nabla on non-scalar tensor requires explicit indices");
  }

  std::unique_ptr<IndexedExpr> result =
      makeDeriv(derivIdx, std::move(tensorArgExpr), checker);

  std::unordered_set<std::string> baseUsed;
  for (const auto &idx : tensorArg.tensorIndexNames) {
    if (!idx.empty())
      baseUsed.insert(idx);
  }
  baseUsed.insert(derivIdx);
  if (reservedIndices) {
    for (const auto &idx : *reservedIndices) {
      if (!idx.empty())
        baseUsed.insert(idx);
    }
  }

  for (size_t slot = 0; slot < tensorArg.tensorIndexNames.size(); ++slot) {
    const std::string &slotIdx = tensorArg.tensorIndexNames[slot];
    if (slotIdx.empty()) {
      throw std::runtime_error(
          "nabla on non-scalar tensor requires explicit indices");
    }
    if (!isSpatialIndexName(slotIdx)) {
      throw std::runtime_error("Invalid tensor index '" + slotIdx +
                               "' in nabla expansion");
    }

    std::string dummy = chooseFreshSpatialIndex(baseUsed, "nabla expansion");
    bool slotIsUp = false;
    if (slot < tensorArg.tensorIndexIsUp.size()) {
      slotIsUp = tensorArg.tensorIndexIsUp[slot];
    } else {
      slotIsUp = slot < static_cast<size_t>(tensorArg.up);
    }

    std::vector<std::string> gammaIndices;
    if (slotIsUp) {
      // +Gamma^i_{k m} T^m...
      gammaIndices = {slotIdx, derivIdx, dummy};
    } else {
      // -Gamma^m_{i k} T_...m...
      gammaIndices = {dummy, slotIdx, derivIdx};
    }

    auto gamma = makeVar(connectionName, gammaIndices, fields, checker);
    auto shifted = cloneTensorWithReplacedIndex(tensorArg, slot, dummy, checker);
    auto correction = makeContract(
        makeBinaryChecked('*', std::move(gamma), std::move(shifted), checker),
        checker);

    if (slotIsUp) {
      result = makeBinaryChecked('+', std::move(result), std::move(correction),
                                 checker);
    } else {
      result = makeBinaryChecked('-', std::move(result), std::move(correction),
                                 checker);
    }
  }

  checker.infer(result.get());
  return result;
}

static std::unique_ptr<IndexedExpr> raiseWithInverseMetric(
    std::unique_ptr<IndexedExpr> expr, const std::string &raisedIdx,
    const std::string &contractIdx, const std::string &invgName,
    const std::unordered_map<std::string, const FieldDecl *> &fields,
    TensorTypeChecker &checker) {
  if (auto *bin = dynamic_cast<IndexedBinary *>(expr.get())) {
    if (bin->op == '+' || bin->op == '-') {
      char op = bin->op;
      auto lhsRaised = raiseWithInverseMetric(
          std::move(bin->lhs), raisedIdx, contractIdx, invgName, fields, checker);
      auto rhsRaised = raiseWithInverseMetric(
          std::move(bin->rhs), raisedIdx, contractIdx, invgName, fields, checker);
      return makeBinaryChecked(op, std::move(lhsRaised), std::move(rhsRaised),
                               checker);
    }
  }

  auto raised = makeBinaryChecked(
      '*', makeVar(invgName, {raisedIdx, contractIdx}, fields, checker),
      std::move(expr), checker);
  return makeContract(std::move(raised), checker);
}

static TensorType tensorTypeFromDesc(const TensorTypeDesc &desc) {
  return TensorType{desc.up, desc.down};
}
static bool isScalarDesc(const TensorTypeDesc &desc) {
  return desc.up == 0 && desc.down == 0 && desc.kind == TensorKind::Scalar;
}

static constexpr const char *kErrMissingSimulationBlock =
    "E1001: missing simulation block in executable mode";

static constexpr const char *kWarnMissingSimulationBlock =
    "W1001: missing simulation block in symbolic mode";

static constexpr const char *kWarnInverseMetricMissing =
    "W1002: inverse_metric field is missing while metrics are declared";

static constexpr const char *kWarnMetricMissing =
    "W1003: metric field is missing while inverse_metric fields are declared";

namespace {
struct LocalScopeGuard {
  std::unordered_map<std::string, bool> &locals;
  std::unordered_map<std::string, bool> saved;

  LocalScopeGuard(std::unordered_map<std::string, bool> &localsIn,
                  std::unordered_map<std::string, bool> replacement)
      : locals(localsIn), saved(localsIn) {
    locals = std::move(replacement);
  }

  ~LocalScopeGuard() { locals = std::move(saved); }
};
} // namespace

void SemanticAnalyzer::validateSpatialIndex(const std::string &idx) {
  if (!core::isSpatialIndexName(idx)) {
    throw std::runtime_error("Invalid tensor index '" + idx +
                             "'. Allowed: {i, j, k, l, m, n}.");
  }
}

int SemanticAnalyzer::resolveIndex(const std::string &name) {
  auto it = coordIndex.find(name);
  if (it == coordIndex.end())
    throw std::runtime_error("Unknown tensor index: " + name);
  return it->second;
}

bool SemanticAnalyzer::isSimpleIndexSwap(const IndexedExpr *lhs,
                                         const IndexedExpr *rhs) const {
  auto lVar = dynamic_cast<const IndexedVar *>(lhs);
  auto rVar = dynamic_cast<const IndexedVar *>(rhs);
  if (!lVar || !rVar)
    return false;
  if (lVar->name != rVar->name)
    return false;
  if (lVar->tensorIndexNames.size() != 2 || rVar->tensorIndexNames.size() != 2)
    return false;
  return lVar->tensorIndexNames[0] == rVar->tensorIndexNames[1] &&
         lVar->tensorIndexNames[1] == rVar->tensorIndexNames[0];
}

bool SemanticAnalyzer::isNegatedSwap(const IndexedExpr *lhs,
                                     const IndexedExpr *rhs) const {
  auto bin = dynamic_cast<const IndexedBinary *>(rhs);
  if (!bin || bin->op != '*')
    return false;
  const IndexedExpr *other = nullptr;
  double coeff = 0.0;
  if (auto num = dynamic_cast<const IndexedNumber *>(bin->lhs.get())) {
    coeff = num->value;
    other = bin->rhs.get();
  } else if (auto num = dynamic_cast<const IndexedNumber *>(bin->rhs.get())) {
    coeff = num->value;
    other = bin->lhs.get();
  }
  if (coeff == -1.0 && other)
    return isSimpleIndexSwap(lhs, other);
  return false;
}

bool SemanticAnalyzer::containsExplicitMetricAntisymmetry(
    const IndexedExpr *expr) const {
  if (!expr)
    return false;
  if (auto bin = dynamic_cast<const IndexedBinary *>(expr)) {
    if (bin->op == '-') {
      if (isSimpleIndexSwap(bin->lhs.get(), bin->rhs.get()))
        return true;
    }
    if (bin->op == '+') {
      if (isNegatedSwap(bin->lhs.get(), bin->rhs.get()) ||
          isNegatedSwap(bin->rhs.get(), bin->lhs.get()))
        return true;
    }
    return containsExplicitMetricAntisymmetry(bin->lhs.get()) ||
           containsExplicitMetricAntisymmetry(bin->rhs.get());
  }
  if (auto call = dynamic_cast<const IndexedCall *>(expr)) {
    for (const auto &arg : call->args)
      if (containsExplicitMetricAntisymmetry(arg.get()))
        return true;
  }
  return false;
}

void SemanticAnalyzer::enforceMetricFieldRules(const FieldDecl &field) {
  if (field.isMetric) {
    if (field.up != 0 || field.down != 2) {
      throw std::runtime_error("metric field '" + field.name +
                               "' must be covariant rank-2");
    }
    if (field.indices.size() != 2) {
      throw std::runtime_error("metric field '" + field.name +
                               "' must declare exactly two indices");
    }
    metricFieldCount++;
  } else if (field.isInverseMetric) {
    if (field.up != 2 || field.down != 0) {
      throw std::runtime_error("inverse_metric field '" + field.name +
                               "' must be contravariant rank-2");
    }
    if (field.indices.size() != 2) {
      throw std::runtime_error("inverse_metric field '" + field.name +
                               "' must declare exactly two indices");
    }
    inverseMetricFieldCount++;
  }
}

// --- TRANSFORM EXPR MISE À JOUR ---

std::unique_ptr<IndexedExpr> SemanticAnalyzer::transformExpr(const Expr *e) {
  if (auto n = dynamic_cast<const NumberExpr *>(e))
    return std::make_unique<IndexedNumber>(n->value);

  if (auto v = dynamic_cast<const VarExpr *>(e)) {
    if (auto it = coordIndex.find(v->name); it != coordIndex.end()) {
      auto iv =
          std::make_unique<IndexedVar>(v->name, IndexedVarKind::Coordinate);
      iv->coordIndex = it->second;
      iv->tensorKind = TensorKind::Scalar;
      return iv;
    }
    if (locals.count(v->name)) {
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Local);
      iv->tensorKind = TensorKind::Scalar;
      return iv;
    }

    if (params.count(v->name)) {
      auto iv =
          std::make_unique<IndexedVar>(v->name, IndexedVarKind::Parameter);
      iv->tensorKind = TensorKind::Scalar;
      return iv;
    }

    if (auto itf = fields.find(v->name); itf != fields.end()) {
      const FieldDecl *fd = itf->second;
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Field);
      iv->tensorKind = fd->kind;
      iv->up = fd->up;
      iv->down = fd->down;
      return iv;
    }

    throw std::runtime_error("Unknown identifier: " + v->name);
  }

  if (auto b = dynamic_cast<const BinaryExpr *>(e))
    return std::make_unique<IndexedBinary>(b->op, transformExpr(b->lhs.get()),
                                           transformExpr(b->rhs.get()));

  if (auto p = dynamic_cast<const ParenExpr *>(e))
    return transformExpr(p->inner.get());

  if (auto iv = dynamic_cast<const IndexedVarExpr *>(e)) {
    auto it = fields.find(iv->base);
    if (it == fields.end())
      throw std::runtime_error("Unknown indexed tensor: " + iv->base);

    const FieldDecl *fd = it->second;
    size_t expected = static_cast<size_t>(fd->up + fd->down);

    if (iv->indices.size() != expected)
      throw std::runtime_error("Tensor '" + iv->base + "' expects " +
                               std::to_string(expected) + " indices, got " +
                               std::to_string(iv->indices.size()));

    auto out = std::make_unique<IndexedVar>(iv->base, IndexedVarKind::Field);
    out->tensorKind = fd->kind;
    out->up = fd->up;
    out->down = fd->down;

    size_t pos = 0;
    for (size_t i = 0; i < iv->indices.size(); ++i) {
      const auto &idx = iv->indices[i];
      if (!coordIndex.count(idx)) {
        validateSpatialIndex(idx);
        coordIndex[idx] = -2;
      }
      int off = resolveIndex(idx);
      out->tensorIndices.push_back(off);
      out->tensorIndexNames.push_back(idx);
      int idxOff = 0;
      if (i < iv->indexOffsets.size())
        idxOff = iv->indexOffsets[i];
      out->tensorIndexOffsets.push_back(idxOff);
      bool isUp = pos < static_cast<size_t>(fd->up);
      out->tensorIndexIsUp.push_back(isUp);
      ++pos;
    }
    return out;
  }

  if (auto c = dynamic_cast<const CallExpr *>(e)) {
    if (c->callee == "trace") {
      if (c->args.size() != 1)
        throw std::runtime_error("trace() expects exactly 1 argument");
      TensorTypeChecker checker;
      auto arg = transformExpr(c->args[0].get());
      auto out = makeContract(std::move(arg), checker);
      checker.infer(out.get());
      return out;
    }

    if (c->callee == "laplacian") {
      if (c->args.size() != 1)
        throw std::runtime_error("laplacian() expects exactly 1 argument");
      TensorTypeChecker checker;
      auto arg = transformExpr(c->args[0].get());
      TensorType argT = checker.infer(arg.get());
      if (!argT.isScalar())
        throw std::runtime_error("laplacian() expects scalar argument");

      auto d1 = makeDeriv("i", std::move(arg), checker);
      auto d2 = makeDeriv("i", std::move(d1), checker);
      auto out = makeContract(std::move(d2), checker);
      checker.infer(out.get());
      return out;
    }

    const bool isCovariantNabla =
        c->callee.size() == 7 && c->callee.rfind("nabla_", 0) == 0;
    const bool isContravariantNabla =
        c->callee.size() == 7 && c->callee.rfind("nabla^", 0) == 0;
    if (isCovariantNabla || isContravariantNabla) {
      if (c->args.size() != 1)
        throw std::runtime_error("nabla expects exactly 1 argument");

      std::string derivIdx(1, c->callee[6]);
      validateSpatialIndex(derivIdx);

      TensorTypeChecker checker;
      auto arg = transformExpr(c->args[0].get());
      TensorType argT = checker.infer(arg.get());
      const bool isContravariant = isContravariantNabla;

      if (argT.isScalar()) {
        if (!isContravariant)
          return makeDeriv(derivIdx, std::move(arg), checker);

        std::string invgName;
        int inverseCount = 0;
        for (const auto &kv : fields) {
          if (kv.second->isInverseMetric) {
            invgName = kv.first;
            ++inverseCount;
          }
        }
        if (inverseCount > 1) {
          throw std::runtime_error(
              "nabla^ on scalar requires a unique inverse_metric field "
              "(multiple declared)");
        }
        if (invgName.empty()) {
          throw std::runtime_error(
              "nabla^ on scalar requires 'inverse_metric'");
        }

        std::unordered_set<std::string> used;
        collectExprIndexNames(arg.get(), used);
        used.insert(derivIdx);
        std::string lowerIdx = chooseFreshSpatialIndex(used, "nabla^ scalar");

        auto cov = makeDeriv(lowerIdx, std::move(arg), checker);
        auto raised = makeBinaryChecked(
            '*', makeVar(invgName, {derivIdx, lowerIdx}, fields, checker),
            std::move(cov), checker);
        auto out = makeContract(std::move(raised), checker);
        checker.infer(out.get());
        return out;
      }

      auto metrics = resolveMetricNames(fields);
      const std::string &gName = metrics.first;
      const std::string &invgName = metrics.second;
      const std::string connectionName = resolveConnectionFieldName(fields);

      const bool canExpandFromMetric = !gName.empty() && !invgName.empty();
      const bool canExpandFromConnection = !connectionName.empty();
      if (!canExpandFromMetric && !canExpandFromConnection) {
        throw std::runtime_error(
            "nabla on non-scalar tensor requires either "
            "'metric'+'inverse_metric' or a connection tensor "
            "mixed_tensor(up=1,down=2)");
      }

      auto *tensorVar = dynamic_cast<IndexedVar *>(arg.get());
      if (!tensorVar) {
        throw std::runtime_error(
            "nabla on non-scalar tensor requires an indexed tensor argument");
      }
      if (tensorVar->tensorIndexNames.size() != static_cast<size_t>(argT.rank())) {
        throw std::runtime_error(
            "nabla on non-scalar tensor requires explicit indices");
      }

      IndexedVar tensorSnapshot = *tensorVar;

      if (!isContravariant) {
        if (canExpandFromMetric) {
          return expandCovariantNablaForTensor(
              tensorSnapshot, std::move(arg), derivIdx, gName, invgName, fields,
              checker);
        }
        return expandCovariantNablaWithConnectionForTensor(
            tensorSnapshot, std::move(arg), derivIdx, connectionName, fields,
            checker);
      }

      if (invgName.empty()) {
        throw std::runtime_error(
            "nabla^ on non-scalar tensor requires 'inverse_metric'");
      }

      std::unordered_set<std::string> used;
      for (const auto &idx : tensorSnapshot.tensorIndexNames) {
        if (!idx.empty())
          used.insert(idx);
      }
      used.insert(derivIdx);
      std::string covIdx = chooseFreshSpatialIndex(used, "nabla^ expansion");

      std::unique_ptr<IndexedExpr> covExpanded;
      std::unordered_set<std::string> reservedForCovariantExpansion;
      reservedForCovariantExpansion.insert(derivIdx);
      if (canExpandFromMetric) {
        covExpanded = expandCovariantNablaForTensor(
            tensorSnapshot, std::move(arg), covIdx, gName, invgName, fields,
            checker, &reservedForCovariantExpansion);
      } else {
        covExpanded = expandCovariantNablaWithConnectionForTensor(
            tensorSnapshot, std::move(arg), covIdx, connectionName, fields,
            checker, &reservedForCovariantExpansion);
      }
      auto out = raiseWithInverseMetric(std::move(covExpanded), derivIdx, covIdx,
                                        invgName, fields, checker);
      checker.infer(out.get());
      return out;
    }

    const ExternDecl *externDecl = nullptr;
    if (auto itExt = externSignatures.find(c->callee);
        itExt != externSignatures.end()) {
      externDecl = itExt->second;
      if (c->args.size() != externDecl->params.size()) {
        throw std::runtime_error(
            "extern function '" + c->callee + "' expects " +
            std::to_string(externDecl->params.size()) + " arguments, got " +
            std::to_string(c->args.size()));
      }
    }

    if (mode == CompilationMode::Executable &&
        !isExecutableBuiltin(c->callee) && !externDecl) {
      throw std::runtime_error(
          "executable mode requires implementation for function '" + c->callee +
          "'");
    }
    auto out = std::make_unique<IndexedCall>();
    out->callee = c->callee;
    out->isExtern = externDecl != nullptr;
    out->declaredArity = c->args.size();
    if (externDecl) {
      out->returnType = externDecl->returnType;
      out->paramTypes = externDecl->params;
    }
    TensorTypeChecker argChecker(hasConnectionTensor);
    for (size_t i = 0; i < c->args.size(); ++i) {
      auto transformed = transformExpr(c->args[i].get());
      if (externDecl) {
        TensorType actual = argChecker.infer(transformed.get());
        TensorType expected = tensorTypeFromDesc(externDecl->params[i]);
        if (!actual.sameVariance(expected)) {
          throw std::runtime_error("extern function '" + c->callee +
                                   "' argument " + std::to_string(i + 1) +
                                   " variance mismatch");
        }
      }
      out->args.push_back(std::move(transformed));
    }
    return out;
  }

  throw std::runtime_error("Unsupported expr in semantic analyzer");
}

SemanticAnalyzer::SemanticAnalyzer(const Program &p, CompilationMode m)
    : prog(p), mode(m) {
  for (const auto &paramName : prog.params) {
    if (!params.insert(paramName).second) {
      throw std::runtime_error("Parameter redeclared: " + paramName);
    }
  }

  for (const auto &ext : prog.externs) {
    if (!externSignatures.emplace(ext.name, &ext).second) {
      throw std::runtime_error("Extern function redeclared: " + ext.name);
    }
    if (mode == CompilationMode::Executable && !isScalarDesc(ext.returnType)) {
      throw std::runtime_error("executable mode extern '" + ext.name +
                               "' must return scalar");
    }
  }

  std::unordered_set<std::string> metricNames;
  for (const auto &metric : prog.metrics) {
    if (params.count(metric.name)) {
      throw std::runtime_error("Name collision: parameter '" + metric.name +
                               "' conflicts with metric '" + metric.name +
                               "'");
    }
    if (!metricNames.insert(metric.name).second) {
      throw std::runtime_error("Metric redeclared: " + metric.name);
    }
  }

  for (const auto &f : prog.fields) {
    if (params.count(f.name)) {
      throw std::runtime_error("Name collision: parameter '" + f.name +
                               "' conflicts with field '" + f.name + "'");
    }
    if (metricNames.count(f.name)) {
      throw std::runtime_error("Name collision: field '" + f.name +
                               "' conflicts with metric '" + f.name + "'");
    }
    if (fields.count(f.name))
      throw std::runtime_error("Field redeclared: " + f.name);
    enforceMetricFieldRules(f);
    if ((f.up == 1 && f.down == 2) ||
        ((f.up + f.down) == 3 &&
         (f.name == "Gamma" || f.name == "GammaU" || f.name == "Christoffel"))) {
      hasConnectionTensor = true;
    }
    fields[f.name] = &f;
  }

  if (metricFieldCount > 0 && inverseMetricFieldCount == 0) {
    warnings.push_back(kWarnInverseMetricMissing);
  }
  if (inverseMetricFieldCount > 0 && metricFieldCount == 0) {
    warnings.push_back(kWarnMetricMissing);
  }

  for (const auto &m : prog.metrics) {
    for (const auto &entry : m.entries) {
      if (entry.lhs.indices.empty())
        metricScalarLocals[entry.lhs.base] = true;
    }
  }
  for (const auto &m : prog.metrics) {
    if (fields.count(m.name)) {
      throw std::runtime_error("Name collision: metric '" + m.name +
                               "' conflicts with existing field '" + m.name +
                               "'");
    }
    FieldDecl fd;
    fd.kind = TensorKind::CovTensor2;
    fd.name = m.name;
    fd.up = 0;
    fd.down = 2;

    fd.isMetric = true;

    syntheticMetricFields.push_back(fd);
    fields[m.name] = &syntheticMetricFields.back();
  }
  if (!prog.simulation) {
    if (mode == CompilationMode::Executable) {
      throw std::runtime_error(
          std::string(kErrMissingSimulationBlock) +
          ". Add `simulation { dimension = <N> resolution = [...] time { dt = ... "
          "integrator = ... } spatial { scheme = ... derivative = ... order = ... } }` "
          "or use --symbolic.");
    }
    simulationMissing = true;
    warnings.push_back(std::string(kWarnMissingSimulationBlock) +
                       "; proceeding without simulation metadata");
  } else {
    validateSimulation(*prog.simulation);
  }

  if (prog.initialData) {
    validateInitialData(*prog.initialData);
  }
}

IndexedMetric SemanticAnalyzer::analyzeMetric(const MetricDecl &decl) {
  LocalScopeGuard localsScope(locals, metricScalarLocals);
  coordIndex.clear();
  locals.clear();

  IndexedMetric out;
  out.name = decl.name;
  out.rank = 2;
  out.coords = decl.indices;

  for (size_t i = 0; i < decl.indices.size(); ++i)
    coordIndex[decl.indices[i]] = static_cast<int>(i);

  TensorTypeChecker checker(hasConnectionTensor);

  for (const auto &entry : decl.entries) {
    IndexedAssignment a;
    a.tensor = entry.lhs.base;

    if (!entry.lhs.indices.empty()) {
      if (entry.lhs.indices.size() != 2)
        throw std::runtime_error(
            "Metric tensor '" + entry.lhs.base + "' must have 2 indices (got " +
            std::to_string(entry.lhs.indices.size()) + ")");
      for (const auto &idx : entry.lhs.indices)
        a.indexOffsets.push_back(resolveIndex(idx));
    }

    a.rhs = transformExpr(entry.rhs.get());
    checker.checkMetricAssignment(a);
    checker.infer(a.rhs.get());
    out.assignments.push_back(std::move(a));
  }

  return out;
}

IndexedEvolution SemanticAnalyzer::analyzeEvolution(const EvolutionDecl &evo) {
  LocalScopeGuard localsScope(locals, std::unordered_map<std::string, bool>{});
  coordIndex.clear();
  locals.clear();

  IndexedEvolution out;
  out.name = evo.name;

  for (const auto &eq : evo.equations)
    for (const auto &idx : eq.indices) {
      validateSpatialIndex(idx);
      coordIndex[idx] = -1;
    }

  for (const auto &tmp : evo.tempAssignments) {
    if (!tmp.lhs.indices.empty()) {
      continue;
    }

    if (params.count(tmp.lhs.base)) {
      throw std::runtime_error("Cannot redeclare parameter '" + tmp.lhs.base +
                               "' as local");
    }

    if (fields.count(tmp.lhs.base)) {
      throw std::runtime_error("Cannot redeclare field '" + tmp.lhs.base +
                               "' as local");
    }

    locals[tmp.lhs.base] = true;
  }

  TensorTypeChecker checker(hasConnectionTensor);

  for (const auto &eq : evo.equations) {

    auto it = fields.find(eq.fieldName);
    if (it == fields.end())
      throw std::runtime_error("Unknown field in evolution: " + eq.fieldName);

    const FieldDecl *fd = it->second;
    size_t expectedRank = static_cast<size_t>(fd->up + fd->down);

    if (eq.indices.size() != expectedRank) {
      throw std::runtime_error(
          "Wrong number of indices in evolution for field '" + eq.fieldName +
          "': expected " + std::to_string(expectedRank) + ", got " +
          std::to_string(eq.indices.size()));
    }

    IndexedEvolutionEq ie;
    ie.fieldName = eq.fieldName;
    ie.indices = eq.indices;
    ie.rhs = transformExpr(eq.rhs.get());

    if (fd->isMetric || fd->isInverseMetric) {
      if (containsExplicitMetricAntisymmetry(ie.rhs.get())) {
        throw std::runtime_error("metric field '" + fd->name +
                                 "' assignments must be symmetric");
      }
    }

    TensorType lhsType = {fd->up, fd->down};
    checker.checkAssignmentVariance(lhsType, ie.indices, ie.rhs.get());
    checker.infer(ie.rhs.get());
    ie.rhs->inferredType.kind = fd->kind;
    ie.rhs->inferredType.up = fd->up;
    ie.rhs->inferredType.down = fd->down;

    out.equations.push_back(std::move(ie));
  }

  for (const auto &tmp : evo.tempAssignments) {
    auto rhs = transformExpr(tmp.rhs.get());
    TensorType lhsType{0, 0};
    checker.checkAssignmentVariance(lhsType, tmp.lhs.indices, rhs.get());
    checker.infer(rhs.get());

    IndexedAssignment ia;
    ia.tensor = tmp.lhs.base;
    for (auto &idx : tmp.lhs.indices)
      ia.indexOffsets.push_back(resolveIndex(idx));
    ia.rhs = std::move(rhs);
    out.temp.push_back(std::move(ia));
  }

  return out;
}

void SemanticAnalyzer::validateSimulation(const SimulationConfig &sim) {
  if (sim.dimension <= 0) {
    throw std::runtime_error("simulation dimension must be >= 1");
  }

  if ((int)sim.resolution.size() != sim.dimension) {
    throw std::runtime_error(
        "resolution size (" + std::to_string(sim.resolution.size()) +
        ") does not match dimension (" + std::to_string(sim.dimension) + ")");
  }

  for (int r : sim.resolution) {
    if (r <= 0)
      throw std::runtime_error("resolution entries must be > 0");
  }

  if (sim.time.dt <= 0.0) {
    throw std::runtime_error("time.dt must be > 0");
  }

  if (sim.spatial.scheme == SpatialScheme::FiniteDifference) {
    if (sim.spatial.order < 2)
      throw std::runtime_error("FD order must be >= 2");

    if (sim.spatial.order % 2 != 0)
      throw std::runtime_error("FD order must be even");
  }

  if (sim.spatial.scheme == SpatialScheme::Spectral) {
    if (sim.spatial.order != 0) {
      throw std::runtime_error(
          "spectral scheme does not use finite-difference order");
    }
  }
}
} // namespace tensorium
