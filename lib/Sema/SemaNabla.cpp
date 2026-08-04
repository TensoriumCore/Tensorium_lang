#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Sema/tensor_type_checker.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tensorium {
namespace {

bool isLegacyNamedConnectionField(const std::string &name,
                                  const FieldDecl *fd) {
  if (!fd)
    return false;
  return (fd->up + fd->down) == 3 &&
         (name == "Gamma" || name == "GammaU" || name == "Christoffel");
}

std::unique_ptr<IndexedVar>
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

std::unique_ptr<IndexedVar> makeConnectionVar(
    const std::string &name, const std::vector<std::string> &indices,
    const std::unordered_map<std::string, const FieldDecl *> &fields,
    TensorTypeChecker &checker) {
  auto it = fields.find(name);
  if (it == fields.end())
    throw std::runtime_error("Connection field not found: " + name);
  const FieldDecl *fd = it->second;

  const bool isCanonicalConnection = fd->up == 1 && fd->down == 2;
  if (isCanonicalConnection)
    return makeVar(name, indices, fields, checker);

  if (!isLegacyNamedConnectionField(name, fd))
    return makeVar(name, indices, fields, checker);

  // Legacy 'con_tensor3 Gamma[..]' declarations are interpreted as
  // connection tensors Gamma^i_{jk} for covariant-derivative expansion.
  auto v = std::make_unique<IndexedVar>(name, IndexedVarKind::Field);
  v->tensorKind = TensorKind::MixedTensor;
  v->up = 1;
  v->down = 2;
  v->tensorIndexNames = indices;
  for (size_t i = 0; i < indices.size(); ++i) {
    v->tensorIndexIsUp.push_back(i == 0);
    v->tensorIndexOffsets.push_back(0);
  }
  checker.infer(v.get());
  return v;
}

std::unique_ptr<IndexedCall> makeDeriv(const std::string &idx,
                                       std::unique_ptr<IndexedExpr> arg,
                                       TensorTypeChecker &checker) {
  auto call = std::make_unique<IndexedCall>();
  call->callee = "d_" + idx;
  call->args.push_back(std::move(arg));
  checker.infer(call.get());
  return call;
}

std::unique_ptr<IndexedCall>
makeContract(std::unique_ptr<IndexedExpr> arg, TensorTypeChecker &checker) {
  auto call = std::make_unique<IndexedCall>();
  call->callee = "contract";
  call->args.push_back(std::move(arg));
  call->isExtern = false;
  checker.infer(call.get());
  return call;
}

bool isSpatialIndexName(const std::string &idx) {
  return idx.size() == 1 && core::isSpatialIndexChar(idx[0]);
}

void collectExprIndexNames(const IndexedExpr *e,
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
        (c->callee.rfind("nabla_", 0) == 0 ||
         c->callee.rfind("nabla^", 0) == 0)) {
      std::string idx(1, c->callee[6]);
      if (isSpatialIndexName(idx))
        out.insert(idx);
    }
    for (const auto &arg : c->args)
      collectExprIndexNames(arg.get(), out);
  }
}

std::string chooseFreshSpatialIndex(const std::unordered_set<std::string> &used,
                                    const std::string &context) {
  for (const char c : {'i', 'j', 'k', 'l', 'm', 'n'}) {
    std::string idx(1, c);
    if (!used.count(idx))
      return idx;
  }
  throw std::runtime_error("nabla expansion ran out of fresh spatial indices in " +
                           context);
}

std::pair<std::string, std::string> resolveMetricNames(
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

std::string resolveConnectionFieldName(
    const std::unordered_map<std::string, const FieldDecl *> &fields) {
  auto isConnection = [](const std::string &name, const FieldDecl *fd) {
    if (!fd)
      return false;
    if (fd->up == 1 && fd->down == 2)
      return true;
    return isLegacyNamedConnectionField(name, fd);
  };

  auto pickNamed = [&](const std::string &name) -> std::string {
    auto it = fields.find(name);
    if (it == fields.end())
      return {};
    if (!isConnection(it->first, it->second))
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
    if (!isConnection(kv.first, kv.second))
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

std::unique_ptr<IndexedVar>
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

std::unique_ptr<IndexedExpr>
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

ChristoffelTerms buildChristoffelTerms(
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

  auto termA = makeDeriv(d1, makeVar(g_name, {d2, dum}, fields, checker),
                         checker);
  auto termB = makeDeriv(d2, makeVar(g_name, {d1, dum}, fields, checker),
                         checker);
  auto termC = makeDeriv(dum, makeVar(g_name, {d1, d2}, fields, checker),
                         checker);

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

std::unique_ptr<IndexedExpr> expandCovariantNablaForTensor(
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
        makeBinaryChecked('*', std::move(gammaTerms.a), std::move(repA),
                          checker),
        checker);
    auto corrB = makeContract(
        makeBinaryChecked('*', std::move(gammaTerms.b), std::move(repB),
                          checker),
        checker);
    auto corrC = makeContract(
        makeBinaryChecked('*', std::move(gammaTerms.c), std::move(repC),
                          checker),
        checker);

    auto halfA = makeBinaryChecked('*', std::make_unique<IndexedNumber>(0.5),
                                   std::move(corrA), checker);
    auto halfB = makeBinaryChecked('*', std::make_unique<IndexedNumber>(0.5),
                                   std::move(corrB), checker);
    auto halfC = makeBinaryChecked('*', std::make_unique<IndexedNumber>(0.5),
                                   std::move(corrC), checker);

    auto sumAB =
        makeBinaryChecked('+', std::move(halfA), std::move(halfB), checker);
    auto correction =
        makeBinaryChecked('-', std::move(sumAB), std::move(halfC), checker);

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

std::unique_ptr<IndexedExpr> expandCovariantNablaWithConnectionForTensor(
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

    auto gamma = makeConnectionVar(connectionName, gammaIndices, fields, checker);
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

std::unique_ptr<IndexedExpr> raiseWithInverseMetric(
    std::unique_ptr<IndexedExpr> expr, const std::string &raisedIdx,
    const std::string &contractIdx, const std::string &invgName,
    const std::unordered_map<std::string, const FieldDecl *> &fields,
    TensorTypeChecker &checker) {
  if (auto *bin = dynamic_cast<IndexedBinary *>(expr.get())) {
    if (bin->op == '+' || bin->op == '-') {
      char op = bin->op;
      auto lhsRaised = raiseWithInverseMetric(
          std::move(bin->lhs), raisedIdx, contractIdx, invgName, fields,
          checker);
      auto rhsRaised = raiseWithInverseMetric(
          std::move(bin->rhs), raisedIdx, contractIdx, invgName, fields,
          checker);
      return makeBinaryChecked(op, std::move(lhsRaised), std::move(rhsRaised),
                               checker);
    }
  }

  auto raised = makeBinaryChecked(
      '*', makeVar(invgName, {raisedIdx, contractIdx}, fields, checker),
      std::move(expr), checker);
  return makeContract(std::move(raised), checker);
}

} // namespace

std::unique_ptr<IndexedExpr>
SemanticAnalyzer::transformNablaCall(const CallExpr &call,
                                     bool isContravariant) {
  if (call.args.size() != 1)
    throw std::runtime_error("nabla expects exactly 1 argument");

  std::string derivIdx(1, call.callee[6]);
  validateSpatialIndex(derivIdx);

  if (analyzingConstraintProblem && constraintGeometryAvailable) {
    auto out = std::make_unique<IndexedCall>();
    out->callee = call.callee;
    out->declaredArity = 1;
    out->args.push_back(transformExpr(call.args[0].get()));
    TensorTypeChecker geometryChecker(true);
    const TensorType argumentType =
        geometryChecker.infer(out->args.front().get());
    if (!argumentType.isScalar() &&
        !dynamic_cast<const IndexedVar *>(out->args.front().get())) {
      throw std::runtime_error(
          "nabla on non-scalar tensor requires an indexed tensor argument");
    }
    geometryChecker.infer(out.get());
    return out;
  }

  TensorTypeChecker checker;
  auto arg = transformExpr(call.args[0].get());
  TensorType argT = checker.infer(arg.get());

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
      throw std::runtime_error("nabla^ on scalar requires 'inverse_metric'");
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

} // namespace tensorium
