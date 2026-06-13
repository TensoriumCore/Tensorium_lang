#include "ExprLowering.h"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Lowering/TensorTypeConversion.hpp"

#include <map>
#include <string>
#include <vector>

namespace tensorium::backend {

static bool hasTensorRank(const tensorium::IndexedExpr *e) {
  return e && (e->inferredType.up + e->inferredType.down) > 0;
}

static bool parsePartialDerivativeName(const std::string &name,
                                       std::string &coordIndex) {
  if (name.size() != 3 || name[0] != 'd' || name[1] != '_')
    return false;
  if (!tensorium::core::isSpatialIndexChar(name[2]))
    return false;
  coordIndex.assign(1, name[2]);
  return true;
}

static bool parseCovariantDerivativeName(const std::string &name,
                                         bool &contravariant,
                                         std::string &coordIndex) {
  if (name.size() == 7 && name.rfind("nabla_", 0) == 0 &&
      tensorium::core::isSpatialIndexChar(name[6])) {
    contravariant = false;
    coordIndex.assign(1, name[6]);
    return true;
  }
  if (name.size() == 7 && name.rfind("nabla^", 0) == 0 &&
      tensorium::core::isSpatialIndexChar(name[6])) {
    contravariant = true;
    coordIndex.assign(1, name[6]);
    return true;
  }
  return false;
}

static bool tryExtractIndexName(const tensorium::IndexedExpr *e,
                                std::string &outName) {
  auto *v = dynamic_cast<const tensorium::IndexedVar *>(e);
  if (!v || v->name.size() != 1)
    return false;
  if (!tensorium::core::isTensorIndexChar(v->name[0]))
    return false;
  outName = v->name;
  return true;
}

static void collectIndexCounts(const tensorium::IndexedExpr *e,
                               std::map<std::string, int> &counts) {
  using namespace tensorium;
  if (!e)
    return;

  if (auto *v = dynamic_cast<const IndexedVar *>(e)) {
    for (const auto &name : v->tensorIndexNames) {
      if (!name.empty() && core::isTensorIndexName(name))
        counts[name] += 1;
    }
    return;
  }

  if (auto *b = dynamic_cast<const IndexedBinary *>(e)) {
    collectIndexCounts(b->lhs.get(), counts);
    collectIndexCounts(b->rhs.get(), counts);
    return;
  }

  if (auto *c = dynamic_cast<const IndexedCall *>(e)) {
    if (c->callee == "contract") {
      // A contract(...) contributes only its free indices to the surrounding
      // expression. This allows outer expressions to contract against those
      // free indices (for example gammaU[j,k] * contract(...[i,k]...)).
      if (c->args.empty())
        return;
      std::map<std::string, int> local;
      collectIndexCounts(c->args[0].get(), local);
      for (const auto &[idx, count] : local) {
        if (count == 1)
          counts[idx] += 1;
      }
      return;
    }

    for (const auto &arg : c->args)
      collectIndexCounts(arg.get(), counts);

    std::string idx;
    if (parsePartialDerivativeName(c->callee, idx)) {
      counts[idx] += 1;
      return;
    }

    bool contra = false;
    if (parseCovariantDerivativeName(c->callee, contra, idx)) {
      counts[idx] += 1;
      return;
    }

    if (c->callee == "covariant_derivative" && c->args.size() >= 2 &&
        tryExtractIndexName(c->args[1].get(), idx)) {
      counts[idx] += 1;
      return;
    }
  }
}

static std::vector<std::string>
collectRepeatedIndices(const tensorium::IndexedExpr *e) {
  std::map<std::string, int> counts;
  collectIndexCounts(e, counts);

  std::vector<std::string> repeated;
  for (const auto &[idx, count] : counts) {
    if (count >= 2)
      repeated.push_back(idx);
  }
  return repeated;
}
static std::unique_ptr<VarIR>
makeIndexedFieldRef(const std::string &fieldName,
                    const std::vector<std::string> &indexNames, int up,
                    int down) {
  auto out = std::make_unique<VarIR>(fieldName, VarKind::Field);
  out->tensorIndexNames = indexNames;
  out->exprType = lowering::makeTensorType(up, down);
  return out;
}

static const tensorium::IndexedVar *
asFieldVar(const tensorium::IndexedExpr *e) {
  auto *v = dynamic_cast<const tensorium::IndexedVar *>(e);
  if (!v || v->kind != tensorium::IndexedVarKind::Field)
    return nullptr;
  return v;
}

std::unique_ptr<ExprIR>
lowerIndexedExpr(const tensorium::IndexedExpr *e,
                bool materializeImplicitContraction,
                bool hasConnectionTensor);

static std::unique_ptr<ExprIR>
lowerChristoffelBuiltin(const tensorium::IndexedCall *call) {
  if (!call || call->args.size() != 2)
    return std::make_unique<CallIR>("<invalid_christoffel>");

  auto *gammaArg = asFieldVar(call->args[0].get());
  auto *gammaUArg = asFieldVar(call->args[1].get());
  if (!gammaArg || !gammaUArg)
    return std::make_unique<CallIR>("<invalid_christoffel>");

  const std::string gammaName = gammaArg->name;
  const std::string gammaUName = gammaUArg->name;

  auto gamma_lk = makeIndexedFieldRef(gammaName, {"l", "k"}, 0, 2);
  auto gamma_lj = makeIndexedFieldRef(gammaName, {"l", "j"}, 0, 2);
  auto gamma_jk = makeIndexedFieldRef(gammaName, {"j", "k"}, 0, 2);
  auto gammaU_il = makeIndexedFieldRef(gammaUName, {"i", "l"}, 2, 0);

  auto dj_gamma_lk =
      std::make_unique<PartialDerivativeIR>(std::move(gamma_lk), "j");
  dj_gamma_lk->exprType = lowering::makeTensorType(0, 3);

  auto dk_gamma_lj =
      std::make_unique<PartialDerivativeIR>(std::move(gamma_lj), "k");
  dk_gamma_lj->exprType = lowering::makeTensorType(0, 3);

  auto dl_gamma_jk =
      std::make_unique<PartialDerivativeIR>(std::move(gamma_jk), "l");
  dl_gamma_jk->exprType = lowering::makeTensorType(0, 3);

  auto add = std::make_unique<BinaryIR>("+", std::move(dj_gamma_lk),
                                        std::move(dk_gamma_lj));
  add->exprType = lowering::makeTensorType(0, 3);

  auto sum = std::make_unique<BinaryIR>("-", std::move(add),
                                        std::move(dl_gamma_jk));
  sum->exprType = lowering::makeTensorType(0, 3);

  auto product =
      std::make_unique<TensorProductIR>(std::move(gammaU_il), std::move(sum));
  product->exprType = lowering::makeTensorType(2, 3);

  auto contraction = std::make_unique<ContractionIR>(std::move(product));
  contraction->summedIndices = {"l"};
  contraction->exprType = lowering::makeTensorType(1, 2);

  auto half = std::make_unique<NumberIR>(0.5);
  half->exprType = lowering::makeTensorType(0, 0);

  auto out =
      std::make_unique<BinaryIR>("*", std::move(half), std::move(contraction));
  out->exprType = lowering::lowerTensorType(call->inferredType);
  return out;
}

static std::string componentIndexName(const tensorium::IndexedCall *call) {
  if (!call || call->args.empty())
    return "?";
  auto *field = dynamic_cast<const tensorium::IndexedVar *>(call->args[0].get());
  if (!field || field->tensorIndexNames.size() != 1)
    return "?";
  return field->tensorIndexNames.front();
}

static std::unique_ptr<ExprIR>
lowerVectorLaplacianExpr(const tensorium::IndexedExpr *arg,
                         bool materializeImplicitContraction,
                         bool hasConnectionTensor,
                         const tensorium::ir::TensorType &resultType,
                         const std::string &dummyIndex) {
  auto first = std::make_unique<PartialDerivativeIR>(
      lowerIndexedExpr(arg, materializeImplicitContraction,
                       hasConnectionTensor),
      dummyIndex);
  first->exprType =
      lowering::makeTensorType(arg->inferredType.up, arg->inferredType.down + 1);
  auto second =
      std::make_unique<PartialDerivativeIR>(std::move(first), dummyIndex);
  second->exprType =
      lowering::makeTensorType(arg->inferredType.up, arg->inferredType.down + 2);
  auto trace = std::make_unique<TraceIR>(std::move(second));
  trace->tracedIndices = {dummyIndex};
  trace->exprType = resultType;
  return trace;
}

std::unique_ptr<ExprIR>
lowerIndexedExpr(const tensorium::IndexedExpr *e,
                bool materializeImplicitContraction,
                bool hasConnectionTensor) {
  using namespace tensorium;

  if (!e)
    return nullptr;

  if (auto n = dynamic_cast<const IndexedNumber *>(e)) {
    auto out = std::make_unique<NumberIR>(n->value);
    out->exprType = lowering::lowerTensorType(n->inferredType);
    return out;
  }

  if (auto v = dynamic_cast<const IndexedVar *>(e)) {
    VarKind k = VarKind::Field;
    int coord = -1;
    switch (v->kind) {
    case IndexedVarKind::Field:
      k = VarKind::Field;
      break;
    case IndexedVarKind::Parameter:
      k = VarKind::Param;
      break;
    case IndexedVarKind::Local:
      k = VarKind::Local;
      break;
    case IndexedVarKind::Coordinate:
      k = VarKind::Coord;
      coord = v->coordIndex;
      break;
    }

    auto out = std::make_unique<VarIR>(v->name, k);
    out->coordIndex = coord;
    out->tensorIndexNames = v->tensorIndexNames;
    out->exprType = lowering::lowerTensorType(v->inferredType);
    return out;
  }

  if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
    auto lhs = lowerIndexedExpr(b->lhs.get(), materializeImplicitContraction,
                                hasConnectionTensor);
    auto rhs = lowerIndexedExpr(b->rhs.get(), materializeImplicitContraction,
                                hasConnectionTensor);

    std::unique_ptr<ExprIR> out;
    if (b->op == '*' && hasTensorRank(b->lhs.get()) && hasTensorRank(b->rhs.get())) {
      auto product = std::make_unique<TensorProductIR>(std::move(lhs), std::move(rhs));
      product->exprType = lowering::lowerTensorType(b->inferredType);
      out = std::move(product);
    } else {
      auto binary = std::make_unique<BinaryIR>(std::string(1, b->op),
                                               std::move(lhs), std::move(rhs));
      binary->exprType = lowering::lowerTensorType(b->inferredType);
      out = std::move(binary);
    }

    if (materializeImplicitContraction && b->op == '*') {
      auto summed = collectRepeatedIndices(b);
      if (!summed.empty()) {
        auto contraction = std::make_unique<ContractionIR>(std::move(out));
        contraction->summedIndices = std::move(summed);
        contraction->exprType = lowering::lowerTensorType(b->inferredType);
        return contraction;
      }
    }
    return out;
  }

  if (auto c = dynamic_cast<const IndexedCall *>(e)) {
    std::string coordIndex;
    if (parsePartialDerivativeName(c->callee, coordIndex)) {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_derivative>");
      auto deriv = std::make_unique<PartialDerivativeIR>(
          lowerIndexedExpr(c->args[0].get(), materializeImplicitContraction,
                           hasConnectionTensor),
          coordIndex);
      deriv->exprType = lowering::lowerTensorType(c->inferredType);
      return deriv;
    }

    bool contra = false;
    if (parseCovariantDerivativeName(c->callee, contra, coordIndex)) {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_covariant_derivative>");
      auto deriv = std::make_unique<CovariantDerivativeIR>(
          lowerIndexedExpr(c->args[0].get(), materializeImplicitContraction,
                           hasConnectionTensor),
          coordIndex);
      deriv->contravariant = contra;
      deriv->hasConnectionTensor = hasConnectionTensor;
      deriv->exprType = lowering::lowerTensorType(c->inferredType);
      return deriv;
    }

    if (c->callee == "covariant_derivative") {
      if (c->args.size() < 2)
        return std::make_unique<CallIR>("<invalid_covariant_derivative>");
      if (!tryExtractIndexName(c->args[1].get(), coordIndex))
        coordIndex = "?";
      auto deriv = std::make_unique<CovariantDerivativeIR>(
          lowerIndexedExpr(c->args[0].get(), materializeImplicitContraction,
                           hasConnectionTensor),
          coordIndex);
      deriv->hasConnectionTensor = hasConnectionTensor;
      deriv->exprType = lowering::lowerTensorType(c->inferredType);
      return deriv;
    }

    if (c->callee == "gradient" || c->callee == "grad") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_gradient>");
      auto grad = std::make_unique<GradientIR>(
          lowerIndexedExpr(c->args[0].get(), materializeImplicitContraction,
                           hasConnectionTensor));
      grad->exprType = lowering::lowerTensorType(c->inferredType);
      return grad;
    }

    if (c->callee == "divergence" || c->callee == "div") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_divergence>");
      auto div = std::make_unique<DivergenceIR>(
          lowerIndexedExpr(c->args[0].get(), materializeImplicitContraction,
                           hasConnectionTensor));
      if (c->args.size() >= 2) {
        std::string idx;
        if (tryExtractIndexName(c->args[1].get(), idx))
          div->contractedIndex = idx;
      }
      div->exprType = lowering::lowerTensorType(c->inferredType);
      return div;
    }

    if (c->callee == "trace") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_trace>");
      auto trace = std::make_unique<TraceIR>(
          lowerIndexedExpr(c->args[0].get(), false, hasConnectionTensor));
      for (size_t i = 1; i < c->args.size(); ++i) {
        std::string idx;
        if (tryExtractIndexName(c->args[i].get(), idx))
          trace->tracedIndices.push_back(idx);
      }
      if (trace->tracedIndices.empty())
        trace->tracedIndices = collectRepeatedIndices(c->args[0].get());
      trace->exprType = lowering::lowerTensorType(c->inferredType);
      return trace;
    }

    if (c->callee == "vector_laplacian") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_vector_laplacian>");
      return lowerVectorLaplacianExpr(
          c->args[0].get(), materializeImplicitContraction,
          hasConnectionTensor, lowering::lowerTensorType(c->inferredType), "j");
    }

    if (c->callee == "york_vector_laplacian_diag") {
      if (c->args.empty())
        return std::make_unique<CallIR>(
            "<invalid_york_vector_laplacian_diag>");
      const std::string componentIndex = componentIndexName(c);
      if (!tensorium::core::isTensorIndexName(componentIndex))
        return std::make_unique<CallIR>(
            "<invalid_york_vector_laplacian_diag>");
      auto out = std::make_unique<CallIR>("york_vector_laplacian_diag");
      out->args.push_back(lowerIndexedExpr(c->args[0].get(),
                                           materializeImplicitContraction,
                                           hasConnectionTensor));
      out->exprType = lowering::lowerTensorType(c->inferredType);
      return out;
    }

    if (c->callee == "york_vector_laplacian") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_york_vector_laplacian>");
      const std::string componentIndex = componentIndexName(c);
      if (!tensorium::core::isTensorIndexName(componentIndex))
        return std::make_unique<CallIR>("<invalid_york_vector_laplacian>");
      auto out = std::make_unique<CallIR>("york_vector_laplacian");
      out->args.push_back(lowerIndexedExpr(c->args[0].get(),
                                           materializeImplicitContraction,
                                           hasConnectionTensor));
      out->exprType = lowering::lowerTensorType(c->inferredType);
      return out;
    }

    if (c->callee == "index_permute") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_index_permute>");
      auto permute = std::make_unique<IndexPermuteIR>(
          lowerIndexedExpr(c->args[0].get(), false, hasConnectionTensor),
          std::vector<std::string>{});
      for (size_t i = 1; i < c->args.size(); ++i) {
        std::string idx;
        if (tryExtractIndexName(c->args[i].get(), idx))
          permute->order.push_back(idx);
      }
      permute->exprType = lowering::lowerTensorType(c->inferredType);
      return permute;
    }

    if (c->callee == "index_rename") {
      if (c->args.size() != 3)
        return std::make_unique<CallIR>("<invalid_index_rename>");
      std::string from;
      std::string to;
      if (!tryExtractIndexName(c->args[1].get(), from) ||
          !tryExtractIndexName(c->args[2].get(), to)) {
        return std::make_unique<CallIR>("<invalid_index_rename>");
      }
      auto rename = std::make_unique<IndexRenameIR>(
          lowerIndexedExpr(c->args[0].get(), false, hasConnectionTensor),
          from, to);
      rename->exprType = lowering::lowerTensorType(c->inferredType);
      return rename;
    }

    if (c->callee == "contract") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_contract>");
      auto contraction = std::make_unique<ContractionIR>(
          lowerIndexedExpr(c->args[0].get(), false, hasConnectionTensor));
      contraction->summedIndices = collectRepeatedIndices(c->args[0].get());
      contraction->exprType = lowering::lowerTensorType(c->inferredType);
      return contraction;
    }

    if (c->callee == "christoffel") {
      return lowerChristoffelBuiltin(c);
    }

    auto out = std::make_unique<CallIR>(c->callee);
    out->isExtern = c->isExtern;
    out->externArity = c->declaredArity;
    out->returnType = lowering::lowerTensorType(c->returnType);
    out->paramTypes.reserve(c->paramTypes.size());
    for (const auto &paramType : c->paramTypes)
      out->paramTypes.push_back(lowering::lowerTensorType(paramType));
    out->args.reserve(c->args.size());
    for (const auto &a : c->args)
      out->args.push_back(
          lowerIndexedExpr(a.get(), materializeImplicitContraction,
                           hasConnectionTensor));
    out->exprType = lowering::lowerTensorType(c->inferredType);
    return out;
  }

  return std::make_unique<CallIR>("<unknown>");
}

} // namespace tensorium::backend
