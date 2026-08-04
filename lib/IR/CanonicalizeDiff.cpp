#include "tensorium/Validation/IRCanonicalize.hpp"

#include "tensorium/Core/IndexSet.h"

#include <memory>
#include <string>
#include <utility>

namespace tensorium::validation {
namespace {

static bool hasConnectionTensor(const backend::ModuleIR &module) {
  for (const auto &field : module.fields) {
    if ((field.tensorType.up == 1 && field.tensorType.down == 2) ||
        (field.tensorType.rank() == 3 &&
         (field.name == "Gamma" || field.name == "GammaU" ||
          field.name == "Christoffel"))) {
      return true;
    }
  }
  return false;
}

static std::string defaultDerivativeIndex() {
  return std::string(1, core::kTensorIndices.front());
}

static void canonicalizeExpr(std::unique_ptr<backend::ExprIR> &expr,
                             bool connectionAvailable) {
  using backend::ExprIR;
  if (!expr)
    return;

  switch (expr->kind) {
  case ExprIR::Kind::Number:
  case ExprIR::Kind::Var:
    return;
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<backend::BinaryIR *>(expr.get());
    canonicalizeExpr(bin->lhs, connectionAvailable);
    canonicalizeExpr(bin->rhs, connectionAvailable);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<backend::CallIR *>(expr.get());
    for (auto &arg : call->args)
      canonicalizeExpr(arg, connectionAvailable);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<backend::TensorProductIR *>(expr.get());
    canonicalizeExpr(prod->lhs, connectionAvailable);
    canonicalizeExpr(prod->rhs, connectionAvailable);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<backend::ContractionIR *>(expr.get());
    canonicalizeExpr(ctr->in, connectionAvailable);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<backend::IndexRenameIR *>(expr.get());
    canonicalizeExpr(rename->in, connectionAvailable);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<backend::IndexPermuteIR *>(expr.get());
    canonicalizeExpr(perm->in, connectionAvailable);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<backend::TraceIR *>(expr.get());
    canonicalizeExpr(trace->in, connectionAvailable);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<backend::PartialDerivativeIR *>(expr.get());
    canonicalizeExpr(diff->in, connectionAvailable);
    if (diff->coordIndex.empty())
      diff->coordIndex = defaultDerivativeIndex();
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<backend::GradientIR *>(expr.get());
    canonicalizeExpr(grad->in, connectionAvailable);

    auto partial = std::make_unique<backend::PartialDerivativeIR>(
        std::move(grad->in), defaultDerivativeIndex());
    partial->exprType = grad->exprType;
    expr = std::move(partial);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<backend::CovariantDerivativeIR *>(expr.get());
    canonicalizeExpr(diff->in, connectionAvailable);
    if (diff->derivIndex.empty())
      diff->derivIndex = defaultDerivativeIndex();
    diff->hasConnectionTensor =
        diff->hasConnectionTensor || connectionAvailable;
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<backend::DivergenceIR *>(expr.get());
    canonicalizeExpr(div->in, connectionAvailable);

    std::string derivIndex = div->contractedIndex;
    if (derivIndex.empty())
      derivIndex = defaultDerivativeIndex();

    auto cov = std::make_unique<backend::CovariantDerivativeIR>(
        std::move(div->in), derivIndex);
    cov->hasConnectionTensor = connectionAvailable;
    if (cov->in) {
      cov->exprType = cov->in->exprType;
      cov->exprType.down += 1;
    } else {
      cov->exprType = div->exprType;
    }

    auto contraction = std::make_unique<backend::ContractionIR>(std::move(cov));
    contraction->summedIndices.push_back(derivIndex);
    contraction->exprType = div->exprType;
    expr = std::move(contraction);
    return;
  }
  }
}

} // namespace

void canonicalizeDifferentialIR(backend::ModuleIR &module) {
  const bool connectionAvailable = hasConnectionTensor(module);

  for (auto &evolution : module.evolutions) {
    for (auto &temp : evolution.temporaries)
      canonicalizeExpr(temp.rhs, connectionAvailable);
    for (auto &equation : evolution.equations)
      canonicalizeExpr(equation.rhs, connectionAvailable);
  }
  if (module.constraintProblem) {
    auto &problem = *module.constraintProblem;
    const bool constraintConnectionAvailable =
        connectionAvailable || problem.geometry.enabled;
    if (problem.geometry.enabled) {
      canonicalizeExpr(problem.geometry.radialScale,
                       constraintConnectionAvailable);
      canonicalizeExpr(problem.geometry.tangentialScale,
                       constraintConnectionAvailable);
    }
    for (auto &equation : problem.equations)
      canonicalizeExpr(equation.residual, constraintConnectionAvailable);
    for (auto &boundary : problem.boundaries)
      for (auto &condition : boundary.conditions)
        canonicalizeExpr(condition.rhs, constraintConnectionAvailable);
    for (auto &seed : problem.seeds)
      canonicalizeExpr(seed.rhs, constraintConnectionAvailable);
    if (problem.cttReconstruction.enabled)
      canonicalizeExpr(problem.cttReconstruction.meanCurvature,
                       constraintConnectionAvailable);
  }
}

} // namespace tensorium::validation
