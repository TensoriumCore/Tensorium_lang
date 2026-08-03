#include "tensorium/Validation/IRVerifier.hpp"

#include "tensorium/Core/IndexSet.h"

#include <algorithm>
#include <string>

namespace tensorium::validation {
namespace {

struct VerifyContext {
  bool connectionAvailable = false;
  ValidationResult *result = nullptr;
};

static void emitError(VerifyContext &ctx, const std::string &message) {
  ctx.result->diags.push_back({Diagnostic::Kind::Error, message});
}

static bool hasConnectionTensor(const backend::ModuleIR &module) {
  for (const auto &field : module.fields) {
    if ((field.name == "Gamma" || field.name == "GammaU" ||
         field.name == "Christoffel") &&
        field.tensorType.rank() == 3) {
      return true;
    }
  }
  return false;
}

static void verifyExpr(const backend::ExprIR *expr, VerifyContext &ctx) {
  using backend::ExprIR;
  if (!expr) {
    emitError(ctx, "IR verifier: null expression node");
    return;
  }

  switch (expr->kind) {
  case ExprIR::Kind::Number:
  case ExprIR::Kind::Var:
    return;
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const backend::BinaryIR *>(expr);
    verifyExpr(bin->lhs.get(), ctx);
    verifyExpr(bin->rhs.get(), ctx);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<const backend::CallIR *>(expr);
    for (const auto &arg : call->args)
      verifyExpr(arg.get(), ctx);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const backend::TensorProductIR *>(expr);
    verifyExpr(prod->lhs.get(), ctx);
    verifyExpr(prod->rhs.get(), ctx);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<const backend::ContractionIR *>(expr);
    verifyExpr(ctr->in.get(), ctx);

    if (ctr->summedIndices.empty()) {
      emitError(ctx, "IR verifier: contraction has no summed indices");
      return;
    }

    if (!std::is_sorted(ctr->summedIndices.begin(), ctr->summedIndices.end())) {
      emitError(ctx, "IR verifier: contraction indices are not canonicalized");
    }
    if (std::adjacent_find(ctr->summedIndices.begin(),
                           ctr->summedIndices.end()) !=
        ctr->summedIndices.end()) {
      emitError(ctx, "IR verifier: contraction indices must be unique");
    }

    for (const auto &idx : ctr->summedIndices) {
      if (!core::isTensorIndexName(idx)) {
        emitError(ctx, "IR verifier: invalid contraction index '" + idx + "'");
      }
    }
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    verifyExpr(rename->in.get(), ctx);
    if (!core::isTensorIndexName(rename->from) ||
        !core::isTensorIndexName(rename->to)) {
      emitError(ctx, "IR verifier: invalid index_rename arguments");
    }
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<const backend::IndexPermuteIR *>(expr);
    verifyExpr(perm->in.get(), ctx);
    for (const auto &idx : perm->order) {
      if (!core::isTensorIndexName(idx)) {
        emitError(ctx,
                  "IR verifier: invalid index_permute index '" + idx + "'");
      }
    }
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const backend::TraceIR *>(expr);
    verifyExpr(trace->in.get(), ctx);
    if (trace->tracedIndices.empty()) {
      emitError(ctx, "IR verifier: trace has no traced indices");
      return;
    }
    for (const auto &idx : trace->tracedIndices) {
      if (!core::isTensorIndexName(idx)) {
        emitError(ctx, "IR verifier: invalid trace index '" + idx + "'");
      }
    }
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<const backend::PartialDerivativeIR *>(expr);
    verifyExpr(diff->in.get(), ctx);
    if (!core::isSpatialIndexName(diff->coordIndex)) {
      emitError(ctx, "IR verifier: invalid partial derivative index '" +
                         diff->coordIndex + "'");
    }
    return;
  }
  case ExprIR::Kind::Gradient:
    emitError(ctx, "IR verifier: uncanonicalized gradient operation");
    return;
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<const backend::CovariantDerivativeIR *>(expr);
    verifyExpr(diff->in.get(), ctx);
    if (!core::isSpatialIndexName(diff->derivIndex)) {
      emitError(ctx, "IR verifier: invalid covariant derivative index '" +
                         diff->derivIndex + "'");
    }
    if (!diff->hasConnectionTensor) {
      emitError(ctx,
                "IR verifier: covariant derivative requires connection tensor");
    }
    return;
  }
  case ExprIR::Kind::Divergence:
    emitError(ctx, "IR verifier: uncanonicalized divergence operation");
    return;
  }
}

} // namespace

ValidationResult verifyIR(const backend::ModuleIR &module) {
  ValidationResult result;
  VerifyContext ctx;
  ctx.connectionAvailable = hasConnectionTensor(module);
  ctx.result = &result;

  for (const auto &evolution : module.evolutions) {
    for (const auto &temp : evolution.temporaries)
      verifyExpr(temp.rhs.get(), ctx);
    for (const auto &equation : evolution.equations)
      verifyExpr(equation.rhs.get(), ctx);
  }

  if (module.constraintProblem) {
    const auto &problem = *module.constraintProblem;
    for (const auto &equation : problem.equations)
      verifyExpr(equation.residual.get(), ctx);
    for (const auto &boundary : problem.boundaries)
      for (const auto &condition : boundary.conditions)
        verifyExpr(condition.rhs.get(), ctx);
    for (const auto &seed : problem.seeds)
      verifyExpr(seed.rhs.get(), ctx);
    if (problem.cttReconstruction.enabled)
      verifyExpr(problem.cttReconstruction.meanCurvature.get(), ctx);
  }

  return result;
}

} // namespace tensorium::validation
