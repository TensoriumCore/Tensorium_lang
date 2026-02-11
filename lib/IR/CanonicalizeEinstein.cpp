#include "tensorium/Validation/IRCanonicalize.hpp"

#include "tensorium/Core/IndexSet.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tensorium::validation {
namespace {

static void sortAndUnique(std::vector<std::string> &indices) {
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
}

static void collectIndexUses(const backend::ExprIR *expr,
                             std::map<std::string, int> &counts) {
  using backend::ExprIR;
  if (!expr)
    return;

  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return;
  case ExprIR::Kind::Var: {
    auto *var = static_cast<const backend::VarIR *>(expr);
    for (const auto &name : var->tensorIndexNames) {
      if (core::isTensorIndexName(name))
        counts[name] += 1;
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const backend::BinaryIR *>(expr);
    collectIndexUses(bin->lhs.get(), counts);
    collectIndexUses(bin->rhs.get(), counts);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<const backend::CallIR *>(expr);
    for (const auto &arg : call->args)
      collectIndexUses(arg.get(), counts);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const backend::TensorProductIR *>(expr);
    collectIndexUses(prod->lhs.get(), counts);
    collectIndexUses(prod->rhs.get(), counts);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<const backend::ContractionIR *>(expr);
    collectIndexUses(ctr->in.get(), counts);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    collectIndexUses(rename->in.get(), counts);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<const backend::IndexPermuteIR *>(expr);
    collectIndexUses(perm->in.get(), counts);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const backend::TraceIR *>(expr);
    collectIndexUses(trace->in.get(), counts);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<const backend::PartialDerivativeIR *>(expr);
    collectIndexUses(diff->in.get(), counts);
    if (core::isTensorIndexName(diff->coordIndex))
      counts[diff->coordIndex] += 1;
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<const backend::GradientIR *>(expr);
    collectIndexUses(grad->in.get(), counts);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<const backend::CovariantDerivativeIR *>(expr);
    collectIndexUses(diff->in.get(), counts);
    if (core::isTensorIndexName(diff->derivIndex))
      counts[diff->derivIndex] += 1;
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<const backend::DivergenceIR *>(expr);
    collectIndexUses(div->in.get(), counts);
    if (core::isTensorIndexName(div->contractedIndex))
      counts[div->contractedIndex] += 1;
    return;
  }
  }
}

static std::vector<std::string>
collectRepeatedIndices(const backend::ExprIR *expr) {
  std::map<std::string, int> counts;
  collectIndexUses(expr, counts);

  std::vector<std::string> repeated;
  for (const auto &[name, count] : counts) {
    if (count >= 2)
      repeated.push_back(name);
  }
  return repeated;
}

static void collectUsedIndexNames(const backend::ExprIR *expr,
                                  std::set<std::string> &used) {
  using backend::ExprIR;
  if (!expr)
    return;

  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return;
  case ExprIR::Kind::Var: {
    auto *var = static_cast<const backend::VarIR *>(expr);
    for (const auto &name : var->tensorIndexNames) {
      if (core::isTensorIndexName(name))
        used.insert(name);
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const backend::BinaryIR *>(expr);
    collectUsedIndexNames(bin->lhs.get(), used);
    collectUsedIndexNames(bin->rhs.get(), used);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<const backend::CallIR *>(expr);
    for (const auto &arg : call->args)
      collectUsedIndexNames(arg.get(), used);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const backend::TensorProductIR *>(expr);
    collectUsedIndexNames(prod->lhs.get(), used);
    collectUsedIndexNames(prod->rhs.get(), used);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<const backend::ContractionIR *>(expr);
    collectUsedIndexNames(ctr->in.get(), used);
    for (const auto &name : ctr->summedIndices) {
      if (core::isTensorIndexName(name))
        used.insert(name);
    }
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    collectUsedIndexNames(rename->in.get(), used);
    if (core::isTensorIndexName(rename->from))
      used.insert(rename->from);
    if (core::isTensorIndexName(rename->to))
      used.insert(rename->to);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<const backend::IndexPermuteIR *>(expr);
    collectUsedIndexNames(perm->in.get(), used);
    for (const auto &name : perm->order) {
      if (core::isTensorIndexName(name))
        used.insert(name);
    }
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const backend::TraceIR *>(expr);
    collectUsedIndexNames(trace->in.get(), used);
    for (const auto &name : trace->tracedIndices) {
      if (core::isTensorIndexName(name))
        used.insert(name);
    }
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<const backend::PartialDerivativeIR *>(expr);
    collectUsedIndexNames(diff->in.get(), used);
    if (core::isTensorIndexName(diff->coordIndex))
      used.insert(diff->coordIndex);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<const backend::GradientIR *>(expr);
    collectUsedIndexNames(grad->in.get(), used);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<const backend::CovariantDerivativeIR *>(expr);
    collectUsedIndexNames(diff->in.get(), used);
    if (core::isTensorIndexName(diff->derivIndex))
      used.insert(diff->derivIndex);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<const backend::DivergenceIR *>(expr);
    collectUsedIndexNames(div->in.get(), used);
    if (core::isTensorIndexName(div->contractedIndex))
      used.insert(div->contractedIndex);
    return;
  }
  }
}

static std::string pickFreshIndex(const std::set<std::string> &used) {
  for (char c : core::kTensorIndices) {
    std::string candidate(1, c);
    if (used.find(candidate) == used.end())
      return candidate;
  }
  return {};
}

static void canonicalizeExpr(std::unique_ptr<backend::ExprIR> &expr) {
  using backend::ExprIR;
  if (!expr)
    return;

  switch (expr->kind) {
  case ExprIR::Kind::Number:
  case ExprIR::Kind::Var:
    return;
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<backend::BinaryIR *>(expr.get());
    canonicalizeExpr(bin->lhs);
    canonicalizeExpr(bin->rhs);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<backend::CallIR *>(expr.get());
    for (auto &arg : call->args)
      canonicalizeExpr(arg);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<backend::TensorProductIR *>(expr.get());
    canonicalizeExpr(prod->lhs);
    canonicalizeExpr(prod->rhs);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<backend::ContractionIR *>(expr.get());
    canonicalizeExpr(ctr->in);
    sortAndUnique(ctr->summedIndices);

    if (ctr->summedIndices.empty()) {
      expr = std::move(ctr->in);
      return;
    }

    std::map<std::string, int> counts;
    collectIndexUses(ctr->in.get(), counts);

    std::set<std::string> used;
    collectUsedIndexNames(ctr->in.get(), used);
    for (const auto &name : ctr->summedIndices)
      used.insert(name);

    for (auto &name : ctr->summedIndices) {
      auto it = counts.find(name);
      if (it == counts.end() || it->second <= 2)
        continue;

      std::string fresh = pickFreshIndex(used);
      if (fresh.empty() || fresh == name)
        continue;

      auto rename =
          std::make_unique<backend::IndexRenameIR>(std::move(ctr->in), name, fresh);
      rename->exprType = rename->in ? rename->in->exprType : ctr->exprType;
      ctr->in = std::move(rename);
      name = fresh;
      used.insert(fresh);
    }

    sortAndUnique(ctr->summedIndices);

    if (ctr->in && ctr->in->kind != ExprIR::Kind::TensorProduct) {
      auto trace = std::make_unique<backend::TraceIR>(std::move(ctr->in));
      trace->tracedIndices = ctr->summedIndices;
      trace->exprType = ctr->exprType;
      expr = std::move(trace);
    }
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<backend::IndexRenameIR *>(expr.get());
    canonicalizeExpr(rename->in);
    if (rename->from.empty() || rename->to.empty() || rename->from == rename->to)
      expr = std::move(rename->in);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<backend::IndexPermuteIR *>(expr.get());
    canonicalizeExpr(perm->in);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<backend::TraceIR *>(expr.get());
    canonicalizeExpr(trace->in);

    if (trace->tracedIndices.empty())
      trace->tracedIndices = collectRepeatedIndices(trace->in.get());
    sortAndUnique(trace->tracedIndices);

    if (trace->tracedIndices.empty()) {
      expr = std::move(trace->in);
      return;
    }

    if (trace->in && trace->in->kind == ExprIR::Kind::TensorProduct) {
      auto ctr = std::make_unique<backend::ContractionIR>(std::move(trace->in));
      ctr->summedIndices = trace->tracedIndices;
      ctr->exprType = trace->exprType;
      expr = std::move(ctr);
    }
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<backend::PartialDerivativeIR *>(expr.get());
    canonicalizeExpr(diff->in);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<backend::GradientIR *>(expr.get());
    canonicalizeExpr(grad->in);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<backend::CovariantDerivativeIR *>(expr.get());
    canonicalizeExpr(diff->in);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<backend::DivergenceIR *>(expr.get());
    canonicalizeExpr(div->in);
    return;
  }
  }
}

} // namespace

void canonicalizeEinsteinIR(backend::ModuleIR &module) {
  for (auto &evolution : module.evolutions) {
    for (auto &temp : evolution.temporaries)
      canonicalizeExpr(temp.rhs);
    for (auto &equation : evolution.equations)
      canonicalizeExpr(equation.rhs);
  }
}

} // namespace tensorium::validation
