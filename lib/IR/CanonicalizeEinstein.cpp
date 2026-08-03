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

static void filterTensorIndexNames(std::vector<std::string> &indices) {
  indices.erase(std::remove_if(indices.begin(), indices.end(),
                               [](const std::string &idx) {
                                 return !core::isTensorIndexName(idx);
                               }),
                indices.end());
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

static void collectFreeIndices(const backend::ExprIR *expr,
                               std::set<std::string> &free) {
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
        free.insert(name);
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const backend::BinaryIR *>(expr);
    collectFreeIndices(bin->lhs.get(), free);
    collectFreeIndices(bin->rhs.get(), free);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<const backend::CallIR *>(expr);
    for (const auto &arg : call->args)
      collectFreeIndices(arg.get(), free);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const backend::TensorProductIR *>(expr);
    collectFreeIndices(prod->lhs.get(), free);
    collectFreeIndices(prod->rhs.get(), free);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<const backend::ContractionIR *>(expr);
    collectFreeIndices(ctr->in.get(), free);
    for (const auto &name : ctr->summedIndices) {
      if (core::isTensorIndexName(name))
        free.erase(name);
    }
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    collectFreeIndices(rename->in.get(), free);
    if (core::isTensorIndexName(rename->from) &&
        core::isTensorIndexName(rename->to)) {
      auto it = free.find(rename->from);
      if (it != free.end()) {
        free.erase(it);
        free.insert(rename->to);
      }
    }
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<const backend::IndexPermuteIR *>(expr);
    collectFreeIndices(perm->in.get(), free);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const backend::TraceIR *>(expr);
    collectFreeIndices(trace->in.get(), free);
    for (const auto &name : trace->tracedIndices) {
      if (core::isTensorIndexName(name))
        free.erase(name);
    }
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<const backend::PartialDerivativeIR *>(expr);
    collectFreeIndices(diff->in.get(), free);
    if (core::isTensorIndexName(diff->coordIndex))
      free.insert(diff->coordIndex);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<const backend::GradientIR *>(expr);
    collectFreeIndices(grad->in.get(), free);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<const backend::CovariantDerivativeIR *>(expr);
    collectFreeIndices(diff->in.get(), free);
    if (core::isTensorIndexName(diff->derivIndex))
      free.insert(diff->derivIndex);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<const backend::DivergenceIR *>(expr);
    collectFreeIndices(div->in.get(), free);
    if (core::isTensorIndexName(div->contractedIndex))
      free.erase(div->contractedIndex);
    return;
  }
  }
}

static std::vector<std::string>
pickCanonicalDummyTargets(const std::set<std::string> &free, size_t needed) {
  std::vector<std::string> out;
  out.reserve(needed);

  for (char c : core::kTensorIndices) {
    if (out.size() >= needed)
      break;
    std::string candidate(1, c);
    if (free.find(candidate) == free.end())
      out.push_back(std::move(candidate));
  }
  return out;
}

static void remapIndexName(const std::map<std::string, std::string> &mapping,
                           std::string &name) {
  auto it = mapping.find(name);
  if (it != mapping.end())
    name = it->second;
}

static void remapIndexList(const std::map<std::string, std::string> &mapping,
                           std::vector<std::string> &indices) {
  for (auto &idx : indices)
    remapIndexName(mapping, idx);
}

static void
applyIndexMapping(backend::ExprIR *expr,
                  const std::map<std::string, std::string> &mapping) {
  using backend::ExprIR;
  if (!expr || mapping.empty())
    return;

  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return;
  case ExprIR::Kind::Var: {
    auto *var = static_cast<backend::VarIR *>(expr);
    remapIndexList(mapping, var->tensorIndexNames);
    return;
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<backend::BinaryIR *>(expr);
    applyIndexMapping(bin->lhs.get(), mapping);
    applyIndexMapping(bin->rhs.get(), mapping);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<backend::CallIR *>(expr);
    for (auto &arg : call->args)
      applyIndexMapping(arg.get(), mapping);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<backend::TensorProductIR *>(expr);
    applyIndexMapping(prod->lhs.get(), mapping);
    applyIndexMapping(prod->rhs.get(), mapping);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<backend::ContractionIR *>(expr);
    remapIndexList(mapping, ctr->summedIndices);
    applyIndexMapping(ctr->in.get(), mapping);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<backend::IndexRenameIR *>(expr);
    remapIndexName(mapping, rename->from);
    remapIndexName(mapping, rename->to);
    applyIndexMapping(rename->in.get(), mapping);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<backend::IndexPermuteIR *>(expr);
    remapIndexList(mapping, perm->order);
    applyIndexMapping(perm->in.get(), mapping);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<backend::TraceIR *>(expr);
    remapIndexList(mapping, trace->tracedIndices);
    applyIndexMapping(trace->in.get(), mapping);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<backend::PartialDerivativeIR *>(expr);
    remapIndexName(mapping, diff->coordIndex);
    applyIndexMapping(diff->in.get(), mapping);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<backend::GradientIR *>(expr);
    applyIndexMapping(grad->in.get(), mapping);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<backend::CovariantDerivativeIR *>(expr);
    remapIndexName(mapping, diff->derivIndex);
    applyIndexMapping(diff->in.get(), mapping);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<backend::DivergenceIR *>(expr);
    remapIndexName(mapping, div->contractedIndex);
    applyIndexMapping(div->in.get(), mapping);
    return;
  }
  }
}

static std::map<std::string, std::string>
buildDummyAlphaRenamingMap(const backend::ContractionIR &ctr) {
  std::set<std::string> free;
  collectFreeIndices(ctr.in.get(), free);
  for (const auto &idx : ctr.summedIndices)
    free.erase(idx);

  std::vector<std::string> targets =
      pickCanonicalDummyTargets(free, ctr.summedIndices.size());

  std::map<std::string, std::string> mapping;
  for (size_t i = 0; i < ctr.summedIndices.size(); ++i) {
    if (i >= targets.size())
      break;
    const std::string &from = ctr.summedIndices[i];
    const std::string &to = targets[i];
    if (from != to)
      mapping[from] = to;
  }
  return mapping;
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

    if (ctr->in && ctr->in->kind == ExprIR::Kind::Contraction) {
      auto inner = static_cast<backend::ContractionIR *>(ctr->in.get());
      for (const auto &idx : inner->summedIndices)
        ctr->summedIndices.push_back(idx);
      ctr->in = std::move(inner->in);
    }

    filterTensorIndexNames(ctr->summedIndices);
    sortAndUnique(ctr->summedIndices);

    if (ctr->summedIndices.empty()) {
      expr = std::move(ctr->in);
      return;
    }

    auto renameMap = buildDummyAlphaRenamingMap(*ctr);
    applyIndexMapping(ctr->in.get(), renameMap);
    remapIndexList(renameMap, ctr->summedIndices);
    sortAndUnique(ctr->summedIndices);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<backend::IndexRenameIR *>(expr.get());
    canonicalizeExpr(rename->in);
    if (!core::isTensorIndexName(rename->from) ||
        !core::isTensorIndexName(rename->to) || rename->from == rename->to) {
      expr = std::move(rename->in);
      return;
    }

    std::map<std::string, std::string> mapping;
    mapping.emplace(rename->from, rename->to);
    applyIndexMapping(rename->in.get(), mapping);
    expr = std::move(rename->in);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<backend::IndexPermuteIR *>(expr.get());
    canonicalizeExpr(perm->in);
    if (perm->in && perm->in->kind == ExprIR::Kind::IndexPermute) {
      auto *inner = static_cast<backend::IndexPermuteIR *>(perm->in.get());
      if (inner->order == perm->order) {
        expr = std::move(perm->in);
        return;
      }
    }
    if (perm->order.empty())
      expr = std::move(perm->in);
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

    auto ctr = std::make_unique<backend::ContractionIR>(std::move(trace->in));
    ctr->summedIndices = trace->tracedIndices;
    ctr->exprType = trace->exprType;
    expr = std::move(ctr);
    canonicalizeExpr(expr);
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
  if (module.constraintProblem) {
    auto &problem = *module.constraintProblem;
    for (auto &equation : problem.equations)
      canonicalizeExpr(equation.residual);
    for (auto &boundary : problem.boundaries)
      for (auto &condition : boundary.conditions)
        canonicalizeExpr(condition.rhs);
    for (auto &seed : problem.seeds)
      canonicalizeExpr(seed.rhs);
  }
}

} // namespace tensorium::validation
