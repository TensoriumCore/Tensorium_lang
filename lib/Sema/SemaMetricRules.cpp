#include "tensorium/Sema/Sema.hpp"

#include <stdexcept>

namespace tensorium {

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

} // namespace tensorium
