#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Sema/tensor_type_checker.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tensorium {
namespace {
struct LocalScopeGuard {
  std::unordered_map<std::string, TensorTypeDesc> &locals;
  std::unordered_map<std::string, TensorTypeDesc> saved;

  LocalScopeGuard(std::unordered_map<std::string, TensorTypeDesc> &localsIn,
                  std::unordered_map<std::string, TensorTypeDesc> replacement)
      : locals(localsIn), saved(localsIn) {
    locals = std::move(replacement);
  }

  ~LocalScopeGuard() { locals = std::move(saved); }
};

static TensorKind deduceKind(int up, int down) {
  if (up == 0 && down == 0)
    return TensorKind::Scalar;
  if (up == 1 && down == 0)
    return TensorKind::Vector;
  if (up == 0 && down == 1)
    return TensorKind::Covector;
  if (up == 0 && down == 2)
    return TensorKind::CovTensor2;
  if (up == 2 && down == 0)
    return TensorKind::ConTensor2;
  if (up == 0 && down == 3)
    return TensorKind::CovTensor3;
  if (up == 3 && down == 0)
    return TensorKind::ConTensor3;
  if (up == 0 && down == 4)
    return TensorKind::CovTensor4;
  if (up == 4 && down == 0)
    return TensorKind::ConTensor4;
  return TensorKind::MixedTensor;
}
} // namespace

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
  LocalScopeGuard localsScope(
      locals, std::unordered_map<std::string, TensorTypeDesc>{});
  coordIndex.clear();
  locals.clear();

  IndexedEvolution out;
  out.name = evo.name;

  for (const auto &tmp : evo.tempAssignments)
    for (const auto &idx : tmp.lhs.indices) {
      validateSpatialIndex(idx);
      coordIndex[idx] = -1;
    }

  for (const auto &eq : evo.equations)
    for (const auto &idx : eq.indices) {
      validateSpatialIndex(idx);
      coordIndex[idx] = -1;
    }

  TensorTypeChecker checker(hasConnectionTensor);
  std::unordered_set<std::string> tempNames;

  for (const auto &tmp : evo.tempAssignments) {
    if (params.count(tmp.lhs.base)) {
      throw std::runtime_error("Cannot redeclare parameter '" + tmp.lhs.base +
                               "' as local");
    }

    if (fields.count(tmp.lhs.base)) {
      throw std::runtime_error("Cannot redeclare field '" + tmp.lhs.base +
                               "' as local");
    }

    if (!tempNames.insert(tmp.lhs.base).second) {
      throw std::runtime_error("Cannot redeclare local '" + tmp.lhs.base + "'");
    }

    if (tmp.lhs.indices.empty())
      locals[tmp.lhs.base] = TensorTypeDesc{TensorKind::Scalar, 0, 0};
  }

  for (const auto &tmp : evo.tempAssignments) {
    auto rhs = transformExpr(tmp.rhs.get());
    TensorType rhsType = checker.infer(rhs.get());
    TensorType lhsType{rhsType.up, rhsType.down};
    checker.checkAssignmentVariance(lhsType, tmp.lhs.indices, rhs.get());

    IndexedAssignment ia;
    ia.tensor = tmp.lhs.base;
    ia.indices = tmp.lhs.indices;
    for (auto &idx : tmp.lhs.indices)
      ia.indexOffsets.push_back(resolveIndex(idx));
    ia.rhs = std::move(rhs);
    out.temp.push_back(std::move(ia));

    locals[tmp.lhs.base] =
        TensorTypeDesc{deduceKind(rhsType.up, rhsType.down), rhsType.up,
                       rhsType.down};
  }

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

  return out;
}

IndexedPrint SemanticAnalyzer::analyzePrint(const PrintDecl &decl) {
  LocalScopeGuard localsScope(
      locals, std::unordered_map<std::string, TensorTypeDesc>{});
  coordIndex.clear();
  locals.clear();

  auto isEvolutionTemporary = [this](const std::string &name) {
    for (const auto &evo : prog.evolutions)
      for (const auto &tmp : evo.tempAssignments)
        if (tmp.lhs.base == name)
          return true;
    return false;
  };

  auto rejectLocalTemporary = [&](const std::string &name) {
    if (isEvolutionTemporary(name)) {
      throw std::runtime_error(
          "print() cannot print evolution temporary '" + name +
          "' from top level; print a declared field or materialize the "
          "temporary as a field");
    }
  };

  if (const auto *var = dynamic_cast<const VarExpr *>(decl.expr.get()))
    rejectLocalTemporary(var->name);
  if (const auto *indexed =
          dynamic_cast<const IndexedVarExpr *>(decl.expr.get()))
    rejectLocalTemporary(indexed->base);

  TensorTypeChecker checker(hasConnectionTensor);
  IndexedPrint out;
  out.expr = transformExpr(decl.expr.get());
  TensorType type = checker.infer(out.expr.get());

  const auto *field = dynamic_cast<const IndexedVar *>(out.expr.get());
  if (!field || field->kind != IndexedVarKind::Field) {
    throw std::runtime_error("print() currently expects a declared field");
  }

  if (type.rank() > 0 &&
      field->tensorIndexNames.size() != static_cast<size_t>(type.rank())) {
    throw std::runtime_error("print() of tensor field '" + field->name +
                             "' requires explicit indices");
  }

  return out;
}

} // namespace tensorium
