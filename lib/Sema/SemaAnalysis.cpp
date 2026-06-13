#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Core/TensorTypes.hpp"
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

static bool sameTensorShape(const FieldDecl &field,
                            const TensorTypeDesc &desc) {
  return field.up == desc.up && field.down == desc.down;
}

static size_t tensorRank(const TensorTypeDesc &desc) {
  return static_cast<size_t>(desc.up + desc.down);
}

static void collectAstFieldRefs(const Expr *expr,
                                const std::unordered_map<std::string,
                                                         const FieldDecl *> &fields,
                                std::unordered_set<std::string> &out) {
  if (!expr)
    return;
  if (auto var = dynamic_cast<const VarExpr *>(expr)) {
    if (fields.count(var->name))
      out.insert(var->name);
    return;
  }
  if (auto idx = dynamic_cast<const IndexedVarExpr *>(expr)) {
    if (fields.count(idx->base))
      out.insert(idx->base);
    return;
  }
  if (auto bin = dynamic_cast<const BinaryExpr *>(expr)) {
    collectAstFieldRefs(bin->lhs.get(), fields, out);
    collectAstFieldRefs(bin->rhs.get(), fields, out);
    return;
  }
  if (auto call = dynamic_cast<const CallExpr *>(expr)) {
    for (const auto &arg : call->args)
      collectAstFieldRefs(arg.get(), fields, out);
    return;
  }
  if (auto paren = dynamic_cast<const ParenExpr *>(expr))
    collectAstFieldRefs(paren->inner.get(), fields, out);
}

static void seedCoordinateSymbols(const SimulationConfig *sim,
                                  std::unordered_map<std::string, int> &coords) {
  if (!sim)
    return;

  auto add = [&](const std::string &name, int axis) {
    if (sim->dimension > axis)
      coords[name] = axis;
  };

  if (sim->coordinates == CoordinateSystem::Cartesian) {
    add("x", 0);
    add("y", 1);
    add("z", 2);
    return;
  }

  if (sim->coordinates == CoordinateSystem::Spherical) {
    add("r", 0);
    add("theta", 1);
    add("phi", 2);
    return;
  }

  if (sim->coordinates == CoordinateSystem::Cylindrical) {
    add("rho", 0);
    add("phi", 1);
    add("z", 2);
  }
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
  seedCoordinateSymbols(prog.simulation.get(), coordIndex);
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
        core::makeTensorTypeDesc(rhsType.up, rhsType.down);
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

IndexedEvolution
SemanticAnalyzer::analyzeConstraint(const ConstraintDecl &decl) {
  LocalScopeGuard localsScope(
      locals, std::unordered_map<std::string, TensorTypeDesc>{});
  coordIndex.clear();
  seedCoordinateSymbols(prog.simulation.get(), coordIndex);
  locals.clear();

  IndexedEvolution out;
  out.name = decl.name;

  const bool hasExplicitRoles =
      !decl.unknowns.empty() || !decl.freeFields.empty();
  std::unordered_set<std::string> unknownNames;
  std::unordered_set<std::string> freeNames;
  auto validateRole = [&](const ConstraintFieldRoleDecl &role,
                          const char *roleName,
                          std::unordered_set<std::string> &names) {
    if (!names.insert(role.name).second)
      throw std::runtime_error(std::string("constraints ") + roleName +
                               " redeclared: " + role.name);
    auto it = fields.find(role.name);
    if (it == fields.end()) {
      throw std::runtime_error(std::string("constraints ") + roleName +
                               " references unknown field '" + role.name + "'");
    }
    if (!sameTensorShape(*it->second, role.type)) {
      throw std::runtime_error(std::string("constraints ") + roleName +
                               " type mismatch for field '" + role.name + "'");
    }
    if (role.indices.size() != tensorRank(role.type)) {
      throw std::runtime_error(std::string("constraints ") + roleName +
                               " index count mismatch for field '" +
                               role.name + "'");
    }
  };
  for (const auto &unknown : decl.unknowns)
    validateRole(unknown, "unknown", unknownNames);
  for (const auto &freeField : decl.freeFields)
    validateRole(freeField, "free", freeNames);
  for (const auto &name : unknownNames) {
    if (freeNames.count(name)) {
      throw std::runtime_error("constraints field cannot be both unknown and "
                               "free: " +
                               name);
    }
  }

  for (const auto &tmp : decl.tempAssignments)
    for (const auto &idx : tmp.lhs.indices) {
      validateSpatialIndex(idx);
      coordIndex[idx] = -1;
    }

  for (const auto &eq : decl.residuals)
    for (const auto &idx : eq.indices) {
      validateSpatialIndex(idx);
      coordIndex[idx] = -1;
    }

  for (const auto &eq : decl.residuals) {
    if (!eq.unknownFieldName.empty() && !fields.count(eq.unknownFieldName)) {
      throw std::runtime_error("Unknown field in constraints residual unknown: " +
                               eq.unknownFieldName);
    }
    if (hasExplicitRoles) {
      if (eq.unknownFieldName.empty()) {
        throw std::runtime_error("constraints residual '" + eq.fieldName +
                                 "' requires `for <unknown>` when unknown/free "
                                 "roles are declared");
      }
      if (!unknownNames.count(eq.unknownFieldName)) {
        throw std::runtime_error("constraints residual '" + eq.fieldName +
                                 "' references non-unknown field '" +
                                 eq.unknownFieldName + "'");
      }
    }
  }

  if (hasExplicitRoles) {
    std::unordered_set<std::string> usedFields;
    for (const auto &tmp : decl.tempAssignments)
      collectAstFieldRefs(tmp.rhs.get(), fields, usedFields);
    for (const auto &eq : decl.residuals)
      collectAstFieldRefs(eq.rhs.get(), fields, usedFields);
    for (const std::string &fieldName : usedFields) {
      if (!unknownNames.count(fieldName) && !freeNames.count(fieldName)) {
        throw std::runtime_error("constraints field '" + fieldName +
                                 "' must be declared unknown or free");
      }
    }
  }

  std::unordered_set<std::string> residualNames;
  for (const auto &eq : decl.residuals)
    residualNames.insert(eq.fieldName);
  for (const auto &boundary : decl.boundaryConditions) {
    if (!residualNames.count(boundary.residualName)) {
      throw std::runtime_error("boundary condition references unknown residual '" +
                               boundary.residualName + "'");
    }
    if (boundary.face != "lower_x1" && boundary.face != "upper_x1" &&
        boundary.face != "lower_x2" && boundary.face != "upper_x2" &&
        boundary.face != "lower_x3" && boundary.face != "upper_x3") {
      throw std::runtime_error("unknown boundary face '" + boundary.face + "'");
    }
    if (boundary.kind != "dirichlet" && boundary.kind != "robin") {
      throw std::runtime_error("unknown boundary kind '" + boundary.kind + "'");
    }
    if (boundary.derivativeKind != "normal" &&
        boundary.derivativeKind != "radial") {
      throw std::runtime_error("unknown boundary derivative kind '" +
                               boundary.derivativeKind + "'");
    }
    auto validateBoundaryCoordinate = [](const std::string &coordinate) {
      return coordinate.empty() || coordinate == "x1" || coordinate == "x2" ||
             coordinate == "x3" || coordinate == "x" || coordinate == "y" ||
             coordinate == "z" || coordinate == "r" || coordinate == "radius";
    };
    if (!validateBoundaryCoordinate(boundary.valueCoefficientCoordinate)) {
      throw std::runtime_error("unknown boundary coordinate '" +
                               boundary.valueCoefficientCoordinate + "'");
    }
    if (!validateBoundaryCoordinate(
            boundary.normalDerivativeCoefficientCoordinate)) {
      throw std::runtime_error(
          "unknown boundary coordinate '" +
          boundary.normalDerivativeCoefficientCoordinate + "'");
    }
    if (!validateBoundaryCoordinate(boundary.targetValueCoordinate)) {
      throw std::runtime_error("unknown boundary coordinate '" +
                               boundary.targetValueCoordinate + "'");
    }
  }

  TensorTypeChecker checker(hasConnectionTensor);
  std::unordered_set<std::string> tempNames;

  for (const auto &tmp : decl.tempAssignments) {
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

  for (const auto &tmp : decl.tempAssignments) {
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
        core::makeTensorTypeDesc(rhsType.up, rhsType.down);
  }

  for (const auto &eq : decl.residuals) {

    auto it = fields.find(eq.fieldName);
    if (it == fields.end())
      throw std::runtime_error("Unknown field in constraints residual: " +
                               eq.fieldName);

    const FieldDecl *fd = it->second;
    size_t expectedRank = static_cast<size_t>(fd->up + fd->down);

    if (eq.indices.size() != expectedRank) {
      throw std::runtime_error(
          "Wrong number of indices in constraints residual for field '" +
          eq.fieldName + "': expected " + std::to_string(expectedRank) +
          ", got " + std::to_string(eq.indices.size()));
    }

    IndexedEvolutionEq ie;
    ie.fieldName = eq.fieldName;
    ie.unknownFieldName = eq.unknownFieldName;
    ie.indices = eq.indices;
    ie.rhs = transformExpr(eq.rhs.get());

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
