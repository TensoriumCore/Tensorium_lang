#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Sema/CallSupport.hpp"
#include "tensorium/Sema/tensor_type_checker.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace tensorium {

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

    if (auto itu = unknowns.find(v->name); itu != unknowns.end()) {
      const ConstraintUnknownDecl *decl = itu->second;
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Unknown);
      iv->tensorKind = decl->type.kind;
      iv->up = decl->type.up;
      iv->down = decl->type.down;
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
    const FieldDecl *field = nullptr;
    const ConstraintUnknownDecl *unknown = nullptr;
    if (auto it = fields.find(iv->base); it != fields.end())
      field = it->second;
    if (auto it = unknowns.find(iv->base); it != unknowns.end())
      unknown = it->second;
    if (!field && !unknown)
      throw std::runtime_error("Unknown indexed tensor: " + iv->base);

    const int up = field ? field->up : unknown->type.up;
    const int down = field ? field->down : unknown->type.down;
    const TensorKind tensorKind = field ? field->kind : unknown->type.kind;
    size_t expected = static_cast<size_t>(up + down);

    if (iv->indices.size() != expected)
      throw std::runtime_error("Tensor '" + iv->base + "' expects " +
                               std::to_string(expected) + " indices, got " +
                               std::to_string(iv->indices.size()));

    auto out = std::make_unique<IndexedVar>(
        iv->base, field ? IndexedVarKind::Field : IndexedVarKind::Unknown);
    out->tensorKind = tensorKind;
    out->up = up;
    out->down = down;

    size_t pos = 0;
    for (auto &idx : iv->indices) {
      if (!coordIndex.count(idx)) {
        validateSpatialIndex(idx);
        coordIndex[idx] = -2;
      }
      int off = resolveIndex(idx);
      out->tensorIndices.push_back(off);
      out->tensorIndexNames.push_back(idx);
      bool isUp = pos < static_cast<size_t>(up);
      out->tensorIndexIsUp.push_back(isUp);
      ++pos;
    }
    return out;
  }

  if (auto c = dynamic_cast<const CallExpr *>(e)) {
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
                               "' conflicts with metric '" + metric.name + "'");
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
    if ((f.name == "Gamma" || f.name == "GammaU" || f.name == "Christoffel") &&
        (f.up + f.down) == 3) {
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
    syntheticMetricFields.push_back(fd);
    fields[m.name] = &syntheticMetricFields.back();
  }

  if (prog.initialData && prog.initialData->hasConstraintProblem) {
    for (const auto &unknown : prog.initialData->constraintProblem.unknowns) {
      if (params.count(unknown.name) || fields.count(unknown.name) ||
          metricNames.count(unknown.name)) {
        throw std::runtime_error("Name collision for constraint unknown '" +
                                 unknown.name + "'");
      }
      if (!unknowns.emplace(unknown.name, &unknown).second) {
        throw std::runtime_error("Constraint unknown redeclared: " +
                                 unknown.name);
      }
    }
  }

  const bool constraintOnly =
      prog.initialData && prog.initialData->hasConstraintProblem &&
      !prog.initialData->hasMetric4 && !prog.initialData->hasDecomposed &&
      prog.evolutions.empty();
  if (!prog.simulation) {
    simulationMissing = true;
    if (!constraintOnly) {
      if (mode == CompilationMode::Executable) {
        throw std::runtime_error(std::string(kErrMissingSimulationBlock) +
                                 ". Add `simulation { dimension = <N> "
                                 "resolution = [...] time { dt = ... "
                                 "integrator = ... } spatial { scheme = ... "
                                 "derivative = ... order = ... } }` "
                                 "or use --symbolic.");
      }
      warnings.push_back(std::string(kWarnMissingSimulationBlock) +
                         "; proceeding without simulation metadata");
    }
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

IndexedConstraintProblem SemanticAnalyzer::analyzeConstraintProblem(
    const ConstraintProblemDecl &problem) {
  LocalScopeGuard localsScope(locals, std::unordered_map<std::string, bool>{});
  coordIndex.clear();

  if (!problem.domains.empty()) {
    const auto &domain = problem.domains.front();
    const std::vector<std::string> cartesian = {"x", "y", "z"};
    const std::vector<std::string> spherical = {"r", "theta", "phi"};
    const std::vector<std::string> cylindrical = {"rho", "phi", "z"};
    const std::vector<std::string> *coordinates = nullptr;
    if (domain.coordinates == "cartesian")
      coordinates = &cartesian;
    else if (domain.coordinates == "spherical")
      coordinates = &spherical;
    else if (domain.coordinates == "cylindrical")
      coordinates = &cylindrical;
    if (coordinates) {
      for (size_t i = 0;
           i < domain.resolution.size() && i < coordinates->size(); ++i)
        coordIndex[(*coordinates)[i]] = static_cast<int>(i);
    }
  }

  auto registerIndices = [this](const std::vector<std::string> &indices) {
    for (const auto &idx : indices) {
      validateSpatialIndex(idx);
      if (!coordIndex.count(idx))
        coordIndex[idx] = -2;
    }
  };
  for (const auto &equation : problem.equations)
    registerIndices(equation.indices);
  for (const auto &boundary : problem.boundaries)
    for (const auto &condition : boundary.conditions)
      registerIndices(condition.lhs.indices);
  for (const auto &seed : problem.seeds)
    registerIndices(seed.lhs.indices);

  TensorTypeChecker checker(hasConnectionTensor);
  IndexedConstraintProblem out;
  out.name = problem.name;

  for (const auto &equation : problem.equations) {
    IndexedConstraintEquation indexed;
    indexed.name = equation.name;
    indexed.type = equation.type;
    indexed.indices = equation.indices;
    indexed.residual = transformExpr(equation.residual.get());
    checker.checkAssignmentVariance(tensorTypeFromDesc(equation.type),
                                    equation.indices, indexed.residual.get());
    checker.infer(indexed.residual.get());
    out.equations.push_back(std::move(indexed));
  }

  auto lowerAssignment = [this, &checker](const Assignment &assignment) {
    auto it = unknowns.find(assignment.lhs.base);
    if (it == unknowns.end()) {
      throw std::runtime_error(
          "constraint assignment targets unknown symbol '" +
          assignment.lhs.base + "'");
    }
    const ConstraintUnknownDecl &decl = *it->second;
    const size_t rank = static_cast<size_t>(decl.type.up + decl.type.down);
    if (assignment.lhs.indices.size() != rank) {
      throw std::runtime_error("constraint unknown '" + decl.name +
                               "' expects " + std::to_string(rank) +
                               " indices, got " +
                               std::to_string(assignment.lhs.indices.size()));
    }

    IndexedConstraintAssignment indexed;
    indexed.unknown = decl.name;
    indexed.indices = assignment.lhs.indices;
    indexed.rhs = transformExpr(assignment.rhs.get());
    checker.checkAssignmentVariance(tensorTypeFromDesc(decl.type),
                                    assignment.lhs.indices, indexed.rhs.get());
    checker.infer(indexed.rhs.get());
    return indexed;
  };

  for (const auto &boundary : problem.boundaries) {
    IndexedConstraintBoundary indexed;
    indexed.region = boundary.region;
    for (const auto &condition : boundary.conditions)
      indexed.conditions.push_back(lowerAssignment(condition));
    out.boundaries.push_back(std::move(indexed));
  }
  for (const auto &seed : problem.seeds)
    out.seeds.push_back(lowerAssignment(seed));

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
