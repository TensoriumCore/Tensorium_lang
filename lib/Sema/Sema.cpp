#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Sema/CallSupport.hpp"
#include "tensorium/Sema/tensor_type_checker.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

namespace tensorium {

static TensorType tensorTypeFromDesc(const TensorTypeDesc &desc) {
  return TensorType{desc.up, desc.down};
}

static bool isScalarDesc(const TensorTypeDesc &desc) {
  return desc.up == 0 && desc.down == 0 && desc.kind == TensorKind::Scalar;
}

static bool exprStructuralEqual(const Expr *lhs, const Expr *rhs) {
  if (!lhs || !rhs)
    return lhs == rhs;

  if (auto ln = dynamic_cast<const NumberExpr *>(lhs)) {
    auto rn = dynamic_cast<const NumberExpr *>(rhs);
    return rn && ln->value == rn->value;
  }
  if (auto lv = dynamic_cast<const VarExpr *>(lhs)) {
    auto rv = dynamic_cast<const VarExpr *>(rhs);
    return rv && lv->name == rv->name;
  }
  if (auto lb = dynamic_cast<const BinaryExpr *>(lhs)) {
    auto rb = dynamic_cast<const BinaryExpr *>(rhs);
    return rb && lb->op == rb->op &&
           exprStructuralEqual(lb->lhs.get(), rb->lhs.get()) &&
           exprStructuralEqual(lb->rhs.get(), rb->rhs.get());
  }
  if (auto lc = dynamic_cast<const CallExpr *>(lhs)) {
    auto rc = dynamic_cast<const CallExpr *>(rhs);
    if (!rc || lc->callee != rc->callee || lc->args.size() != rc->args.size())
      return false;
    for (size_t i = 0; i < lc->args.size(); ++i) {
      if (!exprStructuralEqual(lc->args[i].get(), rc->args[i].get()))
        return false;
    }
    return true;
  }
  if (auto lp = dynamic_cast<const ParenExpr *>(lhs)) {
    auto rp = dynamic_cast<const ParenExpr *>(rhs);
    return rp && exprStructuralEqual(lp->inner.get(), rp->inner.get());
  }
  if (auto li = dynamic_cast<const IndexedVarExpr *>(lhs)) {
    auto ri = dynamic_cast<const IndexedVarExpr *>(rhs);
    return ri && li->base == ri->base && li->indices == ri->indices;
  }
  return false;
}

static bool isCoordinateAlias(const std::string &name) {
  static const std::set<std::string> kCoordinateAliases = {
      "t", "x", "y", "z", "r", "rho", "theta", "phi"};
  return kCoordinateAliases.count(name) != 0;
}

static bool isAllowedCoordinateName(const SimulationConfig &sim,
                                    const std::string &name) {
  if (sim.coordinates == CoordinateSystem::Spherical) {
    if (name == "r")
      return sim.dimension >= 1;
    if (name == "theta")
      return sim.dimension >= 2;
    if (name == "phi")
      return sim.dimension >= 3;
    return false;
  }

  if (sim.coordinates == CoordinateSystem::Cartesian) {
    if (name == "x")
      return sim.dimension >= 1;
    if (name == "y")
      return sim.dimension >= 2;
    if (name == "z")
      return sim.dimension >= 3;
    return false;
  }

  if (sim.coordinates == CoordinateSystem::Cylindrical) {
    if (name == "rho")
      return sim.dimension >= 1;
    if (name == "phi")
      return sim.dimension >= 2;
    if (name == "z")
      return sim.dimension >= 3;
    return false;
  }

  return false;
}

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
  if (lVar->tensorIndexNames.size() != 2 ||
      rVar->tensorIndexNames.size() != 2)
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

    if (auto itf = fields.find(v->name); itf != fields.end()) {
      const FieldDecl *fd = itf->second;
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Field);
      iv->tensorKind = fd->kind;
      iv->up = fd->up;
      iv->down = fd->down;
      return iv;
    }

    auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Parameter);
    iv->tensorKind = TensorKind::Scalar;
    return iv;
  }

  if (auto b = dynamic_cast<const BinaryExpr *>(e))
    return std::make_unique<IndexedBinary>(b->op, transformExpr(b->lhs.get()),
                                           transformExpr(b->rhs.get()));

  if (auto p = dynamic_cast<const ParenExpr *>(e))
    return transformExpr(p->inner.get());

  if (auto iv = dynamic_cast<const IndexedVarExpr *>(e)) {
    auto it = fields.find(iv->base);
    if (it == fields.end())
      throw std::runtime_error("Unknown indexed tensor: " + iv->base);

    const FieldDecl *fd = it->second;
    size_t expected = static_cast<size_t>(fd->up + fd->down);

    if (iv->indices.size() != expected)
      throw std::runtime_error("Tensor '" + iv->base + "' expects " +
                               std::to_string(expected) + " indices, got " +
                               std::to_string(iv->indices.size()));

    auto out = std::make_unique<IndexedVar>(iv->base, IndexedVarKind::Field);
    out->tensorKind = fd->kind;
    out->up = fd->up;
    out->down = fd->down;

    size_t pos = 0;
    for (auto &idx : iv->indices) {
      if (!coordIndex.count(idx)) {
        validateSpatialIndex(idx);
        coordIndex[idx] = -2;
      }
      int off = resolveIndex(idx);
      out->tensorIndices.push_back(off);
      out->tensorIndexNames.push_back(idx);
      bool isUp = pos < static_cast<size_t>(fd->up);
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
        throw std::runtime_error("extern function '" + c->callee +
                                 "' expects " +
                                 std::to_string(externDecl->params.size()) +
                                 " arguments, got " +
                                 std::to_string(c->args.size()));
      }
    }

    if (mode == CompilationMode::Executable &&
        !isExecutableBuiltin(c->callee) && !externDecl) {
      throw std::runtime_error(
          "executable mode requires implementation for function '" +
          c->callee + "'");
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

void SemanticAnalyzer::validateInitialDataExpr(const Expr *expr,
                                               const std::string &context) {
  if (!expr)
    throw std::runtime_error("initial_data " + context +
                             " contains empty expression");

  if (auto n = dynamic_cast<const NumberExpr *>(expr)) {
    (void)n;
    return;
  }

  if (auto v = dynamic_cast<const VarExpr *>(expr)) {
    if (!prog.simulation)
      return;
    if (isCoordinateAlias(v->name) &&
        !isAllowedCoordinateName(*prog.simulation, v->name)) {
      throw std::runtime_error("initial_data " + context +
                               " uses coordinate '" + v->name +
                               "' incompatible with simulation coordinates");
    }
    return;
  }

  if (auto b = dynamic_cast<const BinaryExpr *>(expr)) {
    validateInitialDataExpr(b->lhs.get(), context);
    validateInitialDataExpr(b->rhs.get(), context);
    return;
  }

  if (auto p = dynamic_cast<const ParenExpr *>(expr)) {
    validateInitialDataExpr(p->inner.get(), context);
    return;
  }

  if (auto c = dynamic_cast<const CallExpr *>(expr)) {
    static const std::set<std::string> kAllowedScalarFns = {
        "sin", "sqrt"};
    if (kAllowedScalarFns.count(c->callee) == 0) {
      throw std::runtime_error("initial_data " + context +
                               " uses unsupported scalar function '" +
                               c->callee + "'");
    }
    for (const auto &arg : c->args)
      validateInitialDataExpr(arg.get(), context);
    return;
  }

  if (auto idx = dynamic_cast<const IndexedVarExpr *>(expr)) {
    throw std::runtime_error("initial_data " + context +
                             " must be scalar expression, got indexed term '" +
                             idx->base + "'");
  }

  throw std::runtime_error("initial_data " + context +
                           " contains unsupported expression node");
}

void SemanticAnalyzer::validateInitialData(const InitialDataDecl &init) {
  if (!prog.simulation) {
    throw std::runtime_error("initial_data requires simulation block");
  }

  if (init.hasMetric4) {
    if (init.metric4.indices.size() != 2) {
      throw std::runtime_error("initial_data metric4 expects exactly 2 indices");
    }
    if (init.metric4.components.size() != 4) {
      throw std::runtime_error("initial_data metric4 expects a 4x4 matrix");
    }
    for (size_t i = 0; i < init.metric4.components.size(); ++i) {
      if (init.metric4.components[i].size() != 4) {
        throw std::runtime_error("initial_data metric4 row " +
                                 std::to_string(i) +
                                 " must have 4 entries");
      }
      for (size_t j = 0; j < init.metric4.components[i].size(); ++j) {
        validateInitialDataExpr(
            init.metric4.components[i][j].get(),
            "metric4[" + std::to_string(i) + "," + std::to_string(j) + "]");
      }
    }

    if (init.enforceSymmetry) {
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
          if (!exprStructuralEqual(init.metric4.components[i][j].get(),
                                   init.metric4.components[j][i].get())) {
            throw std::runtime_error(
                "initial_data metric4 symmetry violation at (" +
                std::to_string(i) + "," + std::to_string(j) + ")");
          }
        }
      }
    }
  }

  if (init.hasDecomposed) {
    if (!init.decomposed.alpha) {
      throw std::runtime_error("initial_data requires alpha expression");
    }
    validateInitialDataExpr(init.decomposed.alpha.get(), "alpha");

    if (init.decomposed.beta.size() != 3) {
      throw std::runtime_error("initial_data beta must have 3 entries");
    }
    for (size_t i = 0; i < init.decomposed.beta.size(); ++i) {
      validateInitialDataExpr(init.decomposed.beta[i].get(),
                              "beta[" + std::to_string(i) + "]");
    }

    if (init.decomposed.gamma.size() != 3) {
      throw std::runtime_error("initial_data gamma must be 3x3");
    }
    for (size_t i = 0; i < init.decomposed.gamma.size(); ++i) {
      if (init.decomposed.gamma[i].size() != 3) {
        throw std::runtime_error("initial_data gamma row " +
                                 std::to_string(i) +
                                 " must have 3 entries");
      }
      for (size_t j = 0; j < init.decomposed.gamma[i].size(); ++j) {
        validateInitialDataExpr(
            init.decomposed.gamma[i][j].get(),
            "gamma[" + std::to_string(i) + "," + std::to_string(j) + "]");
      }
    }

    if (!init.decomposed.gammaU.empty()) {
      if (init.decomposed.gammaU.size() != 3) {
        throw std::runtime_error("initial_data gammaU must be 3x3");
      }
      for (size_t i = 0; i < init.decomposed.gammaU.size(); ++i) {
        if (init.decomposed.gammaU[i].size() != 3) {
          throw std::runtime_error("initial_data gammaU row " +
                                   std::to_string(i) +
                                   " must have 3 entries");
        }
        for (size_t j = 0; j < init.decomposed.gammaU[i].size(); ++j) {
          validateInitialDataExpr(
              init.decomposed.gammaU[i][j].get(),
              "gammaU[" + std::to_string(i) + "," + std::to_string(j) + "]");
        }
      }
    }
  }

  if (init.hasMetric4) {
    auto it = fields.find(init.metric4.name);
    if (it != fields.end()) {
      const FieldDecl *fd = it->second;
      const bool validMetricType =
          (fd->kind == TensorKind::Metric) ||
          (fd->kind == TensorKind::CovTensor2 && fd->up == 0 && fd->down == 2);
      if (!validMetricType) {
        throw std::runtime_error("initial_data metric4 target '" +
                                 init.metric4.name +
                                 "' must be metric or covariant rank-2 field");
      }
    }
  }

  if (init.split3p1.enabled) {
    if (!init.hasMetric4 && !init.hasDecomposed) {
      throw std::runtime_error(
          "split_3p1 mapping requires metric4 or decomposed initial_data definition");
    }

    auto validateTarget = [this](const TensorAccess &target, int expUp,
                                 int expDown, const std::string &label,
                                 bool allowMetricLike = false,
                                 bool allowInverseMetricLike = false) {
      auto it = fields.find(target.base);
      if (it == fields.end()) {
        throw std::runtime_error("split_3p1 target field '" + target.base +
                                 "' for " + label + " is not declared");
      }
      const FieldDecl *fd = it->second;

      bool typeOk = (fd->up == expUp && fd->down == expDown);
      if (allowMetricLike && fd->kind == TensorKind::Metric &&
          expUp == 0 && expDown == 2)
        typeOk = true;
      if (allowInverseMetricLike && fd->kind == TensorKind::InverseMetric &&
          expUp == 2 && expDown == 0)
        typeOk = true;
      if (!typeOk) {
        throw std::runtime_error("split_3p1 target '" + target.base +
                                 "' for " + label + " has wrong tensor type");
      }

      const size_t expectedRank = static_cast<size_t>(expUp + expDown);
      if (target.indices.size() != expectedRank) {
        throw std::runtime_error("split_3p1 target '" + target.base +
                                 "' for " + label +
                                 " has wrong number of indices");
      }
    };

    if (init.split3p1.hasAlpha) {
      validateTarget(init.split3p1.alphaTarget, 0, 0, "alpha");
    }
    if (init.split3p1.hasBeta) {
      validateTarget(init.split3p1.betaTarget, 0, 1, "beta");
    }
    if (init.split3p1.hasGamma) {
      validateTarget(init.split3p1.gammaTarget, 0, 2, "gamma",
                     true, false);
    }
    if (init.split3p1.hasGammaU) {
      validateTarget(init.split3p1.gammaUTarget, 2, 0, "gammaU",
                     false, true);
    }
  }
}

SemanticAnalyzer::SemanticAnalyzer(const Program &p, CompilationMode m)
    : prog(p), mode(m) {
  for (const auto &ext : prog.externs) {
    if (!externSignatures.emplace(ext.name, &ext).second) {
      throw std::runtime_error("Extern function redeclared: " + ext.name);
    }
    if (mode == CompilationMode::Executable && !isScalarDesc(ext.returnType)) {
      throw std::runtime_error("executable mode extern '" + ext.name +
                               "' must return scalar");
    }
  }

  for (const auto &f : prog.fields) {
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
    std::cerr << "[Tensorium] warning: inverse_metric field is missing while metrics are declared\n";
  }
  if (inverseMetricFieldCount > 0 && metricFieldCount == 0) {
    std::cerr << "[Tensorium] warning: metric field is missing while inverse_metric fields are declared\n";
  }

  for (const auto &m : prog.metrics) {
    for (const auto &entry : m.entries) {
      if (entry.lhs.indices.empty())
        locals[entry.lhs.base] = true;
    }
  }
  for (const auto &m : prog.metrics) {
    FieldDecl fd;
    fd.kind = TensorKind::CovTensor2;
    fd.name = m.name;
    fd.up = 0;
    fd.down = 2;
    syntheticMetricFields.push_back(fd);
    fields[m.name] = &syntheticMetricFields.back();
  }
  if (!prog.simulation) {
    if (mode == CompilationMode::Executable) {
      throw std::runtime_error("missing simulation block");
    }
    simulationMissing = true;
    std::cerr << "[Tensorium] warning: missing simulation block (symbolic mode)\n";
  } else {
    validateSimulation(*prog.simulation);
  }

  if (prog.initialData) {
    validateInitialData(*prog.initialData);
  }
}

IndexedMetric SemanticAnalyzer::analyzeMetric(const MetricDecl &decl) {
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
