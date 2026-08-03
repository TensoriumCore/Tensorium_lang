
#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Core/IndexSet.h"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensorium::backend {

static std::unique_ptr<InitExprIR> lowerInitExpr(const tensorium::Expr *expr) {
  using namespace tensorium;
  if (!expr)
    return nullptr;

  if (auto n = dynamic_cast<const NumberExpr *>(expr)) {
    return std::make_unique<InitNumberIR>(n->value);
  }
  if (auto v = dynamic_cast<const VarExpr *>(expr))
    return std::make_unique<InitSymbolIR>(v->name);
  if (auto idx = dynamic_cast<const IndexedVarExpr *>(expr)) {
    std::string sym = idx->base + "[";
    for (size_t i = 0; i < idx->indices.size(); ++i) {
      sym += idx->indices[i];
      if (i + 1 < idx->indices.size())
        sym += ",";
    }
    sym += "]";
    return std::make_unique<InitSymbolIR>(sym);
  }
  if (auto p = dynamic_cast<const ParenExpr *>(expr))
    return lowerInitExpr(p->inner.get());
  if (auto b = dynamic_cast<const BinaryExpr *>(expr)) {
    return std::make_unique<InitBinaryIR>(b->op, lowerInitExpr(b->lhs.get()),
                                          lowerInitExpr(b->rhs.get()));
  }
  if (auto c = dynamic_cast<const CallExpr *>(expr)) {
    auto out = std::make_unique<InitCallIR>(c->callee);
    out->args.reserve(c->args.size());
    for (const auto &arg : c->args)
      out->args.push_back(lowerInitExpr(arg.get()));
    return out;
  }
  return std::make_unique<InitSymbolIR>("<expr>");
}

static tensorium::ir::TensorType
lowerTensorType(const tensorium::TensorTypeDesc &d) {
  tensorium::ir::TensorType out;
  out.up = d.up;
  out.down = d.down;
  return out;
}

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

static tensorium::ir::TensorType makeTensorType(int up, int down) {
  tensorium::ir::TensorType out;
  out.up = up;
  out.down = down;
  return out;
}

static std::unique_ptr<VarIR>
makeIndexedFieldRef(const std::string &fieldName,
                    const std::vector<std::string> &indexNames, int up,
                    int down) {
  auto out = std::make_unique<VarIR>(fieldName, VarKind::Field);
  out->tensorIndexNames = indexNames;
  out->exprType = makeTensorType(up, down);
  return out;
}

static const tensorium::IndexedVar *
asFieldVar(const tensorium::IndexedExpr *e) {
  auto *v = dynamic_cast<const tensorium::IndexedVar *>(e);
  if (!v || v->kind != tensorium::IndexedVarKind::Field)
    return nullptr;
  return v;
}

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
  dj_gamma_lk->exprType = makeTensorType(0, 3);

  auto dk_gamma_lj =
      std::make_unique<PartialDerivativeIR>(std::move(gamma_lj), "k");
  dk_gamma_lj->exprType = makeTensorType(0, 3);

  auto dl_gamma_jk =
      std::make_unique<PartialDerivativeIR>(std::move(gamma_jk), "l");
  dl_gamma_jk->exprType = makeTensorType(0, 3);

  auto add = std::make_unique<BinaryIR>("+", std::move(dj_gamma_lk),
                                        std::move(dk_gamma_lj));
  add->exprType = makeTensorType(0, 3);

  auto sum =
      std::make_unique<BinaryIR>("-", std::move(add), std::move(dl_gamma_jk));
  sum->exprType = makeTensorType(0, 3);

  auto product =
      std::make_unique<TensorProductIR>(std::move(gammaU_il), std::move(sum));
  product->exprType = makeTensorType(2, 3);

  auto contraction = std::make_unique<ContractionIR>(std::move(product));
  contraction->summedIndices = {"l"};
  contraction->exprType = makeTensorType(1, 2);

  auto half = std::make_unique<NumberIR>(0.5);
  half->exprType = makeTensorType(0, 0);

  auto out =
      std::make_unique<BinaryIR>("*", std::move(half), std::move(contraction));
  out->exprType = lowerTensorType(call->inferredType);
  return out;
}

static std::unique_ptr<ExprIR>
lowerIndexedExpr(const tensorium::IndexedExpr *e,
                 bool materializeImplicitContraction,
                 bool hasConnectionTensor) {
  using namespace tensorium;

  if (!e)
    return nullptr;

  if (auto n = dynamic_cast<const IndexedNumber *>(e)) {
    auto out = std::make_unique<NumberIR>(n->value);
    out->exprType = lowerTensorType(n->inferredType);
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
    case IndexedVarKind::Unknown:
      k = VarKind::Unknown;
      break;
    }

    auto out = std::make_unique<VarIR>(v->name, k);
    out->coordIndex = coord;
    out->tensorIndexNames = v->tensorIndexNames;
    out->exprType = lowerTensorType(v->inferredType);
    return out;
  }

  if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
    auto lhs = lowerIndexedExpr(b->lhs.get(), materializeImplicitContraction,
                                hasConnectionTensor);
    auto rhs = lowerIndexedExpr(b->rhs.get(), materializeImplicitContraction,
                                hasConnectionTensor);

    std::unique_ptr<ExprIR> out;
    if (b->op == '*' && hasTensorRank(b->lhs.get()) &&
        hasTensorRank(b->rhs.get())) {
      auto product =
          std::make_unique<TensorProductIR>(std::move(lhs), std::move(rhs));
      product->exprType = lowerTensorType(b->inferredType);
      out = std::move(product);
    } else {
      auto binary = std::make_unique<BinaryIR>(std::string(1, b->op),
                                               std::move(lhs), std::move(rhs));
      binary->exprType = lowerTensorType(b->inferredType);
      out = std::move(binary);
    }

    if (materializeImplicitContraction && b->op == '*') {
      auto summed = collectRepeatedIndices(b);
      if (!summed.empty()) {
        auto contraction = std::make_unique<ContractionIR>(std::move(out));
        contraction->summedIndices = std::move(summed);
        contraction->exprType = lowerTensorType(b->inferredType);
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
      deriv->exprType = lowerTensorType(c->inferredType);
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
      deriv->exprType = lowerTensorType(c->inferredType);
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
      deriv->exprType = lowerTensorType(c->inferredType);
      return deriv;
    }

    if (c->callee == "gradient" || c->callee == "grad") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_gradient>");
      auto grad = std::make_unique<GradientIR>(
          lowerIndexedExpr(c->args[0].get(), materializeImplicitContraction,
                           hasConnectionTensor));
      grad->exprType = lowerTensorType(c->inferredType);
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
      div->exprType = lowerTensorType(c->inferredType);
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
      trace->exprType = lowerTensorType(c->inferredType);
      return trace;
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
      permute->exprType = lowerTensorType(c->inferredType);
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
          lowerIndexedExpr(c->args[0].get(), false, hasConnectionTensor), from,
          to);
      rename->exprType = lowerTensorType(c->inferredType);
      return rename;
    }

    if (c->callee == "contract") {
      if (c->args.empty())
        return std::make_unique<CallIR>("<invalid_contract>");
      auto contraction = std::make_unique<ContractionIR>(
          lowerIndexedExpr(c->args[0].get(), false, hasConnectionTensor));
      contraction->summedIndices = collectRepeatedIndices(c->args[0].get());
      contraction->exprType = lowerTensorType(c->inferredType);
      return contraction;
    }

    if (c->callee == "christoffel") {
      return lowerChristoffelBuiltin(c);
    }

    auto out = std::make_unique<CallIR>(c->callee);
    out->isExtern = c->isExtern;
    out->externArity = c->declaredArity;
    out->returnType = lowerTensorType(c->returnType);
    out->paramTypes.reserve(c->paramTypes.size());
    for (const auto &paramType : c->paramTypes)
      out->paramTypes.push_back(lowerTensorType(paramType));
    out->args.reserve(c->args.size());
    for (const auto &a : c->args)
      out->args.push_back(lowerIndexedExpr(
          a.get(), materializeImplicitContraction, hasConnectionTensor));
    out->exprType = lowerTensorType(c->inferredType);
    return out;
  }

  return std::make_unique<CallIR>("<unknown>");
}

static std::string renderPrintLabel(const tensorium::IndexedVar &var) {
  std::string label = var.name;
  if (!var.tensorIndexNames.empty()) {
    label += "[";
    for (size_t i = 0; i < var.tensorIndexNames.size(); ++i) {
      label += var.tensorIndexNames[i];
      if (i + 1 < var.tensorIndexNames.size())
        label += ",";
    }
    label += "]";
  }
  return label;
}

static PrintIR lowerPrint(const tensorium::IndexedPrint &print) {
  const auto *var = dynamic_cast<const tensorium::IndexedVar *>(print.expr.get());
  if (!var || var->kind != tensorium::IndexedVarKind::Field)
    throw std::runtime_error("print() lowering expects a field reference");

  PrintIR out;
  out.label = renderPrintLabel(*var);
  out.fieldName = var->name;
  out.indices = var->tensorIndexNames;
  out.tensorType = lowerTensorType(print.expr->inferredType);
  return out;
}

FieldKind BackendBuilder::lowerFieldKind(TensorKind k) {
  switch (k) {
  case TensorKind::Scalar:
    return FieldKind::Scalar;
  case TensorKind::Vector:
    return FieldKind::Vector;
  case TensorKind::Covector:
    return FieldKind::Covector;
  case TensorKind::CovTensor2:
    return FieldKind::CovTensor2;
  case TensorKind::ConTensor2:
    return FieldKind::ConTensor2;
  case TensorKind::CovTensor3:
    return FieldKind::CovTensor3;
  case TensorKind::ConTensor3:
    return FieldKind::ConTensor3;
  case TensorKind::CovTensor4:
    return FieldKind::CovTensor4;
  case TensorKind::ConTensor4:
    return FieldKind::ConTensor4;
  case TensorKind::MixedTensor:
    return FieldKind::MixedTensor;
  case TensorKind::Metric:
    return FieldKind::CovTensor2;
  case TensorKind::InverseMetric:
    return FieldKind::ConTensor2;
  }
  return FieldKind::Scalar;
}

static SimulationIR lowerSimulation(const tensorium::SimulationConfig &sim) {
  SimulationIR out;

  switch (sim.coordinates) {
  case tensorium::CoordinateSystem::Cartesian:
    out.coords = CoordSystem::Cartesian;
    break;
  case tensorium::CoordinateSystem::Spherical:
    out.coords = CoordSystem::Spherical;
    break;
  case tensorium::CoordinateSystem::Cylindrical:
    out.coords = CoordSystem::Cylindrical;
    break;
  }

  out.dimension = sim.dimension;
  out.resolution = sim.resolution;

  // Time
  out.time.dt = sim.time.dt;
  switch (sim.time.integrator) {
  case tensorium::TimeIntegrator::Euler:
    out.time.integrator = backend::TimeIntegrator::Euler;
    break;
  case tensorium::TimeIntegrator::RK3:
    out.time.integrator = backend::TimeIntegrator::RK3;
    break;
  case tensorium::TimeIntegrator::RK4:
    out.time.integrator = backend::TimeIntegrator::RK4;
    break;
  }

  // Spatial
  out.spatial.order = sim.spatial.order;

  out.spatial.scheme =
      (sim.spatial.scheme == tensorium::SpatialScheme::FiniteDifference)
          ? backend::SpatialScheme::FD
          : backend::SpatialScheme::Spectral;

  out.spatial.derivative =
      (sim.spatial.derivative == tensorium::DerivativeScheme::Centered)
          ? backend::DerivativeScheme::Centered
          : backend::DerivativeScheme::Upwind;

  return out;
}

ModuleIR BackendBuilder::build(const Program &prog,
                               tensorium::SemanticAnalyzer &sem) {
  ModuleIR mod;
  bool hasConnectionTensor = false;

  if (!prog.constraints.empty() && !prog.evolutions.empty()) {
    throw std::runtime_error(
        "constraints blocks cannot be mixed with evolution blocks in the "
        "current residual-kernel ABI");
  }
  mod.hasResidualConstraints = !prog.constraints.empty();

  if (prog.simulation)
    mod.simulation = lowerSimulation(*prog.simulation);

  if (prog.initialData) {
    InitialDataIR init;
    init.hasMetric4 = prog.initialData->hasMetric4;
    init.hasDecomposed = prog.initialData->hasDecomposed;

    if (prog.initialData->hasMetric4) {
      init.metric4.name = prog.initialData->metric4.name;
      init.metric4.indices = prog.initialData->metric4.indices;
      init.metric4.enforceSymmetry = prog.initialData->enforceSymmetry;

      if (prog.simulation) {
        switch (prog.simulation->coordinates) {
        case tensorium::CoordinateSystem::Cartesian:
          init.metric4.coordSystem = "cartesian";
          break;
        case tensorium::CoordinateSystem::Spherical:
          init.metric4.coordSystem = "spherical";
          break;
        case tensorium::CoordinateSystem::Cylindrical:
          init.metric4.coordSystem = "cylindrical";
          break;
        }
      }

      for (const auto &row : prog.initialData->metric4.components) {
        for (const auto &entry : row)
          init.metric4.components.push_back(lowerInitExpr(entry.get()));
      }
    }

    if (prog.initialData->hasDecomposed) {
      init.decomposed.alphaExpr =
          lowerInitExpr(prog.initialData->decomposed.alpha.get());
      for (const auto &entry : prog.initialData->decomposed.beta)
        init.decomposed.betaExpr.push_back(lowerInitExpr(entry.get()));
      for (const auto &row : prog.initialData->decomposed.gamma) {
        for (const auto &entry : row)
          init.decomposed.gammaExpr.push_back(lowerInitExpr(entry.get()));
      }
      for (const auto &row : prog.initialData->decomposed.gammaU) {
        for (const auto &entry : row)
          init.decomposed.gammaUExpr.push_back(lowerInitExpr(entry.get()));
      }
    }

    init.split3p1.enabled = prog.initialData->split3p1.enabled;
    init.split3p1.hasAlpha = prog.initialData->split3p1.hasAlpha;
    init.split3p1.hasBeta = prog.initialData->split3p1.hasBeta;
    init.split3p1.hasGamma = prog.initialData->split3p1.hasGamma;
    init.split3p1.hasGammaU = prog.initialData->split3p1.hasGammaU;
    if (init.split3p1.hasAlpha)
      init.split3p1.alphaField = prog.initialData->split3p1.alphaTarget.base;
    if (init.split3p1.hasBeta)
      init.split3p1.betaField = prog.initialData->split3p1.betaTarget.base;
    if (init.split3p1.hasGamma)
      init.split3p1.gammaField = prog.initialData->split3p1.gammaTarget.base;
    if (init.split3p1.hasGammaU)
      init.split3p1.gammaUField = prog.initialData->split3p1.gammaUTarget.base;

    mod.initialData = std::move(init);
  }

  mod.fields.reserve(prog.fields.size());
  for (const auto &f : prog.fields) {
    FieldIR out;
    out.name = f.name;
    out.kind = lowerFieldKind(f.kind);
    out.tensorType.up = f.up;
    out.tensorType.down = f.down;
    if ((f.up == 1 && f.down == 2) ||
        ((f.up + f.down) == 3 &&
         (f.name == "Gamma" || f.name == "GammaU" || f.name == "Christoffel"))) {
      hasConnectionTensor = true;
    }
    mod.fields.push_back(std::move(out));
  }

  if (prog.initialData && prog.initialData->hasConstraintProblem) {
    const auto &problem = prog.initialData->constraintProblem;
    auto indexed = sem.analyzeConstraintProblem(problem);
    ConstraintProblemIR out;
    out.name = problem.name;

    for (const auto &domain : problem.domains) {
      SpectralDomainIR lowered;
      lowered.name = domain.name;
      lowered.coordinates = domain.coordinates;
      lowered.topology = domain.topology;
      lowered.resolution = domain.resolution;
      lowered.basis = domain.basis;
      lowered.bounds = domain.bounds;
      out.domains.push_back(std::move(lowered));
    }
    for (const auto &unknown : problem.unknowns) {
      ConstraintUnknownIR lowered;
      lowered.name = unknown.name;
      lowered.tensorType = lowerTensorType(unknown.type);
      lowered.indices = unknown.indices;
      out.unknowns.push_back(std::move(lowered));
    }
    for (const auto &equation : indexed.equations) {
      ConstraintEquationIR lowered;
      lowered.name = equation.name;
      lowered.tensorType = lowerTensorType(equation.type);
      lowered.indices = equation.indices;
      lowered.residual =
          lowerIndexedExpr(equation.residual.get(), true, hasConnectionTensor);
      out.equations.push_back(std::move(lowered));
    }

    auto lowerAssignment =
        [hasConnectionTensor](const IndexedConstraintAssignment &assignment) {
          ConstraintAssignmentIR lowered;
          lowered.unknown = assignment.unknown;
          lowered.indices = assignment.indices;
          lowered.rhs =
              lowerIndexedExpr(assignment.rhs.get(), true, hasConnectionTensor);
          return lowered;
        };
    for (const auto &boundary : indexed.boundaries) {
      ConstraintBoundaryIR lowered;
      lowered.region = boundary.region;
      for (const auto &condition : boundary.conditions)
        lowered.conditions.push_back(lowerAssignment(condition));
      out.boundaries.push_back(std::move(lowered));
    }
    for (const auto &interface : problem.interfaces) {
      ConstraintInterfaceIR lowered;
      lowered.innerDomain = interface.innerDomain;
      lowered.outerDomain = interface.outerDomain;
      out.interfaces.push_back(std::move(lowered));
    }
    for (const auto &seed : indexed.seeds)
      out.seeds.push_back(lowerAssignment(seed));

    if (indexed.cttReconstruction.enabled) {
      out.cttReconstruction.enabled = true;
      out.cttReconstruction.conformalFactor =
          indexed.cttReconstruction.conformalFactor;
      out.cttReconstruction.radialVectorPotential =
          indexed.cttReconstruction.radialVectorPotential;
      out.cttReconstruction.meanCurvature =
          lowerIndexedExpr(indexed.cttReconstruction.meanCurvature.get(), true,
                           hasConnectionTensor);
    }

    out.solve.nonlinear = problem.solve.nonlinear;
    out.solve.linear = problem.solve.linear;
    out.solve.tolerance = problem.solve.tolerance;
    out.solve.maxIterations = problem.solve.maxIterations;
    mod.constraintProblem = std::move(out);
  }

  mod.evolutions.reserve(prog.evolutions.size());
  for (const auto &evo : prog.evolutions) {
    auto indexed = sem.analyzeEvolution(evo);

    EvolutionIR out;
    out.name = indexed.name;
    out.equations.reserve(indexed.equations.size());

    for (const auto &eq : indexed.equations) {
      EquationIR oeq;
      oeq.fieldName = eq.fieldName;
      oeq.indices = eq.indices;
      oeq.rhs = lowerIndexedExpr(eq.rhs.get(), true, hasConnectionTensor);
      out.equations.push_back(std::move(oeq));
    }

    out.temporaries.reserve(indexed.temp.size());
    for (const auto &tmp : indexed.temp) {
      TempAssignIR ot;
      ot.name = tmp.tensor;
      ot.indices = tmp.indices;
      ot.indexOffsets = tmp.indexOffsets;
      ot.rhs = lowerIndexedExpr(tmp.rhs.get(), true, hasConnectionTensor);
      out.temporaries.push_back(std::move(ot));
    }

    mod.evolutions.push_back(std::move(out));
  }

  for (const auto &constraints : prog.constraints) {
    auto indexed = sem.analyzeConstraint(constraints);

    EvolutionIR out;
    out.name = indexed.name;
    out.equations.reserve(indexed.equations.size());

    for (const auto &eq : indexed.equations) {
      EquationIR oeq;
      oeq.fieldName = eq.fieldName;
      oeq.indices = eq.indices;
      oeq.rhs = lowerIndexedExpr(eq.rhs.get(), true, hasConnectionTensor);
      out.equations.push_back(std::move(oeq));
    }

    out.temporaries.reserve(indexed.temp.size());
    for (const auto &tmp : indexed.temp) {
      TempAssignIR ot;
      ot.name = tmp.tensor;
      ot.indices = tmp.indices;
      ot.indexOffsets = tmp.indexOffsets;
      ot.rhs = lowerIndexedExpr(tmp.rhs.get(), true, hasConnectionTensor);
      out.temporaries.push_back(std::move(ot));
    }

    mod.evolutions.push_back(std::move(out));
  }

  mod.prints.reserve(prog.prints.size());
  for (const auto &print : prog.prints) {
    auto indexed = sem.analyzePrint(print);
    mod.prints.push_back(lowerPrint(indexed));
  }

  return mod;
}

} // namespace tensorium::backend
