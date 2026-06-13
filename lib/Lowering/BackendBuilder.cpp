
#include "tensorium/Lowering/BackendBuilder.hpp"
#include "tensorium/Lowering/TensorTypeConversion.hpp"
#include "ExprLowering.h"

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

static std::string spectralComponentName(const std::string &base,
                                         int component) {
  return base + std::to_string(component + 1);
}

static std::string spectralComponentCoordName(int component) {
  switch (component) {
  case 0:
    return "i";
  case 1:
    return "j";
  default:
    return "k";
  }
}

static bool isRankOneField(const std::vector<FieldIR> &fields,
                           const std::string &name) {
  for (const auto &field : fields)
    if (field.name == name)
      return field.tensorType.rank() == 1;
  return false;
}

static bool hasFieldIR(const std::vector<FieldIR> &fields,
                       const std::string &name) {
  for (const auto &field : fields)
    if (field.name == name)
      return true;
  return false;
}

static void ensureScalarComponentFields(std::vector<FieldIR> &fields,
                                        const std::string &base) {
  if (!isRankOneField(fields, base))
    return;
  for (int component = 0; component < 3; ++component) {
    const std::string name = spectralComponentName(base, component);
    if (hasFieldIR(fields, name))
      continue;
    FieldIR out;
    out.name = name;
    out.kind = FieldKind::Scalar;
    out.tensorType = lowering::makeTensorType(0, 0);
    fields.push_back(std::move(out));
  }
}

static void componentizeRankOneFieldRefs(ExprIR *expr,
                                         const std::vector<FieldIR> &fields,
                                         const std::string &componentIndex,
                                         int component) {
  if (!expr)
    return;
  switch (expr->kind) {
  case ExprIR::Kind::Var: {
    auto *var = static_cast<VarIR *>(expr);
    if (var->vkind == VarKind::Field && isRankOneField(fields, var->name) &&
        var->tensorIndexNames.size() == 1 &&
        var->tensorIndexNames[0] == componentIndex) {
      var->name = spectralComponentName(var->name, component);
      var->tensorIndexNames.clear();
      var->exprType = lowering::makeTensorType(0, 0);
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<BinaryIR *>(expr);
    componentizeRankOneFieldRefs(bin->lhs.get(), fields, componentIndex,
                                 component);
    componentizeRankOneFieldRefs(bin->rhs.get(), fields, componentIndex,
                                 component);
    bin->exprType = lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<TensorProductIR *>(expr);
    componentizeRankOneFieldRefs(prod->lhs.get(), fields, componentIndex,
                                 component);
    componentizeRankOneFieldRefs(prod->rhs.get(), fields, componentIndex,
                                 component);
    prod->exprType = lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<CallIR *>(expr);
    std::string vectorBase;
    if (call->callee == "york_vector_laplacian" && call->args.size() == 1) {
      if (auto *var = dynamic_cast<VarIR *>(call->args[0].get());
          var && var->vkind == VarKind::Field)
        vectorBase = var->name;
    }
    for (auto &arg : call->args)
      componentizeRankOneFieldRefs(arg.get(), fields, componentIndex,
                                   component);
    if (call->callee == "york_vector_laplacian_diag")
      call->callee += "_" + spectralComponentCoordName(component);
    if (call->callee == "york_vector_laplacian" && !vectorBase.empty())
      call->callee += "_" + vectorBase + "_" + std::to_string(component);
    call->exprType = lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *deriv = static_cast<PartialDerivativeIR *>(expr);
    componentizeRankOneFieldRefs(deriv->in.get(), fields, componentIndex,
                                 component);
    if (deriv->coordIndex == componentIndex)
      deriv->coordIndex = spectralComponentCoordName(component);
    deriv->exprType =
        lowering::makeTensorType(deriv->in ? deriv->in->exprType.up : 0,
                       deriv->in ? deriv->in->exprType.down + 1 : 1);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *contract = static_cast<ContractionIR *>(expr);
    componentizeRankOneFieldRefs(contract->in.get(), fields, componentIndex,
                                 component);
    contract->exprType = lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<IndexRenameIR *>(expr);
    componentizeRankOneFieldRefs(rename->in.get(), fields, componentIndex,
                                 component);
    rename->exprType = rename->in ? rename->in->exprType : lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *permute = static_cast<IndexPermuteIR *>(expr);
    componentizeRankOneFieldRefs(permute->in.get(), fields, componentIndex,
                                 component);
    permute->exprType =
        permute->in ? permute->in->exprType : lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<TraceIR *>(expr);
    componentizeRankOneFieldRefs(trace->in.get(), fields, componentIndex,
                                 component);
    trace->exprType = lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<GradientIR *>(expr);
    componentizeRankOneFieldRefs(grad->in.get(), fields, componentIndex,
                                 component);
    grad->exprType =
        lowering::makeTensorType(grad->in ? grad->in->exprType.up : 0,
                       grad->in ? grad->in->exprType.down + 1 : 1);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *deriv = static_cast<CovariantDerivativeIR *>(expr);
    componentizeRankOneFieldRefs(deriv->in.get(), fields, componentIndex,
                                 component);
    deriv->exprType =
        lowering::makeTensorType(deriv->in ? deriv->in->exprType.up : 0,
                       deriv->in ? deriv->in->exprType.down + 1 : 1);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<DivergenceIR *>(expr);
    componentizeRankOneFieldRefs(div->in.get(), fields, componentIndex,
                                 component);
    div->exprType = lowering::makeTensorType(0, 0);
    return;
  }
  case ExprIR::Kind::Number:
    expr->exprType = lowering::makeTensorType(0, 0);
    return;
  }
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
  out.tensorType = lowering::lowerTensorType(print.expr->inferredType);
  return out;
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
                               tensorium::lowering::SemanticAnalysis &semantics) {
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
    out.kind = lowering::lowerFieldKind(f.kind);
    out.tensorType.up = f.up;
    out.tensorType.down = f.down;
    if ((f.up == 1 && f.down == 2) ||
        ((f.up + f.down) == 3 &&
         (f.name == "Gamma" || f.name == "GammaU" || f.name == "Christoffel"))) {
      hasConnectionTensor = true;
    }
    mod.fields.push_back(std::move(out));
  }

  mod.evolutions.reserve(prog.evolutions.size());
  for (const auto &evo : prog.evolutions) {
    auto indexed = semantics.analyzeEvolution(evo);

    EvolutionIR out;
    out.name = indexed.name;
    out.equations.reserve(indexed.equations.size());

    for (const auto &eq : indexed.equations) {
      EquationIR oeq;
      oeq.fieldName = eq.fieldName;
      oeq.unknownFieldName = eq.unknownFieldName;
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
    auto indexed = semantics.analyzeConstraint(constraints);

    EvolutionIR out;
    out.name = indexed.name;
    out.constraintUnknowns.reserve(constraints.unknowns.size());
    for (const auto &unknown : constraints.unknowns) {
      ensureScalarComponentFields(mod.fields, unknown.name);
      if (unknown.type.up + unknown.type.down == 1) {
        for (int component = 0; component < 3; ++component) {
          ConstraintFieldRoleIR role;
          role.name = spectralComponentName(unknown.name, component);
          role.tensorType = lowering::makeTensorType(0, 0);
          out.constraintUnknowns.push_back(std::move(role));
        }
      } else {
        ConstraintFieldRoleIR role;
        role.name = unknown.name;
        role.tensorType = lowering::lowerTensorType(unknown.type);
        out.constraintUnknowns.push_back(std::move(role));
      }
    }
    out.constraintFreeFields.reserve(constraints.freeFields.size());
    for (const auto &freeField : constraints.freeFields) {
      ensureScalarComponentFields(mod.fields, freeField.name);
      if (freeField.type.up + freeField.type.down == 1) {
        for (int component = 0; component < 3; ++component) {
          ConstraintFieldRoleIR role;
          role.name = spectralComponentName(freeField.name, component);
          role.tensorType = lowering::makeTensorType(0, 0);
          out.constraintFreeFields.push_back(std::move(role));
        }
      } else {
        ConstraintFieldRoleIR role;
        role.name = freeField.name;
        role.tensorType = lowering::lowerTensorType(freeField.type);
        out.constraintFreeFields.push_back(std::move(role));
      }
    }
    out.equations.reserve(indexed.equations.size());

    for (const auto &eq : indexed.equations) {
      ensureScalarComponentFields(mod.fields, eq.fieldName);
      ensureScalarComponentFields(mod.fields, eq.unknownFieldName);
      if (eq.indices.size() == 1 && isRankOneField(mod.fields, eq.fieldName) &&
          isRankOneField(mod.fields, eq.unknownFieldName)) {
        for (int component = 0; component < 3; ++component) {
          EquationIR oeq;
          oeq.fieldName = spectralComponentName(eq.fieldName, component);
          oeq.unknownFieldName =
              spectralComponentName(eq.unknownFieldName, component);
          oeq.rhs = lowerIndexedExpr(eq.rhs.get(), true, hasConnectionTensor);
          componentizeRankOneFieldRefs(oeq.rhs.get(), mod.fields,
                                       eq.indices.front(), component);
          oeq.rhs->exprType = lowering::makeTensorType(0, 0);
          out.equations.push_back(std::move(oeq));
        }
      } else {
        EquationIR oeq;
        oeq.fieldName = eq.fieldName;
        oeq.unknownFieldName = eq.unknownFieldName;
        oeq.indices = eq.indices;
        oeq.rhs = lowerIndexedExpr(eq.rhs.get(), true, hasConnectionTensor);
        out.equations.push_back(std::move(oeq));
      }
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

    out.boundaryConditions.reserve(constraints.boundaryConditions.size());
    for (const auto &boundary : constraints.boundaryConditions) {
      const int componentCount =
          isRankOneField(mod.fields, boundary.residualName) ? 3 : 1;
      for (int component = 0; component < componentCount; ++component) {
        BoundaryConditionIR bc;
        bc.residualName =
            componentCount == 1
                ? boundary.residualName
                : spectralComponentName(boundary.residualName, component);
        bc.face = boundary.face;
        bc.kind = boundary.kind;
        bc.valueCoefficient = boundary.valueCoefficient;
        bc.normalDerivativeCoefficient =
            boundary.normalDerivativeCoefficient;
        bc.targetValue = boundary.targetValue;
        bc.derivativeKind = boundary.derivativeKind;
        bc.valueCoefficientCoordinate = boundary.valueCoefficientCoordinate;
        bc.normalDerivativeCoefficientCoordinate =
            boundary.normalDerivativeCoefficientCoordinate;
        bc.targetValueCoordinate = boundary.targetValueCoordinate;
        out.boundaryConditions.push_back(std::move(bc));
      }
    }

    mod.evolutions.push_back(std::move(out));
  }

  mod.prints.reserve(prog.prints.size());
  for (const auto &print : prog.prints) {
    auto indexed = semantics.analyzePrint(print);
    mod.prints.push_back(lowerPrint(indexed));
  }

  return mod;
}

} // namespace tensorium::backend
