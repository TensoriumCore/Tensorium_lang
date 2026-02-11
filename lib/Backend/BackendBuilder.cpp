
#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Core/IndexSet.h"

#include <map>
#include <string>
#include <vector>

namespace tensorium::backend {

static tensorium::ir::TensorType lowerTensorType(const tensorium::TensorTypeDesc &d) {
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
    if (b->op == '*' && hasTensorRank(b->lhs.get()) && hasTensorRank(b->rhs.get())) {
      auto product = std::make_unique<TensorProductIR>(std::move(lhs), std::move(rhs));
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
          lowerIndexedExpr(c->args[0].get(), false, hasConnectionTensor),
          from, to);
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

    auto out = std::make_unique<CallIR>(c->callee);
    out->isExtern = c->isExtern;
    out->externArity = c->declaredArity;
    out->returnType = lowerTensorType(c->returnType);
    out->paramTypes.reserve(c->paramTypes.size());
    for (const auto &paramType : c->paramTypes)
      out->paramTypes.push_back(lowerTensorType(paramType));
    out->args.reserve(c->args.size());
    for (const auto &a : c->args)
      out->args.push_back(
          lowerIndexedExpr(a.get(), materializeImplicitContraction,
                           hasConnectionTensor));
    out->exprType = lowerTensorType(c->inferredType);
    return out;
  }

  return std::make_unique<CallIR>("<unknown>");
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

  if (prog.simulation)
    mod.simulation = lowerSimulation(*prog.simulation);

  mod.fields.reserve(prog.fields.size());
  for (const auto &f : prog.fields) {
    FieldIR out;
    out.name = f.name;
    out.kind = lowerFieldKind(f.kind);
    out.tensorType.up = f.up;
    out.tensorType.down = f.down;
    if ((f.name == "Gamma" || f.name == "GammaU" || f.name == "Christoffel") &&
        (f.up + f.down) == 3) {
      hasConnectionTensor = true;
    }
    mod.fields.push_back(std::move(out));
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
      ot.indexOffsets = tmp.indexOffsets;
      ot.rhs = lowerIndexedExpr(tmp.rhs.get(), true, hasConnectionTensor);
      out.temporaries.push_back(std::move(ot));
    }

    mod.evolutions.push_back(std::move(out));
  }

  return mod;
}

} // namespace tensorium::backend
