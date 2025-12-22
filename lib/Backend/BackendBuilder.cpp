
#include "tensorium/Backend/BackendBuilder.hpp"

namespace tensorium::backend {

static std::unique_ptr<ExprIR>
lowerIndexedExpr(const tensorium::IndexedExpr *e) {
  using namespace tensorium;

  if (!e)
    return nullptr;

  if (auto n = dynamic_cast<const IndexedNumber *>(e)) {
    return std::make_unique<NumberIR>(n->value);
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
    out->tensorIndexNames =
        v->tensorIndexNames; // optional, but useful for debug
    return out;
  }

  if (auto b = dynamic_cast<const IndexedBinary *>(e)) {

    return std::make_unique<BinaryIR>(std::string(1, b->op),
                                      lowerIndexedExpr(b->lhs.get()),
                                      lowerIndexedExpr(b->rhs.get()));
  }

  if (auto c = dynamic_cast<const IndexedCall *>(e)) {
    auto out = std::make_unique<CallIR>(c->callee);
    out->args.reserve(c->args.size());
    for (const auto &a : c->args)
      out->args.push_back(lowerIndexedExpr(a.get()));
    return out;
  }

  // fallback: should not happen
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
  case TensorKind::MixedTensor:
    return FieldKind::MixedTensor;
  }
  return FieldKind::Scalar;
}

static SimulationIR lowerSimulation(const tensorium::SimulationConfig &sim) {
  SimulationIR out;

  // Coordinates
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

  // Simulation (optional)
  if (prog.simulation)
    mod.simulation = lowerSimulation(*prog.simulation);

  // Fields
  mod.fields.reserve(prog.fields.size());
  for (const auto &f : prog.fields) {
    FieldIR out;
    out.name = f.name;
    out.kind = lowerFieldKind(f.kind);
    out.up = f.up;
    out.down = f.down;
    mod.fields.push_back(std::move(out));
  }

  // Evolutions → Indexed → ExprIR
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
      oeq.rhs = lowerIndexedExpr(eq.rhs.get());
      out.equations.push_back(std::move(oeq));
    }

    mod.evolutions.push_back(std::move(out));
  }

  return mod;
}

} // namespace tensorium::backend
