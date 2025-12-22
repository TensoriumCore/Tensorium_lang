#include "tensorium/AST/ASTPrinter.hpp"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"
#include <iostream>
#include <vector>

namespace tensorium {

void printField(const FieldDecl &f) {
  std::cout << "field ";

  switch (f.kind) {
  case TensorKind::Scalar:
    std::cout << "scalar ";
    break;
  case TensorKind::Vector:
    std::cout << "vector ";
    break;
  case TensorKind::Covector:
    std::cout << "covector ";
    break;
  case TensorKind::CovTensor2:
    std::cout << "cov_tensor2 ";
    break;
  case TensorKind::ConTensor2:
    std::cout << "con_tensor2 ";
    break;
  case TensorKind::MixedTensor:
    std::cout << "mixed_tensor ";
    break;
  }
  std::cout << f.name << "  (up=" << f.up << ", down=" << f.down << ")";
  if (!f.indices.empty()) {
    std::cout << "[";
    for (size_t i = 0; i < f.indices.size(); ++i) {
      std::cout << f.indices[i];
      if (i + 1 < f.indices.size())
        std::cout << ",";
    }
    std::cout << "]";
  }
  std::cout << "\n";
}

void printSimulation(const SimulationConfig &sim) {
  std::cout << "\n=== Simulation ===\n";

  std::cout << "Coordinates: ";
  switch (sim.coordinates) {
  case CoordinateSystem::Cartesian:
    std::cout << "cartesian";
    break;
  case CoordinateSystem::Spherical:
    std::cout << "spherical";
    break;
  case CoordinateSystem::Cylindrical:
    std::cout << "cylindrical";
    break;
  }
  std::cout << "\n";

  std::cout << "Dimension: " << sim.dimension << "\n";

  std::cout << "Resolution: [";
  for (size_t i = 0; i < sim.resolution.size(); ++i) {
    std::cout << sim.resolution[i];
    if (i + 1 < sim.resolution.size())
      std::cout << ",";
  }
  std::cout << "]\n";

  std::cout << "Time:\n";
  std::cout << "  dt = " << sim.time.dt << "\n";
  std::cout << "  integrator = ";
  switch (sim.time.integrator) {
  case TimeIntegrator::Euler:
    std::cout << "euler";
    break;
  case TimeIntegrator::RK3:
    std::cout << "rk3";
    break;
  case TimeIntegrator::RK4:
    std::cout << "rk4";
    break;
  }
  std::cout << "\n";

  std::cout << "Spatial:\n";
  std::cout << "  scheme = ";
  switch (sim.spatial.scheme) {
  case SpatialScheme::FiniteDifference:
    std::cout << "fd";
    break;
  case SpatialScheme::Spectral:
    std::cout << "spectral";
    break;
  }
  std::cout << "\n";

  std::cout << "  derivative = ";
  switch (sim.spatial.derivative) {
  case DerivativeScheme::Centered:
    std::cout << "centered";
    break;
  case DerivativeScheme::Upwind:
    std::cout << "upwind";
    break;
  }
  std::cout << "\n";

  std::cout << "  order = " << sim.spatial.order << "\n";
}

void printMetric(const MetricDecl &m, int idx) {
  std::cout << "\n=== Metric #" << idx << " ===\n";
  std::cout << "Metric name: " << m.name << "\n";
  std::cout << "Header indices: ";
  for (const auto &id : m.indices)
    std::cout << id << " ";
  std::cout << "\n";

  std::cout << "Number of entries: " << m.entries.size() << "\n";
  for (const auto &e : m.entries) {
    std::cout << "  ";
    std::cout << e.lhs.base;
    if (!e.lhs.indices.empty()) {
      std::cout << "(";
      for (size_t i = 0; i < e.lhs.indices.size(); ++i) {
        std::cout << e.lhs.indices[i];
        if (i + 1 < e.lhs.indices.size())
          std::cout << ",";
      }
      std::cout << ")";
    }
    std::cout << " = ";
    if (e.rhs)
      printExpr(e.rhs.get());
    else
      std::cout << "<null>";
    std::cout << "\n";
  }
}

void printIndexedExpr(const IndexedExpr *e);
void printIndexedBinary(const IndexedBinary *b) {
  std::cout << "(";
  printIndexedExpr(b->lhs.get());
  std::cout << " " << b->op << " ";
  printIndexedExpr(b->rhs.get());
  std::cout << ")";
}

void printIndexedExpr(const IndexedExpr *e) {
  if (auto n = dynamic_cast<const IndexedNumber *>(e)) {
    std::cout << n->value;
    return;
  }

  if (auto v = dynamic_cast<const IndexedVar *>(e)) {
    std::cout << v->name;

    if (!v->tensorIndexNames.empty()) {
      std::cout << "[";
      for (size_t i = 0; i < v->tensorIndexNames.size(); ++i) {
        std::cout << v->tensorIndexNames[i];
        if (i + 1 < v->tensorIndexNames.size())
          std::cout << ",";
      }
      std::cout << "]";
      return;
    }

    std::cout << "[";

    switch (v->kind) {
    case IndexedVarKind::Coordinate:
      std::cout << "coord:" << v->coordIndex;
      break;
    case IndexedVarKind::Local:
      std::cout << "local";
      break;
    case IndexedVarKind::Field:
      std::cout << "field";
      break;
    case IndexedVarKind::Parameter:
      std::cout << "param";
      break;
    }

    if (v->up > 0 || v->down > 0) {
      std::cout << ", tensor(up=" << v->up << ",down=" << v->down << ")";
    }

    std::cout << "]";
    return;
  }

  if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
    printIndexedBinary(b);
    return;
  }

  if (auto c = dynamic_cast<const IndexedCall *>(e)) {
    std::cout << c->callee << "(";
    for (size_t i = 0; i < c->args.size(); ++i) {
      printIndexedExpr(c->args[i].get());
      if (i + 1 < c->args.size())
        std::cout << ", ";
    }
    std::cout << ")";
    return;
  }

  std::cout << "<unknown>";
}

void printIndexedMetric(const IndexedMetric &m) {
  std::cout << "\n=== Indexed Metric ===\n";
  std::cout << m.name << " (rank=" << m.rank << ")\n";

  for (const auto &a : m.assignments) {
    std::cout << "  " << a.tensor;
    if (!a.indexOffsets.empty()) {
      std::cout << "(";
      for (size_t i = 0; i < a.indexOffsets.size(); ++i) {
        std::cout << a.indexOffsets[i];
        if (i + 1 < a.indexOffsets.size())
          std::cout << ",";
      }
      std::cout << ")";
    }
    std::cout << " = ";
    printIndexedExpr(a.rhs.get());
    std::cout << "\n";
  }
}

void printIndexedEvolution(const IndexedEvolution &evo) {
  std::cout << "\n=== Indexed Evolution (" << evo.name << ") ===\n";
  for (const auto &eq : evo.equations) {
    std::cout << "  dt " << eq.fieldName;
    if (!eq.indices.empty()) {
      std::cout << "[";
      for (size_t i = 0; i < eq.indices.size(); ++i) {
        std::cout << eq.indices[i];
        if (i + 1 < eq.indices.size())
          std::cout << ",";
      }
      std::cout << "]";
    }
    std::cout << " = ";
    printIndexedExpr(eq.rhs.get());
    std::cout << "\n";
  }
}

void printEvolution(const EvolutionDecl &evo, int idx) {
  std::cout << "\n=== Evolution #" << idx << " (" << evo.name << ") ===\n";

  for (const auto &eq : evo.equations) {
    std::cout << "  dt " << eq.fieldName;
    if (!eq.indices.empty()) {
      std::cout << "[";
      for (size_t i = 0; i < eq.indices.size(); ++i) {
        std::cout << eq.indices[i];
        if (i + 1 < eq.indices.size())
          std::cout << ",";
      }
      std::cout << "]";
    }
    std::cout << " = ";
    printExpr(eq.rhs.get());
    std::cout << "\n";
  }

  if (!evo.tempAssignments.empty()) {
    std::cout << "  -- Locals --\n";
    for (const auto &tmp : evo.tempAssignments) {
      std::cout << "  " << tmp.lhs.base;
      if (!tmp.lhs.indices.empty()) {
        std::cout << "[";
        for (size_t i = 0; i < tmp.lhs.indices.size(); ++i) {
          std::cout << tmp.lhs.indices[i];
          if (i + 1 < tmp.lhs.indices.size())
            std::cout << ",";
        }
        std::cout << "]";
      }
      std::cout << " = ";
      printExpr(tmp.rhs.get());
      std::cout << "\n";
    }
  }
}

} // namespace tensorium
