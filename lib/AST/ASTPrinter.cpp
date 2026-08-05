#include "tensorium/AST/ASTPrinter.hpp"
#include "tensorium/AST/AST.hpp"
#include <iostream>

namespace tensorium {

class ASTPrinter : public ExprVisitor {
public:
  void visit(const NumberExpr &E) override { std::cout << E.value; }

  void visit(const VarExpr &E) override { std::cout << E.name; }

  void visit(const BinaryExpr &E) override {
    std::cout << "(";
    E.lhs->accept(*this);
    std::cout << " " << E.op << " ";
    E.rhs->accept(*this);
    std::cout << ")";
  }

  void visit(const ParenExpr &E) override {
    std::cout << "(";
    E.inner->accept(*this);
    std::cout << ")";
  }

  void visit(const CallExpr &E) override {
    std::cout << E.callee << "(";
    for (size_t i = 0; i < E.args.size(); ++i) {
      E.args[i]->accept(*this);
      if (i + 1 < E.args.size())
        std::cout << ", ";
    }
    std::cout << ")";
  }

  void visit(const IndexedVarExpr &E) override {
    std::cout << E.base << "[";
    for (size_t i = 0; i < E.indices.size(); ++i) {
      std::cout << E.indices[i];
      int off = 0;
      if (i < E.indexOffsets.size())
        off = E.indexOffsets[i];
      if (off > 0)
        std::cout << "+" << off;
      else if (off < 0)
        std::cout << off;
      if (i + 1 < E.indices.size())
        std::cout << ",";
    }
    std::cout << "]";
  }
};

void printExpr(const Expr *e) {
  if (!e)
    return;
  ASTPrinter P;
  e->accept(P);
}

void printProgram(const Program &prog) {
  std::cout << "=== Program AST ===\n";
  if (!prog.params.empty()) {
    std::cout << "\nParams:\n  params {";
    for (size_t i = 0; i < prog.params.size(); ++i) {
      std::cout << prog.params[i];
      if (i + 1 < prog.params.size())
        std::cout << ", ";
    }
    std::cout << "}\n";
  }
  if (!prog.externs.empty()) {
    std::cout << "\nExtern Functions:\n";
    auto printTensorType = [](const TensorTypeDesc &t) {
      switch (t.kind) {
      case TensorKind::Scalar:
        std::cout << "scalar";
        return;
      case TensorKind::Vector:
        std::cout << "vector";
        return;
      case TensorKind::Covector:
        std::cout << "covector";
        return;
      case TensorKind::CovTensor2:
        std::cout << "cov_tensor2";
        return;
      case TensorKind::ConTensor2:
        std::cout << "con_tensor2";
        return;
      case TensorKind::CovTensor3:
        std::cout << "cov_tensor3";
        return;
      case TensorKind::ConTensor3:
        std::cout << "con_tensor3";
        return;
      case TensorKind::CovTensor4:
        std::cout << "cov_tensor4";
        return;
      case TensorKind::ConTensor4:
        std::cout << "con_tensor4";
        return;
      case TensorKind::MixedTensor:
        std::cout << "mixed_tensor(up=" << t.up << ",down=" << t.down << ")";
        return;
      case TensorKind::Metric:
        std::cout << "metric";
        return;
      case TensorKind::InverseMetric:
        std::cout << "inverse_metric";
        return;
      }
    };
    for (const auto &ext : prog.externs) {
      std::cout << "  extern ";
      printTensorType(ext.returnType);
      std::cout << " " << ext.name << "(";
      for (size_t i = 0; i < ext.params.size(); ++i) {
        printTensorType(ext.params[i]);
        if (i + 1 < ext.params.size())
          std::cout << ", ";
      }
      std::cout << ")\n";
    }
  }
  if (!prog.fields.empty()) {
    std::cout << "\nFields:\n";
    for (const auto &f : prog.fields) {
      std::cout << "  field ";

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
        std::cout << "cov_tensor2";
        break;
      case TensorKind::ConTensor2:
        std::cout << "con_tensor2";
        break;
      case TensorKind::CovTensor3:
        std::cout << "cov_tensor3";
        break;
      case TensorKind::ConTensor3:
        std::cout << "con_tensor3";
        break;
      case TensorKind::ConTensor4:
        std::cout << "con_tensor4";
        break;
      case TensorKind::CovTensor4:
        std::cout << "cov_tensor4";
        break;
      case TensorKind::MixedTensor:
        std::cout << "mixed_tensor";
        break;
      case TensorKind::Metric:
        std::cout << "metric";
        break;
      case TensorKind::InverseMetric:
        std::cout << "inverse_metric";
        break;
      }

      std::cout << f.name;

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
  }

  if (prog.simulation) {
    const auto &sim = *prog.simulation;

    std::cout << "\nSimulation:\n";
    std::cout << "  dimension = " << sim.dimension << "\n";

    std::cout << "  resolution = [";
    for (size_t i = 0; i < sim.resolution.size(); ++i) {
      std::cout << sim.resolution[i];
      if (i + 1 < sim.resolution.size())
        std::cout << ",";
    }
    std::cout << "]\n";

    std::cout << "  time:\n";
    std::cout << "    dt = " << sim.time.dt << "\n";

    std::cout << "  spatial:\n";
    std::cout << "    order = " << sim.spatial.order << "\n";
  }

  if (prog.initialData) {
    std::cout << "\nInitialData:\n";
    std::cout << "  enforce_symmetry = "
              << (prog.initialData->enforceSymmetry ? "true" : "false") << "\n";
    if (prog.initialData->hasMetric4) {
      std::cout << "  metric4 " << prog.initialData->metric4.name << "[";
      for (size_t i = 0; i < prog.initialData->metric4.indices.size(); ++i) {
        std::cout << prog.initialData->metric4.indices[i];
        if (i + 1 < prog.initialData->metric4.indices.size())
          std::cout << ",";
      }
      std::cout << "] = <4x4 matrix>\n";
    }
    if (prog.initialData->hasDecomposed) {
      std::cout << "  alpha = ";
      printExpr(prog.initialData->decomposed.alpha.get());
      std::cout << "\n";
      std::cout << "  beta = <3 entries>\n";
      std::cout << "  gamma = <3x3 matrix>\n";
      if (!prog.initialData->decomposed.gammaU.empty())
        std::cout << "  gammaU = <3x3 matrix>\n";
    }
    if (prog.initialData->split3p1.enabled) {
      std::cout << "  split_3p1 mappings:\n";
      auto printTarget = [](const TensorAccess &t) {
        std::cout << t.base;
        if (!t.indices.empty()) {
          std::cout << "[";
          for (size_t i = 0; i < t.indices.size(); ++i) {
            std::cout << t.indices[i];
            if (i + 1 < t.indices.size())
              std::cout << ",";
          }
          std::cout << "]";
        }
      };
      if (prog.initialData->split3p1.hasAlpha) {
        std::cout << "    alpha -> ";
        printTarget(prog.initialData->split3p1.alphaTarget);
        std::cout << "\n";
      }
      if (prog.initialData->split3p1.hasBeta) {
        std::cout << "    beta -> ";
        printTarget(prog.initialData->split3p1.betaTarget);
        std::cout << "\n";
      }
      if (prog.initialData->split3p1.hasGamma) {
        std::cout << "    gamma -> ";
        printTarget(prog.initialData->split3p1.gammaTarget);
        std::cout << "\n";
      }
      if (prog.initialData->split3p1.hasGammaU) {
        std::cout << "    gammaU -> ";
        printTarget(prog.initialData->split3p1.gammaUTarget);
        std::cout << "\n";
      }
    }
    if (prog.initialData->hasSpectralProblem) {
      const auto &spectral = prog.initialData->spectralProblem;
      std::cout << "  spectral system=" << spectral.system
                << " coordinate_map=" << spectral.coordinateMap
                << " unknown_map=" << spectral.unknownMap
                << " field_projector=" << spectral.fieldProjector
                << " reconstruction=" << spectral.reconstruction << "\n";
      std::cout << "    resolution=[";
      for (size_t i = 0; i < spectral.resolution.size(); ++i) {
        std::cout << spectral.resolution[i];
        if (i + 1 < spectral.resolution.size())
          std::cout << ",";
      }
      std::cout << "] basis=[";
      for (size_t i = 0; i < spectral.basis.size(); ++i) {
        std::cout << spectral.basis[i];
        if (i + 1 < spectral.basis.size())
          std::cout << ",";
      }
      std::cout << "] parameters=" << spectral.parameters.size() << "\n";
      std::cout << "    solve nonlinear=" << spectral.solve.nonlinear
                << " linear=" << spectral.solve.linear
                << " tolerance=" << spectral.solve.tolerance
                << " max_iterations=" << spectral.solve.maxIterations
                << " preconditioner=" << spectral.solve.preconditioner
                << "\n";
    }
    if (prog.initialData->hasConstraintProblem) {
      const auto &problem = prog.initialData->constraintProblem;
      std::cout << "  constraint_problem " << problem.name << ":\n";
      if (problem.geometry.enabled) {
        std::cout << "    geometry " << problem.geometry.kind << " metric="
                  << problem.geometry.metricName << " inverse_metric="
                  << problem.geometry.inverseMetricName << "\n";
      }
      for (const auto &domain : problem.domains) {
        std::cout << "    domain " << domain.name << " (" << domain.coordinates
                  << ", " << domain.topology << ", basis=" << domain.basis
                  << ", resolution=[";
        for (size_t i = 0; i < domain.resolution.size(); ++i) {
          std::cout << domain.resolution[i];
          if (i + 1 < domain.resolution.size())
            std::cout << ",";
        }
        std::cout << "]";
        if (!domain.bounds.empty()) {
          std::cout << ", bounds=[";
          for (size_t i = 0; i < domain.bounds.size(); ++i) {
            std::cout << domain.bounds[i];
            if (i + 1 < domain.bounds.size())
              std::cout << ",";
          }
          std::cout << "]";
        }
        std::cout << ")\n";
      }
      for (const auto &unknown : problem.unknowns) {
        std::cout << "    unknown "
                  << (unknown.symmetric ? "symmetric " : "") << unknown.name
                  << " (up=" << unknown.type.up << ",down=" << unknown.type.down
                  << ")\n";
      }
      for (const auto &equation : problem.equations) {
        std::cout << "    equation " << equation.name << " = ";
        printExpr(equation.residual.get());
        std::cout << "\n";
      }
      for (const auto &boundary : problem.boundaries)
        std::cout << "    boundary " << boundary.region << " ("
                  << boundary.conditions.size() << " conditions)\n";
      for (const auto &interface : problem.interfaces)
        std::cout << "    interface " << interface.innerDomain << " -> "
                  << interface.outerDomain << "\n";
      if (problem.cttReconstruction.enabled) {
        std::cout << "    reconstruct ctt conformal_factor="
                  << problem.cttReconstruction.conformalFactor;
        if (!problem.cttReconstruction.radialVectorPotential.empty())
          std::cout << " radial_vector="
                    << problem.cttReconstruction.radialVectorPotential;
        if (!problem.cttReconstruction.conformalElectricRadial.empty())
          std::cout << " conformal_electric_radial="
                    << problem.cttReconstruction.conformalElectricRadial;
        std::cout << " mean_curvature=";
        printExpr(problem.cttReconstruction.meanCurvature.get());
        std::cout << "\n";
      }
      std::cout << "    solve nonlinear=" << problem.solve.nonlinear
                << " linear=" << problem.solve.linear
                << " tolerance=" << problem.solve.tolerance
                << " max_iterations=" << problem.solve.maxIterations << "\n";
    }
  }

  if (!prog.prints.empty()) {
    std::cout << "\nPrints:\n";
    for (const auto &print : prog.prints) {
      std::cout << "  print(";
      printExpr(print.expr.get());
      std::cout << ");\n";
    }
  }

  for (const auto &evo : prog.evolutions) {
    std::cout << "Evolution " << evo.name << " {\n";
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
    std::cout << "}\n";
  }

  for (const auto &constraints : prog.constraints) {
    std::cout << "Constraints " << constraints.name << " {\n";
    for (const auto &tmp : constraints.tempAssignments) {
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
    for (const auto &eq : constraints.residuals) {
      std::cout << "  residual " << eq.fieldName;
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
    std::cout << "}\n";
  }
}

} // namespace tensorium
