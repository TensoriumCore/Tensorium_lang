#pragma once

#include "tensorium/Backend/DomainIR.hpp"
#include <iostream>

namespace tensorium::backend {

inline void printInitExpr(const InitExprIR *e) {
  if (!e) {
    std::cout << "<null>";
    return;
  }
  switch (e->kind) {
  case InitExprIR::Kind::Number: {
    auto *n = static_cast<const InitNumberIR *>(e);
    std::cout << n->value;
    return;
  }
  case InitExprIR::Kind::Symbol: {
    auto *s = static_cast<const InitSymbolIR *>(e);
    std::cout << s->name;
    return;
  }
  case InitExprIR::Kind::Binary: {
    auto *b = static_cast<const InitBinaryIR *>(e);
    std::cout << "(";
    printInitExpr(b->lhs.get());
    std::cout << b->op;
    printInitExpr(b->rhs.get());
    std::cout << ")";
    return;
  }
  case InitExprIR::Kind::Call: {
    auto *c = static_cast<const InitCallIR *>(e);
    std::cout << c->callee << "(";
    for (size_t i = 0; i < c->args.size(); ++i) {
      printInitExpr(c->args[i].get());
      if (i + 1 < c->args.size())
        std::cout << ",";
    }
    std::cout << ")";
    return;
  }
  }
}

inline void printExprIR(const ExprIR *e) {
  if (!e) {
    std::cout << "<null>";
    return;
  }

  auto printType = [&]() {
    std::cout << "[u=" << e->exprType.up << ",d=" << e->exprType.down << "]";
  };

  switch (e->kind) {
  case ExprIR::Kind::Number: {
    auto *n = static_cast<const NumberIR *>(e);
    std::cout << n->value;
    printType();
    return;
  }
  case ExprIR::Kind::Var: {
    auto *v = static_cast<const VarIR *>(e);
    std::cout << v->name << "[";
    switch (v->vkind) {
    case VarKind::Field:
      std::cout << "field";
      break;
    case VarKind::Param:
      std::cout << "param";
      break;
    case VarKind::Local:
      std::cout << "local";
      break;
    case VarKind::Coord:
      std::cout << "coord:" << v->coordIndex;
      break;
    case VarKind::Unknown:
      std::cout << "unknown";
      break;
    }
    std::cout << "]";
    printType();
    if (!v->tensorIndexNames.empty()) {
      std::cout << "{";
      for (size_t i = 0; i < v->tensorIndexNames.size(); ++i) {
        std::cout << v->tensorIndexNames[i];
        if (i + 1 < v->tensorIndexNames.size())
          std::cout << ",";
      }
      std::cout << "}";
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    auto *b = static_cast<const BinaryIR *>(e);
    std::cout << "(";
    printExprIR(b->lhs.get());
    std::cout << " " << b->op << " ";
    printExprIR(b->rhs.get());
    std::cout << ")";
    printType();
    return;
  }
  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(e);
    std::cout << c->callee << "(";
    for (size_t i = 0; i < c->args.size(); ++i) {
      printExprIR(c->args[i].get());
      if (i + 1 < c->args.size())
        std::cout << ", ";
    }
    std::cout << ")";
    printType();
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *p = static_cast<const TensorProductIR *>(e);
    std::cout << "tensor_product(";
    printExprIR(p->lhs.get());
    std::cout << ", ";
    printExprIR(p->rhs.get());
    std::cout << ")";
    printType();
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *c = static_cast<const ContractionIR *>(e);
    std::cout << "contraction(";
    printExprIR(c->in.get());
    if (!c->summedIndices.empty()) {
      std::cout << "; sum=[";
      for (size_t i = 0; i < c->summedIndices.size(); ++i) {
        std::cout << c->summedIndices[i];
        if (i + 1 < c->summedIndices.size())
          std::cout << ",";
      }
      std::cout << "]";
    }
    std::cout << ")";
    printType();
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *r = static_cast<const IndexRenameIR *>(e);
    std::cout << "index_rename(";
    printExprIR(r->in.get());
    std::cout << "; " << r->from << "->" << r->to << ")";
    printType();
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *p = static_cast<const IndexPermuteIR *>(e);
    std::cout << "index_permute(";
    printExprIR(p->in.get());
    std::cout << "; order=[";
    for (size_t i = 0; i < p->order.size(); ++i) {
      std::cout << p->order[i];
      if (i + 1 < p->order.size())
        std::cout << ",";
    }
    std::cout << "])";
    printType();
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *t = static_cast<const TraceIR *>(e);
    std::cout << "trace(";
    printExprIR(t->in.get());
    if (!t->tracedIndices.empty()) {
      std::cout << "; idx=[";
      for (size_t i = 0; i < t->tracedIndices.size(); ++i) {
        std::cout << t->tracedIndices[i];
        if (i + 1 < t->tracedIndices.size())
          std::cout << ",";
      }
      std::cout << "]";
    }
    std::cout << ")";
    printType();
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *d = static_cast<const PartialDerivativeIR *>(e);
    std::cout << "partial_" << d->coordIndex << "(";
    printExprIR(d->in.get());
    std::cout << ")";
    printType();
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *g = static_cast<const GradientIR *>(e);
    std::cout << "gradient(";
    printExprIR(g->in.get());
    std::cout << ")";
    printType();
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *d = static_cast<const CovariantDerivativeIR *>(e);
    std::cout << "covariant_" << d->derivIndex << "(";
    printExprIR(d->in.get());
    std::cout << "; contra=" << (d->contravariant ? "true" : "false")
              << ", gamma=" << (d->hasConnectionTensor ? "present" : "missing")
              << ")";
    printType();
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *d = static_cast<const DivergenceIR *>(e);
    std::cout << "divergence(";
    printExprIR(d->in.get());
    if (!d->contractedIndex.empty())
      std::cout << "; idx=" << d->contractedIndex;
    std::cout << ")";
    printType();
    return;
  }
  }
}

inline void printModuleIR(const ModuleIR &m) {
  std::cout << "BackendModuleIR:\n";

  if (m.simulation) {
    std::cout << "  Simulation:\n";
    std::cout << "    dim = " << m.simulation->dimension << "\n";
    std::cout << "    dt  = " << m.simulation->time.dt << "\n";
  }

  if (m.initialData) {
    std::cout << "  InitialData:\n";
    if (m.initialData->hasMetric4) {
      std::cout << "    metric4 " << m.initialData->metric4.name << " [";
      for (size_t i = 0; i < m.initialData->metric4.indices.size(); ++i) {
        std::cout << m.initialData->metric4.indices[i];
        if (i + 1 < m.initialData->metric4.indices.size())
          std::cout << ",";
      }
      std::cout << "]\n";
    }
    if (m.initialData->hasDecomposed) {
      std::cout << "    alpha = ";
      printInitExpr(m.initialData->decomposed.alphaExpr.get());
      std::cout << "\n";
      if (!m.initialData->decomposed.gammaUExpr.empty()) {
        std::cout << "    gammaU = [";
        for (size_t i = 0; i < m.initialData->decomposed.gammaUExpr.size();
             ++i) {
          printInitExpr(m.initialData->decomposed.gammaUExpr[i].get());
          if (i + 1 < m.initialData->decomposed.gammaUExpr.size())
            std::cout << ",";
        }
        std::cout << "]\n";
      }
    }
    if (m.initialData->split3p1.enabled) {
      std::cout << "    split_3p1:\n";
      if (m.initialData->split3p1.hasAlpha)
        std::cout << "      alpha -> " << m.initialData->split3p1.alphaField
                  << "\n";
      if (m.initialData->split3p1.hasBeta)
        std::cout << "      beta -> " << m.initialData->split3p1.betaField
                  << "\n";
      if (m.initialData->split3p1.hasGamma)
        std::cout << "      gamma -> " << m.initialData->split3p1.gammaField
                  << "\n";
      if (m.initialData->split3p1.hasGammaU)
        std::cout << "      gammaU -> " << m.initialData->split3p1.gammaUField
                  << "\n";
    }
  }

  if (m.constraintProblem) {
    const auto &problem = *m.constraintProblem;
    std::cout << "  ConstraintProblem " << problem.name << ":\n";
    if (problem.geometry.enabled) {
      std::cout << "    Geometry " << problem.geometry.kind << " metric="
                << problem.geometry.metricName << " inverse_metric="
                << problem.geometry.inverseMetricName << " radial_scale=";
      printExprIR(problem.geometry.radialScale.get());
      std::cout << " tangential_scale=";
      printExprIR(problem.geometry.tangentialScale.get());
      std::cout << "\n";
    }
    std::cout << "    Domains:\n";
    for (const auto &domain : problem.domains) {
      std::cout << "      " << domain.name
                << " coordinates=" << domain.coordinates
                << " topology=" << domain.topology << " basis=" << domain.basis
                << " resolution=[";
      for (size_t i = 0; i < domain.resolution.size(); ++i) {
        std::cout << domain.resolution[i];
        if (i + 1 < domain.resolution.size())
          std::cout << ",";
      }
      std::cout << "]";
      if (!domain.bounds.empty()) {
        std::cout << " bounds=[";
        for (size_t i = 0; i < domain.bounds.size(); ++i) {
          std::cout << domain.bounds[i];
          if (i + 1 < domain.bounds.size())
            std::cout << ",";
        }
        std::cout << "]";
      }
      std::cout << "\n";
    }
    std::cout << "    Unknowns:\n";
    for (const auto &unknown : problem.unknowns)
      std::cout << "      " << unknown.name << " (up=" << unknown.tensorType.up
                << ",down=" << unknown.tensorType.down
                << ",symmetric=" << (unknown.symmetric ? "true" : "false")
                << ")\n";
    std::cout << "    Residuals:\n";
    for (const auto &equation : problem.equations) {
      std::cout << "      " << equation.name << " = ";
      printExprIR(equation.residual.get());
      std::cout << "\n";
    }
    std::cout << "    Boundaries:\n";
    for (const auto &boundary : problem.boundaries) {
      std::cout << "      " << boundary.region << ":\n";
      for (const auto &condition : boundary.conditions) {
        std::cout << "        " << condition.unknown << " = ";
        printExprIR(condition.rhs.get());
        std::cout << "\n";
      }
    }
    std::cout << "    Interfaces:\n";
    for (const auto &interface : problem.interfaces)
      std::cout << "      " << interface.innerDomain << " -> "
                << interface.outerDomain << " (C0,C1)\n";
    std::cout << "    Seeds:\n";
    for (const auto &seed : problem.seeds) {
      std::cout << "      " << seed.unknown << " = ";
      printExprIR(seed.rhs.get());
      std::cout << "\n";
    }
    if (problem.cttReconstruction.enabled) {
      std::cout << "    Reconstruct CTT: conformal_factor="
                << problem.cttReconstruction.conformalFactor
                << " radial_vector="
                << problem.cttReconstruction.radialVectorPotential
                << " mean_curvature=";
      printExprIR(problem.cttReconstruction.meanCurvature.get());
      std::cout << "\n";
    }
    std::cout << "    Solve: nonlinear=" << problem.solve.nonlinear
              << " linear=" << problem.solve.linear
              << " tolerance=" << problem.solve.tolerance
              << " max_iterations=" << problem.solve.maxIterations << "\n";
  }

  std::cout << "  Fields:\n";
  for (const auto &f : m.fields) {
    std::cout << "    " << f.name << " (up=" << f.tensorType.up
              << ",down=" << f.tensorType.down << ")\n";
  }

  if (!m.prints.empty()) {
    std::cout << "  Prints:\n";
    for (const auto &p : m.prints) {
      std::cout << "    print " << p.label << " (up=" << p.tensorType.up
                << ",down=" << p.tensorType.down << ")\n";
    }
  }

  std::cout << "  Evolutions:\n";
  for (const auto &evo : m.evolutions) {
    std::cout << "    Evolution " << evo.name << " {\n";
    for (const auto &eq : evo.equations) {
      std::cout << "      dt " << eq.fieldName;
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
      printExprIR(eq.rhs.get());
      std::cout << "\n";
    }
    std::cout << "    }\n";
  }
}

} // namespace tensorium::backend
