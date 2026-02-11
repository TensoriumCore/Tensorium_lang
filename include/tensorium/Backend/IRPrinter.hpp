
#pragma once
#include "tensorium/Backend/DomainIR.hpp"
#include <iostream>

namespace tensorium::backend {

inline void printExprIR(const ExprIR *e) {
  if (!e) {
    std::cout << "<null>";
    return;
  }

  auto printType = [&]() {
    std::cout << "[u=" << e->exprType.up << ",d=" << e->exprType.down
              << "]";
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
              << ", gamma="
              << (d->hasConnectionTensor ? "present" : "missing") << ")";
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

  std::cout << "  Fields:\n";
  for (const auto &f : m.fields) {
    std::cout << "    " << f.name << " (up=" << f.tensorType.up << ",down=" << f.tensorType.down
              << ")\n";
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
