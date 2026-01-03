
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
    std::cout << "    " << f.name << " (up=" << f.up << ",down=" << f.down
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
