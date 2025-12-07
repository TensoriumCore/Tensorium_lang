#pragma once
#include "../semantics/indexed_ast.hpp"
#include <iostream>

static void printIndexedExpr(const IndexedExpr *e);

static void printIndexedBinary(const IndexedBinary *b) {
  std::cout << "(";
  printIndexedExpr(b->lhs.get());
  std::cout << " " << b->op << " ";
  printIndexedExpr(b->rhs.get());
  std::cout << ")";
}

static void printIndexedExpr(const IndexedExpr *e) {
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

static void printIndexedMetric(const IndexedMetric &m) {
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

static void printIndexedEvolution(const IndexedEvolution &evo) {
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
