#pragma once
#include "../ast.hpp"
#include "../semantics/indexed_ast.hpp"
#include <iostream>

static void printExpr(const Expr *e);

static void printNumber(const NumberExpr *n) { std::cout << n->value; }

static void printVar(const VarExpr *v) { std::cout << v->name; }

static void printBinary(const BinaryExpr *b) {
  std::cout << "(";
  printExpr(b->lhs.get());
  std::cout << " " << b->op << " ";
  printExpr(b->rhs.get());
  std::cout << ")";
}

static void printParen(const ParenExpr *p) {
  std::cout << "(";
  printExpr(p->inner.get());
  std::cout << ")";
}

static void printCall(const CallExpr *c) {
  std::cout << c->callee << "(";
  for (size_t i = 0; i < c->args.size(); ++i) {
    printExpr(c->args[i].get());
    if (i + 1 < c->args.size())
      std::cout << ", ";
  }
  std::cout << ")";
}

static void printExpr(const Expr *e) {
  if (auto n = dynamic_cast<const NumberExpr *>(e)) {
    printNumber(n);
  } else if (auto v = dynamic_cast<const VarExpr *>(e)) {
    printVar(v);
  } else if (auto b = dynamic_cast<const BinaryExpr *>(e)) {
    printBinary(b);
  } else if (auto p = dynamic_cast<const ParenExpr *>(e)) {
    printParen(p);
  } else if (auto c = dynamic_cast<const CallExpr *>(e)) {
    printCall(c);
  } else if (auto iv = dynamic_cast<const IndexedVarExpr *>(e)) {
    std::cout << iv->base << "[";
    for (size_t i = 0; i < iv->indices.size(); ++i) {
      std::cout << iv->indices[i];
      if (i + 1 < iv->indices.size())
        std::cout << ",";
    }
    std::cout << "]";
  } else {
    std::cout << "<unknown_expr>";
  }
}

static void printField(const FieldDecl &f) {
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

static void printMetric(const MetricDecl &m, int idx) {
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


static void printEvolution(const EvolutionDecl &evo, int idx) {
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
