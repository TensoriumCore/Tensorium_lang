
#pragma once
#include "indexed_ast.hpp"
#include <stdexcept>
#include <string>

struct TensorType {
  int rank;
  bool isScalar() const { return rank == 0; }
};

class TensorTypeChecker {
public:
  TensorType infer(const IndexedExpr *e) const {
    if (auto n = dynamic_cast<const IndexedNumber *>(e)) {
      (void)n;
      return TensorType{0};
    }

    if (auto v = dynamic_cast<const IndexedVar *>(e)) {
      switch (v->kind) {
      case IndexedVarKind::Coordinate:
      case IndexedVarKind::Local:
      case IndexedVarKind::Parameter:
        return TensorType{0};
      case IndexedVarKind::Field:
        switch (v->tensorKind) {
        case TensorKind::Scalar:
          return TensorType{0};
        case TensorKind::Vector:
          return TensorType{1};
        case TensorKind::Tensor2:
          return TensorType{2};
        }
      }
    }

    if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
      TensorType lt = infer(b->lhs.get());
      TensorType rt = infer(b->rhs.get());

      if (b->op == '+' || b->op == '-') {
        if (lt.rank != rt.rank)
          throw std::runtime_error("incompatible tensor ranks for + or -");
        return lt;
      }

      if (b->op == '*') {
        if (lt.rank != 0 && rt.rank != 0)
          throw std::runtime_error(
              "tensor-tensor product not supported yet (both non-scalar)");
        return (lt.rank != 0) ? lt : rt;
      }

      if (b->op == '/') {
        if (rt.rank != 0)
          throw std::runtime_error("division by non-scalar tensor");
        return lt;
      }

      return lt;
    }

    if (auto c = dynamic_cast<const IndexedCall *>(e)) {
      for (auto &arg : c->args) {
        TensorType t = infer(arg.get());
        if (!t.isScalar())
          throw std::runtime_error("function '" + c->callee +
                                   "' expects scalar arguments");
      }
      return TensorType{0};
    }

    throw std::runtime_error("unsupported expression in tensor type inference");
  }

  void checkMetricAssignment(const IndexedAssignment &a) const {
    TensorType t = infer(a.rhs.get());
    if (!t.isScalar()) {
      throw std::runtime_error(
          "metric assignment to '" + a.tensor +
          "' must be scalar (got tensor rank=" + std::to_string(t.rank) + ")");
    }
  }
};
