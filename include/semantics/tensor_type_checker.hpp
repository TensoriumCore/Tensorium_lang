#pragma once
#include "indexed_ast.hpp"
#include <stdexcept>
#include <string>

struct TensorType {
  int up = 0;
  int down = 0;

  bool isScalar() const { return up == 0 && down == 0; }
  int rank() const { return up + down; }

  bool sameVariance(const TensorType &o) const {
    return up == o.up && down == o.down;
  }
};

class TensorTypeChecker {
public:
  TensorType infer(const IndexedExpr *e) const {
    if (!e)
      throw std::runtime_error("null expression in tensor type inference");

    if (auto call = dynamic_cast<const IndexedCall *>(e)) {
      if (call->callee == "contract") {
        return TensorType{0, 0}; 
      }
    }

    if (dynamic_cast<const IndexedNumber *>(e)) {
      return TensorType{0, 0};
    }

    if (auto v = dynamic_cast<const IndexedVar *>(e)) {
      switch (v->tensorKind) {
      case TensorKind::Scalar:
        return TensorType{0, 0};
      case TensorKind::Vector:
        return TensorType{1, 0};
      case TensorKind::Covector:
        return TensorType{0, 1};
      case TensorKind::CovTensor2:
        return TensorType{0, 2};
      case TensorKind::ConTensor2:
        return TensorType{2, 0};
      case TensorKind::MixedTensor:
        return TensorType{v->up, v->down};
      }
    }

    if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
      TensorType lt = infer(b->lhs.get());
      TensorType rt = infer(b->rhs.get());

      if (b->op == '+' || b->op == '-') {
        if (!lt.sameVariance(rt))
          throw std::runtime_error(
              "tensor addition/subtraction requires identical variance");
        return lt;
      }

      if (b->op == '*') {
        if (lt.isScalar())
          return rt;
        if (rt.isScalar())
          return lt;
        throw std::runtime_error("tensor * tensor requires explicit contraction");
      }

      if (b->op == '/') {
        if (!rt.isScalar())
          throw std::runtime_error(
              "division by non-scalar tensor is not allowed");
        return lt;
      }

      return lt;
    }

    if (auto call = dynamic_cast<const IndexedCall *>(e)) {
      for (auto &arg : call->args) {
        TensorType t = infer(arg.get());
        if (!t.isScalar())
          throw std::runtime_error("function '" + call->callee +
                                   "' expects scalar argument");
      }
      return TensorType{0, 0};
    }

    throw std::runtime_error("unsupported expression in tensor type inference");
  }

  void checkAssignmentVariance(const TensorType &lhs,
                               const IndexedExpr *rhs) const {
    TensorType r = infer(rhs);
    if (!lhs.sameVariance(r)) {
      throw std::runtime_error("tensor assignment mismatch: LHS(" +
                               std::to_string(lhs.up) + "," +
                               std::to_string(lhs.down) + ") vs RHS(" +
                               std::to_string(r.up) + "," +
                               std::to_string(r.down) + ")");
    }
  }

  void checkMetricAssignment(const IndexedAssignment &a) const {
    TensorType t = infer(a.rhs.get());
    if (!t.isScalar()) {
      throw std::runtime_error(
          "metric assignment to '" + a.tensor +
          "' must be scalar (got tensor rank=" + std::to_string(t.rank()) +
          ")");
    }
  }
};
