#pragma once
#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/IndexedAST.hpp"
#include <stdexcept>
#include <string>

namespace tensorium {

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
  bool isPartialDerivative(const std::string &name) const {
    if (name.size() != 3)
      return false;
    if (name[0] != 'd' || name[1] != '_')
      return false;
    char c = name[2];
    return (c == 'i' || c == 'j' || c == 'k' || c == 'l' || c == 'm' ||
            c == 'n');
  }

  bool isCovariantDerivative(const std::string &name, bool &contravariant,
                             char &index) const {
    if (name.size() == 7 && name.rfind("nabla_", 0) == 0) {
      index = name[6];
      contravariant = false;
      return true;
    }
    if (name.size() == 7 && name.rfind("nabla^", 0) == 0) {
      index = name[6];
      contravariant = true;
      return true;
    }
    return false;
  }

  static bool isTensorIndexChar(char c) {
    return (c == 'i' || c == 'j' || c == 'k' || c == 'l' || c == 'm' ||
            c == 'n');
  }

  void collectIndexCounts(const IndexedExpr *e, int counts[256]) const {
    if (!e)
      return;

    if (auto v = dynamic_cast<const IndexedVar *>(e)) {
      for (const auto &name : v->tensorIndexNames) {
        if (!name.empty()) {
          char c = name[0];
          if (isTensorIndexChar(c))
            counts[(unsigned char)c]++;
        }
      }
      return;
    }

    if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
      collectIndexCounts(b->lhs.get(), counts);
      collectIndexCounts(b->rhs.get(), counts);
      return;
    }

    if (auto c = dynamic_cast<const IndexedCall *>(e)) {
      for (const auto &arg : c->args)
        collectIndexCounts(arg.get(), counts);
      return;
    }
  }

public:
  TensorType infer(const IndexedExpr *e) const {
    if (!e)
      throw std::runtime_error("null expression in tensor type inference");

    if (dynamic_cast<const IndexedNumber *>(e))
      return TensorType{0, 0};

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
        return TensorType{lt.up + rt.up, lt.down + rt.down};
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
      const std::string &cal = call->callee;

      if (cal == "contract") {
        if (call->args.size() != 1)
          throw std::runtime_error("contract() expects 1 argument");

        const IndexedExpr *arg = call->args[0].get();
        TensorType t = infer(arg);

        int counts[256] = {0};
        collectIndexCounts(arg, counts);

        int pairs = 0;
        for (char idx : {'i', 'j', 'k', 'l', 'm', 'n'}) {
          int c = counts[(unsigned char)idx];
          if (c < 0)
            throw std::runtime_error("internal error: negative index count");
          pairs += (c / 2);
        }

        if (pairs == 0)
          throw std::runtime_error(
              "contract() expects at least one repeated index");

        int remove = 2 * pairs;
        int r = t.rank();
        if (r < remove)
          throw std::runtime_error(
              "contract() contraction exceeds tensor rank");

        int up = t.up;
        int down = t.down;

        int rem = remove;
        int takeDown = (down < rem) ? down : rem;
        down -= takeDown;
        rem -= takeDown;

        int takeUp = (up < rem) ? up : rem;
        up -= takeUp;
        rem -= takeUp;

        if (rem != 0)
          throw std::runtime_error(
              "contract() could not remove requested rank");

        return TensorType{up, down};
      }
      if (isPartialDerivative(cal)) {
        if (call->args.size() != 1)
          throw std::runtime_error("d_* expects exactly 1 argument");
        TensorType argT = infer(call->args[0].get());
        if (!argT.isScalar())
          throw std::runtime_error("d_* expects scalar argument");
        return TensorType{0, 1};
      }

      bool contra = false;
      char idx = 0;
      if (isCovariantDerivative(cal, contra, idx)) {
        if (call->args.size() != 1)
          throw std::runtime_error("nabla expects exactly 1 argument");
        TensorType t = infer(call->args[0].get());
        if (contra)
          return TensorType{t.up + 1, t.down};
        return TensorType{t.up, t.down + 1};
      }

      if (cal == "laplacian") {
        if (call->args.size() != 1)
          throw std::runtime_error("laplacian() expects exactly 1 argument");
        TensorType argT = infer(call->args[0].get());
        if (!argT.isScalar())
          throw std::runtime_error("laplacian() expects scalar argument");
        return TensorType{0, 0};
      }

      for (auto &arg : call->args) {
        TensorType t = infer(arg.get());
        if (!t.isScalar())
          throw std::runtime_error("function '" + cal +
                                   "' expects scalar argument");
      }
      return TensorType{0, 0};
    }

    throw std::runtime_error("unsupported expression in tensor type inference");
  }

  void checkAssignmentVariance(const TensorType &lhs,
                               const IndexedExpr *rhs) const {
    TensorType rhsType = infer(rhs);
    if (!lhs.sameVariance(rhsType)) {
      throw std::runtime_error(
          "tensor assignment mismatch: LHS(" + std::to_string(lhs.up) + "," +
          std::to_string(lhs.down) + ") vs RHS(" + std::to_string(rhsType.up) +
          "," + std::to_string(rhsType.down) + ")");
    }
  }

  void checkMetricAssignment(const IndexedAssignment &a) const {
    TensorType t = infer(a.rhs.get());
    if (!t.isScalar()) {
      throw std::runtime_error("metric assignment to '" + a.tensor +
                               "' must be scalar (got tensor rank=" +
                               std::to_string(t.rank()) + ")");
    }
  }
};

} // namespace tensorium
