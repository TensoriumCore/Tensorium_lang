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
  static TensorKind deduceKind(int up, int down) {
    if (up == 0 && down == 0)
      return TensorKind::Scalar;
    if (up == 1 && down == 0)
      return TensorKind::Vector;
    if (up == 0 && down == 1)
      return TensorKind::Covector;
    if (up == 0 && down == 2)
      return TensorKind::CovTensor2;
    if (up == 2 && down == 0)
      return TensorKind::ConTensor2;
    if (up == 0 && down == 3)
      return TensorKind::CovTensor3;
    if (up == 3 && down == 0)
      return TensorKind::ConTensor3;
    if (up == 0 && down == 4)
      return TensorKind::CovTensor4;
    if (up == 4 && down == 0)
      return TensorKind::ConTensor4;
    return TensorKind::MixedTensor;
  }

  static void annotateType(const IndexedExpr *expr, const TensorType &tt) {
    auto *mut = const_cast<IndexedExpr *>(expr);
    mut->inferredType.kind = deduceKind(tt.up, tt.down);
    mut->inferredType.up = tt.up;
    mut->inferredType.down = tt.down;
  }

  TensorType tensorTypeFromDesc(const TensorTypeDesc &desc) const {
    return TensorType{desc.up, desc.down};
  }

  bool isPartialDerivative(const std::string &name) const {
    if (name.size() != 3)
      return false;
    if (name[0] != 'd' || name[1] != '_')
      return false;
    char c = name[2];
    return (c == 'i' || c == 'j' || c == 'k' || c == 'l' || c == 'm' ||
            c == 'n');
  }

  bool isScalarExpr(const IndexedExpr *e) const {
    try {
      return inferImpl(e, true).isScalar();
    } catch (...) {
      return false;
    }
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
      const std::string &cal = c->callee;

      if (cal == "contract") {
        if (c->args.size() != 1)
          throw std::runtime_error("contract() expects 1 argument");

        int tmp[256] = {0};
        collectIndexCounts(c->args[0].get(), tmp);

        for (char idx : {'i', 'j', 'k', 'l', 'm', 'n'}) {
          int cc = tmp[(unsigned char)idx];
          if (cc == 0)
            continue;
          if (cc == 1) {
            counts[(unsigned char)idx]++;
            continue;
          }
          if (cc == 2)
            continue;
          throw std::runtime_error(
              std::string("Ambiguous contraction: index '") + idx +
              "' appears " + std::to_string(cc) + " times.");
        }
        return;
      }

      for (const auto &arg : c->args)
        collectIndexCounts(arg.get(), counts);

      if (isPartialDerivative(cal)) {
        char idx = cal[2];
        counts[(unsigned char)idx]++;
        return;
      }

      bool contra = false;
      char nidx = 0;
      if (isCovariantDerivative(cal, contra, nidx)) {
        counts[(unsigned char)nidx]++;
        return;
      }

      return;
    }
  }

  void collectAdditiveTerms(const IndexedExpr *e,
                            std::vector<const IndexedExpr *> &out) const {
    if (!e)
      return;

    if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
      if (b->op == '+' || b->op == '-') {
        collectAdditiveTerms(b->lhs.get(), out);
        collectAdditiveTerms(b->rhs.get(), out);
        return;
      }

      if (b->op == '*') {
        const IndexedExpr *L = b->lhs.get();
        const IndexedExpr *R = b->rhs.get();
        if (isScalarExpr(L)) {
          collectAdditiveTerms(R, out);
          return;
        }
        if (isScalarExpr(R)) {
          collectAdditiveTerms(L, out);
          return;
        }
      }

      if (b->op == '/') {
        const IndexedExpr *R = b->rhs.get();
        if (isScalarExpr(R)) {
          collectAdditiveTerms(b->lhs.get(), out);
          return;
        }
      }
    }

    out.push_back(e);
  }

public:
  TensorType inferImpl(const IndexedExpr *e, bool allowRepeated) const {
    if (!e)
      throw std::runtime_error("null expression in tensor type inference");

    if (dynamic_cast<const IndexedNumber *>(e)) {
      TensorType t{0, 0};
      annotateType(e, t);
      return t;
    }

    if (auto v = dynamic_cast<const IndexedVar *>(e)) {
      if (!allowRepeated) {
        std::unordered_set<std::string> seen;
        for (const auto &name : v->tensorIndexNames) {
          if (!name.empty()) {
            if (!seen.insert(name).second) {
              throw std::runtime_error("Implicit trace '" + v->name + "[" +
                                       name + "," + name +
                                       "]' is forbidden; use explicit trace()");
            }
          }
        }
      }

      TensorType t;
      switch (v->tensorKind) {
      case TensorKind::Scalar:
        t = {0, 0};
        break;
      case TensorKind::Vector:
        t = {1, 0};
        break;
      case TensorKind::Covector:
        t = {0, 1};
        break;
      case TensorKind::CovTensor2:
        t = {0, 2};
        break;
      case TensorKind::ConTensor2:
        t = {2, 0};
        break;
      case TensorKind::CovTensor3:
        t = {0, 3};
        break;
      case TensorKind::ConTensor3:
        t = {3, 0};
        break;
      case TensorKind::CovTensor4:
        t = {0, 4};
        break;
      case TensorKind::ConTensor4:
        t = {4, 0};
        break;
      case TensorKind::MixedTensor:
        t = {v->up, v->down};
        break;
      }
      annotateType(e, t);
      return t;
    }

    if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
      TensorType lt = inferImpl(b->lhs.get(), allowRepeated);
      TensorType rt = inferImpl(b->rhs.get(), allowRepeated);

      if (b->op == '+' || b->op == '-') {
        if (!lt.sameVariance(rt))
          throw std::runtime_error(
              "tensor addition/subtraction requires identical variance");
        annotateType(e, lt);
        return lt;
      }

      if (b->op == '*') {
        TensorType res{lt.up + rt.up, lt.down + rt.down};
        annotateType(e, res);
        return res;
      }

      if (b->op == '/') {
        if (!rt.isScalar())
          throw std::runtime_error(
              "division by non-scalar tensor is not allowed");
        annotateType(e, lt);
        return lt;
      }

      annotateType(e, lt);
      return lt;
    }

    if (auto call = dynamic_cast<const IndexedCall *>(e)) {
      const std::string &cal = call->callee;

      if (cal == "contract") {
        if (call->args.size() != 1)
          throw std::runtime_error("contract() expects 1 argument");

        const IndexedExpr *arg = call->args[0].get();
        TensorType t = inferImpl(arg, true);

        int counts[256] = {0};
        collectIndexCounts(arg, counts);

        int freeCount = 0;
        int contracted = 0;

        for (char idx : {'i', 'j', 'k', 'l', 'm', 'n'}) {
          int c = counts[(unsigned char)idx];
          if (c == 0)
            continue;
          if (c == 1) {
            freeCount += 1;
            continue;
          }
          if (c == 2) {
            contracted += 1;
            continue;
          }
          throw std::runtime_error(
              std::string("Ambiguous contraction: index '") + idx +
              "' appears " + std::to_string(c) + " times.");
        }

        if (contracted == 0)
          throw std::runtime_error(
              "contract() expects at least one repeated index");

        if (t.up == 0 && t.down > 0) {
          TensorType res{0, freeCount};
          annotateType(e, res);
          return res;
        }

        if (t.down == 0 && t.up > 0) {
          TensorType res{freeCount, 0};
          annotateType(e, res);
          return res;
        }

        int remove = 2 * contracted;
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
              "internal error: contract() could not remove requested rank");

        TensorType res{up, down};
        annotateType(e, res);
        return res;
      }

      if (isPartialDerivative(cal)) {
        if (call->args.size() != 1)
          throw std::runtime_error("d_* expects exactly 1 argument");
        TensorType argT = inferImpl(call->args[0].get(), allowRepeated);
        TensorType res{argT.up, argT.down + 1};
        annotateType(e, res);
        return res;
      }

      bool contra = false;
      char idx = 0;
      if (isCovariantDerivative(cal, contra, idx)) {
        if (call->args.size() != 1)
          throw std::runtime_error("nabla expects exactly 1 argument");
        TensorType t = inferImpl(call->args[0].get(), allowRepeated);
        TensorType res = contra ? TensorType{t.up + 1, t.down}
                                : TensorType{t.up, t.down + 1};
        annotateType(e, res);
        return res;
      }

      if (cal == "laplacian") {
        if (call->args.size() != 1)
          throw std::runtime_error("laplacian() expects exactly 1 argument");
        TensorType argT = inferImpl(call->args[0].get(), allowRepeated);
        if (!argT.isScalar())
          throw std::runtime_error("laplacian() expects scalar argument");
        TensorType res{0, 0};
        annotateType(e, res);
        return res;
      }

      if (call->isExtern) {
        if (call->paramTypes.size() != call->args.size())
          throw std::runtime_error("extern call parameter mismatch");
        for (size_t i = 0; i < call->args.size(); ++i) {
          TensorType expected = tensorTypeFromDesc(call->paramTypes[i]);
          TensorType actual = inferImpl(call->args[i].get(), allowRepeated);
          if (!actual.sameVariance(expected)) {
            throw std::runtime_error("extern function '" + cal +
                                     "' argument variance mismatch");
          }
        }
        TensorType ret = tensorTypeFromDesc(call->returnType);
        annotateType(e, ret);
        return ret;
      }

      for (auto &arg : call->args) {
        TensorType t = inferImpl(arg.get(), allowRepeated);
        if (!t.isScalar())
          throw std::runtime_error("function '" + cal +
                                   "' expects scalar argument");
      }

      TensorType res{0, 0};
      annotateType(e, res);
      return res;
    }

    throw std::runtime_error("unsupported expression in tensor type inference");
  }

  TensorType infer(const IndexedExpr *e) const { return inferImpl(e, false); }
  void checkAssignmentVariance(const TensorType &lhs,
                               const std::vector<std::string> &lhsIndexNames,
                               const IndexedExpr *rhs) const {
    TensorType rhsRaw = infer(rhs);

    bool lhsSet[256] = {false};
    for (const auto &nm : lhsIndexNames) {
      if (nm.empty())
        continue;
      char c = nm[0];
      if (!isTensorIndexChar(c))
        throw std::runtime_error("Invalid tensor index '" + nm + "'");
      lhsSet[(unsigned char)c] = true;
    }

    std::vector<const IndexedExpr *> terms;
    collectAdditiveTerms(rhs, terms);

    for (const IndexedExpr *t : terms) {
      int counts[256] = {0};
      collectIndexCounts(t, counts);

      for (char idx : {'i', 'j', 'k', 'l', 'm', 'n'}) {
        int c = counts[(unsigned char)idx];
        bool inLhs = lhsSet[(unsigned char)idx];

        if (c == 0)
          continue;

        if (c >= 3) {
          throw std::runtime_error(
              std::string("Ambiguous contraction: index '") + idx +
              "' appears " + std::to_string(c) + " times.");
        }

        if (inLhs) {
          if (c != 1) {
            throw std::runtime_error(
                std::string("Invalid Einstein: LHS index '") + idx +
                "' must appear exactly once in RHS.");
          }
        } else {
          if (c == 1) {
            throw std::runtime_error(std::string("Free index '") + idx +
                                     "' appears only in RHS and not LHS.");
          }
        }
      }
    }

    // Variance/type check: rely on infer(rhs) which already accounts for
    // contract() and derivative arity rules.
    TensorType rhsEff = rhsRaw;

    if (!lhs.sameVariance(rhsEff)) {
      throw std::runtime_error(
          "tensor assignment mismatch: LHS(" + std::to_string(lhs.up) + "," +
          std::to_string(lhs.down) + ") vs RHS(" + std::to_string(rhsEff.up) +
          "," + std::to_string(rhsEff.down) + ")");
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
