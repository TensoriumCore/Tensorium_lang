#pragma once
#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/IndexedAST.hpp"
#include "tensorium/Core/IndexSet.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

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
  struct IndexVarianceInfo {
    int contravariant = 0;
    int covariant = 0;
    bool metricCoupling = false;
  };
  struct IndexAnalysisEntry {
    int count = 0;
    IndexVarianceInfo variance;
    bool insideExplicitContraction = false;
    bool outsideExplicitContraction = false;
  };
  struct IndexAnalysisResult {
    std::array<IndexAnalysisEntry, 256> entries{};
    std::vector<std::string> freeIndices;
    std::vector<std::string> summedIndices;
    std::vector<std::string> ambiguousIndices;
    std::vector<std::string> duplicateWithinTensor;
  };
  bool connectionTensorAvailable = false;
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

  static int getDeclaredUpCount(const IndexedVar *v) {
    if (!v)
      return 0;
    switch (v->tensorKind) {
    case TensorKind::Scalar:
      return 0;
    case TensorKind::Vector:
      return 1;
    case TensorKind::Covector:
      return 0;
    case TensorKind::CovTensor2:
      return 0;
    case TensorKind::ConTensor2:
      return 2;
    case TensorKind::CovTensor3:
      return 0;
    case TensorKind::ConTensor3:
      return 3;
    case TensorKind::CovTensor4:
      return 0;
    case TensorKind::ConTensor4:
      return 4;
    case TensorKind::MixedTensor:
      return v->up;
    case TensorKind::Metric:
      return 0;
    case TensorKind::InverseMetric:
      return 2;
    }
    return 0;
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
    return core::isSpatialIndexChar(name[2]);
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

  bool isGradientBuiltin(const std::string &name) const {
    return name == "gradient" || name == "grad";
  }

  bool isDivergenceBuiltin(const std::string &name) const {
    return name == "divergence" || name == "div";
  }

  bool isChristoffelBuiltin(const std::string &name) const {
    return name == "christoffel";
  }

  bool isCovariantDerivativeBuiltin(const IndexedCall *call,
                                    bool &contravariant, char &index) const {
    if (!call)
      return false;
    if (isCovariantDerivative(call->callee, contravariant, index))
      return true;
    if (call->callee != "covariant_derivative")
      return false;
    if (call->args.size() != 2)
      throw std::runtime_error(
          "covariant_derivative(tensor, index) expects exactly 2 arguments");
    auto *idxVar = dynamic_cast<const IndexedVar *>(call->args[1].get());
    if (!idxVar || idxVar->name.size() != 1 ||
        !core::isSpatialIndexChar(idxVar->name[0])) {
      throw std::runtime_error(
          "covariant_derivative second argument must be a spatial index name");
    }
    contravariant = false;
    index = idxVar->name[0];
    return true;
  }

  void collectIndexAnalysis(const IndexedExpr *e, bool insideExplicitContract,
                            IndexAnalysisResult &analysis) const {
    if (!e)
      return;

    if (auto v = dynamic_cast<const IndexedVar *>(e)) {
      std::unordered_set<std::string> seen;
      for (const auto &name : v->tensorIndexNames) {
        if (!name.empty()) {
          if (!seen.insert(name).second)
            analysis.duplicateWithinTensor.push_back(v->name + "[" + name + "]");
          char c = name[0];
          if (core::isTensorIndexChar(c))
            analysis.entries[(unsigned char)c].count++;
          if (insideExplicitContract)
            analysis.entries[(unsigned char)c].insideExplicitContraction = true;
          else
            analysis.entries[(unsigned char)c].outsideExplicitContraction = true;
        }
      }
      for (size_t i = 0; i < v->tensorIndexNames.size(); ++i) {
        const auto &name = v->tensorIndexNames[i];
        if (name.empty())
          continue;
        char c = name[0];
        if (!core::isTensorIndexChar(c))
          continue;
        auto &entry = analysis.entries[(unsigned char)c];
        bool isUp = false;
        if (i < v->tensorIndexIsUp.size()) {
          isUp = v->tensorIndexIsUp[i];
        } else {
          isUp = static_cast<int>(i) < getDeclaredUpCount(v);
        }
        if (isUp)
          entry.variance.contravariant += 1;
        else
          entry.variance.covariant += 1;
        if (v->tensorKind == TensorKind::Metric ||
            v->tensorKind == TensorKind::InverseMetric)
          entry.variance.metricCoupling = true;
      }
      return;
    }

    if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
      collectIndexAnalysis(b->lhs.get(), insideExplicitContract, analysis);
      collectIndexAnalysis(b->rhs.get(), insideExplicitContract, analysis);
      return;
    }

    if (auto c = dynamic_cast<const IndexedCall *>(e)) {
      const std::string &cal = c->callee;

      if (isChristoffelBuiltin(cal)) {
        // christoffel(...) materializes tensor structure during IR lowering.
        // Skip raw argument index counting here to avoid false free/bound
        // collisions from helper arguments like gamma/gammaU.
        return;
      }

      if (cal == "contract") {
        if (c->args.size() != 1)
          throw std::runtime_error("contract() expects 1 argument");
        collectIndexAnalysis(c->args[0].get(), true, analysis);
        return;
      }

      for (const auto &arg : c->args)
        collectIndexAnalysis(arg.get(), insideExplicitContract, analysis);

      if (isPartialDerivative(cal)) {
        char idx = cal[2];
        auto &entry = analysis.entries[(unsigned char)idx];
        entry.count++;
        // d_i(...) introduces a covariant derivative index.
        entry.variance.covariant += 1;
        return;
      }

      bool contra = false;
      char nidx = 0;
      if (isCovariantDerivativeBuiltin(c, contra, nidx)) {
        analysis.entries[(unsigned char)nidx].count++;
        if (contra)
          analysis.entries[(unsigned char)nidx].variance.contravariant += 1;
        else
          analysis.entries[(unsigned char)nidx].variance.covariant += 1;
        return;
      }

      if (isGradientBuiltin(cal) || isDivergenceBuiltin(cal)) {
        if (c->args.empty())
          throw std::runtime_error(cal + "() expects at least 1 argument");
        return;
      }

      return;
    }
  }

  IndexAnalysisResult analyzeIndices(const IndexedExpr *e) const {
    IndexAnalysisResult analysis;
    collectIndexAnalysis(e, false, analysis);

    for (char idx : core::kTensorIndices) {
      const auto &entry = analysis.entries[(unsigned char)idx];
      if (entry.count == 1)
        analysis.freeIndices.push_back(std::string(1, idx));
      else if (entry.count == 2)
        analysis.summedIndices.push_back(std::string(1, idx));
      else if (entry.count >= 3)
        analysis.ambiguousIndices.push_back(std::string(1, idx));
    }

    return analysis;
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

  TensorType inferContractResultType(const IndexedExpr *arg,
                                     const TensorType &argType) const {
    auto analysis = analyzeIndices(arg);

    if (!analysis.ambiguousIndices.empty()) {
      throw std::runtime_error(
          std::string("Ambiguous contraction: index '") +
          analysis.ambiguousIndices.front() +
          "' appears 3 or more times.");
    }

    const int contracted = static_cast<int>(analysis.summedIndices.size());
    if (contracted == 0)
      throw std::runtime_error("contract() expects at least one repeated index");

    int up = argType.up;
    int down = argType.down;
    int unresolvedPairs = 0;

    for (const auto &name : analysis.summedIndices) {
      const auto &entry = analysis.entries[(unsigned char)name[0]];
      const auto &variance = entry.variance;
      const bool mixedVariance =
          (variance.contravariant > 0 && variance.covariant > 0);
      if (!mixedVariance) {
        ++unresolvedPairs;
        continue;
      }
      if (up == 0 || down == 0) {
        throw std::runtime_error(
            "internal error: mixed-variance contraction rank underflow");
      }
      --up;
      --down;
    }

    // Compatibility rule: legacy Tensorium accepted same-variance contractions
    // and removed two ranks with down-priority. Keep that behavior while
    // fixing mixed-variance contractions to remove one up + one down.
    int rem = 2 * unresolvedPairs;
    int takeDown = (down < rem) ? down : rem;
    down -= takeDown;
    rem -= takeDown;

    int takeUp = (up < rem) ? up : rem;
    up -= takeUp;
    rem -= takeUp;

    if (rem != 0) {
      throw std::runtime_error(
          "internal error: contract() could not remove requested rank");
    }

    const int expectedRank = argType.rank() - (2 * contracted);
    if (expectedRank < 0 || expectedRank != (up + down)) {
      throw std::runtime_error(
          "internal error: contract() produced inconsistent inferred rank");
    }

    return TensorType{up, down};
  }

public:
  explicit TensorTypeChecker(bool hasConnectionTensor = false)
      : connectionTensorAvailable(hasConnectionTensor) {}

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
      case TensorKind::Metric:
        t = {0, 2};
        break;
      case TensorKind::InverseMetric:
        t = {2, 0};
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

      if (isChristoffelBuiltin(cal)) {
        if (call->args.size() != 2)
          throw std::runtime_error(
              "christoffel(gamma, gammaU) expects exactly 2 arguments");
        TensorType gammaT = inferImpl(call->args[0].get(), allowRepeated);
        TensorType gammaUT = inferImpl(call->args[1].get(), allowRepeated);
        if (!(gammaT.up == 0 && gammaT.down == 2)) {
          throw std::runtime_error(
              "christoffel() first argument must be covariant rank-2");
        }
        if (!(gammaUT.up == 2 && gammaUT.down == 0)) {
          throw std::runtime_error(
              "christoffel() second argument must be contravariant rank-2");
        }
        TensorType res{1, 2};
        annotateType(e, res);
        return res;
      }

      if (cal == "contract") {
        if (call->args.size() != 1)
          throw std::runtime_error(cal + "() expects 1 argument");

        const IndexedExpr *arg = call->args[0].get();
        TensorType t = inferImpl(arg, true);
        TensorType res = inferContractResultType(arg, t);
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
      if (isCovariantDerivativeBuiltin(call, contra, idx)) {
        if (!connectionTensorAvailable) {
          throw std::runtime_error(
              "Covariant derivative requires connection tensor Gamma (rank-3 field)");
        }
        TensorType t = inferImpl(call->args[0].get(), allowRepeated);
        TensorType res = contra ? TensorType{t.up + 1, t.down}
                                : TensorType{t.up, t.down + 1};
        annotateType(e, res);
        return res;
      }

      if (isGradientBuiltin(cal)) {
        if (call->args.size() != 1)
          throw std::runtime_error("gradient() expects exactly 1 argument");
        TensorType t = inferImpl(call->args[0].get(), allowRepeated);
        TensorType res{t.up, t.down + 1};
        annotateType(e, res);
        return res;
      }

      if (isDivergenceBuiltin(cal)) {
        if (call->args.size() != 1)
          throw std::runtime_error("divergence() expects exactly 1 argument");
        if (!connectionTensorAvailable) {
          throw std::runtime_error(
              "Divergence requires connection tensor Gamma (rank-3 field)");
        }
        TensorType t = inferImpl(call->args[0].get(), allowRepeated);
        if (t.rank() == 0) {
          throw std::runtime_error("divergence() expects non-scalar argument");
        }
        TensorType res = (t.up > 0) ? TensorType{t.up - 1, t.down}
                                    : TensorType{t.up, t.down - 1};
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
      if (!core::isTensorIndexChar(c))
        throw std::runtime_error("Invalid tensor index '" + nm + "'");
      lhsSet[(unsigned char)c] = true;
    }

    std::vector<const IndexedExpr *> terms;
    collectAdditiveTerms(rhs, terms);

    for (const IndexedExpr *t : terms) {
      auto analysis = analyzeIndices(t);
      if (!analysis.ambiguousIndices.empty()) {
        throw std::runtime_error(
            std::string("Ambiguous contraction: index '") +
            analysis.ambiguousIndices.front() +
            "' appears 3 or more times.");
      }

      for (char idx : core::kTensorIndices) {
        const auto &entry = analysis.entries[(unsigned char)idx];
        int c = entry.count;
        bool inLhs = lhsSet[(unsigned char)idx];

        if (c == 0)
          continue;

        if (inLhs) {
          if (c != 1) {
            throw std::runtime_error(
                std::string("Index collision: symbol '") + idx +
                "' is both free and bound; rename one index in RHS.");
          }
        } else {
          if (c == 1) {
            throw std::runtime_error(std::string("Free index '") + idx +
                                     "' appears only in RHS and not LHS.");
          }
          if (c == 2) {
            bool explicitOnly =
                entry.insideExplicitContraction &&
                !entry.outsideExplicitContraction;
            if (explicitOnly)
              continue;
            const auto &info = entry.variance;
            bool mixedVariance =
                (info.contravariant > 0 && info.covariant > 0);
            if (!mixedVariance && !info.metricCoupling) {
              throw std::runtime_error(std::string("Implicit contraction of index '") +
                                       idx +
                                       "' requires explicit metric or inverse metric");
            }
          }
        }
      }

    }

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
