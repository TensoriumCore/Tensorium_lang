#pragma once
#include "tensorium/AST/AST.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tensorium {

struct IndexedExpr {
  virtual ~IndexedExpr() = default;
  TensorTypeDesc inferredType{TensorKind::Scalar, 0, 0};
};

struct IndexedNumber : IndexedExpr {
  double value;
  explicit IndexedNumber(double v) : value(v) {}
};

enum class IndexedVarKind { Coordinate, Local, Field, Parameter };

struct IndexedVar : IndexedExpr {
  std::string name;
  IndexedVarKind kind;
  TensorKind tensorKind = TensorKind::Scalar;
  int up = 0;
  int down = 0;
  int coordIndex = -1;

  std::vector<int> tensorIndices;
  std::vector<std::string> tensorIndexNames;
  std::vector<bool> tensorIndexIsUp;

  IndexedVar(std::string n, IndexedVarKind k, int cidx = -1)
      : name(std::move(n)), kind(k), coordIndex(cidx) {}
};

struct IndexedBinary : IndexedExpr {
  char op;
  std::unique_ptr<IndexedExpr> lhs;
  std::unique_ptr<IndexedExpr> rhs;
  IndexedBinary(char o, std::unique_ptr<IndexedExpr> L,
                std::unique_ptr<IndexedExpr> R)
      : op(o), lhs(std::move(L)), rhs(std::move(R)) {}
};

struct IndexedCall : IndexedExpr {
  std::string callee;
  std::vector<std::unique_ptr<IndexedExpr>> args;
  bool isExtern = false;
  size_t declaredArity = 0;
  TensorTypeDesc returnType;
  std::vector<TensorTypeDesc> paramTypes;
};

struct IndexedAssignment {
  std::string tensor;
  std::vector<int> indexOffsets;
  std::unique_ptr<IndexedExpr> rhs;
};

struct IndexedMetric {
  std::string name;
  int rank;
  std::vector<std::string> coords;
  std::vector<IndexedAssignment> assignments;
};

struct IndexedEvolutionEq {
  std::string fieldName;
  std::vector<std::string> indices;
  std::unique_ptr<IndexedExpr> rhs;
};

struct IndexedEvolution {
  std::string name;
  std::vector<IndexedEvolutionEq> equations;
  std::vector<IndexedAssignment> temp;
};

} // namespace tensorium
