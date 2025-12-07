#pragma once
#include <memory>
#include <string>
#include <vector>

enum class IndexedVarKind {
  Coordinate,
  Local,
  Parameter,
  Field,
};

struct IndexedExpr {
  virtual ~IndexedExpr() = default;
};

struct IndexedNumber : IndexedExpr {
  double value;
  explicit IndexedNumber(double v) : value(v) {}
};

struct IndexedVar : IndexedExpr {
  IndexedVarKind kind;
  std::string name;
  int coordIndex; 
  std::vector<int> tensorIndices;  
  std::vector<std::string> tensorIndexNames; 

  IndexedVar(std::string n, IndexedVarKind k, int idx = -1)
      : kind(k), name(std::move(n)), coordIndex(idx) {}
};

struct IndexedBinary : IndexedExpr {
  char op;
  std::unique_ptr<IndexedExpr> lhs, rhs;
  IndexedBinary(char o, std::unique_ptr<IndexedExpr> l,
                std::unique_ptr<IndexedExpr> r)
      : op(o), lhs(std::move(l)), rhs(std::move(r)) {}
};

struct IndexedCall : IndexedExpr {
  std::string callee;
  std::vector<std::unique_ptr<IndexedExpr>> args;
};

struct IndexedAssignment {
  std::string tensor;
  std::vector<int> indexOffsets; 
  std::unique_ptr<IndexedExpr> rhs;
};

struct IndexedMetric {
  std::string name;
  int rank = 2;
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
};
