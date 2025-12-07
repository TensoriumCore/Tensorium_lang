#pragma once
#include <memory>
#include <string>
#include <vector>

enum class TensorKind {
  Scalar,
  Vector,
  Covector,
  CovTensor2,
  ConTensor2,
  MixedTensor
};

struct Expr {
  virtual ~Expr() = default;
};

struct NumberExpr : Expr {
  double value;
  explicit NumberExpr(double v) : value(v) {}
};

struct CallExpr : Expr {
  std::string callee;
  std::vector<std::unique_ptr<Expr>> args;
};

struct VarExpr : Expr {
  std::string name;
  explicit VarExpr(std::string n) : name(std::move(n)) {}
};

struct BinaryExpr : Expr {
  std::unique_ptr<Expr> lhs;
  std::unique_ptr<Expr> rhs;
  char op;
  BinaryExpr(std::unique_ptr<Expr> l, char o, std::unique_ptr<Expr> r)
      : lhs(std::move(l)), rhs(std::move(r)), op(o) {}
};

struct ParenExpr : Expr {
  std::unique_ptr<Expr> inner;
  explicit ParenExpr(std::unique_ptr<Expr> e) : inner(std::move(e)) {}
};

struct IndexedVarExpr : Expr {
  std::string base;
  std::vector<std::string> indices;

  IndexedVarExpr(std::string b, std::vector<std::string> idx)
      : base(std::move(b)), indices(std::move(idx)) {}
};

struct TensorAccess {
  std::string base;
  std::vector<std::string> indices;
};

struct Assignment {
  TensorAccess lhs;
  std::unique_ptr<Expr> rhs;
};

struct MetricDecl {
  std::string name;
  std::vector<std::string> indices;
  std::vector<Assignment> entries;
};

struct FieldDecl {
  TensorKind kind;
  std::string name;
  std::vector<std::string> indices;
  int up = 0;
  int down = 0;
};

struct EvolutionEq {
  std::string fieldName;
  std::vector<std::string> indices;
  std::unique_ptr<Expr> rhs;
};

struct EvolutionDecl {
  std::string name;
  std::vector<EvolutionEq> equations;
  std::vector<Assignment> tempAssignments;
};

struct Program {
  std::vector<FieldDecl> fields;
  std::vector<MetricDecl> metrics;
  std::vector<EvolutionDecl> evolutions;
};
