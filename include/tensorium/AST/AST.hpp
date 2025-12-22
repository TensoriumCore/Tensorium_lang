#pragma once
#include <memory>
#include <string>
#include <vector>

namespace tensorium {

enum class TensorKind {
  Scalar,
  Vector,
  Covector,
  CovTensor2,
  ConTensor2,
  MixedTensor
};

struct NumberExpr;
struct VarExpr;
struct BinaryExpr;
struct CallExpr;
struct ParenExpr;
struct IndexedVarExpr;

struct ExprVisitor {
  virtual ~ExprVisitor() = default;
  virtual void visit(const NumberExpr &) = 0;
  virtual void visit(const VarExpr &) = 0;
  virtual void visit(const BinaryExpr &) = 0;
  virtual void visit(const CallExpr &) = 0;
  virtual void visit(const ParenExpr &) = 0;
  virtual void visit(const IndexedVarExpr &) = 0;
};

struct Expr {
  virtual ~Expr() = default;
  virtual void accept(ExprVisitor &v) const = 0;
};

struct NumberExpr : Expr {
  double value;
  explicit NumberExpr(double v) : value(v) {}
  void accept(ExprVisitor &v) const override { v.visit(*this); }
};

struct VarExpr : Expr {
  std::string name;
  explicit VarExpr(std::string n) : name(std::move(n)) {}
  void accept(ExprVisitor &v) const override { v.visit(*this); }
};

struct BinaryExpr : Expr {
  std::unique_ptr<Expr> lhs, rhs;
  char op;
  BinaryExpr(std::unique_ptr<Expr> l, char o, std::unique_ptr<Expr> r)
      : lhs(std::move(l)), rhs(std::move(r)), op(o) {}
  void accept(ExprVisitor &v) const override { v.visit(*this); }
};

struct ParenExpr : Expr {
  std::unique_ptr<Expr> inner;
  explicit ParenExpr(std::unique_ptr<Expr> e) : inner(std::move(e)) {}
  void accept(ExprVisitor &v) const override { v.visit(*this); }
};

struct CallExpr : Expr {
  std::string callee;
  std::vector<std::unique_ptr<Expr>> args;
  void accept(ExprVisitor &v) const override { v.visit(*this); }
};

struct IndexedVarExpr : Expr {
  std::string base;
  std::vector<std::string> indices;
  IndexedVarExpr(std::string b, std::vector<std::string> idx)
      : base(std::move(b)), indices(std::move(idx)) {}
  void accept(ExprVisitor &v) const override { v.visit(*this); }
};

// Structures Top-Level
struct TensorAccess {
  std::string base;
  std::vector<std::string> indices;
};
struct Assignment {
  TensorAccess lhs;
  std::unique_ptr<Expr> rhs;
};

struct FieldDecl {
  TensorKind kind;
  std::string name;
  std::vector<std::string> indices;
  int up = 0;
  int down = 0;
};

struct MetricDecl {
  std::string name;
  std::vector<std::string> indices;
  std::vector<Assignment> entries;
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

enum class CoordinateSystem { Cartesian, Spherical, Cylindrical };

enum class TimeIntegrator { Euler, RK3, RK4 };

enum class SpatialScheme { FiniteDifference, Spectral };

enum class DerivativeScheme { Centered, Upwind };

struct TimeConfig {
  double dt = 0.0;
  TimeIntegrator integrator = TimeIntegrator::RK4;
};

struct SpatialConfig {
  SpatialScheme scheme = SpatialScheme::FiniteDifference;
  DerivativeScheme derivative = DerivativeScheme::Centered;
  int order = 2;
};

struct SimulationConfig {
  CoordinateSystem coordinates = CoordinateSystem::Cartesian;
  int dimension = 3;
  std::vector<int> resolution;
  TimeConfig time;
  SpatialConfig spatial;
};

struct Program {
  std::vector<FieldDecl> fields;
  std::vector<MetricDecl> metrics;
  std::vector<EvolutionDecl> evolutions;
  std::unique_ptr<SimulationConfig> simulation;
};
} // namespace tensorium
