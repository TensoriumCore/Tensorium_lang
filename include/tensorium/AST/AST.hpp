#pragma once
#include <cstddef>
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
  CovTensor3,
  ConTensor3,
  ConTensor4,
  CovTensor4,
  MixedTensor,
  Metric,
  InverseMetric
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
  std::vector<int> indexOffsets;
  IndexedVarExpr(std::string b, std::vector<std::string> idx,
                 std::vector<int> offs = {})
      : base(std::move(b)), indices(std::move(idx)),
        indexOffsets(std::move(offs)) {
    if (indexOffsets.size() != indices.size())
      indexOffsets.assign(indices.size(), 0);
  }
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

struct TensorTypeDesc {
  TensorKind kind = TensorKind::Scalar;
  int up = 0;
  int down = 0;
};

struct ExternDecl {
  std::string name;
  TensorTypeDesc returnType;
  std::vector<TensorTypeDesc> params;
  size_t paramCount = 0;
};

struct FieldDecl {
  TensorKind kind;
  std::string name;
  std::vector<std::string> indices;
  int up = 0;
  int down = 0;
  bool isMetric = false;
  bool isInverseMetric = false;
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

struct ConstraintEq {
  std::string fieldName;
  std::string unknownFieldName;
  std::vector<std::string> indices;
  std::unique_ptr<Expr> rhs;
};

struct ConstraintFieldRoleDecl {
  TensorTypeDesc type;
  std::string name;
  std::vector<std::string> indices;
};

struct BoundaryConditionDecl {
  std::string residualName;
  std::string face;
  std::string kind;
  double valueCoefficient = 1.0;
  double normalDerivativeCoefficient = 0.0;
  double targetValue = 0.0;
  std::string derivativeKind = "normal";
  std::string valueCoefficientCoordinate;
  std::string normalDerivativeCoefficientCoordinate;
  std::string targetValueCoordinate;
};

struct ConstraintDecl {
  std::string name;
  std::vector<ConstraintFieldRoleDecl> unknowns;
  std::vector<ConstraintFieldRoleDecl> freeFields;
  std::vector<ConstraintEq> residuals;
  std::vector<Assignment> tempAssignments;
  std::vector<BoundaryConditionDecl> boundaryConditions;
};

struct PrintDecl {
  std::unique_ptr<Expr> expr;
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

struct Metric4InitDecl {
  std::string name;
  std::vector<std::string> indices;
  std::vector<std::vector<std::unique_ptr<Expr>>> components;
};

struct DecomposedMetricInitDecl {
  std::unique_ptr<Expr> alpha;
  std::vector<std::unique_ptr<Expr>> beta;
  std::vector<std::vector<std::unique_ptr<Expr>>> gamma;
  std::vector<std::vector<std::unique_ptr<Expr>>> gammaU;
};

struct Split3P1BindingDecl {
  bool enabled = false;
  bool hasAlpha = false;
  bool hasBeta = false;
  bool hasGamma = false;
  bool hasGammaU = false;
  TensorAccess alphaTarget;
  TensorAccess betaTarget;
  TensorAccess gammaTarget;
  TensorAccess gammaUTarget;
};

struct InitialDataDecl {
  bool enforceSymmetry = false;
  bool hasMetric4 = false;
  bool hasDecomposed = false;
  Metric4InitDecl metric4;
  DecomposedMetricInitDecl decomposed;
  Split3P1BindingDecl split3p1;
};

struct Program {
  std::vector<std::string> params;
  std::vector<ExternDecl> externs;
  std::vector<FieldDecl> fields;
  std::vector<MetricDecl> metrics;
  std::vector<EvolutionDecl> evolutions;
  std::vector<ConstraintDecl> constraints;
  std::vector<PrintDecl> prints;
  std::unique_ptr<SimulationConfig> simulation;
  std::unique_ptr<InitialDataDecl> initialData;
};
} // namespace tensorium
