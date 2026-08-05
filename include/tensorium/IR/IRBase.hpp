#pragma once

#include "tensorium/IR/TensorType.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorium::backend {

enum class CoordSystem { Cartesian, Spherical, Cylindrical };
enum class TimeIntegrator { Euler, RK3, RK4 };
enum class SpatialScheme { FD, Spectral };
enum class DerivativeScheme { Centered, Upwind };

struct TimeIR {
  double dt = 0.0;
  TimeIntegrator integrator = TimeIntegrator::Euler;
};

struct SpatialIR {
  SpatialScheme scheme = SpatialScheme::FD;
  DerivativeScheme derivative = DerivativeScheme::Centered;
  int order = 2;
};

struct SimulationIR {
  CoordSystem coords = CoordSystem::Cartesian;
  int dimension = 0;
  std::vector<int> resolution;
  TimeIR time;
  SpatialIR spatial;
};

enum class FieldKind {
  Scalar,
  Vector,
  Covector,
  CovTensor2,
  ConTensor2,
  CovTensor3,
  ConTensor3,
  CovTensor4,
  ConTensor4,
  MixedTensor
};

struct FieldIR {
  std::string name;
  FieldKind kind = FieldKind::Scalar;
  tensorium::ir::TensorType tensorType;
};

enum class VarKind { Field, Param, Local, Coord, Unknown };

struct ExprIR {
  enum class Kind {
    Number,
    Var,
    Binary,
    Call,
    TensorProduct,
    Contraction,
    IndexRename,
    IndexPermute,
    Trace,
    PartialDerivative,
    Gradient,
    CovariantDerivative,
    Divergence
  };

  Kind kind;
  tensorium::ir::TensorType exprType;

  virtual ~ExprIR() = default;
  explicit ExprIR(Kind k) : kind(k) {}
};

struct NumberIR final : ExprIR {
  double value;
  explicit NumberIR(double v) : ExprIR(Kind::Number), value(v) {}
};

struct VarIR final : ExprIR {
  std::string name;
  VarKind vkind = VarKind::Field;
  int coordIndex = -1;
  std::vector<std::string> tensorIndexNames;
  VarIR(std::string n, VarKind k)
      : ExprIR(Kind::Var), name(std::move(n)), vkind(k) {}
};

struct BinaryIR final : ExprIR {
  std::string op;
  std::unique_ptr<ExprIR> lhs;
  std::unique_ptr<ExprIR> rhs;
  BinaryIR(std::string o, std::unique_ptr<ExprIR> L, std::unique_ptr<ExprIR> R)
      : ExprIR(Kind::Binary), op(std::move(o)), lhs(std::move(L)),
        rhs(std::move(R)) {}
};

struct CallIR final : ExprIR {
  std::string callee;
  std::vector<std::unique_ptr<ExprIR>> args;
  bool isExtern = false;
  size_t externArity = 0;
  tensorium::ir::TensorType returnType;
  std::vector<tensorium::ir::TensorType> paramTypes;
  explicit CallIR(std::string c) : ExprIR(Kind::Call), callee(std::move(c)) {}
};

struct EquationIR {
  std::string fieldName;
  std::vector<std::string> indices;
  std::unique_ptr<ExprIR> rhs;
};

struct TempAssignIR {
  std::string name;
  std::vector<std::string> indices;
  std::vector<int> indexOffsets;
  std::unique_ptr<ExprIR> rhs;
};

struct EvolutionIR {
  std::string name;
  std::vector<EquationIR> equations;
  std::vector<TempAssignIR> temporaries;
};

struct PrintIR {
  std::string label;
  std::string fieldName;
  std::vector<std::string> indices;
  tensorium::ir::TensorType tensorType;
};

struct InitExprIR {
  enum class Kind { Number, Symbol, Binary, Call };
  Kind kind;
  virtual ~InitExprIR() = default;
  explicit InitExprIR(Kind k) : kind(k) {}
};

struct InitNumberIR final : InitExprIR {
  double value;
  explicit InitNumberIR(double v) : InitExprIR(Kind::Number), value(v) {}
};

struct InitSymbolIR final : InitExprIR {
  std::string name;
  explicit InitSymbolIR(std::string n)
      : InitExprIR(Kind::Symbol), name(std::move(n)) {}
};

struct InitBinaryIR final : InitExprIR {
  char op;
  std::unique_ptr<InitExprIR> lhs;
  std::unique_ptr<InitExprIR> rhs;
  InitBinaryIR(char o, std::unique_ptr<InitExprIR> L,
               std::unique_ptr<InitExprIR> R)
      : InitExprIR(Kind::Binary), op(o), lhs(std::move(L)),
        rhs(std::move(R)) {}
};

struct InitCallIR final : InitExprIR {
  std::string callee;
  std::vector<std::unique_ptr<InitExprIR>> args;
  explicit InitCallIR(std::string c)
      : InitExprIR(Kind::Call), callee(std::move(c)) {}
};

struct Metric4InitIR {
  std::string name;
  std::vector<std::string> indices;
  std::vector<std::unique_ptr<InitExprIR>> components;
  bool enforceSymmetry = false;
  std::string coordSystem;
};

struct DecomposedInitIR {
  std::unique_ptr<InitExprIR> alphaExpr;
  std::vector<std::unique_ptr<InitExprIR>> betaExpr;
  std::vector<std::unique_ptr<InitExprIR>> gammaExpr;
  std::vector<std::unique_ptr<InitExprIR>> gammaUExpr;
};

struct Split3P1BindingIR {
  bool enabled = false;
  bool hasAlpha = false;
  bool hasBeta = false;
  bool hasGamma = false;
  bool hasGammaU = false;
  std::string alphaField;
  std::string betaField;
  std::string gammaField;
  std::string gammaUField;
};

struct InitialDataIR {
  bool hasMetric4 = false;
  bool hasDecomposed = false;
  Metric4InitIR metric4;
  DecomposedInitIR decomposed;
  Split3P1BindingIR split3p1;
};

struct SpectralDomainIR {
  std::string name;
  std::string coordinates;
  std::string topology;
  std::vector<int> resolution;
  std::string basis;
  std::vector<double> bounds;
};

struct ConstraintUnknownIR {
  std::string name;
  tensorium::ir::TensorType tensorType;
  std::vector<std::string> indices;
  bool symmetric = false;
};

struct ConstraintEquationIR {
  std::string name;
  tensorium::ir::TensorType tensorType;
  std::vector<std::string> indices;
  std::unique_ptr<ExprIR> residual;
};

struct ConstraintAssignmentIR {
  std::string unknown;
  std::vector<std::string> indices;
  std::unique_ptr<ExprIR> rhs;
};

struct ConstraintBoundaryIR {
  std::string region;
  std::vector<ConstraintAssignmentIR> conditions;
};

struct ConstraintInterfaceIR {
  std::string innerDomain;
  std::string outerDomain;
};

struct ConstraintSolveIR {
  std::string nonlinear;
  std::string linear;
  double tolerance = 1.0e-10;
  int maxIterations = 30;
  double linearTolerance = 1.0e-10;
  double linearRelativeTolerance = 1.0e-10;
  int maxLinearIterations = 64;
  int restart = 0;
  std::string preconditioner = "none";
  int preconditionerSweeps = 4;
  double jvpRelativeStep = 1.4901161193847656e-8;
  double jvpAbsoluteStep = 0.0;
};

struct SpectralParameterBindingIR {
  std::string name;
  double value = 0.0;
};

struct SpectralInitialDataIR {
  std::string name;
  std::string system;
  std::string coordinateMap;
  std::vector<int> resolution;
  std::vector<std::string> basis;
  std::vector<std::string> coordinateParameters;
  std::string unknownMap;
  std::vector<double> unknownMapParameters;
  std::string fieldProjector;
  std::string reconstruction;
  std::vector<SpectralParameterBindingIR> parameters;
  ConstraintSolveIR solve;
};

struct ConstraintCttReconstructionIR {
  bool enabled = false;
  std::string conformalFactor;
  std::string radialVectorPotential;
  std::string conformalElectricRadial;
  std::unique_ptr<ExprIR> meanCurvature;
};

struct ConstraintGeometryIR {
  bool enabled = false;
  std::string kind;
  std::string metricName;
  std::string inverseMetricName;
  std::unique_ptr<ExprIR> radialScale;
  std::unique_ptr<ExprIR> tangentialScale;
};

struct ConstraintProblemIR {
  std::string name;
  ConstraintGeometryIR geometry;
  std::vector<SpectralDomainIR> domains;
  std::vector<ConstraintUnknownIR> unknowns;
  std::vector<ConstraintEquationIR> equations;
  std::vector<ConstraintBoundaryIR> boundaries;
  std::vector<ConstraintInterfaceIR> interfaces;
  std::vector<ConstraintAssignmentIR> seeds;
  ConstraintCttReconstructionIR cttReconstruction;
  ConstraintSolveIR solve;
};

struct ModuleIR {
  std::optional<SimulationIR> simulation;
  std::optional<InitialDataIR> initialData;
  std::optional<ConstraintProblemIR> constraintProblem;
  std::optional<SpectralInitialDataIR> spectralInitialData;
  bool hasResidualConstraints = false;
  std::vector<FieldIR> fields;
  std::vector<EvolutionIR> evolutions;
  std::vector<PrintIR> prints;
};

} // namespace tensorium::backend
