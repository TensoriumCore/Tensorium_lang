
#pragma once
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
  int up = 0;
  int down = 0;
};

enum class VarKind { Field, Param, Local, Coord };

struct ExprIR {
  enum class Kind { Number, Var, Binary, Call };
  Kind kind;

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
  explicit CallIR(std::string c) : ExprIR(Kind::Call), callee(std::move(c)) {}
};

struct EquationIR {
  std::string fieldName;
  std::vector<std::string> indices;
  std::unique_ptr<ExprIR> rhs;
};

struct TempAssignIR {
  std::string name;
  std::vector<int> indexOffsets;
  std::unique_ptr<ExprIR> rhs;
};

struct EvolutionIR {
  std::string name;
  std::vector<EquationIR> equations;
  std::vector<TempAssignIR> temporaries;
};

struct ModuleIR {
  std::optional<SimulationIR> simulation;
  std::vector<FieldIR> fields;
  std::vector<EvolutionIR> evolutions;
};

} // namespace tensorium::backend
