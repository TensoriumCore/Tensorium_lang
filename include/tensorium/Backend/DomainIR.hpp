
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

enum class VarKind { Field, Param, Local, Coord };

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

struct TensorProductIR final : ExprIR {
  std::unique_ptr<ExprIR> lhs;
  std::unique_ptr<ExprIR> rhs;
  TensorProductIR(std::unique_ptr<ExprIR> L, std::unique_ptr<ExprIR> R)
      : ExprIR(Kind::TensorProduct), lhs(std::move(L)), rhs(std::move(R)) {}
};

struct ContractionIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::vector<std::string> summedIndices;
  explicit ContractionIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Contraction), in(std::move(expr)) {}
};

struct IndexRenameIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string from;
  std::string to;
  IndexRenameIR(std::unique_ptr<ExprIR> expr, std::string fromIndex,
                std::string toIndex)
      : ExprIR(Kind::IndexRename), in(std::move(expr)),
        from(std::move(fromIndex)), to(std::move(toIndex)) {}
};

struct IndexPermuteIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::vector<std::string> order;
  IndexPermuteIR(std::unique_ptr<ExprIR> expr, std::vector<std::string> outOrder)
      : ExprIR(Kind::IndexPermute), in(std::move(expr)),
        order(std::move(outOrder)) {}
};

struct TraceIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::vector<std::string> tracedIndices;
  explicit TraceIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Trace), in(std::move(expr)) {}
};

struct PartialDerivativeIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string coordIndex;
  PartialDerivativeIR(std::unique_ptr<ExprIR> expr, std::string idx)
      : ExprIR(Kind::PartialDerivative), in(std::move(expr)),
        coordIndex(std::move(idx)) {}
};

struct GradientIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  explicit GradientIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Gradient), in(std::move(expr)) {}
};

struct CovariantDerivativeIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string derivIndex;
  bool contravariant = false;
  bool hasConnectionTensor = false;
  CovariantDerivativeIR(std::unique_ptr<ExprIR> expr, std::string idx)
      : ExprIR(Kind::CovariantDerivative), in(std::move(expr)),
        derivIndex(std::move(idx)) {}
};

struct DivergenceIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string contractedIndex;
  explicit DivergenceIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Divergence), in(std::move(expr)) {}
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
