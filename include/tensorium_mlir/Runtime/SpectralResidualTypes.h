#pragma once

#include "tensorium_mlir/Runtime/SpectralGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef TENSORIUM_SPECTRAL_RESIDUAL_ABI_TYPES_H
#define TENSORIUM_SPECTRAL_RESIDUAL_ABI_TYPES_H

typedef struct tensorium_spectral_residual_point {
  std::int64_t i;
  std::int64_t j;
  std::int64_t k;
  std::int64_t index;
  double logical[3];
  double physical[3];
  double value;
  double d1;
  double d2;
  double d3;
  double d11;
  double d12;
  double d13;
  double d22;
  double d23;
  double d33;
  const double *aux_values;
  std::int64_t aux_count;
} tensorium_spectral_residual_point;

typedef double (*tensorium_spectral_residual_kernel_fn)(
    const tensorium_spectral_residual_point *point, const double *params,
    std::int64_t param_count, void *user_data);

typedef double (*tensorium_spectral_residual_jvp_kernel_fn)(
    const tensorium_spectral_residual_point *point,
    const tensorium_spectral_residual_point *direction, const double *params,
    std::int64_t param_count, void *user_data);

typedef int (*tensorium_spectral_residual_grid_kernel_fn)(
    std::int64_t n_points, const double *params, std::int64_t param_count,
    const double *value, const double *d1, const double *d2, const double *d3,
    const double *d11, const double *d12, const double *d13,
    const double *d22, const double *d23, const double *d33,
    const double *const *aux_fields, std::int64_t aux_count, const double *x1,
    const double *x2, const double *x3, double *out, void *user_data);

typedef void (*tensorium_spectral_coordinate_map_fn)(
    const double *logical, double *physical, const double *params,
    std::int64_t param_count, void *user_data);

typedef struct tensorium_spectral_residual_kernel_desc {
  const char *symbol_name;
  tensorium_spectral_residual_kernel_fn evaluate;
  const char *jvp_symbol_name;
  tensorium_spectral_residual_jvp_kernel_fn evaluate_jvp;
  void *user_data;
} tensorium_spectral_residual_kernel_desc;

typedef struct tensorium_spectral_residual_grid_kernel_desc {
  const char *symbol_name;
  tensorium_spectral_residual_grid_kernel_fn evaluate;
  void *user_data;
} tensorium_spectral_residual_grid_kernel_desc;

typedef struct tensorium_spectral_coordinate_map_desc {
  const char *symbol_name;
  tensorium_spectral_coordinate_map_fn map;
  void *user_data;
} tensorium_spectral_coordinate_map_desc;

typedef struct tensorium_spectral_residual_system_equation_desc {
  const char *residual_name;
  const char *unknown_name;
  std::int64_t unknown_index;
  std::int64_t point_kernel_index;
  std::int64_t grid_kernel_index;
  const char *const *param_names;
  std::int64_t param_count;
  const char *const *auxiliary_names;
  const std::int64_t *auxiliary_unknown_indices;
  std::int64_t auxiliary_count;
} tensorium_spectral_residual_system_equation_desc;

typedef struct tensorium_spectral_residual_system_desc {
  const char *symbol_name;
  const char *const *unknown_names;
  std::int64_t unknown_count;
  const tensorium_spectral_residual_system_equation_desc *equations;
  std::int64_t equation_count;
} tensorium_spectral_residual_system_desc;

typedef struct tensorium_spectral_initial_data_desc {
  std::int64_t abi_version;
  const char *symbol_name;
  const char *system_name;
  const char *coordinate_map;
  const std::int64_t *resolution;
  const char *const *basis;
  std::int64_t dimension;
  const char *const *coordinate_parameter_names;
  std::int64_t coordinate_parameter_count;
  const char *unknown_map;
  const double *unknown_map_parameters;
  std::int64_t unknown_map_parameter_count;
  const char *field_projector;
  const char *reconstruction;
  const char *const *parameter_names;
  const double *parameter_values;
  std::int64_t parameter_count;
  const char *nonlinear_solver;
  const char *linear_solver;
  double residual_tolerance;
  std::int64_t max_newton_steps;
  double linear_tolerance;
  double linear_relative_tolerance;
  std::int64_t max_linear_iterations;
  std::int64_t restart;
  const char *preconditioner;
  std::int64_t preconditioner_sweeps;
  double jvp_relative_step;
  double jvp_absolute_step;
} tensorium_spectral_initial_data_desc;

#endif /* TENSORIUM_SPECTRAL_RESIDUAL_ABI_TYPES_H */

namespace tensorium_mlir::runtime {

using SpectralAuxiliaryUnknownIndex = std::int64_t;

inline constexpr SpectralAuxiliaryUnknownIndex kSpectralStaticAuxiliary = -1;

struct SpectralResidualKernel {
  std::string symbolName;
  tensorium_spectral_residual_kernel_fn evaluate = nullptr;
  std::string jvpSymbolName;
  tensorium_spectral_residual_jvp_kernel_fn evaluateJvp = nullptr;
  void *userData = nullptr;

  SpectralResidualKernel() = default;
  SpectralResidualKernel(std::string symbolName,
                         tensorium_spectral_residual_kernel_fn evaluate,
                         void *userData = nullptr)
      : symbolName(std::move(symbolName)), evaluate(evaluate),
        userData(userData) {}
  SpectralResidualKernel(
      std::string symbolName,
      tensorium_spectral_residual_kernel_fn evaluate,
      std::string jvpSymbolName,
      tensorium_spectral_residual_jvp_kernel_fn evaluateJvp,
      void *userData = nullptr)
      : symbolName(std::move(symbolName)), evaluate(evaluate),
        jvpSymbolName(std::move(jvpSymbolName)), evaluateJvp(evaluateJvp),
        userData(userData) {}
};

struct SpectralResidualGridKernel {
  std::string symbolName;
  tensorium_spectral_residual_grid_kernel_fn evaluate = nullptr;
  void *userData = nullptr;
};

struct SpectralCoordinateMap {
  std::string symbolName = "tensorium_spectral_identity_map";
  tensorium_spectral_coordinate_map_fn map = nullptr;
  void *userData = nullptr;
};

using SpectralDerivativeMapFn = void (*)(
    const double logical[3], const SpectralPointDerivatives3D *logicalDerivatives,
    SpectralPointDerivatives3D *physicalDerivatives, const double *params,
    std::int64_t paramCount, void *userData);

struct SpectralDerivativeMap {
  std::string symbolName = "tensorium_spectral_identity_derivative_map";
  SpectralDerivativeMapFn transform = nullptr;
  void *userData = nullptr;
};

using SpectralUnknownMapFn = SpectralDerivativeMapFn;

struct SpectralUnknownMap {
  std::string symbolName = "tensorium_spectral_identity_unknown_map";
  SpectralUnknownMapFn transform = nullptr;
  void *userData = nullptr;
};

using SpectralFieldProjectorFn = void (*)(const SpectralGrid3D *grid,
                                          double *values,
                                          std::int64_t valueCount,
                                          void *userData);

struct SpectralFieldProjector {
  std::string symbolName = "tensorium_spectral_identity_field_projector";
  SpectralFieldProjectorFn project = nullptr;
  void *userData = nullptr;
};

struct SpectralResidualProblem {
  const SpectralGrid3D *grid = nullptr;
  SpectralResidualKernel kernel;
  std::span<const double> params{};
  std::span<const std::vector<double>> auxiliaryFields{};
  SpectralCoordinateMap coordinateMap{};
  std::span<const double> coordinateParams{};
  SpectralResidualGridKernel gridKernel{};
  SpectralDerivativeMap derivativeMap{};
  SpectralUnknownMap unknownMap{};
  std::span<const double> unknownMapParams{};
  SpectralFieldProjector fieldProjector{};
};

struct SpectralResidualAssemblyResult {
  std::vector<double> values;
  double l2Norm = 0.0;
  double maxAbs = 0.0;
  bool finite = true;
  bool usedGeneratedGridKernel = false;

  std::size_t size() const { return values.size(); }
};

struct SpectralResidualSystemEquation {
  SpectralResidualProblem problem;
  std::size_t unknownIndex = 0;
  std::string residualName;
  std::span<const SpectralAuxiliaryUnknownIndex> auxiliaryUnknownIndices{};
};

struct SpectralResidualSystemProblem {
  const SpectralGrid3D *grid = nullptr;
  std::span<const SpectralResidualSystemEquation> equations{};
};

struct SpectralResidualSystemAssemblyResult {
  std::vector<double> values;
  std::vector<SpectralResidualAssemblyResult> equationResults;
  std::size_t equationCount = 0;
  std::size_t pointsPerEquation = 0;
  double l2Norm = 0.0;
  double maxAbs = 0.0;
  bool finite = true;
  bool usedGeneratedGridKernels = false;

  std::size_t size() const { return values.size(); }
};

struct SpectralResidualSystemJacobianVectorProductResult {
  std::vector<double> values;
  double step = 0.0;
  double l2Norm = 0.0;
  double maxAbs = 0.0;
  bool finite = true;
  bool usedGeneratedGridKernels = false;
  bool usedGeneratedJvpKernels = false;

  std::size_t size() const { return values.size(); }
};

struct SpectralGeneratedResidualSystemEquationInputs {
  std::span<const double> params{};
  std::span<const std::vector<double>> auxiliaryFields{};
};

struct SpectralGeneratedResidualSystem {
  const SpectralGrid3D *grid = nullptr;
  std::string symbolName;
  std::vector<SpectralResidualSystemEquation> equations;

  SpectralResidualSystemProblem view() const {
    return SpectralResidualSystemProblem{
        grid, std::span<const SpectralResidualSystemEquation>(equations.data(),
                                                              equations.size())};
  }
};

struct SpectralJacobianVectorProductOptions {
  double relativeStep = 1.4901161193847656e-8;
  double absoluteStep = 0.0;
  bool centeredDifference = true;
};

struct SpectralJacobianVectorProductResult {
  std::vector<double> values;
  double step = 0.0;
  double l2Norm = 0.0;
  double maxAbs = 0.0;
  bool finite = true;
  bool usedGeneratedJvpKernel = false;

  std::size_t size() const { return values.size(); }
};

enum class SpectralEllipticSolveStatus {
  MaxSteps,
  Converged,
  InvalidResidual,
  LinearSolveFailed,
  LineSearchFailed,
  InvalidInput,
};

enum class SpectralLinearSolveKind {
  Auto,
  DenseJacobian,
  MatrixFreeGMRES,
};

enum class SpectralPreconditionerKind {
  None,
  DiagonalJVP,
  DenseLaplacianShift,
  ModalLaplacianShift,
  MappedFiniteDifferenceLaplacianShift,
  MappedFiniteDifferenceMultigrid,
};

struct SpectralEllipticSolveOptions {
  int maxNewtonSteps = 8;
  double residualTolerance = 0.0;
  double residualRatioTarget = 0.0;
  double initialDamping = 1.0;
  double lineSearchReduction = 0.5;
  double minDamping = 1.0e-6;
  int maxLineSearchSteps = 16;
  double linearPivotTolerance = 1.0e-12;
  std::size_t denseJacobianMaxUnknowns = 2048;
  SpectralLinearSolveKind linearSolver = SpectralLinearSolveKind::Auto;
  int gmresMaxIterations = 64;
  int gmresRestart = 0;
  double gmresTolerance = 1.0e-10;
  double gmresRelativeTolerance = 1.0e-10;
  SpectralPreconditionerKind gmresPreconditioner =
      SpectralPreconditionerKind::None;
  double preconditionerPivotTolerance = 1.0e-12;
  double preconditionerLaplacianShift = 0.0;
  std::vector<double> preconditionerLaplacianShifts{};
  int preconditionerRelaxationSweeps = 4;
  double preconditionerRelaxationOmega = 1.0;
  int preconditionerMultigridPreSweeps = 3;
  int preconditionerMultigridPostSweeps = 3;
  double preconditionerMultigridRelaxationOmega = 1.0;
  bool preconditionerMultigridUseLocalReaction = true;
  SpectralJacobianVectorProductOptions jvpOptions{};
};

struct SpectralEllipticSolveResult {
  SpectralEllipticSolveStatus status = SpectralEllipticSolveStatus::MaxSteps;
  int steps = 0;
  int maxSteps = 0;
  std::size_t unknowns = 0;
  double initialResidualL2 = 0.0;
  double finalResidualL2 = 0.0;
  double finalResidualMaxAbs = 0.0;
  double residualRatio = std::numeric_limits<double>::infinity();
  double lastDamping = 0.0;
  double finalLinearResidualL2 = std::numeric_limits<double>::infinity();
  int linearIterations = 0;
  bool usedGeneratedGridKernel = false;
  bool usedMatrixFreeGMRES = false;
  bool usedPreconditioner = false;
  bool usedFieldProjector = false;

  bool converged() const {
    return status == SpectralEllipticSolveStatus::Converged;
  }
  bool residualIsFinite() const {
    return std::isfinite(initialResidualL2) && std::isfinite(finalResidualL2);
  }
};

} // namespace tensorium_mlir::runtime
