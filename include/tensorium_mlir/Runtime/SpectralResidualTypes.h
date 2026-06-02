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

#endif /* TENSORIUM_SPECTRAL_RESIDUAL_ABI_TYPES_H */

namespace tensorium_mlir::runtime {

using SpectralAuxiliaryUnknownIndex = std::int64_t;

inline constexpr SpectralAuxiliaryUnknownIndex kSpectralStaticAuxiliary = -1;

struct SpectralResidualKernel {
  std::string symbolName;
  tensorium_spectral_residual_kernel_fn evaluate = nullptr;
  void *userData = nullptr;
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

struct SpectralResidualProblem {
  const SpectralGrid3D *grid = nullptr;
  SpectralResidualKernel kernel;
  std::span<const double> params{};
  std::span<const std::vector<double>> auxiliaryFields{};
  SpectralCoordinateMap coordinateMap{};
  std::span<const double> coordinateParams{};
  SpectralResidualGridKernel gridKernel{};
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
  double gmresTolerance = 1.0e-10;
  double gmresRelativeTolerance = 1.0e-10;
  SpectralPreconditionerKind gmresPreconditioner =
      SpectralPreconditionerKind::None;
  double preconditionerPivotTolerance = 1.0e-12;
  double preconditionerLaplacianShift = 0.0;
  std::vector<double> preconditionerLaplacianShifts{};
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

  bool converged() const {
    return status == SpectralEllipticSolveStatus::Converged;
  }
  bool residualIsFinite() const {
    return std::isfinite(initialResidualL2) && std::isfinite(finalResidualL2);
  }
};

} // namespace tensorium_mlir::runtime
