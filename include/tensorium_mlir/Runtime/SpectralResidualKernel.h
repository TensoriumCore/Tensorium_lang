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

#endif /* TENSORIUM_SPECTRAL_RESIDUAL_ABI_TYPES_H */

namespace tensorium_mlir::runtime {

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

  bool converged() const {
    return status == SpectralEllipticSolveStatus::Converged;
  }
  bool residualIsFinite() const {
    return std::isfinite(initialResidualL2) && std::isfinite(finalResidualL2);
  }
};

inline SpectralResidualKernel
spectralResidualKernelFromDesc(const tensorium_spectral_residual_kernel_desc &desc) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("spectral residual kernel symbol is empty");
  if (!desc.evaluate)
    throw std::runtime_error("spectral residual kernel callback is null");
  return SpectralResidualKernel{desc.symbol_name, desc.evaluate,
                                desc.user_data};
}

inline SpectralResidualGridKernel spectralResidualGridKernelFromDesc(
    const tensorium_spectral_residual_grid_kernel_desc &desc) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("spectral residual grid kernel symbol is empty");
  if (!desc.evaluate)
    throw std::runtime_error("spectral residual grid kernel callback is null");
  return SpectralResidualGridKernel{desc.symbol_name, desc.evaluate,
                                    desc.user_data};
}

inline SpectralCoordinateMap
spectralCoordinateMapFromDesc(const tensorium_spectral_coordinate_map_desc &desc) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("spectral coordinate map symbol is empty");
  if (!desc.map)
    throw std::runtime_error("spectral coordinate map callback is null");
  return SpectralCoordinateMap{desc.symbol_name, desc.map, desc.user_data};
}

inline void validateSpectralDerivativeBundle(const SpectralGrid3D &grid,
                                             const SpectralDerivatives3D &derivs) {
  const std::size_t size = grid.size();
  if (derivs.value.size() != size || derivs.d1.size() != size ||
      derivs.d2.size() != size || derivs.d3.size() != size ||
      derivs.d11.size() != size || derivs.d12.size() != size ||
      derivs.d13.size() != size || derivs.d22.size() != size ||
      derivs.d23.size() != size || derivs.d33.size() != size) {
    throw std::runtime_error("spectral derivative bundle size mismatch");
  }
}

inline double spectralVectorMaxAbs(std::span<const double> values) {
  double out = 0.0;
  for (double value : values) {
    if (!std::isfinite(value))
      return value;
    out = std::max(out, std::fabs(value));
  }
  return out;
}

inline double spectralVectorL2Norm(std::span<const double> values) {
  if (values.empty())
    return 0.0;
  double sum = 0.0;
  for (double value : values) {
    if (!std::isfinite(value))
      return value;
    sum += value * value;
  }
  return std::sqrt(sum / static_cast<double>(values.size()));
}

inline bool spectralVectorIsFinite(std::span<const double> values) {
  for (double value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

inline SpectralResidualAssemblyResult
makeSpectralResidualAssemblyResult(std::vector<double> values,
                                   bool usedGeneratedGridKernel = false) {
  SpectralResidualAssemblyResult result;
  result.values = std::move(values);
  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = spectralVectorIsFinite(result.values);
  result.usedGeneratedGridKernel = usedGeneratedGridKernel;
  return result;
}

inline SpectralJacobianVectorProductResult
makeSpectralJacobianVectorProductResult(std::vector<double> values,
                                        double step) {
  SpectralJacobianVectorProductResult result;
  result.values = std::move(values);
  result.step = step;
  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = spectralVectorIsFinite(result.values);
  return result;
}

inline const SpectralGrid3D &
requireSpectralResidualGrid(const SpectralResidualProblem &problem) {
  if (!problem.grid)
    throw std::runtime_error("spectral residual problem grid is null");
  return *problem.grid;
}

inline tensorium_spectral_residual_point makeSpectralResidualPoint(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    std::size_t i, std::size_t j, std::size_t k,
    const SpectralCoordinateMap &coordinateMap,
    std::span<const double> coordinateParams,
    std::span<const double> auxiliaryValues = {}) {
  const SpectralPoint3D point = grid.point(i, j, k);
  const SpectralPointDerivatives3D u =
      grid.pointDerivatives(derivs, point.index);

  tensorium_spectral_residual_point out{};
  out.i = static_cast<std::int64_t>(point.i);
  out.j = static_cast<std::int64_t>(point.j);
  out.k = static_cast<std::int64_t>(point.k);
  out.index = static_cast<std::int64_t>(point.index);
  out.logical[0] = point.x1;
  out.logical[1] = point.x2;
  out.logical[2] = point.x3;
  out.physical[0] = point.x1;
  out.physical[1] = point.x2;
  out.physical[2] = point.x3;
  if (coordinateMap.map) {
    coordinateMap.map(out.logical, out.physical, coordinateParams.data(),
                      static_cast<std::int64_t>(coordinateParams.size()),
                      coordinateMap.userData);
  }
  out.value = u.value;
  out.d1 = u.d1;
  out.d2 = u.d2;
  out.d3 = u.d3;
  out.d11 = u.d11;
  out.d12 = u.d12;
  out.d13 = u.d13;
  out.d22 = u.d22;
  out.d23 = u.d23;
  out.d33 = u.d33;
  out.aux_values = auxiliaryValues.data();
  out.aux_count = static_cast<std::int64_t>(auxiliaryValues.size());
  return out;
}

inline std::vector<double> evaluateSpectralResidualWithAuxFields(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  validateSpectralDerivativeBundle(grid, derivs);
  if (!kernel.evaluate)
    throw std::runtime_error("spectral residual kernel callback is null");
  for (const auto &field : auxiliaryFields) {
    if (field.size() != grid.size())
      throw std::runtime_error("spectral auxiliary field size mismatch");
  }

  std::vector<double> out(grid.size(), 0.0);
  std::vector<double> pointAux(auxiliaryFields.size(), 0.0);
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const std::size_t pointIndex = grid.index(i, j, k);
        for (std::size_t aux = 0; aux < auxiliaryFields.size(); ++aux)
          pointAux[aux] = auxiliaryFields[aux][pointIndex];
        const auto point = makeSpectralResidualPoint(
            grid, derivs, i, j, k, coordinateMap, coordinateParams, pointAux);
        out[static_cast<std::size_t>(point.index)] =
            kernel.evaluate(&point, params.data(),
                            static_cast<std::int64_t>(params.size()),
                            kernel.userData);
      }
    }
  }
  return out;
}

inline std::vector<double> evaluateSpectralResidual(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  return evaluateSpectralResidualWithAuxFields(grid, derivs, kernel, params, {},
                                               coordinateMap, coordinateParams);
}

inline std::vector<double> evaluateSpectralResidual(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  return evaluateSpectralResidual(grid, grid.derivatives(values), kernel,
                                  params, coordinateMap, coordinateParams);
}

inline std::vector<double> evaluateSpectralResidualWithAuxFields(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  return evaluateSpectralResidualWithAuxFields(
      grid, grid.derivatives(values), kernel, params, auxiliaryFields,
      coordinateMap, coordinateParams);
}

inline std::array<std::vector<double>, 3> makeSpectralPhysicalCoordinateBuffers(
    const SpectralGrid3D &grid, const SpectralCoordinateMap &coordinateMap,
    std::span<const double> coordinateParams) {
  std::array<std::vector<double>, 3> coords = {
      std::vector<double>(grid.size(), 0.0),
      std::vector<double>(grid.size(), 0.0),
      std::vector<double>(grid.size(), 0.0)};

  double logical[3] = {0.0, 0.0, 0.0};
  double physical[3] = {0.0, 0.0, 0.0};
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const SpectralPoint3D point = grid.point(i, j, k);
        logical[0] = point.x1;
        logical[1] = point.x2;
        logical[2] = point.x3;
        physical[0] = point.x1;
        physical[1] = point.x2;
        physical[2] = point.x3;
        if (coordinateMap.map) {
          coordinateMap.map(logical, physical, coordinateParams.data(),
                            static_cast<std::int64_t>(
                                coordinateParams.size()),
                            coordinateMap.userData);
        }
        coords[0][point.index] = physical[0];
        coords[1][point.index] = physical[1];
        coords[2][point.index] = physical[2];
      }
    }
  }
  return coords;
}

inline std::vector<double> evaluateSpectralResidualWithGridKernel(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    const SpectralResidualGridKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  validateSpectralDerivativeBundle(grid, derivs);
  if (!kernel.evaluate)
    throw std::runtime_error("spectral residual grid kernel callback is null");
  for (const auto &field : auxiliaryFields) {
    if (field.size() != grid.size())
      throw std::runtime_error("spectral auxiliary field size mismatch");
  }

  std::vector<const double *> auxiliaryPointers;
  auxiliaryPointers.reserve(auxiliaryFields.size());
  for (const auto &field : auxiliaryFields)
    auxiliaryPointers.push_back(field.data());

  const auto coords =
      makeSpectralPhysicalCoordinateBuffers(grid, coordinateMap,
                                            coordinateParams);
  std::vector<double> out(grid.size(), 0.0);
  const int status = kernel.evaluate(
      static_cast<std::int64_t>(grid.size()), params.data(),
      static_cast<std::int64_t>(params.size()), derivs.value.data(),
      derivs.d1.data(), derivs.d2.data(), derivs.d3.data(), derivs.d11.data(),
      derivs.d12.data(), derivs.d13.data(), derivs.d22.data(),
      derivs.d23.data(), derivs.d33.data(),
      auxiliaryPointers.empty() ? nullptr : auxiliaryPointers.data(),
      static_cast<std::int64_t>(auxiliaryPointers.size()), coords[0].data(),
      coords[1].data(), coords[2].data(), out.data(), kernel.userData);
  if (status != 0)
    throw std::runtime_error("spectral residual grid kernel failed: " +
                             std::to_string(status));
  return out;
}

inline SpectralResidualAssemblyResult assembleSpectralResidual(
    const SpectralResidualProblem &problem,
    const SpectralDerivatives3D &derivs) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  if (problem.gridKernel.evaluate) {
    return makeSpectralResidualAssemblyResult(
        evaluateSpectralResidualWithGridKernel(
            grid, derivs, problem.gridKernel, problem.params,
            problem.auxiliaryFields, problem.coordinateMap,
            problem.coordinateParams),
        true);
  }
  return makeSpectralResidualAssemblyResult(evaluateSpectralResidualWithAuxFields(
      grid, derivs, problem.kernel, problem.params, problem.auxiliaryFields,
      problem.coordinateMap, problem.coordinateParams));
}

inline SpectralResidualAssemblyResult assembleSpectralResidual(
    const SpectralResidualProblem &problem, const std::vector<double> &values) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  if (values.size() != grid.size())
    throw std::runtime_error("spectral residual state size mismatch");
  return assembleSpectralResidual(problem, grid.derivatives(values));
}

inline SpectralResidualAssemblyResult assembleSpectralResidual(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields = {},
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  const SpectralResidualProblem problem{&grid, kernel, params, auxiliaryFields,
                                        coordinateMap, coordinateParams};
  return assembleSpectralResidual(problem, values);
}

inline const SpectralGrid3D &requireSpectralResidualSystemGrid(
    const SpectralResidualSystemProblem &system) {
  if (!system.grid)
    throw std::runtime_error("spectral residual system grid is null");
  return *system.grid;
}

inline SpectralResidualSystemAssemblyResult assembleSpectralResidualSystem(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> unknownFields) {
  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  if (system.equations.empty())
    throw std::runtime_error("spectral residual system has no equations");
  for (const auto &field : unknownFields) {
    if (field.size() != grid.size())
      throw std::runtime_error("spectral residual system unknown size mismatch");
  }

  SpectralResidualSystemAssemblyResult result;
  result.equationCount = system.equations.size();
  result.pointsPerEquation = grid.size();
  result.values.reserve(result.equationCount * result.pointsPerEquation);
  result.equationResults.reserve(result.equationCount);
  result.usedGeneratedGridKernels = true;

  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= unknownFields.size())
      throw std::runtime_error(
          "spectral residual system equation unknown index out of range");
    SpectralResidualProblem problem = equation.problem;
    if (!problem.grid)
      problem.grid = &grid;
    if (problem.grid != &grid)
      throw std::runtime_error("spectral residual system grid mismatch");

    const auto residual =
        assembleSpectralResidual(problem, unknownFields[equation.unknownIndex]);
    result.usedGeneratedGridKernels =
        result.usedGeneratedGridKernels && residual.usedGeneratedGridKernel;
    result.finite = result.finite && residual.finite;
    result.values.insert(result.values.end(), residual.values.begin(),
                         residual.values.end());
    result.equationResults.push_back(std::move(residual));
  }

  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = result.finite && spectralVectorIsFinite(result.values);
  return result;
}

inline double spectralJacobianVectorProductStep(
    const SpectralGrid3D &grid, std::span<const double> values,
    std::span<const double> direction,
    const SpectralJacobianVectorProductOptions &options) {
  if (values.size() != grid.size())
    throw std::runtime_error("spectral residual state size mismatch");
  if (direction.size() != grid.size())
    throw std::runtime_error("spectral residual direction size mismatch");
  if (!(options.relativeStep > 0.0) || !std::isfinite(options.relativeStep))
    throw std::runtime_error("spectral JVP relative step must be positive");
  if (options.absoluteStep < 0.0 || !std::isfinite(options.absoluteStep))
    throw std::runtime_error("spectral JVP absolute step must be finite");

  const double stateMax = spectralVectorMaxAbs(values);
  const double directionMax = spectralVectorMaxAbs(direction);
  if (!std::isfinite(stateMax) || !std::isfinite(directionMax))
    throw std::runtime_error("spectral JVP state and direction must be finite");
  if (directionMax == 0.0)
    return 0.0;
  return std::max(options.absoluteStep,
                  options.relativeStep * std::max(1.0, stateMax) /
                      directionMax);
}

inline SpectralJacobianVectorProductResult
evaluateSpectralJacobianVectorProduct(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    const std::vector<double> &direction,
    const SpectralJacobianVectorProductOptions &options = {}) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  const double step =
      spectralJacobianVectorProductStep(grid, values, direction, options);
  if (step == 0.0)
    return makeSpectralJacobianVectorProductResult(
        std::vector<double>(grid.size(), 0.0), step);

  std::vector<double> plus(values.size(), 0.0);
  for (std::size_t p = 0; p < values.size(); ++p)
    plus[p] = values[p] + step * direction[p];
  const auto plusResidual = assembleSpectralResidual(problem, plus);

  std::vector<double> out(values.size(), 0.0);
  if (options.centeredDifference) {
    std::vector<double> minus(values.size(), 0.0);
    for (std::size_t p = 0; p < values.size(); ++p)
      minus[p] = values[p] - step * direction[p];
    const auto minusResidual = assembleSpectralResidual(problem, minus);
    const double scale = 0.5 / step;
    for (std::size_t p = 0; p < values.size(); ++p)
      out[p] = (plusResidual.values[p] - minusResidual.values[p]) * scale;
  } else {
    const auto baseResidual = assembleSpectralResidual(problem, values);
    const double scale = 1.0 / step;
    for (std::size_t p = 0; p < values.size(); ++p)
      out[p] = (plusResidual.values[p] - baseResidual.values[p]) * scale;
  }

  return makeSpectralJacobianVectorProductResult(std::move(out), step);
}

inline SpectralJacobianVectorProductResult
evaluateSpectralJacobianVectorProduct(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const std::vector<double> &direction, const SpectralResidualKernel &kernel,
    std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields = {},
    const SpectralJacobianVectorProductOptions &options = {},
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  const SpectralResidualProblem problem{&grid, kernel, params, auxiliaryFields,
                                        coordinateMap, coordinateParams};
  return evaluateSpectralJacobianVectorProduct(problem, values, direction,
                                               options);
}

inline double spectralResidualRatio(double initialResidualL2,
                                    double residualL2) {
  if (initialResidualL2 > 0.0)
    return residualL2 / initialResidualL2;
  return residualL2 == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

inline bool reachedSpectralResidualTarget(
    const SpectralEllipticSolveResult &result,
    const SpectralEllipticSolveOptions &options) {
  if (!result.residualIsFinite())
    return false;
  if (options.residualTolerance > 0.0 &&
      result.finalResidualL2 <= options.residualTolerance)
    return true;
  if (options.residualRatioTarget > 0.0 &&
      result.residualRatio <= options.residualRatioTarget)
    return true;
  return false;
}

inline bool solveDenseLinearSystem(std::vector<double> matrix,
                                   std::vector<double> rhs,
                                   std::vector<double> &solution,
                                   double pivotTolerance) {
  const std::size_t n = rhs.size();
  if (matrix.size() != n * n)
    return false;
  solution.assign(n, 0.0);

  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivotRow = col;
    double pivotAbs = std::fabs(matrix[col * n + col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      const double candidate = std::fabs(matrix[row * n + col]);
      if (candidate > pivotAbs) {
        pivotAbs = candidate;
        pivotRow = row;
      }
    }
    if (!(pivotAbs > pivotTolerance) || !std::isfinite(pivotAbs))
      return false;
    if (pivotRow != col) {
      for (std::size_t j = col; j < n; ++j)
        std::swap(matrix[col * n + j], matrix[pivotRow * n + j]);
      std::swap(rhs[col], rhs[pivotRow]);
    }

    const double pivot = matrix[col * n + col];
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = matrix[row * n + col] / pivot;
      matrix[row * n + col] = 0.0;
      for (std::size_t j = col + 1; j < n; ++j)
        matrix[row * n + j] -= factor * matrix[col * n + j];
      rhs[row] -= factor * rhs[col];
    }
  }

  for (std::size_t rev = 0; rev < n; ++rev) {
    const std::size_t row = n - 1 - rev;
    double sum = rhs[row];
    for (std::size_t j = row + 1; j < n; ++j)
      sum -= matrix[row * n + j] * solution[j];
    const double diagonal = matrix[row * n + row];
    if (!(std::fabs(diagonal) > pivotTolerance) || !std::isfinite(diagonal))
      return false;
    solution[row] = sum / diagonal;
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

inline double spectralVectorDot(std::span<const double> lhs,
                                std::span<const double> rhs) {
  if (lhs.size() != rhs.size())
    throw std::runtime_error("spectral vector dot size mismatch");
  double out = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i)
    out += lhs[i] * rhs[i];
  return out;
}

inline double spectralVectorEuclideanNorm(std::span<const double> values) {
  return std::sqrt(std::max(0.0, spectralVectorDot(values, values)));
}

struct SpectralGMRESResult {
  bool converged = false;
  int iterations = 0;
  double residualL2 = std::numeric_limits<double>::infinity();
  std::vector<double> solution;
};

inline bool solveSpectralLeastSquaresNormalEquations(
    const std::vector<double> &hessenberg, std::size_t rows, std::size_t cols,
    std::size_t leadingDim, double beta, double pivotTolerance,
    std::vector<double> &solution, double &residualL2,
    std::size_t vectorSize) {
  std::vector<double> normal(cols * cols, 0.0);
  std::vector<double> rhs(cols, 0.0);
  for (std::size_t col = 0; col < cols; ++col) {
    rhs[col] = beta * hessenberg[col];
    for (std::size_t other = 0; other < cols; ++other) {
      double sum = 0.0;
      for (std::size_t row = 0; row < rows; ++row)
        sum += hessenberg[row * leadingDim + col] *
               hessenberg[row * leadingDim + other];
      normal[col * cols + other] = sum;
    }
  }

  if (!solveDenseLinearSystem(std::move(normal), std::move(rhs), solution,
                              pivotTolerance))
    return false;

  double residualSquared = 0.0;
  for (std::size_t row = 0; row < rows; ++row) {
    double value = row == 0 ? beta : 0.0;
    for (std::size_t col = 0; col < cols; ++col)
      value -= hessenberg[row * leadingDim + col] * solution[col];
    residualSquared += value * value;
  }
  residualL2 =
      std::sqrt(std::max(0.0, residualSquared) /
                static_cast<double>(std::max<std::size_t>(1, vectorSize)));
  return std::isfinite(residualL2);
}

inline SpectralGMRESResult solveSpectralGMRESByJVP(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    std::span<const double> rhs,
    const SpectralEllipticSolveOptions &options) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  const std::size_t n = grid.size();
  SpectralGMRESResult result;
  result.solution.assign(n, 0.0);
  if (rhs.size() != n || values.size() != n || options.gmresMaxIterations < 0)
    return result;

  const double rhsEuclidean = spectralVectorEuclideanNorm(rhs);
  const double rhsL2 =
      rhsEuclidean / std::sqrt(static_cast<double>(std::max<std::size_t>(1, n)));
  const double target = std::max(options.gmresTolerance,
                                 options.gmresRelativeTolerance * rhsL2);
  result.residualL2 = rhsL2;
  if (rhsL2 <= target) {
    result.converged = true;
    return result;
  }
  const std::size_t maxIterations = std::min<std::size_t>(
      n, static_cast<std::size_t>(options.gmresMaxIterations));
  if (maxIterations == 0)
    return result;

  std::vector<double> basis((maxIterations + 1) * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    basis[i] = rhs[i] / rhsEuclidean;

  std::vector<double> hessenberg((maxIterations + 1) * maxIterations, 0.0);
  std::vector<double> arnoldiVector(n, 0.0);
  std::vector<double> y;
  std::vector<double> bestY;
  std::size_t bestColumns = 0;

  for (std::size_t col = 0; col < maxIterations; ++col) {
    std::span<const double> direction(&basis[col * n], n);
    std::vector<double> directionVector(direction.begin(), direction.end());
    const auto jvp =
        evaluateSpectralJacobianVectorProduct(problem, values, directionVector,
                                              options.jvpOptions);
    if (!jvp.finite || jvp.values.size() != n)
      return result;
    arnoldiVector = jvp.values;

    for (std::size_t row = 0; row <= col; ++row) {
      std::span<const double> basisVector(&basis[row * n], n);
      const double h = spectralVectorDot(arnoldiVector, basisVector);
      hessenberg[row * maxIterations + col] = h;
      for (std::size_t i = 0; i < n; ++i)
        arnoldiVector[i] -= h * basisVector[i];
    }

    const double nextNorm = spectralVectorEuclideanNorm(arnoldiVector);
    hessenberg[(col + 1) * maxIterations + col] = nextNorm;
    if (nextNorm > options.linearPivotTolerance && col + 1 < maxIterations + 1) {
      for (std::size_t i = 0; i < n; ++i)
        basis[(col + 1) * n + i] = arnoldiVector[i] / nextNorm;
    }

    const std::size_t columns = col + 1;
    const std::size_t rows = col + 2;
    double projectedResidualL2 = std::numeric_limits<double>::infinity();
    if (!solveSpectralLeastSquaresNormalEquations(
            hessenberg, rows, columns, maxIterations, rhsEuclidean,
            options.linearPivotTolerance, y, projectedResidualL2, n)) {
      return result;
    }

    result.iterations = static_cast<int>(columns);
    result.residualL2 = projectedResidualL2;
    bestY = y;
    bestColumns = columns;
    if (projectedResidualL2 <= target) {
      result.converged = true;
      break;
    }
    if (nextNorm <= options.linearPivotTolerance)
      break;
  }

  if (bestColumns == 0)
    return result;
  result.solution.assign(n, 0.0);
  for (std::size_t col = 0; col < bestColumns; ++col) {
    for (std::size_t i = 0; i < n; ++i)
      result.solution[i] += bestY[col] * basis[col * n + i];
  }
  return result;
}

inline bool buildDenseSpectralJacobianByJVP(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    const SpectralEllipticSolveOptions &options,
    std::vector<double> &jacobian) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  const std::size_t n = grid.size();
  if (values.size() != n)
    throw std::runtime_error("spectral Newton state size mismatch");
  if (n > options.denseJacobianMaxUnknowns)
    return false;

  jacobian.assign(n * n, 0.0);
  std::vector<double> direction(n, 0.0);
  for (std::size_t col = 0; col < n; ++col) {
    direction[col] = 1.0;
    const auto jvp =
        evaluateSpectralJacobianVectorProduct(problem, values, direction,
                                              options.jvpOptions);
    direction[col] = 0.0;
    if (!jvp.finite || jvp.values.size() != n)
      return false;
    for (std::size_t row = 0; row < n; ++row)
      jacobian[row * n + col] = jvp.values[row];
  }
  return true;
}

inline void updateSpectralSolveResidualState(
    SpectralEllipticSolveResult &result,
    const SpectralResidualAssemblyResult &residual) {
  result.finalResidualL2 = residual.l2Norm;
  result.finalResidualMaxAbs = residual.maxAbs;
  result.residualRatio =
      spectralResidualRatio(result.initialResidualL2, result.finalResidualL2);
  result.usedGeneratedGridKernel =
      result.usedGeneratedGridKernel || residual.usedGeneratedGridKernel;
}

inline SpectralEllipticSolveResult solveSpectralNewton(
    const SpectralResidualProblem &problem, std::vector<double> &values,
    const SpectralEllipticSolveOptions &options = {}) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  SpectralEllipticSolveResult result;
  result.maxSteps = options.maxNewtonSteps;
  result.unknowns = grid.size();

  if (values.size() != grid.size() || options.maxNewtonSteps < 0 ||
      options.maxLineSearchSteps < 0 ||
      !(options.initialDamping > 0.0) ||
      !(options.lineSearchReduction > 0.0 &&
        options.lineSearchReduction < 1.0) ||
      !(options.minDamping > 0.0) ||
      !(options.linearPivotTolerance > 0.0) ||
      options.gmresMaxIterations < 0 || options.gmresTolerance < 0.0 ||
      options.gmresRelativeTolerance < 0.0) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }

  auto residual = assembleSpectralResidual(problem, values);
  result.initialResidualL2 = residual.l2Norm;
  updateSpectralSolveResidualState(result, residual);
  if (!residual.finite) {
    result.status = SpectralEllipticSolveStatus::InvalidResidual;
    return result;
  }
  if (reachedSpectralResidualTarget(result, options)) {
    result.status = SpectralEllipticSolveStatus::Converged;
    return result;
  }

  const std::size_t n = grid.size();
  const bool denseAllowed =
      options.denseJacobianMaxUnknowns > 0 &&
      n <= options.denseJacobianMaxUnknowns;
  const bool useDense =
      options.linearSolver == SpectralLinearSolveKind::DenseJacobian ||
      (options.linearSolver == SpectralLinearSolveKind::Auto && denseAllowed);
  if (useDense && !denseAllowed) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }

  for (int step = 1; step <= options.maxNewtonSteps; ++step) {
    std::vector<double> rhs(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      rhs[i] = -residual.values[i];

    std::vector<double> correction;
    if (useDense) {
      std::vector<double> jacobian;
      if (!buildDenseSpectralJacobianByJVP(problem, values, options,
                                           jacobian)) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      if (!solveDenseLinearSystem(std::move(jacobian), std::move(rhs),
                                  correction,
                                  options.linearPivotTolerance)) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      result.linearIterations += static_cast<int>(n);
      result.finalLinearResidualL2 = 0.0;
    } else {
      const auto linear =
          solveSpectralGMRESByJVP(problem, values, rhs, options);
      result.linearIterations += linear.iterations;
      result.finalLinearResidualL2 = linear.residualL2;
      result.usedMatrixFreeGMRES = true;
      if (!linear.converged || linear.solution.size() != n) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      correction = linear.solution;
    }

    bool accepted = false;
    double damping = options.initialDamping;
    std::vector<double> candidate(values.size(), 0.0);
    SpectralResidualAssemblyResult candidateResidual;
    for (int attempt = 0; attempt <= options.maxLineSearchSteps; ++attempt) {
      for (std::size_t i = 0; i < n; ++i)
        candidate[i] = values[i] + damping * correction[i];
      candidateResidual = assembleSpectralResidual(problem, candidate);
      if (candidateResidual.finite &&
          (candidateResidual.l2Norm < residual.l2Norm ||
           candidateResidual.l2Norm <= options.residualTolerance ||
           residual.l2Norm == 0.0)) {
        accepted = true;
        break;
      }
      damping *= options.lineSearchReduction;
      if (damping < options.minDamping)
        break;
    }

    if (!accepted) {
      result.status = SpectralEllipticSolveStatus::LineSearchFailed;
      return result;
    }

    values = std::move(candidate);
    residual = std::move(candidateResidual);
    result.steps = step;
    result.lastDamping = damping;
    updateSpectralSolveResidualState(result, residual);
    if (!residual.finite) {
      result.status = SpectralEllipticSolveStatus::InvalidResidual;
      return result;
    }
    if (reachedSpectralResidualTarget(result, options)) {
      result.status = SpectralEllipticSolveStatus::Converged;
      return result;
    }
  }

  result.status = SpectralEllipticSolveStatus::MaxSteps;
  return result;
}

} // namespace tensorium_mlir::runtime
