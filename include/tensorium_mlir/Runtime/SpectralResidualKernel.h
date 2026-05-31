#pragma once

#include "tensorium_mlir/Runtime/SpectralGrid.h"

#include <algorithm>
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

typedef void (*tensorium_spectral_coordinate_map_fn)(
    const double *logical, double *physical, const double *params,
    std::int64_t param_count, void *user_data);

typedef struct tensorium_spectral_residual_kernel_desc {
  const char *symbol_name;
  tensorium_spectral_residual_kernel_fn evaluate;
  void *user_data;
} tensorium_spectral_residual_kernel_desc;

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
};

struct SpectralResidualAssemblyResult {
  std::vector<double> values;
  double l2Norm = 0.0;
  double maxAbs = 0.0;
  bool finite = true;

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

inline SpectralResidualKernel
spectralResidualKernelFromDesc(const tensorium_spectral_residual_kernel_desc &desc) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("spectral residual kernel symbol is empty");
  if (!desc.evaluate)
    throw std::runtime_error("spectral residual kernel callback is null");
  return SpectralResidualKernel{desc.symbol_name, desc.evaluate,
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
makeSpectralResidualAssemblyResult(std::vector<double> values) {
  SpectralResidualAssemblyResult result;
  result.values = std::move(values);
  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = spectralVectorIsFinite(result.values);
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

inline SpectralResidualAssemblyResult assembleSpectralResidual(
    const SpectralResidualProblem &problem,
    const SpectralDerivatives3D &derivs) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
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

} // namespace tensorium_mlir::runtime
