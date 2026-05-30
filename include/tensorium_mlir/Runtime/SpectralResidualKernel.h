#pragma once

#include "tensorium_mlir/Runtime/SpectralGrid.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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

inline tensorium_spectral_residual_point makeSpectralResidualPoint(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    std::size_t i, std::size_t j, std::size_t k,
    const SpectralCoordinateMap &coordinateMap,
    std::span<const double> coordinateParams) {
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
  return out;
}

inline std::vector<double> evaluateSpectralResidual(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  validateSpectralDerivativeBundle(grid, derivs);
  if (!kernel.evaluate)
    throw std::runtime_error("spectral residual kernel callback is null");

  std::vector<double> out(grid.size(), 0.0);
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const auto point = makeSpectralResidualPoint(
            grid, derivs, i, j, k, coordinateMap, coordinateParams);
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
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  return evaluateSpectralResidual(grid, grid.derivatives(values), kernel,
                                  params, coordinateMap, coordinateParams);
}

} // namespace tensorium_mlir::runtime
