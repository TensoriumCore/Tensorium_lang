#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualTypes.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace tensorium_mlir::runtime {

// Maps the solver variable v to the physical residual variable
// u = scale * (q_axis - boundary) * v. The complete logical derivative bundle
// is transformed by the product rule before any coordinate derivative map is
// applied.
inline void linearBoundaryFactorUnknownMap(
    const double logical[3],
    const SpectralPointDerivatives3D *solverDerivatives,
    SpectralPointDerivatives3D *physicalDerivatives, const double *params,
    std::int64_t paramCount, void *) {
  if (!logical || !solverDerivatives || !physicalDerivatives || !params ||
      paramCount != 3 || !std::isfinite(params[0]) ||
      !std::isfinite(params[1]) || !std::isfinite(params[2])) {
    throw std::runtime_error(
        "linear boundary-factor unknown map requires axis, boundary, and "
        "scale parameters");
  }

  if (params[0] < 0.0 || params[0] > 2.0)
    throw std::runtime_error(
        "linear boundary-factor unknown map axis must be 0, 1, or 2");
  const auto roundedAxis = static_cast<std::int64_t>(std::llround(params[0]));
  if (std::abs(params[0] - static_cast<double>(roundedAxis)) > 1.0e-12) {
    throw std::runtime_error(
        "linear boundary-factor unknown map axis must be 0, 1, or 2");
  }

  const std::size_t axis = static_cast<std::size_t>(roundedAxis);
  const double scale = params[2];
  const double weight = scale * (logical[axis] - params[1]);
  const std::array<double, 3> first = {
      solverDerivatives->d1, solverDerivatives->d2, solverDerivatives->d3};
  const std::array<std::array<double, 3>, 3> second = {{
      {solverDerivatives->d11, solverDerivatives->d12, solverDerivatives->d13},
      {solverDerivatives->d12, solverDerivatives->d22, solverDerivatives->d23},
      {solverDerivatives->d13, solverDerivatives->d23, solverDerivatives->d33},
  }};

  std::array<double, 3> mappedFirst{};
  std::array<std::array<double, 3>, 3> mappedSecond{};
  for (std::size_t i = 0; i < 3; ++i) {
    mappedFirst[i] = weight * first[i];
    if (i == axis)
      mappedFirst[i] += scale * solverDerivatives->value;
    for (std::size_t j = 0; j < 3; ++j) {
      mappedSecond[i][j] = weight * second[i][j];
      if (i == axis)
        mappedSecond[i][j] += scale * first[j];
      if (j == axis)
        mappedSecond[i][j] += scale * first[i];
    }
  }

  physicalDerivatives->value = weight * solverDerivatives->value;
  physicalDerivatives->d1 = mappedFirst[0];
  physicalDerivatives->d2 = mappedFirst[1];
  physicalDerivatives->d3 = mappedFirst[2];
  physicalDerivatives->d11 = mappedSecond[0][0];
  physicalDerivatives->d12 = mappedSecond[0][1];
  physicalDerivatives->d13 = mappedSecond[0][2];
  physicalDerivatives->d22 = mappedSecond[1][1];
  physicalDerivatives->d23 = mappedSecond[1][2];
  physicalDerivatives->d33 = mappedSecond[2][2];
}

inline SpectralUnknownMap makeLinearBoundaryFactorUnknownMap() {
  return SpectralUnknownMap{
      "tensorium_spectral_linear_boundary_factor_unknown_map",
      &linearBoundaryFactorUnknownMap, nullptr};
}

} // namespace tensorium_mlir::runtime
