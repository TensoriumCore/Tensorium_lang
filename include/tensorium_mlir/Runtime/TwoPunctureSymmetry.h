#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace tensorium_mlir::runtime {

// Project a scalar field onto the even subspace of
// (A, B, phi) -> (A, -B, phi + pi). Under the TwoPuncture coordinate map this
// is Cartesian inversion (x, y, z) -> (-x, -y, -z). Equal-mass binaries with
// opposite tangential momenta may use this optional reduction; asymmetric
// binaries must leave it disabled.
inline void twoPunctureInversionEvenFieldProjector(const SpectralGrid3D *grid,
                                                   double *values,
                                                   std::int64_t valueCount,
                                                   void *) {
  if (!grid || !values || valueCount < 0 ||
      static_cast<std::size_t>(valueCount) != grid->size()) {
    throw std::runtime_error(
        "two-puncture inversion projector received invalid field data");
  }
  if (grid->axis(2).basis != SpectralBasis::FourierPeriodic ||
      grid->n3() == 0 || grid->n3() % 2 != 0) {
    throw std::runtime_error(
        "two-puncture inversion projector requires an even Fourier axis");
  }
  for (std::size_t j = 0; j < grid->n2(); ++j) {
    const std::size_t reflectedJ = grid->n2() - 1 - j;
    const double reflectionError =
        std::fabs(grid->axis(1).points[j] + grid->axis(1).points[reflectedJ]);
    if (reflectionError > 1.0e-12) {
      throw std::runtime_error(
          "two-puncture inversion projector requires a B-reflection grid");
    }
  }

  for (std::size_t k = 0; k < grid->n3(); ++k) {
    const std::size_t rotatedK = (k + grid->n3() / 2) % grid->n3();
    for (std::size_t j = 0; j < grid->n2(); ++j) {
      const std::size_t reflectedJ = grid->n2() - 1 - j;
      for (std::size_t i = 0; i < grid->n1(); ++i) {
        const std::size_t index = grid->index(i, j, k);
        const std::size_t image = grid->index(i, reflectedJ, rotatedK);
        if (index >= image)
          continue;
        const double average = 0.5 * (values[index] + values[image]);
        values[index] = average;
        values[image] = average;
      }
    }
  }
}

inline SpectralFieldProjector makeTwoPunctureInversionEvenFieldProjector() {
  return SpectralFieldProjector{
      "tensorium_spectral_two_puncture_inversion_even_projector",
      &twoPunctureInversionEvenFieldProjector, nullptr};
}

inline double
measureTwoPunctureInversionParityError(const SpectralGrid3D &grid,
                                       std::span<const double> values) {
  if (values.size() != grid.size() || grid.n3() == 0 || grid.n3() % 2 != 0)
    throw std::runtime_error(
        "two-puncture inversion parity measurement requires a valid field");
  double maxError = 0.0;
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const std::size_t rotatedK = (k + grid.n3() / 2) % grid.n3();
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const std::size_t reflectedJ = grid.n2() - 1 - j;
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        maxError = std::max(
            maxError, std::fabs(values[grid.index(i, j, k)] -
                                values[grid.index(i, reflectedJ, rotatedK)]));
      }
    }
  }
  return maxError;
}

} // namespace tensorium_mlir::runtime
