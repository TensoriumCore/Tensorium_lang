#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace tensorium_mlir::runtime {

inline void validateTwoPunctureInversionGrid(const SpectralGrid3D *grid) {
  if (!grid)
    throw std::runtime_error(
        "two-puncture inversion projector received a null grid");
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
}

inline void projectTwoPunctureInversionParity(const SpectralGrid3D &grid,
                                               std::span<double> values,
                                               bool even) {
  if (values.size() != grid.size())
    throw std::runtime_error(
        "two-puncture inversion projector received invalid component data");
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const std::size_t rotatedK = (k + grid.n3() / 2) % grid.n3();
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const std::size_t reflectedJ = grid.n2() - 1 - j;
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const std::size_t index = grid.index(i, j, k);
        const std::size_t image = grid.index(i, reflectedJ, rotatedK);
        if (index >= image)
          continue;
        const double projected =
            0.5 * (values[index] + (even ? values[image] : -values[image]));
        values[index] = projected;
        values[image] = even ? projected : -projected;
      }
    }
  }
}

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
  validateTwoPunctureInversionGrid(grid);
  projectTwoPunctureInversionParity(
      *grid, std::span<double>(values, static_cast<std::size_t>(valueCount)),
      true);
}

// Logical derivatives of an inversion-even scalar have a fixed parity under
// (A, B, phi) -> (A, -B, phi + pi). Enforce that parity before the singular
// bispherical-to-Cartesian derivative map amplifies round-off near rho=0.
inline void twoPunctureInversionEvenDerivativeProjector(
    const SpectralGrid3D *grid, SpectralDerivatives3D *derivatives, void *) {
  validateTwoPunctureInversionGrid(grid);
  if (!derivatives)
    throw std::runtime_error(
        "two-puncture inversion projector received null derivatives");

  projectTwoPunctureInversionParity(*grid, derivatives->value, true);
  projectTwoPunctureInversionParity(*grid, derivatives->d1, true);
  projectTwoPunctureInversionParity(*grid, derivatives->d2, false);
  projectTwoPunctureInversionParity(*grid, derivatives->d3, true);
  projectTwoPunctureInversionParity(*grid, derivatives->d11, true);
  projectTwoPunctureInversionParity(*grid, derivatives->d12, false);
  projectTwoPunctureInversionParity(*grid, derivatives->d13, true);
  projectTwoPunctureInversionParity(*grid, derivatives->d22, true);
  projectTwoPunctureInversionParity(*grid, derivatives->d23, false);
  projectTwoPunctureInversionParity(*grid, derivatives->d33, true);
}

inline SpectralFieldProjector makeTwoPunctureInversionEvenFieldProjector() {
  return SpectralFieldProjector{
      "tensorium_spectral_two_puncture_inversion_even_projector",
      &twoPunctureInversionEvenFieldProjector, nullptr,
      &twoPunctureInversionEvenDerivativeProjector};
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
