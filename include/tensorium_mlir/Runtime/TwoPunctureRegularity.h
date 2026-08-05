#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualTypes.h"
#include "tensorium_mlir/Runtime/TwoPunctureSymmetry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace tensorium_mlir::runtime {

struct TwoPunctureScalarRegularityDiagnostics {
  double innerAxisPhiVariation = 0.0;
  double positiveOuterAxisPhiVariation = 0.0;
  double negativeOuterAxisPhiVariation = 0.0;

  double maxPhiVariation() const {
    return std::max({innerAxisPhiVariation, positiveOuterAxisPhiVariation,
                     negativeOuterAxisPhiVariation});
  }
};

inline void requireTwoPunctureScalarRegularityGrid(const SpectralGrid3D &grid) {
  if (grid.axis(0).basis != SpectralBasis::ChebyshevZeros ||
      grid.axis(1).basis != SpectralBasis::ChebyshevZeros ||
      grid.axis(2).basis != SpectralBasis::FourierPeriodic) {
    throw std::runtime_error(
        "two-puncture scalar regularity requires Chebyshev-Chebyshev-Fourier "
        "axes");
  }
  for (std::size_t dim = 0; dim < 2; ++dim) {
    for (double point : grid.axis(dim).points) {
      if (!(point > -1.0 && point < 1.0)) {
        throw std::runtime_error(
            "two-puncture scalar regularity requires axes inside (-1,1)");
      }
    }
  }
}

inline double spectralPhiVariation(std::span<const double> values) {
  if (values.empty())
    return 0.0;
  double mean = 0.0;
  for (double value : values)
    mean += value;
  mean /= static_cast<double>(values.size());
  double variation = 0.0;
  for (double value : values)
    variation = std::max(variation, std::fabs(value - mean));
  return variation;
}

inline TwoPunctureScalarRegularityDiagnostics
measureTwoPunctureScalarRegularity(const SpectralGrid3D &grid,
                                   const std::vector<double> &values) {
  requireTwoPunctureScalarRegularityGrid(grid);
  if (values.size() != grid.size())
    throw std::runtime_error("two-puncture regularity field size mismatch");

  TwoPunctureScalarRegularityDiagnostics out;
  std::vector<double> phiTrace(grid.n3(), 0.0);

  for (std::size_t j = 0; j < grid.n2(); ++j) {
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      std::vector<double> line(grid.n1(), 0.0);
      for (std::size_t i = 0; i < grid.n1(); ++i)
        line[i] = values[grid.index(i, j, k)];
      phiTrace[k] = grid.axis(0).interpolate(line, -1.0);
    }
    out.innerAxisPhiVariation =
        std::max(out.innerAxisPhiVariation, spectralPhiVariation(phiTrace));
  }

  for (std::size_t i = 0; i < grid.n1(); ++i) {
    for (std::size_t side = 0; side < 2; ++side) {
      const double boundary = side == 0 ? -1.0 : 1.0;
      for (std::size_t k = 0; k < grid.n3(); ++k) {
        std::vector<double> line(grid.n2(), 0.0);
        for (std::size_t j = 0; j < grid.n2(); ++j)
          line[j] = values[grid.index(i, j, k)];
        phiTrace[k] = grid.axis(1).interpolate(line, boundary);
      }
      double &variation = side == 0 ? out.positiveOuterAxisPhiVariation
                                    : out.negativeOuterAxisPhiVariation;
      variation = std::max(variation, spectralPhiVariation(phiTrace));
    }
  }
  return out;
}

// Remove nonzero Fourier traces from A=-1 and B=+-1. These three logical
// boundaries all map to the Cartesian x axis, where a continuous scalar has a
// unique value independent of phi. The correction is a transfinite linear
// extension of the boundary trace. It preserves the phi-average while changing
// only the non-axisymmetric component needed to regularize those traces.
inline void
twoPunctureScalarRegularityFieldProjector(const SpectralGrid3D *grid,
                                          double *values,
                                          std::int64_t valueCount, void *) {
  if (!grid || !values || valueCount < 0 ||
      static_cast<std::size_t>(valueCount) != grid->size()) {
    throw std::runtime_error(
        "two-puncture regularity projector received invalid field data");
  }
  requireTwoPunctureScalarRegularityGrid(*grid);

  std::vector<double> mean(grid->n1() * grid->n2(), 0.0);
  for (std::size_t j = 0; j < grid->n2(); ++j) {
    for (std::size_t i = 0; i < grid->n1(); ++i) {
      for (std::size_t k = 0; k < grid->n3(); ++k)
        mean[i + grid->n1() * j] += values[grid->index(i, j, k)];
      mean[i + grid->n1() * j] /= static_cast<double>(grid->n3());
    }
  }

  std::vector<double> regularized(values, values + grid->size());
  std::vector<double> line(std::max(grid->n1(), grid->n2()), 0.0);
  for (std::size_t k = 0; k < grid->n3(); ++k) {
    for (std::size_t i = 0; i < grid->n1(); ++i) {
      for (std::size_t j = 0; j < grid->n2(); ++j) {
        line[j] = regularized[grid->index(i, j, k)] - mean[i + grid->n1() * j];
      }
      const double atMinus = grid->axis(1).interpolate(
          std::vector<double>(line.begin(), line.begin() + grid->n2()), -1.0);
      const double atPlus = grid->axis(1).interpolate(
          std::vector<double>(line.begin(), line.begin() + grid->n2()), 1.0);
      for (std::size_t j = 0; j < grid->n2(); ++j) {
        const double B = grid->axis(1).points[j];
        regularized[grid->index(i, j, k)] -=
            0.5 * (1.0 - B) * atMinus + 0.5 * (1.0 + B) * atPlus;
      }
    }

    for (std::size_t j = 0; j < grid->n2(); ++j) {
      for (std::size_t i = 0; i < grid->n1(); ++i) {
        line[i] = regularized[grid->index(i, j, k)] - mean[i + grid->n1() * j];
      }
      const double atInner = grid->axis(0).interpolate(
          std::vector<double>(line.begin(), line.begin() + grid->n1()), -1.0);
      for (std::size_t i = 0; i < grid->n1(); ++i) {
        const double A = grid->axis(0).points[i];
        regularized[grid->index(i, j, k)] -= 0.5 * (1.0 - A) * atInner;
      }
    }
  }
  std::copy(regularized.begin(), regularized.end(), values);
}

inline SpectralFieldProjector makeTwoPunctureScalarRegularityFieldProjector() {
  return SpectralFieldProjector{
      "tensorium_spectral_two_puncture_scalar_regularity_projector",
      &twoPunctureScalarRegularityFieldProjector, nullptr};
}

inline void twoPunctureInversionEvenRegularityFieldProjector(
    const SpectralGrid3D *grid, double *values, std::int64_t valueCount,
    void *userData) {
  twoPunctureScalarRegularityFieldProjector(grid, values, valueCount, userData);
  twoPunctureInversionEvenFieldProjector(grid, values, valueCount, userData);
}

inline SpectralFieldProjector
makeTwoPunctureInversionEvenRegularityFieldProjector() {
  return SpectralFieldProjector{
      "tensorium_spectral_two_puncture_inversion_even_regularity_projector",
      &twoPunctureInversionEvenRegularityFieldProjector, nullptr};
}

} // namespace tensorium_mlir::runtime
