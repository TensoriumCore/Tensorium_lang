#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <span>
#include <vector>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralResidualProblem;
using tensorium_mlir::runtime::assembleSpectralResidual;
using tensorium_mlir::runtime::spectralResidualGridKernelFromDesc;
using tensorium_mlir::runtime::spectralResidualKernelFromDesc;

double manufacturedU(double x, double y, double z) {
  return (4.0 * x * x * x - 3.0 * x) + y * y + 0.25 * std::cos(2.0 * z);
}

double manufacturedLaplacian(double x, double, double z) {
  return 24.0 * x + 2.0 - std::cos(2.0 * z);
}

double manufacturedSource(double x, double y, double z) {
  return x * y + 0.125 * std::sin(3.0 * z);
}

double maxAbs(const std::vector<double> &values) {
  double out = 0.0;
  for (double value : values)
    out = std::max(out, std::abs(value));
  return out;
}

} // namespace

int main() {
  try {
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral point kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral grid kernel");

    const auto pointKernel =
        spectralResidualKernelFromDesc(tensorium_spectral_residual_kernels[0]);
    const auto gridKernel = spectralResidualGridKernelFromDesc(
        tensorium_spectral_residual_grid_kernels[0]);

    const double alpha = -0.25;
    const double params[] = {alpha};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(8),
                        SpectralAxis::chebyshevZeros(7),
                        SpectralAxis::fourierPeriodic(12));

    std::vector<double> values(grid.size(), 0.0);
    std::vector<double> source(grid.size(), 0.0);

    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double zk = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double yj = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double xi = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          values[p] = manufacturedU(xi, yj, zk);
          source[p] = manufacturedSource(xi, yj, zk);
        }
      }
    }

    auto derivs = grid.derivatives(values);
    const std::array<std::vector<double>, 1> auxiliaryFields{source};
    SpectralResidualProblem problem{
        &grid,
        pointKernel,
        std::span<const double>(params, 1),
        std::span<const std::vector<double>>(auxiliaryFields.data(),
                                             auxiliaryFields.size())};
    problem.gridKernel = gridKernel;
    const auto fastResidual = assembleSpectralResidual(problem, derivs);
    if (!fastResidual.usedGeneratedGridKernel) {
      std::fprintf(stderr,
                   "spectral residual assembly did not use generated grid "
                   "kernel\n");
      return 3;
    }

    problem.gridKernel = {};
    const auto fallbackResidual = assembleSpectralResidual(problem, derivs);
    if (fallbackResidual.usedGeneratedGridKernel) {
      std::fprintf(stderr,
                   "spectral residual fallback unexpectedly used grid kernel\n");
      return 3;
    }

    std::vector<double> errors(grid.size(), 0.0);
    std::vector<double> fallbackDelta(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double zk = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double yj = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double xi = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          const double expected = manufacturedLaplacian(xi, yj, zk) +
                                  alpha * manufacturedU(xi, yj, zk) +
                                  manufacturedSource(xi, yj, zk);
          errors[p] = fastResidual.values[p] - expected;
          fallbackDelta[p] =
              fastResidual.values[p] - fallbackResidual.values[p];
        }
      }
    }

    const double error = maxAbs(errors);
    const double fallbackError = maxAbs(fallbackDelta);
    std::printf("[generated-spectral-global-residual] max error = %.17g\n",
                error);
    std::printf(
        "[generated-spectral-global-residual] fast/fallback delta = %.17g\n",
        fallbackError);
    if (error > 4e-10 || fallbackError > 1e-12) {
      std::fprintf(stderr, "generated spectral global residual mismatch\n");
      return 4;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated spectral global residual runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
