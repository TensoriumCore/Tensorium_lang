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
using tensorium_mlir::runtime::evaluateSpectralResidualWithAuxFields;
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
                  "expected at least one generated spectral residual kernel");
    const auto kernel =
        spectralResidualKernelFromDesc(tensorium_spectral_residual_kernels[0]);

    const double alpha = -0.25;
    const double params[] = {alpha};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(8),
                        SpectralAxis::chebyshevZeros(7),
                        SpectralAxis::fourierPeriodic(12));

    std::vector<double> values(grid.size(), 0.0);
    std::vector<double> source(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          values[p] = manufacturedU(x, y, z);
          source[p] = manufacturedSource(x, y, z);
        }
      }
    }

    const std::array<std::vector<double>, 1> auxiliaryFields{source};
    const auto residual = evaluateSpectralResidualWithAuxFields(
        grid, values, kernel, std::span<const double>(params, 1),
        std::span<const std::vector<double>>(auxiliaryFields.data(),
                                             auxiliaryFields.size()));

    std::vector<double> errors(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          const double expected = manufacturedLaplacian(x, y, z) +
                                  alpha * manufacturedU(x, y, z) +
                                  manufacturedSource(x, y, z);
          errors[p] = residual[p] - expected;
        }
      }
    }

    const double error = maxAbs(errors);
    std::printf("[generated-spectral-aux-residual] kernel = %s\n",
                kernel.symbolName.c_str());
    std::printf("[generated-spectral-aux-residual] max error = %.17g\n",
                error);
    if (error > 4e-10) {
      std::fprintf(stderr, "generated spectral aux residual mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "generated spectral aux residual runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
