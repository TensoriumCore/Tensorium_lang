#include "tensorium_mlir/Runtime/SpectralGrid.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralGrid3D;

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
    const double alpha = -0.25;
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(8),
                        SpectralAxis::chebyshevZeros(7),
                        SpectralAxis::fourierPeriodic(12));

    std::vector<double> values(grid.size(), 0.0);
    std::vector<double> source(grid.size(), 0.0);
    std::vector<double> x(grid.size(), 0.0);
    std::vector<double> y(grid.size(), 0.0);
    std::vector<double> z(grid.size(), 0.0);

    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double zk = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double yj = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double xi = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          values[p] = manufacturedU(xi, yj, zk);
          source[p] = manufacturedSource(xi, yj, zk);
          x[p] = xi;
          y[p] = yj;
          z[p] = zk;
        }
      }
    }

    auto derivs = grid.derivatives(values);
    std::vector<double> residual(grid.size(), 0.0);
    tensorium_call_spectral_residual_grid_H(
        static_cast<int64_t>(grid.size()), alpha, derivs.value.data(),
        derivs.d1.data(), derivs.d2.data(), derivs.d3.data(),
        derivs.d11.data(), derivs.d12.data(), derivs.d13.data(),
        derivs.d22.data(), derivs.d23.data(), derivs.d33.data(),
        source.data(), x.data(), y.data(), z.data(), residual.data());

    std::vector<double> errors(grid.size(), 0.0);
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
          errors[p] = residual[p] - expected;
        }
      }
    }

    const double error = maxAbs(errors);
    std::printf("[generated-spectral-global-residual] max error = %.17g\n",
                error);
    if (error > 4e-10) {
      std::fprintf(stderr, "generated spectral global residual mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated spectral global residual runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
