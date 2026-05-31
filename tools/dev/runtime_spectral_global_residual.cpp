#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralJacobianVectorProductOptions;
using tensorium_mlir::runtime::SpectralResidualKernel;
using tensorium_mlir::runtime::SpectralResidualProblem;
using tensorium_mlir::runtime::assembleSpectralResidual;
using tensorium_mlir::runtime::evaluateSpectralJacobianVectorProduct;

double residualKernel(const tensorium_spectral_residual_point *point,
                      const double *params, std::int64_t paramCount, void *) {
  if (paramCount != 1)
    throw std::runtime_error("spectral global residual test expects 1 param");
  if (point->aux_count != 1 || !point->aux_values)
    throw std::runtime_error("spectral global residual test expects 1 aux");
  return point->d11 + point->d22 + point->d33 + params[0] * point->value +
         point->aux_values[0];
}

double manufacturedU(double x, double y, double z) {
  return (4.0 * x * x * x - 3.0 * x) + y * y + 0.25 * std::cos(2.0 * z);
}

double manufacturedLaplacian(double x, double, double z) {
  return 24.0 * x + 2.0 - std::cos(2.0 * z);
}

double manufacturedSource(double x, double y, double z) {
  return x * y + 0.125 * std::sin(3.0 * z);
}

double manufacturedDirection(double x, double y, double z) {
  return 0.35 * (4.0 * x * x * x - 3.0 * x) - 0.2 * y * y +
         0.125 * std::cos(3.0 * z);
}

double manufacturedDirectionLaplacian(double x, double, double z) {
  return 8.4 * x - 0.4 - 1.125 * std::cos(3.0 * z);
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
    const double params[] = {alpha};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(9),
                        SpectralAxis::chebyshevZeros(8),
                        SpectralAxis::fourierPeriodic(16));

    std::vector<double> values(grid.size(), 0.0);
    std::vector<double> direction(grid.size(), 0.0);
    std::vector<double> source(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          values[p] = manufacturedU(x, y, z);
          direction[p] = manufacturedDirection(x, y, z);
          source[p] = manufacturedSource(x, y, z);
        }
      }
    }

    const std::array<std::vector<double>, 1> auxiliaryFields{source};
    const SpectralResidualKernel kernel{"tensorium_runtime_spectral_poisson",
                                        &residualKernel, nullptr};
    const SpectralResidualProblem problem{
        &grid,
        kernel,
        std::span<const double>(params, 1),
        std::span<const std::vector<double>>(auxiliaryFields.data(),
                                             auxiliaryFields.size())};

    const auto assembled = assembleSpectralResidual(problem, values);
    std::vector<double> residualErrors(grid.size(), 0.0);
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
          residualErrors[p] = assembled.values[p] - expected;
        }
      }
    }

    const double residualError = maxAbs(residualErrors);
    std::printf("[spectral-global-residual] residual max error = %.17g\n",
                residualError);
    std::printf("[spectral-global-residual] residual l2 = %.17g max = %.17g\n",
                assembled.l2Norm, assembled.maxAbs);
    if (!assembled.finite || assembled.size() != grid.size() ||
        residualError > 4e-10) {
      std::fprintf(stderr, "spectral global residual assembly mismatch\n");
      return 3;
    }

    SpectralJacobianVectorProductOptions jvpOptions;
    jvpOptions.relativeStep = 1.0e-5;
    const auto jvp =
        evaluateSpectralJacobianVectorProduct(problem, values, direction,
                                              jvpOptions);
    std::vector<double> jvpErrors(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          const double expected = manufacturedDirectionLaplacian(x, y, z) +
                                  alpha * manufacturedDirection(x, y, z);
          jvpErrors[p] = jvp.values[p] - expected;
        }
      }
    }

    const double jvpError = maxAbs(jvpErrors);
    std::printf("[spectral-global-residual] jvp step = %.17g\n", jvp.step);
    std::printf("[spectral-global-residual] jvp max error = %.17g\n",
                jvpError);
    if (!jvp.finite || jvp.size() != grid.size() || jvp.step <= 0.0 ||
        jvpError > 5e-8) {
      std::fprintf(stderr, "spectral global residual JVP mismatch\n");
      return 4;
    }

    const std::vector<double> zeroDirection(grid.size(), 0.0);
    const auto zeroJvp =
        evaluateSpectralJacobianVectorProduct(problem, values, zeroDirection,
                                              jvpOptions);
    if (zeroJvp.step != 0.0 || zeroJvp.maxAbs != 0.0 ||
        zeroJvp.size() != grid.size()) {
      std::fprintf(stderr, "spectral zero-direction JVP mismatch\n");
      return 5;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "spectral global residual runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
