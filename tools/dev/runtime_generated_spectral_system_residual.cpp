#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralResidualProblem;
using tensorium_mlir::runtime::SpectralResidualSystemEquation;
using tensorium_mlir::runtime::SpectralResidualSystemProblem;
using tensorium_mlir::runtime::assembleSpectralResidualSystem;
using tensorium_mlir::runtime::spectralResidualGridKernelFromDesc;
using tensorium_mlir::runtime::spectralResidualKernelFromDesc;

const tensorium_spectral_residual_kernel_desc &
findPointKernel(const char *symbol) {
  for (int i = 0; i < TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT; ++i) {
    if (tensorium_spectral_residual_kernels[i].symbol_name &&
        std::strcmp(tensorium_spectral_residual_kernels[i].symbol_name,
                    symbol) == 0) {
      return tensorium_spectral_residual_kernels[i];
    }
  }
  throw std::runtime_error(std::string("missing spectral point kernel: ") +
                           symbol);
}

const tensorium_spectral_residual_grid_kernel_desc &
findGridKernel(const char *symbol) {
  for (int i = 0; i < TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT; ++i) {
    if (tensorium_spectral_residual_grid_kernels[i].symbol_name &&
        std::strcmp(tensorium_spectral_residual_grid_kernels[i].symbol_name,
                    symbol) == 0) {
      return tensorium_spectral_residual_grid_kernels[i];
    }
  }
  throw std::runtime_error(std::string("missing spectral grid kernel: ") +
                           symbol);
}

double exactU(double x, double y, double z) {
  return (4.0 * x * x * x - 3.0 * x) + 0.5 * y * y +
         0.2 * std::cos(2.0 * z);
}

double lapU(double x, double, double z) {
  return 24.0 * x + 1.0 - 0.8 * std::cos(2.0 * z);
}

double exactV(double x, double y, double z) {
  return 0.25 * (2.0 * x * x - 1.0) - 0.4 * y * y * y +
         0.15 * std::sin(3.0 * z);
}

double lapV(double, double y, double z) {
  return 1.0 - 2.4 * y - 1.35 * std::sin(3.0 * z);
}

double maxAbs(std::span<const double> values) {
  double out = 0.0;
  for (double value : values)
    out = std::max(out, std::abs(value));
  return out;
}

} // namespace

int main() {
  try {
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 2,
                  "expected at least two generated spectral point kernels");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 2,
                  "expected at least two generated spectral grid kernels");

    const auto huPoint =
        spectralResidualKernelFromDesc(findPointKernel("tensorium_spectral_residual_Hu"));
    const auto hvPoint =
        spectralResidualKernelFromDesc(findPointKernel("tensorium_spectral_residual_Hv"));
    const auto huGrid = spectralResidualGridKernelFromDesc(
        findGridKernel("tensorium_spectral_residual_grid_Hu"));
    const auto hvGrid = spectralResidualGridKernelFromDesc(
        findGridKernel("tensorium_spectral_residual_grid_Hv"));

    const double alpha = 0.5;
    const double beta = -0.35;
    const double coupling = 0.125;
    const double huParams[] = {alpha, coupling};
    const double hvParams[] = {beta, coupling};

    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(7),
                        SpectralAxis::chebyshevZeros(6),
                        SpectralAxis::fourierPeriodic(10));

    std::vector<double> u(grid.size(), 0.0);
    std::vector<double> v(grid.size(), 0.0);
    std::vector<double> sourceU(grid.size(), 0.0);
    std::vector<double> sourceV(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          u[p] = exactU(x, y, z);
          v[p] = exactV(x, y, z);
          sourceU[p] = -(lapU(x, y, z) + alpha * u[p] + coupling * v[p]);
          sourceV[p] = -(lapV(x, y, z) + beta * v[p] + coupling * u[p]);
        }
      }
    }

    const std::array<std::vector<double>, 2> unknownFields{u, v};
    const std::array<std::vector<double>, 2> huAuxiliaryFields{sourceU, v};
    const std::array<std::vector<double>, 2> hvAuxiliaryFields{sourceV, u};

    SpectralResidualProblem huProblem{
        &grid,
        huPoint,
        std::span<const double>(huParams, 2),
        std::span<const std::vector<double>>(huAuxiliaryFields.data(),
                                             huAuxiliaryFields.size())};
    huProblem.gridKernel = huGrid;

    SpectralResidualProblem hvProblem{
        &grid,
        hvPoint,
        std::span<const double>(hvParams, 2),
        std::span<const std::vector<double>>(hvAuxiliaryFields.data(),
                                             hvAuxiliaryFields.size())};
    hvProblem.gridKernel = hvGrid;

    const std::array<SpectralResidualSystemEquation, 2> equations{{
        SpectralResidualSystemEquation{huProblem, 0, "Hu"},
        SpectralResidualSystemEquation{hvProblem, 1, "Hv"},
    }};
    const SpectralResidualSystemProblem system{
        &grid, std::span<const SpectralResidualSystemEquation>(
                   equations.data(), equations.size())};

    const auto result = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(unknownFields.data(),
                                                     unknownFields.size()));
    const double huMax = result.equationResults.empty()
                             ? std::numeric_limits<double>::infinity()
                             : result.equationResults[0].maxAbs;
    const double hvMax = result.equationResults.size() < 2
                             ? std::numeric_limits<double>::infinity()
                             : result.equationResults[1].maxAbs;

    std::printf("[generated-spectral-system] equations = %zu points = %zu\n",
                result.equationCount, result.pointsPerEquation);
    std::printf("[generated-spectral-system] Hu max = %.17g\n", huMax);
    std::printf("[generated-spectral-system] Hv max = %.17g\n", hvMax);
    std::printf("[generated-spectral-system] system l2 = %.17g max = %.17g\n",
                result.l2Norm, result.maxAbs);

    if (!result.finite || !result.usedGeneratedGridKernels ||
        result.equationCount != 2 || result.pointsPerEquation != grid.size() ||
        result.size() != 2 * grid.size() || huMax > 6e-10 || hvMax > 6e-10 ||
        result.maxAbs > 6e-10) {
      std::fprintf(stderr, "generated spectral system residual mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated spectral system residual runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
