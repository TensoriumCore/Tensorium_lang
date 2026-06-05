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
using tensorium_mlir::runtime::SpectralGeneratedResidualSystem;
using tensorium_mlir::runtime::SpectralGeneratedResidualSystemEquationInputs;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::assembleSpectralResidualSystem;
using tensorium_mlir::runtime::makeSpectralResidualSystemFromDesc;

struct ConvergenceResult {
  std::size_t nCheb = 0;
  std::size_t nFourier = 0;
  double exactResidualMax = std::numeric_limits<double>::infinity();
  double exactResidualL2 = std::numeric_limits<double>::infinity();
};

double exactU(double x, double y, double z) {
  return 0.04 * std::exp(0.35 * x - 0.2 * y) + 0.025 * std::cos(3.0 * z);
}

double lapU(double x, double y, double z) {
  const double expPart = 0.04 * std::exp(0.35 * x - 0.2 * y);
  return (0.35 * 0.35 + 0.2 * 0.2) * expPart -
         9.0 * 0.025 * std::cos(3.0 * z);
}

double extrinsicA2(double x, double y, double z) {
  return 0.35 * (1.1 + 0.15 * std::cos(x) + 0.08 * y * y +
                 0.04 * std::sin(z));
}

double seventh(double value) {
  const double value2 = value * value;
  const double value4 = value2 * value2;
  return value4 * value2 * value;
}

ConvergenceResult runCase(const tensorium_spectral_residual_system_desc &desc,
                          std::size_t nCheb, std::size_t nFourier) {
  const double background = 1.0;
  const double params[] = {background};
  SpectralGrid3D grid(SpectralAxis::chebyshevZeros(nCheb),
                      SpectralAxis::chebyshevZeros(nCheb),
                      SpectralAxis::fourierPeriodic(nFourier));

  std::vector<double> expected(grid.size(), 0.0);
  std::vector<double> a2(grid.size(), 0.0);
  std::vector<double> source(grid.size(), 0.0);
  std::array<std::vector<double>, 1> exactFields{
      std::vector<double>(grid.size(), 0.0)};

  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const double z = grid.axis(2).points[k];
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const double y = grid.axis(1).points[j];
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const double x = grid.axis(0).points[i];
        const std::size_t p = grid.index(i, j, k);
        expected[p] = exactU(x, y, z);
        exactFields[0][p] = expected[p];
        const double psi = background + expected[p];
        a2[p] = extrinsicA2(x, y, z);
        source[p] = -(lapU(x, y, z) + 0.125 * a2[p] / seventh(psi));
      }
    }
  }

  const std::array<std::vector<double>, 2> auxiliaryFields{a2, source};
  const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
      systemInputs{{
          SpectralGeneratedResidualSystemEquationInputs{
              std::span<const double>(params, 1),
              std::span<const std::vector<double>>(auxiliaryFields.data(),
                                                   auxiliaryFields.size())},
      }};
  const SpectralGeneratedResidualSystem generatedSystem =
      makeSpectralResidualSystemFromDesc(
          desc, grid, tensorium_spectral_residual_kernels,
          TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
          tensorium_spectral_residual_grid_kernels,
          TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
          std::span<const SpectralGeneratedResidualSystemEquationInputs>(
              systemInputs.data(), systemInputs.size()));
  const auto system = generatedSystem.view();

  const auto exactResidual = assembleSpectralResidualSystem(
      system, std::span<const std::vector<double>>(exactFields.data(),
                                                   exactFields.size()));

  ConvergenceResult out;
  out.nCheb = nCheb;
  out.nFourier = nFourier;
  out.exactResidualMax = exactResidual.maxAbs;
  out.exactResidualL2 = exactResidual.l2Norm;

  if (!exactResidual.finite || !exactResidual.usedGeneratedGridKernels) {
    std::printf(
        "[generated-spectral-lichnerowicz-convergence] failed N=(%zu,%zu,%zu) exact l2 = %.17g exact max = %.17g\n",
        out.nCheb, out.nCheb, out.nFourier, out.exactResidualL2,
        out.exactResidualMax);
    throw std::runtime_error("Lichnerowicz exact residual evaluation failed");
  }
  return out;
}

} // namespace

int main() {
  try {
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral point kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral grid kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT >= 1,
                  "expected at least one generated spectral residual system");

    const auto &systemDesc = tensorium_spectral_residual_systems[0];
    if (!systemDesc.symbol_name ||
        std::strcmp(systemDesc.symbol_name,
                    "SpectralLichnerowiczManufactured3D") != 0 ||
        systemDesc.unknown_count != 1 || systemDesc.equation_count != 1) {
      throw std::runtime_error(
          "unexpected generated Lichnerowicz spectral system metadata");
    }

    const std::array<ConvergenceResult, 3> results{
        runCase(systemDesc, 4, 8),
        runCase(systemDesc, 6, 12),
        runCase(systemDesc, 8, 16),
    };

    for (const auto &result : results) {
      std::printf(
          "[generated-spectral-lichnerowicz-convergence] N=(%zu,%zu,%zu) exact l2 = %.17g exact max = %.17g\n",
          result.nCheb, result.nCheb, result.nFourier,
          result.exactResidualL2, result.exactResidualMax);
    }

    if (!(results[1].exactResidualMax < 0.35 * results[0].exactResidualMax &&
          results[2].exactResidualMax < 0.35 * results[1].exactResidualMax)) {
      std::fprintf(stderr,
                   "Lichnerowicz exact residual did not converge spectrally\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated Lichnerowicz convergence runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
