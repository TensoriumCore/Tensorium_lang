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
using tensorium_mlir::runtime::SpectralEllipticSolveOptions;
using tensorium_mlir::runtime::SpectralGeneratedResidualSystemEquationInputs;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralPreconditionerKind;
using tensorium_mlir::runtime::assembleSpectralResidualSystem;
using tensorium_mlir::runtime::makeSpectralResidualSystemFromDesc;
using tensorium_mlir::runtime::solveSpectralNewton;

double exactPsi(double x, double y, double z) {
  return 1.0 + 0.08 * (4.0 * x * x * x - 3.0 * x) + 0.05 * y * y +
         0.03 * std::cos(2.0 * z);
}

double lapPsi(double x, double, double z) {
  return 1.92 * x + 0.1 - 0.12 * std::cos(2.0 * z);
}

double fifth(double value) {
  const double value2 = value * value;
  return value2 * value2 * value;
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
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral point kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral grid kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT >= 1,
                  "expected at least one generated spectral residual system");

    const auto &systemDesc = tensorium_spectral_residual_systems[0];
    if (!systemDesc.symbol_name ||
        std::strcmp(systemDesc.symbol_name,
                    "SpectralHamiltonianToyNonlinear3D") != 0 ||
        systemDesc.unknown_count != 1 || systemDesc.equation_count != 1) {
      throw std::runtime_error(
          "unexpected generated nonlinear spectral system metadata");
    }

    const double alpha = 0.08;
    const double mass = 0.75;
    const double params[] = {alpha, mass};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::fourierPeriodic(8));

    std::vector<double> expected(grid.size(), 0.0);
    std::vector<double> source(grid.size(), 0.0);
    std::array<std::vector<double>, 1> solutionFields{
        std::vector<double>(grid.size(), 0.0)};
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          expected[p] = exactPsi(x, y, z);
          solutionFields[0][p] =
              expected[p] + 0.02 * (2.0 * x * x - 1.0) -
              0.01 * std::cos(2.0 * z);
          source[p] = -(lapPsi(x, y, z) + mass * expected[p] +
                        alpha * fifth(expected[p]));
        }
      }
    }

    const std::array<std::vector<double>, 1> auxiliaryFields{source};
    const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(params, 2),
                std::span<const std::vector<double>>(auxiliaryFields.data(),
                                                     auxiliaryFields.size())},
        }};
    const auto generatedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, grid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
        std::span<const SpectralGeneratedResidualSystemEquationInputs>(
            systemInputs.data(), systemInputs.size()));
    const auto system = generatedSystem.view();

    const auto initialResidual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solutionFields.data(),
                                                     solutionFields.size()));

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 8;
    options.residualTolerance = 4e-10;
    options.residualRatioTarget = 1e-12;
    options.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
    options.denseJacobianMaxUnknowns = 1;
    options.gmresMaxIterations = 256;
    options.gmresTolerance = 8e-12;
    options.gmresRelativeTolerance = 1e-13;
    options.gmresPreconditioner =
        SpectralPreconditionerKind::ModalLaplacianShift;
    options.preconditionerLaplacianShift = mass;
    options.jvpOptions.relativeStep = 1e-6;
    options.linearPivotTolerance = 1e-13;
    options.preconditionerPivotTolerance = 1e-12;

    const auto solveResult = solveSpectralNewton(
        system, std::span<std::vector<double>>(solutionFields.data(),
                                               solutionFields.size()),
        options);
    const auto finalResidual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solutionFields.data(),
                                                     solutionFields.size()));

    std::vector<double> errors(grid.size(), 0.0);
    for (std::size_t p = 0; p < grid.size(); ++p)
      errors[p] = solutionFields[0][p] - expected[p];
    const double solutionError = maxAbs(errors);

    std::printf("[generated-spectral-nonlinear] initial residual l2 = %.17g\n",
                initialResidual.l2Norm);
    std::printf("[generated-spectral-nonlinear] steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-nonlinear] final residual l2 = %.17g\n",
                solveResult.finalResidualL2);
    std::printf("[generated-spectral-nonlinear] final residual max = %.17g\n",
                finalResidual.maxAbs);
    std::printf("[generated-spectral-nonlinear] linear iterations = %d\n",
                solveResult.linearIterations);
    std::printf("[generated-spectral-nonlinear] linear residual l2 = %.17g\n",
                solveResult.finalLinearResidualL2);
    std::printf("[generated-spectral-nonlinear] used preconditioner = %d\n",
                solveResult.usedPreconditioner ? 1 : 0);
    std::printf("[generated-spectral-nonlinear] solution max error = %.17g\n",
                solutionError);

    if (!initialResidual.usedGeneratedGridKernels ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !solveResult.usedMatrixFreeGMRES || !solveResult.usedPreconditioner ||
        !finalResidual.usedGeneratedGridKernels ||
        solveResult.finalResidualL2 > 8e-10 ||
        finalResidual.maxAbs > 6e-9 || solutionError > 5e-8) {
      std::fprintf(stderr, "generated nonlinear spectral solve mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated nonlinear spectral runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
