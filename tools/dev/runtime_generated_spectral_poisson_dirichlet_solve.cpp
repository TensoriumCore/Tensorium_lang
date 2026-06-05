#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralBoundaryCondition;
using tensorium_mlir::runtime::SpectralBoundaryConditionKind;
using tensorium_mlir::runtime::SpectralBoundaryFace;
using tensorium_mlir::runtime::SpectralEllipticSolveOptions;
using tensorium_mlir::runtime::SpectralGeneratedResidualSystemEquationInputs;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralPreconditionerKind;
using tensorium_mlir::runtime::kSpectralPi;
using tensorium_mlir::runtime::assembleSpectralResidualSystem;
using tensorium_mlir::runtime::makeSpectralResidualSystemFromDesc;
using tensorium_mlir::runtime::solveSpectralNewton;

double exactU(double x, double y, double z) {
  return (1.0 - x * x) * (1.0 - y * y) * std::cos(2.0 * z);
}

double laplacianExactU(double x, double y, double z) {
  const double xPart = 1.0 - x * x;
  const double yPart = 1.0 - y * y;
  return (-2.0 * yPart - 2.0 * xPart - 4.0 * xPart * yPart) *
         std::cos(2.0 * z);
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
        std::strcmp(systemDesc.symbol_name, "SpectralPoissonDirichlet3D") !=
            0 ||
        systemDesc.unknown_count != 1 || systemDesc.equation_count != 1) {
      throw std::runtime_error(
          "unexpected generated Poisson Dirichlet spectral system metadata");
    }

    SpectralGrid3D grid(SpectralAxis::chebyshevLobatto(6, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(6, -1.0, 1.0),
                        SpectralAxis::fourierPeriodic(10, 2.0 * kSpectralPi,
                                                      0.0));

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
          expected[p] = exactU(x, y, z);
          source[p] = -laplacianExactU(x, y, z);
          solutionFields[0][p] = 0.08 * (1.0 - x * x) *
                                 (1.0 - y * y) * std::cos(z);
        }
      }
    }

    const std::array<std::vector<double>, 1> auxiliaryFields{source};
    const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(),
                std::span<const std::vector<double>>(auxiliaryFields.data(),
                                                     auxiliaryFields.size())},
        }};
    auto generatedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, grid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
        std::span<const SpectralGeneratedResidualSystemEquationInputs>(
            systemInputs.data(), systemInputs.size()));

    const std::array<SpectralBoundaryCondition, 4> boundaryConditions{{
        SpectralBoundaryCondition{
            SpectralBoundaryFace::LowerX1,
            SpectralBoundaryConditionKind::Dirichlet, 1.0, 0.0, 0.0},
        SpectralBoundaryCondition{
            SpectralBoundaryFace::UpperX1,
            SpectralBoundaryConditionKind::Dirichlet, 1.0, 0.0, 0.0},
        SpectralBoundaryCondition{
            SpectralBoundaryFace::LowerX2,
            SpectralBoundaryConditionKind::Dirichlet, 1.0, 0.0, 0.0},
        SpectralBoundaryCondition{
            SpectralBoundaryFace::UpperX2,
            SpectralBoundaryConditionKind::Dirichlet, 1.0, 0.0, 0.0},
    }};
    generatedSystem.equations[0].problem.boundaryConditions =
        std::span<const SpectralBoundaryCondition>(boundaryConditions.data(),
                                                   boundaryConditions.size());
    const auto system = generatedSystem.view();

    const auto initialResidual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solutionFields.data(),
                                                     solutionFields.size()));

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 4;
    options.residualTolerance = 2e-10;
    options.residualRatioTarget = 1e-12;
    options.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
    options.denseJacobianMaxUnknowns = 1;
    options.gmresMaxIterations = 256;
    options.gmresTolerance = 2e-10;
    options.gmresRelativeTolerance = 1e-12;
    options.gmresPreconditioner =
        SpectralPreconditionerKind::DenseLaplacianShift;
    options.preconditionerLaplacianShift = 0.0;
    options.jvpOptions.relativeStep = 1e-7;
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

    std::printf("[generated-spectral-poisson-dirichlet] initial residual l2 = %.17g max = %.17g\n",
                initialResidual.l2Norm, initialResidual.maxAbs);
    std::printf("[generated-spectral-poisson-dirichlet] steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-poisson-dirichlet] final residual l2 = %.17g max = %.17g\n",
                solveResult.finalResidualL2, finalResidual.maxAbs);
    std::printf("[generated-spectral-poisson-dirichlet] linear iterations = %d residual = %.17g\n",
                solveResult.linearIterations, solveResult.finalLinearResidualL2);
    std::printf("[generated-spectral-poisson-dirichlet] solution max error = %.17g\n",
                solutionError);

    if (!initialResidual.usedGeneratedGridKernels ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !solveResult.usedMatrixFreeGMRES || !solveResult.usedPreconditioner ||
        !finalResidual.usedGeneratedGridKernels ||
        solveResult.finalResidualL2 > 2e-10 ||
        finalResidual.maxAbs > 2e-9 || solutionError > 2e-7) {
      std::fprintf(stderr,
                   "generated Poisson Dirichlet spectral solve mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated Poisson Dirichlet spectral runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
