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
using tensorium_mlir::runtime::SpectralEllipticSolveOptions;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralLinearPreconditioner;
using tensorium_mlir::runtime::SpectralPreconditionerKind;
using tensorium_mlir::runtime::SpectralResidualProblem;
using tensorium_mlir::runtime::applySpectralPreconditioner;
using tensorium_mlir::runtime::assembleSpectralResidual;
using tensorium_mlir::runtime::buildSpectralScalarPreconditioner;
using tensorium_mlir::runtime::solveSpectralNewton;
using tensorium_mlir::runtime::spectralResidualGridKernelFromDesc;
using tensorium_mlir::runtime::spectralResidualKernelFromDesc;

double manufacturedU(double x, double y, double z) {
  return (4.0 * x * x * x - 3.0 * x) + y * y + 0.25 * std::cos(2.0 * z);
}

double manufacturedLaplacian(double x, double, double z) {
  return 24.0 * x + 2.0 - std::cos(2.0 * z);
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

    const double alpha = 0.75;
    const double params[] = {alpha};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::fourierPeriodic(6));

    std::vector<double> expected(grid.size(), 0.0);
    std::vector<double> source(grid.size(), 0.0);
    std::vector<double> solution(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          expected[p] = manufacturedU(x, y, z);
          source[p] =
              -(manufacturedLaplacian(x, y, z) + alpha * expected[p]);
        }
      }
    }

    const std::array<std::vector<double>, 1> auxiliaryFields{source};
    SpectralResidualProblem problem{
        &grid,
        pointKernel,
        std::span<const double>(params, 1),
        std::span<const std::vector<double>>(auxiliaryFields.data(),
                                             auxiliaryFields.size())};
    problem.gridKernel = gridKernel;

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 4;
    options.residualTolerance = 5e-11;
    options.residualRatioTarget = 1e-12;
    options.linearSolver = SpectralLinearSolveKind::Auto;
    options.denseJacobianMaxUnknowns = 1;
    options.gmresMaxIterations = 128;
    options.gmresTolerance = 1e-12;
    options.gmresRelativeTolerance = 1e-13;
    options.gmresPreconditioner =
        SpectralPreconditionerKind::ModalLaplacianShift;
    options.preconditionerLaplacianShift = alpha;
    options.jvpOptions.relativeStep = 1e-6;
    options.linearPivotTolerance = 1e-13;
    options.preconditionerPivotTolerance = 1e-13;

    std::vector<double> modalProbe(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        for (std::size_t i = 0; i < grid.n1(); ++i)
          modalProbe[grid.index(i, j, k)] = std::cos(2.0 * z);
      }
    }
    std::vector<double> denseApplied = modalProbe;
    std::vector<double> modalApplied = modalProbe;
    SpectralEllipticSolveOptions denseOptions = options;
    denseOptions.gmresPreconditioner =
        SpectralPreconditionerKind::DenseLaplacianShift;
    SpectralLinearPreconditioner densePreconditioner;
    SpectralLinearPreconditioner modalPreconditioner;
    if (!buildSpectralScalarPreconditioner(problem, solution, denseOptions,
                                           densePreconditioner) ||
        !buildSpectralScalarPreconditioner(problem, solution, options,
                                           modalPreconditioner) ||
        !applySpectralPreconditioner(
            densePreconditioner, denseApplied,
            denseOptions.preconditionerPivotTolerance) ||
        !applySpectralPreconditioner(
            modalPreconditioner, modalApplied,
            options.preconditionerPivotTolerance)) {
      std::fprintf(stderr, "spectral preconditioner comparison setup failed\n");
      return 3;
    }
    std::vector<double> preconditionerDelta(grid.size(), 0.0);
    for (std::size_t p = 0; p < grid.size(); ++p)
      preconditionerDelta[p] = modalApplied[p] - denseApplied[p];
    const double modalDenseDelta = maxAbs(preconditionerDelta);
    std::printf(
        "[generated-spectral-newton] modal/dense preconditioner delta = %.17g\n",
        modalDenseDelta);
    if (modalDenseDelta > 5e-11) {
      std::fprintf(stderr, "modal spectral preconditioner differs from dense\n");
      return 4;
    }

    const auto solveResult = solveSpectralNewton(problem, solution, options);
    const auto finalResidual = assembleSpectralResidual(problem, solution);

    std::vector<double> errors(grid.size(), 0.0);
    for (std::size_t p = 0; p < grid.size(); ++p)
      errors[p] = solution[p] - expected[p];

    const double solutionError = maxAbs(errors);
    std::printf("[generated-spectral-newton] steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-newton] initial residual l2 = %.17g\n",
                solveResult.initialResidualL2);
    std::printf("[generated-spectral-newton] final residual l2 = %.17g\n",
                solveResult.finalResidualL2);
    std::printf("[generated-spectral-newton] linear iterations = %d\n",
                solveResult.linearIterations);
    std::printf("[generated-spectral-newton] linear residual l2 = %.17g\n",
                solveResult.finalLinearResidualL2);
    std::printf("[generated-spectral-newton] used preconditioner = %d\n",
                solveResult.usedPreconditioner ? 1 : 0);
    std::printf("[generated-spectral-newton] final residual max = %.17g\n",
                finalResidual.maxAbs);
    std::printf("[generated-spectral-newton] solution max error = %.17g\n",
                solutionError);

    if (!solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !solveResult.usedMatrixFreeGMRES || !solveResult.usedPreconditioner ||
        !finalResidual.usedGeneratedGridKernel ||
        solveResult.finalResidualL2 > 2e-10 || finalResidual.maxAbs > 2e-9 ||
        solutionError > 3e-8) {
      std::fprintf(stderr, "generated spectral Newton solve failed\n");
      return 5;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "generated spectral Newton runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
