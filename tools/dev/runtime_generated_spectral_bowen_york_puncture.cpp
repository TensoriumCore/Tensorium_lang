#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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
using tensorium_mlir::runtime::assembleGeneratedSpectralResidualSystem;
using tensorium_mlir::runtime::makeGeneratedSpectralResidualSystem;
using tensorium_mlir::runtime::solveGeneratedSpectralEllipticSystem;

struct PunctureParams {
  double eps2 = 0.16;
  double mass = 0.32;
  double px = 0.04;
  double x0 = 0.12;
  double y0 = -0.08;
  double z0 = 0.0;
};

double psiSingular(double x, double y, double z, const PunctureParams &params) {
  const double dx = x - params.x0;
  const double dy = y - params.y0;
  const double dz = z - params.z0;
  const double r2 = dx * dx + dy * dy + dz * dz + params.eps2;
  return 1.0 + 0.5 * params.mass / std::sqrt(r2);
}

double minPsi(const SpectralGrid3D &grid, std::span<const double> u,
              const PunctureParams &params) {
  double out = std::numeric_limits<double>::infinity();
#pragma omp parallel for collapse(3) reduction(min : out) schedule(static)
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const double z = grid.axis(2).points[k];
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const double y = grid.axis(1).points[j];
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const double x = grid.axis(0).points[i];
        const std::size_t p = grid.index(i, j, k);
        out = std::min(out, psiSingular(x, y, z, params) + u[p]);
      }
    }
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

    const PunctureParams params;
    const double rawParams[] = {params.eps2, params.mass, params.px,
                                params.x0,  params.y0,   params.z0};
    SpectralGrid3D grid(SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0));

    std::array<std::vector<double>, 1> solutionFields{
        std::vector<double>(grid.size(), 0.0)};
    const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(rawParams, 6),
                std::span<const std::vector<double>>()},
        }};
    const auto generatedSystem = makeGeneratedSpectralResidualSystem(
        tensorium_spectral_residual_systems,
        TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
        "SpectralBowenYorkRegularizedPuncture3D", grid,
        tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
        std::span<const SpectralGeneratedResidualSystemEquationInputs>(
            systemInputs.data(), systemInputs.size()),
        1, 1);
    const auto initialResidual = assembleGeneratedSpectralResidualSystem(
        generatedSystem,
        std::span<const std::vector<double>>(solutionFields.data(),
                                             solutionFields.size()));

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 6;
    options.residualTolerance = 0.0;
    options.residualRatioTarget = 0.98;
    options.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
    options.denseJacobianMaxUnknowns = 1;
    options.gmresMaxIterations = 384;
    options.gmresTolerance =
        std::min(3e-3, std::max(1e-7, 0.9 * initialResidual.l2Norm));
    options.gmresRelativeTolerance = 0.0;
    options.gmresPreconditioner =
        SpectralPreconditionerKind::DenseLaplacianShift;
    options.preconditionerLaplacianShift = -0.02;
    options.jvpOptions.relativeStep = 1e-6;
    options.linearPivotTolerance = 1e-13;
    options.preconditionerPivotTolerance = 1e-12;

    const double initialMinPsi = minPsi(grid, solutionFields[0], params);
    const auto run = solveGeneratedSpectralEllipticSystem(
        generatedSystem,
        std::span<std::vector<double>>(solutionFields.data(),
                                       solutionFields.size()),
        options);
    const auto &solveResult = run.solveResult;
    const auto &finalResidual = run.finalResidual;
    const double finalMinPsi = minPsi(grid, solutionFields[0], params);

    std::printf("[generated-spectral-bowen-york-puncture] initial residual l2 = %.17g max = %.17g\n",
                initialResidual.l2Norm, initialResidual.maxAbs);
    std::printf("[generated-spectral-bowen-york-puncture] initial min psi = %.17g\n",
                initialMinPsi);
    std::printf("[generated-spectral-bowen-york-puncture] steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-bowen-york-puncture] final residual l2 = %.17g max = %.17g\n",
                solveResult.finalResidualL2, finalResidual.maxAbs);
    std::printf("[generated-spectral-bowen-york-puncture] residual ratio = %.17g\n",
                solveResult.residualRatio);
    std::printf("[generated-spectral-bowen-york-puncture] linear iterations = %d residual = %.17g\n",
                solveResult.linearIterations, solveResult.finalLinearResidualL2);
    std::printf("[generated-spectral-bowen-york-puncture] final min psi = %.17g\n",
                finalMinPsi);

    if (!initialResidual.finite || !initialResidual.usedGeneratedGridKernels ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !solveResult.usedMatrixFreeGMRES || !solveResult.usedPreconditioner ||
        !finalResidual.finite || !finalResidual.usedGeneratedGridKernels ||
        !(finalMinPsi > 0.0) ||
        !(solveResult.finalResidualL2 < initialResidual.l2Norm) ||
        !(finalResidual.maxAbs < initialResidual.maxAbs)) {
      std::fprintf(stderr,
                   "generated Bowen-York puncture spectral solve mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated Bowen-York puncture spectral runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
