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

struct PunctureParams {
  double eps2 = 0.08;
  double mass = 0.35;
  double px = 0.08;
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

    const auto &systemDesc = tensorium_spectral_residual_systems[0];
    if (!systemDesc.symbol_name ||
        std::strcmp(systemDesc.symbol_name,
                    "SpectralBowenYorkRegularizedPuncture3D") != 0 ||
        systemDesc.unknown_count != 1 || systemDesc.equation_count != 1) {
      throw std::runtime_error(
          "unexpected generated Bowen-York puncture spectral system metadata");
    }

    const PunctureParams params;
    const double rawParams[] = {params.eps2, params.mass, params.px,
                                params.x0,  params.y0,   params.z0};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(5, -1.0, 1.0),
                        SpectralAxis::chebyshevZeros(5, -1.0, 1.0),
                        SpectralAxis::fourierPeriodic(10, 2.0, -1.0));

    std::array<std::vector<double>, 1> solutionFields{
        std::vector<double>(grid.size(), 0.0)};
    const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(rawParams, 6),
                std::span<const std::vector<double>>()},
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
    const double initialMinPsi = minPsi(grid, solutionFields[0], params);

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 6;
    options.residualTolerance = 0.0;
    options.residualRatioTarget = 0.98;
    options.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
    options.denseJacobianMaxUnknowns = 1;
    options.gmresMaxIterations = 384;
    options.gmresTolerance = 3e-3;
    options.gmresRelativeTolerance = 0.0;
    options.gmresPreconditioner =
        SpectralPreconditionerKind::ModalLaplacianShift;
    options.preconditionerLaplacianShift = -0.02;
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
