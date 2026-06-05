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

struct ContinuationStage {
  const char *name = "";
  PunctureParams params{};
  double residualTolerance = 0.0;
  double requiredRatio = 1.0;
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

SpectralEllipticSolveOptions makeOptions(double ratioTarget,
                                         double residualTolerance,
                                         double initialResidualL2) {
  SpectralEllipticSolveOptions options;
  options.maxNewtonSteps = 8;
  options.residualTolerance = residualTolerance;
  options.residualRatioTarget = ratioTarget;
  options.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
  options.denseJacobianMaxUnknowns = 1;
  options.gmresMaxIterations = 384;
  options.gmresTolerance =
      std::min(3e-3, std::max(1e-7, 0.9 * initialResidualL2));
  options.gmresRelativeTolerance = 0.0;
  options.gmresPreconditioner =
      SpectralPreconditionerKind::ModalLaplacianShift;
  options.preconditionerLaplacianShift = -0.02;
  options.jvpOptions.relativeStep = 1e-6;
  options.linearPivotTolerance = 1e-13;
  options.preconditionerPivotTolerance = 1e-12;
  return options;
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

    const std::array<ContinuationStage, 2> stages{{
        ContinuationStage{"wide-easy",
                          PunctureParams{0.22, 0.30, 0.02, 0.12, -0.08, 0.0},
                          2e-4,
                          0.999},
        ContinuationStage{"wide",
                          PunctureParams{0.16, 0.32, 0.04, 0.12, -0.08, 0.0},
                          0.0,
                          0.999},
    }};

    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(5, -1.0, 1.0),
                        SpectralAxis::chebyshevZeros(5, -1.0, 1.0),
                        SpectralAxis::fourierPeriodic(10, 2.0, -1.0));
    std::array<std::vector<double>, 1> solutionFields{
        std::vector<double>(grid.size(), 0.0)};

    double firstResidualL2 = 0.0;
    double lastResidualL2 = 0.0;
    double lastResidualMax = 0.0;
    double lastMinPsi = 0.0;
    bool usedGeneratedGridKernel = false;
    bool usedGMRES = false;
    bool usedPreconditioner = false;

    for (std::size_t stageIndex = 0; stageIndex < stages.size();
         ++stageIndex) {
      const auto &stage = stages[stageIndex];
      const double rawParams[] = {stage.params.eps2, stage.params.mass,
                                  stage.params.px,   stage.params.x0,
                                  stage.params.y0,   stage.params.z0};
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
      if (stageIndex == 0)
        firstResidualL2 = initialResidual.l2Norm;

      const auto options =
          makeOptions(stage.requiredRatio, stage.residualTolerance,
                      initialResidual.l2Norm);
      const auto solveResult = solveSpectralNewton(
          system, std::span<std::vector<double>>(solutionFields.data(),
                                                 solutionFields.size()),
          options);
      const auto finalResidual = assembleSpectralResidualSystem(
          system, std::span<const std::vector<double>>(solutionFields.data(),
                                                       solutionFields.size()));
      lastResidualL2 = solveResult.finalResidualL2;
      lastResidualMax = finalResidual.maxAbs;
      lastMinPsi = minPsi(grid, solutionFields[0], stage.params);
      usedGeneratedGridKernel =
          usedGeneratedGridKernel || initialResidual.usedGeneratedGridKernels ||
          solveResult.usedGeneratedGridKernel ||
          finalResidual.usedGeneratedGridKernels;
      usedGMRES = usedGMRES || solveResult.usedMatrixFreeGMRES;
      usedPreconditioner =
          usedPreconditioner || solveResult.usedPreconditioner;

      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s initial l2 = %.17g max = %.17g\n",
                  stage.name, initialResidual.l2Norm, initialResidual.maxAbs);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s steps = %d status = %d\n",
                  stage.name, solveResult.steps,
                  static_cast<int>(solveResult.status));
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s final l2 = %.17g max = %.17g ratio = %.17g\n",
                  stage.name, solveResult.finalResidualL2, finalResidual.maxAbs,
                  solveResult.residualRatio);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s linear iterations = %d residual = %.17g min psi = %.17g\n",
                  stage.name, solveResult.linearIterations,
                  solveResult.finalLinearResidualL2, lastMinPsi);

      const bool solvedByInitialTolerance =
          solveResult.converged() && solveResult.steps == 0 &&
          stage.residualTolerance > 0.0 &&
          initialResidual.l2Norm <= stage.residualTolerance;
      const bool solvedByNewton =
          solveResult.converged() && solveResult.usedGeneratedGridKernel &&
          solveResult.usedMatrixFreeGMRES && solveResult.usedPreconditioner &&
          solveResult.finalResidualL2 < initialResidual.l2Norm;
      const bool madeLinearProgress =
          solveResult.usedMatrixFreeGMRES && solveResult.usedPreconditioner &&
          solveResult.linearIterations > 0 &&
          solveResult.finalLinearResidualL2 < initialResidual.l2Norm;
      if (!initialResidual.finite || !initialResidual.usedGeneratedGridKernels ||
          !(solvedByInitialTolerance || solvedByNewton || madeLinearProgress) ||
          !finalResidual.finite || !finalResidual.usedGeneratedGridKernels ||
          !(lastMinPsi > 0.0)) {
        std::fprintf(stderr,
                     "generated Bowen-York puncture continuation stage failed\n");
        return 3;
      }
    }

    std::printf("[generated-spectral-bowen-york-puncture-continuation] first residual l2 = %.17g\n",
                firstResidualL2);
    std::printf("[generated-spectral-bowen-york-puncture-continuation] target residual l2 = %.17g max = %.17g\n",
                lastResidualL2, lastResidualMax);

    if (!usedGeneratedGridKernel || !usedGMRES || !usedPreconditioner ||
        !(lastResidualL2 > 0.0) || !(lastMinPsi > 0.0)) {
      std::fprintf(stderr,
                   "generated Bowen-York puncture continuation mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated Bowen-York puncture continuation runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
