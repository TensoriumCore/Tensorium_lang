#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
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
using tensorium_mlir::runtime::makeGeneratedSpectralResidualSystem;
using tensorium_mlir::runtime::solveGeneratedSpectralEllipticSystem;

double fifth(double value) {
  const double value2 = value * value;
  return value2 * value2 * value;
}

double seventh(double value) {
  const double value2 = value * value;
  const double value4 = value2 * value2;
  return value4 * value2 * value;
}

double boundaryBump1D(double value) {
  const double oneMinusSquare = 1.0 - value * value;
  return oneMinusSquare * oneMinusSquare;
}

double boundaryBump1DLaplacian(double value) {
  return -4.0 + 12.0 * value * value;
}

double boundaryBump(double x, double y, double z) {
  return boundaryBump1D(x) * boundaryBump1D(y) * boundaryBump1D(z);
}

double boundaryBumpLaplacian(double x, double y, double z) {
  const double bx = boundaryBump1D(x);
  const double by = boundaryBump1D(y);
  const double bz = boundaryBump1D(z);
  return boundaryBump1DLaplacian(x) * by * bz +
         bx * boundaryBump1DLaplacian(y) * bz +
         bx * by * boundaryBump1DLaplacian(z);
}

double exactPsi(double x, double y, double z) {
  return 1.0 + 0.035 * boundaryBump(x, y, z);
}

double lapExactPsi(double x, double y, double z) {
  return 0.035 * boundaryBumpLaplacian(x, y, z);
}

double freeA2(double x, double y, double z) {
  return 0.07 * (1.0 + 0.15 * x + 0.1 * y * y + 0.05 * std::cos(z));
}

double freeK(double x, double y, double) { return 0.12 + 0.04 * x * y; }

double freeRho(double x, double, double z) {
  return 0.02 * (1.0 + 0.1 * x * x + 0.2 * z);
}

double freeRbar(double x, double y, double z) {
  return 0.03 * (x - y + 0.5 * z * z);
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

    const double matterCoeff = 0.4;
    const double modifiedCoeff = 0.9;
    const double params[] = {matterCoeff, modifiedCoeff};

    SpectralGrid3D grid(SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0));

    std::vector<double> expected(grid.size(), 0.0);
    std::vector<double> a2(grid.size(), 0.0);
    std::vector<double> kTrace(grid.size(), 0.0);
    std::vector<double> rbar(grid.size(), 0.0);
    std::vector<double> modSource(grid.size(), 0.0);
    std::vector<double> rho(grid.size(), 0.0);
    std::array<std::vector<double>, 1> solutionFields{
        std::vector<double>(grid.size(), 0.0)};

    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          a2[p] = freeA2(x, y, z);
          kTrace[p] = freeK(x, y, z);
          rbar[p] = freeRbar(x, y, z);
          rho[p] = freeRho(x, y, z);

          expected[p] = exactPsi(x, y, z);
          const double psi = expected[p];
          const double psi5 = fifth(psi);
          const double base =
              lapExactPsi(x, y, z) - 0.125 * rbar[p] * psi +
              0.125 * a2[p] / seventh(psi) -
              0.08333333333333333 * kTrace[p] * kTrace[p] * psi5 +
              matterCoeff * rho[p] * psi5;
          modSource[p] = -base / modifiedCoeff;

          solutionFields[0][p] = expected[p] - 0.008 * boundaryBump(x, y, z);
        }
      }
    }

    const std::array<std::vector<double>, 5> auxiliaryFields{
        a2, kTrace, rbar, modSource, rho};
    const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(params, 2),
                std::span<const std::vector<double>>(auxiliaryFields.data(),
                                                     auxiliaryFields.size())},
        }};
    const auto generatedSystem = makeGeneratedSpectralResidualSystem(
        tensorium_spectral_residual_systems,
        TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
        "SpectralYorkLichnerowiczConstraint3D", grid,
        tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
        std::span<const SpectralGeneratedResidualSystemEquationInputs>(
            systemInputs.data(), systemInputs.size()),
        1, 1);

    if (generatedSystem.equations.size() != 1 ||
        generatedSystem.equations[0].residualName != "H" ||
        generatedSystem.equations[0].unknownIndex != 0) {
      throw std::runtime_error("unexpected York Lichnerowicz equation mapping");
    }

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 8;
    options.residualTolerance = 8e-10;
    options.residualRatioTarget = 1e-12;
    options.linearSolver = SpectralLinearSolveKind::DenseJacobian;
    options.denseJacobianMaxUnknowns = 256;
    options.gmresMaxIterations = 256;
    options.gmresTolerance = 8e-11;
    options.gmresRelativeTolerance = 1e-13;
    options.gmresPreconditioner =
        SpectralPreconditionerKind::ModalLaplacianShift;
    options.preconditionerLaplacianShift = -0.02;
    options.jvpOptions.relativeStep = 1e-6;
    options.linearPivotTolerance = 1e-13;
    options.preconditionerPivotTolerance = 1e-12;

    const auto run = solveGeneratedSpectralEllipticSystem(
        generatedSystem,
        std::span<std::vector<double>>(solutionFields.data(),
                                       solutionFields.size()),
        options);
    const auto &initialResidual = run.initialResidual;
    const auto &solveResult = run.solveResult;
    const auto &finalResidual = run.finalResidual;

    std::vector<double> errors(grid.size(), 0.0);
    for (std::size_t p = 0; p < grid.size(); ++p)
      errors[p] = solutionFields[0][p] - expected[p];
    const double solutionError = maxAbs(errors);

    std::printf("[generated-spectral-york-lichnerowicz] initial residual l2 = %.17g\n",
                initialResidual.l2Norm);
    std::printf("[generated-spectral-york-lichnerowicz] steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-york-lichnerowicz] final residual l2 = %.17g\n",
                solveResult.finalResidualL2);
    std::printf("[generated-spectral-york-lichnerowicz] final residual max = %.17g\n",
                finalResidual.maxAbs);
    std::printf("[generated-spectral-york-lichnerowicz] linear iterations = %d\n",
                solveResult.linearIterations);
    std::printf("[generated-spectral-york-lichnerowicz] linear residual l2 = %.17g\n",
                solveResult.finalLinearResidualL2);
    std::printf("[generated-spectral-york-lichnerowicz] used preconditioner = %d\n",
                solveResult.usedPreconditioner ? 1 : 0);
    std::printf("[generated-spectral-york-lichnerowicz] solution max error = %.17g\n",
                solutionError);

    if (!initialResidual.usedGeneratedGridKernels ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !finalResidual.usedGeneratedGridKernels ||
        solveResult.finalResidualL2 > 1e-8 || finalResidual.maxAbs > 4e-8 ||
        solutionError > 2e-7) {
      std::fprintf(stderr,
                   "generated York Lichnerowicz spectral solve mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated York Lichnerowicz spectral runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
