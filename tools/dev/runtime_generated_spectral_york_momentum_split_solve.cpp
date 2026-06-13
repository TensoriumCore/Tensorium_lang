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
using tensorium_mlir::runtime::assembleSpectralResidualSystem;
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

double bump1D(double value) {
  const double q = 1.0 - value * value;
  return q * q;
}

double bump1DLaplacian(double value) { return -4.0 + 12.0 * value * value; }

double bump(double x, double y, double z) {
  return bump1D(x) * bump1D(y) * bump1D(z);
}

double bumpLaplacian(double x, double y, double z) {
  const double bx = bump1D(x);
  const double by = bump1D(y);
  const double bz = bump1D(z);
  return bump1DLaplacian(x) * by * bz +
         bx * bump1DLaplacian(y) * bz +
         bx * by * bump1DLaplacian(z);
}

double exactPsi(double x, double y, double z) {
  return 1.0 + 0.03 * bump(x, y, z);
}

double lapExactPsi(double x, double y, double z) {
  return 0.03 * bumpLaplacian(x, y, z);
}

double exactW1(double x, double y, double z) {
  return 0.018 * bump(x, y, z);
}

double exactW2(double x, double y, double z) {
  return -0.014 * bump(x, y, z);
}

double exactW3(double x, double y, double z) {
  return 0.011 * bump(x, y, z);
}

double lapExactW1(double x, double y, double z) {
  return 0.018 * bumpLaplacian(x, y, z);
}

double lapExactW2(double x, double y, double z) {
  return -0.014 * bumpLaplacian(x, y, z);
}

double lapExactW3(double x, double y, double z) {
  return 0.011 * bumpLaplacian(x, y, z);
}

double freeA2(double x, double y, double z) {
  return 0.06 * (1.0 + 0.1 * x + 0.08 * y * y + 0.03 * z);
}

double freeK(double x, double y, double) { return 0.1 + 0.03 * x * y; }

double freeRho(double x, double, double z) {
  return 0.015 * (1.0 + 0.08 * x * x + 0.1 * z);
}

double freeRbar(double x, double y, double z) {
  return 0.025 * (x - 0.5 * y + 0.25 * z * z);
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
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 4,
                  "expected four generated spectral point kernels");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 4,
                  "expected four generated spectral grid kernels");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT >= 1,
                  "expected at least one generated spectral residual system");

    const double matterCoeff = 0.35;
    const double modifiedCoeff = 0.8;
    const double vectorCoupling = 0.22;
    const double vectorMass = 0.4;
    const double momentumPsiCoupling = 0.18;
    const double hParams[] = {matterCoeff, modifiedCoeff, vectorCoupling};
    const double mParams[] = {momentumPsiCoupling, vectorMass};

    SpectralGrid3D grid(SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0));

    std::array<std::vector<double>, 4> exactFields{
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0)};
    std::array<std::vector<double>, 4> solutionFields{
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0)};

    std::vector<double> a2(grid.size(), 0.0);
    std::vector<double> kTrace(grid.size(), 0.0);
    std::vector<double> rbar(grid.size(), 0.0);
    std::vector<double> modSource(grid.size(), 0.0);
    std::vector<double> rho(grid.size(), 0.0);
    std::vector<double> j1(grid.size(), 0.0);
    std::vector<double> j2(grid.size(), 0.0);
    std::vector<double> j3(grid.size(), 0.0);

    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          exactFields[0][p] = exactPsi(x, y, z);
          exactFields[1][p] = exactW1(x, y, z);
          exactFields[2][p] = exactW2(x, y, z);
          exactFields[3][p] = exactW3(x, y, z);
          a2[p] = freeA2(x, y, z);
          kTrace[p] = freeK(x, y, z);
          rbar[p] = freeRbar(x, y, z);
          rho[p] = freeRho(x, y, z);

          const double psi = exactFields[0][p];
          const double psi5 = fifth(psi);
          const double wsum =
              exactFields[1][p] + exactFields[2][p] + exactFields[3][p];
          const double hBase =
              lapExactPsi(x, y, z) - 0.125 * rbar[p] * psi +
              0.125 * a2[p] / seventh(psi) -
              0.08333333333333333 * kTrace[p] * kTrace[p] * psi5 +
              matterCoeff * rho[p] * psi5 + vectorCoupling * wsum;
          modSource[p] = -hBase / modifiedCoeff;

          j1[p] = -(lapExactW1(x, y, z) + vectorMass * exactFields[1][p] +
                    momentumPsiCoupling * (psi - 1.0));
          j2[p] = -(lapExactW2(x, y, z) + vectorMass * exactFields[2][p] +
                    momentumPsiCoupling * (psi - 1.0));
          j3[p] = -(lapExactW3(x, y, z) + vectorMass * exactFields[3][p] +
                    momentumPsiCoupling * (psi - 1.0));

          const double perturb = 0.004 * bump(x, y, z);
          solutionFields[0][p] = exactFields[0][p] - perturb;
          solutionFields[1][p] = exactFields[1][p] + 0.7 * perturb;
          solutionFields[2][p] = exactFields[2][p] - 0.5 * perturb;
          solutionFields[3][p] = exactFields[3][p] + 0.3 * perturb;
        }
      }
    }

    const std::array<std::vector<double>, 8> hAuxiliaryFields{
        a2, kTrace, rbar, exactFields[1], exactFields[2],
        exactFields[3], modSource, rho};
    const std::array<std::vector<double>, 2> m1AuxiliaryFields{j1,
                                                               exactFields[0]};
    const std::array<std::vector<double>, 2> m2AuxiliaryFields{j2,
                                                               exactFields[0]};
    const std::array<std::vector<double>, 2> m3AuxiliaryFields{j3,
                                                               exactFields[0]};

    const std::array<SpectralGeneratedResidualSystemEquationInputs, 4>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(hParams, 3),
                std::span<const std::vector<double>>(hAuxiliaryFields.data(),
                                                     hAuxiliaryFields.size())},
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(mParams, 2),
                std::span<const std::vector<double>>(m1AuxiliaryFields.data(),
                                                     m1AuxiliaryFields.size())},
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(mParams, 2),
                std::span<const std::vector<double>>(m2AuxiliaryFields.data(),
                                                     m2AuxiliaryFields.size())},
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(mParams, 2),
                std::span<const std::vector<double>>(m3AuxiliaryFields.data(),
                                                     m3AuxiliaryFields.size())},
        }};

    const auto generatedSystem = makeGeneratedSpectralResidualSystem(
        tensorium_spectral_residual_systems,
        TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
        "SpectralYorkMomentumSplitConstraint3D", grid,
        tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
        std::span<const SpectralGeneratedResidualSystemEquationInputs>(
            systemInputs.data(), systemInputs.size()),
        4, 4);
    const auto system = generatedSystem.view();

    if (generatedSystem.equations.size() != 4 ||
        generatedSystem.equations[0].residualName != "H" ||
        generatedSystem.equations[1].residualName != "M1" ||
        generatedSystem.equations[2].residualName != "M2" ||
        generatedSystem.equations[3].residualName != "M3" ||
        generatedSystem.equations[0].unknownIndex != 0 ||
        generatedSystem.equations[1].unknownIndex != 1 ||
        generatedSystem.equations[2].unknownIndex != 2 ||
        generatedSystem.equations[3].unknownIndex != 3) {
      throw std::runtime_error("unexpected York momentum equation mapping");
    }

    const auto exactResidual = assembleSpectralResidualSystem(
        system,
        std::span<const std::vector<double>>(exactFields.data(),
                                             exactFields.size()));
    std::printf("[generated-spectral-york-momentum] exact residual l2 = %.17g max = %.17g\n",
                exactResidual.l2Norm, exactResidual.maxAbs);
    if (!exactResidual.finite || !exactResidual.usedGeneratedGridKernels ||
        exactResidual.maxAbs > 2e-11) {
      std::fprintf(stderr, "York momentum exact residual mismatch\n");
      return 3;
    }

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 8;
    options.residualTolerance = 2e-9;
    options.residualRatioTarget = 1e-12;
    options.linearSolver = SpectralLinearSolveKind::DenseJacobian;
    options.denseJacobianMaxUnknowns = 600;
    options.jvpOptions.relativeStep = 1e-6;
    options.linearPivotTolerance = 1e-13;

    const auto run = solveGeneratedSpectralEllipticSystem(
        generatedSystem,
        std::span<std::vector<double>>(solutionFields.data(),
                                       solutionFields.size()),
        options);
    const auto &solveResult = run.solveResult;
    const auto &finalResidual = run.finalResidual;

    std::vector<double> errors(4 * grid.size(), 0.0);
    for (std::size_t field = 0; field < exactFields.size(); ++field) {
      for (std::size_t p = 0; p < grid.size(); ++p)
        errors[field * grid.size() + p] =
            solutionFields[field][p] - exactFields[field][p];
    }
    const double solutionError = maxAbs(errors);

    std::printf("[generated-spectral-york-momentum] steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-york-momentum] initial residual l2 = %.17g\n",
                solveResult.initialResidualL2);
    std::printf("[generated-spectral-york-momentum] final residual l2 = %.17g\n",
                solveResult.finalResidualL2);
    std::printf("[generated-spectral-york-momentum] final residual max = %.17g\n",
                finalResidual.maxAbs);
    std::printf("[generated-spectral-york-momentum] linear iterations = %d\n",
                solveResult.linearIterations);
    std::printf("[generated-spectral-york-momentum] solution max error = %.17g\n",
                solutionError);

    if (!run.initialResidual.usedGeneratedGridKernels ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !finalResidual.usedGeneratedGridKernels ||
        solveResult.finalResidualL2 > 2e-9 || finalResidual.maxAbs > 2e-8 ||
        solutionError > 2e-7) {
      std::fprintf(stderr, "generated York momentum spectral solve mismatch\n");
      return 4;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated York momentum spectral runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
