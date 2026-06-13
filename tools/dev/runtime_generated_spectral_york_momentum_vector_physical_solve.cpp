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
using tensorium_mlir::runtime::makeGeneratedSpectralResidualSystem;
using tensorium_mlir::runtime::solveGeneratedSpectralEllipticSystem;

double bump1D(double value) {
  const double q = 1.0 - value * value;
  return q * q;
}

double bump(double x, double y, double z) {
  return bump1D(x) * bump1D(y) * bump1D(z);
}

bool allFinite(std::span<const double> values) {
  for (double value : values)
    if (!std::isfinite(value))
      return false;
  return true;
}

double minValue(std::span<const double> values) {
  double out = values.empty() ? 0.0 : values.front();
  for (double value : values)
    out = std::min(out, value);
  return out;
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

    const double matterCoeff = 0.0;
    const double modifiedCoeff = 0.0;
    const double vectorMass = 0.35;
    const double hParams[] = {matterCoeff, modifiedCoeff};
    const double mParams[] = {vectorMass};

    SpectralGrid3D grid(SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(5, -1.0, 1.0));

    std::array<std::vector<double>, 4> solutionFields{
        std::vector<double>(grid.size(), 1.0),
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0)};

    std::vector<double> a2(grid.size(), 0.0);
    std::vector<double> kTrace(grid.size(), 0.0);
    std::vector<double> rbar(grid.size(), 0.0);
    std::vector<double> modSource(grid.size(), 0.0);
    std::vector<double> rho(grid.size(), 0.0);
    std::array<std::vector<double>, 3> jFields{
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0)};

    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          const double b = bump(x, y, z);
          a2[p] = 0.006 * (1.0 + 0.15 * b);
          jFields[0][p] = 0.01 * b;
          jFields[1][p] = -0.007 * b;
          jFields[2][p] = 0.005 * b;
        }
      }
    }

    const std::array<std::vector<double>, 5> hAuxiliaryFields{
        a2, kTrace, rbar, modSource, rho};
    const std::array<std::vector<double>, 1> m1AuxiliaryFields{jFields[0]};
    const std::array<std::vector<double>, 1> m2AuxiliaryFields{jFields[1]};
    const std::array<std::vector<double>, 1> m3AuxiliaryFields{jFields[2]};

    const std::array<SpectralGeneratedResidualSystemEquationInputs, 4>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(hParams, 2),
                std::span<const std::vector<double>>(hAuxiliaryFields.data(),
                                                     hAuxiliaryFields.size())},
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(mParams, 1),
                std::span<const std::vector<double>>(m1AuxiliaryFields.data(),
                                                     m1AuxiliaryFields.size())},
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(mParams, 1),
                std::span<const std::vector<double>>(m2AuxiliaryFields.data(),
                                                     m2AuxiliaryFields.size())},
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(mParams, 1),
                std::span<const std::vector<double>>(m3AuxiliaryFields.data(),
                                                     m3AuxiliaryFields.size())},
        }};

    const auto generatedSystem = makeGeneratedSpectralResidualSystem(
        tensorium_spectral_residual_systems,
        TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
        "SpectralYorkMomentumVectorConstraint3D", grid,
        tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
        std::span<const SpectralGeneratedResidualSystemEquationInputs>(
            systemInputs.data(), systemInputs.size()),
        4, 4);

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 10;
    options.residualTolerance = 2e-9;
    options.residualRatioTarget = 2e-8;
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

    const double minPsi = minValue(solutionFields[0]);
    double maxW = 0.0;
    for (std::size_t component = 1; component < solutionFields.size();
         ++component)
      maxW = std::max(maxW, maxAbs(solutionFields[component]));

    std::printf("[generated-spectral-york-vector-physical] steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-york-vector-physical] initial residual l2 = %.17g\n",
                solveResult.initialResidualL2);
    std::printf("[generated-spectral-york-vector-physical] final residual l2 = %.17g\n",
                solveResult.finalResidualL2);
    std::printf("[generated-spectral-york-vector-physical] final residual max = %.17g\n",
                finalResidual.maxAbs);
    std::printf("[generated-spectral-york-vector-physical] residual ratio = %.17g\n",
                solveResult.residualRatio);
    std::printf("[generated-spectral-york-vector-physical] min psi = %.17g max W = %.17g\n",
                minPsi, maxW);

    const bool fieldsFinite =
        allFinite(solutionFields[0]) && allFinite(solutionFields[1]) &&
        allFinite(solutionFields[2]) && allFinite(solutionFields[3]);
    if (!run.initialResidual.usedGeneratedGridKernels ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !finalResidual.finite || !finalResidual.usedGeneratedGridKernels ||
        !(solveResult.finalResidualL2 < solveResult.initialResidualL2) ||
        solveResult.residualRatio > 2e-5 || finalResidual.maxAbs > 2e-8 ||
        !fieldsFinite || !(minPsi > 0.0) || !(maxW > 0.0)) {
      std::fprintf(stderr,
                   "generated York vector physical spectral solve mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated York vector physical runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
