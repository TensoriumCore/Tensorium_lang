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
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralJacobianVectorProductOptions;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralGeneratedResidualSystemEquationInputs;
using tensorium_mlir::runtime::assembleSpectralResidualSystem;
using tensorium_mlir::runtime::evaluateSpectralResidualSystemJacobianVectorProduct;
using tensorium_mlir::runtime::makeSpectralResidualSystemFromDesc;
using tensorium_mlir::runtime::solveSpectralNewton;

double exactU(double x, double y, double z) {
  return (4.0 * x * x * x - 3.0 * x) + 0.5 * y * y +
         0.2 * std::cos(2.0 * z);
}

double lapU(double x, double, double z) {
  return 24.0 * x + 1.0 - 0.8 * std::cos(2.0 * z);
}

double exactV(double x, double y, double z) {
  return 0.25 * (2.0 * x * x - 1.0) - 0.4 * y * y * y +
         0.15 * std::sin(3.0 * z);
}

double lapV(double, double y, double z) {
  return 1.0 - 2.4 * y - 1.35 * std::sin(3.0 * z);
}

double dirU(double x, double y, double z) {
  return 0.3 * (4.0 * x * x * x - 3.0 * x) - 0.15 * y * y +
         0.07 * std::cos(2.0 * z);
}

double lapDirU(double x, double, double z) {
  return 7.2 * x - 0.3 - 0.28 * std::cos(2.0 * z);
}

double dirV(double x, double y, double z) {
  return -0.2 * (2.0 * x * x - 1.0) + 0.11 * y * y * y +
         0.05 * std::sin(3.0 * z);
}

double lapDirV(double, double y, double z) {
  return -0.8 + 0.66 * y - 0.45 * std::sin(3.0 * z);
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
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 2,
                  "expected at least two generated spectral point kernels");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 2,
                  "expected at least two generated spectral grid kernels");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT >= 1,
                  "expected at least one generated spectral residual system");

    const auto &systemDesc = tensorium_spectral_residual_systems[0];
    if (!systemDesc.symbol_name ||
        std::strcmp(systemDesc.symbol_name, "SpectralTwoFieldSystem3D") != 0 ||
        systemDesc.unknown_count != 2 || systemDesc.equation_count != 2) {
      throw std::runtime_error("unexpected generated spectral system metadata");
    }

    const double alpha = 0.5;
    const double beta = -0.35;
    const double coupling = 0.125;
    const double huParams[] = {alpha, coupling};
    const double hvParams[] = {beta, coupling};

    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::fourierPeriodic(8));

    std::vector<double> u(grid.size(), 0.0);
    std::vector<double> v(grid.size(), 0.0);
    std::vector<double> du(grid.size(), 0.0);
    std::vector<double> dv(grid.size(), 0.0);
    std::vector<double> sourceU(grid.size(), 0.0);
    std::vector<double> sourceV(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          u[p] = exactU(x, y, z);
          v[p] = exactV(x, y, z);
          du[p] = dirU(x, y, z);
          dv[p] = dirV(x, y, z);
          sourceU[p] = -(lapU(x, y, z) + alpha * u[p] + coupling * v[p]);
          sourceV[p] = -(lapV(x, y, z) + beta * v[p] + coupling * u[p]);
        }
      }
    }

    const std::array<std::vector<double>, 2> unknownFields{u, v};
    const std::array<std::vector<double>, 2> directionFields{du, dv};
    const std::array<std::vector<double>, 2> huAuxiliaryFields{sourceU, v};
    const std::array<std::vector<double>, 2> hvAuxiliaryFields{sourceV, u};
    const std::array<SpectralGeneratedResidualSystemEquationInputs, 2>
        systemInputs{{
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(huParams, 2),
                std::span<const std::vector<double>>(huAuxiliaryFields.data(),
                                                     huAuxiliaryFields.size())},
            SpectralGeneratedResidualSystemEquationInputs{
                std::span<const double>(hvParams, 2),
                std::span<const std::vector<double>>(hvAuxiliaryFields.data(),
                                                     hvAuxiliaryFields.size())},
        }};
    const auto generatedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, grid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
        std::span<const SpectralGeneratedResidualSystemEquationInputs>(
            systemInputs.data(), systemInputs.size()));
    const auto system = generatedSystem.view();

    if (generatedSystem.equations.size() != 2 ||
        generatedSystem.equations[0].residualName != "Hu" ||
        generatedSystem.equations[1].residualName != "Hv" ||
        generatedSystem.equations[0].unknownIndex != 0 ||
        generatedSystem.equations[1].unknownIndex != 1) {
      throw std::runtime_error("unexpected generated spectral equation mapping");
    }

    const auto result = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(unknownFields.data(),
                                                     unknownFields.size()));
    const double huMax = result.equationResults.empty()
                             ? std::numeric_limits<double>::infinity()
                             : result.equationResults[0].maxAbs;
    const double hvMax = result.equationResults.size() < 2
                             ? std::numeric_limits<double>::infinity()
                             : result.equationResults[1].maxAbs;

    std::printf("[generated-spectral-system] equations = %zu points = %zu\n",
                result.equationCount, result.pointsPerEquation);
    std::printf("[generated-spectral-system] Hu max = %.17g\n", huMax);
    std::printf("[generated-spectral-system] Hv max = %.17g\n", hvMax);
    std::printf("[generated-spectral-system] system l2 = %.17g max = %.17g\n",
                result.l2Norm, result.maxAbs);

    if (!result.finite || !result.usedGeneratedGridKernels ||
        result.equationCount != 2 || result.pointsPerEquation != grid.size() ||
        result.size() != 2 * grid.size() || huMax > 6e-10 || hvMax > 6e-10 ||
        result.maxAbs > 6e-10) {
      std::fprintf(stderr, "generated spectral system residual mismatch\n");
      return 3;
    }

    SpectralJacobianVectorProductOptions jvpOptions;
    jvpOptions.relativeStep = 1.0e-6;
    const auto jvp = evaluateSpectralResidualSystemJacobianVectorProduct(
        system, std::span<const std::vector<double>>(unknownFields.data(),
                                                     unknownFields.size()),
        std::span<const std::vector<double>>(directionFields.data(),
                                             directionFields.size()),
        jvpOptions);

    std::vector<double> jvpErrors(2 * grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      const double z = grid.axis(2).points[k];
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        const double y = grid.axis(1).points[j];
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const double x = grid.axis(0).points[i];
          const std::size_t p = grid.index(i, j, k);
          const double expectedHu =
              lapDirU(x, y, z) + alpha * du[p] + coupling * dv[p];
          const double expectedHv =
              lapDirV(x, y, z) + beta * dv[p] + coupling * du[p];
          jvpErrors[p] = jvp.values[p] - expectedHu;
          jvpErrors[grid.size() + p] =
              jvp.values[grid.size() + p] - expectedHv;
        }
      }
    }

    const double jvpError = maxAbs(jvpErrors);
    std::printf("[generated-spectral-system] jvp step = %.17g\n", jvp.step);
    std::printf("[generated-spectral-system] jvp l2 = %.17g max = %.17g\n",
                jvp.l2Norm, jvp.maxAbs);
    std::printf("[generated-spectral-system] jvp max error = %.17g\n",
                jvpError);
    if (!jvp.finite || !jvp.usedGeneratedGridKernels ||
        jvp.size() != 2 * grid.size() || jvp.step <= 0.0 ||
        jvpError > 2e-8) {
      std::fprintf(stderr, "generated spectral system JVP mismatch\n");
      return 4;
    }

    std::array<std::vector<double>, 2> solutionFields{
        std::vector<double>(grid.size(), 0.0),
        std::vector<double>(grid.size(), 0.0)};
    SpectralEllipticSolveOptions solveOptions;
    solveOptions.maxNewtonSteps = 4;
    solveOptions.residualTolerance = 8e-10;
    solveOptions.residualRatioTarget = 1e-12;
    solveOptions.linearSolver = SpectralLinearSolveKind::Auto;
    solveOptions.denseJacobianMaxUnknowns = 1;
    solveOptions.gmresMaxIterations = 512;
    solveOptions.gmresTolerance = 2e-12;
    solveOptions.gmresRelativeTolerance = 1e-13;
    solveOptions.jvpOptions.relativeStep = 1e-6;
    solveOptions.linearPivotTolerance = 1e-13;

    const auto solveResult = solveSpectralNewton(
        system, std::span<std::vector<double>>(solutionFields.data(),
                                               solutionFields.size()),
        solveOptions);
    const auto finalResidual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solutionFields.data(),
                                                     solutionFields.size()));

    std::vector<double> solutionErrors(2 * grid.size(), 0.0);
    for (std::size_t p = 0; p < grid.size(); ++p) {
      solutionErrors[p] = solutionFields[0][p] - u[p];
      solutionErrors[grid.size() + p] = solutionFields[1][p] - v[p];
    }
    const double solutionError = maxAbs(solutionErrors);
    std::printf("[generated-spectral-system] solve steps = %d status = %d\n",
                solveResult.steps, static_cast<int>(solveResult.status));
    std::printf("[generated-spectral-system] solve initial l2 = %.17g\n",
                solveResult.initialResidualL2);
    std::printf("[generated-spectral-system] solve final l2 = %.17g\n",
                solveResult.finalResidualL2);
    std::printf("[generated-spectral-system] solve linear iterations = %d\n",
                solveResult.linearIterations);
    std::printf("[generated-spectral-system] solve linear l2 = %.17g\n",
                solveResult.finalLinearResidualL2);
    std::printf("[generated-spectral-system] solve final max = %.17g\n",
                finalResidual.maxAbs);
    std::printf("[generated-spectral-system] solve max error = %.17g\n",
                solutionError);
    if (!solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !solveResult.usedMatrixFreeGMRES ||
        !finalResidual.usedGeneratedGridKernels ||
        solveResult.finalResidualL2 > 1e-9 || finalResidual.maxAbs > 8e-9 ||
        solutionError > 8e-8) {
      std::fprintf(stderr, "generated spectral system solve mismatch\n");
      return 5;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated spectral system residual runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
