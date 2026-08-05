#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"
#include "tensorium_mlir/Runtime/TwoPunctureMap.h"

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

using tensorium_mlir::runtime::assembleSpectralResidual;
using tensorium_mlir::runtime::makeTwoPunctureCoordinateMap;
using tensorium_mlir::runtime::makeTwoPunctureDerivativeMap;
using tensorium_mlir::runtime::solveSpectralNewton;
using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralEllipticSolveOptions;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralPointDerivatives3D;
using tensorium_mlir::runtime::spectralResidualGridKernelFromDesc;
using tensorium_mlir::runtime::spectralResidualKernelFromDesc;
using tensorium_mlir::runtime::SpectralResidualProblem;
using tensorium_mlir::runtime::twoPunctureDerivativeMap;

constexpr double kConstant = 0.08;
constexpr double kA = 0.02;
constexpr double kB = -0.015;
constexpr double kPhi = 0.01;

SpectralPointDerivatives3D manufacturedLogicalDerivatives(double A, double B,
                                                          double phi) {
  const double v = kConstant + kA * A + kB * B + kPhi * std::cos(phi);
  SpectralPointDerivatives3D out;
  out.value = (A - 1.0) * v;
  out.d1 = v + (A - 1.0) * kA;
  out.d2 = (A - 1.0) * kB;
  out.d3 = -(A - 1.0) * kPhi * std::sin(phi);
  out.d11 = 2.0 * kA;
  out.d12 = kB;
  out.d13 = -kPhi * std::sin(phi);
  out.d22 = 0.0;
  out.d23 = 0.0;
  out.d33 = -(A - 1.0) * kPhi * std::cos(phi);
  return out;
}

double maxAbsDifference(const std::vector<double> &lhs,
                        const std::vector<double> &rhs) {
  double out = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i)
    out = std::max(out, std::abs(lhs[i] - rhs[i]));
  return out;
}

} // namespace

int main() {
  try {
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 1,
                  "expected a generated spectral point kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 1,
                  "expected a generated spectral grid kernel");

    const auto pointKernel =
        spectralResidualKernelFromDesc(tensorium_spectral_residual_kernels[0]);
    const auto gridKernel = spectralResidualGridKernelFromDesc(
        tensorium_spectral_residual_grid_kernels[0]);

    constexpr double halfSeparation = 1.25;
    constexpr double shift = 0.61;
    const std::array<double, 1> mapParams = {halfSeparation};
    const std::array<double, 1> residualParams = {shift};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::fourierPeriodic(6));

    std::vector<double> expected(grid.size(), 0.0);
    std::vector<double> source(grid.size(), 0.0);
    std::vector<double> solution(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const auto point = grid.point(i, j, k);
          const double logical[3] = {point.x1, point.x2, point.x3};
          const SpectralPointDerivatives3D logicalDerivatives =
              manufacturedLogicalDerivatives(point.x1, point.x2, point.x3);
          SpectralPointDerivatives3D physicalDerivatives;
          twoPunctureDerivativeMap(logical, &logicalDerivatives,
                                   &physicalDerivatives, mapParams.data(),
                                   static_cast<std::int64_t>(mapParams.size()),
                                   nullptr);
          expected[point.index] = logicalDerivatives.value;
          source[point.index] = -(physicalDerivatives.laplacian() +
                                  shift * logicalDerivatives.value);
        }
      }
    }

    const std::array<std::vector<double>, 1> auxiliaryFields = {source};
    SpectralResidualProblem problem{
        &grid, pointKernel, residualParams,
        std::span<const std::vector<double>>(auxiliaryFields.data(),
                                             auxiliaryFields.size())};
    problem.coordinateMap = makeTwoPunctureCoordinateMap();
    problem.coordinateParams = mapParams;
    problem.gridKernel = gridKernel;
    problem.derivativeMap = makeTwoPunctureDerivativeMap();

    const auto initialResidual = assembleSpectralResidual(problem, solution);
    SpectralResidualProblem pointProblem = problem;
    pointProblem.gridKernel = {};
    const auto pointInitialResidual =
        assembleSpectralResidual(pointProblem, solution);
    const double initialKernelDifference =
        maxAbsDifference(initialResidual.values, pointInitialResidual.values);
    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 3;
    options.residualTolerance = 2.0e-8;
    options.residualRatioTarget = 1.0e-9;
    options.linearSolver = SpectralLinearSolveKind::DenseJacobian;
    options.denseJacobianMaxUnknowns = grid.size();
    options.jvpOptions.relativeStep = 2.0e-6;
    options.jvpOptions.absoluteStep = 1.0e-8;
    options.linearPivotTolerance = 1.0e-13;

    const auto solveResult = solveSpectralNewton(problem, solution, options);
    const auto finalResidual = assembleSpectralResidual(problem, solution);
    const double solutionError = maxAbsDifference(solution, expected);

    std::printf("[two-puncture] collocation points = %zu\n", grid.size());
    std::printf("[two-puncture] initial residual l2 = %.17g\n",
                initialResidual.l2Norm);
    std::printf("[two-puncture] point/grid residual delta = %.17g\n",
                initialKernelDifference);
    std::printf("[two-puncture] final residual l2 = %.17g\n",
                finalResidual.l2Norm);
    std::printf("[two-puncture] final residual max = %.17g\n",
                finalResidual.maxAbs);
    std::printf("[two-puncture] solution max error = %.17g\n", solutionError);

    if (!solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !finalResidual.usedGeneratedGridKernel ||
        initialResidual.l2Norm < 1.0e-4 || initialKernelDifference > 2.0e-12 ||
        finalResidual.l2Norm > 2.0e-8 || finalResidual.maxAbs > 2.0e-7 ||
        solutionError > 3.0e-7) {
      std::fprintf(stderr, "mapped generated two-puncture solve failed\n");
      return 3;
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "mapped generated two-puncture solve threw: %s\n",
                 error.what());
    return 2;
  }
  return 0;
}
