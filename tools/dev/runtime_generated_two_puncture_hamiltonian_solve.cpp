#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"
#include "tensorium_mlir/Runtime/SpectralUnknownMaps.h"
#include "tensorium_mlir/Runtime/TwoPunctureDiagnostics.h"
#include "tensorium_mlir/Runtime/TwoPunctureMap.h"
#include "tensorium_mlir/Runtime/TwoPunctureSymmetry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using tensorium_mlir::runtime::assembleSpectralResidualSystem;
using tensorium_mlir::runtime::makeLinearBoundaryFactorUnknownMap;
using tensorium_mlir::runtime::makeSpectralResidualSystemFromDesc;
using tensorium_mlir::runtime::makeTwoPunctureAdmDiagnostics;
using tensorium_mlir::runtime::makeTwoPunctureCoordinateMap;
using tensorium_mlir::runtime::makeTwoPunctureDerivativeMap;
using tensorium_mlir::runtime::makeTwoPunctureInversionEvenFieldProjector;
using tensorium_mlir::runtime::mapTwoPunctureCoordinates;
using tensorium_mlir::runtime::measureTwoPunctureInversionParityError;
using tensorium_mlir::runtime::solveSpectralNewton;
using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralEllipticSolveOptions;
using tensorium_mlir::runtime::SpectralGeneratedResidualSystemEquationInputs;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralPreconditionerKind;

} // namespace

int main() {
  try {
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT == 1,
                  "expected one generated spectral point kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT == 1,
                  "expected one generated spectral grid kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT == 1,
                  "expected one generated spectral residual system");

    const auto &systemDesc = tensorium_spectral_residual_systems[0];
    if (!systemDesc.symbol_name ||
        std::strcmp(systemDesc.symbol_name,
                    "SpectralTwoPunctureHamiltonian3D") != 0 ||
        systemDesc.unknown_count != 1 || systemDesc.equation_count != 1 ||
        systemDesc.equations[0].param_count != 15) {
      throw std::runtime_error(
          "unexpected generated two-puncture residual metadata");
    }

    constexpr double halfSeparation = 1.4;
    constexpr double mass1 = 0.55;
    constexpr double mass2 = 0.55;
    std::array<double, 15> physicalParams{};
    const auto paramIndex = [&](const char *name) {
      const auto &equation = systemDesc.equations[0];
      for (std::int64_t i = 0; i < equation.param_count; ++i) {
        if (equation.param_names[i] &&
            std::strcmp(equation.param_names[i], name) == 0)
          return static_cast<std::size_t>(i);
      }
      throw std::runtime_error(std::string("missing generated parameter: ") +
                               name);
    };
    const auto setParam = [&](const char *name, double value) {
      physicalParams[paramIndex(name)] = value;
    };
    setParam("b", halfSeparation);
    setParam("m1", mass1);
    setParam("m2", mass2);
    const std::array<double, 1> coordinateParams = {halfSeparation};
    const std::array<double, 3> unknownMapParams = {0.0, 1.0, 1.0};
    SpectralGrid3D grid(SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::chebyshevZeros(4),
                        SpectralAxis::fourierPeriodic(6));

    const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
        systemInputs{{SpectralGeneratedResidualSystemEquationInputs{
            physicalParams, {}}}};
    auto generatedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, grid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT, systemInputs);
    auto &problem = generatedSystem.equations[0].problem;
    problem.coordinateMap = makeTwoPunctureCoordinateMap();
    problem.coordinateParams = coordinateParams;
    problem.derivativeMap = makeTwoPunctureDerivativeMap();
    problem.unknownMap = makeLinearBoundaryFactorUnknownMap();
    problem.unknownMapParams = unknownMapParams;
    problem.fieldProjector = makeTwoPunctureInversionEvenFieldProjector();
    const auto system = generatedSystem.view();

    std::vector<double> projectorProbe(grid.size(), 0.0);
    const std::size_t probeIndex = grid.index(0, 0, 0);
    const std::size_t probeImage =
        grid.index(0, grid.n2() - 1, grid.n3() / 2);
    projectorProbe[probeIndex] = 1.0;
    projectorProbe[probeImage] = -3.0;
    const double projectorErrorBefore =
        measureTwoPunctureInversionParityError(grid, projectorProbe);
    problem.fieldProjector.project(
        &grid, projectorProbe.data(),
        static_cast<std::int64_t>(projectorProbe.size()),
        problem.fieldProjector.userData);
    const double projectorErrorAfter =
        measureTwoPunctureInversionParityError(grid, projectorProbe);

    std::array<std::vector<double>, 1> solverFields{
        std::vector<double>(grid.size(), 0.0)};

    const auto validateSinglePunctureContraction =
        [&](bool spinCase) -> double {
      physicalParams.fill(0.0);
      setParam("b", halfSeparation);
      const std::array<double, 3> vector = {0.07, -0.04, 0.03};
      const std::array<const char *, 3> momentumNames = {"p1x", "p1y", "p1z"};
      const std::array<const char *, 3> spinNames = {"s1x", "s1y", "s1z"};
      const auto &names = spinCase ? spinNames : momentumNames;
      for (std::size_t component = 0; component < vector.size(); ++component)
        setParam(names[component], vector[component]);
      const auto residual = assembleSpectralResidualSystem(
          system, std::span<const std::vector<double>>(solverFields.data(),
                                                       solverFields.size()));
      double maxRelativeError = 0.0;
      const double magnitude2 =
          vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
      for (std::size_t k = 0; k < grid.n3(); ++k) {
        for (std::size_t j = 0; j < grid.n2(); ++j) {
          for (std::size_t i = 0; i < grid.n1(); ++i) {
            const auto logical = grid.point(i, j, k);
            const auto physical = mapTwoPunctureCoordinates(
                logical.x1, logical.x2, logical.x3, halfSeparation);
            const std::array<double, 3> displacement = {
                physical.x - halfSeparation, physical.y, physical.z};
            const double radius = std::sqrt(displacement[0] * displacement[0] +
                                            displacement[1] * displacement[1] +
                                            displacement[2] * displacement[2]);
            const double dot =
                (vector[0] * displacement[0] + vector[1] * displacement[1] +
                 vector[2] * displacement[2]) /
                radius;
            const double expectedA2 =
                spinCase
                    ? 18.0 * (magnitude2 - dot * dot) / std::pow(radius, 6.0)
                    : 4.5 * (magnitude2 + 2.0 * dot * dot) /
                          std::pow(radius, 4.0);
            const double expectedResidual = 0.125 * expectedA2;
            maxRelativeError = std::max(
                maxRelativeError,
                std::abs(residual.values[logical.index] - expectedResidual) /
                    (1.0 + std::abs(expectedResidual)));
          }
        }
      }
      return maxRelativeError;
    };

    const double momentumContractionError =
        validateSinglePunctureContraction(false);
    const double spinContractionError = validateSinglePunctureContraction(true);

    physicalParams.fill(0.0);
    setParam("b", halfSeparation);
    setParam("m1", mass1);
    setParam("m2", mass2);
    const auto timeSymmetricResidual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solverFields.data(),
                                                     solverFields.size()));
    if (!timeSymmetricResidual.finite ||
        !timeSymmetricResidual.usedGeneratedGridKernels ||
        timeSymmetricResidual.maxAbs > 2.0e-12) {
      std::fprintf(stderr,
                   "Brill-Lindquist time-symmetric residual is not zero\n");
      return 3;
    }

    // Equal and opposite tangential Bowen-York momenta.
    setParam("p1y", 0.08);
    setParam("p2y", -0.08);
    const auto initialResidual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solverFields.data(),
                                                     solverFields.size()));

    SpectralEllipticSolveOptions options;
    options.maxNewtonSteps = 12;
    options.residualTolerance = 2.0e-8;
    options.residualRatioTarget = 1.0e-9;
    options.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
    options.denseJacobianMaxUnknowns = 1;
    options.gmresMaxIterations = 64;
    options.gmresTolerance = 1.0e-10;
    options.gmresRelativeTolerance = 1.0e-10;
    options.gmresPreconditioner = SpectralPreconditionerKind::DiagonalJVP;
    options.jvpOptions.relativeStep = 2.0e-6;
    options.jvpOptions.absoluteStep = 1.0e-8;
    options.linearPivotTolerance = 1.0e-13;

    const auto solveResult =
        solveSpectralNewton(system,
                            std::span<std::vector<double>>(solverFields.data(),
                                                           solverFields.size()),
                            options);
    const auto finalResidual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solverFields.data(),
                                                     solverFields.size()));

    double minPsi = std::numeric_limits<double>::infinity();
    double maxCorrection = 0.0;
    double maxOuterCorrection = 0.0;
    const double outerA = grid.axis(0).points.front();
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const auto logical = grid.point(i, j, k);
          const auto physical = mapTwoPunctureCoordinates(
              logical.x1, logical.x2, logical.x3, halfSeparation);
          const double r1 = std::sqrt(
              (physical.x - halfSeparation) * (physical.x - halfSeparation) +
              physical.y * physical.y + physical.z * physical.z);
          const double r2 = std::sqrt(
              (physical.x + halfSeparation) * (physical.x + halfSeparation) +
              physical.y * physical.y + physical.z * physical.z);
          const double correction =
              (logical.x1 - 1.0) * solverFields[0][logical.index];
          const double psi =
              1.0 + 0.5 * mass1 / r1 + 0.5 * mass2 / r2 + correction;
          minPsi = std::min(minPsi, psi);
          maxCorrection = std::max(maxCorrection, std::abs(correction));
          if (logical.x1 == outerA)
            maxOuterCorrection =
                std::max(maxOuterCorrection, std::abs(correction));
        }
      }
    }

    constexpr double probeA = 0.0;
    constexpr double probeB = 0.0;
    constexpr double probePhi = 0.37;
    const std::array<double, 3> momentum1 = {0.0, 0.08, 0.0};
    const std::array<double, 3> momentum2 = {0.0, -0.08, 0.0};
    const std::array<double, 3> zeroSpin = {0.0, 0.0, 0.0};
    const double coarseProbe =
        (probeA - 1.0) *
        grid.interpolate(solverFields[0], probeA, probeB, probePhi);
    const double coarseVInfinity =
        grid.interpolate(solverFields[0], 1.0, 0.0, 0.0);
    const auto coarseAdm = makeTwoPunctureAdmDiagnostics(
        halfSeparation, mass1, mass2, coarseVInfinity, momentum1, momentum2,
        zeroSpin, zeroSpin);

    SpectralGrid3D fineGrid(SpectralAxis::chebyshevZeros(5),
                            SpectralAxis::chebyshevZeros(5),
                            SpectralAxis::fourierPeriodic(8));
    auto fineGeneratedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, fineGrid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT, systemInputs);
    auto &fineProblem = fineGeneratedSystem.equations[0].problem;
    fineProblem.coordinateMap = makeTwoPunctureCoordinateMap();
    fineProblem.coordinateParams = coordinateParams;
    fineProblem.derivativeMap = makeTwoPunctureDerivativeMap();
    fineProblem.unknownMap = makeLinearBoundaryFactorUnknownMap();
    fineProblem.unknownMapParams = unknownMapParams;
    fineProblem.fieldProjector = makeTwoPunctureInversionEvenFieldProjector();
    const auto fineSystem = fineGeneratedSystem.view();
    std::array<std::vector<double>, 1> fineSolverFields{
        std::vector<double>(fineGrid.size(), 0.0)};
    SpectralEllipticSolveOptions fineOptions = options;
    const auto fineSolveResult = solveSpectralNewton(
        fineSystem,
        std::span<std::vector<double>>(fineSolverFields.data(),
                                       fineSolverFields.size()),
        fineOptions);
    const auto fineResidual = assembleSpectralResidualSystem(
        fineSystem, std::span<const std::vector<double>>(
                        fineSolverFields.data(), fineSolverFields.size()));
    const double fineProbe =
        (probeA - 1.0) *
        fineGrid.interpolate(fineSolverFields[0], probeA, probeB, probePhi);
    const double fineVInfinity =
        fineGrid.interpolate(fineSolverFields[0], 1.0, 0.0, 0.0);
    const auto fineAdm = makeTwoPunctureAdmDiagnostics(
        halfSeparation, mass1, mass2, fineVInfinity, momentum1, momentum2,
        zeroSpin, zeroSpin);
    const double probeDelta = std::abs(fineProbe - coarseProbe);
    const double admDelta = std::abs(fineAdm.energy - coarseAdm.energy);

    SpectralGrid3D veryCoarseGrid(SpectralAxis::chebyshevZeros(3),
                                  SpectralAxis::chebyshevZeros(3),
                                  SpectralAxis::fourierPeriodic(4));
    auto veryCoarseGeneratedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, veryCoarseGrid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT, systemInputs);
    auto &veryCoarseProblem = veryCoarseGeneratedSystem.equations[0].problem;
    veryCoarseProblem.coordinateMap = makeTwoPunctureCoordinateMap();
    veryCoarseProblem.coordinateParams = coordinateParams;
    veryCoarseProblem.derivativeMap = makeTwoPunctureDerivativeMap();
    veryCoarseProblem.unknownMap = makeLinearBoundaryFactorUnknownMap();
    veryCoarseProblem.unknownMapParams = unknownMapParams;
    veryCoarseProblem.fieldProjector =
        makeTwoPunctureInversionEvenFieldProjector();
    const auto veryCoarseSystem = veryCoarseGeneratedSystem.view();
    std::array<std::vector<double>, 1> veryCoarseSolverFields{
        std::vector<double>(veryCoarseGrid.size(), 0.0)};
    SpectralEllipticSolveOptions veryCoarseOptions = options;
    const auto veryCoarseSolveResult = solveSpectralNewton(
        veryCoarseSystem,
        std::span<std::vector<double>>(veryCoarseSolverFields.data(),
                                       veryCoarseSolverFields.size()),
        veryCoarseOptions);
    const double veryCoarseProbe =
        (probeA - 1.0) * veryCoarseGrid.interpolate(veryCoarseSolverFields[0],
                                                    probeA, probeB, probePhi);
    const double veryCoarseVInfinity =
        veryCoarseGrid.interpolate(veryCoarseSolverFields[0], 1.0, 0.0, 0.0);
    const auto veryCoarseAdm = makeTwoPunctureAdmDiagnostics(
        halfSeparation, mass1, mass2, veryCoarseVInfinity, momentum1, momentum2,
        zeroSpin, zeroSpin);
    const double veryCoarseProbeDelta = std::abs(coarseProbe - veryCoarseProbe);
    const double veryCoarseAdmDelta =
        std::abs(coarseAdm.energy - veryCoarseAdm.energy);

    const double fineSymmetryError = measureTwoPunctureInversionParityError(
        fineGrid, fineSolverFields[0]);

    std::printf("[two-puncture-hamiltonian] Brill-Lindquist residual max = "
                "%.17g\n",
                timeSymmetricResidual.maxAbs);
    std::printf("[two-puncture-hamiltonian] Bowen-York momentum A2 relative "
                "error = %.17g\n",
                momentumContractionError);
    std::printf("[two-puncture-hamiltonian] Bowen-York spin A2 relative error "
                "= %.17g\n",
                spinContractionError);
    std::printf("[two-puncture-hamiltonian] inversion projector error = "
                "%.17g -> %.17g\n",
                projectorErrorBefore, projectorErrorAfter);
    std::printf("[two-puncture-hamiltonian] boosted initial residual l2 = "
                "%.17g\n",
                initialResidual.l2Norm);
    std::printf("[two-puncture-hamiltonian] boosted final residual l2 = "
                "%.17g\n",
                finalResidual.l2Norm);
    std::printf("[two-puncture-hamiltonian] boosted final residual max = "
                "%.17g\n",
                finalResidual.maxAbs);
    std::printf("[two-puncture-hamiltonian] coarse Newton status/steps/linear = "
                "%d/%d/%d, linear residual = %.17g\n",
                static_cast<int>(solveResult.status), solveResult.steps,
                solveResult.linearIterations,
                solveResult.finalLinearResidualL2);
    std::printf("[two-puncture-hamiltonian] fine Newton status/steps/linear = "
                "%d/%d/%d, linear residual = %.17g\n",
                static_cast<int>(fineSolveResult.status), fineSolveResult.steps,
                fineSolveResult.linearIterations,
                fineSolveResult.finalLinearResidualL2);
    std::printf("[two-puncture-hamiltonian] very-coarse Newton "
                "status/steps/linear = %d/%d/%d, linear residual = %.17g\n",
                static_cast<int>(veryCoarseSolveResult.status),
                veryCoarseSolveResult.steps,
                veryCoarseSolveResult.linearIterations,
                veryCoarseSolveResult.finalLinearResidualL2);
    std::printf("[two-puncture-hamiltonian] correction max = %.17g\n",
                maxCorrection);
    std::printf("[two-puncture-hamiltonian] outer correction max = %.17g\n",
                maxOuterCorrection);
    std::printf("[two-puncture-hamiltonian] minimum psi = %.17g\n", minPsi);
    std::printf("[two-puncture-hamiltonian] coarse/fine probe delta = %.17g\n",
                probeDelta);
    std::printf("[two-puncture-hamiltonian] very-coarse/coarse probe delta = "
                "%.17g\n",
                veryCoarseProbeDelta);
    std::printf("[two-puncture-hamiltonian] coarse ADM energy = %.17g\n",
                coarseAdm.energy);
    std::printf("[two-puncture-hamiltonian] fine ADM energy = %.17g\n",
                fineAdm.energy);
    std::printf("[two-puncture-hamiltonian] coarse/fine ADM delta = %.17g\n",
                admDelta);
    std::printf("[two-puncture-hamiltonian] very-coarse/coarse ADM delta = "
                "%.17g\n",
                veryCoarseAdmDelta);
    std::printf("[two-puncture-hamiltonian] fine orbital symmetry error = "
                "%.17g\n",
                fineSymmetryError);

    if (momentumContractionError > 2.0e-12 || spinContractionError > 2.0e-12 ||
        projectorErrorBefore < 1.0 || projectorErrorAfter != 0.0 ||
        projectorProbe[probeIndex] != -1.0 ||
        projectorProbe[probeImage] != -1.0 ||
        !initialResidual.finite || initialResidual.l2Norm < 1.0e-5 ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !solveResult.usedMatrixFreeGMRES || !solveResult.usedPreconditioner ||
        !solveResult.usedFieldProjector ||
        !finalResidual.finite || !finalResidual.usedGeneratedGridKernels ||
        finalResidual.l2Norm > 2.0e-8 || finalResidual.maxAbs > 2.0e-7 ||
        !(maxCorrection > 1.0e-8) || !(minPsi > 0.0) ||
        !(maxOuterCorrection < maxCorrection) || !fineSolveResult.converged() ||
        !fineSolveResult.usedMatrixFreeGMRES ||
        !fineSolveResult.usedPreconditioner ||
        !fineSolveResult.usedFieldProjector ||
        !veryCoarseSolveResult.converged() ||
        !veryCoarseSolveResult.usedMatrixFreeGMRES ||
        !veryCoarseSolveResult.usedPreconditioner ||
        !veryCoarseSolveResult.usedFieldProjector || !fineResidual.finite ||
        fineResidual.l2Norm > 2.0e-8 || fineResidual.maxAbs > 2.0e-7 ||
        probeDelta > 1.0e-3 || admDelta > 2.0e-3 ||
        fineSymmetryError > 2.0e-8 || !(probeDelta < veryCoarseProbeDelta) ||
        !(admDelta < veryCoarseAdmDelta) ||
        std::abs(fineAdm.linearMomentum[1]) > 1.0e-14 ||
        std::abs(fineAdm.angularMomentum[2] -
                 2.0 * halfSeparation * momentum1[1]) > 1.0e-14) {
      std::fprintf(stderr, "physical two-puncture Hamiltonian solve failed\n");
      return 4;
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "two-puncture Hamiltonian runner failed: %s\n",
                 error.what());
    return 2;
  }
  return 0;
}
