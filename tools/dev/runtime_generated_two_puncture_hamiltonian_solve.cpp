#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"
#include "tensorium_mlir/Runtime/SpectralUnknownMaps.h"
#include "tensorium_mlir/Runtime/TwoPunctureDiagnostics.h"
#include "tensorium_mlir/Runtime/TwoPunctureHandoff.h"
#include "tensorium_mlir/Runtime/TwoPunctureMap.h"
#include "tensorium_mlir/Runtime/TwoPunctureMassCalibration.h"
#include "tensorium_mlir/Runtime/TwoPunctureRegularity.h"
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
using tensorium_mlir::runtime::
    buildSpectralMappedFiniteDifferenceLaplacianShift;
using tensorium_mlir::runtime::calibrateTwoPunctureBareMasses;
using tensorium_mlir::runtime::evaluateTwoPunctureBowenYorkTensor;
using tensorium_mlir::runtime::evaluateTwoPunctureBssnPoint;
using tensorium_mlir::runtime::interpolateTwoPunctureBssnToCartesianGrid;
using tensorium_mlir::runtime::invertTwoPunctureCoordinates;
using tensorium_mlir::runtime::makeLinearBoundaryFactorUnknownMap;
using tensorium_mlir::runtime::makeSpectralResidualSystemFromDesc;
using tensorium_mlir::runtime::makeTwoPunctureAdmDiagnostics;
using tensorium_mlir::runtime::makeTwoPunctureCoordinateMap;
using tensorium_mlir::runtime::makeTwoPunctureDerivativeMap;
using tensorium_mlir::runtime::makeTwoPunctureInversionEvenFieldProjector;
using tensorium_mlir::runtime::makeTwoPunctureLocalMassDiagnostics;
using tensorium_mlir::runtime::makeTwoPunctureScalarRegularityFieldProjector;
using tensorium_mlir::runtime::mapTwoPunctureCoordinates;
using tensorium_mlir::runtime::measureTwoPunctureInversionParityError;
using tensorium_mlir::runtime::measureTwoPunctureScalarRegularity;
using tensorium_mlir::runtime::sampleTwoPunctureRegularField;
using tensorium_mlir::runtime::solveSpectralNewton;
using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralEllipticSolveOptions;
using tensorium_mlir::runtime::SpectralGeneratedResidualSystemEquationInputs;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralPreconditionerKind;
using tensorium_mlir::runtime::TwoPunctureBssnGridBuffers;
using tensorium_mlir::runtime::TwoPunctureCartesianGridView;
using tensorium_mlir::runtime::TwoPunctureGaugeSeed;
using tensorium_mlir::runtime::TwoPunctureMassCalibrationOptions;
using tensorium_mlir::runtime::TwoPuncturePhysicalParameters;
using tensorium_mlir::runtime::updateTwoPunctureBareMasses;

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
      const auto findIndex = [&](const char *candidate) {
        for (std::int64_t i = 0; i < equation.param_count; ++i) {
          if (equation.param_names[i] &&
              std::strcmp(equation.param_names[i], candidate) == 0)
            return static_cast<std::size_t>(i);
        }
        return static_cast<std::size_t>(equation.param_count);
      };
      const std::size_t canonical = findIndex(name);
      if (canonical != static_cast<std::size_t>(equation.param_count))
        return canonical;
      static constexpr std::array<std::pair<const char *, const char *>, 12>
          kDescriptiveNames = {{{"p1x", "P1_x"},
                                {"p1y", "P1_y"},
                                {"p1z", "P1_z"},
                                {"s1x", "S1_x"},
                                {"s1y", "S1_y"},
                                {"s1z", "S1_z"},
                                {"p2x", "P2_x"},
                                {"p2y", "P2_y"},
                                {"p2z", "P2_z"},
                                {"s2x", "S2_x"},
                                {"s2y", "S2_y"},
                                {"s2z", "S2_z"}}};
      for (const auto &[canonicalName, descriptiveName] : kDescriptiveNames) {
        if (std::strcmp(name, canonicalName) == 0) {
          const std::size_t descriptive = findIndex(descriptiveName);
          if (descriptive != static_cast<std::size_t>(equation.param_count))
            return descriptive;
          break;
        }
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
    const std::size_t probeImage = grid.index(0, grid.n2() - 1, grid.n3() / 2);
    projectorProbe[probeIndex] = 1.0;
    projectorProbe[probeImage] = -3.0;
    const double projectorErrorBefore =
        measureTwoPunctureInversionParityError(grid, projectorProbe);
    const auto inversionProjector =
        makeTwoPunctureInversionEvenFieldProjector();
    inversionProjector.project(&grid, projectorProbe.data(),
                               static_cast<std::int64_t>(projectorProbe.size()),
                               inversionProjector.userData);
    const double projectorErrorAfter =
        measureTwoPunctureInversionParityError(grid, projectorProbe);

    std::vector<double> regularityProbe(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          const auto point = grid.point(i, j, k);
          regularityProbe[point.index] =
              0.2 +
              std::cos(point.x3) * (1.0 + 0.3 * point.x1 - 0.2 * point.x2);
        }
      }
    }
    const double regularityErrorBefore =
        measureTwoPunctureScalarRegularity(grid, regularityProbe)
            .maxPhiVariation();
    const auto regularityProjector =
        makeTwoPunctureScalarRegularityFieldProjector();
    regularityProjector.project(
        &grid, regularityProbe.data(),
        static_cast<std::int64_t>(regularityProbe.size()),
        regularityProjector.userData);
    const double regularityErrorAfter =
        measureTwoPunctureScalarRegularity(grid, regularityProbe)
            .maxPhiVariation();
    const auto regularityProjectedOnce = regularityProbe;
    regularityProjector.project(
        &grid, regularityProbe.data(),
        static_cast<std::int64_t>(regularityProbe.size()),
        regularityProjector.userData);
    double regularityIdempotenceError = 0.0;
    for (std::size_t p = 0; p < regularityProbe.size(); ++p) {
      regularityIdempotenceError =
          std::max(regularityIdempotenceError,
                   std::abs(regularityProbe[p] - regularityProjectedOnce[p]));
    }

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
    options.gmresMaxIterations = 128;
    options.gmresRestart = 24;
    options.gmresTolerance = 1.0e-10;
    options.gmresRelativeTolerance = 1.0e-10;
    options.gmresPreconditioner =
        SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift;
    options.preconditionerRelaxationSweeps = 6;
    options.jvpOptions.relativeStep = 2.0e-6;
    options.jvpOptions.absoluteStep = 1.0e-8;
    options.linearPivotTolerance = 1.0e-13;

    const auto mappedStencil =
        buildSpectralMappedFiniteDifferenceLaplacianShift(
            problem, options.preconditionerLaplacianShift,
            options.preconditionerPivotTolerance);
    std::size_t maxStencilWidth = 0;
    for (std::size_t row = 0; row < mappedStencil.size; ++row) {
      maxStencilWidth =
          std::max(maxStencilWidth, mappedStencil.rowOffsets[row + 1] -
                                        mappedStencil.rowOffsets[row]);
    }

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
    const auto coarsePunctureSample =
        sampleTwoPunctureRegularField(grid, solverFields[0]);
    const auto coarseLocalMasses = makeTwoPunctureLocalMassDiagnostics(
        halfSeparation, mass1, mass2, coarsePunctureSample.values[0],
        coarsePunctureSample.values[1]);
    const auto coarseRegularity =
        measureTwoPunctureScalarRegularity(grid, solverFields[0]);

    SpectralGrid3D fineGrid(SpectralAxis::chebyshevZeros(7),
                            SpectralAxis::chebyshevZeros(7),
                            SpectralAxis::fourierPeriodic(12));
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
    std::array<std::vector<double>, 1> multigridSolverFields{
        std::vector<double>(fineGrid.size(), 0.0)};
    SpectralEllipticSolveOptions multigridOptions = fineOptions;
    multigridOptions.gmresPreconditioner =
        SpectralPreconditionerKind::MappedFiniteDifferenceMultigrid;
    multigridOptions.preconditionerMultigridPreSweeps = 3;
    multigridOptions.preconditionerMultigridPostSweeps = 3;
    multigridOptions.preconditionerMultigridRelaxationOmega = 1.0;
    const auto multigridSolveResult = solveSpectralNewton(
        fineSystem,
        std::span<std::vector<double>>(multigridSolverFields.data(),
                                       multigridSolverFields.size()),
        multigridOptions);
    const auto multigridResidual = assembleSpectralResidualSystem(
        fineSystem, std::span<const std::vector<double>>(
                        multigridSolverFields.data(),
                        multigridSolverFields.size()));
    const double fineProbe =
        (probeA - 1.0) *
        fineGrid.interpolate(fineSolverFields[0], probeA, probeB, probePhi);
    const double fineVInfinity =
        fineGrid.interpolate(fineSolverFields[0], 1.0, 0.0, 0.0);
    const auto fineAdm = makeTwoPunctureAdmDiagnostics(
        halfSeparation, mass1, mass2, fineVInfinity, momentum1, momentum2,
        zeroSpin, zeroSpin);
    const auto finePunctureSample =
        sampleTwoPunctureRegularField(fineGrid, fineSolverFields[0]);
    const auto fineLocalMasses = makeTwoPunctureLocalMassDiagnostics(
        halfSeparation, mass1, mass2, finePunctureSample.values[0],
        finePunctureSample.values[1]);
    const double probeDelta = std::abs(fineProbe - coarseProbe);
    const double admDelta = std::abs(fineAdm.energy - coarseAdm.energy);
    const double punctureMassDelta = std::max(
        std::abs(fineLocalMasses.admMasses[0] - coarseLocalMasses.admMasses[0]),
        std::abs(fineLocalMasses.admMasses[1] -
                 coarseLocalMasses.admMasses[1]));

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
    const auto veryCoarsePunctureSample = sampleTwoPunctureRegularField(
        veryCoarseGrid, veryCoarseSolverFields[0]);
    const auto veryCoarseLocalMasses = makeTwoPunctureLocalMassDiagnostics(
        halfSeparation, mass1, mass2, veryCoarsePunctureSample.values[0],
        veryCoarsePunctureSample.values[1]);
    const auto veryCoarseRegularity = measureTwoPunctureScalarRegularity(
        veryCoarseGrid, veryCoarseSolverFields[0]);
    const double veryCoarseProbeDelta = std::abs(coarseProbe - veryCoarseProbe);
    const double veryCoarseAdmDelta =
        std::abs(coarseAdm.energy - veryCoarseAdm.energy);
    const double veryCoarsePunctureMassDelta =
        std::max(std::abs(coarseLocalMasses.admMasses[0] -
                          veryCoarseLocalMasses.admMasses[0]),
                 std::abs(coarseLocalMasses.admMasses[1] -
                          veryCoarseLocalMasses.admMasses[1]));

    const double fineSymmetryError =
        measureTwoPunctureInversionParityError(fineGrid, fineSolverFields[0]);
    const auto fineRegularity =
        measureTwoPunctureScalarRegularity(fineGrid, fineSolverFields[0]);

    const std::array<double, 2> algebraicTargets = {0.63, 0.41};
    const std::array<double, 2> algebraicRegularField = {0.03, 0.07};
    const auto algebraicBareMasses = updateTwoPunctureBareMasses(
        halfSeparation, algebraicTargets, algebraicRegularField);
    const auto algebraicLocalMasses = makeTwoPunctureLocalMassDiagnostics(
        halfSeparation, algebraicBareMasses[0], algebraicBareMasses[1],
        algebraicRegularField[0], algebraicRegularField[1]);
    const double algebraicMassError = std::max(
        std::abs(algebraicLocalMasses.admMasses[0] - algebraicTargets[0]),
        std::abs(algebraicLocalMasses.admMasses[1] - algebraicTargets[1]));

    std::array<std::vector<double>, 1> calibrationFields{
        std::vector<double>(grid.size(), 0.0)};
    double calibrationMaxPhiVariation = 0.0;
    bool calibrationUsedMatrixFree = false;
    TwoPunctureMassCalibrationOptions calibrationOptions;
    calibrationOptions.maxIterations = 8;
    calibrationOptions.absoluteTolerance = 5.0e-9;
    calibrationOptions.relativeTolerance = 5.0e-9;
    const auto calibrationResult = calibrateTwoPunctureBareMasses(
        halfSeparation, coarseLocalMasses.admMasses,
        std::array<double, 2>{0.50, 0.50},
        [&](const std::array<double, 2> &bareMasses,
            std::array<double, 2> &regularField) {
          setParam("m1", bareMasses[0]);
          setParam("m2", bareMasses[1]);
          const auto calibrationSolve = solveSpectralNewton(
              system,
              std::span<std::vector<double>>(calibrationFields.data(),
                                             calibrationFields.size()),
              options);
          calibrationUsedMatrixFree =
              calibrationUsedMatrixFree || calibrationSolve.usedMatrixFreeGMRES;
          if (!calibrationSolve.converged())
            return false;
          const auto sample =
              sampleTwoPunctureRegularField(grid, calibrationFields[0]);
          regularField = sample.values;
          calibrationMaxPhiVariation =
              std::max({calibrationMaxPhiVariation, sample.maxPhiVariation[0],
                        sample.maxPhiVariation[1]});
          return true;
        },
        calibrationOptions);
    setParam("m1", mass1);
    setParam("m2", mass2);

    // Table 1 of Ansorg, Bruegmann, and Tichy, Phys. Rev. D 70,
    // 064011 (2004): the q=0.1 non-spinning test-mass sequence member.
    const double publishedHeavyMass = 1.0;
    const double publishedLightMass = 0.1 * publishedHeavyMass;
    const double publishedDistance =
        publishedHeavyMass * (2.5 + std::sqrt(6.0));
    const double publishedHalfSeparation = 0.5 * publishedDistance;
    const double publishedLightVelocity =
        4.0 * std::sqrt(3.0) / (5.0 + 2.0 * std::sqrt(6.0));
    const double publishedMomentum =
        publishedLightMass * publishedLightVelocity;
    const std::array<double, 1> publishedCoordinateParams = {
        publishedHalfSeparation};
    physicalParams.fill(0.0);
    setParam("b", publishedHalfSeparation);
    setParam("m1", publishedLightMass);
    setParam("m2", publishedHeavyMass);
    setParam("p1y", -publishedMomentum);
    setParam("p2y", publishedMomentum);
    SpectralGrid3D publishedGrid(SpectralAxis::chebyshevZeros(10),
                                 SpectralAxis::chebyshevZeros(10),
                                 SpectralAxis::fourierPeriodic(16));
    auto publishedGeneratedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, publishedGrid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT, systemInputs);
    auto &publishedProblem = publishedGeneratedSystem.equations[0].problem;
    publishedProblem.coordinateMap = makeTwoPunctureCoordinateMap();
    publishedProblem.coordinateParams = publishedCoordinateParams;
    publishedProblem.derivativeMap = makeTwoPunctureDerivativeMap();
    publishedProblem.unknownMap = makeLinearBoundaryFactorUnknownMap();
    publishedProblem.unknownMapParams = unknownMapParams;
    const auto publishedSystem = publishedGeneratedSystem.view();
    std::array<std::vector<double>, 1> publishedFields{
        std::vector<double>(publishedGrid.size(), 0.0)};
    const auto publishedSolve =
        solveSpectralNewton(publishedSystem,
                            std::span<std::vector<double>>(
                                publishedFields.data(), publishedFields.size()),
                            fineOptions);
    const auto publishedResidual = assembleSpectralResidualSystem(
        publishedSystem, std::span<const std::vector<double>>(
                             publishedFields.data(), publishedFields.size()));
    const auto publishedPunctureSample =
        sampleTwoPunctureRegularField(publishedGrid, publishedFields[0]);
    const double publishedVInfinity =
        publishedGrid.interpolate(publishedFields[0], 1.0, 0.0, 0.0);
    const double publishedScaledHeavy = 2.0 * publishedDistance *
                                        publishedPunctureSample.values[1] /
                                        publishedLightMass;
    const double publishedScaledInfinity = -4.0 * publishedHalfSeparation *
                                           publishedVInfinity /
                                           publishedLightMass;
    const double publishedMaxRelativeError = std::max(
        {std::abs(publishedPunctureSample.values[0] - 0.03417) / 0.03417,
         std::abs(publishedScaledHeavy - 0.2011) / 0.2011,
         std::abs(publishedScaledInfinity - 0.1688) / 0.1688});

    TwoPuncturePhysicalParameters handoffParameters;
    handoffParameters.halfSeparation = publishedHalfSeparation;
    handoffParameters.bareMasses = {publishedLightMass, publishedHeavyMass};
    handoffParameters.momenta = {
        {{0.0, -publishedMomentum, 0.0}, {0.0, publishedMomentum, 0.0}}};
    constexpr std::size_t handoffPointCount = 5;
    const std::array<double, 3> handoffA = {-0.3, 0.1, 0.6};
    const std::array<double, 3> handoffB = {-0.2, 0.35, -0.4};
    const std::array<double, 3> handoffPhi = {0.4, 1.7, 5.5};
    std::array<double, handoffPointCount> handoffX{};
    std::array<double, handoffPointCount> handoffY{};
    std::array<double, handoffPointCount> handoffZ{};
    for (std::size_t point = 0; point < handoffA.size(); ++point) {
      const auto physical =
          mapTwoPunctureCoordinates(handoffA[point], handoffB[point],
                                    handoffPhi[point], publishedHalfSeparation);
      handoffX[point] = physical.x;
      handoffY[point] = physical.y;
      handoffZ[point] = physical.z;
    }
    handoffX[3] = publishedHalfSeparation;
    handoffX[4] = -publishedHalfSeparation;

    std::array<double, handoffPointCount> handoffChi{};
    std::array<double, handoffPointCount> handoffMeanCurvature{};
    std::array<double, handoffPointCount> handoffCorrection{};
    std::array<double, handoffPointCount> handoffPsi{};
    std::array<double, handoffPointCount> handoffLapse{};
    std::array<std::array<double, handoffPointCount>, 9> handoffMetric{};
    std::array<std::array<double, handoffPointCount>, 9> handoffInverse{};
    std::array<std::array<double, handoffPointCount>, 9> handoffExtrinsic{};
    std::array<std::array<double, handoffPointCount>, 3> handoffConnection{};
    std::array<std::array<double, handoffPointCount>, 3> handoffShift{};
    TwoPunctureBssnGridBuffers handoffBuffers;
    handoffBuffers.chi = handoffChi.data();
    handoffBuffers.meanCurvature = handoffMeanCurvature.data();
    handoffBuffers.regularCorrection = handoffCorrection.data();
    handoffBuffers.conformalFactor = handoffPsi.data();
    handoffBuffers.lapse = handoffLapse.data();
    for (std::size_t component = 0; component < 9; ++component) {
      handoffBuffers.conformalMetric[component] =
          handoffMetric[component].data();
      handoffBuffers.inverseConformalMetric[component] =
          handoffInverse[component].data();
      handoffBuffers.traceFreeExtrinsicCurvature[component] =
          handoffExtrinsic[component].data();
    }
    for (std::size_t component = 0; component < 3; ++component) {
      handoffBuffers.conformalConnection[component] =
          handoffConnection[component].data();
      handoffBuffers.shift[component] = handoffShift[component].data();
    }
    TwoPunctureGaugeSeed handoffGauge;
    handoffGauge.lapse = 0.75;
    handoffGauge.shift = {0.1, -0.2, 0.3};
    interpolateTwoPunctureBssnToCartesianGrid(
        publishedProblem, publishedFields[0], handoffParameters,
        TwoPunctureCartesianGridView{
            handoffPointCount,
            {handoffX.data(), handoffY.data(), handoffZ.data()}},
        handoffBuffers, handoffGauge);

    double handoffLogicalError = 0.0;
    double handoffCorrectionError = 0.0;
    double handoffAlgebraicError = 0.0;
    const double twoPi = 2.0 * std::acos(-1.0);
    for (std::size_t point = 0; point < handoffA.size(); ++point) {
      const auto logical = invertTwoPunctureCoordinates(
          handoffX[point], handoffY[point], handoffZ[point],
          publishedHalfSeparation);
      const double phiDifference = std::abs(logical.phi - handoffPhi[point]);
      const double periodicPhiDifference =
          std::min(phiDifference, std::abs(phiDifference - twoPi));
      handoffLogicalError = std::max(
          {handoffLogicalError, std::abs(logical.A - handoffA[point]),
           std::abs(logical.B - handoffB[point]), periodicPhiDifference});
      const double expectedCorrection =
          (handoffA[point] - 1.0) *
          publishedGrid.interpolate(publishedFields[0], handoffA[point],
                                    handoffB[point], handoffPhi[point]);
      handoffCorrectionError =
          std::max(handoffCorrectionError,
                   std::abs(handoffCorrection[point] - expectedCorrection));
      const double expectedChi = std::pow(handoffPsi[point], -4.0);
      handoffAlgebraicError = std::max(
          handoffAlgebraicError, std::abs(handoffChi[point] - expectedChi));
      double trace = 0.0;
      for (std::size_t i = 0; i < 3; ++i) {
        trace += handoffExtrinsic[3 * i + i][point];
        for (std::size_t j = 0; j < 3; ++j) {
          const std::size_t component = 3 * i + j;
          const double delta = i == j ? 1.0 : 0.0;
          handoffAlgebraicError =
              std::max({handoffAlgebraicError,
                        std::abs(handoffMetric[component][point] - delta),
                        std::abs(handoffInverse[component][point] - delta),
                        std::abs(handoffExtrinsic[component][point] -
                                 handoffExtrinsic[3 * j + i][point])});
        }
        handoffAlgebraicError = std::max(handoffAlgebraicError,
                                         std::abs(handoffConnection[i][point]));
        handoffAlgebraicError =
            std::max(handoffAlgebraicError,
                     std::abs(handoffShift[i][point] - handoffGauge.shift[i]));
      }
      handoffAlgebraicError =
          std::max({handoffAlgebraicError, std::abs(trace),
                    std::abs(handoffMeanCurvature[point]),
                    std::abs(handoffLapse[point] - handoffGauge.lapse)});
    }
    for (std::size_t point = 3; point < handoffPointCount; ++point) {
      handoffAlgebraicError =
          std::max(handoffAlgebraicError, std::abs(handoffChi[point]));
      for (std::size_t component = 0; component < 9; ++component)
        handoffAlgebraicError =
            std::max(handoffAlgebraicError,
                     std::abs(handoffExtrinsic[component][point]));
    }
    const bool handoffPuncturesValid = std::isinf(handoffPsi[3]) &&
                                       std::isinf(handoffPsi[4]) &&
                                       std::isfinite(handoffCorrection[3]) &&
                                       std::isfinite(handoffCorrection[4]);

    const std::array<double, 3> constraintPoint = {0.2, 1.1, 0.7};
    const double differenceStep = 2.0e-3;
    const auto evaluateHandoff = [&](const std::array<double, 3> &point) {
      return evaluateTwoPunctureBssnPoint(publishedProblem, publishedFields[0],
                                          handoffParameters, point[0], point[1],
                                          point[2]);
    };
    const auto constraintCenter = evaluateHandoff(constraintPoint);
    double laplacianPsi = 0.0;
    for (std::size_t dim = 0; dim < 3; ++dim) {
      auto plus = constraintPoint;
      auto minus = constraintPoint;
      plus[dim] += differenceStep;
      minus[dim] -= differenceStep;
      laplacianPsi += (evaluateHandoff(plus).conformalFactor -
                       2.0 * constraintCenter.conformalFactor +
                       evaluateHandoff(minus).conformalFactor) /
                      (differenceStep * differenceStep);
    }
    const auto conformalExtrinsic = evaluateTwoPunctureBowenYorkTensor(
        constraintPoint[0], constraintPoint[1], constraintPoint[2],
        handoffParameters);
    double conformalExtrinsicSquared = 0.0;
    for (double component : conformalExtrinsic)
      conformalExtrinsicSquared += component * component;
    const double handoffHamiltonianResidual =
        laplacianPsi + 0.125 * conformalExtrinsicSquared /
                           std::pow(constraintCenter.conformalFactor, 7.0);

    double handoffMomentumResidual = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
      double divergence = 0.0;
      for (std::size_t j = 0; j < 3; ++j) {
        auto plus = constraintPoint;
        auto minus = constraintPoint;
        plus[j] += differenceStep;
        minus[j] -= differenceStep;
        const auto plusTensor = evaluateTwoPunctureBowenYorkTensor(
            plus[0], plus[1], plus[2], handoffParameters);
        const auto minusTensor = evaluateTwoPunctureBowenYorkTensor(
            minus[0], minus[1], minus[2], handoffParameters);
        divergence += (plusTensor[3 * i + j] - minusTensor[3 * i + j]) /
                      (2.0 * differenceStep);
      }
      handoffMomentumResidual =
          std::max(handoffMomentumResidual, std::abs(divergence));
    }

    auto spinningHandoffParameters = handoffParameters;
    spinningHandoffParameters.momenta = {};
    spinningHandoffParameters.spins = {{{0.03, -0.02, 0.05}, {0.0, 0.0, 0.0}}};
    const std::array<double, 3> spinProbe = {publishedHalfSeparation + 0.7,
                                             -0.4, 0.6};
    const auto spinTensor = evaluateTwoPunctureBowenYorkTensor(
        spinProbe[0], spinProbe[1], spinProbe[2], spinningHandoffParameters);
    const std::array<double, 3> spinDisplacement = {
        spinProbe[0] - publishedHalfSeparation, spinProbe[1], spinProbe[2]};
    const double spinRadius = std::hypot(
        spinDisplacement[0], spinDisplacement[1], spinDisplacement[2]);
    std::array<double, 3> spinNormal{};
    double spinSquared = 0.0;
    double spinNormalProduct = 0.0;
    for (std::size_t component = 0; component < 3; ++component) {
      spinNormal[component] = spinDisplacement[component] / spinRadius;
      const double spin = spinningHandoffParameters.spins[0][component];
      spinSquared += spin * spin;
      spinNormalProduct += spin * spinNormal[component];
    }
    double spinTensorSquared = 0.0;
    for (double component : spinTensor)
      spinTensorSquared += component * component;
    const double expectedSpinTensorSquared =
        18.0 * (spinSquared - spinNormalProduct * spinNormalProduct) /
        std::pow(spinRadius, 6.0);
    const double handoffSpinRelativeError =
        std::abs(spinTensorSquared - expectedSpinTensorSquared) /
        expectedSpinTensorSquared;

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
    std::printf("[two-puncture-hamiltonian] scalar regularity projector error "
                "= %.17g -> %.17g, idempotence %.17g\n",
                regularityErrorBefore, regularityErrorAfter,
                regularityIdempotenceError);
    std::printf("[two-puncture-hamiltonian] mapped sparse stencil nnz/width = "
                "%zu/%zu\n",
                mappedStencil.values.size(), maxStencilWidth);
    std::printf("[two-puncture-hamiltonian] boosted initial residual l2 = "
                "%.17g\n",
                initialResidual.l2Norm);
    std::printf("[two-puncture-hamiltonian] boosted final residual l2 = "
                "%.17g\n",
                finalResidual.l2Norm);
    std::printf("[two-puncture-hamiltonian] boosted final residual max = "
                "%.17g\n",
                finalResidual.maxAbs);
    std::printf(
        "[two-puncture-hamiltonian] coarse Newton status/steps/linear = "
        "%d/%d/%d, linear residual = %.17g\n",
        static_cast<int>(solveResult.status), solveResult.steps,
        solveResult.linearIterations, solveResult.finalLinearResidualL2);
    std::printf("[two-puncture-hamiltonian] fine Newton status/steps/linear = "
                "%d/%d/%d, linear residual = %.17g\n",
                static_cast<int>(fineSolveResult.status), fineSolveResult.steps,
                fineSolveResult.linearIterations,
                fineSolveResult.finalLinearResidualL2);
    std::printf("[two-puncture-hamiltonian] two-grid Newton "
                "status/steps/linear = %d/%d/%d, residual %.17g\n",
                static_cast<int>(multigridSolveResult.status),
                multigridSolveResult.steps,
                multigridSolveResult.linearIterations,
                multigridResidual.l2Norm);
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
    std::printf("[two-puncture-hamiltonian] fine scalar-axis regularity error "
                "= %.17g\n",
                fineRegularity.maxPhiVariation());
    std::printf("[two-puncture-hamiltonian] scalar-axis regularity samples "
                "= %.17g -> %.17g -> %.17g\n",
                veryCoarseRegularity.maxPhiVariation(),
                coarseRegularity.maxPhiVariation(),
                fineRegularity.maxPhiVariation());
    std::printf("[two-puncture-hamiltonian] coarse puncture ADM masses = "
                "%.17g %.17g\n",
                coarseLocalMasses.admMasses[0], coarseLocalMasses.admMasses[1]);
    std::printf("[two-puncture-hamiltonian] fine puncture ADM masses = "
                "%.17g %.17g\n",
                fineLocalMasses.admMasses[0], fineLocalMasses.admMasses[1]);
    std::printf("[two-puncture-hamiltonian] puncture-mass refinement delta = "
                "%.17g -> %.17g\n",
                veryCoarsePunctureMassDelta, punctureMassDelta);
    std::printf("[two-puncture-hamiltonian] puncture phi variation = "
                "%.17g %.17g\n",
                coarsePunctureSample.maxPhiVariation[0],
                coarsePunctureSample.maxPhiVariation[1]);
    std::printf("[two-puncture-hamiltonian] bare-mass calibration = "
                "%.17g %.17g in %d solves, max error %.17g\n",
                calibrationResult.bareMasses[0],
                calibrationResult.bareMasses[1], calibrationResult.iterations,
                calibrationResult.maxMassError);
    std::printf("[two-puncture-hamiltonian] published q=0.1 solve "
                "status/steps/linear = %d/%d/%d, residual %.17g\n",
                static_cast<int>(publishedSolve.status), publishedSolve.steps,
                publishedSolve.linearIterations, publishedResidual.l2Norm);
    std::printf("[two-puncture-hamiltonian] published q=0.1 observables = "
                "%.17g %.17g %.17g\n",
                publishedPunctureSample.values[0], publishedScaledHeavy,
                publishedScaledInfinity);
    std::printf("[two-puncture-hamiltonian] published q=0.1 max relative "
                "error = %.17g\n",
                publishedMaxRelativeError);
    std::printf("[two-puncture-hamiltonian] BSSN handoff logical/correction/"
                "algebraic error = %.17g %.17g %.17g\n",
                handoffLogicalError, handoffCorrectionError,
                handoffAlgebraicError);
    std::printf("[two-puncture-hamiltonian] BSSN handoff H/M residual = "
                "%.17g %.17g\n",
                handoffHamiltonianResidual, handoffMomentumResidual);
    std::printf("[two-puncture-hamiltonian] BSSN handoff Bowen-York spin "
                "relative error = %.17g\n",
                handoffSpinRelativeError);

    if (momentumContractionError > 2.0e-12 || spinContractionError > 2.0e-12 ||
        !publishedSolve.converged() ||
        !publishedSolve.usedGeneratedGridKernel ||
        !publishedSolve.usedMatrixFreeGMRES ||
        !publishedSolve.usedPreconditioner ||
        publishedSolve.usedFieldProjector || !publishedResidual.finite ||
        publishedResidual.l2Norm > 2.0e-8 ||
        publishedResidual.maxAbs > 2.0e-7 ||
        publishedMaxRelativeError > 5.0e-2 || !handoffPuncturesValid ||
        handoffLogicalError > 2.0e-12 || handoffCorrectionError > 2.0e-12 ||
        handoffAlgebraicError > 2.0e-12 ||
        std::abs(handoffHamiltonianResidual) > 1.0e-4 ||
        handoffMomentumResidual > 1.0e-6 ||
        handoffSpinRelativeError > 2.0e-12 || projectorErrorBefore < 1.0 ||
        projectorErrorAfter != 0.0 || projectorProbe[probeIndex] != -1.0 ||
        projectorProbe[probeImage] != -1.0 || regularityErrorBefore < 0.5 ||
        regularityErrorAfter > 2.0e-12 ||
        regularityIdempotenceError > 2.0e-12 ||
        fineRegularity.maxPhiVariation() > 5.0e-3 ||
        mappedStencil.size != grid.size() || maxStencilWidth > 7 ||
        mappedStencil.values.size() > 7 * grid.size() ||
        !(fineRegularity.maxPhiVariation() <
          coarseRegularity.maxPhiVariation()) ||
        algebraicMassError > 2.0e-14 || !calibrationResult.converged() ||
        !calibrationUsedMatrixFree ||
        std::abs(calibrationResult.bareMasses[0] - mass1) > 2.0e-6 ||
        std::abs(calibrationResult.bareMasses[1] - mass2) > 2.0e-6 ||
        calibrationMaxPhiVariation > 2.0e-3 ||
        !(punctureMassDelta < veryCoarsePunctureMassDelta) ||
        !initialResidual.finite || initialResidual.l2Norm < 1.0e-5 ||
        !solveResult.converged() || !solveResult.usedGeneratedGridKernel ||
        !solveResult.usedMatrixFreeGMRES || !solveResult.usedPreconditioner ||
        !solveResult.usedFieldProjector || !finalResidual.finite ||
        !finalResidual.usedGeneratedGridKernels ||
        finalResidual.l2Norm > 2.0e-8 || finalResidual.maxAbs > 2.0e-7 ||
        !(maxCorrection > 1.0e-8) || !(minPsi > 0.0) ||
        !(maxOuterCorrection < maxCorrection) || !fineSolveResult.converged() ||
        !fineSolveResult.usedMatrixFreeGMRES ||
        !fineSolveResult.usedPreconditioner ||
        !fineSolveResult.usedFieldProjector ||
        fineSolveResult.linearIterations <= options.gmresRestart ||
        !multigridSolveResult.converged() ||
        !multigridSolveResult.usedMatrixFreeGMRES ||
        !multigridSolveResult.usedPreconditioner ||
        multigridSolveResult.linearIterations >=
            fineSolveResult.linearIterations ||
        !multigridResidual.finite || multigridResidual.l2Norm > 2.0e-8 ||
        multigridResidual.maxAbs > 2.0e-7 ||
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
