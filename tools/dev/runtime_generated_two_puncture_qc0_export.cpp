#include "tensorium_mlir/Runtime/SpectralEllipticSolver.h"
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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace tensorium_mlir::runtime;

std::size_t parseSize(const char *text, const char *name, std::size_t minimum) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(text, &consumed);
  if (text[consumed] != '\0' || parsed < minimum ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return static_cast<std::size_t>(parsed);
}

double parsePositiveDouble(const char *text, const char *name) {
  std::size_t consumed = 0;
  const double parsed = std::stod(text, &consumed);
  if (text[consumed] != '\0' || !(parsed > 0.0) || !std::isfinite(parsed))
    throw std::runtime_error(std::string("invalid ") + name);
  return parsed;
}

template <std::size_t Components>
std::array<double *, Components>
allocateComponentPointers(std::array<std::vector<double>, Components> &storage,
                          std::size_t pointCount) {
  std::array<double *, Components> pointers{};
  for (std::size_t component = 0; component < Components; ++component) {
    storage[component].assign(pointCount, 0.0);
    pointers[component] = storage[component].data();
  }
  return pointers;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 1 && argc != 7) {
      std::cerr << "usage: " << argv[0]
                << " [output.csv nA nB nPhi slice_n half_width]\n";
      return 2;
    }

    const std::string outputPath =
        argc == 7 ? argv[1] : "tensorium_qc0_bssn_slice.csv";
    const std::size_t nA = argc == 7 ? parseSize(argv[2], "nA", 3) : 10;
    const std::size_t nB = argc == 7 ? parseSize(argv[3], "nB", 3) : 10;
    const std::size_t nPhi = argc == 7 ? parseSize(argv[4], "nPhi", 4) : 16;
    const std::size_t sliceN =
        argc == 7 ? parseSize(argv[5], "slice_n", 3) : 129;
    const double halfWidth =
        argc == 7 ? parsePositiveDouble(argv[6], "half_width") : 8.0;
    if (nPhi % 2 != 0)
      throw std::runtime_error("nPhi must be even for QC0 inversion symmetry");
    if (sliceN > std::numeric_limits<std::size_t>::max() / sliceN)
      throw std::runtime_error("slice_n is too large");

    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT == 1);
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT == 1);
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT == 1);
    const auto &systemDesc = tensorium_spectral_residual_systems[0];
    if (!systemDesc.symbol_name ||
        std::strcmp(systemDesc.symbol_name,
                    "SpectralTwoPunctureHamiltonian3D") != 0 ||
        systemDesc.unknown_count != 1 || systemDesc.equation_count != 1 ||
        systemDesc.equations[0].param_count != 15) {
      throw std::runtime_error("unexpected generated QC0 residual metadata");
    }

    // Published QC0 parameters used by the Einstein Toolkit QC0 setup.
    constexpr double halfSeparation = 1.168642873;
    constexpr double bareMassPlus = 0.453;
    constexpr double bareMassMinus = 0.453;
    constexpr double momentum = 0.3331917498;
    const std::array<double, 3> momentumPlus = {0.0, momentum, 0.0};
    const std::array<double, 3> momentumMinus = {0.0, -momentum, 0.0};
    const std::array<double, 3> zeroSpin = {0.0, 0.0, 0.0};

    std::array<double, 15> physicalParams{};
    const auto paramIndex = [&](const char *name) {
      const auto &equation = systemDesc.equations[0];
      for (std::int64_t index = 0; index < equation.param_count; ++index) {
        if (equation.param_names[index] &&
            std::strcmp(equation.param_names[index], name) == 0)
          return static_cast<std::size_t>(index);
      }
      throw std::runtime_error(std::string("missing generated parameter: ") +
                               name);
    };
    const auto setParam = [&](const char *name, double value) {
      physicalParams[paramIndex(name)] = value;
    };
    setParam("b", halfSeparation);
    setParam("m1", bareMassPlus);
    setParam("m2", bareMassMinus);
    setParam("p1y", momentumPlus[1]);
    setParam("p2y", momentumMinus[1]);

    SpectralGrid3D spectralGrid(SpectralAxis::chebyshevZeros(nA),
                                SpectralAxis::chebyshevZeros(nB),
                                SpectralAxis::fourierPeriodic(nPhi));
    const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
        systemInputs{{SpectralGeneratedResidualSystemEquationInputs{
            physicalParams, {}}}};
    auto generatedSystem = makeSpectralResidualSystemFromDesc(
        systemDesc, spectralGrid, tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT, systemInputs);
    auto &problem = generatedSystem.equations[0].problem;
    const std::array<double, 1> coordinateParams = {halfSeparation};
    const std::array<double, 3> unknownMapParams = {0.0, 1.0, 1.0};
    problem.coordinateMap = makeTwoPunctureCoordinateMap();
    problem.coordinateParams = coordinateParams;
    problem.derivativeMap = makeTwoPunctureDerivativeMap();
    problem.unknownMap = makeLinearBoundaryFactorUnknownMap();
    problem.unknownMapParams = unknownMapParams;
    problem.fieldProjector = makeTwoPunctureInversionEvenFieldProjector();
    const auto system = generatedSystem.view();

    SpectralEllipticSolveOptions solveOptions;
    solveOptions.maxNewtonSteps = 16;
    solveOptions.residualTolerance = 2.0e-8;
    solveOptions.residualRatioTarget = 1.0e-9;
    solveOptions.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
    solveOptions.denseJacobianMaxUnknowns = 1;
    solveOptions.gmresMaxIterations = 1024;
    solveOptions.gmresRestart = 64;
    // Use an inexact Newton forcing term. Early Krylov solves track the current
    // nonlinear residual; the independent nonlinear stopping test remains
    // strict and the absolute floor controls the final Newton steps.
    solveOptions.gmresTolerance = 1.0e-9;
    solveOptions.gmresRelativeTolerance = 2.0e-2;
    solveOptions.gmresPreconditioner =
        SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift;
    solveOptions.preconditionerRelaxationSweeps = 12;
    solveOptions.jvpOptions.relativeStep = 2.0e-6;
    solveOptions.jvpOptions.absoluteStep = 1.0e-8;
    solveOptions.linearPivotTolerance = 1.0e-13;

    std::array<std::vector<double>, 1> solverFields{
        std::vector<double>(spectralGrid.size(), 0.0)};
    const auto solveResult =
        solveSpectralNewton(system,
                            std::span<std::vector<double>>(solverFields.data(),
                                                           solverFields.size()),
                            solveOptions);
    const auto residual = assembleSpectralResidualSystem(
        system, std::span<const std::vector<double>>(solverFields.data(),
                                                     solverFields.size()));
    std::cout << std::setprecision(17)
              << "[qc0] solve status / steps / GMRES = "
              << static_cast<int>(solveResult.status) << " / "
              << solveResult.steps << " / " << solveResult.linearIterations
              << "\n"
              << "[qc0] final linear residual = "
              << solveResult.finalLinearResidualL2 << "\n"
              << "[qc0] solve residual L2 / max = " << residual.l2Norm << " / "
              << residual.maxAbs << "\n";
    if (!solveResult.converged() || !residual.finite ||
        residual.l2Norm > solveOptions.residualTolerance ||
        residual.maxAbs > 2.0e-6) {
      throw std::runtime_error("QC0 Hamiltonian solve did not converge");
    }

    const double vInfinity =
        spectralGrid.interpolate(solverFields[0], 1.0, 0.0, 0.0);
    const auto adm = makeTwoPunctureAdmDiagnostics(
        halfSeparation, bareMassPlus, bareMassMinus, vInfinity, momentumPlus,
        momentumMinus, zeroSpin, zeroSpin);
    const auto punctureSample =
        sampleTwoPunctureRegularField(spectralGrid, solverFields[0]);
    const auto localMasses = makeTwoPunctureLocalMassDiagnostics(
        halfSeparation, bareMassPlus, bareMassMinus, punctureSample.values[0],
        punctureSample.values[1]);
    const auto regularity =
        measureTwoPunctureScalarRegularity(spectralGrid, solverFields[0]);
    if (regularity.maxPhiVariation() > 1.0e-3)
      throw std::runtime_error("QC0 axis regularity check failed");

    const std::size_t targetPointCount = sliceN * sliceN;
    std::vector<double> x(targetPointCount, 0.0);
    std::vector<double> y(targetPointCount, 0.0);
    std::vector<double> z(targetPointCount, 0.0);
    const double spacing = 2.0 * halfWidth / static_cast<double>(sliceN - 1);
    for (std::size_t i = 0; i < sliceN; ++i) {
      for (std::size_t j = 0; j < sliceN; ++j) {
        const std::size_t point = i * sliceN + j;
        x[point] = -halfWidth + spacing * static_cast<double>(i);
        y[point] = -halfWidth + spacing * static_cast<double>(j);
      }
    }

    std::vector<double> chi(targetPointCount, 0.0);
    std::vector<double> meanCurvature(targetPointCount, 0.0);
    std::vector<double> correction(targetPointCount, 0.0);
    std::vector<double> psi(targetPointCount, 0.0);
    std::array<std::vector<double>, 9> conformalMetric;
    std::array<std::vector<double>, 9> inverseConformalMetric;
    std::array<std::vector<double>, 9> traceFreeExtrinsicCurvature;
    std::array<std::vector<double>, 3> conformalConnection;
    std::array<std::vector<double>, 3> shift;
    TwoPunctureBssnGridBuffers outputs;
    outputs.chi = chi.data();
    outputs.meanCurvature = meanCurvature.data();
    outputs.regularCorrection = correction.data();
    outputs.conformalFactor = psi.data();
    outputs.conformalMetric =
        allocateComponentPointers(conformalMetric, targetPointCount);
    outputs.inverseConformalMetric =
        allocateComponentPointers(inverseConformalMetric, targetPointCount);
    outputs.traceFreeExtrinsicCurvature = allocateComponentPointers(
        traceFreeExtrinsicCurvature, targetPointCount);
    outputs.conformalConnection =
        allocateComponentPointers(conformalConnection, targetPointCount);
    outputs.shift = allocateComponentPointers(shift, targetPointCount);

    TwoPuncturePhysicalParameters handoffParameters;
    handoffParameters.halfSeparation = halfSeparation;
    handoffParameters.bareMasses = {bareMassPlus, bareMassMinus};
    handoffParameters.momenta = {momentumPlus, momentumMinus};
    handoffParameters.spins = {zeroSpin, zeroSpin};
    interpolateTwoPunctureBssnToCartesianGrid(
        problem, solverFields[0], handoffParameters,
        TwoPunctureCartesianGridView{targetPointCount,
                                     {x.data(), y.data(), z.data()}},
        outputs);

    std::vector<double> lapse(targetPointCount, 0.0);
    double maxTrace = 0.0;
    double minChi = std::numeric_limits<double>::infinity();
    double maxChi = 0.0;
    for (std::size_t point = 0; point < targetPointCount; ++point) {
      lapse[point] = std::sqrt(chi[point]);
      minChi = std::min(minChi, chi[point]);
      maxChi = std::max(maxChi, chi[point]);
      const double trace = traceFreeExtrinsicCurvature[0][point] +
                           traceFreeExtrinsicCurvature[4][point] +
                           traceFreeExtrinsicCurvature[8][point];
      maxTrace = std::max(maxTrace, std::abs(trace));
      if (!std::isfinite(chi[point]) || !std::isfinite(lapse[point]) ||
          !std::isfinite(trace)) {
        throw std::runtime_error(
            "QC0 handoff produced a non-finite BSSN field");
      }
    }

    std::ofstream csv(outputPath);
    if (!csv)
      throw std::runtime_error("cannot open QC0 CSV output: " + outputPath);
    csv << std::setprecision(17);
    csv << "i,j,x,y,z,u,psi,chi,alpha,gammatilde_xx,gammatilde_xy,"
           "gammatilde_xz,gammatilde_yy,gammatilde_yz,gammatilde_zz,"
           "Atilde_xx,Atilde_xy,Atilde_xz,Atilde_yy,Atilde_yz,Atilde_zz,K,"
           "Gamma_x,Gamma_y,Gamma_z,beta_x,beta_y,beta_z\n";
    for (std::size_t i = 0; i < sliceN; ++i) {
      for (std::size_t j = 0; j < sliceN; ++j) {
        const std::size_t point = i * sliceN + j;
        csv << i << ',' << j << ',' << x[point] << ',' << y[point] << ','
            << z[point] << ',' << correction[point] << ',' << psi[point] << ','
            << chi[point] << ',' << lapse[point] << ','
            << conformalMetric[0][point] << ',' << conformalMetric[1][point]
            << ',' << conformalMetric[2][point] << ','
            << conformalMetric[4][point] << ',' << conformalMetric[5][point]
            << ',' << conformalMetric[8][point] << ','
            << traceFreeExtrinsicCurvature[0][point] << ','
            << traceFreeExtrinsicCurvature[1][point] << ','
            << traceFreeExtrinsicCurvature[2][point] << ','
            << traceFreeExtrinsicCurvature[4][point] << ','
            << traceFreeExtrinsicCurvature[5][point] << ','
            << traceFreeExtrinsicCurvature[8][point] << ','
            << meanCurvature[point] << ',' << conformalConnection[0][point]
            << ',' << conformalConnection[1][point] << ','
            << conformalConnection[2][point] << ',' << shift[0][point] << ','
            << shift[1][point] << ',' << shift[2][point] << '\n';
      }
    }
    csv.close();
    if (!csv)
      throw std::runtime_error("failed while writing QC0 CSV output");

    const std::string metadataPath = outputPath + ".json";
    std::ofstream metadata(metadataPath);
    if (!metadata)
      throw std::runtime_error("cannot open QC0 metadata output: " +
                               metadataPath);
    metadata << std::setprecision(17);
    metadata << "{\n"
             << "  \"case\": \"QC0\",\n"
             << "  \"formulation\": \"Bowen-York puncture / BSSN\",\n"
             << "  \"half_separation\": " << halfSeparation << ",\n"
             << "  \"bare_masses\": [" << bareMassPlus << ", " << bareMassMinus
             << "],\n"
             << "  \"momenta\": [[0, " << momentum << ", 0], [0, " << -momentum
             << ", 0]],\n"
             << "  \"spins\": [[0, 0, 0], [0, 0, 0]],\n"
             << "  \"spectral_resolution\": [" << nA << ", " << nB << ", "
             << nPhi << "],\n"
             << "  \"slice_resolution\": [" << sliceN << ", " << sliceN
             << ", 1],\n"
             << "  \"slice_half_width\": " << halfWidth << ",\n"
             << "  \"slice_spacing\": " << spacing << ",\n"
             << "  \"newton_steps\": " << solveResult.steps << ",\n"
             << "  \"linear_iterations\": " << solveResult.linearIterations
             << ",\n"
             << "  \"residual_l2\": " << residual.l2Norm << ",\n"
             << "  \"residual_max\": " << residual.maxAbs << ",\n"
             << "  \"adm_energy\": " << adm.energy << ",\n"
             << "  \"adm_linear_momentum\": [" << adm.linearMomentum[0] << ", "
             << adm.linearMomentum[1] << ", " << adm.linearMomentum[2] << "],\n"
             << "  \"adm_angular_momentum\": [" << adm.angularMomentum[0]
             << ", " << adm.angularMomentum[1] << ", " << adm.angularMomentum[2]
             << "],\n"
             << "  \"puncture_adm_masses\": [" << localMasses.admMasses[0]
             << ", " << localMasses.admMasses[1] << "],\n"
             << "  \"axis_regularity_error\": " << regularity.maxPhiVariation()
             << ",\n"
             << "  \"bssn_trace_error\": " << maxTrace << ",\n"
             << "  \"gauge\": {\"lapse\": \"psi^-2\", \"shift\": [0, 0, 0]},\n"
             << "  \"fields\": [\"chi\", \"alpha\", \"gammatilde_ij\", "
                "\"Atilde_ij\", \"K\", \"Gamma_i\", \"beta_i\"],\n"
             << "  \"layout\": \"Cartesian z=0 slice, row-major i*slice_n+j\"\n"
             << "}\n";
    metadata.close();
    if (!metadata)
      throw std::runtime_error("failed while writing QC0 metadata output");

    std::cout << std::setprecision(17) << "[qc0] spectral grid = " << nA << 'x'
              << nB << 'x' << nPhi << " (" << spectralGrid.size()
              << " unknowns)\n"
              << "[qc0] Newton steps / GMRES iterations = " << solveResult.steps
              << " / " << solveResult.linearIterations << "\n"
              << "[qc0] residual L2 / max = " << residual.l2Norm << " / "
              << residual.maxAbs << "\n"
              << "[qc0] ADM energy / Jz = " << adm.energy << " / "
              << adm.angularMomentum[2] << "\n"
              << "[qc0] puncture ADM masses = " << localMasses.admMasses[0]
              << " " << localMasses.admMasses[1] << "\n"
              << "[qc0] axis regularity / BSSN trace error = "
              << regularity.maxPhiVariation() << " / " << maxTrace << "\n"
              << "[qc0] chi range on slice = [" << minChi << ", " << maxChi
              << "]\n"
              << "[qc0] CSV = " << outputPath << "\n"
              << "[qc0] metadata = " << metadataPath << '\n';
  } catch (const std::exception &error) {
    std::cerr << "QC0 export failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
