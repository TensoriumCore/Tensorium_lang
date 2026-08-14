#include "tensorium_mlir/Runtime/GeneratedInitialDataIO.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parseSize(const char *text, const char *name, std::size_t minimum) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(text, &consumed);
  if (text[consumed] != '\0' || parsed < minimum ||
      parsed > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<std::size_t>(parsed);
}

double parsePositiveDouble(const char *text, const char *name) {
  std::size_t consumed = 0;
  const double parsed = std::stod(text, &consumed);
  if (text[consumed] != '\0' || !(parsed > 0.0) || !std::isfinite(parsed))
    throw std::runtime_error(std::string("invalid ") + name);
  return parsed;
}

std::array<std::size_t, 3> parseResolution3D(const char *text) {
  std::array<std::size_t, 3> resolution{};
  if (!text || text[0] == '\0')
    return resolution;
  std::string normalized(text);
  for (char &character : normalized) {
    if (character == 'x' || character == 'X' || character == ',')
      character = ' ';
  }
  std::istringstream input(normalized);
  std::string trailing;
  if (!(input >> resolution[0] >> resolution[1] >> resolution[2]) ||
      (input >> trailing) ||
      std::any_of(resolution.begin(), resolution.end(),
                  [](std::size_t value) { return value < 3; })) {
    throw std::runtime_error(
        "invalid initial_data resolution; expected N1xN2xN3 with values >= 3");
  }
  return resolution;
}

std::vector<std::array<std::size_t, 3>>
parseContinuationResolutions(const char *text) {
  std::vector<std::array<std::size_t, 3>> resolutions;
  if (!text || text[0] == '\0')
    return resolutions;
  std::istringstream input(text);
  std::string stage;
  while (std::getline(input, stage, ';')) {
    if (stage.empty())
      throw std::runtime_error(
          "invalid initial_data continuation; empty resolution stage");
    resolutions.push_back(parseResolution3D(stage.c_str()));
  }
  return resolutions;
}

bool hasResolution(const std::array<std::size_t, 3> &resolution) {
  return std::any_of(resolution.begin(), resolution.end(),
                     [](std::size_t value) { return value != 0; });
}

} // namespace

int main(int argc, char **argv) {
  using namespace tensorium_mlir::runtime;
  try {
    if (argc != 1 && argc != 4) {
      std::cerr << "usage: " << argv[0] << " [output.csv slice_n half_width]\n";
      return 2;
    }

    static_assert(TENSORIUM_SPECTRAL_INITIAL_DATA_COUNT == 1);
    const std::string outputPath =
        argc == 4 ? argv[1] : "tensorium_qc0_bssn_slice.csv";
    GeneratedInitialDataSliceOptions slice;
    if (argc == 4) {
      slice.resolution = parseSize(argv[2], "slice_n", 3);
      slice.halfWidth = parsePositiveDouble(argv[3], "half_width");
    }

    const auto solveStart = std::chrono::steady_clock::now();
    const char *preconditionerOverride =
        std::getenv("TENSORIUM_INITIAL_DATA_PRECONDITIONER");
    const char *preconditionerSweepsText =
        std::getenv("TENSORIUM_INITIAL_DATA_PRECONDITIONER_SWEEPS");
    const auto resolutionOverride =
        parseResolution3D(std::getenv("TENSORIUM_INITIAL_DATA_RESOLUTION"));
    auto continuationResolutions = parseContinuationResolutions(
        std::getenv("TENSORIUM_INITIAL_DATA_CONTINUATION"));
    if (!continuationResolutions.empty() && hasResolution(resolutionOverride) &&
        continuationResolutions.back() != resolutionOverride) {
      continuationResolutions.push_back(resolutionOverride);
    }
    int preconditionerSweepsOverride = 0;
    if (preconditionerSweepsText && preconditionerSweepsText[0] != '\0') {
      const std::size_t parsedSweeps =
          parseSize(preconditionerSweepsText, "preconditioner_sweeps", 1);
      if (parsedSweeps >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("invalid preconditioner_sweeps");
      preconditionerSweepsOverride = static_cast<int>(parsedSweeps);
    }
    auto solution =
        continuationResolutions.empty()
            ? solveGeneratedSpectralInitialData(
                  tensorium_spectral_initial_data[0],
                  tensorium_spectral_residual_systems,
                  TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
                  tensorium_spectral_residual_kernels,
                  TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
                  tensorium_spectral_residual_grid_kernels,
                  TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT, {},
                  preconditionerOverride, preconditionerSweepsOverride,
                  resolutionOverride)
            : solveGeneratedSpectralInitialDataContinuation(
                  tensorium_spectral_initial_data[0],
                  tensorium_spectral_residual_systems,
                  TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
                  tensorium_spectral_residual_kernels,
                  TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
                  tensorium_spectral_residual_grid_kernels,
                  TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
                  continuationResolutions, {}, preconditionerOverride,
                  preconditionerSweepsOverride);
    const auto solveEnd = std::chrono::steady_clock::now();
    const double solveSeconds =
        std::chrono::duration<double>(solveEnd - solveStart).count();
    solution.solveWallSeconds = solveSeconds;
    const std::size_t rejectedPointIndex =
        solution.projectedOutResidualMaxIndex % solution.grid->size();
    const std::size_t rejectedI = rejectedPointIndex % solution.grid->n1();
    const std::size_t rejectedLine = rejectedPointIndex / solution.grid->n1();
    const std::size_t rejectedJ = rejectedLine % solution.grid->n2();
    const std::size_t rejectedK = rejectedLine / solution.grid->n2();
    const auto rejectedPoint =
        solution.grid->point(rejectedI, rejectedJ, rejectedK);
    const auto residualMaxIterator = std::max_element(
        solution.residual.values.begin(), solution.residual.values.end(),
        [](double lhs, double rhs) { return std::abs(lhs) < std::abs(rhs); });
    const std::size_t residualMaxFlatIndex =
        residualMaxIterator == solution.residual.values.end()
            ? 0
            : static_cast<std::size_t>(std::distance(
                  solution.residual.values.begin(), residualMaxIterator));
    const std::size_t residualMaxPointIndex =
        residualMaxFlatIndex % solution.grid->size();
    const std::size_t residualMaxI =
        residualMaxPointIndex % solution.grid->n1();
    const std::size_t residualMaxLine =
        residualMaxPointIndex / solution.grid->n1();
    const std::size_t residualMaxJ = residualMaxLine % solution.grid->n2();
    const std::size_t residualMaxK = residualMaxLine / solution.grid->n2();
    const auto residualMaxPoint =
        solution.grid->point(residualMaxI, residualMaxJ, residualMaxK);
    const bool usesGeneratedJvp = std::all_of(
        solution.generatedSystem.equations.begin(),
        solution.generatedSystem.equations.end(), [](const auto &equation) {
          return equation.problem.kernel.evaluateJvp != nullptr;
        });

    for (std::size_t stage = 0; stage < solution.continuationStages.size();
         ++stage) {
      const auto &report = solution.continuationStages[stage];
      std::cout << std::setprecision(17) << "[initial_data] continuation stage "
                << (stage + 1) << '/' << solution.continuationStages.size()
                << " = " << report.resolution[0] << 'x' << report.resolution[1]
                << 'x' << report.resolution[2] << ", status "
                << static_cast<int>(report.solveResult.status) << ", Newton "
                << report.solveResult.steps << ", linear "
                << report.solveResult.linearIterations << ", residual "
                << report.solveResult.initialResidualL2 << " -> "
                << report.residualL2 << ", wall " << report.solveWallSeconds
                << " s\n";
    }

    std::cout
        << std::setprecision(17) << "[initial_data] case = "
        << tensorium_spectral_initial_data[0].symbol_name << '\n'
        << "[initial_data] solve status / steps / linear iterations = "
        << static_cast<int>(solution.solveResult.status) << " / "
        << solution.solveResult.steps << " / "
        << solution.solveResult.linearIterations << '\n'
        << "[initial_data] final linear residual L2 = "
        << solution.solveResult.finalLinearResidualL2 << '\n'
        << "[initial_data] JVP = "
        << (usesGeneratedJvp ? "compiled forward mode"
                             : "finite-difference fallback")
        << '\n'
        << "[initial_data] preconditioner = "
        << (preconditionerOverride && preconditionerOverride[0] != '\0'
                ? preconditionerOverride
                : tensorium_spectral_initial_data[0].preconditioner)
        << '\n'
        << "[initial_data] preconditioner sweeps = "
        << (preconditionerSweepsOverride > 0
                ? preconditionerSweepsOverride
                : tensorium_spectral_initial_data[0].preconditioner_sweeps)
        << '\n'
        << "[initial_data] solve wall time = " << solveSeconds << " s\n"
        << "[initial_data] residual L2 / max = " << solution.residual.l2Norm
        << " / " << solution.residual.maxAbs << '\n'
        << "[initial_data] residual max logical point = ["
        << residualMaxPoint.x1 << ", " << residualMaxPoint.x2 << ", "
        << residualMaxPoint.x3 << "]\n"
        << "[initial_data] raw residual L2 / max = "
        << solution.rawResidual.l2Norm << " / " << solution.rawResidual.maxAbs
        << '\n'
        << "[initial_data] projected-out residual L2 / max = "
        << solution.projectedOutResidualL2 << " / "
        << solution.projectedOutResidualMaxAbs << '\n'
        << "[initial_data] projected-out max logical point = ["
        << rejectedPoint.x1 << ", " << rejectedPoint.x2 << ", "
        << rejectedPoint.x3 << "]\n";
    if (!solution.converged() ||
        solution.residual.l2Norm > solution.options.residualTolerance)
      throw std::runtime_error("generated initial_data solve did not converge");

    const std::string reconstruction =
        tensorium_spectral_initial_data[0].reconstruction;
    const auto exportStart = std::chrono::steady_clock::now();
    const auto report =
        reconstruction == "none"
            ? exportGeneratedInitialDataCollocationCsv(solution, outputPath)
            : exportGeneratedInitialDataBssnSlice(solution, outputPath, slice);
    const auto exportEnd = std::chrono::steady_clock::now();
    const double exportSeconds =
        std::chrono::duration<double>(exportEnd - exportStart).count();
    std::cout << "[initial_data] spectral grid = " << solution.grid->n1() << 'x'
              << solution.grid->n2() << 'x' << solution.grid->n3() << " ("
              << solution.grid->size() << " points per field)\n";
    if (reconstruction == "two_puncture_bssn") {
      std::cout << "[initial_data] ADM energy / Jz = " << report.admEnergy
                << " / " << report.admAngularMomentum[2] << '\n'
                << "[initial_data] puncture ADM masses = "
                << report.punctureAdmMasses[0] << ' '
                << report.punctureAdmMasses[1] << '\n'
                << "[initial_data] axis regularity / BSSN trace error = "
                << report.regularityError << " / " << report.bssnTraceError
                << '\n'
                << "[initial_data] chi range on slice = [" << report.minChi
                << ", " << report.maxChi << "]\n";
    }
    std::cout << "[initial_data] export wall time = " << exportSeconds
              << " s\n";
    std::cout << "[initial_data] CSV = " << report.csvPath << '\n'
              << "[initial_data] metadata = " << report.metadataPath << '\n';
  } catch (const std::exception &error) {
    std::cerr << "initial_data export failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
