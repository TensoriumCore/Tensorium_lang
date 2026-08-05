#include "tensorium_mlir/Runtime/GeneratedInitialDataIO.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::size_t parseSize(const char *text, const char *name,
                      std::size_t minimum) {
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

} // namespace

int main(int argc, char **argv) {
  using namespace tensorium_mlir::runtime;
  try {
    if (argc != 1 && argc != 4) {
      std::cerr << "usage: " << argv[0]
                << " [output.csv slice_n half_width]\n";
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
    auto solution = solveGeneratedSpectralInitialData(
        tensorium_spectral_initial_data[0],
        tensorium_spectral_residual_systems,
        TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
        tensorium_spectral_residual_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
        tensorium_spectral_residual_grid_kernels,
        TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT);
    const auto solveEnd = std::chrono::steady_clock::now();
    const double solveSeconds =
        std::chrono::duration<double>(solveEnd - solveStart).count();

    std::cout << std::setprecision(17)
              << "[initial_data] case = "
              << tensorium_spectral_initial_data[0].symbol_name << '\n'
              << "[initial_data] solve status / steps / linear iterations = "
              << static_cast<int>(solution.solveResult.status) << " / "
              << solution.solveResult.steps << " / "
              << solution.solveResult.linearIterations << '\n'
              << "[initial_data] solve wall time = " << solveSeconds
              << " s\n"
              << "[initial_data] residual L2 / max = "
              << solution.residual.l2Norm << " / "
              << solution.residual.maxAbs << '\n';
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
    std::cout << "[initial_data] spectral grid = " << solution.grid->n1()
              << 'x' << solution.grid->n2() << 'x' << solution.grid->n3()
              << " (" << solution.grid->size() << " points per field)\n";
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
