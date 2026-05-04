#include "tensorium/Sema/Sema.hpp"

#include <stdexcept>
#include <string>

namespace tensorium {

void SemanticAnalyzer::validateSimulation(const SimulationConfig &sim) {
  if (sim.dimension <= 0) {
    throw std::runtime_error("simulation dimension must be >= 1");
  }

  if ((int)sim.resolution.size() != sim.dimension) {
    throw std::runtime_error(
        "resolution size (" + std::to_string(sim.resolution.size()) +
        ") does not match dimension (" + std::to_string(sim.dimension) + ")");
  }

  for (int r : sim.resolution) {
    if (r <= 0)
      throw std::runtime_error("resolution entries must be > 0");
  }

  if (sim.time.dt <= 0.0) {
    throw std::runtime_error("time.dt must be > 0");
  }

  if (sim.spatial.scheme == SpatialScheme::FiniteDifference) {
    if (sim.spatial.order < 2)
      throw std::runtime_error("FD order must be >= 2");

    if (sim.spatial.order % 2 != 0)
      throw std::runtime_error("FD order must be even");
  }

  if (sim.spatial.scheme == SpatialScheme::Spectral) {
    if (sim.spatial.order != 0) {
      throw std::runtime_error(
          "spectral scheme does not use finite-difference order");
    }
  }
}

} // namespace tensorium
