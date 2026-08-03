#pragma once

#include "tensorium/Backend/DomainIR.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorium::solver {

struct ConstraintSolveRequest {
  std::unordered_map<std::string, double> parameters;
};

struct ConstraintDomainSolution {
  std::string name;
  std::size_t offset = 0;
  std::size_t pointCount = 0;
  bool compactified = false;
};

struct ConstraintUnknownSolution {
  std::string name;
  std::size_t contravariantRank = 0;
  std::size_t covariantRank = 0;
  std::size_t componentCount = 1;
  std::size_t pointsPerComponent = 0;
};

struct RadialCttPhysicalSolution {
  std::string basis = "flat_spherical_orthonormal_coframe";
  std::string conformalFactorUnknown;
  std::string radialVectorPotentialUnknown;
  std::vector<double> meanCurvature;
  std::vector<double> spatialMetricRadial;
  std::vector<double> spatialMetricTangential;
  std::vector<double> extrinsicCurvatureRadial;
  std::vector<double> extrinsicCurvatureTangential;
};

struct ConstraintSolution {
  bool converged = false;
  std::size_t iterations = 0;
  double residualNorm = 0.0;
  std::vector<double> coordinates;
  std::vector<ConstraintDomainSolution> domains;
  std::vector<ConstraintUnknownSolution> unknownLayouts;
  std::unordered_map<std::string, std::vector<double>> unknowns;
  std::optional<RadialCttPhysicalSolution> physicalCtt;
  std::vector<double> residualHistory;
};

enum class CttTargetCoordinates { Spherical, Cartesian };

struct CttTargetGrid {
  CttTargetCoordinates coordinates = CttTargetCoordinates::Spherical;
  std::size_t pointCount = 0;
  // Spherical: r, theta, phi. Cartesian: x, y, z.
  std::array<const double *, 3> coordinateComponents{};
};

struct CttEvolutionBuffers {
  // Structure-of-arrays, row-major tensor component: component = 3*i + j.
  std::array<double *, 9> spatialMetric{};
  std::array<double *, 9> inverseSpatialMetric{};
  std::array<double *, 9> extrinsicCurvature{};
  double *meanCurvature = nullptr;
};

// Coupled scalar/rank-one radial constraint backend with Chebyshev-Lobatto
// domains, optional compactified infinity, C0/C1 matching, and Newton solve.
ConstraintSolution
solveRadialConstraintProblem(const backend::ModuleIR &module,
                             const ConstraintSolveRequest &request = {});

// Spectrally interpolates reconstructed radial CTT profiles and lifts them to
// full 3x3 physical tensors on a spherical or Cartesian target grid.
void interpolateRadialCttToGrid(const ConstraintSolution &solution,
                                const CttTargetGrid &target,
                                const CttEvolutionBuffers &outputs);

} // namespace tensorium::solver
