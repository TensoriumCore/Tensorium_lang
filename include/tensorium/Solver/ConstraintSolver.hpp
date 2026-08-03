#pragma once

#include "tensorium/Backend/DomainIR.hpp"

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

// Coupled scalar/rank-one radial constraint backend with Chebyshev-Lobatto
// domains, optional compactified infinity, C0/C1 matching, and Newton solve.
ConstraintSolution
solveRadialConstraintProblem(const backend::ModuleIR &module,
                             const ConstraintSolveRequest &request = {});

} // namespace tensorium::solver
