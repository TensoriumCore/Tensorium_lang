#pragma once

#include "tensorium/Backend/DomainIR.hpp"

#include <cstddef>
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

struct ConstraintSolution {
  bool converged = false;
  std::size_t iterations = 0;
  double residualNorm = 0.0;
  std::vector<double> coordinates;
  std::vector<ConstraintDomainSolution> domains;
  std::unordered_map<std::string, std::vector<double>> unknowns;
  std::vector<double> residualHistory;
};

// Scalar radial constraint backend with Chebyshev-Lobatto shell domains,
// optional compactified infinity, C0/C1 interface matching, and Newton solve.
ConstraintSolution
solveRadialConstraintProblem(const backend::ModuleIR &module,
                             const ConstraintSolveRequest &request = {});

} // namespace tensorium::solver
