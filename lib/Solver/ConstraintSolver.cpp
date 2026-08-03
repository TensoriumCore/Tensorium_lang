#include "tensorium/Solver/ConstraintSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensorium::solver {
namespace {

using Matrix = std::vector<double>;

struct RadialGrid {
  std::string name;
  std::size_t size = 0;
  bool compactified = false;
  std::vector<double> radius;
  Matrix firstDerivative;
  Matrix secondDerivative;
};

struct DualGrid {
  std::vector<double> value;
  std::vector<double> tangent;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("constraint solver: " + message);
}

double &at(Matrix &matrix, std::size_t n, std::size_t row, std::size_t col) {
  return matrix[row * n + col];
}

double at(const Matrix &matrix, std::size_t n, std::size_t row,
          std::size_t col) {
  return matrix[row * n + col];
}

std::vector<double> applyMatrix(const Matrix &matrix, std::size_t n,
                                const std::vector<double> &values) {
  if (values.size() != n)
    fail("internal matrix/vector size mismatch");
  std::vector<double> result(n, 0.0);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t col = 0; col < n; ++col)
      result[row] += at(matrix, n, row, col) * values[col];
  return result;
}

Matrix multiplyMatrices(const Matrix &lhs, const Matrix &rhs, std::size_t n) {
  Matrix result(n * n, 0.0);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t k = 0; k < n; ++k) {
      const double left = at(lhs, n, row, k);
      for (std::size_t col = 0; col < n; ++col)
        at(result, n, row, col) += left * at(rhs, n, k, col);
    }
  }
  return result;
}

RadialGrid buildChebyshevLobattoGrid(const backend::SpectralDomainIR &domain) {
  if (domain.resolution.size() != 1)
    fail("radial backend requires a one-dimensional domain resolution");
  const int requestedSize = domain.resolution.front();
  if (requestedSize < 3)
    fail("Chebyshev-Lobatto resolution must be at least 3");
  if (requestedSize > 257)
    fail("dense radial backend limits resolution to 257 points");
  RadialGrid grid;
  grid.name = domain.name;
  grid.size = static_cast<std::size_t>(requestedSize);
  grid.compactified = domain.topology == "compactified";
  grid.radius.resize(grid.size);
  const double pi = std::acos(-1.0);
  std::vector<double> spectralCoordinate(grid.size);
  for (std::size_t i = 0; i < grid.size; ++i) {
    spectralCoordinate[i] = -std::cos(pi * static_cast<double>(i) /
                                      static_cast<double>(grid.size - 1));
  }

  // Chebyshev-Lobatto barycentric weights. The common scale cancels in the
  // differentiation matrix, so these remain stable at high resolution and
  // for wide physical shells.
  std::vector<double> weights(grid.size, 1.0);
  for (std::size_t i = 0; i < grid.size; ++i) {
    const double endpointScale = (i == 0 || i + 1 == grid.size) ? 0.5 : 1.0;
    weights[i] = (i % 2 == 0 ? 1.0 : -1.0) * endpointScale;
  }

  Matrix spectralFirst(grid.size * grid.size, 0.0);
  for (std::size_t row = 0; row < grid.size; ++row) {
    double offDiagonalSum = 0.0;
    for (std::size_t col = 0; col < grid.size; ++col) {
      if (row == col)
        continue;
      const double entry =
          weights[col] /
          (weights[row] * (spectralCoordinate[row] - spectralCoordinate[col]));
      at(spectralFirst, grid.size, row, col) = entry;
      offDiagonalSum += entry;
    }
    at(spectralFirst, grid.size, row, row) = -offDiagonalSum;
  }
  const Matrix spectralSecond =
      multiplyMatrices(spectralFirst, spectralFirst, grid.size);

  grid.firstDerivative.assign(grid.size * grid.size, 0.0);
  grid.secondDerivative.assign(grid.size * grid.size, 0.0);
  if (grid.compactified) {
    if (domain.bounds.size() != 1)
      fail("compactified radial domain requires bounds = [inner_radius]");
    const double innerRadius = domain.bounds[0];
    for (std::size_t row = 0; row < grid.size; ++row) {
      const double oneMinusX = 1.0 - spectralCoordinate[row];
      grid.radius[row] = oneMinusX == 0.0
                             ? std::numeric_limits<double>::infinity()
                             : 2.0 * innerRadius / oneMinusX;
      const double dxDr = (oneMinusX * oneMinusX) / (2.0 * innerRadius);
      const double d2xDr2 = -(oneMinusX * oneMinusX * oneMinusX) /
                            (2.0 * innerRadius * innerRadius);
      for (std::size_t col = 0; col < grid.size; ++col) {
        at(grid.firstDerivative, grid.size, row, col) =
            dxDr * at(spectralFirst, grid.size, row, col);
        at(grid.secondDerivative, grid.size, row, col) =
            dxDr * dxDr * at(spectralSecond, grid.size, row, col) +
            d2xDr2 * at(spectralFirst, grid.size, row, col);
      }
    }
  } else {
    if (domain.bounds.size() != 2)
      fail("finite radial domain requires bounds = [r_min, r_max]");
    const double lower = domain.bounds[0];
    const double upper = domain.bounds[1];
    const double midpoint = 0.5 * (lower + upper);
    const double halfWidth = 0.5 * (upper - lower);
    for (std::size_t row = 0; row < grid.size; ++row) {
      grid.radius[row] = midpoint + halfWidth * spectralCoordinate[row];
      for (std::size_t col = 0; col < grid.size; ++col) {
        at(grid.firstDerivative, grid.size, row, col) =
            at(spectralFirst, grid.size, row, col) / halfWidth;
        at(grid.secondDerivative, grid.size, row, col) =
            at(spectralSecond, grid.size, row, col) / (halfWidth * halfWidth);
      }
    }
  }
  return grid;
}

DualGrid constantGrid(std::size_t size, double value) {
  return {std::vector<double>(size, value), std::vector<double>(size, 0.0)};
}

DualGrid applyRadialLaplacian(const DualGrid &input, const RadialGrid &grid) {
  DualGrid result;
  const auto first = applyMatrix(grid.firstDerivative, grid.size, input.value);
  const auto second =
      applyMatrix(grid.secondDerivative, grid.size, input.value);
  const auto tangentFirst =
      applyMatrix(grid.firstDerivative, grid.size, input.tangent);
  const auto tangentSecond =
      applyMatrix(grid.secondDerivative, grid.size, input.tangent);
  result.value.resize(grid.size);
  result.tangent.resize(grid.size);
  for (std::size_t i = 0; i < grid.size; ++i) {
    result.value[i] = second[i] + (2.0 / grid.radius[i]) * first[i];
    result.tangent[i] =
        tangentSecond[i] + (2.0 / grid.radius[i]) * tangentFirst[i];
  }
  return result;
}

DualGrid evalExpr(const backend::ExprIR *expr, const RadialGrid &grid,
                  const std::string &unknownName,
                  const std::vector<double> &unknown,
                  const std::vector<double> &unknownTangent,
                  const std::unordered_map<std::string, double> &parameters) {
  using backend::ExprIR;
  if (!expr)
    fail("encountered a null expression");

  switch (expr->kind) {
  case ExprIR::Kind::Number: {
    const auto *number = static_cast<const backend::NumberIR *>(expr);
    return constantGrid(grid.size, number->value);
  }
  case ExprIR::Kind::Var: {
    const auto *variable = static_cast<const backend::VarIR *>(expr);
    switch (variable->vkind) {
    case backend::VarKind::Unknown:
      if (variable->name != unknownName)
        fail("unsupported additional unknown '" + variable->name + "'");
      return {unknown, unknownTangent};
    case backend::VarKind::Param: {
      auto it = parameters.find(variable->name);
      if (it == parameters.end())
        fail("missing parameter '" + variable->name + "'");
      return constantGrid(grid.size, it->second);
    }
    case backend::VarKind::Coord:
      if (variable->coordIndex != 0 && variable->name != "r")
        fail("only the radial coordinate is executable");
      return {grid.radius, std::vector<double>(grid.size, 0.0)};
    case backend::VarKind::Field:
    case backend::VarKind::Local:
      fail("unsupported variable '" + variable->name + "'");
    }
    break;
  }
  case ExprIR::Kind::Binary: {
    const auto *binary = static_cast<const backend::BinaryIR *>(expr);
    DualGrid lhs = evalExpr(binary->lhs.get(), grid, unknownName, unknown,
                            unknownTangent, parameters);
    DualGrid rhs = evalExpr(binary->rhs.get(), grid, unknownName, unknown,
                            unknownTangent, parameters);
    DualGrid result = constantGrid(grid.size, 0.0);
    for (std::size_t i = 0; i < grid.size; ++i) {
      if (binary->op == "+") {
        result.value[i] = lhs.value[i] + rhs.value[i];
        result.tangent[i] = lhs.tangent[i] + rhs.tangent[i];
      } else if (binary->op == "-") {
        result.value[i] = lhs.value[i] - rhs.value[i];
        result.tangent[i] = lhs.tangent[i] - rhs.tangent[i];
      } else if (binary->op == "*") {
        result.value[i] = lhs.value[i] * rhs.value[i];
        result.tangent[i] =
            lhs.tangent[i] * rhs.value[i] + lhs.value[i] * rhs.tangent[i];
      } else if (binary->op == "/") {
        result.value[i] = lhs.value[i] / rhs.value[i];
        result.tangent[i] =
            (lhs.tangent[i] * rhs.value[i] - lhs.value[i] * rhs.tangent[i]) /
            (rhs.value[i] * rhs.value[i]);
      } else if (binary->op == "^") {
        result.value[i] = std::pow(lhs.value[i], rhs.value[i]);
        if (rhs.tangent[i] == 0.0) {
          result.tangent[i] = rhs.value[i] *
                              std::pow(lhs.value[i], rhs.value[i] - 1.0) *
                              lhs.tangent[i];
        } else {
          if (lhs.value[i] <= 0.0)
            fail("differentiating a variable exponent requires positive base");
          result.tangent[i] =
              result.value[i] * (rhs.tangent[i] * std::log(lhs.value[i]) +
                                 rhs.value[i] * lhs.tangent[i] / lhs.value[i]);
        }
      } else {
        fail("unsupported binary operator '" + binary->op + "'");
      }
    }
    return result;
  }
  case ExprIR::Kind::Call: {
    const auto *call = static_cast<const backend::CallIR *>(expr);
    if (call->args.size() != 1)
      fail("radial call '" + call->callee + "' expects one argument");
    DualGrid argument = evalExpr(call->args[0].get(), grid, unknownName,
                                 unknown, unknownTangent, parameters);
    if (call->callee == "laplacian")
      return applyRadialLaplacian(argument, grid);

    DualGrid result = constantGrid(grid.size, 0.0);
    for (std::size_t i = 0; i < grid.size; ++i) {
      if (call->callee == "sin") {
        result.value[i] = std::sin(argument.value[i]);
        result.tangent[i] = std::cos(argument.value[i]) * argument.tangent[i];
      } else if (call->callee == "cos") {
        result.value[i] = std::cos(argument.value[i]);
        result.tangent[i] = -std::sin(argument.value[i]) * argument.tangent[i];
      } else if (call->callee == "sqrt") {
        result.value[i] = std::sqrt(argument.value[i]);
        result.tangent[i] = argument.tangent[i] / (2.0 * result.value[i]);
      } else if (call->callee == "exp") {
        result.value[i] = std::exp(argument.value[i]);
        result.tangent[i] = result.value[i] * argument.tangent[i];
      } else {
        fail("unsupported scalar call '" + call->callee + "'");
      }
    }
    return result;
  }
  case ExprIR::Kind::PartialDerivative: {
    const auto *derivative =
        static_cast<const backend::PartialDerivativeIR *>(expr);
    DualGrid input = evalExpr(derivative->in.get(), grid, unknownName, unknown,
                              unknownTangent, parameters);
    return {applyMatrix(grid.firstDerivative, grid.size, input.value),
            applyMatrix(grid.firstDerivative, grid.size, input.tangent)};
  }
  case ExprIR::Kind::TensorProduct:
  case ExprIR::Kind::Contraction:
  case ExprIR::Kind::IndexRename:
  case ExprIR::Kind::IndexPermute:
  case ExprIR::Kind::Trace:
  case ExprIR::Kind::Gradient:
  case ExprIR::Kind::CovariantDerivative:
  case ExprIR::Kind::Divergence:
    fail("expression kind is not executable by the scalar radial backend");
  }
  fail("unreachable expression evaluation state");
}

const backend::ConstraintAssignmentIR &
getBoundaryCondition(const backend::ConstraintProblemIR &problem,
                     const std::string &region,
                     const std::string &unknownName) {
  for (const auto &boundary : problem.boundaries) {
    if (boundary.region != region)
      continue;
    for (const auto &condition : boundary.conditions)
      if (condition.unknown == unknownName)
        return condition;
    fail("boundary '" + region + "' does not constrain '" + unknownName + "'");
  }
  fail("missing boundary region '" + region + "'");
}

struct RadialDomainLayout {
  std::vector<RadialGrid> grids;
  std::vector<std::size_t> offsets;
  std::size_t totalSize = 0;
};

RadialDomainLayout
buildDomainLayout(const backend::ConstraintProblemIR &problem) {
  RadialDomainLayout layout;
  layout.grids.reserve(problem.domains.size());
  layout.offsets.reserve(problem.domains.size());

  for (std::size_t i = 0; i < problem.domains.size(); ++i) {
    const auto &domain = problem.domains[i];
    if (domain.coordinates != "spherical")
      fail("radial backend requires spherical domains");
    if (domain.basis != "chebyshev")
      fail("radial backend requires basis = chebyshev in every domain");
    if (domain.topology != "shell" && domain.topology != "compactified")
      fail("radial backend supports shell and compactified topologies");
    if (domain.topology == "compactified" && i + 1 != problem.domains.size())
      fail("a compactified domain must be the final radial domain");
    layout.offsets.push_back(layout.totalSize);
    layout.grids.push_back(buildChebyshevLobattoGrid(domain));
    layout.totalSize += layout.grids.back().size;
  }

  if (layout.totalSize > 513)
    fail("dense radial backend limits the total resolution to 513 points");
  if (layout.grids.size() == 1) {
    if (!problem.interfaces.empty())
      fail("a single radial domain cannot declare an interface");
    return layout;
  }
  if (problem.interfaces.size() + 1 != layout.grids.size())
    fail("multidomain radial backend requires one interface between each "
         "consecutive domain");

  for (std::size_t i = 0; i + 1 < layout.grids.size(); ++i) {
    const auto &interface = problem.interfaces[i];
    if (interface.innerDomain != problem.domains[i].name ||
        interface.outerDomain != problem.domains[i + 1].name) {
      fail("radial interfaces must connect consecutive domains in declaration "
           "order");
    }
    const double leftRadius = layout.grids[i].radius.back();
    const double rightRadius = layout.grids[i + 1].radius.front();
    const double scale =
        std::max({1.0, std::abs(leftRadius), std::abs(rightRadius)});
    if (!std::isfinite(leftRadius) || !std::isfinite(rightRadius) ||
        std::abs(leftRadius - rightRadius) > 1.0e-12 * scale) {
      fail("radial interface '" + interface.innerDomain + " -> " +
           interface.outerDomain + "' has incompatible physical radii");
    }
  }
  return layout;
}

std::vector<double> domainSlice(const std::vector<double> &values,
                                std::size_t offset, std::size_t size) {
  if (offset + size > values.size())
    fail("internal multidomain vector size mismatch");
  return {values.begin() + static_cast<std::ptrdiff_t>(offset),
          values.begin() + static_cast<std::ptrdiff_t>(offset + size)};
}

DualGrid
evaluateResidual(const backend::ConstraintProblemIR &problem,
                 const backend::ConstraintEquationIR &equation,
                 const RadialDomainLayout &layout,
                 const std::string &unknownName,
                 const std::vector<double> &unknown,
                 const std::vector<double> &unknownTangent,
                 const std::unordered_map<std::string, double> &parameters) {
  if (unknown.size() != layout.totalSize ||
      unknownTangent.size() != layout.totalSize)
    fail("internal multidomain unknown size mismatch");

  DualGrid residual = constantGrid(layout.totalSize, 0.0);
  std::vector<std::vector<double>> localUnknowns;
  std::vector<std::vector<double>> localTangents;
  localUnknowns.reserve(layout.grids.size());
  localTangents.reserve(layout.grids.size());
  for (std::size_t domainIndex = 0; domainIndex < layout.grids.size();
       ++domainIndex) {
    const auto &grid = layout.grids[domainIndex];
    const std::size_t offset = layout.offsets[domainIndex];
    localUnknowns.push_back(domainSlice(unknown, offset, grid.size));
    localTangents.push_back(domainSlice(unknownTangent, offset, grid.size));
    DualGrid localResidual =
        evalExpr(equation.residual.get(), grid, unknownName,
                 localUnknowns.back(), localTangents.back(), parameters);
    for (std::size_t i = 1; i + 1 < grid.size; ++i) {
      residual.value[offset + i] = localResidual.value[i];
      residual.tangent[offset + i] = localResidual.tangent[i];
    }
  }

  const auto &inner = getBoundaryCondition(problem, "inner", unknownName);
  const auto &outer = getBoundaryCondition(problem, "outer", unknownName);
  const auto &firstGrid = layout.grids.front();
  const auto &lastGrid = layout.grids.back();
  DualGrid innerValue =
      evalExpr(inner.rhs.get(), firstGrid, unknownName, localUnknowns.front(),
               localTangents.front(), parameters);
  DualGrid outerValue =
      evalExpr(outer.rhs.get(), lastGrid, unknownName, localUnknowns.back(),
               localTangents.back(), parameters);
  residual.value.front() =
      localUnknowns.front().front() - innerValue.value.front();
  residual.tangent.front() =
      localTangents.front().front() - innerValue.tangent.front();
  residual.value.back() = localUnknowns.back().back() - outerValue.value.back();
  residual.tangent.back() =
      localTangents.back().back() - outerValue.tangent.back();

  for (std::size_t i = 0; i + 1 < layout.grids.size(); ++i) {
    const auto &leftGrid = layout.grids[i];
    const auto &rightGrid = layout.grids[i + 1];
    const std::size_t leftRow = layout.offsets[i] + leftGrid.size - 1;
    const std::size_t rightRow = layout.offsets[i + 1];
    residual.value[leftRow] =
        localUnknowns[i].back() - localUnknowns[i + 1].front();
    residual.tangent[leftRow] =
        localTangents[i].back() - localTangents[i + 1].front();

    const auto leftDerivative =
        applyMatrix(leftGrid.firstDerivative, leftGrid.size, localUnknowns[i]);
    const auto rightDerivative = applyMatrix(
        rightGrid.firstDerivative, rightGrid.size, localUnknowns[i + 1]);
    const auto leftTangentDerivative =
        applyMatrix(leftGrid.firstDerivative, leftGrid.size, localTangents[i]);
    const auto rightTangentDerivative = applyMatrix(
        rightGrid.firstDerivative, rightGrid.size, localTangents[i + 1]);
    residual.value[rightRow] = leftDerivative.back() - rightDerivative.front();
    residual.tangent[rightRow] =
        leftTangentDerivative.back() - rightTangentDerivative.front();
  }
  return residual;
}

double infinityNorm(const std::vector<double> &values) {
  double norm = 0.0;
  for (double value : values)
    norm = std::max(norm, std::abs(value));
  return norm;
}

std::vector<double> solveDense(Matrix matrix, std::vector<double> rhs,
                               std::size_t n) {
  const double pivotFloor = 64.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    for (std::size_t row = col + 1; row < n; ++row)
      if (std::abs(at(matrix, n, row, col)) >
          std::abs(at(matrix, n, pivot, col)))
        pivot = row;
    if (std::abs(at(matrix, n, pivot, col)) <= pivotFloor)
      fail("Newton Jacobian is singular");
    if (pivot != col) {
      for (std::size_t k = col; k < n; ++k)
        std::swap(at(matrix, n, col, k), at(matrix, n, pivot, k));
      std::swap(rhs[col], rhs[pivot]);
    }
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = at(matrix, n, row, col) / at(matrix, n, col, col);
      at(matrix, n, row, col) = 0.0;
      for (std::size_t k = col + 1; k < n; ++k)
        at(matrix, n, row, k) -= factor * at(matrix, n, col, k);
      rhs[row] -= factor * rhs[col];
    }
  }

  std::vector<double> solution(n, 0.0);
  for (std::size_t reverse = 0; reverse < n; ++reverse) {
    const std::size_t row = n - 1 - reverse;
    double value = rhs[row];
    for (std::size_t col = row + 1; col < n; ++col)
      value -= at(matrix, n, row, col) * solution[col];
    solution[row] = value / at(matrix, n, row, row);
  }
  return solution;
}

} // namespace

ConstraintSolution
solveRadialConstraintProblem(const backend::ModuleIR &module,
                             const ConstraintSolveRequest &request) {
  if (!module.constraintProblem)
    fail("module has no constraint problem");
  const auto &problem = *module.constraintProblem;
  if (problem.domains.empty())
    fail("radial backend requires at least one domain");
  if (problem.unknowns.size() != 1 ||
      !problem.unknowns.front().tensorType.isScalar())
    fail("radial backend currently requires exactly one scalar unknown");
  if (problem.equations.size() != 1 ||
      !problem.equations.front().tensorType.isScalar())
    fail("radial backend currently requires exactly one scalar equation");
  if (problem.solve.nonlinear != "newton" || problem.solve.linear != "direct")
    fail("radial backend currently requires newton with a direct linear solve");

  const auto &unknownDecl = problem.unknowns.front();
  const auto &equation = problem.equations.front();
  RadialDomainLayout layout = buildDomainLayout(problem);
  const std::vector<double> zeroTangent(layout.totalSize, 0.0);
  std::vector<double> unknown(layout.totalSize, 0.0);

  for (const auto &seed : problem.seeds) {
    if (seed.unknown != unknownDecl.name)
      continue;
    for (std::size_t domainIndex = 0; domainIndex < layout.grids.size();
         ++domainIndex) {
      const auto &grid = layout.grids[domainIndex];
      const std::size_t offset = layout.offsets[domainIndex];
      auto localUnknown = domainSlice(unknown, offset, grid.size);
      std::vector<double> localZero(grid.size, 0.0);
      auto localSeed = evalExpr(seed.rhs.get(), grid, unknownDecl.name,
                                localUnknown, localZero, request.parameters)
                           .value;
      std::copy(localSeed.begin(), localSeed.end(), unknown.begin() + offset);
    }
    break;
  }

  ConstraintSolution solution;
  solution.coordinates.reserve(layout.totalSize);
  solution.domains.reserve(layout.grids.size());
  for (std::size_t i = 0; i < layout.grids.size(); ++i) {
    const auto &grid = layout.grids[i];
    solution.coordinates.insert(solution.coordinates.end(), grid.radius.begin(),
                                grid.radius.end());
    solution.domains.push_back(
        {grid.name, layout.offsets[i], grid.size, grid.compactified});
  }
  const std::size_t maxIterations =
      static_cast<std::size_t>(problem.solve.maxIterations);

  for (std::size_t iteration = 0; iteration <= maxIterations; ++iteration) {
    DualGrid residual =
        evaluateResidual(problem, equation, layout, unknownDecl.name, unknown,
                         zeroTangent, request.parameters);
    const double norm = infinityNorm(residual.value);
    solution.residualHistory.push_back(norm);
    solution.residualNorm = norm;
    solution.iterations = iteration;
    if (!std::isfinite(norm))
      fail("residual contains a non-finite value");
    if (norm <= problem.solve.tolerance) {
      solution.converged = true;
      break;
    }
    if (iteration == maxIterations)
      break;

    Matrix jacobian(layout.totalSize * layout.totalSize, 0.0);
    for (std::size_t col = 0; col < layout.totalSize; ++col) {
      std::vector<double> direction(layout.totalSize, 0.0);
      direction[col] = 1.0;
      DualGrid differentiated =
          evaluateResidual(problem, equation, layout, unknownDecl.name, unknown,
                           direction, request.parameters);
      for (std::size_t row = 0; row < layout.totalSize; ++row)
        at(jacobian, layout.totalSize, row, col) = differentiated.tangent[row];
    }

    std::vector<double> rhs(layout.totalSize);
    for (std::size_t i = 0; i < layout.totalSize; ++i)
      rhs[i] = -residual.value[i];
    const std::vector<double> update =
        solveDense(std::move(jacobian), std::move(rhs), layout.totalSize);

    bool accepted = false;
    double damping = 1.0;
    for (int lineSearch = 0; lineSearch < 16; ++lineSearch) {
      std::vector<double> candidate = unknown;
      for (std::size_t i = 0; i < layout.totalSize; ++i)
        candidate[i] += damping * update[i];
      const double candidateNorm = infinityNorm(
          evaluateResidual(problem, equation, layout, unknownDecl.name,
                           candidate, zeroTangent, request.parameters)
              .value);
      if (std::isfinite(candidateNorm) && candidateNorm < norm) {
        unknown = std::move(candidate);
        accepted = true;
        break;
      }
      damping *= 0.5;
    }
    if (!accepted)
      break;
  }

  solution.unknowns.emplace(unknownDecl.name, std::move(unknown));
  return solution;
}

} // namespace tensorium::solver
