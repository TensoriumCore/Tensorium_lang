#include "tensorium/Solver/ConstraintSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
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

struct ComponentField {
  std::vector<DualGrid> components;
};

using UnknownState = std::unordered_map<std::string, ComponentField>;
using ComponentEnvironment = std::unordered_map<std::string, std::size_t>;

constexpr std::size_t kSpatialComponentCount = 3;

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

DualGrid applyRadialDerivative(const DualGrid &input, const RadialGrid &grid) {
  return {applyMatrix(grid.firstDerivative, grid.size, input.value),
          applyMatrix(grid.firstDerivative, grid.size, input.tangent)};
}

// Flat conformal vector Laplacian for a spherically symmetric vector
// V^i = w(r) n^i. The returned scalar is the radial amplitude of
// (Delta_L V)^i = (4/3) (w'' + 2 w'/r - 2 w/r^2) n^i.
DualGrid applyRadialConformalVectorLaplacian(const DualGrid &input,
                                             const RadialGrid &grid) {
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
    const double inverseRadius = 1.0 / grid.radius[i];
    const double inverseRadiusSquared = inverseRadius * inverseRadius;
    result.value[i] =
        (4.0 / 3.0) * (second[i] + 2.0 * inverseRadius * first[i] -
                       2.0 * inverseRadiusSquared * input.value[i]);
    result.tangent[i] =
        (4.0 / 3.0) *
        (tangentSecond[i] + 2.0 * inverseRadius * tangentFirst[i] -
         2.0 * inverseRadiusSquared * input.tangent[i]);
  }
  return result;
}

DualGrid evalExpr(const backend::ExprIR *expr, const RadialGrid &grid,
                  const UnknownState &unknowns,
                  const ComponentEnvironment &componentEnvironment,
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
    case backend::VarKind::Unknown: {
      auto it = unknowns.find(variable->name);
      if (it == unknowns.end())
        fail("missing discrete values for unknown '" + variable->name + "'");
      if (it->second.components.size() == 1)
        return it->second.components.front();
      if (variable->tensorIndexNames.empty())
        fail("tensor unknown '" + variable->name +
             "' requires component indices");
      std::size_t flatComponent = 0;
      for (const auto &indexName : variable->tensorIndexNames) {
        auto component = componentEnvironment.find(indexName);
        if (component == componentEnvironment.end())
          fail("unbound tensor component index '" + indexName + "'");
        flatComponent =
            flatComponent * kSpatialComponentCount + component->second;
      }
      if (flatComponent >= it->second.components.size())
        fail("tensor component is outside the unknown layout");
      return it->second.components[flatComponent];
    }
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
    DualGrid lhs = evalExpr(binary->lhs.get(), grid, unknowns,
                            componentEnvironment, parameters);
    DualGrid rhs = evalExpr(binary->rhs.get(), grid, unknowns,
                            componentEnvironment, parameters);
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
    DualGrid argument = evalExpr(call->args[0].get(), grid, unknowns,
                                 componentEnvironment, parameters);
    if (call->callee == "laplacian")
      return applyRadialLaplacian(argument, grid);
    if (call->callee == "radial_derivative")
      return applyRadialDerivative(argument, grid);
    if (call->callee == "radial_conformal_vector_laplacian")
      return applyRadialConformalVectorLaplacian(argument, grid);

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
    DualGrid input = evalExpr(derivative->in.get(), grid, unknowns,
                              componentEnvironment, parameters);
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
    fail("expression kind is not executable by the component radial backend");
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

struct UnknownComponentLayout {
  std::string name;
  std::size_t rank = 0;
  std::size_t componentCount = 1;
  std::size_t firstComponent = 0;
};

struct ComponentSystemLayout {
  std::vector<UnknownComponentLayout> unknowns;
  std::size_t totalComponents = 0;
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

ComponentSystemLayout
buildComponentLayout(const backend::ConstraintProblemIR &problem) {
  if (problem.unknowns.empty() ||
      problem.unknowns.size() != problem.equations.size())
    fail("radial backend requires one equation per unknown");

  ComponentSystemLayout layout;
  layout.unknowns.reserve(problem.unknowns.size());
  for (std::size_t i = 0; i < problem.unknowns.size(); ++i) {
    const auto &unknown = problem.unknowns[i];
    const auto &equation = problem.equations[i];
    const std::size_t rank =
        static_cast<std::size_t>(unknown.tensorType.rank());
    if (rank > 1)
      fail("radial backend currently supports scalar and rank-one unknowns");
    if (unknown.tensorType.up != equation.tensorType.up ||
        unknown.tensorType.down != equation.tensorType.down)
      fail("unknown '" + unknown.name +
           "' and its equation must have identical tensor variance");
    if (unknown.indices.size() != rank || equation.indices.size() != rank)
      fail("component layout rank does not match declared free indices");
    const std::size_t componentCount = rank == 0 ? 1 : kSpatialComponentCount;
    layout.unknowns.push_back(
        {unknown.name, rank, componentCount, layout.totalComponents});
    layout.totalComponents += componentCount;
  }
  return layout;
}

ComponentEnvironment
makeComponentEnvironment(const std::vector<std::string> &indices,
                         std::size_t component) {
  ComponentEnvironment environment;
  if (indices.empty())
    return environment;
  if (indices.size() != 1 || component >= kSpatialComponentCount)
    fail("unsupported tensor component environment");
  environment.emplace(indices.front(), component);
  return environment;
}

std::vector<UnknownState>
buildLocalStates(const backend::ConstraintProblemIR &problem,
                 const RadialDomainLayout &domainLayout,
                 const ComponentSystemLayout &componentLayout,
                 const std::vector<double> &values,
                 const std::vector<double> &tangents) {
  const std::size_t dofCount =
      componentLayout.totalComponents * domainLayout.totalSize;
  if (values.size() != dofCount || tangents.size() != dofCount)
    fail("internal component vector size mismatch");

  std::vector<UnknownState> states(domainLayout.grids.size());
  for (std::size_t domainIndex = 0; domainIndex < domainLayout.grids.size();
       ++domainIndex) {
    const auto &grid = domainLayout.grids[domainIndex];
    for (std::size_t unknownIndex = 0;
         unknownIndex < componentLayout.unknowns.size(); ++unknownIndex) {
      const auto &unknownLayout = componentLayout.unknowns[unknownIndex];
      ComponentField field;
      field.components.reserve(unknownLayout.componentCount);
      for (std::size_t component = 0; component < unknownLayout.componentCount;
           ++component) {
        const std::size_t offset = (unknownLayout.firstComponent + component) *
                                       domainLayout.totalSize +
                                   domainLayout.offsets[domainIndex];
        field.components.push_back({domainSlice(values, offset, grid.size),
                                    domainSlice(tangents, offset, grid.size)});
      }
      states[domainIndex].emplace(problem.unknowns[unknownIndex].name,
                                  std::move(field));
    }
  }
  return states;
}

DualGrid
evaluateResidual(const backend::ConstraintProblemIR &problem,
                 const RadialDomainLayout &domainLayout,
                 const ComponentSystemLayout &componentLayout,
                 const std::vector<double> &unknown,
                 const std::vector<double> &unknownTangent,
                 const std::unordered_map<std::string, double> &parameters) {
  const std::size_t dofCount =
      componentLayout.totalComponents * domainLayout.totalSize;
  DualGrid residual = constantGrid(dofCount, 0.0);
  const auto localStates = buildLocalStates(
      problem, domainLayout, componentLayout, unknown, unknownTangent);

  const auto &firstGrid = domainLayout.grids.front();
  const auto &lastGrid = domainLayout.grids.back();
  for (std::size_t unknownIndex = 0;
       unknownIndex < componentLayout.unknowns.size(); ++unknownIndex) {
    const auto &unknownLayout = componentLayout.unknowns[unknownIndex];
    const auto &equation = problem.equations[unknownIndex];
    const auto &inner =
        getBoundaryCondition(problem, "inner", unknownLayout.name);
    const auto &outer =
        getBoundaryCondition(problem, "outer", unknownLayout.name);

    for (std::size_t component = 0; component < unknownLayout.componentCount;
         ++component) {
      const std::size_t rowBase =
          (unknownLayout.firstComponent + component) * domainLayout.totalSize;
      const ComponentEnvironment equationEnvironment =
          makeComponentEnvironment(equation.indices, component);

      for (std::size_t domainIndex = 0; domainIndex < domainLayout.grids.size();
           ++domainIndex) {
        const auto &grid = domainLayout.grids[domainIndex];
        const std::size_t offset = rowBase + domainLayout.offsets[domainIndex];
        DualGrid localResidual =
            evalExpr(equation.residual.get(), grid, localStates[domainIndex],
                     equationEnvironment, parameters);
        for (std::size_t i = 1; i + 1 < grid.size; ++i) {
          residual.value[offset + i] = localResidual.value[i];
          residual.tangent[offset + i] = localResidual.tangent[i];
        }
      }

      const ComponentEnvironment innerEnvironment =
          makeComponentEnvironment(inner.indices, component);
      const ComponentEnvironment outerEnvironment =
          makeComponentEnvironment(outer.indices, component);
      const DualGrid innerValue =
          evalExpr(inner.rhs.get(), firstGrid, localStates.front(),
                   innerEnvironment, parameters);
      const DualGrid outerValue =
          evalExpr(outer.rhs.get(), lastGrid, localStates.back(),
                   outerEnvironment, parameters);
      const auto &firstUnknown =
          localStates.front().at(unknownLayout.name).components[component];
      const auto &lastUnknown =
          localStates.back().at(unknownLayout.name).components[component];
      residual.value[rowBase] =
          firstUnknown.value.front() - innerValue.value.front();
      residual.tangent[rowBase] =
          firstUnknown.tangent.front() - innerValue.tangent.front();
      residual.value[rowBase + domainLayout.totalSize - 1] =
          lastUnknown.value.back() - outerValue.value.back();
      residual.tangent[rowBase + domainLayout.totalSize - 1] =
          lastUnknown.tangent.back() - outerValue.tangent.back();

      for (std::size_t i = 0; i + 1 < domainLayout.grids.size(); ++i) {
        const auto &leftGrid = domainLayout.grids[i];
        const auto &rightGrid = domainLayout.grids[i + 1];
        const auto &leftUnknown =
            localStates[i].at(unknownLayout.name).components[component];
        const auto &rightUnknown =
            localStates[i + 1].at(unknownLayout.name).components[component];
        const std::size_t leftRow =
            rowBase + domainLayout.offsets[i] + leftGrid.size - 1;
        const std::size_t rightRow = rowBase + domainLayout.offsets[i + 1];
        residual.value[leftRow] =
            leftUnknown.value.back() - rightUnknown.value.front();
        residual.tangent[leftRow] =
            leftUnknown.tangent.back() - rightUnknown.tangent.front();

        const auto leftDerivative = applyMatrix(
            leftGrid.firstDerivative, leftGrid.size, leftUnknown.value);
        const auto rightDerivative = applyMatrix(
            rightGrid.firstDerivative, rightGrid.size, rightUnknown.value);
        const auto leftTangentDerivative = applyMatrix(
            leftGrid.firstDerivative, leftGrid.size, leftUnknown.tangent);
        const auto rightTangentDerivative = applyMatrix(
            rightGrid.firstDerivative, rightGrid.size, rightUnknown.tangent);
        residual.value[rightRow] =
            leftDerivative.back() - rightDerivative.front();
        residual.tangent[rightRow] =
            leftTangentDerivative.back() - rightTangentDerivative.front();
      }
    }
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

RadialCttPhysicalSolution reconstructRadialCtt(
    const backend::ConstraintProblemIR &problem,
    const RadialDomainLayout &domainLayout,
    const ComponentSystemLayout &componentLayout,
    const std::vector<double> &unknown,
    const std::unordered_map<std::string, double> &parameters) {
  const auto &reconstruction = problem.cttReconstruction;
  auto findScalarLayout =
      [&](const std::string &name,
          const std::string &role) -> const UnknownComponentLayout & {
    auto it = std::find_if(
        componentLayout.unknowns.begin(), componentLayout.unknowns.end(),
        [&](const UnknownComponentLayout &layout) {
          return layout.name == name;
        });
    if (it == componentLayout.unknowns.end())
      fail("CTT reconstruction " + role + " unknown '" + name +
           "' is missing from the component layout");
    if (it->componentCount != 1)
      fail("CTT reconstruction " + role + " must be scalar");
    return *it;
  };

  const auto &psiLayout = findScalarLayout(reconstruction.conformalFactor,
                                           "conformal factor");
  const auto &wLayout = findScalarLayout(reconstruction.radialVectorPotential,
                                         "radial vector potential");
  const std::vector<double> zeroTangent(unknown.size(), 0.0);
  const auto localStates = buildLocalStates(
      problem, domainLayout, componentLayout, unknown, zeroTangent);

  RadialCttPhysicalSolution physical;
  physical.conformalFactorUnknown = reconstruction.conformalFactor;
  physical.radialVectorPotentialUnknown =
      reconstruction.radialVectorPotential;
  physical.meanCurvature.reserve(domainLayout.totalSize);
  physical.spatialMetricRadial.reserve(domainLayout.totalSize);
  physical.spatialMetricTangential.reserve(domainLayout.totalSize);
  physical.extrinsicCurvatureRadial.reserve(domainLayout.totalSize);
  physical.extrinsicCurvatureTangential.reserve(domainLayout.totalSize);

  for (std::size_t domainIndex = 0;
       domainIndex < domainLayout.grids.size(); ++domainIndex) {
    const auto &grid = domainLayout.grids[domainIndex];
    const auto &state = localStates[domainIndex];
    const auto &psi = state.at(psiLayout.name).components.front().value;
    const auto &w = state.at(wLayout.name).components.front().value;
    const auto wPrime = applyMatrix(grid.firstDerivative, grid.size, w);
    const auto meanCurvature =
        evalExpr(reconstruction.meanCurvature.get(), grid, state, {}, parameters)
            .value;

    for (std::size_t i = 0; i < grid.size; ++i) {
      if (!std::isfinite(psi[i]) || psi[i] <= 0.0)
        fail("CTT reconstruction requires a finite positive conformal factor");
      if (!std::isfinite(meanCurvature[i]))
        fail("CTT reconstruction mean curvature is non-finite");
      const double inverseRadius = 1.0 / grid.radius[i];
      const double longitudinalAmplitude = wPrime[i] - w[i] * inverseRadius;
      const double psiSquared = psi[i] * psi[i];
      const double psiFourth = psiSquared * psiSquared;
      const double inversePsiSquared = 1.0 / psiSquared;
      const double traceContribution =
          (1.0 / 3.0) * psiFourth * meanCurvature[i];
      const double radialExtrinsic =
          inversePsiSquared * (4.0 / 3.0) * longitudinalAmplitude +
          traceContribution;
      const double tangentialExtrinsic =
          inversePsiSquared * (-2.0 / 3.0) * longitudinalAmplitude +
          traceContribution;

      physical.meanCurvature.push_back(meanCurvature[i]);
      physical.spatialMetricRadial.push_back(psiFourth);
      physical.spatialMetricTangential.push_back(psiFourth);
      physical.extrinsicCurvatureRadial.push_back(radialExtrinsic);
      physical.extrinsicCurvatureTangential.push_back(tangentialExtrinsic);
    }
  }
  return physical;
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
  if (problem.solve.nonlinear != "newton" || problem.solve.linear != "direct")
    fail("radial backend currently requires newton with a direct linear solve");

  RadialDomainLayout layout = buildDomainLayout(problem);
  ComponentSystemLayout componentLayout = buildComponentLayout(problem);
  const std::size_t dofCount =
      componentLayout.totalComponents * layout.totalSize;
  if (dofCount > 513)
    fail("dense radial backend limits coupled systems to 513 component "
         "degrees of freedom");
  const std::vector<double> zeroTangent(dofCount, 0.0);
  std::vector<double> unknown(dofCount, 0.0);

  for (const auto &seed : problem.seeds) {
    auto target = std::find_if(problem.unknowns.begin(), problem.unknowns.end(),
                               [&](const backend::ConstraintUnknownIR &decl) {
                                 return decl.name == seed.unknown;
                               });
    if (target == problem.unknowns.end())
      continue;
    const std::size_t targetIndex = static_cast<std::size_t>(
        std::distance(problem.unknowns.begin(), target));
    const auto &targetLayout = componentLayout.unknowns[targetIndex];
    const auto localStates = buildLocalStates(problem, layout, componentLayout,
                                              unknown, zeroTangent);
    for (std::size_t domainIndex = 0; domainIndex < layout.grids.size();
         ++domainIndex) {
      const auto &grid = layout.grids[domainIndex];
      for (std::size_t component = 0; component < targetLayout.componentCount;
           ++component) {
        const ComponentEnvironment environment =
            makeComponentEnvironment(seed.indices, component);
        const std::size_t offset =
            (targetLayout.firstComponent + component) * layout.totalSize +
            layout.offsets[domainIndex];
        auto localSeed =
            evalExpr(seed.rhs.get(), grid, localStates[domainIndex],
                     environment, request.parameters)
                .value;
        std::copy(localSeed.begin(), localSeed.end(), unknown.begin() + offset);
      }
    }
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
  solution.unknownLayouts.reserve(componentLayout.unknowns.size());
  for (std::size_t i = 0; i < componentLayout.unknowns.size(); ++i) {
    const auto &unknownLayout = componentLayout.unknowns[i];
    const auto &tensorType = problem.unknowns[i].tensorType;
    solution.unknownLayouts.push_back(
        {unknownLayout.name, static_cast<std::size_t>(tensorType.up),
         static_cast<std::size_t>(tensorType.down),
         unknownLayout.componentCount, layout.totalSize});
  }
  const std::size_t maxIterations =
      static_cast<std::size_t>(problem.solve.maxIterations);

  for (std::size_t iteration = 0; iteration <= maxIterations; ++iteration) {
    DualGrid residual =
        evaluateResidual(problem, layout, componentLayout, unknown, zeroTangent,
                         request.parameters);
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

    Matrix jacobian(dofCount * dofCount, 0.0);
    for (std::size_t col = 0; col < dofCount; ++col) {
      std::vector<double> direction(dofCount, 0.0);
      direction[col] = 1.0;
      DualGrid differentiated =
          evaluateResidual(problem, layout, componentLayout, unknown, direction,
                           request.parameters);
      for (std::size_t row = 0; row < dofCount; ++row)
        at(jacobian, dofCount, row, col) = differentiated.tangent[row];
    }

    std::vector<double> rhs(dofCount);
    for (std::size_t i = 0; i < dofCount; ++i)
      rhs[i] = -residual.value[i];
    const std::vector<double> update =
        solveDense(std::move(jacobian), std::move(rhs), dofCount);

    bool accepted = false;
    double damping = 1.0;
    for (int lineSearch = 0; lineSearch < 16; ++lineSearch) {
      std::vector<double> candidate = unknown;
      for (std::size_t i = 0; i < dofCount; ++i)
        candidate[i] += damping * update[i];
      const double candidateNorm = infinityNorm(
          evaluateResidual(problem, layout, componentLayout, candidate,
                           zeroTangent, request.parameters)
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

  for (std::size_t unknownIndex = 0;
       unknownIndex < componentLayout.unknowns.size(); ++unknownIndex) {
    const auto &unknownLayout = componentLayout.unknowns[unknownIndex];
    const std::size_t offset = unknownLayout.firstComponent * layout.totalSize;
    solution.unknowns.emplace(
        unknownLayout.name,
        domainSlice(unknown, offset,
                    unknownLayout.componentCount * layout.totalSize));
  }
  if (problem.cttReconstruction.enabled) {
    solution.physicalCtt = reconstructRadialCtt(
        problem, layout, componentLayout, unknown, request.parameters);
  }
  return solution;
}

} // namespace tensorium::solver
