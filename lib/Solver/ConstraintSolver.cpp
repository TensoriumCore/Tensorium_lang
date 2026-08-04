#include "tensorium/Solver/ConstraintSolver.hpp"

#include <algorithm>
#include <array>
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
  std::size_t rank = 0;
  bool symmetric = false;
};

using UnknownState = std::unordered_map<std::string, ComponentField>;
using ComponentEnvironment = std::unordered_map<std::string, std::size_t>;

struct RadialGeometry {
  bool enabled = false;
  std::string metricName;
  std::string inverseMetricName;
  std::vector<double> radialScale;
  std::vector<double> tangentialScale;
  std::vector<double> radialScaleDerivative;
  std::vector<double> tangentialScaleDerivative;
  std::vector<double> connectionRate;
};

constexpr std::size_t kSpatialComponentCount = 3;
constexpr std::size_t kSymmetricRankTwoComponentCount = 6;
constexpr std::array<std::array<std::size_t, 2>,
                     kSymmetricRankTwoComponentCount>
    kSymmetricRankTwoIndices = {{{0, 0}, {0, 1}, {0, 2},
                                 {1, 1}, {1, 2}, {2, 2}}};
constexpr std::array<std::array<std::size_t, kSpatialComponentCount>,
                     kSpatialComponentCount>
    kSymmetricRankTwoComponents = {
        {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}};

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

DualGrid multiplyPointwise(const DualGrid &lhs, const DualGrid &rhs,
                           std::size_t size) {
  if (lhs.value.size() != size || lhs.tangent.size() != size ||
      rhs.value.size() != size || rhs.tangent.size() != size)
    fail("internal dual-grid size mismatch");

  DualGrid result = constantGrid(size, 0.0);
  for (std::size_t i = 0; i < size; ++i) {
    result.value[i] = lhs.value[i] * rhs.value[i];
    result.tangent[i] =
        lhs.tangent[i] * rhs.value[i] + lhs.value[i] * rhs.tangent[i];
  }
  return result;
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

DualGrid applyGeometryScalarLaplacian(const DualGrid &input,
                                      const RadialGrid &grid,
                                      const RadialGeometry &geometry) {
  if (!geometry.enabled || geometry.radialScale.size() != grid.size ||
      geometry.tangentialScale.size() != grid.size ||
      geometry.radialScaleDerivative.size() != grid.size ||
      geometry.tangentialScaleDerivative.size() != grid.size)
    fail("invalid spherical-orthonormal geometry layout");

  const auto first = applyMatrix(grid.firstDerivative, grid.size, input.value);
  const auto second = applyMatrix(grid.secondDerivative, grid.size, input.value);
  const auto tangentFirst =
      applyMatrix(grid.firstDerivative, grid.size, input.tangent);
  const auto tangentSecond =
      applyMatrix(grid.secondDerivative, grid.size, input.tangent);
  DualGrid result = constantGrid(grid.size, 0.0);
  for (std::size_t point = 0; point < grid.size; ++point) {
    const double radialScale = geometry.radialScale[point];
    const double tangentialScale = geometry.tangentialScale[point];
    const double inverseRadialSquared =
        1.0 / (radialScale * radialScale);
    const double inverseRadius = std::isinf(grid.radius[point])
                                     ? 0.0
                                     : 1.0 / grid.radius[point];
    const double firstCoefficient =
        inverseRadialSquared *
        (2.0 * inverseRadius +
         2.0 * geometry.tangentialScaleDerivative[point] / tangentialScale -
         geometry.radialScaleDerivative[point] / radialScale);
    result.value[point] = inverseRadialSquared * second[point] +
                          firstCoefficient * first[point];
    result.tangent[point] = inverseRadialSquared * tangentSecond[point] +
                            firstCoefficient * tangentFirst[point];
  }
  return result;
}

DualGrid applyFrameDerivative(const DualGrid &input, const RadialGrid &grid,
                              const RadialGeometry &geometry,
                              std::size_t direction) {
  if (direction != 0)
    return constantGrid(grid.size, 0.0);
  const auto derivative =
      applyMatrix(grid.firstDerivative, grid.size, input.value);
  const auto tangentDerivative =
      applyMatrix(grid.firstDerivative, grid.size, input.tangent);
  DualGrid result = constantGrid(grid.size, 0.0);
  for (std::size_t point = 0; point < grid.size; ++point) {
    result.value[point] = derivative[point] / geometry.radialScale[point];
    result.tangent[point] =
        tangentDerivative[point] / geometry.radialScale[point];
  }
  return result;
}

double frameConnection(const RadialGeometry &geometry, std::size_t output,
                       std::size_t input, std::size_t direction,
                       std::size_t point) {
  if (direction == 1) {
    if (output == 1 && input == 0)
      return geometry.connectionRate[point];
    if (output == 0 && input == 1)
      return -geometry.connectionRate[point];
  }
  if (direction == 2) {
    if (output == 2 && input == 0)
      return geometry.connectionRate[point];
    if (output == 0 && input == 2)
      return -geometry.connectionRate[point];
  }
  return 0.0;
}

struct TensorSlot {
  std::string index;
  bool contravariant = false;
};

std::vector<TensorSlot> collectTensorSlots(const backend::ExprIR *expr) {
  using backend::ExprIR;
  if (!expr)
    fail("cannot collect tensor slots from a null expression");

  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return {};
  case ExprIR::Kind::Var: {
    const auto *variable = static_cast<const backend::VarIR *>(expr);
    std::vector<TensorSlot> slots;
    slots.reserve(variable->tensorIndexNames.size());
    for (std::size_t slot = 0; slot < variable->tensorIndexNames.size();
         ++slot) {
      slots.push_back(
          {variable->tensorIndexNames[slot],
           slot < static_cast<std::size_t>(variable->exprType.up)});
    }
    return slots;
  }
  case ExprIR::Kind::Binary: {
    const auto *binary = static_cast<const backend::BinaryIR *>(expr);
    auto lhs = collectTensorSlots(binary->lhs.get());
    auto rhs = collectTensorSlots(binary->rhs.get());
    const auto rank = static_cast<std::size_t>(expr->exprType.rank());
    if (lhs.size() == rank)
      return lhs;
    if (rhs.size() == rank)
      return rhs;
    if (rank == 0)
      return {};
    fail("cannot determine the free tensor slots of a binary expression");
  }
  case ExprIR::Kind::Call: {
    const auto *call = static_cast<const backend::CallIR *>(expr);
    if (expr->exprType.rank() == 0)
      return {};
    if (call->args.empty())
      fail("tensor call has no argument from which to recover indices");
    return collectTensorSlots(call->args.front().get());
  }
  case ExprIR::Kind::TensorProduct: {
    const auto *product = static_cast<const backend::TensorProductIR *>(expr);
    auto slots = collectTensorSlots(product->lhs.get());
    auto rhs = collectTensorSlots(product->rhs.get());
    slots.insert(slots.end(), rhs.begin(), rhs.end());
    return slots;
  }
  case ExprIR::Kind::Contraction: {
    const auto *contraction =
        static_cast<const backend::ContractionIR *>(expr);
    auto slots = collectTensorSlots(contraction->in.get());
    slots.erase(
        std::remove_if(slots.begin(), slots.end(), [&](const TensorSlot &slot) {
          return std::find(contraction->summedIndices.begin(),
                           contraction->summedIndices.end(),
                           slot.index) != contraction->summedIndices.end();
        }),
        slots.end());
    return slots;
  }
  case ExprIR::Kind::IndexRename: {
    const auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    auto slots = collectTensorSlots(rename->in.get());
    for (auto &slot : slots)
      if (slot.index == rename->from)
        slot.index = rename->to;
    return slots;
  }
  case ExprIR::Kind::IndexPermute: {
    const auto *permute = static_cast<const backend::IndexPermuteIR *>(expr);
    return collectTensorSlots(permute->in.get());
  }
  case ExprIR::Kind::Trace: {
    const auto *trace = static_cast<const backend::TraceIR *>(expr);
    auto slots = collectTensorSlots(trace->in.get());
    slots.erase(
        std::remove_if(slots.begin(), slots.end(), [&](const TensorSlot &slot) {
          return std::find(trace->tracedIndices.begin(),
                           trace->tracedIndices.end(),
                           slot.index) != trace->tracedIndices.end();
        }),
        slots.end());
    return slots;
  }
  case ExprIR::Kind::PartialDerivative: {
    const auto *derivative =
        static_cast<const backend::PartialDerivativeIR *>(expr);
    auto slots = collectTensorSlots(derivative->in.get());
    slots.push_back({derivative->coordIndex, false});
    return slots;
  }
  case ExprIR::Kind::CovariantDerivative: {
    const auto *derivative =
        static_cast<const backend::CovariantDerivativeIR *>(expr);
    auto slots = collectTensorSlots(derivative->in.get());
    slots.push_back({derivative->derivIndex, derivative->contravariant});
    return slots;
  }
  case ExprIR::Kind::Divergence: {
    const auto *divergence = static_cast<const backend::DivergenceIR *>(expr);
    auto slots = collectTensorSlots(divergence->in.get());
    slots.erase(
        std::remove_if(slots.begin(), slots.end(), [&](const TensorSlot &slot) {
          return slot.index == divergence->contractedIndex;
        }),
        slots.end());
    return slots;
  }
  case ExprIR::Kind::Gradient:
    fail("gradient must be canonicalized before tensor slot evaluation");
  }
  fail("unreachable tensor slot collection state");
}

void validateFreeTensorSlots(const backend::ExprIR *expr,
                             const std::vector<TensorSlot> &slots) {
  const auto expected = static_cast<std::size_t>(expr->exprType.rank());
  if (slots.size() != expected) {
    fail("tensor expression exposes " + std::to_string(slots.size()) +
         " free component slots but its type has rank " +
         std::to_string(expected));
  }
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    if (slots[slot].index.empty())
      fail("tensor expression contains an unnamed free component slot");
    for (std::size_t previous = 0; previous < slot; ++previous) {
      if (slots[previous].index == slots[slot].index) {
        fail("tensor expression has repeated free index '" +
             slots[slot].index + "'");
      }
    }
  }
}

DualGrid evalExpr(const backend::ExprIR *expr, const RadialGrid &grid,
                  const UnknownState &unknowns,
                  const ComponentEnvironment &componentEnvironment,
                  const RadialGeometry *geometry,
                  const std::unordered_map<std::string, double> &parameters);

DualGrid evalCovariantDerivative(
    const backend::ExprIR *input, std::size_t direction,
    const RadialGrid &grid, const UnknownState &unknowns,
    const ComponentEnvironment &componentEnvironment,
    const RadialGeometry &geometry,
    const std::unordered_map<std::string, double> &parameters) {
  DualGrid value = evalExpr(input, grid, unknowns, componentEnvironment,
                            &geometry, parameters);
  DualGrid result = applyFrameDerivative(value, grid, geometry, direction);
  auto slots = collectTensorSlots(input);
  validateFreeTensorSlots(input, slots);

  for (const auto &slot : slots) {
    auto slotComponentIt = componentEnvironment.find(slot.index);
    if (slotComponentIt == componentEnvironment.end())
      fail("unbound covariant tensor slot index '" + slot.index + "'");
    const std::size_t slotComponent = slotComponentIt->second;

    for (std::size_t replacement = 0;
         replacement < kSpatialComponentCount; ++replacement) {
      ComponentEnvironment shiftedEnvironment = componentEnvironment;
      shiftedEnvironment[slot.index] = replacement;
      DualGrid shifted = evalExpr(input, grid, unknowns, shiftedEnvironment,
                                  &geometry, parameters);
      for (std::size_t point = 0; point < grid.size; ++point) {
        const double coefficient =
            slot.contravariant
                ? frameConnection(geometry, slotComponent, replacement,
                                  direction, point)
                : -frameConnection(geometry, replacement, slotComponent,
                                   direction, point);
        result.value[point] += coefficient * shifted.value[point];
        result.tangent[point] += coefficient * shifted.tangent[point];
      }
    }
  }
  return result;
}

DualGrid applyGeometryRoughLaplacian(
    const backend::ExprIR *input, const RadialGrid &grid,
    const UnknownState &unknowns,
    const ComponentEnvironment &componentEnvironment,
    const RadialGeometry &geometry,
    const std::unordered_map<std::string, double> &parameters) {
  auto inputSlots = collectTensorSlots(input);
  validateFreeTensorSlots(input, inputSlots);
  DualGrid result = constantGrid(grid.size, 0.0);

  for (std::size_t direction = 0; direction < kSpatialComponentCount;
       ++direction) {
    DualGrid first = evalCovariantDerivative(
        input, direction, grid, unknowns, componentEnvironment, geometry,
        parameters);
    DualGrid second = applyFrameDerivative(first, grid, geometry, direction);

    // The outer derivative acts on every original tensor slot of the first
    // derivative.
    for (const auto &slot : inputSlots) {
      auto slotComponentIt = componentEnvironment.find(slot.index);
      if (slotComponentIt == componentEnvironment.end())
        fail("unbound rough-laplacian tensor slot index '" + slot.index +
             "'");
      const std::size_t slotComponent = slotComponentIt->second;
      for (std::size_t replacement = 0;
           replacement < kSpatialComponentCount; ++replacement) {
        ComponentEnvironment shiftedEnvironment = componentEnvironment;
        shiftedEnvironment[slot.index] = replacement;
        DualGrid shiftedFirst = evalCovariantDerivative(
            input, direction, grid, unknowns, shiftedEnvironment, geometry,
            parameters);
        for (std::size_t point = 0; point < grid.size; ++point) {
          const double coefficient =
              slot.contravariant
                  ? frameConnection(geometry, slotComponent, replacement,
                                    direction, point)
                  : -frameConnection(geometry, replacement, slotComponent,
                                     direction, point);
          second.value[point] += coefficient * shiftedFirst.value[point];
          second.tangent[point] +=
              coefficient * shiftedFirst.tangent[point];
        }
      }
    }

    // The first derivative contributes one additional covariant slot. This
    // term is -Gamma^b_{a a} D_b and supplies both the scalar radial measure
    // term and the derivative-index connection for tensors.
    for (std::size_t replacement = 0;
         replacement < kSpatialComponentCount; ++replacement) {
      DualGrid shiftedFirst = evalCovariantDerivative(
          input, replacement, grid, unknowns, componentEnvironment, geometry,
          parameters);
      for (std::size_t point = 0; point < grid.size; ++point) {
        const double coefficient =
            -frameConnection(geometry, replacement, direction, direction,
                             point);
        second.value[point] += coefficient * shiftedFirst.value[point];
        second.tangent[point] +=
            coefficient * shiftedFirst.tangent[point];
      }
    }

    for (std::size_t point = 0; point < grid.size; ++point) {
      result.value[point] += second.value[point];
      result.tangent[point] += second.tangent[point];
    }
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
                  const RadialGeometry *geometry,
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
      if (variable->tensorIndexNames.size() != it->second.rank)
        fail("tensor access rank does not match the unknown layout");

      std::array<std::size_t, 2> symmetricIndices{};
      std::size_t flatComponent = 0;
      for (std::size_t index = 0;
           index < variable->tensorIndexNames.size(); ++index) {
        const auto &indexName = variable->tensorIndexNames[index];
        auto component = componentEnvironment.find(indexName);
        if (component == componentEnvironment.end())
          fail("unbound tensor component index '" + indexName + "'");
        if (it->second.symmetric)
          symmetricIndices[index] = component->second;
        else
          flatComponent = flatComponent * kSpatialComponentCount +
                          component->second;
      }
      if (it->second.symmetric) {
        if (it->second.rank != 2)
          fail("symmetric component mapping requires a rank-two unknown");
        flatComponent = kSymmetricRankTwoComponents[symmetricIndices[0]]
                                                    [symmetricIndices[1]];
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
      if (geometry && geometry->enabled &&
          (variable->name == geometry->metricName ||
           variable->name == geometry->inverseMetricName)) {
        if (variable->tensorIndexNames.size() != 2)
          fail("geometry metric access requires two component indices");
        auto first =
            componentEnvironment.find(variable->tensorIndexNames.front());
        auto second =
            componentEnvironment.find(variable->tensorIndexNames.back());
        if (first == componentEnvironment.end() ||
            second == componentEnvironment.end())
          fail("geometry metric access has an unbound component index");
        return constantGrid(grid.size,
                            first->second == second->second ? 1.0 : 0.0);
      }
      fail("unsupported fixed field '" + variable->name + "'");
    case backend::VarKind::Local:
      fail("unsupported variable '" + variable->name + "'");
    }
    break;
  }
  case ExprIR::Kind::Binary: {
    const auto *binary = static_cast<const backend::BinaryIR *>(expr);
    DualGrid lhs = evalExpr(binary->lhs.get(), grid, unknowns,
                            componentEnvironment, geometry, parameters);
    DualGrid rhs = evalExpr(binary->rhs.get(), grid, unknowns,
                            componentEnvironment, geometry, parameters);
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
    if (call->callee == "laplacian" && geometry && geometry->enabled) {
      const int rank = call->args[0]->exprType.rank();
      if (rank > 2)
        fail("rough laplacian supports tensor arguments through rank two");
      if (rank != 0)
        return applyGeometryRoughLaplacian(
            call->args[0].get(), grid, unknowns, componentEnvironment,
            *geometry, parameters);
    }
    DualGrid argument = evalExpr(call->args[0].get(), grid, unknowns,
                                 componentEnvironment, geometry, parameters);
    if (call->callee == "laplacian") {
      if (geometry && geometry->enabled)
        return applyGeometryScalarLaplacian(argument, grid, *geometry);
      return applyRadialLaplacian(argument, grid);
    }
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
                              componentEnvironment, geometry, parameters);
    if (geometry && geometry->enabled) {
      auto direction = componentEnvironment.find(derivative->coordIndex);
      if (direction == componentEnvironment.end())
        fail("unbound frame derivative index '" + derivative->coordIndex + "'");
      return applyFrameDerivative(input, grid, *geometry, direction->second);
    }
    return {applyMatrix(grid.firstDerivative, grid.size, input.value),
            applyMatrix(grid.firstDerivative, grid.size, input.tangent)};
  }
  case ExprIR::Kind::Contraction: {
    // The current semantic pipeline canonicalizes laplacian(u) into
    // contraction(partial_i(partial_i(u))). In a spherically symmetric radial
    // solve this denotes u'' + 2 u' / r, not merely the one-dimensional
    // second derivative.
    const auto *contraction =
        static_cast<const backend::ContractionIR *>(expr);
    const auto *outer = dynamic_cast<const backend::PartialDerivativeIR *>(
        contraction->in.get());
    const auto *inner =
        outer ? dynamic_cast<const backend::PartialDerivativeIR *>(
                    outer->in.get())
              : nullptr;
    if (inner && outer->coordIndex == inner->coordIndex &&
        contraction->summedIndices.size() == 1 &&
        contraction->summedIndices.front() == outer->coordIndex) {
      DualGrid input = evalExpr(inner->in.get(), grid, unknowns,
                                componentEnvironment, geometry, parameters);
      if (geometry && geometry->enabled)
        return applyGeometryScalarLaplacian(input, grid, *geometry);
      return applyRadialLaplacian(input, grid);
    }

    if (contraction->summedIndices.empty())
      fail("Einstein contraction has no summed indices");
    if (contraction->summedIndices.size() > 2)
      fail("radial Einstein contraction supports at most two summed indices");

    ComponentEnvironment contractedEnvironment = componentEnvironment;
    for (const auto &index : contraction->summedIndices) {
      if (contractedEnvironment.contains(index))
        fail("Einstein contraction index '" + index +
             "' collides with a free component index");
    }

    DualGrid result = constantGrid(grid.size, 0.0);
    auto accumulate = [&](auto &&self, std::size_t depth) -> void {
      if (depth == contraction->summedIndices.size()) {
        DualGrid term = evalExpr(contraction->in.get(), grid, unknowns,
                                 contractedEnvironment, geometry, parameters);
        for (std::size_t point = 0; point < grid.size; ++point) {
          result.value[point] += term.value[point];
          result.tangent[point] += term.tangent[point];
        }
        return;
      }

      const auto &index = contraction->summedIndices[depth];
      for (std::size_t component = 0; component < kSpatialComponentCount;
           ++component) {
        contractedEnvironment.emplace(index, component);
        self(self, depth + 1);
        contractedEnvironment.erase(index);
      }
    };
    accumulate(accumulate, 0);
    return result;
  }
  case ExprIR::Kind::TensorProduct: {
    const auto *product = static_cast<const backend::TensorProductIR *>(expr);
    DualGrid lhs = evalExpr(product->lhs.get(), grid, unknowns,
                            componentEnvironment, geometry, parameters);
    DualGrid rhs = evalExpr(product->rhs.get(), grid, unknowns,
                            componentEnvironment, geometry, parameters);
    return multiplyPointwise(lhs, rhs, grid.size);
  }
  case ExprIR::Kind::CovariantDerivative: {
    const auto *derivative =
        static_cast<const backend::CovariantDerivativeIR *>(expr);
    if (!geometry || !geometry->enabled)
      fail("covariant derivative requires spherical-orthonormal geometry");
    auto directionIt = componentEnvironment.find(derivative->derivIndex);
    if (directionIt == componentEnvironment.end())
      fail("unbound covariant derivative index '" + derivative->derivIndex +
           "'");
    return evalCovariantDerivative(
        derivative->in.get(), directionIt->second, grid, unknowns,
        componentEnvironment, *geometry, parameters);
  }
  case ExprIR::Kind::IndexRename:
  case ExprIR::Kind::IndexPermute:
  case ExprIR::Kind::Trace:
  case ExprIR::Kind::Gradient:
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
  bool symmetric = false;
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

std::vector<RadialGeometry> buildRadialGeometries(
    const backend::ConstraintProblemIR &problem,
    const RadialDomainLayout &domainLayout,
    const std::unordered_map<std::string, double> &parameters) {
  std::vector<RadialGeometry> geometries(domainLayout.grids.size());
  if (!problem.geometry.enabled)
    return geometries;
  if (problem.geometry.kind != "spherical_orthonormal")
    fail("unsupported constraint geometry '" + problem.geometry.kind + "'");
  if (!problem.geometry.radialScale || !problem.geometry.tangentialScale)
    fail("spherical-orthonormal geometry has missing scale expressions");

  const UnknownState noUnknowns;
  const ComponentEnvironment noComponents;
  for (std::size_t domainIndex = 0;
       domainIndex < domainLayout.grids.size(); ++domainIndex) {
    const auto &grid = domainLayout.grids[domainIndex];
    auto &geometry = geometries[domainIndex];
    geometry.enabled = true;
    geometry.metricName = problem.geometry.metricName;
    geometry.inverseMetricName = problem.geometry.inverseMetricName;
    geometry.radialScale =
        evalExpr(problem.geometry.radialScale.get(), grid, noUnknowns,
                 noComponents, nullptr, parameters)
            .value;
    geometry.tangentialScale =
        evalExpr(problem.geometry.tangentialScale.get(), grid, noUnknowns,
                 noComponents, nullptr, parameters)
            .value;
    geometry.radialScaleDerivative = applyMatrix(
        grid.firstDerivative, grid.size, geometry.radialScale);
    geometry.tangentialScaleDerivative = applyMatrix(
        grid.firstDerivative, grid.size, geometry.tangentialScale);
    geometry.connectionRate.resize(grid.size);

    for (std::size_t point = 0; point < grid.size; ++point) {
      const double radialScale = geometry.radialScale[point];
      const double tangentialScale = geometry.tangentialScale[point];
      if (!std::isfinite(radialScale) || radialScale <= 0.0)
        fail("geometry radial_scale must be finite and strictly positive");
      if (!std::isfinite(tangentialScale) || tangentialScale <= 0.0)
        fail("geometry tangential_scale must be finite and strictly positive");
      if (!std::isfinite(geometry.radialScaleDerivative[point]) ||
          !std::isfinite(geometry.tangentialScaleDerivative[point]))
        fail("geometry scale derivative is non-finite");
      const double inverseRadius = std::isinf(grid.radius[point])
                                       ? 0.0
                                       : 1.0 / grid.radius[point];
      geometry.connectionRate[point] =
          (inverseRadius +
           geometry.tangentialScaleDerivative[point] / tangentialScale) /
          radialScale;
      if (!std::isfinite(geometry.connectionRate[point]))
        fail("geometry connection coefficient is non-finite");
    }
  }
  return geometries;
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
    if (rank > 2)
      fail("radial backend currently supports unknowns up to rank two");
    if (unknown.symmetric &&
        !((unknown.tensorType.up == 0 && unknown.tensorType.down == 2) ||
          (unknown.tensorType.up == 2 && unknown.tensorType.down == 0))) {
      fail("symmetric unknown '" + unknown.name +
           "' must be covariant or contravariant rank two");
    }
    if (unknown.tensorType.up != equation.tensorType.up ||
        unknown.tensorType.down != equation.tensorType.down)
      fail("unknown '" + unknown.name +
           "' and its equation must have identical tensor variance");
    if (unknown.indices.size() != rank || equation.indices.size() != rank)
      fail("component layout rank does not match declared free indices");
    std::size_t componentCount = 1;
    for (std::size_t index = 0; index < rank; ++index)
      componentCount *= kSpatialComponentCount;
    if (unknown.symmetric)
      componentCount = kSymmetricRankTwoComponentCount;
    layout.unknowns.push_back(
        {unknown.name, rank, unknown.symmetric, componentCount,
         layout.totalComponents});
    layout.totalComponents += componentCount;
  }
  return layout;
}

ComponentEnvironment
makeComponentEnvironment(const std::vector<std::string> &indices,
                         std::size_t component, bool symmetric) {
  ComponentEnvironment environment;
  if (indices.empty()) {
    if (component != 0)
      fail("scalar component index must be zero");
    return environment;
  }

  if (symmetric) {
    if (indices.size() != 2 ||
        component >= kSymmetricRankTwoComponentCount) {
      fail("invalid symmetric rank-two component environment");
    }
    for (std::size_t index = 0; index < indices.size(); ++index) {
      if (!environment
               .emplace(indices[index],
                        kSymmetricRankTwoIndices[component][index])
               .second) {
        fail("tensor component layout requires distinct free indices");
      }
    }
    return environment;
  }

  std::size_t componentCount = 1;
  for (std::size_t index = 0; index < indices.size(); ++index)
    componentCount *= kSpatialComponentCount;
  if (component >= componentCount)
    fail("tensor component is outside the declared rank");

  for (std::size_t reverse = 0; reverse < indices.size(); ++reverse) {
    const std::size_t index = indices.size() - 1 - reverse;
    const std::size_t indexComponent = component % kSpatialComponentCount;
    component /= kSpatialComponentCount;
    if (!environment.emplace(indices[index], indexComponent).second)
      fail("tensor component layout requires distinct free indices");
  }
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
      field.rank = unknownLayout.rank;
      field.symmetric = unknownLayout.symmetric;
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
                 const std::vector<RadialGeometry> &geometries,
                 const ComponentSystemLayout &componentLayout,
                 const std::vector<double> &unknown,
                 const std::vector<double> &unknownTangent,
                 const std::unordered_map<std::string, double> &parameters) {
  const std::size_t dofCount =
      componentLayout.totalComponents * domainLayout.totalSize;
  if (geometries.size() != domainLayout.grids.size())
    fail("internal radial geometry/domain count mismatch");
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
          makeComponentEnvironment(equation.indices, component,
                                   unknownLayout.symmetric);

      for (std::size_t domainIndex = 0; domainIndex < domainLayout.grids.size();
           ++domainIndex) {
        const auto &grid = domainLayout.grids[domainIndex];
        const RadialGeometry *geometry =
            geometries[domainIndex].enabled ? &geometries[domainIndex] : nullptr;
        const std::size_t offset = rowBase + domainLayout.offsets[domainIndex];
        DualGrid localResidual =
            evalExpr(equation.residual.get(), grid, localStates[domainIndex],
                     equationEnvironment, geometry, parameters);
        for (std::size_t i = 1; i + 1 < grid.size; ++i) {
          residual.value[offset + i] = localResidual.value[i];
          residual.tangent[offset + i] = localResidual.tangent[i];
        }
      }

      const ComponentEnvironment innerEnvironment =
          makeComponentEnvironment(inner.indices, component,
                                   unknownLayout.symmetric);
      const ComponentEnvironment outerEnvironment =
          makeComponentEnvironment(outer.indices, component,
                                   unknownLayout.symmetric);
      const DualGrid innerValue =
          evalExpr(inner.rhs.get(), firstGrid, localStates.front(),
                   innerEnvironment,
                   geometries.front().enabled ? &geometries.front() : nullptr,
                   parameters);
      const DualGrid outerValue =
          evalExpr(outer.rhs.get(), lastGrid, localStates.back(),
                   outerEnvironment,
                   geometries.back().enabled ? &geometries.back() : nullptr,
                   parameters);
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
    const std::vector<RadialGeometry> &geometries,
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
        evalExpr(reconstruction.meanCurvature.get(), grid, state, {},
                 geometries[domainIndex].enabled ? &geometries[domainIndex]
                                                 : nullptr,
                 parameters)
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

const ConstraintDomainSolution &
selectInterpolationDomain(const ConstraintSolution &solution, double radius) {
  if (!std::isfinite(radius) || radius <= 0.0)
    fail("CTT target-grid radii must be finite and strictly positive");
  for (const auto &domain : solution.domains) {
    if (domain.pointCount < 2 ||
        domain.offset + domain.pointCount > solution.coordinates.size())
      fail("invalid domain metadata in CTT solution");
    const double lower = solution.coordinates[domain.offset];
    const double upper =
        solution.coordinates[domain.offset + domain.pointCount - 1];
    double scale = std::max({1.0, std::abs(radius), std::abs(lower)});
    if (std::isfinite(upper))
      scale = std::max(scale, std::abs(upper));
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (radius + tolerance < lower)
      continue;
    if (domain.compactified || radius <= upper + tolerance)
      return domain;
  }
  fail("target radius lies outside the solved CTT domains");
}

double interpolateDomainProfile(const ConstraintSolution &solution,
                                const ConstraintDomainSolution &domain,
                                const std::vector<double> &profile,
                                double radius) {
  if (profile.size() != solution.coordinates.size())
    fail("CTT profile size does not match solution coordinates");
  const std::size_t n = domain.pointCount;
  const double lower = solution.coordinates[domain.offset];
  double spectralCoordinate = 0.0;
  if (domain.compactified) {
    spectralCoordinate = 1.0 - 2.0 * lower / radius;
  } else {
    const double upper = solution.coordinates[domain.offset + n - 1];
    spectralCoordinate =
        (2.0 * radius - lower - upper) / (upper - lower);
  }
  spectralCoordinate = std::clamp(spectralCoordinate, -1.0, 1.0);

  const double pi = std::acos(-1.0);
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double node =
        -std::cos(pi * static_cast<double>(i) / static_cast<double>(n - 1));
    const double distance = spectralCoordinate - node;
    if (std::abs(distance) <=
        32.0 * std::numeric_limits<double>::epsilon())
      return profile[domain.offset + i];
    const double endpointScale = (i == 0 || i + 1 == n) ? 0.5 : 1.0;
    const double weight = (i % 2 == 0 ? 1.0 : -1.0) * endpointScale;
    const double term = weight / distance;
    numerator += term * profile[domain.offset + i];
    denominator += term;
  }
  return numerator / denominator;
}

void storeSymmetricTensor(const std::array<double *, 9> &output,
                          std::size_t point,
                          const std::array<double, 9> &value) {
  for (std::size_t component = 0; component < value.size(); ++component)
    output[component][point] = value[component];
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
  const std::vector<RadialGeometry> geometries =
      buildRadialGeometries(problem, layout, request.parameters);
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
            makeComponentEnvironment(seed.indices, component,
                                     targetLayout.symmetric);
        const std::size_t offset =
            (targetLayout.firstComponent + component) * layout.totalSize +
            layout.offsets[domainIndex];
        auto localSeed =
            evalExpr(seed.rhs.get(), grid, localStates[domainIndex],
                     environment,
                     geometries[domainIndex].enabled ? &geometries[domainIndex]
                                                     : nullptr,
                     request.parameters)
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
         unknownLayout.componentCount, layout.totalSize,
         unknownLayout.symmetric});
  }
  const std::size_t maxIterations =
      static_cast<std::size_t>(problem.solve.maxIterations);

  for (std::size_t iteration = 0; iteration <= maxIterations; ++iteration) {
    DualGrid residual =
        evaluateResidual(problem, layout, geometries, componentLayout, unknown,
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

    Matrix jacobian(dofCount * dofCount, 0.0);
    for (std::size_t col = 0; col < dofCount; ++col) {
      std::vector<double> direction(dofCount, 0.0);
      direction[col] = 1.0;
      DualGrid differentiated =
          evaluateResidual(problem, layout, geometries, componentLayout,
                           unknown, direction, request.parameters);
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
          evaluateResidual(problem, layout, geometries, componentLayout,
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
        problem, layout, geometries, componentLayout, unknown,
        request.parameters);
  }
  return solution;
}

void interpolateRadialCttToGrid(const ConstraintSolution &solution,
                                const CttTargetGrid &target,
                                const CttEvolutionBuffers &outputs) {
  if (!solution.converged)
    fail("cannot interpolate an unconverged constraint solution");
  if (!solution.physicalCtt)
    fail("constraint solution has no reconstructed CTT physical fields");
  if (target.pointCount == 0)
    fail("CTT target grid must contain at least one point");
  for (const double *coordinate : target.coordinateComponents)
    if (!coordinate)
      fail("CTT target grid has a null coordinate component");
  for (std::size_t component = 0; component < 9; ++component) {
    if (!outputs.spatialMetric[component] ||
        !outputs.inverseSpatialMetric[component] ||
        !outputs.extrinsicCurvature[component])
      fail("CTT evolution output has a null tensor component");
  }

  const auto &physical = *solution.physicalCtt;
  for (std::size_t point = 0; point < target.pointCount; ++point) {
    for (const double *coordinate : target.coordinateComponents)
      if (!std::isfinite(coordinate[point]))
        fail("CTT target-grid coordinates must be finite");
    double radius = 0.0;
    std::array<double, 3> radialUnit{};
    double theta = 0.0;
    if (target.coordinates == CttTargetCoordinates::Spherical) {
      radius = target.coordinateComponents[0][point];
      theta = target.coordinateComponents[1][point];
    } else {
      const double x = target.coordinateComponents[0][point];
      const double y = target.coordinateComponents[1][point];
      const double z = target.coordinateComponents[2][point];
      radius = std::sqrt(x * x + y * y + z * z);
      if (radius > 0.0)
        radialUnit = {x / radius, y / radius, z / radius};
    }

    const auto &domain = selectInterpolationDomain(solution, radius);
    const double gammaRadial = interpolateDomainProfile(
        solution, domain, physical.spatialMetricRadial, radius);
    const double gammaTangential = interpolateDomainProfile(
        solution, domain, physical.spatialMetricTangential, radius);
    const double kRadial = interpolateDomainProfile(
        solution, domain, physical.extrinsicCurvatureRadial, radius);
    const double kTangential = interpolateDomainProfile(
        solution, domain, physical.extrinsicCurvatureTangential, radius);
    const double meanCurvature = interpolateDomainProfile(
        solution, domain, physical.meanCurvature, radius);
    if (gammaRadial <= 0.0 || gammaTangential <= 0.0 ||
        !std::isfinite(gammaRadial) || !std::isfinite(gammaTangential) ||
        !std::isfinite(kRadial) || !std::isfinite(kTangential) ||
        !std::isfinite(meanCurvature))
      fail("interpolated CTT physical tensor is invalid");

    std::array<double, 9> gamma{};
    std::array<double, 9> gammaInverse{};
    std::array<double, 9> extrinsic{};
    if (target.coordinates == CttTargetCoordinates::Spherical) {
      const double radiusSquared = radius * radius;
      const double sinTheta = std::sin(theta);
      const double azimuthScale = radiusSquared * sinTheta * sinTheta;
      gamma[0] = gammaRadial;
      gamma[4] = radiusSquared * gammaTangential;
      gamma[8] = azimuthScale * gammaTangential;
      gammaInverse[0] = 1.0 / gammaRadial;
      gammaInverse[4] = 1.0 / (radiusSquared * gammaTangential);
      gammaInverse[8] =
          azimuthScale == 0.0
              ? std::numeric_limits<double>::infinity()
              : 1.0 / (azimuthScale * gammaTangential);
      extrinsic[0] = kRadial;
      extrinsic[4] = radiusSquared * kTangential;
      extrinsic[8] = azimuthScale * kTangential;
    } else {
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
          const std::size_t component = 3 * i + j;
          const double delta = i == j ? 1.0 : 0.0;
          const double radialProjector = radialUnit[i] * radialUnit[j];
          gamma[component] =
              gammaTangential * delta +
              (gammaRadial - gammaTangential) * radialProjector;
          gammaInverse[component] =
              delta / gammaTangential +
              (1.0 / gammaRadial - 1.0 / gammaTangential) *
                  radialProjector;
          extrinsic[component] =
              kTangential * delta +
              (kRadial - kTangential) * radialProjector;
        }
      }
    }
    storeSymmetricTensor(outputs.spatialMetric, point, gamma);
    storeSymmetricTensor(outputs.inverseSpatialMetric, point, gammaInverse);
    storeSymmetricTensor(outputs.extrinsicCurvature, point, extrinsic);
    if (outputs.meanCurvature)
      outputs.meanCurvature[point] = meanCurvature;
  }
}

void initializeBssnFromRadialCtt(const ConstraintSolution &solution,
                                 const CttTargetGrid &target,
                                 const CttBssnBuffers &outputs,
                                 const BssnGaugeSeed &gauge) {
  if (target.coordinates != CttTargetCoordinates::Cartesian)
    fail("BSSN initialization currently requires Cartesian target coordinates");
  if (!outputs.chi || !outputs.meanCurvature)
    fail("CTT BSSN output has a null scalar component");
  for (std::size_t component = 0; component < 9; ++component) {
    if (!outputs.conformalMetric[component] ||
        !outputs.inverseConformalMetric[component] ||
        !outputs.traceFreeExtrinsicCurvature[component])
      fail("CTT BSSN output has a null tensor component");
  }
  const bool hasAnyShift = std::any_of(
      outputs.shift.begin(), outputs.shift.end(), [](const double *component) {
        return component != nullptr;
      });
  const bool hasAllShift = std::all_of(
      outputs.shift.begin(), outputs.shift.end(), [](const double *component) {
        return component != nullptr;
      });
  if (hasAnyShift && !hasAllShift)
    fail("CTT BSSN shift output must provide all three components or none");
  if (!std::isfinite(gauge.lapse) ||
      !std::all_of(gauge.shift.begin(), gauge.shift.end(),
                   [](double value) { return std::isfinite(value); }))
    fail("CTT BSSN gauge seed must be finite");

  std::array<std::vector<double>, 9> physicalMetric;
  std::array<std::vector<double>, 9> physicalInverseMetric;
  std::array<std::vector<double>, 9> physicalExtrinsicCurvature;
  auto allocatePointers = [&](std::array<std::vector<double>, 9> &storage) {
    std::array<double *, 9> pointers{};
    for (std::size_t component = 0; component < storage.size(); ++component) {
      storage[component].resize(target.pointCount);
      pointers[component] = storage[component].data();
    }
    return pointers;
  };
  std::vector<double> physicalMeanCurvature(target.pointCount);
  CttEvolutionBuffers physicalOutputs;
  physicalOutputs.spatialMetric = allocatePointers(physicalMetric);
  physicalOutputs.inverseSpatialMetric =
      allocatePointers(physicalInverseMetric);
  physicalOutputs.extrinsicCurvature =
      allocatePointers(physicalExtrinsicCurvature);
  physicalOutputs.meanCurvature = physicalMeanCurvature.data();
  interpolateRadialCttToGrid(solution, target, physicalOutputs);

  for (std::size_t point = 0; point < target.pointCount; ++point) {
    const double a = physicalMetric[0][point];
    const double b = physicalMetric[1][point];
    const double c = physicalMetric[2][point];
    const double d = physicalMetric[3][point];
    const double e = physicalMetric[4][point];
    const double f = physicalMetric[5][point];
    const double g = physicalMetric[6][point];
    const double h = physicalMetric[7][point];
    const double i = physicalMetric[8][point];
    const double determinant =
        a * (e * i - f * h) - b * (d * i - f * g) +
        c * (d * h - e * g);
    if (!(determinant > 0.0) || !std::isfinite(determinant))
      fail("CTT physical metric must have a finite positive determinant");

    const double chi = std::cbrt(1.0 / determinant);
    const double meanCurvature = physicalMeanCurvature[point];
    outputs.chi[point] = chi;
    outputs.meanCurvature[point] = meanCurvature;
    for (std::size_t component = 0; component < 9; ++component) {
      outputs.conformalMetric[component][point] =
          chi * physicalMetric[component][point];
      outputs.inverseConformalMetric[component][point] =
          physicalInverseMetric[component][point] / chi;
      outputs.traceFreeExtrinsicCurvature[component][point] =
          chi * (physicalExtrinsicCurvature[component][point] -
                 physicalMetric[component][point] * meanCurvature / 3.0);
    }
    if (outputs.lapse)
      outputs.lapse[point] = gauge.lapse;
    if (hasAllShift) {
      for (std::size_t component = 0; component < 3; ++component)
        outputs.shift[component][point] = gauge.shift[component];
    }
  }
}

} // namespace tensorium::solver
