#pragma once

#include "tensorium_mlir/Runtime/GeneratedHostStorage.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

namespace tensorium_mlir::runtime {

enum class EllipticSolveStatus {
  MaxSteps,
  Converged,
  InvalidResidual,
};

struct EllipticSolveResult {
  std::string kernelSymbol;
  EllipticSolveStatus status = EllipticSolveStatus::MaxSteps;
  int steps = 0;
  int maxSteps = 0;
  std::int64_t stencilRadius = 1;
  double dt = 0.0;
  double initialResidualL2 = 0.0;
  double finalResidualL2 = 0.0;
  double residualRatio = std::numeric_limits<double>::infinity();

  bool converged() const { return status == EllipticSolveStatus::Converged; }
  bool residualIsFinite() const {
    return std::isfinite(initialResidualL2) && std::isfinite(finalResidualL2);
  }
};

using EllipticSolveObserver = void (*)(const EllipticSolveResult &,
                                       GeneratedHostStorage &, void *);

struct EllipticSolveOptions {
  double dt = 0.0;
  int maxSteps = 0;
  double residualTolerance = 0.0;
  double residualRatioTarget = 0.0;
  double jacobiWeight = 2.0 / 3.0;
  double jacobiDiagonal = 0.0;
  std::size_t expectedEulerUpdateCount = 0;
  bool preferParallelKernel = true;
  GeneratedHostGridSpacing spacing{};
  EllipticSolveObserver observer = nullptr;
  void *observerUserData = nullptr;
};

inline std::int64_t flatGridIndex(std::int64_t i, std::int64_t j,
                                  std::int64_t k, std::int64_t ny,
                                  std::int64_t nz) {
  return (i * ny + j) * nz + k;
}

inline const GeneratedHostKernelBindingPlan &
requireResidualGridKernel(const GeneratedHostStorage &storage,
                          bool preferParallel = true) {
  const char *parallelFirst[] = {"tensorium_residual_grid_parallel",
                                 "tensorium_residual_grid_affine",
                                 "tensorium_rhs_grid_parallel",
                                 "tensorium_rhs_grid_affine"};
  const char *serialFirst[] = {"tensorium_residual_grid_affine",
                               "tensorium_residual_grid_parallel",
                               "tensorium_rhs_grid_affine",
                               "tensorium_rhs_grid_parallel"};
  const auto candidates = preferParallel
                              ? std::span<const char *const>(parallelFirst)
                              : std::span<const char *const>(serialFirst);
  for (const char *candidate : candidates) {
    if (const auto *plan = storage.findKernelPlan(candidate))
      return *plan;
  }
  throw std::runtime_error("missing residual/rhs grid plan");
}

inline std::int64_t effectiveStencilRadius(
    const GeneratedHostKernelBindingPlan &plan) {
  return plan.stencilRadius > 0 ? plan.stencilRadius : 1;
}

inline double l2InteriorField(const double *values, GeneratedHostGridShape shape,
                              std::int64_t radius) {
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::int64_t i = radius; i < shape.nx - radius; ++i) {
    for (std::int64_t j = radius; j < shape.ny - radius; ++j) {
      for (std::int64_t k = radius; k < shape.nz - radius; ++k) {
        const double value = values[flatGridIndex(i, j, k, shape.ny, shape.nz)];
        if (!std::isfinite(value))
          return value;
        sum += value * value;
        ++count;
      }
    }
  }
  return count == 0 ? 0.0 : std::sqrt(sum / static_cast<double>(count));
}

inline double residualRatio(double initialResidualL2, double residualL2) {
  if (initialResidualL2 > 0.0)
    return residualL2 / initialResidualL2;
  return residualL2 == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

inline double maxAbsInteriorField(const double *values,
                                  GeneratedHostGridShape shape,
                                  std::int64_t radius) {
  double maxValue = 0.0;
  for (std::int64_t i = radius; i < shape.nx - radius; ++i) {
    for (std::int64_t j = radius; j < shape.ny - radius; ++j) {
      for (std::int64_t k = radius; k < shape.nz - radius; ++k) {
        const double value = values[flatGridIndex(i, j, k, shape.ny, shape.nz)];
        if (!std::isfinite(value))
          return value;
        maxValue = std::fmax(maxValue, std::fabs(value));
      }
    }
  }
  return maxValue;
}

inline std::vector<GeneratedHostEulerUpdate>
requireEulerUpdatePairs(const GeneratedHostStorage &storage,
                        std::size_t expectedCount) {
  auto updates = storage.eulerUpdatePairsFromDerivativePrefix();
  if (updates.size() != expectedCount) {
    throw std::runtime_error("Euler update plan mismatch: updates=" +
                             std::to_string(updates.size()));
  }
  return updates;
}

inline void invokeResidualGrid(
    GeneratedHostStorage &storage,
    std::span<const tensorium_host_kernel_adapter_desc> adapters,
    const GeneratedHostKernelBindingPlan &plan, std::span<const double> params,
    GeneratedHostGridSpacing spacing) {
  storage.invoke(adapters, plan.symbolName, params, spacing);
}

inline void runExplicitEulerRelaxation(
    GeneratedHostStorage &storage,
    std::span<const tensorium_host_kernel_adapter_desc> adapters,
    const GeneratedHostKernelBindingPlan &plan,
    std::span<const GeneratedHostEulerUpdate> updates,
    std::span<const double> params, GeneratedHostGridSpacing spacing, double dt,
    int steps) {
  for (int step = 0; step < steps; ++step) {
    storage.applyEulerUpdate(updates, dt);
    invokeResidualGrid(storage, adapters, plan, params, spacing);
  }
}

inline bool reachedResidualTarget(const EllipticSolveResult &result,
                                  const EllipticSolveOptions &options) {
  if (!result.residualIsFinite())
    return false;
  if (options.residualTolerance > 0.0 &&
      result.finalResidualL2 <= options.residualTolerance)
    return true;
  if (options.residualRatioTarget > 0.0 &&
      result.residualRatio <= options.residualRatioTarget)
    return true;
  return false;
}

inline double defaultJacobiDiagonal(GeneratedHostGridSpacing spacing) {
  return 2.0 / (spacing.dx * spacing.dx) + 2.0 / (spacing.dy * spacing.dy) +
         2.0 / (spacing.dz * spacing.dz);
}

inline void applyWeightedJacobiCorrection(double *unknownField,
                                          const double *residualField,
                                          GeneratedHostGridShape shape,
                                          std::int64_t radius, double weight,
                                          double diagonal) {
  if (!unknownField)
    throw std::runtime_error("weighted Jacobi unknown field is null");
  if (!residualField)
    throw std::runtime_error("weighted Jacobi residual field is null");
  if (!(weight > 0.0) || !std::isfinite(weight))
    throw std::runtime_error("weighted Jacobi weight must be finite and positive");
  if (!(diagonal > 0.0) || !std::isfinite(diagonal))
    throw std::runtime_error(
        "weighted Jacobi diagonal must be finite and positive");

  const double scale = weight / diagonal;
  for (std::int64_t i = radius; i < shape.nx - radius; ++i) {
    for (std::int64_t j = radius; j < shape.ny - radius; ++j) {
      for (std::int64_t k = radius; k < shape.nz - radius; ++k) {
        const std::int64_t p = flatGridIndex(i, j, k, shape.ny, shape.nz);
        unknownField[p] += scale * residualField[p];
      }
    }
  }
}

inline EllipticSolveResult solveExplicitEulerRelaxation(
    GeneratedHostStorage &storage,
    std::span<const tensorium_host_kernel_adapter_desc> adapters,
    std::span<const double> params, const double *residualField,
    const EllipticSolveOptions &options) {
  if (options.maxSteps < 0)
    throw std::runtime_error("elliptic solve maxSteps must be non-negative");
  if (!(options.dt >= 0.0) || !std::isfinite(options.dt))
    throw std::runtime_error("elliptic solve dt must be finite and non-negative");
  if (!residualField)
    throw std::runtime_error("elliptic solve residual field is null");

  const auto &plan =
      requireResidualGridKernel(storage, options.preferParallelKernel);
  const auto updates =
      options.expectedEulerUpdateCount > 0
          ? requireEulerUpdatePairs(storage, options.expectedEulerUpdateCount)
          : storage.eulerUpdatePairsFromDerivativePrefix();

  EllipticSolveResult result;
  result.kernelSymbol = plan.symbolName;
  result.maxSteps = options.maxSteps;
  result.stencilRadius = effectiveStencilRadius(plan);
  result.dt = options.dt;

  auto updateResidual = [&]() {
    result.finalResidualL2 =
        l2InteriorField(residualField, storage.shape(), result.stencilRadius);
    result.residualRatio =
        residualRatio(result.initialResidualL2, result.finalResidualL2);
  };
  auto notify = [&]() {
    if (options.observer)
      options.observer(result, storage, options.observerUserData);
  };
  invokeResidualGrid(storage, adapters, plan, params, options.spacing);
  result.initialResidualL2 =
      l2InteriorField(residualField, storage.shape(), result.stencilRadius);
  result.finalResidualL2 = result.initialResidualL2;
  result.residualRatio = residualRatio(result.initialResidualL2,
                                       result.finalResidualL2);
  if (!result.residualIsFinite()) {
    result.status = EllipticSolveStatus::InvalidResidual;
    notify();
    return result;
  }
  if (reachedResidualTarget(result, options)) {
    result.status = EllipticSolveStatus::Converged;
    notify();
    return result;
  }
  notify();

  for (int step = 1; step <= options.maxSteps; ++step) {
    storage.applyEulerUpdate(updates, options.dt);
    invokeResidualGrid(storage, adapters, plan, params, options.spacing);
    result.steps = step;
    updateResidual();
    if (!result.residualIsFinite()) {
      result.status = EllipticSolveStatus::InvalidResidual;
      notify();
      return result;
    }
    if (reachedResidualTarget(result, options)) {
      result.status = EllipticSolveStatus::Converged;
      notify();
      return result;
    }
    notify();
  }

  result.status = EllipticSolveStatus::MaxSteps;
  return result;
}

inline EllipticSolveResult solveWeightedJacobiRelaxation(
    GeneratedHostStorage &storage,
    std::span<const tensorium_host_kernel_adapter_desc> adapters,
    std::span<const double> params, double *unknownField,
    const double *residualField, const EllipticSolveOptions &options) {
  if (options.maxSteps < 0)
    throw std::runtime_error("elliptic solve maxSteps must be non-negative");
  if (!unknownField)
    throw std::runtime_error("weighted Jacobi unknown field is null");
  if (!residualField)
    throw std::runtime_error("weighted Jacobi residual field is null");

  const auto &plan =
      requireResidualGridKernel(storage, options.preferParallelKernel);
  const double diagonal =
      options.jacobiDiagonal > 0.0 ? options.jacobiDiagonal
                                   : defaultJacobiDiagonal(options.spacing);

  EllipticSolveResult result;
  result.kernelSymbol = plan.symbolName;
  result.maxSteps = options.maxSteps;
  result.stencilRadius = effectiveStencilRadius(plan);
  result.dt = 0.0;

  auto updateResidual = [&]() {
    result.finalResidualL2 =
        l2InteriorField(residualField, storage.shape(), result.stencilRadius);
    result.residualRatio =
        residualRatio(result.initialResidualL2, result.finalResidualL2);
  };
  auto notify = [&]() {
    if (options.observer)
      options.observer(result, storage, options.observerUserData);
  };

  invokeResidualGrid(storage, adapters, plan, params, options.spacing);
  result.initialResidualL2 =
      l2InteriorField(residualField, storage.shape(), result.stencilRadius);
  result.finalResidualL2 = result.initialResidualL2;
  result.residualRatio = residualRatio(result.initialResidualL2,
                                       result.finalResidualL2);
  if (!result.residualIsFinite()) {
    result.status = EllipticSolveStatus::InvalidResidual;
    notify();
    return result;
  }
  if (reachedResidualTarget(result, options)) {
    result.status = EllipticSolveStatus::Converged;
    notify();
    return result;
  }
  notify();

  for (int step = 1; step <= options.maxSteps; ++step) {
    applyWeightedJacobiCorrection(unknownField, residualField, storage.shape(),
                                  result.stencilRadius, options.jacobiWeight,
                                  diagonal);
    invokeResidualGrid(storage, adapters, plan, params, options.spacing);
    result.steps = step;
    updateResidual();
    if (!result.residualIsFinite()) {
      result.status = EllipticSolveStatus::InvalidResidual;
      notify();
      return result;
    }
    if (reachedResidualTarget(result, options)) {
      result.status = EllipticSolveStatus::Converged;
      notify();
      return result;
    }
    notify();
  }

  result.status = EllipticSolveStatus::MaxSteps;
  return result;
}

} // namespace tensorium_mlir::runtime
