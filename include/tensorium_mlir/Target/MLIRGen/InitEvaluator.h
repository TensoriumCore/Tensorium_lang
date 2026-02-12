#pragma once

#include "mlir/IR/BuiltinOps.h"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

namespace tensorium_mlir {

struct InitCoordsSoA {
  const double *r = nullptr;
  const double *theta = nullptr;
  const double *phi = nullptr;
};

struct InitOutputsSoA {
  double *alpha = nullptr;
  std::array<double *, 9> gamma{};
  std::array<double *, 9> gammaU{};
};

struct InitEvalDescriptor {
  std::size_t nPoints = 0;
  std::unordered_map<std::string, double> params;
  InitCoordsSoA coords;
  InitOutputsSoA outputs;
};

struct InitEvalResult {
  bool ok = false;
  std::string message;

  static InitEvalResult success() { return {true, ""}; }
  static InitEvalResult failure(std::string msg) {
    return {false, std::move(msg)};
  }
};

// Executes the emitted tensorium_init function for each point in the descriptor.
// This is a front-only numeric evaluator and intentionally supports only the
// init op subset required by the current Schwarzschild milestone.
InitEvalResult evaluateTensoriumInit(::mlir::ModuleOp module,
                                     const InitEvalDescriptor &desc);

} // namespace tensorium_mlir
