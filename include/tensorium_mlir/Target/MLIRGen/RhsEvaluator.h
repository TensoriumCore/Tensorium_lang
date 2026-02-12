#pragma once

#include "mlir/IR/BuiltinOps.h"

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tensorium_mlir {

struct RhsGridSpec {
  unsigned spatialDim = 3;
  std::array<std::size_t, 3> extents{1, 1, 1};
  std::array<double, 3> spacing{1.0, 1.0, 1.0};
};

struct RhsFieldSoA {
  // One pointer per tensor component (row-major over tensor indices).
  std::vector<double *> components;
};

struct RhsEvalDescriptor {
  RhsGridSpec grid;
  // Point where @tensorium_rhs is evaluated.
  std::array<std::size_t, 3> point{0, 0, 0};
  // Per-argument field buffers matching @tensorium_rhs argument order.
  std::vector<RhsFieldSoA> args;
};

struct RhsEvalResult {
  bool ok = false;
  std::string message;

  static RhsEvalResult success() { return {true, ""}; }
  static RhsEvalResult failure(std::string msg) {
    return {false, std::move(msg)};
  }
};

// Executes @tensorium_rhs for a single interior point without LLVM/JIT.
// Supported ops intentionally target the current front pipeline:
// ref, deriv, add/sub/mul/div, contract, promote, dt_assign.
RhsEvalResult evaluateTensoriumRHS(::mlir::ModuleOp module,
                                   const RhsEvalDescriptor &desc);

} // namespace tensorium_mlir
