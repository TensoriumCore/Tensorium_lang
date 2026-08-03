#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "tensorium/IR/IRBase.hpp"

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
  // Read-only field buffers matching @tensorium_rhs argument order.
  std::vector<RhsFieldSoA> args;
  // Separate RHS buffers using the same argument indexing. Entries for fields
  // without a dt assignment must be empty.
  std::vector<RhsFieldSoA> outputs;
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

// Executes @tensorium_rhs over the valid interior grid. Output halo points are
// left untouched. Inputs and outputs must not alias.
RhsEvalResult evaluateTensoriumRHSGrid(::mlir::ModuleOp module,
                                       const RhsEvalDescriptor &desc);

// Advances the evolved fields in state by one explicit time step. The state
// uses @tensorium_rhs argument order; non-evolved fields and halo points remain
// unchanged. All stages are evaluated into separate internal RHS buffers.
RhsEvalResult advanceTensoriumState(
    ::mlir::ModuleOp module, const RhsGridSpec &grid,
    const std::vector<RhsFieldSoA> &state, double dt,
    tensorium::backend::TimeIntegrator integrator);

} // namespace tensorium_mlir
