#pragma once

#include "tensorium/IR/DomainIR.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Location;
class OpBuilder;
class Value;
} // namespace mlir

namespace tensorium_mlir {

void emitEvolutionOps(
    mlir::OpBuilder &b, mlir::Location loc,
    const tensorium::backend::ModuleIR &module,
    const llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg);

} // namespace tensorium_mlir
