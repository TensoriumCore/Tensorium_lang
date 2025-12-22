#pragma once

#include "mlir/IR/DialectRegistry.h"

namespace tensorium_mlir {
void registerAllDialects(mlir::DialectRegistry &registry);
} // namespace tensorium_mlir
