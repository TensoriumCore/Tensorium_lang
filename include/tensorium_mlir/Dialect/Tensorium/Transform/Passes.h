#pragma once

#include "mlir/Pass/Pass.h"

namespace tensorium {
namespace mlir {

std::unique_ptr<::mlir::Pass> createTensoriumNoOpPass();
std::unique_ptr<::mlir::Pass> createTensoriumAnalysisPass();
void registerTensoriumTransformPasses();

} // namespace mlir
} // namespace tensorium

#include "TensoriumPasses.h.inc"
