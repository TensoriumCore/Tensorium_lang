#pragma once
#include "mlir/Pass/Pass.h"

namespace tensorium {
namespace mlir {

std::unique_ptr<::mlir::Pass> createTensoriumAnalysisPass();

} // namespace mlir
} // namespace tensorium
