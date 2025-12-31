#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tensorium::mlir {

std::unique_ptr<::mlir::Pass> createTensoriumEinsteinAnalyzeEinsumPass();

} // namespace tensorium::mlir
