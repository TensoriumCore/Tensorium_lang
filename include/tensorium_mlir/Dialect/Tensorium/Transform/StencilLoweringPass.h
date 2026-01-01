#pragma once
#include <memory>

namespace mlir { class Pass; }

namespace tensorium::mlir {
std::unique_ptr<::mlir::Pass> createTensoriumStencilLoweringPass();
}
