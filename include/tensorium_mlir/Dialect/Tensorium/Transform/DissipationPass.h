#pragma once
#include <memory>
namespace mlir {
class Pass;
}
namespace tensorium::mlir {
std::unique_ptr<::mlir::Pass>
createTensoriumDissipationPass(double strength = 0.1, double dx = 0.1);
}
