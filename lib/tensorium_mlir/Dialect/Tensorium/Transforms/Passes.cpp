#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "mlir/Pass/Pass.h"

namespace tensorium {
namespace mlir {

void registerTensoriumTransformPasses() {
  ::mlir::registerPass([] { return createTensoriumNoOpPass(); });
  ::mlir::registerPass([] { return createTensoriumAnalysisPass(); });
}

} // namespace mlir
} // namespace tensorium
