
#include "tensorium_mlir/Dialect/Tensorium/Transform/NoOpPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace tensorium {
namespace mlir {

namespace {

struct TensoriumNoOpPass
    : public ::mlir::PassWrapper<TensoriumNoOpPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  void runOnOperation() override {llvm::errs() << "[Tensorium] NoOp pass executed\n";}
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumNoOpPass() {
  return std::make_unique<TensoriumNoOpPass>();
}

} // namespace mlir
} // namespace tensorium

