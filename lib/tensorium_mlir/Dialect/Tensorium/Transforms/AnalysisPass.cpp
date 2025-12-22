
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/NoOpPass.h"
#include "llvm/Support/raw_ostream.h"

namespace tensorium {
namespace mlir {

namespace {

struct TensoriumAnalysisPass
    : public ::mlir::PassWrapper<TensoriumAnalysisPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {

  void runOnOperation() override {
    auto module = getOperation();

    size_t numFuncs = 0;
    size_t numBlocks = 0;
    size_t numOps = 0;

    module.walk([&](::mlir::Operation *) { ++numOps; });

    for (auto func : module.getOps<::mlir::func::FuncOp>()) {
      ++numFuncs;
      numBlocks += func.getBody().getBlocks().size();
    }

    llvm::errs() << "[Tensorium][Analysis] Module statistics\n";
    llvm::errs() << "  functions : " << numFuncs << "\n";
    llvm::errs() << "  blocks    : " << numBlocks << "\n";
    llvm::errs() << "  ops       : " << numOps << "\n";

    if (numFuncs == 0) {
      module.emitError("TensoriumAnalysisPass: module contains no functions");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumAnalysisPass() {
  return std::make_unique<TensoriumAnalysisPass>();
}

} // namespace mlir
} // namespace tensorium
