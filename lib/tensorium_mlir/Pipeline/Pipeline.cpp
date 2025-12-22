#include "tensorium_mlir/Pipeline/Pipeline.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

namespace tensorium_mlir {

void buildDefaultPipeline(mlir::PassManager &pm, const PipelineOptions &opt) {
  pm.enableVerifier(opt.verify);

  if (opt.printIR) {
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](mlir::Pass *,
                                     mlir::Operation *) { return true; },
        /*shouldPrintAfterPass=*/
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false);
  }

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

} // namespace tensorium_mlir
