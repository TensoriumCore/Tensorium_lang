#ifndef TENSORIUM_TRANSFORM_PASSES_TD
#define TENSORIUM_TRANSFORM_PASSES_TD

include "mlir/Pass/PassBase.td"

def TensoriumNoOpPass : Pass<"tensorium-noop", "mlir::ModuleOp"> {
  let summary = "Tensorium no-op pass";
  let description = [{
    No-op pass used to validate the Tensorium MLIR pipeline.
  }];
}

#endif // TENSORIUM_TRANSFORM_PASSES_TD
