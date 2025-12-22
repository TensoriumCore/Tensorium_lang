#pragma once

#include "mlir/Pass/PassManager.h"
namespace tensorium_mlir {
struct PipelineOptions {
  bool verify = true;
  bool printIR = false;
};
void buildDefaultPipeline(mlir::PassManager &pm, const PipelineOptions &opt);

} // namespace tensorium_mlir
