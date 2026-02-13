#pragma once

#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
class PassManager;
} // namespace mlir

namespace tensorium_mlir {

void addEinsteinPipelineSafe(mlir::PassManager &pm, const MLIRGenOptions &opts);
void addPostMLIRNormalizationPipeline(mlir::PassManager &pm,
                                      const MLIRGenOptions &opts);
bool lowerModuleToLLVM(mlir::ModuleOp moduleOp, mlir::MLIRContext &ctx,
                       const MLIRGenOptions &opts);

} // namespace tensorium_mlir
