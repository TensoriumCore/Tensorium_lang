#include "MLIRGenPipeline.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/IndexRoleAnalysisPass.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "llvm/Support/raw_ostream.h"

namespace tensorium_mlir {

void addEinsteinPipelineSafe(::mlir::PassManager &pm,
                             const MLIRGenOptions &opts) {
  if (opts.enableMetricLoweringPass) {
    pm.addPass(tensorium::mlir::createTensoriumMetricLoweringPass());
  }
  if (opts.enableInitStdLoweringPass) {
    pm.addPass(tensorium::mlir::createTensoriumInitToStdPass());
  }
  if (opts.enableInitGridScfPass) {
    pm.addPass(tensorium::mlir::createTensoriumInitGridScfPass());
  }
  if (opts.enableInitGridAffinePass) {
    pm.addPass(tensorium::mlir::createTensoriumInitGridAffinePass());
  }
  if (opts.enableRhsGridScfPass) {
    pm.addPass(tensorium::mlir::createTensoriumRhsGridScfPass());
  }
  if (opts.enableRhsGridAffinePass) {
    pm.addPass(tensorium::mlir::createTensoriumRhsGridAffinePass());
  }
  if (opts.enableRhsGridParallelPass) {
    pm.addPass(tensorium::mlir::createTensoriumRhsGridParallelPass());
  }
  if (opts.enableStripSourceFuncsPass) {
    pm.addPass(tensorium::mlir::createTensoriumStripSourceFuncsPass());
  }

  if (opts.enableEinsteinLoweringPass) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinLoweringPass());
  }

  if (opts.enableIndexRoleAnalysisPass) {
    pm.addPass(tensorium::mlir::createTensoriumIndexRoleAnalysisPass());
  }

  const bool needValidity = opts.enableEinsteinValidityPass;
  const bool needCanon = opts.enableEinsteinCanonicalizePass;
  const bool needAnalyze = opts.enableEinsteinAnalyzeEinsumPass || needValidity;
  const bool needIndex = opts.enableIndexAnalyzePass || needValidity;

  if (needIndex) {
    pm.addPass(tensorium::mlir::createTensoriumIndexAnalyzePass());
  }

  if (needAnalyze) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinAnalyzeEinsumPass());
  }

  if (needCanon) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinCanonicalizePass());
  }

  if (needValidity) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinValidityPass());
  }

  if (opts.enableStencilLoweringPass) {
    pm.addPass(
        tensorium::mlir::createTensoriumStencilLoweringPass(opts.dx, opts.order));
  }
  if (opts.enableDissipationPass) {
    pm.addPass(tensorium::mlir::createTensoriumDissipationPass(
        opts.dissipationStrength, opts.dx));
  }
}

void addPostMLIRNormalizationPipeline(::mlir::PassManager &pm,
                                      const MLIRGenOptions &opts) {
  if (opts.enableMLIRInlinePass)
    pm.addPass(mlir::createInlinerPass());
  if (opts.enableMLIRCanonicalizePass)
    pm.addPass(mlir::createCanonicalizerPass());
  if (opts.enableMLIRCSEPass)
    pm.addPass(mlir::createCSEPass());
}

bool lowerModuleToLLVM(mlir::ModuleOp moduleOp, mlir::MLIRContext &ctx,
                       const MLIRGenOptions &opts) {
  mlir::PassManager pm(&ctx);
  if (opts.mlirPrintIRAfterFailure) {
    pm.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/true);
  }
  if (opts.mlirPassTiming) {
    llvm::errs() << "[Tensorium] pass timing: LLVM lowering pipeline\n";
    pm.enableTiming();
  }

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLowerAffinePass());
  if (opts.enableRhsGridParallelPass)
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  if (opts.enableRhsGridParallelPass)
    pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  return mlir::succeeded(pm.run(moduleOp));
}

} // namespace tensorium_mlir
