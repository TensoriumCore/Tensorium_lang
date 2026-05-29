#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"

namespace tensorium_mlir {

void applyOptimizationLevel(MLIRGenOptions &opts, OptimizationLevel level) {
  switch (level) {
  case OptimizationLevel::O0:
    return;
  case OptimizationLevel::O1:
    opts.enableIndexAnalyzePass = true;
    opts.enableEinsteinAnalyzeEinsumPass = true;
    opts.enableEinsteinCanonicalizePass = true;
    opts.enableEinsteinValidityPass = true;
    return;
  case OptimizationLevel::O2:
    applyOptimizationLevel(opts, OptimizationLevel::O1);
    opts.enableEinsteinLoweringPass = true;
    opts.enableStencilLoweringPass = true;
    return;
  case OptimizationLevel::O3:
    applyOptimizationLevel(opts, OptimizationLevel::O2);
    opts.enableMetricLoweringPass = true;
    opts.enableInitStdLoweringPass = true;
    opts.enableInitGridAffinePass = true;
    opts.enableRhsGridAffinePass = true;
    opts.enableStripSourceFuncsPass = true;
    opts.enableMLIRInlinePass = true;
    return;
  }
}

void applyPassOptions(MLIRGenOptions &opts,
                      const MLIRPassOptions &passOptions) {
  opts.enableNoOpPass = opts.enableNoOpPass || passOptions.enableNoOpPass;
  opts.enableAnalysisPass =
      opts.enableAnalysisPass || passOptions.enableAnalysisPass;
  opts.enableEinsteinLoweringPass =
      opts.enableEinsteinLoweringPass ||
      passOptions.enableEinsteinLoweringPass;
  opts.enableIndexRoleAnalysisPass =
      opts.enableIndexRoleAnalysisPass ||
      passOptions.enableIndexRoleAnalysisPass;
  opts.enableEinsteinValidityPass =
      opts.enableEinsteinValidityPass ||
      passOptions.enableEinsteinValidityPass;
  opts.enableIndexAnalyzePass =
      opts.enableIndexAnalyzePass || passOptions.enableIndexAnalyzePass;
  opts.enableEinsteinCanonicalizePass =
      opts.enableEinsteinCanonicalizePass ||
      passOptions.enableEinsteinCanonicalizePass;
  opts.enableEinsteinAnalyzeEinsumPass =
      opts.enableEinsteinAnalyzeEinsumPass ||
      passOptions.enableEinsteinAnalyzeEinsumPass;
  opts.enableMetricLoweringPass =
      opts.enableMetricLoweringPass || passOptions.enableMetricLoweringPass;
  opts.enableInitStdLoweringPass =
      opts.enableInitStdLoweringPass || passOptions.enableInitStdLoweringPass;
  opts.enableInitGridScfPass =
      opts.enableInitGridScfPass || passOptions.enableInitGridScfPass;
  opts.enableInitGridAffinePass =
      opts.enableInitGridAffinePass || passOptions.enableInitGridAffinePass;
  opts.enableRhsGridScfPass =
      opts.enableRhsGridScfPass || passOptions.enableRhsGridScfPass;
  opts.enableRhsGridAffinePass =
      opts.enableRhsGridAffinePass || passOptions.enableRhsGridAffinePass;
  opts.enableRhsGridParallelPass =
      opts.enableRhsGridParallelPass || passOptions.enableRhsGridParallelPass;
  opts.enableStripSourceFuncsPass =
      opts.enableStripSourceFuncsPass || passOptions.enableStripSourceFuncsPass;
  opts.enableStencilLoweringPass =
      opts.enableStencilLoweringPass || passOptions.enableStencilLoweringPass;
  opts.dx = passOptions.dx;
  opts.order = passOptions.order;
  opts.enableDissipationPass =
      opts.enableDissipationPass || passOptions.enableDissipationPass;
  opts.dissipationStrength = passOptions.dissipationStrength;
  opts.enableMLIRCanonicalizePass =
      passOptions.enableMLIRCanonicalizePass;
  opts.enableMLIRCSEPass = passOptions.enableMLIRCSEPass;
  opts.enableMLIRInlinePass =
      opts.enableMLIRInlinePass || passOptions.enableMLIRInlinePass;
  opts.mlirDisableThreading =
      opts.mlirDisableThreading || passOptions.mlirDisableThreading;
  opts.mlirPrintOpOnDiagnostic =
      opts.mlirPrintOpOnDiagnostic || passOptions.mlirPrintOpOnDiagnostic;
  opts.mlirPrintIRAfterFailure =
      opts.mlirPrintIRAfterFailure || passOptions.mlirPrintIRAfterFailure;
  opts.mlirPassTiming = opts.mlirPassTiming || passOptions.mlirPassTiming;
}

MLIRGenOptions makeMLIRGenOptions(OptimizationLevel level) {
  MLIRGenOptions opts;
  applyOptimizationLevel(opts, level);
  return opts;
}

MLIRGenOptions makeMLIRGenOptions(OptimizationLevel level,
                                  const MLIRPassOptions &passOptions) {
  MLIRGenOptions opts = makeMLIRGenOptions(level);
  applyPassOptions(opts, passOptions);
  return opts;
}

} // namespace tensorium_mlir
