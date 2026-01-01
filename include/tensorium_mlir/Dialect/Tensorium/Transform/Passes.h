#pragma once

#include "mlir/Pass/Pass.h"

namespace tensorium {
namespace mlir {

void registerTensoriumTransformPasses();

std::unique_ptr<::mlir::Pass> createTensoriumNoOpPass();
std::unique_ptr<::mlir::Pass> createTensoriumAnalysisPass();
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinLoweringPass();
std::unique_ptr<::mlir::Pass> createTensoriumIndexAnalyzePass();
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinAnalyzeEinsumPass();
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinValidityPass();
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinCanonicalizePass();
std::unique_ptr<::mlir::Pass>
createTensoriumStencilLoweringPass(double dx = 0.1, int order = 2);
std::unique_ptr<::mlir::Pass>
createTensoriumDissipationPass(double strength = 0.1, double dx = 0.1);
} // namespace mlir
} // namespace tensorium

#include "TensoriumPasses.h.inc"
