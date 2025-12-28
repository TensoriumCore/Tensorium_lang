#pragma once

#include "mlir/Pass/Pass.h"

namespace tensorium {
namespace mlir {

void registerTensoriumTransformPasses();
std::unique_ptr<::mlir::Pass> createTensoriumNoOpPass();
std::unique_ptr<::mlir::Pass> createTensoriumAnalysisPass();
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinLoweringPass();

std::unique_ptr<::mlir::Pass> createTensoriumIndexAnalyzePass();
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinLoweringPass();
std::unique_ptr<::mlir::Pass> createTensoriumIndexRoleAnalysisPass();
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinValidityPass();
} // namespace mlir
} // namespace tensorium

#include "TensoriumPasses.h.inc"
