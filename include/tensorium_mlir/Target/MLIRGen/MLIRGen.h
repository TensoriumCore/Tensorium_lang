
#pragma once
#include "tensorium/Backend/DomainIR.hpp"
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T>
class OwningOpRef;
} // namespace mlir

namespace tensorium_mlir {

struct MLIRGenOptions {
  bool enableNoOpPass = false;
  bool enableAnalysisPass = false;
  bool enableEinsteinLoweringPass = false;
  bool enableIndexRoleAnalysisPass = false;
  bool enableEinsteinValidityPass = false;
  bool enableIndexAnalyzePass = false;
  bool enableEinsteinCanonicalizePass = false;
  bool enableEinsteinAnalyzeEinsumPass = false;
  bool enableMetricLoweringPass = false;
  bool enableInitStdLoweringPass = false;
  bool enableInitGridScfPass = false;
  bool enableInitGridAffinePass = false;
  bool enableRhsGridScfPass = false;
  bool enableRhsGridAffinePass = false;
  bool enableStripSourceFuncsPass = false;
  bool enableStencilLoweringPass = false;
  double dx = 0.1;
  int order = 2;
  bool enableDissipationPass = false;
  double dissipationStrength = 0.1;
  // Post-MLIRGen module normalization (default on for stable test output).
  bool enableMLIRCanonicalizePass = true;
  bool enableMLIRCSEPass = true;
  // Optional compaction across function boundaries.
  bool enableMLIRInlinePass = false;
  // MLIR diagnostics/debug toggles.
  bool mlirDisableThreading = false;
  bool mlirPrintOpOnDiagnostic = false;
  bool mlirPrintIRAfterFailure = false;
};

mlir::OwningOpRef<mlir::ModuleOp>
buildMLIRModule(const tensorium::backend::ModuleIR &module,
                mlir::MLIRContext &ctx, const MLIRGenOptions &opts = {},
                bool *pipelineSuccess = nullptr);

bool emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts = {},
              std::string *mlirText = nullptr);

bool emitLLVMIR(const tensorium::backend::ModuleIR &module,
                const MLIRGenOptions &opts = {},
                std::string *llvmIRText = nullptr);
} // namespace tensorium_mlir
