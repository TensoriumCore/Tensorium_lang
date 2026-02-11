
#pragma once
#include "tensorium/Backend/DomainIR.hpp"

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
  bool enableStencilLoweringPass = false;
  double dx = 0.1;
  int order = 2;
  bool enableDissipationPass = false;
  double dissipationStrength = 0.1;
};

mlir::OwningOpRef<mlir::ModuleOp>
buildMLIRModule(const tensorium::backend::ModuleIR &module,
                mlir::MLIRContext &ctx, const MLIRGenOptions &opts = {});

void emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts = {});
} // namespace tensorium_mlir
