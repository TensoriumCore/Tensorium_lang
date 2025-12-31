
#pragma once
#include "tensorium/Backend/DomainIR.hpp"

namespace tensorium_mlir {

struct MLIRGenOptions {
  bool enableNoOpPass = false;
  bool enableAnalysisPass = false;
  bool enableEinsteinLoweringPass = false;
  bool enableIndexRoleAnalysisPass = false;
  bool enableEinsteinValidityPass = false;
  bool enableIndexAnalyzePass = false;
  bool enableEinsteinCanonicalizePass = false;
};

void emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts = {});
} // namespace tensorium_mlir
