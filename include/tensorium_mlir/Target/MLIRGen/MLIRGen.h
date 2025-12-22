
#pragma once
#include "tensorium/Backend/DomainIR.hpp"

namespace tensorium_mlir {

struct MLIRGenOptions {
  bool enableNoOpPass = false;
};

void emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts = {});
} // namespace tensorium_mlir
