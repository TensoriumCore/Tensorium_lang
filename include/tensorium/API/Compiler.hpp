#pragma once

#include "tensorium/IR/DomainIR.hpp"
#include "tensorium/Core/CompilationMode.hpp"
#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"

#include <string>
#include <string_view>
#include <vector>

namespace tensorium::api {

struct CompileOptions {
  CompilationMode mode = CompilationMode::Executable;
  bool runProgramValidation = true;
};

struct CompileResult {
  backend::ModuleIR module;
  std::vector<std::string> warnings;
};

CompileResult parseAndValidateSource(std::string_view source,
                                     const CompileOptions &opts = {});

CompileResult parseAndValidateFile(const std::string &path,
                                   const CompileOptions &opts = {});

std::string emitMLIR(const backend::ModuleIR &module,
                     const tensorium_mlir::MLIRGenOptions &opts = {},
                     bool requireSuccessfulPipeline = true);

std::string emitLLVMIR(const backend::ModuleIR &module,
                       const tensorium_mlir::MLIRGenOptions &opts = {});

std::string compileSourceToLLVMIR(
    std::string_view source, const CompileOptions &compileOpts = {},
    const tensorium_mlir::MLIRGenOptions &mlirOpts = {});

std::string compileFileToLLVMIR(
    const std::string &path, const CompileOptions &compileOpts = {},
    const tensorium_mlir::MLIRGenOptions &mlirOpts = {});

} // namespace tensorium::api
