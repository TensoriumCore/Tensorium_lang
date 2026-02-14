#include "tensorium/API/Compiler.hpp"

#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Validation/IRCanonicalize.hpp"
#include "tensorium/Validation/IRVerifier.hpp"
#include "tensorium/Validation/ProgramValidator.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tensorium::api {
namespace {

std::string readFile(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open file: " + path);

  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void appendDiagnostics(const validation::ValidationResult &result,
                       std::vector<std::string> &warnings,
                       std::vector<std::string> &errors) {
  for (const auto &diag : result.diags) {
    if (diag.kind == validation::Diagnostic::Kind::Error)
      errors.push_back(diag.message);
    else
      warnings.push_back(diag.message);
  }
}

void throwIfErrors(const char *stage, const std::vector<std::string> &errors) {
  if (errors.empty())
    return;

  std::ostringstream oss;
  oss << stage << " failed:";
  for (const auto &err : errors)
    oss << "\n  - " << err;
  throw std::runtime_error(oss.str());
}

} // namespace

CompileResult parseAndValidateSource(std::string_view source,
                                     const CompileOptions &opts) {
  std::string ownedSource(source);

  Lexer lex(ownedSource.c_str());
  Parser parser(lex);
  Program prog = parser.parseProgram();

  SemanticAnalyzer sem(prog, opts.mode);

  CompileResult out;
  out.warnings = sem.getWarnings();

  backend::ModuleIR module = backend::BackendBuilder::build(prog, sem);
  validation::canonicalizeDifferentialIR(module);
  validation::canonicalizeEinsteinIR(module);

  {
    std::vector<std::string> errors;
    auto verifyRes = validation::verifyIR(module);
    appendDiagnostics(verifyRes, out.warnings, errors);
    throwIfErrors("IR verification", errors);
  }

  if (opts.runProgramValidation) {
    std::vector<std::string> errors;
    auto validateRes = validation::validateProgram(module);
    appendDiagnostics(validateRes, out.warnings, errors);
    throwIfErrors("Program validation", errors);
  }

  out.module = std::move(module);
  return out;
}

CompileResult parseAndValidateFile(const std::string &path,
                                   const CompileOptions &opts) {
  return parseAndValidateSource(readFile(path), opts);
}

std::string emitMLIR(const backend::ModuleIR &module,
                     const tensorium_mlir::MLIRGenOptions &opts,
                     bool requireSuccessfulPipeline) {
  mlir::MLIRContext ctx;
  bool pipelineSuccess = true;
  auto moduleOp = tensorium_mlir::buildMLIRModule(module, ctx, opts,
                                                  &pipelineSuccess);

  std::string mlirText;
  llvm::raw_string_ostream os(mlirText);
  moduleOp->print(os);
  os.flush();

  if (requireSuccessfulPipeline && !pipelineSuccess)
    throw std::runtime_error("MLIR pipeline failed");

  return mlirText;
}

std::string emitLLVMIR(const backend::ModuleIR &module,
                       const tensorium_mlir::MLIRGenOptions &opts) {
  std::string llvmIR;
  if (!tensorium_mlir::emitLLVMIR(module, opts, &llvmIR))
    throw std::runtime_error("LLVM IR emission failed");
  return llvmIR;
}

std::string compileSourceToLLVMIR(
    std::string_view source, const CompileOptions &compileOpts,
    const tensorium_mlir::MLIRGenOptions &mlirOpts) {
  auto result = parseAndValidateSource(source, compileOpts);
  return api::emitLLVMIR(result.module, mlirOpts);
}

std::string compileFileToLLVMIR(
    const std::string &path, const CompileOptions &compileOpts,
    const tensorium_mlir::MLIRGenOptions &mlirOpts) {
  auto result = parseAndValidateFile(path, compileOpts);
  return api::emitLLVMIR(result.module, mlirOpts);
}

} // namespace tensorium::api
