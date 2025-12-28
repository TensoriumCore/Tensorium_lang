add_library(tensorium
  AST/ASTPrinter.cpp
  AST/Expr.cpp
  Backend/BackendBuilder.cpp
  Lex/Lexer.cpp
  Parse/Parser.cpp
  Runtime/CpuRuntime.cpp
  Runtime/Eval.cpp
  Sema/Sema.cpp
  Sema/ProgramValidator.cpp
)

target_include_directories(tensorium PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

set(LLVM_TARGET_DEFINITIONS
  ${PROJECT_SOURCE_DIR}/include/tensorium_mlir/Dialect/Tensorium/Transform/Passes.td
)

mlir_tablegen(TensoriumPasses.h.inc
  -gen-pass-decls
  -name Tensorium
  -I ${PROJECT_SOURCE_DIR}/include
  -I ${MLIR_INCLUDE_DIRS}
  -I ${LLVM_INCLUDE_DIRS}
)

add_public_tablegen_target(TensoriumPassesIncGen)

set(LLVM_TARGET_DEFINITIONS
  ${PROJECT_SOURCE_DIR}/include/tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.td
)

mlir_tablegen(TensoriumOps.h.inc
  -gen-op-decls
  -I ${PROJECT_SOURCE_DIR}/include
  -I ${MLIR_INCLUDE_DIRS}
  -I ${LLVM_INCLUDE_DIRS}
)

mlir_tablegen(TensoriumOps.cpp.inc
  -gen-op-defs
  -I ${PROJECT_SOURCE_DIR}/include
  -I ${MLIR_INCLUDE_DIRS}
  -I ${LLVM_INCLUDE_DIRS}
)

add_public_tablegen_target(TensoriumOpsIncGen)

add_library(tensorium_mlir_backend
  tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.cpp
  tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.cpp
  tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/AnalysisPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/NoOpPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/Passes.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGen.cpp
  tensorium_mlir/Init/Registry.cpp
  tensorium_mlir/Init/Passes.cpp
  tensorium_mlir/Pipeline/Pipeline.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinLoweringPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/IndexRoleAnalysisPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinValidityPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/IndexAnalyzePass.cpp
)

add_dependencies(tensorium_mlir_backend
  TensoriumPassesIncGen
  TensoriumOpsIncGen
)

target_include_directories(tensorium_mlir_backend SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/lib
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

# target_include_directories(tensorium_mlir_backend SYSTEM PRIVATE
#   ${LLVM_INCLUDE_DIRS}
#   ${MLIR_INCLUDE_DIRS}
# )

set_target_properties(tensorium_mlir_backend PROPERTIES
  CXX_STANDARD 20
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(tensorium_mlir_backend PRIVATE
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRInferTypeOpInterface
  MLIRBytecodeOpInterface
)
