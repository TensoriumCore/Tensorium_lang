add_library(tensorium
  AST/ASTPrinter.cpp
  AST/Expr.cpp
  Backend/BackendBuilder.cpp
  Lex/Lexer.cpp
  Parse/Parser.cpp
  Runtime/CpuRuntime.cpp
  Runtime/Eval.cpp
  Sema/Sema.cpp
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

add_library(tensorium_mlir_backend
  tensorium_mlir/Target/MLIRGen/MLIRGen.cpp
  tensorium_mlir/Init/Registry.cpp
  tensorium_mlir/Init/Passes.cpp
  tensorium_mlir/Pipeline/Pipeline.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/NoOpPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/Passes.cpp
)

add_dependencies(tensorium_mlir_backend
  TensoriumPassesIncGen
)

target_include_directories(tensorium_mlir_backend PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)


target_include_directories(tensorium_mlir_backend SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

set_target_properties(tensorium_mlir_backend PROPERTIES
  CXX_STANDARD 20
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(tensorium_mlir_backend PRIVATE
  # tensorium
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRMathDialect
)
