add_library(tensoriumCore
  AST/ASTPrinter.cpp
  AST/Expr.cpp
  Lex/Lexer.cpp
  Parse/Parser.cpp
  Runtime/CpuRuntime.cpp
  Runtime/Eval.cpp
  Sema/CallSupport.cpp
  Sema/Sema.cpp
)

target_include_directories(tensoriumCore PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

add_library(tensoriumIR INTERFACE)
target_include_directories(tensoriumIR INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)

add_library(tensoriumLowering
  Backend/BackendBuilder.cpp
)
target_include_directories(tensoriumLowering PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(tensoriumLowering PUBLIC
  tensoriumCore
  tensoriumIR
)

add_library(tensoriumValidation
  Validation/ProgramValidator.cpp
)
target_include_directories(tensoriumValidation PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(tensoriumValidation PUBLIC
  tensoriumIR
)

add_library(tensorium INTERFACE)
target_include_directories(tensorium INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(tensorium INTERFACE
  tensoriumCore
  tensoriumLowering
  tensoriumValidation
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
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinCanonicalizePass.cpp
  tensorium_mlir/Semantic/Einstein.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinAnalyzeEinsumPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/StencilLoweringPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/DissipationPass.cpp
)

add_dependencies(tensorium_mlir_backend
  TensoriumPassesIncGen
  TensoriumOpsIncGen
)

set(TENSORIUM_LLVM_MLIR_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
if (NOT TENSORIUM_LLVM_MLIR_INCLUDE_DIRS)
  get_filename_component(_tensorium_llvm_prefix "${LLVM_DIR}" DIRECTORY)
  get_filename_component(_tensorium_llvm_prefix "${_tensorium_llvm_prefix}" DIRECTORY)
  get_filename_component(_tensorium_llvm_prefix "${_tensorium_llvm_prefix}" DIRECTORY)
  if (EXISTS "${_tensorium_llvm_prefix}/include")
    list(APPEND TENSORIUM_LLVM_MLIR_INCLUDE_DIRS "${_tensorium_llvm_prefix}/include")
  endif()

  get_filename_component(_tensorium_mlir_prefix "${MLIR_DIR}" DIRECTORY)
  get_filename_component(_tensorium_mlir_prefix "${_tensorium_mlir_prefix}" DIRECTORY)
  get_filename_component(_tensorium_mlir_prefix "${_tensorium_mlir_prefix}" DIRECTORY)
  if (EXISTS "${_tensorium_mlir_prefix}/include")
    list(APPEND TENSORIUM_LLVM_MLIR_INCLUDE_DIRS "${_tensorium_mlir_prefix}/include")
  endif()
endif()
target_include_directories(tensorium_mlir_backend SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/lib
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${TENSORIUM_LLVM_MLIR_INCLUDE_DIRS}
)

# CMake may classify the LLVM/MLIR install include directory as implicit and
# elide it from target include flags. Force-pass it to keep this target stable
# across local toolchain layouts.
foreach(_tensorium_inc IN LISTS TENSORIUM_LLVM_MLIR_INCLUDE_DIRS)
  target_compile_options(tensorium_mlir_backend PRIVATE "-isystem${_tensorium_inc}")
endforeach()

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
