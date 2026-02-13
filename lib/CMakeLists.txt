add_library(tensoriumCore
  AST/ASTPrinter.cpp
  AST/Expr.cpp
  Lex/Lexer.cpp
  Parse/Parser.cpp
  Runtime/CpuRuntime.cpp
  Runtime/Eval.cpp
  Sema/CallSupport.cpp
  Sema/Sema.cpp
  Sema/SemaInitialData.cpp
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
  IR/CanonicalizeDiff.cpp
  IR/CanonicalizeEinstein.cpp
  IR/IRVerifier.cpp
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

set(TENSORIUM_MLIR_COMMON_LINK_LIBS
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransforms
  MLIRConvertToLLVMPass
  MLIRConvertToLLVMInterface
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  MLIRMemRefTransforms
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRIndexToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRLLVMCommonConversion
  MLIRReconcileUnrealizedCasts
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRInferTypeOpInterface
  MLIRBytecodeOpInterface
)

function(tensorium_configure_mlir_target target_name)
  target_include_directories(${target_name} SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/lib
    ${CMAKE_CURRENT_BINARY_DIR}
    ${TENSORIUM_LLVM_MLIR_INCLUDE_DIRS}
  )

  # CMake may classify the LLVM/MLIR install include directory as implicit and
  # elide it from target include flags. Force-pass it to keep this target stable
  # across local toolchain layouts.
  foreach(_tensorium_inc IN LISTS TENSORIUM_LLVM_MLIR_INCLUDE_DIRS)
    target_compile_options(${target_name} PRIVATE "-isystem${_tensorium_inc}")
  endforeach()

  set_target_properties(${target_name} PROPERTIES
    CXX_STANDARD 20
    POSITION_INDEPENDENT_CODE ON
  )
endfunction()

add_library(tensoriumMlirIR
  tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.cpp
  tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.cpp
  tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.cpp
)
add_dependencies(tensoriumMlirIR
  TensoriumOpsIncGen
)
tensorium_configure_mlir_target(tensoriumMlirIR)
target_link_libraries(tensoriumMlirIR PUBLIC
  ${TENSORIUM_MLIR_COMMON_LINK_LIBS}
)

add_library(tensoriumMlirSemantic
  tensorium_mlir/Semantic/Einstein.cpp
)
tensorium_configure_mlir_target(tensoriumMlirSemantic)

add_library(tensoriumMlirTransforms
  tensorium_mlir/Dialect/Tensorium/Transforms/AnalysisPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/NoOpPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/Passes.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinLoweringPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/IndexRoleAnalysisPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinValidityPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/IndexAnalyzePass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinCanonicalizePass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinAnalyzeEinsumPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/MetricLoweringPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/InitToStdPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/InitGridScfPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/InitGridAffinePass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/RhsGridScfPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/StripSourceFuncsPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/StencilLoweringPass.cpp
  tensorium_mlir/Dialect/Tensorium/Transforms/DissipationPass.cpp
)
add_dependencies(tensoriumMlirTransforms
  TensoriumPassesIncGen
  TensoriumOpsIncGen
)
tensorium_configure_mlir_target(tensoriumMlirTransforms)
target_link_libraries(tensoriumMlirTransforms PUBLIC
  tensoriumMlirIR
  tensoriumMlirSemantic
  ${TENSORIUM_MLIR_COMMON_LINK_LIBS}
)

add_library(tensoriumMlirCodegen
  tensorium_mlir/Target/MLIRGen/InitEvaluator.cpp
  tensorium_mlir/Target/MLIRGen/RhsEvaluator.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenShared.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenInitialData.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenExpr.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGen.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenPipeline.cpp
  tensorium_mlir/Init/Registry.cpp
  tensorium_mlir/Init/Passes.cpp
  tensorium_mlir/Pipeline/Pipeline.cpp
)
add_dependencies(tensoriumMlirCodegen
  TensoriumPassesIncGen
  TensoriumOpsIncGen
)
tensorium_configure_mlir_target(tensoriumMlirCodegen)
target_link_libraries(tensoriumMlirCodegen PUBLIC
  tensoriumMlirIR
  tensoriumMlirTransforms
  ${TENSORIUM_MLIR_COMMON_LINK_LIBS}
)

add_library(tensorium_mlir_backend INTERFACE)
target_link_libraries(tensorium_mlir_backend INTERFACE
  tensoriumMlirCodegen
  tensoriumMlirTransforms
  tensoriumMlirIR
  tensoriumMlirSemantic
)
