function(tensorium_configure_public_headers target_name)
  target_include_directories(${target_name} PUBLIC
    ${PROJECT_SOURCE_DIR}/include
  )
endfunction()

add_library(tensoriumCore INTERFACE)
target_include_directories(tensoriumCore INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)

add_library(tensoriumAST
  AST/ASTPrinter.cpp
)
tensorium_configure_public_headers(tensoriumAST)
target_link_libraries(tensoriumAST PUBLIC
  tensoriumCore
)

add_library(tensoriumLex
  Lex/Lexer.cpp
)
tensorium_configure_public_headers(tensoriumLex)
target_link_libraries(tensoriumLex PUBLIC
  tensoriumCore
)

add_library(tensoriumParse
  Parse/Parser.cpp
  Parse/ParserDeclarations.cpp
  Parse/ParserExpressions.cpp
  Parse/ParserInitialData.cpp
  Parse/ParserSimulation.cpp
)
tensorium_configure_public_headers(tensoriumParse)
target_link_libraries(tensoriumParse PUBLIC
  tensoriumCore
  tensoriumAST
  tensoriumLex
)

add_library(tensoriumSema
  Sema/CallSupport.cpp
  Sema/SemaAnalysis.cpp
  Sema/SemaConstruction.cpp
  Sema/Sema.cpp
  Sema/SemaInitialData.cpp
  Sema/SemaMetricRules.cpp
  Sema/SemaNabla.cpp
  Sema/SemaSimulation.cpp
)
tensorium_configure_public_headers(tensoriumSema)
target_link_libraries(tensoriumSema PUBLIC
  tensoriumCore
  tensoriumAST
)

add_library(tensoriumRuntime
  Runtime/CpuRuntime.cpp
  Runtime/Eval.cpp
)
tensorium_configure_public_headers(tensoriumRuntime)


add_library(tensoriumIR INTERFACE)
target_include_directories(tensoriumIR INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(tensoriumIR INTERFACE
  tensoriumCore
)

target_link_libraries(tensoriumRuntime PUBLIC
  tensoriumCore
  tensoriumIR
)

add_library(tensoriumFrontend INTERFACE)
target_link_libraries(tensoriumFrontend INTERFACE
  tensoriumAST
  tensoriumLex
  tensoriumParse
  tensoriumSema
)

add_library(tensoriumLowering
  Lowering/BackendBuilder.cpp
)
tensorium_configure_public_headers(tensoriumLowering)
target_link_libraries(tensoriumLowering PUBLIC
  tensoriumCore
  tensoriumIR
  tensoriumAST
)

add_library(tensoriumValidation
  IR/CanonicalizeDiff.cpp
  IR/CanonicalizeEinstein.cpp
  IR/IRVerifier.cpp
  Validation/ProgramValidator.cpp
)
tensorium_configure_public_headers(tensoriumValidation)
target_link_libraries(tensoriumValidation PUBLIC
  tensoriumCore
  tensoriumIR
)

add_library(tensorium INTERFACE)
target_include_directories(tensorium INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(tensorium INTERFACE
  tensoriumCore
  tensoriumFrontend
  tensoriumLowering
  tensoriumRuntime
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

set(TENSORIUM_DIALECT_TD
  ${PROJECT_SOURCE_DIR}/include/tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.td
)
set(TENSORIUM_TYPES_TD
  ${PROJECT_SOURCE_DIR}/include/tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.td
)
set(TENSORIUM_OPS_TD
  ${PROJECT_SOURCE_DIR}/include/tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.td
)

set(LLVM_TARGET_DEFINITIONS
  ${TENSORIUM_TYPES_TD}
)

mlir_tablegen(TensoriumTypes.h.inc
  -gen-typedef-decls
  -I ${PROJECT_SOURCE_DIR}/include
  -I ${MLIR_INCLUDE_DIRS}
  -I ${LLVM_INCLUDE_DIRS}
  DEPENDS ${TENSORIUM_DIALECT_TD}
)

mlir_tablegen(TensoriumTypes.cpp.inc
  -gen-typedef-defs
  -I ${PROJECT_SOURCE_DIR}/include
  -I ${MLIR_INCLUDE_DIRS}
  -I ${LLVM_INCLUDE_DIRS}
  DEPENDS ${TENSORIUM_DIALECT_TD}
)

add_public_tablegen_target(TensoriumTypesIncGen)

set(LLVM_TARGET_DEFINITIONS
  ${TENSORIUM_OPS_TD}
)

mlir_tablegen(TensoriumOps.h.inc
  -gen-op-decls
  -I ${PROJECT_SOURCE_DIR}/include
  -I ${MLIR_INCLUDE_DIRS}
  -I ${LLVM_INCLUDE_DIRS}
  DEPENDS ${TENSORIUM_DIALECT_TD} ${TENSORIUM_TYPES_TD}
)

mlir_tablegen(TensoriumOps.cpp.inc
  -gen-op-defs
  -I ${PROJECT_SOURCE_DIR}/include
  -I ${MLIR_INCLUDE_DIRS}
  -I ${LLVM_INCLUDE_DIRS}
  DEPENDS ${TENSORIUM_DIALECT_TD} ${TENSORIUM_TYPES_TD}
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
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
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
  MLIROpenMPDialect
  MLIRTargetLLVMIRExport
  MLIROpenMPToLLVMIRTranslation
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
  TensoriumTypesIncGen
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
  TensoriumTypesIncGen
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
  tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenInitialData.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenExpr.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenSpectralResidual.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGen.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenOptions.cpp
  tensorium_mlir/Target/MLIRGen/MLIRGenPipeline.cpp
  tensorium_mlir/Init/Registry.cpp
  tensorium_mlir/Init/Passes.cpp
  tensorium_mlir/Pipeline/Pipeline.cpp
)
add_dependencies(tensoriumMlirCodegen
  TensoriumPassesIncGen
  TensoriumTypesIncGen
  TensoriumOpsIncGen
)
tensorium_configure_mlir_target(tensoriumMlirCodegen)
target_link_libraries(tensoriumMlirCodegen PUBLIC
  tensoriumMlirIR
  tensoriumMlirTransforms
  ${TENSORIUM_MLIR_COMMON_LINK_LIBS}
)

add_library(tensoriumMlirRuntime
  tensorium_mlir/Runtime/HostBuffers.cpp
)
tensorium_configure_mlir_target(tensoriumMlirRuntime)
target_link_libraries(tensoriumMlirRuntime PUBLIC
  tensoriumMlirCodegen
)

add_library(tensorium_mlir_backend INTERFACE)
target_link_libraries(tensorium_mlir_backend INTERFACE
  tensoriumMlirRuntime
  tensoriumMlirCodegen
  tensoriumMlirTransforms
  tensoriumMlirIR
  tensoriumMlirSemantic
)

add_library(tensoriumCompilerAPI
  API/Compiler.cpp
)
tensorium_configure_mlir_target(tensoriumCompilerAPI)
target_link_libraries(tensoriumCompilerAPI
  PUBLIC
    tensoriumLowering
    tensoriumValidation
    tensoriumMlirCodegen
  PRIVATE
    tensoriumParse
    tensoriumSema
)
