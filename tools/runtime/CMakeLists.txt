add_executable(Tensorium_abi_probe
  abi_probe.cpp
)

target_link_libraries(Tensorium_abi_probe PRIVATE
  tensoriumParse
  tensoriumSema
  tensoriumLowering
  tensoriumValidation
  tensoriumMlirRuntime
  tensoriumMlirCodegen
)

tensorium_configure_mlir_target(Tensorium_abi_probe)

target_include_directories(Tensorium_abi_probe SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
foreach(_tensorium_inc IN LISTS LLVM_INCLUDE_DIRS MLIR_INCLUDE_DIRS)
  target_compile_options(Tensorium_abi_probe PRIVATE "-isystem${_tensorium_inc}")
endforeach()
