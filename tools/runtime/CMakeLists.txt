add_executable(Tensorium_abi_probe
  abi_probe.cpp
)

target_link_libraries(Tensorium_abi_probe PRIVATE
  tensoriumParse
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

add_executable(Tensorium_constraint_handoff_c_probe
  constraint_handoff_c_probe.c
)
set_target_properties(Tensorium_constraint_handoff_c_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  LINKER_LANGUAGE CXX
)
target_link_libraries(Tensorium_constraint_handoff_c_probe PRIVATE
  tensoriumConstraintHandoff
)

if (BUILD_TESTING)
  add_test(NAME tensorium_constraint_handoff_c_probe
    COMMAND Tensorium_constraint_handoff_c_probe
      ${PROJECT_SOURCE_DIR}/tests/fixtures/gr/ctt_radial_vacuum_solve.tn
      ${PROJECT_SOURCE_DIR}/tests/fixtures/gr/reissner_nordstrom_einstein_maxwell_solve.tn
  )
endif()
