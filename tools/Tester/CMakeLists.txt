add_executable(Tensorium_tester
  main.cpp
  Printer/Printer.cpp
)

target_link_libraries(Tensorium_tester
  PRIVATE
    tensoriumParse
    tensoriumSema
    tensoriumLowering
    tensoriumValidation
)

add_executable(Tensorium_unittests
  UnitTests.cpp
)

target_link_libraries(Tensorium_unittests
  PRIVATE
    tensoriumCompilerAPI
    tensoriumParse
    tensoriumSema
    tensoriumLowering
    tensoriumValidation
    tensoriumMlirRuntime
    tensoriumMlirCodegen
)

tensorium_configure_mlir_target(Tensorium_unittests)

target_include_directories(Tensorium_unittests SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
foreach(_tensorium_inc IN LISTS LLVM_INCLUDE_DIRS MLIR_INCLUDE_DIRS)
  target_compile_options(Tensorium_unittests PRIVATE "-isystem${_tensorium_inc}")
endforeach()
