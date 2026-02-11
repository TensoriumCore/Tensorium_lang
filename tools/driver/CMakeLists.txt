
add_executable(Tensorium_cc
  main.cpp
)

target_link_libraries(Tensorium_cc PRIVATE
  tensoriumLowering
  tensoriumValidation
  tensorium_mlir_backend
)
