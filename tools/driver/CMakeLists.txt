
add_executable(Tensorium_cc
  main.cpp
)

target_link_libraries(Tensorium_cc PRIVATE
  tensoriumLowering
  tensoriumValidation
  tensoriumMlirCodegen
)
