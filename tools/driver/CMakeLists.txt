
add_executable(Tensorium_cc
  main.cpp
)

target_link_libraries(Tensorium_cc PRIVATE
  tensoriumParse
  tensoriumSema
  tensoriumRuntime
  tensoriumLowering
  tensoriumValidation
  tensoriumMlirCodegen
)
