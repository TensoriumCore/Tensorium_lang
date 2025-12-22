
add_executable(Tensorium_cc
  main.cpp
)

target_link_libraries(Tensorium_cc PRIVATE
  tensorium
  tensorium_mlir_backend
)

