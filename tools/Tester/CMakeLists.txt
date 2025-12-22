add_executable(Tensorium_tester
  main.cpp
  Printer/Printer.cpp
)

target_link_libraries(Tensorium_tester
  PRIVATE tensorium
)
