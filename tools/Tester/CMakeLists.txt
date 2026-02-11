add_executable(Tensorium_tester
  main.cpp
  Printer/Printer.cpp
)

target_link_libraries(Tensorium_tester
  PRIVATE tensorium
)

add_executable(Tensorium_unittests
  UnitTests.cpp
)

target_link_libraries(Tensorium_unittests
  PRIVATE tensorium
)
