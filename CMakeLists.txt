cmake_minimum_required(VERSION 3.20)
project(Tensorium LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TENSORIUM_WARNINGS "Enable warnings" ON)

if (TENSORIUM_WARNINGS)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(lib)
add_subdirectory(tools)
