cmake_minimum_required(VERSION 3.20)
project(Tensorium LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TENSORIUM_WARNINGS "Enable warnings" ON)


find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "LLVM_DIR: ${LLVM_DIR}")
message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# Fallback for toolchain packages that do not export include directory
# variables in cache-visible form.
if (NOT LLVM_INCLUDE_DIRS AND DEFINED LLVM_INCLUDE_DIR)
  set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIR}")
endif()
if (NOT MLIR_INCLUDE_DIRS)
  get_filename_component(_tensorium_mlir_prefix "${MLIR_DIR}" DIRECTORY)
  get_filename_component(_tensorium_mlir_prefix "${_tensorium_mlir_prefix}" DIRECTORY)
  get_filename_component(_tensorium_mlir_prefix "${_tensorium_mlir_prefix}" DIRECTORY)
  set(MLIR_INCLUDE_DIRS "${_tensorium_mlir_prefix}/include")
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)


add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})
if (TENSORIUM_WARNINGS)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(lib)
add_subdirectory(tools)
