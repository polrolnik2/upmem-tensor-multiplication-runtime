cmake_minimum_required(VERSION 3.20)
project(pim_matmul C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

option(BUILD_TESTS "Build host tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
set(PARAMS_FILE "${CMAKE_SOURCE_DIR}/defn/params.yaml" CACHE FILEPATH "Path to YAML runtime params")
set(PIM_MATMUL_DPU_BINARY_PATH "" CACHE STRING "Path to DPU kernel binary (set automatically by src/CMakeLists.txt)")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(ParseParams)
parse_params_to_defs(${PARAMS_FILE} RUNTIME_DEFS DPU_DEF_FLAGS)

add_subdirectory(src)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

message(STATUS "Runtime params from ${PARAMS_FILE}: ${RUNTIME_DEFS}")
message(STATUS "DPU kernel binary path: ${PIM_MATMUL_DPU_BINARY_PATH}")
message(STATUS "To use custom DPU binary path, pass: -DPIM_MATMUL_DPU_BINARY_PATH=/path/to/binary")
