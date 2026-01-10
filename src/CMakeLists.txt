# Core library and DPU kernel
set(PIM_DPU_BINARY ${CMAKE_CURRENT_BINARY_DIR}/matrix_multiply_dpu CACHE FILEPATH "Path to built DPU kernel")
set(PIM_MATMUL_DPU_BINARY_PATH "${PIM_DPU_BINARY}" CACHE STRING "DPU kernel binary path for runtime injection" FORCE)

# Get DPU flags directly from dpu-pkg-config
execute_process(
  COMMAND dpu-pkg-config --cflags dpu
  OUTPUT_VARIABLE DPU_CFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND dpu-pkg-config --libs dpu
  OUTPUT_VARIABLE DPU_LDFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(DPU_CFLAGS)
separate_arguments(DPU_LDFLAGS)

add_library(pim_matmul STATIC
  matrix.c
  pim_matrix_multiplication_frame.c
)

target_include_directories(pim_matmul
  PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/lib/simplepim
)

target_compile_definitions(pim_matmul PUBLIC ${RUNTIME_DEFS})

target_compile_definitions(pim_matmul PRIVATE "PIM_MATMUL_DPU_BINARY_PATH=\"${PIM_MATMUL_DPU_BINARY_PATH}\"")

target_compile_options(pim_matmul PUBLIC ${DPU_CFLAGS})

target_link_libraries(pim_matmul PUBLIC ${DPU_LDFLAGS} m)

find_program(DPU_CC dpu-upmem-dpurte-clang HINTS ENV UPMEM_SDK_ROOT PATH_SUFFIXES bin REQUIRED)
set(DPU_KERNEL_SRC ${CMAKE_SOURCE_DIR}/src/dpu/pim_dpu_matrix_multiply.c)
add_custom_command(
  OUTPUT ${PIM_DPU_BINARY}
  COMMAND ${DPU_CC} -O2 -g -I${CMAKE_SOURCE_DIR}/src ${DPU_DEF_FLAGS} -o ${PIM_DPU_BINARY} ${DPU_KERNEL_SRC}
  DEPENDS ${DPU_KERNEL_SRC} ${CMAKE_SOURCE_DIR}/src/dpu_pim_matrix_multiply_kernel_arguments.h
  COMMENT "Building DPU kernel"
)
add_custom_target(dpu_kernel ALL DEPENDS ${PIM_DPU_BINARY})
