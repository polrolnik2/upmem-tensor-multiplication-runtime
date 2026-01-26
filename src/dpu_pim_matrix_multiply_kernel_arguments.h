#ifndef __DPU_PIM_MATRIX_MULTIPLY_KERNEL_ARGUMENTS_H___
#define __DPU_PIM_MATRIX_MULTIPLY_KERNEL_ARGUMENTS_H___

typedef struct {
    uint32_t matrix1_start_offset;
    uint32_t matrix2_start_offset;
    uint32_t result_start_offset;
    uint32_t matrix1_rows;
    uint32_t matrix1_cols;
    uint32_t matrix1_tile_rows;
    uint32_t matrix1_tile_cols;
    uint32_t matrix2_rows;
    uint32_t matrix2_cols;
    uint32_t matrix2_tile_rows;
    uint32_t matrix2_tile_cols;
    uint32_t result_rows;
    uint32_t result_cols;
    uint32_t result_tile_rows;
    uint32_t result_tile_cols;
    uint32_t matrix1_type_size;
    uint32_t matrix2_type_size;
    uint32_t result_type_size;
    uint32_t wram_input_tile_size; // Size of input tile in WRAM
    uint32_t matrix1_original_rows;
    uint32_t matrix1_original_cols;
    uint32_t matrix2_original_rows;
    uint32_t matrix2_original_cols;
    uint32_t inline_load_offset;
    uint32_t inline_load_size;
    uint32_t inline_retrieve_offset;
    uint32_t inline_retrieve_size;
} dpu_pim_matrix_multiply_kernel_arguments_t;

#endif // __DPU_PIM_MATRIX_MULTIPLY_KERNEL_ARGUMENTS_H___