#include <string.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <dpu.h>

#include <math.h>

#include <matrix.h>

#include "dpu_pim_matrix_multiply_kernel_arguments.h"

#include "pim_matrix_multiplication_frame.h"

uint32_t calculate_pad_rows(uint32_t rows, uint32_t element_size) {
    uint32_t col_size = rows * element_size;
    uint32_t pad = (8 - (col_size % 8)) % 8;
    return pad / element_size;
}

uint32_t calculate_pad_cols(uint32_t cols, uint32_t element_size) {
    uint32_t row_size = cols * element_size;
    uint32_t pad = (8 - (row_size % 8)) % 8;
    return pad / element_size;
}

uint32_t get_matrix1_tile_size_bytes(const pim_matrix_multiplication_frame_t* frame) {
    return frame->matrix1_tile_rows * frame->matrix1_tile_cols * frame->matrix1_type_size;
}

uint32_t get_matrix2_tile_size_bytes(const pim_matrix_multiplication_frame_t* frame) {
    return frame->matrix2_tile_rows * frame->matrix2_tile_cols * frame->matrix2_type_size;
}

uint32_t get_result_tile_size_bytes(const pim_matrix_multiplication_frame_t* frame) {
    return frame->result_tile_rows * frame->result_tile_cols * frame->result_type_size;
}

static inline uint32_t align_up(uint32_t value, uint32_t multiple) {
    if (multiple == 0) return value;
    uint32_t remainder = value % multiple;
    return remainder == 0 ? value : value + (multiple - remainder);
}

static inline uint32_t align_rows_to_work_group(uint32_t rows, uint32_t work_group_size) {
    return align_up(rows, work_group_size);
}

static inline uint32_t align_cols_to_work_group(uint32_t cols, uint32_t num_groups) {
    return align_up(cols, num_groups);
}

static inline uint32_t align_rows_to_8bytes(uint32_t rows, uint32_t element_size) {
    return rows + calculate_pad_rows(rows, element_size);
}

static inline uint32_t align_cols_to_8bytes(uint32_t cols, uint32_t element_size) {
    return cols + calculate_pad_cols(cols, element_size);
}

static inline uint32_t align_dim_to_tile(uint32_t dim, uint32_t tile) {
    return align_up(dim, tile);
}

static inline void compute_tile_counts(uint32_t rows, uint32_t cols, uint32_t tile_rows, uint32_t tile_cols,
                                       uint32_t* num_row_tiles, uint32_t* num_col_tiles) {
    *num_row_tiles = (rows + tile_rows - 1) / tile_rows;
    *num_col_tiles = (cols + tile_cols - 1) / tile_cols;
}

static inline uint32_t tiled_buffer_size_bytes(uint32_t num_row_tiles, uint32_t num_col_tiles,
                                               uint32_t tile_rows, uint32_t tile_cols, uint32_t element_size) {
    return num_row_tiles * num_col_tiles * tile_rows * tile_cols * element_size;
}

Matrix * matrix_align(const Matrix *mat) {
    if (!mat) return NULL;
    int32_t pad_rows = calculate_pad_rows(mat->rows, mat->element_size);
    int32_t pad_cols = calculate_pad_cols(mat->cols, mat->element_size);
    Matrix *aligned = matrix_add_cols(mat, pad_cols, NULL);
    if (!aligned) {
        return NULL;
    }
    aligned = matrix_add_rows(aligned, pad_rows, NULL);
    if (!aligned) {
        matrix_free(aligned);
        return NULL;
    }
    return aligned;
}

static void find_optimal_work_group_config(uint32_t num_dpus, uint32_t matrix1_size, uint32_t matrix2_size,
                                          uint32_t* num_work_groups, uint32_t* work_group_size) {
    double best_cost = INFINITY;
    uint32_t best_num_work_groups = 1;
    uint32_t best_work_group_size = num_dpus;
    // Try all divisors of num_dpus to find the optimal configuration
    for (uint32_t nwg = 1; nwg * nwg <= num_dpus; nwg++) {
        if (num_dpus % nwg != 0) continue;
        uint32_t wgs = num_dpus / nwg;
        // Check both (nwg, wgs) and (wgs, nwg) if they are different
        double cost = (double)matrix2_size / nwg + (double)matrix1_size / wgs;
        if (cost < best_cost) {
            best_cost = cost;
            best_num_work_groups = nwg;
            best_work_group_size = wgs;
        }
    }
    *num_work_groups = best_num_work_groups;
    *work_group_size = best_work_group_size;
}

pim_matrix_multiplication_frame_t* create_pim_matrix_multiplication_frame_binary(uint32_t num_dpus, uint32_t dpu_offset,
                                                                        uint32_t matrix1_rows, uint32_t matrix1_cols,
                                                                        uint32_t matrix2_rows, uint32_t matrix2_cols,
                                                                        uint32_t result_rows, uint32_t result_cols,
                                                                        uint32_t matrix1_type_size, uint32_t matrix2_type_size, uint32_t result_type_size,
                                                                        const char* dpu_binary_path) {
    pim_matrix_multiplication_frame_t* frame = (pim_matrix_multiplication_frame_t*)malloc(sizeof(pim_matrix_multiplication_frame_t));
    if (!frame) {
        return NULL;
    }

    if (!dpu_binary_path) {
        fprintf(stderr, "DPU binary path cannot be NULL\n");
        free(frame);
        return NULL;
    }

    struct dpu_set_t set;
    DPU_ASSERT(dpu_alloc(num_dpus, NULL, &set));
    frame->dpu_set = set;

    uint32_t matrix1_size = matrix1_rows * matrix1_cols * matrix1_type_size;
    uint32_t matrix2_size = matrix2_rows * matrix2_cols * matrix2_type_size;

    // Find optimal work group configuration with round numbers
    uint32_t optimal_num_work_groups, optimal_work_group_size;
    find_optimal_work_group_config(num_dpus, matrix1_size, matrix2_size, 
                                  &optimal_num_work_groups, &optimal_work_group_size);
    
    frame->num_work_groups = optimal_num_work_groups;
    frame->work_group_size = optimal_work_group_size;
    frame->num_dpus = num_dpus;
    frame->matrix1_rows = matrix1_rows;
    frame->matrix1_cols = matrix1_cols;

    frame->matrix2_rows = matrix2_rows;
    frame->matrix2_cols = matrix2_cols;

    frame->result_rows = matrix1_rows;
    frame->result_cols = matrix2_cols;

    frame->matrix1_type_size = matrix1_type_size;
    frame->matrix2_type_size = matrix2_type_size;
    frame->result_type_size = result_type_size;

    // Calculate tile dimensions first, before memory layout
    frame->wram_input_tile_size = TILE_SIZE; // Size of input tile in WRAM

    uint32_t num_input_scalars_per_tile = frame->wram_input_tile_size / frame->matrix1_type_size;
    uint32_t tile_dim = (sqrt(num_input_scalars_per_tile));

    while ((tile_dim * tile_dim * frame->matrix1_type_size) % 8 != 0 && tile_dim > 1) {
        tile_dim--;
    }

    frame->matrix1_tile_rows = tile_dim;
    frame->matrix1_tile_cols = tile_dim;
    frame->matrix2_tile_rows = tile_dim;
    frame->matrix2_tile_cols = tile_dim;
    frame->result_tile_rows = tile_dim;
    frame->result_tile_cols = tile_dim;

    // Now calculate memory layout with proper tiled sizes
    uint32_t curr_offset = dpu_offset;
    frame->matrix1_start_offset = curr_offset;

    // Matrix1 memory layout calculation with tiling
    uint32_t matrix1_rows_aligned = align_rows_to_work_group(matrix1_rows, frame->work_group_size);
    uint32_t matrix1_split_rows = matrix1_rows_aligned / frame->work_group_size;
    uint32_t matrix1_rows_8byte_aligned = align_rows_to_8bytes(matrix1_split_rows, frame->matrix1_type_size);
    uint32_t matrix1_rows_tile_aligned = align_dim_to_tile(matrix1_rows_8byte_aligned, frame->matrix1_tile_rows);
    
    uint32_t matrix1_cols_8byte_aligned = align_cols_to_8bytes(matrix1_cols, frame->matrix1_type_size);
    uint32_t matrix1_cols_tile_aligned = align_dim_to_tile(matrix1_cols_8byte_aligned, frame->matrix1_tile_cols);
    
    // Calculate tiled size for matrix1
    uint32_t matrix1_num_row_tiles, matrix1_num_col_tiles;
    compute_tile_counts(matrix1_rows_tile_aligned, matrix1_cols_tile_aligned,
                        frame->matrix1_tile_rows, frame->matrix1_tile_cols,
                        &matrix1_num_row_tiles, &matrix1_num_col_tiles);
    uint32_t matrix1_size_aligned = tiled_buffer_size_bytes(matrix1_num_row_tiles, matrix1_num_col_tiles,
                                                           frame->matrix1_tile_rows, frame->matrix1_tile_cols,
                                                           matrix1_type_size);
    curr_offset += matrix1_size_aligned;
    frame->matrix2_start_offset = curr_offset;

    // Matrix2 memory layout calculation with tiling
    uint32_t matrix2_cols_aligned = align_cols_to_work_group(matrix2_cols, frame->num_work_groups);
    uint32_t matrix2_split_cols = matrix2_cols_aligned / frame->num_work_groups;
    uint32_t matrix2_cols_8byte_aligned = align_cols_to_8bytes(matrix2_split_cols, frame->matrix2_type_size);
    uint32_t matrix2_cols_tile_aligned = align_dim_to_tile(matrix2_cols_8byte_aligned, frame->matrix2_tile_cols);
    
    uint32_t matrix2_rows_8byte_aligned = align_rows_to_8bytes(matrix2_rows, frame->matrix2_type_size);
    uint32_t matrix2_rows_tile_aligned = align_dim_to_tile(matrix2_rows_8byte_aligned, frame->matrix2_tile_rows);
    
    // Calculate tiled size for matrix2  
    uint32_t matrix2_num_row_tiles, matrix2_num_col_tiles;
    compute_tile_counts(matrix2_rows_tile_aligned, matrix2_cols_tile_aligned,
                        frame->matrix2_tile_rows, frame->matrix2_tile_cols,
                        &matrix2_num_row_tiles, &matrix2_num_col_tiles);
    uint32_t matrix2_size_aligned = tiled_buffer_size_bytes(matrix2_num_row_tiles, matrix2_num_col_tiles,
                                                           frame->matrix2_tile_rows, frame->matrix2_tile_cols,
                                                           matrix2_type_size);
    curr_offset += matrix2_size_aligned;
    frame->result_start_offset = curr_offset;

    // Result memory layout calculation with tiling
    uint32_t result_rows_tile_aligned = matrix1_rows_tile_aligned;
    uint32_t result_cols_tile_aligned = matrix2_cols_tile_aligned;
    uint32_t result_num_row_tiles, result_num_col_tiles;
    compute_tile_counts(result_rows_tile_aligned, result_cols_tile_aligned,
                        frame->result_tile_rows, frame->result_tile_cols,
                        &result_num_row_tiles, &result_num_col_tiles);
    uint32_t result_size_aligned = tiled_buffer_size_bytes(result_num_row_tiles, result_num_col_tiles,
                                                          frame->result_tile_rows, frame->result_tile_cols,
                                                          frame->result_type_size);
    curr_offset += result_size_aligned / frame->num_dpus;
    frame->mem_frame_end = curr_offset;

    if (frame->mem_frame_end > 64 * 1024 * 1024) {
        printf("Memory frame exceeds 64MB limit\n");
        free(frame);
        return NULL;
    }

    frame->result_valid = false;

    DPU_ASSERT(dpu_load(frame->dpu_set, dpu_binary_path, NULL));

    return frame;
}

void pim_matrix_multiplication_frame_free(pim_matrix_multiplication_frame_t* frame) {
    if (!frame) return;
    DPU_ASSERT(dpu_free(frame->dpu_set));
    free(frame);
}

void pim_matrix_multiplication_frame_load_first_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix) {
    if (!frame || !matrix) return;
    
    Matrix *matrix_split_aligned = NULL;
    Matrix **submatrices = NULL;
    void **submatrices_data = NULL;
    bool *submatrices_data_populated = NULL;
    
    // Load first matrix into MRAM at the specified offset
    uint32_t matrix1_rows_wg_aligned = align_rows_to_work_group(frame->matrix1_rows, frame->work_group_size);
    uint32_t aligned_rows = matrix1_rows_wg_aligned - frame->matrix1_rows;
    matrix_split_aligned = matrix_add_rows(matrix, aligned_rows, NULL);
    if (!matrix_split_aligned) {
        fprintf(stderr, "Failed to align matrix rows for PIM frame\n");
        goto cleanup;
    }
    
    submatrices = matrix_split_by_rows(matrix_split_aligned, frame->work_group_size);
    if (!submatrices) {
        fprintf(stderr, "Failed to split matrix by rows for PIM frame\n");
        goto cleanup;
    }
    
    submatrices_data = (void**)malloc(frame->work_group_size * sizeof(void*));
    if (!submatrices_data) {
        fprintf(stderr, "Failed to allocate memory for submatrices data\n");
        goto cleanup;
    }
    
    submatrices_data_populated = (bool*)malloc(frame->work_group_size * sizeof(bool));
    if (!submatrices_data_populated) {
        fprintf(stderr, "Failed to allocate memory for submatrices data populated flags\n");
        goto cleanup;
    }
    
    for (uint32_t i = 0; i < frame->work_group_size; i++) {
        submatrices_data_populated[i] = false;
    }
    
    // Load the aligned matrix into MRAM
    uint32_t i;
    struct dpu_set_t dpu;
    DPU_FOREACH(frame->dpu_set, dpu, i) {
        if (!submatrices_data_populated[i % frame->work_group_size]) {
            submatrices[i % frame->work_group_size] = matrix_align(submatrices[i % frame->work_group_size]);
            if (!submatrices[i % frame->work_group_size]) {
                fprintf(stderr, "Failed to align submatrix for PIM frame\n");
                goto cleanup;
            }
            submatrices_data[i % frame->work_group_size] = matrix_get_data_4d_row_major_tiled(submatrices[i % frame->work_group_size], frame->matrix1_tile_rows, frame->matrix1_tile_cols);
            if (!submatrices_data[i % frame->work_group_size]) {
                fprintf(stderr, "Failed to get 4D tiled data from submatrix\n");
                goto cleanup;
            }
            submatrices_data_populated[i % frame->work_group_size] = true;
        }
        DPU_ASSERT(dpu_prepare_xfer(dpu, submatrices_data[i % frame->work_group_size]));
    }
    
    uint32_t offset = frame->matrix1_start_offset;
    // Calculate tiled size: need to account for tiles that may have padding
    uint32_t matrix_rows = submatrices[0]->rows;
    uint32_t matrix_cols = submatrices[0]->cols;
    uint32_t num_row_tiles, num_col_tiles;
    compute_tile_counts(matrix_rows, matrix_cols,
                        frame->matrix1_tile_rows, frame->matrix1_tile_cols,
                        &num_row_tiles, &num_col_tiles);
    uint32_t submatrix_size = tiled_buffer_size_bytes(num_row_tiles, num_col_tiles,
                                                     frame->matrix1_tile_rows, frame->matrix1_tile_cols,
                                                     frame->matrix1_type_size);
    #ifdef DEBUG
    printf("Loading first matrix submatrix size: %u bytes\n", submatrix_size);
    printf("Rows: %u, Cols: %u, Num Row Tiles: %u, Num Col Tiles: %u\n", matrix_rows, matrix_cols, num_row_tiles, num_col_tiles);
    #endif // DEBUG
    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, offset, submatrix_size, DPU_XFER_ASYNC));
    frame->result_valid = false; // Reset result validity after loading new matrix

cleanup:
    if (submatrices_data_populated) free(submatrices_data_populated);
    if (submatrices_data) free(submatrices_data);
    if (submatrices) free(submatrices);
    if (matrix_split_aligned) matrix_free(matrix_split_aligned);
}

void pim_matrix_multiplication_frame_load_second_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix) {
    if (!frame || !matrix) return;
    
    Matrix *matrix_split_aligned = NULL;
    Matrix **submatrices = NULL;
    void **submatrices_data = NULL;
    bool *submatrices_data_populated = NULL;
    
    // Load second matrix into MRAM at the specified offset
    uint32_t matrix2_cols_wg_aligned = align_cols_to_work_group(frame->matrix2_cols, frame->num_work_groups);
    uint32_t aligned_cols = matrix2_cols_wg_aligned - frame->matrix2_cols;
    matrix_split_aligned = matrix_add_cols(matrix, aligned_cols, NULL);
    if (!matrix_split_aligned) {
        fprintf(stderr, "Failed to align matrix cols for PIM frame\n");
        goto cleanup;
    }
    
    submatrices = matrix_split_by_cols(matrix_split_aligned, frame->num_work_groups);
    if (!submatrices) {
        fprintf(stderr, "Failed to split matrix by cols for PIM frame\n");
        goto cleanup;
    }
    
    submatrices_data = (void**)malloc(frame->num_work_groups * sizeof(void*));
    if (!submatrices_data) {
        fprintf(stderr, "Failed to allocate memory for submatrices data\n");
        goto cleanup;
    }
    
    submatrices_data_populated = (bool*)malloc(frame->num_work_groups * sizeof(bool));
    if (!submatrices_data_populated) {
        fprintf(stderr, "Failed to allocate memory for submatrices data populated flags\n");
        goto cleanup;
    }
    
    for (uint32_t i = 0; i < frame->num_work_groups; i++) {
        submatrices_data_populated[i] = false;
    }
    
    // Load the aligned matrix into MRAM
    uint32_t i;
    struct dpu_set_t dpu;
    DPU_FOREACH(frame->dpu_set, dpu, i) {
        if (!submatrices_data_populated[i / frame->work_group_size]) {
            submatrices[i / frame->work_group_size] = matrix_align(submatrices[i / frame->work_group_size]);
            if (!submatrices[i / frame->work_group_size]) {
                fprintf(stderr, "Failed to align submatrix for PIM frame\n");
                goto cleanup;
            }
            // For matrix2, we need column-major data in tiled format
            // First transpose the matrix, then get tiled data, which gives us column-major tiled format
            Matrix* transposed = matrix_transpose(submatrices[i / frame->work_group_size]);
            if (!transposed) {
                fprintf(stderr, "Failed to transpose submatrix for PIM frame\n");
                goto cleanup;
            }
            submatrices_data[i / frame->work_group_size] = matrix_get_data_4d_row_major_tiled(transposed, frame->matrix2_tile_cols, frame->matrix2_tile_rows);
            matrix_free(transposed);
            if (!submatrices_data[i / frame->work_group_size]) {
                fprintf(stderr, "Failed to get 4D tiled data from transposed submatrix\n");
                goto cleanup;
            }
            submatrices_data_populated[i / frame->work_group_size] = true;
        }
        DPU_ASSERT(dpu_prepare_xfer(dpu, submatrices_data[i / frame->work_group_size]));
    }
    
    uint32_t offset = frame->matrix2_start_offset;
    // Calculate tiled size for matrix2: note that we transposed it, so dimensions are swapped
    uint32_t matrix_rows = submatrices[0]->cols;  // After transpose: cols become rows
    uint32_t matrix_cols = submatrices[0]->rows;  // After transpose: rows become cols  
    uint32_t num_row_tiles, num_col_tiles;
    compute_tile_counts(matrix_rows, matrix_cols,
                        frame->matrix2_tile_cols, frame->matrix2_tile_rows,
                        &num_row_tiles, &num_col_tiles);  // Note: using tile_cols for rows, tile_rows for cols
    uint32_t submatrix_size = tiled_buffer_size_bytes(num_row_tiles, num_col_tiles,
                                                     frame->matrix2_tile_rows, frame->matrix2_tile_cols,
                                                     frame->matrix2_type_size);
    #ifdef DEBUG
    printf("Loading first matrix submatrix size: %u bytes\n", submatrix_size);
    printf("Rows: %u, Cols: %u, Num Row Tiles: %u, Num Col Tiles: %u\n", matrix_rows, matrix_cols, num_row_tiles, num_col_tiles);
    #endif // DEBUG
    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, offset, submatrix_size, DPU_XFER_ASYNC));
    frame->result_valid = false; // Reset result validity after loading new matrix

cleanup:
    if (submatrices_data_populated) free(submatrices_data_populated);
    if (submatrices_data) free(submatrices_data);
    if (submatrices) free(submatrices);
    if (matrix_split_aligned) matrix_free(matrix_split_aligned);
}

void pim_matrix_multiplication_frame_execute(pim_matrix_multiplication_frame_t* frame) {
    dpu_pim_matrix_multiply_kernel_arguments_t input_args;
    struct dpu_set_t dpu;
    input_args.matrix1_start_offset = frame->matrix1_start_offset;
    input_args.matrix2_start_offset = frame->matrix2_start_offset;
    input_args.result_start_offset = frame->result_start_offset;
    
    // Calculate matrix1 dimensions aligned to tile sizes
    uint32_t matrix1_rows_wg_aligned = align_rows_to_work_group(frame->result_rows, frame->work_group_size);
    uint32_t matrix1_split_rows = matrix1_rows_wg_aligned / frame->work_group_size;
    uint32_t matrix1_rows_8byte_aligned = align_rows_to_8bytes(matrix1_split_rows, frame->result_type_size);
    input_args.matrix1_rows = align_dim_to_tile(matrix1_rows_8byte_aligned, frame->matrix1_tile_rows);
    
    uint32_t matrix1_cols_8byte_aligned = align_cols_to_8bytes(frame->matrix1_cols, frame->matrix1_type_size);
    input_args.matrix1_cols = align_dim_to_tile(matrix1_cols_8byte_aligned, frame->matrix1_tile_cols);
    
    // Calculate matrix2 dimensions aligned to tile sizes
    uint32_t matrix2_rows_8byte_aligned = align_rows_to_8bytes(frame->matrix2_rows, frame->matrix2_type_size);
    input_args.matrix2_rows = align_dim_to_tile(matrix2_rows_8byte_aligned, frame->matrix2_tile_rows);
    
    uint32_t matrix2_cols_wg_aligned = align_cols_to_work_group(frame->matrix2_cols, frame->num_work_groups);
    uint32_t matrix2_split_cols = matrix2_cols_wg_aligned / frame->num_work_groups;
    uint32_t matrix2_cols_8byte_aligned = align_cols_to_8bytes(matrix2_split_cols, frame->result_type_size);
    input_args.matrix2_cols = align_dim_to_tile(matrix2_cols_8byte_aligned, frame->matrix2_tile_cols);
    
    // Result dimensions should match the aligned matrix dimensions
    input_args.result_rows = input_args.matrix1_rows;
    input_args.result_cols = input_args.matrix2_cols;
    input_args.matrix1_type_size = frame->matrix1_type_size;
    input_args.matrix2_type_size = frame->matrix2_type_size;
    input_args.result_type_size = frame->result_type_size;
    
    // Use pre-calculated tile dimensions from the frame
    input_args.wram_input_tile_size = frame->wram_input_tile_size;
    input_args.matrix1_tile_rows = frame->matrix1_tile_rows;
    input_args.matrix1_tile_cols = frame->matrix1_tile_cols;
    input_args.matrix2_tile_rows = frame->matrix2_tile_rows;
    input_args.matrix2_tile_cols = frame->matrix2_tile_cols;
    input_args.result_tile_rows = frame->result_tile_rows;
    input_args.result_tile_cols = frame->result_tile_cols;
    
    input_args.matrix1_original_rows = matrix1_split_rows;
    input_args.matrix1_original_cols = frame->matrix1_cols;
    input_args.matrix2_original_rows = frame->matrix2_rows;
    input_args.matrix2_original_cols = matrix2_split_cols;

    input_args.inline_load_offset = 0;
    input_args.inline_load_size = 0;  
    input_args.inline_retrieve_offset = 0;
    input_args.inline_retrieve_size = 0;  

    #ifdef DEBUG
    printf("Executing PIM matrix multiplication with parameters:\n");
    printf("Matrix1 Start Offset: %u, Matrix2 Start Offset: %u, Result Start Offset: %u\n",
           input_args.matrix1_start_offset, input_args.matrix2_start_offset, input_args.result_start_offset);
    printf("Matrix1 Rows: %u, Matrix1 Cols: %u\n", input_args.matrix1_rows, input_args.matrix1_cols);
    printf("Matrix2 Rows: %u, Matrix2 Cols: %u\n", input_args.matrix2_rows, input_args.matrix2_cols);
    printf("Result Rows: %u, Result Cols: %u\n", input_args.result_rows, input_args.result_cols);
    printf("Matrix1 Type Size: %u, Matrix2 Type Size: %u, Result Type Size: %u\n",
           input_args.matrix1_type_size, input_args.matrix2_type_size, input_args.result_type_size);
    printf("Matrix1 Tile: %ux%u, Matrix2 Tile: %ux%u, Result Tile: %ux%u\n",
           input_args.matrix1_tile_rows, input_args.matrix1_tile_cols,
           input_args.matrix2_tile_rows, input_args.matrix2_tile_cols,
           input_args.result_tile_rows, input_args.result_tile_cols);
    #endif // DEBUG

    DPU_FOREACH(frame->dpu_set, dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &input_args));
    }

    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_TO_DPU, "MATRIX_MULTIPLY_ARGUMENTS", 0,
                            sizeof(dpu_pim_matrix_multiply_kernel_arguments_t), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(frame->dpu_set, DPU_ASYNCHRONOUS));

    #ifdef DEBUG
    DPU_FOREACH(frame->dpu_set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
    #endif // DEBUG

    frame->result_valid = true; // Mark result as valid after execution
    return;
}

Matrix * pim_matrix_multiplication_frame_get_result(pim_matrix_multiplication_frame_t* frame) {
    if (!frame) {
        fprintf(stderr, "Frame is NULL\n");
        return NULL;
    }
    
    void ***submatrices_data = NULL;
    bool *submatrices_row_populated = NULL;
    Matrix ***submatrices = NULL;
    Matrix **row_submatrices = NULL;
    Matrix *result = NULL;
    
    uint32_t result_rows_wg_aligned = align_rows_to_work_group(frame->result_rows, frame->work_group_size);
    uint32_t result_cols_wg_aligned = align_cols_to_work_group(frame->matrix2_cols, frame->num_work_groups);
    uint32_t result_rows_frame_aligned = result_rows_wg_aligned / frame->work_group_size;
    uint32_t result_cols_frame_aligned = result_cols_wg_aligned / frame->num_work_groups;
    
    // First apply 8-byte alignment padding
    uint32_t result_rows_8byte_aligned = align_rows_to_8bytes(result_rows_frame_aligned, frame->matrix1_type_size);
    uint32_t result_cols_8byte_aligned = align_cols_to_8bytes(result_cols_frame_aligned, frame->matrix2_type_size);
    
    // Then ensure dimensions are multiples of tile dimensions for proper tiling
    uint32_t result_rows_dpu_transfer_aligned = align_dim_to_tile(result_rows_8byte_aligned, frame->result_tile_rows);
    uint32_t result_cols_dpu_transfer_aligned = align_dim_to_tile(result_cols_8byte_aligned, frame->result_tile_cols);
    
    // Calculate tiled data size for result transfer
    uint32_t result_num_row_tiles, result_num_col_tiles;
    compute_tile_counts(result_rows_dpu_transfer_aligned, result_cols_dpu_transfer_aligned,
                        frame->result_tile_rows, frame->result_tile_cols,
                        &result_num_row_tiles, &result_num_col_tiles);
    uint32_t result_size_aligned = tiled_buffer_size_bytes(result_num_row_tiles, result_num_col_tiles,
                                                          frame->result_tile_rows, frame->result_tile_cols,
                                                          frame->result_type_size);
    uint32_t result_submatrices_by_rows = frame->work_group_size;
    uint32_t result_submatrices_by_cols = frame->num_work_groups;
    
    submatrices_data = (void***)malloc(result_submatrices_by_rows * sizeof(void**));
    if (!submatrices_data) {
        fprintf(stderr, "Failed to allocate memory for submatrices data\n");
        goto cleanup;
    }
    
    submatrices_row_populated = (bool*)malloc(result_submatrices_by_rows * sizeof(bool));
    if (!submatrices_row_populated) {
        fprintf(stderr, "Failed to allocate memory for submatrices row populated flags\n");
        goto cleanup;
    }
    
    for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
        submatrices_row_populated[i] = false;
        submatrices_data[i] = NULL;
    }
    
    uint32_t i;
    struct dpu_set_t dpu;
    DPU_FOREACH(frame->dpu_set, dpu, i) {
        uint32_t row = i % result_submatrices_by_rows;
        uint32_t col = i / result_submatrices_by_rows;
        
        if (!submatrices_row_populated[row]) {
            submatrices_data[row] = malloc(result_submatrices_by_cols * sizeof(void*));
            if (!submatrices_data[row]) {
                fprintf(stderr, "Failed to allocate memory for submatrix data row\n");
                goto cleanup;
            }
            for (uint32_t j = 0; j < result_submatrices_by_cols; j++) {
                submatrices_data[row][j] = NULL;
            }
            submatrices_row_populated[row] = true;
        }
        
        submatrices_data[row][col] = malloc(result_size_aligned);
        if (!submatrices_data[row][col]) {
            fprintf(stderr, "Failed to allocate memory for submatrix data element\n");
            goto cleanup;
        }
        DPU_ASSERT(dpu_prepare_xfer(dpu, submatrices_data[row][col]));
    }
    
    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, frame->result_start_offset,
                            result_size_aligned, DPU_XFER_DEFAULT));
    
    submatrices = (Matrix***)malloc(result_submatrices_by_rows * sizeof(Matrix**));
    if (!submatrices) {
        fprintf(stderr, "Failed to allocate memory for submatrices\n");
        goto cleanup;
    }
    
    row_submatrices = (Matrix**)malloc(result_submatrices_by_rows * sizeof(Matrix*));
    if (!row_submatrices) {
        fprintf(stderr, "Failed to allocate memory for row submatrices\n");
        goto cleanup;
    }
    
    for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
        submatrices[i] = NULL;
        row_submatrices[i] = NULL;
    }
    
    for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
        submatrices[i] = (Matrix**)malloc(result_submatrices_by_cols * sizeof(Matrix*));
        if (!submatrices[i]) {
            fprintf(stderr, "Failed to allocate memory for submatrix row %u\n", i);
            goto cleanup;
        }
        
        for (uint32_t j = 0; j < result_submatrices_by_cols; j++) {
            submatrices[i][j] = NULL;
        }
        
        for (uint32_t j = 0; j < result_submatrices_by_cols; j++) {
            // Use the pre-calculated tile counts
            submatrices[i][j] = matrix_create_from_4d_row_major_tiled_array(
                result_num_row_tiles, result_num_col_tiles,
                frame->result_tile_rows, frame->result_tile_cols,
                submatrices_data[i][j], frame->result_type_size);
            if (!submatrices[i][j]) {
                fprintf(stderr, "Failed to create submatrix from row major array\n");
                goto cleanup;
            }
            
            Matrix *extracted = matrix_extract_submatrix(submatrices[i][j], result_rows_frame_aligned, result_cols_frame_aligned);
            if (!extracted) {
                fprintf(stderr, "Failed to extract submatrix\n");
                goto cleanup;
            }
            matrix_free(submatrices[i][j]);
            submatrices[i][j] = extracted;
            #ifdef DEBUG
            printf("Retrieved submatrix (%u, %u):\n", i, j);
            matrix_print(submatrices[i][j], "| %d |");
            #endif // DEBUG
        }
        
        row_submatrices[i] = matrix_join_by_cols(submatrices[i], result_submatrices_by_cols);
        if (!row_submatrices[i]) {
            fprintf(stderr, "Failed to join submatrices by columns\n");
            goto cleanup;
        }
    }
    
    result = matrix_join_by_rows(row_submatrices, result_submatrices_by_rows);
    if (!result) {
        fprintf(stderr, "Failed to join submatrices by rows\n");
        goto cleanup;
    }
    
    Matrix *final_result = matrix_extract_submatrix(result, frame->result_rows, frame->result_cols);
    if (!final_result) {
        fprintf(stderr, "Failed to extract final result submatrix\n");
        goto cleanup;
    }
    
    matrix_free(result);
    result = final_result;
    
    // Clean up intermediate allocations but keep the result
    if (submatrices) {
        for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
            if (submatrices[i]) {
                for (uint32_t j = 0; j < result_submatrices_by_cols; j++) {
                    if (submatrices[i][j]) {
                        matrix_free(submatrices[i][j]);
                    }
                }
                free(submatrices[i]);
            }
        }
        free(submatrices);
    }
    
    if (row_submatrices) {
        for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
            if (row_submatrices[i]) {
                matrix_free(row_submatrices[i]);
            }
        }
        free(row_submatrices);
    }
    
    if (submatrices_data) {
        for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
            if (submatrices_row_populated && submatrices_row_populated[i] && submatrices_data[i]) {
                for (uint32_t j = 0; j < result_submatrices_by_cols; j++) {
                    if (submatrices_data[i][j]) {
                        free(submatrices_data[i][j]);
                    }
                }
                free(submatrices_data[i]);
            }
        }
        free(submatrices_data);
    }
    
    if (submatrices_row_populated) free(submatrices_row_populated);
    
    return result;

cleanup:
    if (result) matrix_free(result);
    
    if (submatrices) {
        for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
            if (submatrices[i]) {
                for (uint32_t j = 0; j < result_submatrices_by_cols; j++) {
                    if (submatrices[i][j]) {
                        matrix_free(submatrices[i][j]);
                    }
                }
                free(submatrices[i]);
            }
        }
        free(submatrices);
    }
    
    if (row_submatrices) {
        for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
            if (row_submatrices[i]) {
                matrix_free(row_submatrices[i]);
            }
        }
        free(row_submatrices);
    }
    
    if (submatrices_data) {
        for (uint32_t i = 0; i < result_submatrices_by_rows; i++) {
            if (submatrices_row_populated && submatrices_row_populated[i] && submatrices_data[i]) {
                for (uint32_t j = 0; j < result_submatrices_by_cols; j++) {
                    if (submatrices_data[i][j]) {
                        free(submatrices_data[i][j]);
                    }
                }
                free(submatrices_data[i]);
            }
        }
        free(submatrices_data);
    }
    
    if (submatrices_row_populated) free(submatrices_row_populated);
    
    return NULL;
}

void pim_matrix_multiplication_frame_sync(pim_matrix_multiplication_frame_t* frame) {
    if (!frame) return;
    DPU_ASSERT(dpu_sync(frame->dpu_set));
}

pim_matrix_multiplication_frame_t* create_pim_matrix_multiplication_frame(
    uint32_t num_dpus, uint32_t dpu_offset, 
    uint32_t matrix1_rows, uint32_t matrix1_cols,
    uint32_t matrix2_rows, uint32_t matrix2_cols,
    uint32_t result_rows, uint32_t result_cols,
    uint32_t matrix1_type_size, uint32_t matrix2_type_size, uint32_t result_type_size) {
    #ifndef PIM_MATMUL_DPU_BINARY_PATH
    #error "PIM_MATMUL_DPU_BINARY_PATH is not defined. Please define it to the path of the DPU binary."
    #endif
    return create_pim_matrix_multiplication_frame_binary(num_dpus, dpu_offset,
                                                  matrix1_rows, matrix1_cols,
                                                  matrix2_rows, matrix2_cols,
                                                  result_rows, result_cols,
                                                  matrix1_type_size, matrix2_type_size, result_type_size,
                                                  PIM_MATMUL_DPU_BINARY_PATH);
}