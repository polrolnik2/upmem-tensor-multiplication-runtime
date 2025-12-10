#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <mram.h>
#include <alloc.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#include "dpu_pim_matrix_multiply_kernel_arguments.h"

__host dpu_pim_matrix_multiply_kernel_arguments_t MATRIX_MULTIPLY_ARGUMENTS;
__dma_aligned void* aux;

// Dual ping-pong buffers for matrix tiles
static __dma_aligned int8_t* matrix1_wram[2];
static __dma_aligned int8_t* matrix2_wram[2];
static __dma_aligned int16_t* result_wram[2];

static uint32_t matrix1_tile_size_bytes;
static uint32_t matrix2_tile_size_bytes;
static uint32_t result_tile_size_bytes;

static uint32_t rows_per_tasklet;

static uint32_t matrix1_tiles_rowwise;
static uint32_t matrix1_tiles_colwise;
static uint32_t matrix2_tiles_rowwise;
static uint32_t matrix2_tiles_colwise;
static uint32_t result_tiles_rowwise;
static uint32_t result_tiles_colwise;

static uint32_t result_elements;

MUTEX_INIT(log_mutex);

BARRIER_INIT(my_barrier, NR_TASKLETS);

static inline void load_A_tile_from_mram(__mram_ptr void *src, __dma_aligned void *dst, uint32_t bytes) {
    for (uint32_t offset = 0; offset < bytes; offset += 2048) {
        mram_read(src + offset, dst + offset, (bytes - offset) < 2048 ? (bytes - offset) : 2048);
    }
}

static inline void load_B_tile_from_mram(__mram_ptr void *src, __dma_aligned void *dst, uint32_t bytes) {
    for (uint32_t offset = 0; offset < bytes; offset += 2048) {
        mram_read(src + offset, dst + offset, (bytes - offset) < 2048 ? (bytes - offset) : 2048);
    }
}

static inline void write_C_tile_to_mram(__dma_aligned void *src, __mram_ptr void *dst, uint32_t bytes) {
    for (uint32_t offset = 0; offset < bytes; offset += 2048) {
        mram_write(src + offset, dst + offset, (bytes - offset) < 2048 ? (bytes - offset) : 2048);
    }
}

void cdot_accumulate(int8_t * A_buf, int8_t * B_buf, int16_t * C_buf,
                          uint32_t start_idx, uint32_t end_idx, 
                          uint32_t m_tile, uint32_t n_tile, uint32_t k_tile) {
    if (me() == 0) {
        return;
    }
    for (int c_idx = start_idx; c_idx < end_idx; ++c_idx) {
        int16_t sum = 0;
        uint32_t i = c_idx / MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
        uint32_t j = c_idx % MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
        for (int kk = 0; kk < k_tile; ++kk) {
            // Matrix B is column-major: B[kk][j] = B_buf[j * k_tile + kk]
            sum += (int16_t)A_buf[i * MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols + kk] * (int16_t)B_buf[j * MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows + kk];
        }
        C_buf[i * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols + j] += sum;
    }
}

/**
 * @brief Main DPU function for matrix multiplication with dual ping-pong buffers
 * 
 * This function implements a dual buffer system where:
 * - Thread 0 handles memory transfers (load/store) on one buffer
 * - Other threads perform computation on the other buffer
 * - Buffers are swapped after each tile computation to eliminate memory wait times
 */
int main() {
    int pid = me();
    
    // Initialize memory heap on tasklet 0
    if (pid == 0) {
        mem_reset(); // Reset the heap

        matrix1_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.matrix1_type_size;
        matrix2_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.matrix2_type_size;
        result_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.result_type_size;

        matrix1_wram[0] = (int8_t*)mem_alloc(matrix1_tile_size_bytes);
        matrix2_wram[0] = (int8_t*)mem_alloc(matrix2_tile_size_bytes);
        result_wram[0] = (int16_t*)mem_alloc(result_tile_size_bytes);
        matrix1_wram[1] = (int8_t*)mem_alloc(matrix1_tile_size_bytes);
        matrix2_wram[1] = (int8_t*)mem_alloc(matrix2_tile_size_bytes);
        result_wram[1] = (int16_t*)mem_alloc(result_tile_size_bytes);
        
        if (!matrix1_wram[0] || !matrix2_wram[0] || !result_wram[0] || 
            !matrix1_wram[1] || !matrix2_wram[1] || !result_wram[1]) {
            printf("[DPU %d] ERROR: Failed to allocate memory for matrix buffers\n", pid);
            return -1;
        }

        rows_per_tasklet = (MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows + NR_TASKLETS - 2) / (NR_TASKLETS - 1);

        matrix1_tiles_rowwise = MATRIX_MULTIPLY_ARGUMENTS.matrix1_rows / MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows;
        matrix1_tiles_colwise = MATRIX_MULTIPLY_ARGUMENTS.matrix1_cols / MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;

        matrix2_tiles_rowwise = MATRIX_MULTIPLY_ARGUMENTS.matrix2_rows / MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows;
        matrix2_tiles_colwise = MATRIX_MULTIPLY_ARGUMENTS.matrix2_cols / MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols;

        result_tiles_rowwise = matrix1_tiles_rowwise;
        result_tiles_colwise = matrix2_tiles_colwise;

        if (MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows == 0 || MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols == 0 ||
            MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows == 0 || MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols == 0 ||
            MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows == 0 || MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols == 0) {
            printf("[DPU %d] ERROR: Division by zero detected in tile size or matrix dimension arguments\n", pid);
            return -3;
        }

        #ifdef DEBUG
        printf("[DPU %d] Tile dimensions debug:\n", pid);
        printf("[DPU %d]   matrix1_tile_rows=%d, matrix1_tile_cols=%d (size=%zu bytes)\n", 
            pid, MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols, matrix1_tile_size_bytes);
        printf("[DPU %d]   matrix2_tile_rows=%d, matrix2_tile_cols=%d (size=%zu bytes)\n", 
            pid, MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols, matrix2_tile_size_bytes);
        printf("[DPU %d]   result_tile_rows=%d, result_tile_cols=%d (size=%zu bytes)\n", 
            pid, MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows, MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols, result_tile_size_bytes);
        printf("[DPU %d]   Calculated tiles: A=%dx%d, B=%dx%d, C=%dx%d\n", 
            pid, matrix1_tiles_rowwise, matrix1_tiles_colwise, 
            matrix2_tiles_rowwise, matrix2_tiles_colwise,
            result_tiles_rowwise, result_tiles_colwise);
        #endif

        if (MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols != MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows) {
            printf("[DPU %d] ERROR: Matrix tile dimensions mismatch: A_cols=%d, B_rows=%d\n", 
                pid, MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols, MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows);
            return -1;
        }

        #ifdef DEBUG
        printf("[DPU %d] Matrix dimensions: A=%dx%d, B=%dx%d, C=%dx%d\n", 
            pid, MATRIX_MULTIPLY_ARGUMENTS.matrix1_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix1_cols,
            MATRIX_MULTIPLY_ARGUMENTS.matrix2_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix2_cols,
            MATRIX_MULTIPLY_ARGUMENTS.result_rows, MATRIX_MULTIPLY_ARGUMENTS.result_cols);
        printf("[DPU %d] Result tiles: %dx%d, Starting ping-pong buffer computation\n", 
            pid, result_tiles_rowwise, result_tiles_colwise);
        #endif
        
        result_elements = MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
    }

    barrier_wait(&my_barrier);

    uint32_t start_idx;
    uint32_t end_idx;

    // Calculate effective dimensions for outermost tiles (only computed once)
    // Last row tile may have padding
    uint32_t last_row_tile_m = MATRIX_MULTIPLY_ARGUMENTS.matrix1_original_rows % MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows;
    if (last_row_tile_m == 0) last_row_tile_m = MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows;
    
    // Last column tile may have padding
    uint32_t last_col_tile_n = MATRIX_MULTIPLY_ARGUMENTS.matrix2_original_cols % MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
    if (last_col_tile_n == 0) last_col_tile_n = MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
    
    // Last k tile may have padding
    uint32_t last_k_tile_k = MATRIX_MULTIPLY_ARGUMENTS.matrix1_original_cols % MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;
    if (last_k_tile_k == 0) last_k_tile_k = MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;


    barrier_wait(&my_barrier);

    
    // Ping-pong buffer implementation with separate result buffer management
    int input_compute_buffer = 0;
    int input_load_buffer = 1;
    int result_buffer = 0;  // Result buffer switches only when we finish a complete result tile
    bool first_iteration = true;
    
    for (int i = 0; i < result_tiles_rowwise; i++) {
        for (int j = 0; j < result_tiles_colwise; j++) {
#ifdef DEBUG
            if (pid == 0) {
                printf("[DPU %d] Processing result tile [%d,%d] using result buffer %d\n", pid, i, j, 0);
            }
#endif  
            // Clear result buffer for new result tile
            if (pid == 0) {
                for (uint32_t idx = 0; idx < result_elements; idx++) {
                    result_wram[0][idx] = 0;
                }
            }

            // Determine effective m and n for this tile (check if it's an outermost tile)
            uint32_t effective_m = (i == result_tiles_rowwise - 1) ? last_row_tile_m : MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows;
            uint32_t effective_n = (j == result_tiles_colwise - 1) ? last_col_tile_n : MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;

            if (pid != 0) {
                uint32_t elements_for_this_tasklet = effective_m * effective_n;
                uint32_t tiles_per_tasklet = (elements_for_this_tasklet + (NR_TASKLETS - 2)) / (NR_TASKLETS - 1);
                start_idx = (pid - 1) * tiles_per_tasklet;
                end_idx = start_idx + tiles_per_tasklet;
                if (end_idx > elements_for_this_tasklet) {
                    end_idx = elements_for_this_tasklet;
                }
            }

            first_iteration = true; // Reset for next result tile

            barrier_wait(&my_barrier);
            
            // Accumulate across all K iterations for this result tile
            for (int k = 0; k < matrix1_tiles_colwise; k++) {
#ifdef DEBUG
                if (pid == 0) {
                    printf("[DPU %d] K iteration %d/%d, input buffers: compute=%d, load=%d, result buffer=%d\n", 
                           pid, k, matrix1_tiles_colwise-1, input_compute_buffer, input_load_buffer, 0);
                }
#endif
                
                // Thread 0: Handle memory operations for next iteration
                if (pid == 0) {
                    // Load new tiles into input_load_buffer
                    __mram_ptr void *mram_addr_A = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix1_start_offset + DPU_MRAM_HEAP_POINTER + 
                        (i * matrix1_tiles_colwise + k) * matrix1_tile_size_bytes);
                    load_A_tile_from_mram(mram_addr_A, matrix1_wram[input_load_buffer], 
                                         matrix1_tile_size_bytes);
                    __mram_ptr void *mram_addr_B = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix2_start_offset + DPU_MRAM_HEAP_POINTER + 
                        (j * matrix2_tiles_rowwise + k) * matrix2_tile_size_bytes);
#ifdef DEBUG
                    printf("[DPU %d] Loading A tile for [%d,%d] from MRAM addr:%p to input buffer %d\n", 
                           pid, i, k, mram_addr_A, input_load_buffer);
                    printf("[DPU %d] Loading B tile for [%d,%d] from MRAM addr:%p to input buffer %d\n", 
                           pid, k, j, mram_addr_B, input_load_buffer);
#endif
                    load_B_tile_from_mram(mram_addr_B, matrix2_wram[input_load_buffer], 
                                         matrix2_tile_size_bytes);
                }

                // Determine effective k for this iteration (check if it's the last k tile)
                uint32_t effective_k = (k == matrix1_tiles_colwise - 1) ? last_k_tile_k : MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;
                
                // All threads except tasklet 0: Compute on input_compute_buffer, accumulate into result_buffer
                if (!first_iteration && pid != 0) {
                    cdot_accumulate(matrix1_wram[input_compute_buffer], matrix2_wram[input_compute_buffer],
                                       result_wram[0],
                                       start_idx, end_idx,
                                       effective_m, effective_n, effective_k);
                }
                
                barrier_wait(&my_barrier);

                // Swap input buffers: what was being loaded becomes the compute buffer
                int temp = input_compute_buffer;
                input_compute_buffer = input_load_buffer;
                input_load_buffer = temp;
                first_iteration = false;

                // Sync after input buffer swap
                barrier_wait(&my_barrier);
            }
            
            // Final computation for the last K iteration
            if (pid != 0) {
                cdot_accumulate(matrix1_wram[input_compute_buffer], matrix2_wram[input_compute_buffer],
                                   result_wram[0],
                                   start_idx, end_idx,
                                   effective_m, effective_n, last_k_tile_k);
            }

            barrier_wait(&my_barrier);
            
            // Now that we've completed this result tile, write it back and switch result buffer
            if (pid == 0) {
                __mram_ptr void *result_mram_addr = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.result_start_offset + DPU_MRAM_HEAP_POINTER + 
                    ((i * result_tiles_colwise + j) * result_tile_size_bytes));
                write_C_tile_to_mram(result_wram[0], result_mram_addr, result_tile_size_bytes);
#ifdef DEBUG
                printf("[DPU %d] Wrote back completed result tile [%d,%d] to MRAM addr:%p from result buffer %d\n", 
                       pid, i, j, result_mram_addr, 0);
#endif
                // Switch to the other result buffer for the next result tile
                result_buffer = 1 - result_buffer;
            }
            
            barrier_wait(&my_barrier);
        }
    }

#ifdef DEBUG
    if (pid == 0) {
        printf("[DPU %d] Matrix multiplication kernel completed successfully\n", pid);
    }
#endif

    // Wait for all operations to complete
    barrier_wait(&my_barrier);
    return 0;
}
