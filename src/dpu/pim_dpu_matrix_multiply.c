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
static __dma_aligned uint8_t* matrix1_wram[2];
static __dma_aligned uint8_t* matrix2_wram[2];
static __dma_aligned uint16_t* result_wram[2];
static bool result_wram_valid[2] = {false, false};
static uint16_t result_writeback_row_tile[2];
static uint16_t result_writeback_col_tile[2];

static size_t matrix1_tile_size_bytes;
static size_t matrix2_tile_size_bytes;
static size_t result_tile_size_bytes;

// Buffer management state
static int current_compute_buffer = 0;
static int current_load_buffer = 1;

MUTEX_INIT(log_mutex);

BARRIER_INIT(my_barrier, NR_TASKLETS);

static inline void load_A_tile_from_mram(__mram_ptr void *src, uint8_t *dst, size_t bytes) {
    for (size_t offset = 0; offset < bytes; offset += 2048) {
        mram_read(src + offset, dst + offset, (bytes - offset) < 2048 ? (bytes - offset) : 2048);
    }
    mram_read(src, dst, bytes);
}

static inline void load_B_tile_from_mram(__mram_ptr void *src, uint8_t *dst, size_t bytes) {
    for (size_t offset = 0; offset < bytes; offset += 2048) {
        mram_read(src + offset, dst + offset, (bytes - offset) < 2048 ? (bytes - offset) : 2048);
    }
}

static inline void write_C_tile_to_mram(uint16_t *src, __mram_ptr void *dst, size_t bytes) {
    for (size_t offset = 0; offset < bytes; offset += 2048) {
        mram_write(src + offset, dst + offset, (bytes - offset) < 2048 ? (bytes - offset) : 2048);
    }
}

void compute_tile_tasklet(int tasklet_id, int n_tasklets,
                          int m_tile, int n_tile, int k_tile,
                          int buffer_idx) {
    // Tasklet 0 doesn't compute, only tasklets 1 to n_tasklets-1 participate
    if (tasklet_id == 0) {
        return; // Tasklet 0 handles memory operations only
    }
    
    int computing_tasklets = n_tasklets - 1; // Exclude tasklet 0
    int effective_tasklet_id = tasklet_id - 1; // Adjust ID for computation (0-based for computing tasklets)
    
    int rows_per_tasklet = (m_tile + computing_tasklets - 1) / computing_tasklets;
    int row0 = effective_tasklet_id * rows_per_tasklet;
    int row_max = (row0 + rows_per_tasklet) < m_tile ? (row0 + rows_per_tasklet) : m_tile;
    
    uint8_t* A_buf = matrix1_wram[buffer_idx];
    uint8_t* B_buf = matrix2_wram[buffer_idx];
    uint16_t* C_buf = result_wram[buffer_idx];

    for (int i = row0; i < row_max; ++i) {
        for (int j = 0; j < n_tile; ++j) {
            uint32_t sum = 0;
            for (int kk = 0; kk < k_tile; ++kk) {
                // Matrix B is column-major: B[kk][j] = B_buf[j * k_tile + kk]
                sum += (uint32_t)A_buf[i * k_tile + kk] * (uint32_t)B_buf[j * k_tile + kk];
            }
            C_buf[i * n_tile + j] += (uint16_t)sum;
        }
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
        
        // Validate that tile size is a multiple of 2048 bytes (required for DMA alignment)
        if (MATRIX_MULTIPLY_ARGUMENTS.wram_input_tile_size % 2048 != 0) {
            printf("[DPU %d] ERROR: Tile size %u is not a multiple of 2048\n", 
                   pid, MATRIX_MULTIPLY_ARGUMENTS.wram_input_tile_size);
            return -2; // Error code for invalid tile size
        }

        matrix1_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.matrix1_type_size;
        matrix2_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.matrix2_type_size;
        result_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.result_type_size;

        matrix1_wram[0] = (uint8_t*)mem_alloc(matrix1_tile_size_bytes);
        matrix2_wram[0] = (uint8_t*)mem_alloc(matrix2_tile_size_bytes);
        result_wram[0] = (uint16_t*)mem_alloc(result_tile_size_bytes);
        matrix1_wram[1] = (uint8_t*)mem_alloc(matrix1_tile_size_bytes);
        matrix2_wram[1] = (uint8_t*)mem_alloc(matrix2_tile_size_bytes);
        result_wram[1] = (uint16_t*)mem_alloc(result_tile_size_bytes);
        
        if (!matrix1_wram[0] || !matrix2_wram[0] || !result_wram[0] || 
            !matrix1_wram[1] || !matrix2_wram[1] || !result_wram[1]) {
            printf("[DPU %d] ERROR: Failed to allocate memory for matrix buffers\n", pid);
            return -1;
        }
        
        // Initialize result buffers to zero
        size_t result_elements = 2 * (MATRIX_MULTIPLY_ARGUMENTS.wram_input_tile_size / sizeof(uint16_t));
        for (size_t i = 0; i < result_elements; i++) {
            result_wram[0][i] = 0;
            result_wram[1][i] = 0;
        }
    }
    
    barrier_wait(&my_barrier);

    if (pid == 0) {
        if (MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows == 0 || MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols == 0 ||
            MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows == 0 || MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols == 0 ||
            MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows == 0 || MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols == 0) {
            printf("[DPU %d] ERROR: Division by zero detected in tile size or matrix dimension arguments\n", pid);
            return -3;
        }
    }
    barrier_wait(&my_barrier);

    uint16_t result_tiles_rowwise = MATRIX_MULTIPLY_ARGUMENTS.result_rows / MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows;
    uint16_t result_tiles_colwise = MATRIX_MULTIPLY_ARGUMENTS.result_cols / MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;

    uint16_t matrix1_tiles_rowwise = MATRIX_MULTIPLY_ARGUMENTS.matrix1_rows / MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows;
    uint16_t matrix1_tiles_colwise = MATRIX_MULTIPLY_ARGUMENTS.matrix1_cols / MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;

    uint16_t matrix2_tiles_rowwise = MATRIX_MULTIPLY_ARGUMENTS.matrix2_rows / MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows;
    uint16_t matrix2_tiles_colwise = MATRIX_MULTIPLY_ARGUMENTS.matrix2_cols / MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols;

    if (pid == 0) {
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
    }

    barrier_wait(&my_barrier);

    if (MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols != MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows) {
        if (pid == 0) {
            printf("[DPU %d] ERROR: Matrix tile dimensions mismatch: A_cols=%d, B_rows=%d\n", 
                   pid, MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols, MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows);
        }
        return -1;
    }

    if (pid == 0) {
        printf("[DPU %d] Matrix dimensions: A=%dx%d, B=%dx%d, C=%dx%d\n", 
               pid, MATRIX_MULTIPLY_ARGUMENTS.matrix1_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix1_cols,
               MATRIX_MULTIPLY_ARGUMENTS.matrix2_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix2_cols,
               MATRIX_MULTIPLY_ARGUMENTS.result_rows, MATRIX_MULTIPLY_ARGUMENTS.result_cols);
        printf("[DPU %d] Result tiles: %dx%d, Starting ping-pong buffer computation\n", 
               pid, result_tiles_rowwise, result_tiles_colwise);
    }

    // // Ping-pong buffer implementation
    int compute_buffer = 0;
    int load_buffer = 1;
    bool first_iteration = true;
    
    for (int i = 0; i < result_tiles_rowwise; i++) {
        for (int j = 0; j < result_tiles_colwise; j++) {
            if (pid == 0) {
                printf("[DPU %d] Processing result tile [%d,%d]\n", pid, i, j);
            }
            
            // Clear result buffer for new result tile
            if (pid == 0) {
                size_t result_elements = MATRIX_MULTIPLY_ARGUMENTS.wram_input_tile_size / sizeof(uint16_t);
                for (size_t idx = 0; idx < result_elements; idx++) {
                    result_wram[compute_buffer][idx] = 0;
                }
            }
            barrier_wait(&my_barrier);
            
            for (int k = 0; k < matrix1_tiles_colwise; k++) {
                if (pid == 0) {
                    printf("[DPU %d] K iteration %d/%d, buffers: compute=%d, load=%d\n", 
                           pid, k, matrix1_tiles_colwise-1, compute_buffer, load_buffer);
                }
                
                // // Thread 0: Handle memory operations for next iteration
                if (pid == 0) {
                    // Store previous results if they exist and are valid
                    if (!first_iteration && result_wram_valid[load_buffer]) {
                        __mram_ptr void *result_mram_addr = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.result_start_offset + DPU_MRAM_HEAP_POINTER + 
                            (result_writeback_row_tile[load_buffer] * result_tiles_colwise + 
                             result_writeback_col_tile[load_buffer]) * result_tile_size_bytes);
                        write_C_tile_to_mram(result_wram[load_buffer], result_mram_addr, 
                                            result_tile_size_bytes);
                        result_wram_valid[load_buffer] = false;
                        printf("[DPU %d] Wrote back result tile [%d,%d] to MRAM addr:%p from buffer %d\n", 
                               pid, result_writeback_row_tile[load_buffer], result_writeback_col_tile[load_buffer], 
                               result_mram_addr, load_buffer);
                    }
                    
                    // Load new tiles into load_buffer
                    __mram_ptr void *mram_addr_A = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix1_start_offset + DPU_MRAM_HEAP_POINTER + 
                        (i * matrix1_tiles_colwise + k) * matrix1_tile_size_bytes);
                    load_A_tile_from_mram(mram_addr_A, matrix1_wram[load_buffer], 
                                         matrix1_tile_size_bytes);
                    printf("[DPU %d] Loaded A tile for [%d,%d] from MRAM addr:%p to buffer %d\n", 
                           pid, i, k, mram_addr_A, load_buffer);
                    __mram_ptr void *mram_addr_B = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix2_start_offset + DPU_MRAM_HEAP_POINTER + 
                        (k * matrix2_tiles_colwise + j) * matrix2_tile_size_bytes);
                    printf("[DPU %d] Loading B tile for [%d,%d] from MRAM addr:%p to buffer %d\n", 
                           pid, k, j, mram_addr_B, load_buffer);
                    load_B_tile_from_mram(mram_addr_B, matrix2_wram[load_buffer], 
                                         matrix2_tile_size_bytes);
                    result_writeback_row_tile[load_buffer] = i;
                    result_writeback_col_tile[load_buffer] = j;
                }
                
                // // All threads except tasklet 0: Compute on compute_buffer while thread 0 was loading load_buffer
                if (!first_iteration && pid != 0) {
                    compute_tile_tasklet(pid, NR_TASKLETS, 
                                       MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows, 
                                       MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols, 
                                       MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols,
                                       compute_buffer);
                }
                
                // // Synchronize before buffer swap
                barrier_wait(&my_barrier);
                
                // // Update result tracking info for the buffer we just computed on
                if (pid == 0 && !first_iteration) {
                    result_wram_valid[compute_buffer] = true;
                }
                
                // // Swap buffers: what was being loaded becomes the compute buffer
                int temp = compute_buffer;
                compute_buffer = load_buffer;
                load_buffer = temp;

                first_iteration = false;

                // // Sync after buffer swap
                barrier_wait(&my_barrier);
            }
            
            // // Process the last loaded tiles (tasklet 0 doesn't compute)
            if (pid != 0) {
                compute_tile_tasklet(pid, NR_TASKLETS, 
                                   MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows, 
                                   MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols, 
                                   MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols,
                                   compute_buffer);
            }
            
            barrier_wait(&my_barrier);
            
            // // Mark final result as ready for writeback
            if (pid == 0) {
                result_wram_valid[compute_buffer] = true;
                result_writeback_row_tile[compute_buffer] = i;
                result_writeback_col_tile[compute_buffer] = j;
            }
        }
    }

    barrier_wait(&my_barrier);
    
    // Final writeback of any remaining results
    if (pid == 0) {
        printf("[DPU %d] Starting final writeback of remaining results\n", pid);
        for (int buf = 0; buf < 2; buf++) {
            if (result_wram_valid[buf]) {
                __mram_ptr void *result_mram_addr = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.result_start_offset + DPU_MRAM_HEAP_POINTER + 
                    (result_writeback_row_tile[buf] * result_tiles_colwise + 
                     result_writeback_col_tile[buf]) * result_tile_size_bytes);
                write_C_tile_to_mram(result_wram[buf], result_mram_addr, 
                                    result_tile_size_bytes);
                result_wram_valid[buf] = false;
            }
        }
        printf("[DPU %d] Matrix multiplication kernel completed successfully\n", pid);
    }

    // Wait for all operations to complete
    barrier_wait(&my_barrier);
    return 0;
}
