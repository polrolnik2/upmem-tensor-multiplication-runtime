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

static size_t matrix1_tile_size_bytes;
static size_t matrix2_tile_size_bytes;
static size_t result_tile_size_bytes;

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
                          int input_buffer_idx, int result_buffer_idx) {
    // Tasklet 0 doesn't compute, only tasklets 1 to n_tasklets-1 participate
    if (tasklet_id == 0) {
        return; // Tasklet 0 handles memory operations only
    }
    
    int computing_tasklets = n_tasklets - 1; // Exclude tasklet 0
    int effective_tasklet_id = tasklet_id - 1; // Adjust ID for computation (0-based for computing tasklets)
    
    int rows_per_tasklet = (m_tile + computing_tasklets - 1) / computing_tasklets;
    int row0 = effective_tasklet_id * rows_per_tasklet;
    int row_max = (row0 + rows_per_tasklet) < m_tile ? (row0 + rows_per_tasklet) : m_tile;
    
    uint8_t* A_buf = matrix1_wram[input_buffer_idx];
    uint8_t* B_buf = matrix2_wram[input_buffer_idx];
    uint16_t* C_buf = result_wram[result_buffer_idx];

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

    uint16_t matrix1_tiles_rowwise = MATRIX_MULTIPLY_ARGUMENTS.matrix1_rows / MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows;
    uint16_t matrix1_tiles_colwise = MATRIX_MULTIPLY_ARGUMENTS.matrix1_cols / MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;

    uint16_t matrix2_tiles_rowwise = MATRIX_MULTIPLY_ARGUMENTS.matrix2_rows / MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows;
    uint16_t matrix2_tiles_colwise = MATRIX_MULTIPLY_ARGUMENTS.matrix2_cols / MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols;

    uint16_t result_tiles_rowwise = matrix1_tiles_rowwise;
    uint16_t result_tiles_colwise = matrix2_tiles_colwise;

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

    // Ping-pong buffer implementation with separate result buffer management
    int input_compute_buffer = 0;
    int input_load_buffer = 1;
    int result_buffer = 0;  // Result buffer switches only when we finish a complete result tile
    size_t result_elements = MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
    bool first_iteration = true;
    
    for (int i = 0; i < result_tiles_rowwise; i++) {
        for (int j = 0; j < result_tiles_colwise; j++) {
            if (pid == 0) {
                printf("[DPU %d] Processing result tile [%d,%d] using result buffer %d\n", pid, i, j, 0);
            }
            
            // Clear result buffer for new result tile
            if (pid == 0) {
                for (size_t idx = 0; idx < result_elements; idx++) {
                    result_wram[0][idx] = 0;
                }
                first_iteration = true; // Reset for next result tile
                printf("[DPU %d] Result tile data: ", pid);
                for (int b = 0; b < 16; b++) {
                    printf("%02X ", result_wram[0][b]);
                }
                printf("\n");
            }
            barrier_wait(&my_barrier);
            
            // Accumulate across all K iterations for this result tile
            for (int k = 0; k < matrix1_tiles_colwise; k++) {
                if (pid == 0) {
                    printf("[DPU %d] K iteration %d/%d, input buffers: compute=%d, load=%d, result buffer=%d\n", 
                           pid, k, matrix1_tiles_colwise-1, input_compute_buffer, input_load_buffer, 0);
                }
                
                // Thread 0: Handle memory operations for next iteration
                if (pid == 0) {
                    // Load new tiles into input_load_buffer
                    __mram_ptr void *mram_addr_A = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix1_start_offset + DPU_MRAM_HEAP_POINTER + 
                        (i * matrix1_tiles_colwise + k) * matrix1_tile_size_bytes);
                    load_A_tile_from_mram(mram_addr_A, matrix1_wram[input_load_buffer], 
                                         matrix1_tile_size_bytes);
                    printf("[DPU %d] Loaded A tile for [%d,%d] from MRAM addr:%p to input buffer %d\n", 
                           pid, i, k, mram_addr_A, input_load_buffer);
                    printf("[DPU %d] A tile data (first 8 bytes): ", pid);
                    for (int b = 0; b < matrix1_tile_size_bytes; b++) {
                        printf("%02X ", matrix1_wram[input_load_buffer][b]);
                    }
                    printf("\n");
                    __mram_ptr void *mram_addr_B = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix2_start_offset + DPU_MRAM_HEAP_POINTER + 
                        (k * matrix2_tiles_colwise + j) * matrix2_tile_size_bytes);
                    printf("[DPU %d] Loading B tile for [%d,%d] from MRAM addr:%p to input buffer %d\n", 
                           pid, k, j, mram_addr_B, input_load_buffer);
                    load_B_tile_from_mram(mram_addr_B, matrix2_wram[input_load_buffer], 
                                         matrix2_tile_size_bytes);
                    printf("Tile B data: ");
                    for (int b = 0; b < matrix2_tile_size_bytes; b++) {
                        printf("%02X ", matrix2_wram[input_load_buffer][b]);
                    }
                    printf("\n");
                }
                
                // All threads except tasklet 0: Compute on input_compute_buffer, accumulate into result_buffer
                if (!first_iteration && pid != 0) {
                    compute_tile_tasklet(pid, NR_TASKLETS, 
                                       MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows, 
                                       MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols, 
                                       MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols,
                                       input_compute_buffer, 0);
                }

                if(pid == 0 && !first_iteration) {
                    printf("[DPU %d] Result tile data: ", pid);
                    for (int b = 0; b < 16; b++) {
                        printf("%02X ", result_wram[0][b]);
                    }
                    printf("\n");
                }
                
                // Synchronize before input buffer swap
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
                compute_tile_tasklet(pid, NR_TASKLETS, 
                                   MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows, 
                                   MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols, 
                                   MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols,
                                   input_compute_buffer, 0);
            }

            if(pid == 0) {
                printf("[DPU %d] Result tile data: ", pid);
                for (int b = 0; b < 16; b++) {
                    printf("%02X ", result_wram[0][b]);
                }
                printf("\n");
            }

            barrier_wait(&my_barrier);
            
            // Now that we've completed this result tile, write it back and switch result buffer
            if (pid == 0) {
                __mram_ptr void *result_mram_addr = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.result_start_offset + DPU_MRAM_HEAP_POINTER + 
                    ((i * result_tiles_colwise + j) * result_tile_size_bytes));
                write_C_tile_to_mram(result_wram[0], result_mram_addr, result_tile_size_bytes);
                printf("[DPU %d] Wrote back completed result tile [%d,%d] to MRAM addr:%p from result buffer %d\n", 
                       pid, i, j, result_mram_addr, 0);
                // Switch to the other result buffer for the next result tile
                result_buffer = 1 - result_buffer;
            }
            
            barrier_wait(&my_barrier);
        }
    }

    if (pid == 0) {
        printf("[DPU %d] Matrix multiplication kernel completed successfully\n", pid);
    }

    // Wait for all operations to complete
    barrier_wait(&my_barrier);
    return 0;
}
