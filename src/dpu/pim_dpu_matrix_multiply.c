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

// Per-tasklet buffers for matrix tiles
static __dma_aligned int8_t* matrix1_wram[NR_TASKLETS];
static __dma_aligned int8_t* matrix2_wram[NR_TASKLETS];
static __dma_aligned int16_t* result_wram[NR_TASKLETS];

static uint32_t matrix1_tile_size_bytes;
static uint32_t matrix2_tile_size_bytes;
static uint32_t result_tile_size_bytes;

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

void compute_tile(int8_t* A_buf, int8_t* B_buf, int16_t* C_buf,
                  uint32_t m_tile, uint32_t n_tile, uint32_t k_tile) {
    for (uint32_t i = 0; i < m_tile; ++i) {
        for (uint32_t j = 0; j < n_tile; ++j) {
            int16_t sum = 0;
            for (uint32_t kk = 0; kk < k_tile; ++kk) {
                // Matrix B is column-major: B[kk][j] = B_buf[j * k_tile + kk]
                sum += (int16_t)A_buf[i * k_tile + kk] * (int16_t)B_buf[j * k_tile + kk];
            }
            C_buf[i * n_tile + j] += sum;
        }
    }
}

/**
 * @brief Main DPU function for matrix multiplication with per-tasklet tile management
 * 
 * This function implements independent per-tasklet processing where:
 * - Each tasklet is assigned a subset of result tiles
 * - Each tasklet independently loads, computes, and stores its tiles
 * - No dedicated memory management tasklet - all tasklets work in parallel
 */
int main() {
    int pid = me();
    
    // Initialize memory heap and allocate per-tasklet buffers
    if (pid == 0) {
        mem_reset(); // Reset the heap

        matrix1_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.matrix1_type_size;
        matrix2_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.matrix2_type_size;
        result_tile_size_bytes = MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols * MATRIX_MULTIPLY_ARGUMENTS.result_type_size;

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

        if (MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols != MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows) {
            printf("[DPU %d] ERROR: Matrix tile dimensions mismatch: A_cols=%d, B_rows=%d\n", 
                pid, MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols, MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows);
            return -1;
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
        printf("[DPU %d] Matrix dimensions: A=%dx%d, B=%dx%d, C=%dx%d\n", 
            pid, MATRIX_MULTIPLY_ARGUMENTS.matrix1_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix1_cols,
            MATRIX_MULTIPLY_ARGUMENTS.matrix2_rows, MATRIX_MULTIPLY_ARGUMENTS.matrix2_cols,
            MATRIX_MULTIPLY_ARGUMENTS.result_rows, MATRIX_MULTIPLY_ARGUMENTS.result_cols);
        printf("[DPU %d] Result tiles: %dx%d, Starting per-tasklet tile processing\n", 
            pid, result_tiles_rowwise, result_tiles_colwise);
        #endif

        result_elements = MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
    }

    barrier_wait(&my_barrier);

    // Each tasklet allocates its own buffers
    matrix1_wram[pid] = (int8_t*)mem_alloc(matrix1_tile_size_bytes);
    matrix2_wram[pid] = (int8_t*)mem_alloc(matrix2_tile_size_bytes);
    result_wram[pid] = (int16_t*)mem_alloc(result_tile_size_bytes);
    
    if (!matrix1_wram[pid] || !matrix2_wram[pid] || !result_wram[pid]) {
        printf("[Tasklet %d] ERROR: Failed to allocate memory for matrix buffers\n", pid);
        return -1;
    }

    barrier_wait(&my_barrier);

    // Calculate total number of result tiles and partition among tasklets
    uint32_t total_result_tiles = result_tiles_rowwise * result_tiles_colwise;
    uint32_t tiles_per_tasklet = (total_result_tiles + NR_TASKLETS - 1) / NR_TASKLETS;
    uint32_t my_tile_start = pid * tiles_per_tasklet;
    uint32_t my_tile_end = (my_tile_start + tiles_per_tasklet) < total_result_tiles ? 
                           (my_tile_start + tiles_per_tasklet) : total_result_tiles;

    #ifdef DEBUG
    printf("[Tasklet %d] Assigned tiles %d to %d (total: %d)\n", 
           pid, my_tile_start, my_tile_end - 1, my_tile_end - my_tile_start);
    #endif

    // Each tasklet processes its assigned result tiles independently
    for (uint32_t tile_idx = my_tile_start; tile_idx < my_tile_end; tile_idx++) {
        // Convert linear tile index to 2D coordinates
        uint32_t i = tile_idx / result_tiles_colwise;  // row index
        uint32_t j = tile_idx % result_tiles_colwise;  // col index

        #ifdef DEBUG
        printf("[Tasklet %d] Processing result tile [%d,%d] (linear index %d)\n", pid, i, j, tile_idx);
        #endif

        // Clear result buffer for this tile
        for (uint32_t idx = 0; idx < result_elements; idx++) {
            result_wram[pid][idx] = 0;
        }

        // Accumulate across all K iterations for this result tile
        for (uint32_t k = 0; k < matrix1_tiles_colwise; k++) {
            #ifdef DEBUG
            printf("[Tasklet %d] K iteration %d/%d for result tile [%d,%d]\n", 
                   pid, k, matrix1_tiles_colwise - 1, i, j);
            #endif

            // Load A tile
            __mram_ptr void *mram_addr_A = (__mram_ptr void *)(
                MATRIX_MULTIPLY_ARGUMENTS.matrix1_start_offset + DPU_MRAM_HEAP_POINTER + 
                (i * matrix1_tiles_colwise + k) * matrix1_tile_size_bytes);
            load_A_tile_from_mram(mram_addr_A, matrix1_wram[pid], matrix1_tile_size_bytes);

            // Load B tile
            __mram_ptr void *mram_addr_B = (__mram_ptr void *)(
                MATRIX_MULTIPLY_ARGUMENTS.matrix2_start_offset + DPU_MRAM_HEAP_POINTER + 
                (j * matrix2_tiles_rowwise + k) * matrix2_tile_size_bytes);
            load_B_tile_from_mram(mram_addr_B, matrix2_wram[pid], matrix2_tile_size_bytes);

            // Compute: C[i,j] += A[i,k] * B[k,j]
            compute_tile(matrix1_wram[pid], matrix2_wram[pid], result_wram[pid],
                        MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows,
                        MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols,
                        MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols);
        }

        // Write completed result tile back to MRAM
        __mram_ptr void *result_mram_addr = (__mram_ptr void *)(
            MATRIX_MULTIPLY_ARGUMENTS.result_start_offset + DPU_MRAM_HEAP_POINTER + 
            ((i * result_tiles_colwise + j) * result_tile_size_bytes));
        write_C_tile_to_mram(result_wram[pid], result_mram_addr, result_tile_size_bytes);

        #ifdef DEBUG
        printf("[Tasklet %d] Completed and wrote back result tile [%d,%d]\n", pid, i, j);
        #endif
    }

    #ifdef DEBUG
    printf("[Tasklet %d] Finished all assigned tiles\n", pid);
    #endif

    // Wait for all tasklets to complete
    barrier_wait(&my_barrier);

    #ifdef DEBUG
    if (pid == 0) {
        printf("[DPU %d] Matrix multiplication kernel completed successfully\n", pid);
    }
    #endif

    return 0;
}
