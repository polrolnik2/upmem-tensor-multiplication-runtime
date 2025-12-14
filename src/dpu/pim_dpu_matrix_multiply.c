#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <mram.h>
#include <alloc.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>
#include <handshake.h>
#include <sem.h>

#include "dpu_pim_matrix_multiply_kernel_arguments.h"

__host dpu_pim_matrix_multiply_kernel_arguments_t MATRIX_MULTIPLY_ARGUMENTS;
__dma_aligned void* aux;

static __dma_aligned int8_t* matrix1_wram[2];
static __dma_aligned int8_t* matrix2_wram[2];
static __dma_aligned int16_t* result_wram[2];

volatile int result_i[2] = {0, 0};
volatile int result_j[2] = {0, 0};

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

typedef enum {
    DMA, COMPUTE
} buffer_state_t;

volatile buffer_state_t input_buffer_states[2] = {DMA, DMA};
volatile buffer_state_t result_buffer_state[2] = {COMPUTE, COMPUTE};

SEMAPHORE_INIT(input_ready_1, 0);
SEMAPHORE_INIT(input_ready_2, 0);
SEMAPHORE_INIT(result_ready_1, 1);
SEMAPHORE_INIT(result_ready_2, 1);
SEMAPHORE_INIT(dma_request, 1);

MUTEX_INIT(status_mutex);

BARRIER_INIT(main_barrier, NR_TASKLETS);

BARRIER_INIT(compute_barrier, NR_TASKLETS-1);

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

void cdot(int8_t * A_buf, int8_t * B_buf, int16_t * C_buf,
                          uint32_t start_idx, uint32_t end_idx, 
                          uint32_t m_tile, uint32_t n_tile, uint32_t k_tile) {
    uint32_t a_addr, b_addr, c_addr;
    uint32_t a_row_start, b_col_start, c_row_start;
    uint32_t i_start = start_idx / n_tile;
    uint32_t j_start = start_idx % n_tile;
    uint32_t i_end = (end_idx - 1) / n_tile;
    uint32_t j_end = (end_idx - 1) % n_tile;
    a_row_start = i_start * MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;
    b_col_start = j_start * MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows;
    c_row_start = i_start * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
    c_addr = c_row_start + j_start;
    uint32_t i = i_start;
    uint32_t j = j_start;
    if (me() == 0) {
        return;
    }
    for (int c_idx = start_idx; c_idx < end_idx; ++c_idx) {
        int16_t sum = 0;
        a_addr = a_row_start;
        b_addr = b_col_start;
        for (int kk = 0; kk < k_tile; ++kk) {
            // Matrix B is column-major: B[kk][j] = B_buf[j * k_tile + kk]
            sum += (int16_t)A_buf[a_addr] * (int16_t)B_buf[b_addr];
            a_addr++;
            b_addr++;
        }
        C_buf[c_addr] = sum;
        c_addr++;
        j++;
        if (j == n_tile) {
            j = 0;
            i++;
        }
        if (i < m_tile && j == 0) {
            a_row_start += MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;
            c_row_start += MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
            b_col_start = 0;
            c_addr = c_row_start;
        } else if (j < n_tile) {
            b_col_start += MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows;
        }
    }
}

void cdot_accumulate(int8_t * A_buf, int8_t * B_buf, int16_t * C_buf,
                          uint32_t start_idx, uint32_t end_idx, 
                          uint32_t m_tile, uint32_t n_tile, uint32_t k_tile) {
    uint32_t a_addr, b_addr, c_addr;
    uint32_t a_row_start, b_col_start, c_row_start;
    uint32_t i_start = start_idx / n_tile;
    uint32_t j_start = start_idx % n_tile;
    uint32_t i_end = (end_idx - 1) / n_tile;
    uint32_t j_end = (end_idx - 1) % n_tile;
    a_row_start = i_start * MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;
    b_col_start = j_start * MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows;
    c_row_start = i_start * MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
    c_addr = c_row_start + j_start;
    uint32_t i = i_start;
    uint32_t j = j_start;
    if (me() == 0) {
        return;
    }
    for (int c_idx = start_idx; c_idx < end_idx; ++c_idx) {
        int16_t sum = 0;
        a_addr = a_row_start;
        b_addr = b_col_start;
        for (int kk = 0; kk < k_tile; ++kk) {
            // Matrix B is column-major: B[kk][j] = B_buf[j * k_tile + kk]
            sum += (int16_t)A_buf[a_addr] * (int16_t)B_buf[b_addr];
            a_addr++;
            b_addr++;
        }
        C_buf[c_addr] += sum;
        c_addr++;
        j++;
        if (j == n_tile) {
            j = 0;
            i++;
        }
        if (i < m_tile && j == 0) {
            a_row_start += MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;
            c_row_start += MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
            b_col_start = 0;
            c_addr = c_row_start;
        } else if (j < n_tile) {
            b_col_start += MATRIX_MULTIPLY_ARGUMENTS.matrix2_tile_rows;
        }
    }
}

void compute_tasklet(uint32_t input_buffer, uint32_t result_buffer, 
                     uint32_t start_idx, uint32_t end_idx,
                     uint32_t effective_m, uint32_t effective_n, uint32_t effective_k, 
                     uint32_t i, uint32_t j, bool first_iteration) {
    if (me() == 1) {
        switch (input_buffer) {
            case 0:
                sem_take(&input_ready_1);
                break;
            case 1:
                sem_take(&input_ready_2);
                break;
        }
        switch (result_buffer) {
            case 0:
                sem_take(&result_ready_1);
                break;
            case 1:
                sem_take(&result_ready_2);
                break;
        }
        if (first_iteration) {
            result_i[result_buffer] = i;
            result_j[result_buffer] = j;
        }
    }
    barrier_wait(&compute_barrier);
    if (first_iteration) {
        cdot(matrix1_wram[input_buffer], matrix2_wram[input_buffer],
             result_wram[result_buffer],
             start_idx, end_idx,
             effective_m, effective_n, effective_k);
    } else {
        cdot_accumulate(matrix1_wram[input_buffer], matrix2_wram[input_buffer],
                        result_wram[result_buffer],
                        start_idx, end_idx,
                        effective_m, effective_n, effective_k);
    }
    #ifdef DEBUG
    if (me() == 1) {
        printf("[DPU %d] Computed tile C(%d, %d) for indices %d to %d using buffer %d\n", me(), 
            result_i[result_buffer], result_j[result_buffer],
            start_idx, end_idx, result_buffer);
    }
    #endif
    if (me() == 1) {
        mutex_lock(status_mutex);
        input_buffer_states[input_buffer] = DMA;
        mutex_unlock(status_mutex);
        sem_give(&dma_request);
        switch (result_buffer) {
            case 0:
                sem_give(&result_ready_1);
                break;
            case 1:
                sem_give(&result_ready_2);
                break;
        }
    }
}

bool dma_tasklet(uint32_t next_i, uint32_t next_j, uint32_t next_k,
                 int input_buffer_index, int output_buffer_index) {
    bool loaded_next = false;
    sem_take(&dma_request);
    if (input_buffer_states[input_buffer_index] == DMA) {
        __mram_ptr void *mram_addr_A = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix1_start_offset + DPU_MRAM_HEAP_POINTER + 
            (next_i * matrix1_tiles_colwise + next_k) * matrix1_tile_size_bytes);
        load_A_tile_from_mram(mram_addr_A, matrix1_wram[input_buffer_index], 
                                matrix1_tile_size_bytes);
        __mram_ptr void *mram_addr_B = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.matrix2_start_offset + DPU_MRAM_HEAP_POINTER + 
            (next_j * matrix2_tiles_rowwise + next_k) * matrix2_tile_size_bytes);
        load_B_tile_from_mram(mram_addr_B, matrix2_wram[input_buffer_index], 
                                matrix2_tile_size_bytes);
        mutex_lock(status_mutex);
        input_buffer_states[input_buffer_index] = COMPUTE;
        loaded_next = true;
        mutex_unlock(status_mutex);
        switch (input_buffer_index) {
            case 0:
                sem_give(&input_ready_1);
                break;
            case 1:
                sem_give(&input_ready_2);
                break;
        }
        #ifdef DEBUG
        printf("[DPU %d] Loaded tiles A(%d, %d) and B(%d, %d) into WRAM\n", me(), next_i, next_k, next_j, next_k);
        #endif
    } 
    if (result_buffer_state[output_buffer_index] == DMA) {
        __mram_ptr void *result_mram_addr = (__mram_ptr void *)(MATRIX_MULTIPLY_ARGUMENTS.result_start_offset + DPU_MRAM_HEAP_POINTER + 
            ((result_i[output_buffer_index] * result_tiles_colwise + result_j[output_buffer_index]) * result_tile_size_bytes));
        write_C_tile_to_mram(result_wram[output_buffer_index], result_mram_addr, result_tile_size_bytes);
        mutex_lock(status_mutex);
        result_buffer_state[output_buffer_index] = COMPUTE;
        mutex_unlock(status_mutex);
        switch (output_buffer_index) {
            case 0:
                sem_give(&result_ready_1);
                break;
            case 1:
                sem_give(&result_ready_2);
                break;
        }
        #ifdef DEBUG
        printf("[DPU %d] Completed writing result tile C(%d, %d) to MRAM\n", me(), result_i[output_buffer_index], result_j[output_buffer_index]);
        #endif
    } 
    return loaded_next;
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

    barrier_wait(&main_barrier);

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

    barrier_wait(&main_barrier);

    int input_compute_buffer = 0;
    int result_compute_buffer = 0; 
    int input_dma_buffer = 0;
    int result_dma_buffer = 0;
    bool first_iteration = true;
    
    if (pid != 0) {
        for (int i = 0; i < result_tiles_rowwise; i++) {
            for (int j = 0; j < result_tiles_colwise; j++) {
                uint32_t effective_m = (i == result_tiles_rowwise - 1) ? last_row_tile_m : MATRIX_MULTIPLY_ARGUMENTS.result_tile_rows;
                uint32_t effective_n = (j == result_tiles_colwise - 1) ? last_col_tile_n : MATRIX_MULTIPLY_ARGUMENTS.result_tile_cols;
                uint32_t elements_for_this_tasklet = effective_m * effective_n;
                uint32_t tiles_per_tasklet = (elements_for_this_tasklet + (NR_TASKLETS - 2)) / (NR_TASKLETS - 1);
                start_idx = (pid - 1) * tiles_per_tasklet;
                end_idx = start_idx + tiles_per_tasklet;
                if (end_idx > elements_for_this_tasklet) {
                    end_idx = elements_for_this_tasklet;
                }
                first_iteration = true;
                for (int k = 0; k < matrix1_tiles_colwise; k++) {
                    uint32_t effective_k = (k == matrix1_tiles_colwise - 1) ? last_k_tile_k : MATRIX_MULTIPLY_ARGUMENTS.matrix1_tile_cols;
                    compute_tasklet(input_compute_buffer, result_compute_buffer,
                                     start_idx, end_idx,
                                     effective_m, effective_n, effective_k, 
                                     i, j, first_iteration);
                    input_compute_buffer = 1 - input_compute_buffer;
                    first_iteration = false;
                    barrier_wait(&compute_barrier);
                }
                if (me() == 1) {
                    mutex_lock(status_mutex);
                    result_buffer_state[result_compute_buffer] = DMA;
                    mutex_unlock(status_mutex);
                    switch (result_compute_buffer) {
                        case 0:
                            sem_take(&result_ready_1);
                            break;
                        case 1:
                            sem_take(&result_ready_2);
                            break;
                    }
                }
                result_compute_buffer = 1 - result_compute_buffer;
                barrier_wait(&compute_barrier);
            }
        }
    } else {
        for (int i = 0; i < result_tiles_rowwise; i++) {
            for (int j = 0; j < result_tiles_colwise; j++) {
                for (int k = 0; k < matrix1_tiles_colwise; k++) {
                    bool result = dma_tasklet(i, j, k, input_dma_buffer, result_dma_buffer);
                    if (!result) {
                        k--;
                        sem_give(&dma_request);
                        continue;
                    }
                    input_dma_buffer = 1 - input_dma_buffer;
                    result_dma_buffer = 1 - result_dma_buffer;
                }
            }
        }
    }

    barrier_wait(&main_barrier);

    if (pid == 0) {
        sem_give(&dma_request);
        dma_tasklet(result_tiles_rowwise-1, result_tiles_colwise-1, matrix1_tiles_colwise-1, 0, 0);
        sem_give(&dma_request);
        dma_tasklet(result_tiles_rowwise-1, result_tiles_colwise-1, matrix1_tiles_colwise-1, 0, 1);
    }
#ifdef DEBUG
    if (pid == 0) {
        printf("[DPU %d] Matrix multiplication kernel completed successfully\n", pid);
    }
#endif

    // Wait for all operations to complete
    barrier_wait(&main_barrier);

    return 0;
}
