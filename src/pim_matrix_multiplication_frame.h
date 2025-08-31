#ifndef __PIM_MATRIX_MULTIPLICATION_FRAME_H___
#define __PIM_MATRIX_MULTIPLICATION_FRAME_H___

#include <dpu.h>

typedef struct {
    uint32_t num_work_groups;
    uint32_t work_group_size;
    uint32_t num_dpus;
    uint32_t matrix1_rows;
    uint32_t matrix1_cols;
    uint32_t matrix2_rows;
    uint32_t matrix2_cols;
    uint32_t result_rows;
    uint32_t result_cols;
    uint32_t matrix1_type_size;       ///< Size of first matrix elements (1 for
    uint32_t matrix2_type_size;       ///< Size of second matrix elements (1 for
    uint32_t result_type_size;        ///< Size of result matrix elements (2 for
    uint32_t matrix1_start_offset;    ///< MRAM offset for first matrix
    uint32_t matrix2_start_offset;    ///< MRAM offset for second matrix
    uint32_t result_start_offset;     ///< MRAM offset for result matrix
    uint32_t mem_frame_end;           ///< MRAM offset for end of memory frame
    
    // Tile dimensions for DPU kernel
    uint32_t matrix1_tile_rows;       ///< Number of rows in matrix1 tiles
    uint32_t matrix1_tile_cols;       ///< Number of columns in matrix1 tiles
    uint32_t matrix2_tile_rows;       ///< Number of rows in matrix2 tiles
    uint32_t matrix2_tile_cols;       ///< Number of columns in matrix2 tiles
    uint32_t result_tile_rows;        ///< Number of rows in result tiles
    uint32_t result_tile_cols;        ///< Number of columns in result tiles
    uint32_t wram_input_tile_size;    ///< Size of input tile in WRAM
    
    bool result_valid;              ///< Flag indicating if result is valid
    struct dpu_set_t dpu_set; ///< DPU set for execution
} pim_matrix_multiplication_frame_t;

/**
 * @brief Create a new PIM matrix multiplication frame - a structure for managing
 *        the state and data of a matrix multiplication operation on a PIM architecture.
 * @details This function allocates memory for the frame, initializes how the matrices should be split to optimize the memory utilization.
 * @param num_dpus Number of DPUs to use.
 * @param dpu_offset Offset for DPU memory.
 * @param matrix1_rows Number of rows in the first matrix.
 * @param matrix1_cols Number of columns in the first matrix.
 * @param matrix2_rows Number of rows in the second matrix.
 * @param matrix2_cols Number of columns in the second matrix.
 * @param result_rows Number of rows in the result matrix.
 * @param result_cols Number of columns in the result matrix.
 * @param matrix1_type_size Size of each element in the first matrix.
 * @param matrix2_type_size Size of each element in the second matrix.
 * @param result_type_size Size of each element in the result matrix.
 * @return Pointer to the new frame, or NULL on failure.
 */
pim_matrix_multiplication_frame_t* create_pim_matrix_multiplication_frame(uint32_t num_dpus, uint32_t dpu_offset, 
                                                                        uint32_t matrix1_rows, uint32_t matrix1_cols,
                                                                        uint32_t matrix2_rows, uint32_t matrix2_cols,
                                                                        uint32_t result_rows, uint32_t result_cols,
                                                                        uint32_t matrix1_type_size, uint32_t matrix2_type_size, uint32_t result_type_size);

/**
 * @brief Load the first matrix (Left side of the multiplication) into the frame.
 * @param frame Pointer to the PIM matrix multiplication frame.
 * @param matrix Pointer to the first matrix.
 */
void pim_matrix_multiplication_frame_load_first_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix);

/**
 * @brief Load the second matrix (Right side of the multiplication) into the frame.
 * @param frame Pointer to the PIM matrix multiplication frame.
 * @param matrix Pointer to the second matrix.
 */
void pim_matrix_multiplication_frame_load_second_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix);

/**
 * @brief Execute the matrix multiplication on the PIM architecture.
 * @param frame Pointer to the PIM matrix multiplication frame.
 */
void pim_matrix_multiplication_frame_execute(pim_matrix_multiplication_frame_t* frame);

/**
 * @brief Get the result of the matrix multiplication from the frame.
 * @details This function retrieves the result matrix after execution. The result is only valid after calling `pim_matrix_multiplication_frame_execute`. The valid flag resets when a new matrix is loaded.
 *          The result is a matrix containing the product of the two matrices loaded into the frame.
 * @param frame Pointer to the PIM matrix multiplication frame.
 * @return Pointer to the resulting matrix, or NULL on failure.
 */
Matrix * pim_matrix_multiplication_frame_get_result(pim_matrix_multiplication_frame_t* frame);

#endif // __PIM_MATRIX_MULTIPLICATION_FRAME_H___