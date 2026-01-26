#ifndef __PIM_MATRIX_MULTIPLICATION_FRAME_H___
#define __PIM_MATRIX_MULTIPLICATION_FRAME_H___

#include <dpu.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * @param dpu_binary_path Path to the DPU kernel binary to load.
 * @return Pointer to the new frame, or NULL on failure.
 */
pim_matrix_multiplication_frame_t* create_pim_matrix_multiplication_frame_binary(uint32_t num_dpus, uint32_t dpu_offset, 
                                                                        uint32_t matrix1_rows, uint32_t matrix1_cols,
                                                                        uint32_t matrix2_rows, uint32_t matrix2_cols,
                                                                        uint32_t result_rows, uint32_t result_cols,
                                                                        uint32_t matrix1_type_size, uint32_t matrix2_type_size, uint32_t result_type_size,
                                                                        const char* dpu_binary_path);
                                                                        
/**
 * @brief Create a new PIM matrix multiplication frame - a structure for managing
 *        the state and data of a matrix multiplication operation on a PIM architecture.
 * @details This function allocates memory for the frame, initializes how the matrices should be split to optimize the memory utilization. 
 * @details It uses a default DPU binary path from the parameter DPU_MATMUL_DPU_BINARY_PATH defined at compile time. This flag is defined when building the project and automatically set by the CMake flow.
 * @details The preffered way to use this library is through the CMake build system integration, which ensures that the DPU binary path is correctly set.
 * @details If this flag is not defined or the CMake build system is not used, this function will not be available. In that case, use `create_pim_matrix_multiplication_frame_binary` to explicitly provide the DPU binary path.
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
pim_matrix_multiplication_frame_t* create_pim_matrix_multiplication_frame(
    uint32_t num_dpus, uint32_t dpu_offset, 
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
    * @brief Execute the matrix multiplication on the PIM architecture with inline data transfers.
    * @details This function allows loading matrices and retrieving results in a single call, optimizing data transfers.
    * @param execution_frame Pointer to the PIM matrix multiplication frame for execution.
    * @param transfer_frame Pointer to the PIM matrix multiplication frame for data transfers.
    * @param load_matrices Flag indicating whether to load the matrices.
    * @param first_matrix Pointer to the first matrix (if loading).
    * @param second_matrix Pointer to the second matrix (if loading).
    * @param retrieve_result Flag indicating whether to retrieve the result matrix.
    * @param result_matrix Pointer to store the resulting matrix (if retrieving).
*/
void pim_matrix_multiplication_frame_execute_and_transfer_inline(pim_matrix_multiplication_frame_t* execution_frame, pim_matrix_multiplication_frame_t* transfer_frame, bool load_matrices, Matrix * first_matrix, Matrix * second_matrix, bool retrieve_result, Matrix ** result_matrix);

/**
    * @brief Macro to execute matrix multiplication with loading and retrieving in one call.
    * @param execution_frame Pointer to the PIM matrix multiplication frame for execution.
    * @param transfer_frame Pointer to the PIM matrix multiplication frame for data transfers.
    * @param first_matrix Pointer to the first matrix to load.
    * @param second_matrix Pointer to the second matrix to load.
    * @param result_matrix Pointer to store the resulting matrix.
*/
#define pim_matrix_multiplication_frame_execute_load_retrieve(execution_frame, transfer_frame, first_matrix, second_matrix, result_matrix) \
    pim_matrix_multiplication_frame_execute_and_transfer_inline(execution_frame, transfer_frame, true, first_matrix, second_matrix, true, result_matrix)

/**
    * @brief Macro to execute matrix multiplication with loading only.
    * @param execution_frame Pointer to the PIM matrix multiplication frame for execution.
    * @param transfer_frame Pointer to the PIM matrix multiplication frame for data transfers.
    * @param first_matrix Pointer to the first matrix to load.
    * @param second_matrix Pointer to the second matrix to load. 

*/
#define pim_matrix_multiplication_frame_execute_load(execution_frame, transfer_frame, first_matrix, second_matrix) \
    pim_matrix_multiplication_frame_execute_and_transfer_inline(execution_frame, transfer_frame, true, first_matrix, second_matrix, false, NULL)

/**
    * @brief Macro to execute matrix multiplication with retrieving only.
    * @param execution_frame Pointer to the PIM matrix multiplication frame for execution.
    * @param transfer_frame Pointer to the PIM matrix multiplication frame for data transfers.
    * @param result_matrix Pointer to store the resulting matrix.

*/
#define pim_matrix_multiplication_frame_execute_retrieve(execution_frame, transfer_frame, result_matrix) \
    pim_matrix_multiplication_frame_execute_and_transfer_inline(execution_frame, transfer_frame, false, NULL, NULL, true, result_matrix)

/**
 * @brief Get the result of the matrix multiplication from the frame.
 * @details This function retrieves the result matrix after execution. The result is only valid after calling `pim_matrix_multiplication_frame_execute`. The valid flag resets when a new matrix is loaded.
 *          The result is a matrix containing the product of the two matrices loaded into the frame.
 * @param frame Pointer to the PIM matrix multiplication frame.
 * @return Pointer to the resulting matrix, or NULL on failure.
 */
Matrix * pim_matrix_multiplication_frame_get_result(pim_matrix_multiplication_frame_t* frame);

/**
 * @brief Free the resources associated with the PIM matrix multiplication frame.
 * @param frame Pointer to the PIM matrix multiplication frame to free.
 */
void pim_matrix_multiplication_frame_free(pim_matrix_multiplication_frame_t* frame);

/**
 * @brief Synchronize the frame, ensuring all operations are complete.
 * @param frame Pointer to the PIM matrix multiplication frame.
 */
void pim_matrix_multiplication_frame_sync(pim_matrix_multiplication_frame_t* frame);

#ifdef __cplusplus
}
#endif


#endif // __PIM_MATRIX_MULTIPLICATION_FRAME_H___