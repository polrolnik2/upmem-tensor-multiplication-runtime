#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_assertions.h"

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"

Matrix* host_multiply_matrices(const Matrix* matrix1, const Matrix* matrix2) {
    if (!matrix1 || !matrix2 || matrix1->cols != matrix2->rows) return NULL;
    uint16_t * result_data_row_major = malloc(matrix1->rows * matrix2->cols * sizeof(uint16_t));
    if (!result_data_row_major) return NULL;
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix2->cols; j++) {
            uint16_t sum = 0;
            for (int k = 0; k < matrix1->cols; k++) {
                uint8_t val1, val2;
                matrix_get(matrix1, i, k, &val1);
                matrix_get(matrix2, k, j, &val2);
                sum += val1 * val2;
            }
            result_data_row_major[i*matrix2->cols + j] = sum;
        }
    }
    Matrix* result = matrix_create_from_row_major_array(matrix1->rows, matrix2->cols, result_data_row_major, sizeof(uint16_t));
    if (!result) {
        free(result_data_row_major);
        return NULL;
    }
    free(result_data_row_major);
    return result;
}

Matrix*  dpu_multiply_matrices(Matrix* matrix1, Matrix* matrix2, uint32_t num_dpus) {
    // Create a sample matrix multiplication frame
    pim_matrix_multiplication_frame_t* frame = create_pim_matrix_multiplication_frame(num_dpus, 0, matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols, matrix1->rows, matrix2->cols,
                                                                                      sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t));
    if (!frame) {
        fprintf(stderr, "Frame creation failed");
        return NULL;
    }
    pim_matrix_multiplication_frame_load_first_matrix(frame, matrix1);
    pim_matrix_multiplication_frame_load_second_matrix(frame, matrix2);
    pim_matrix_multiplication_frame_execute(frame);
    Matrix* result = pim_matrix_multiplication_frame_get_result(frame);
    if (!result) {
        fprintf(stderr, "Result retrieval failed");
        return NULL;
    }
    return result;
}

int test_pim_identity_square_matrix_multiplication() {
    printf("Running test_pim_identity_square_matrix_multiplication...\n");
    // Create two sample matrices 16x16
    uint16_t rows = 16, cols = 16;
    uint8_t data1[16*16], data2[16*16];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = 1; // Sample data for matrix 1
            if (i == j) {
                data2[i*cols + j] = 1; // Sample data for matrix 2
            } else {
                data2[i*cols + j] = 0; // Sample data for matrix 2
            }
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_square_matrix_multiplication() {
    printf("Running test_pim_square_matrix_multiplication...\n");
    // Create two sample matrices 16x16
    uint16_t rows = 16, cols = 16;
    uint8_t data1[16*16], data2[16*16];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = i+j;
            data2[i*cols + j] = i+j; 
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_transfer_misaligned_matrix_multiplication() {
    printf("Running test_pim_frame_misaligned_matrix_multiplication...\n");
    // Create two sample matrices 12x12
    uint16_t rows = 12, cols = 12;
    uint8_t data1[12*12], data2[12*12];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = i+j;
            data2[i*cols + j] = i+j; 
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_frame_misaligned_matrix_multiplication() {
    printf("Running test_pim_frame_misaligned_matrix_multiplication...\n");
    // Create two sample matrices 15x15
    uint16_t rows = 15, cols = 15;
    uint8_t data1[15*15], data2[15*15];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = i+j;
            data2[i*cols + j] = i+j; 
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_rectangular_matrix_multiplication() {
    printf("Running test_pim_rectangular_matrix_multiplication...\n");
    // Create two sample matrices 12x15 and 15x8
    uint16_t rows1 = 12, cols1 = 15;
    uint16_t rows2 = 15, cols2 = 8;
    uint8_t data1[12*15], data2[15*8];
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            data1[i*cols1 + j] = i+j;
        }
    }
    for (int i = 0; i < rows2; i++) {
        for (int j = 0; j < cols2; j++) {
            data2[i*cols2 + j] = i+j;
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows1, cols1, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows2, cols2, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_square_prime_number_of_dpus() {
    printf("Running test_pim_square_matrix_multiplication...\n");
    // Create two sample matrices 16x16
    uint16_t rows = 16, cols = 16;
    uint8_t data1[16*16], data2[16*16];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = i+j;
            data2[i*cols + j] = i+j; 
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 3);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_square_multi_tile_identity() {
    printf("Running test_pim_square_multi_tile_identity...\n");
    // Create two sample matrices 128x128
    uint16_t rows = 128, cols = 128;
    uint8_t data1[128*128], data2[128*128];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = i == j ? 1 : 0; // Sample data for matrix 1
            data2[i*cols + j] = 1; // Sample data for matrix 2
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_square_multi_tile() {
    printf("Running test_pim_square_multi_tile...\n");
    // Create two sample matrices 128x128
    uint16_t rows = 128, cols = 128;
    uint8_t data1[128*128], data2[128*128];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = (i+j) % 256; // Sample data for matrix 1
            data2[i*cols + j] = (i+j) % 256; // Sample data for matrix 2
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_multiple_tiles_per_row() {
    printf("Running test_pim_square_multi_tile...\n");
    // Create two sample matrices 4096x8
    uint16_t rows1 = 8, cols1 = 4096;
    uint16_t rows2 = 4096, cols2 = 8;
    uint8_t data1[8*4096], data2[4096*8];
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            data1[i*cols1 + j] = (i+j) % 256; // Sample data for matrix 1
            data2[i*cols2 + j] = (i+j) % 256; // Sample data for matrix 2
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows1, cols1, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows2, cols2, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    if (!matrix_compare(result, expected_result)) {
        printf(": %dx%d, Result dimensions: %dx%d\n",
            expected_result->rows, expected_result->cols, result->rows, result->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result->rows; i++) {
            for (int j = 0; j < expected_result->cols; j++) {
                int16_t val;
                matrix_get(expected_result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                int16_t val;
                matrix_get(result, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int main() {
    uint32_t fails = 0;
    printf("Running PIM Matrix Multiplication Frame Unittests...\n");
    fails += test_pim_identity_square_matrix_multiplication();
    fails += test_pim_square_matrix_multiplication();
    fails += test_pim_transfer_misaligned_matrix_multiplication();
    fails += test_pim_frame_misaligned_matrix_multiplication();
    fails += test_pim_rectangular_matrix_multiplication();
    fails += test_pim_square_prime_number_of_dpus();
    fails += test_pim_square_multi_tile();
    fails += test_pim_square_multi_tile_identity();
    fails += test_pim_multiple_tiles_per_row();
    if (fails == 0) {
        printf("[PASS] All PIM matrix tests passed!\n");
        return 0;
    } else {
        printf("%d PIM matrix tests failed.\n", fails);
        return 1;
    }
}