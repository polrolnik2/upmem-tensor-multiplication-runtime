#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_assertions.h"

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"

#include "host_multiply_matrices.h"
#include "dpu_multiply_matrices.h"

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

int main() {
    int result = test_pim_transfer_misaligned_matrix_multiplication();
    if (result == 0) {
        printf("[PASS] test_pim_transfer_misaligned_matrix_multiplication passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_pim_transfer_misaligned_matrix_multiplication failed!\n");
        return 1;
    }
}
