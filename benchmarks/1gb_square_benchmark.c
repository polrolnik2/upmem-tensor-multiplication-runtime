#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_assertions.h"

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"

#include "host_multiply_matrices.h"
#include "dpu_multiply_matrices.h"

int main() {
    printf("Running test_pim_rectangular_matrix_multiplication...\n");
    // Create two sample matrices 131072x8192 and 8192x131072
    int rows1 = 131072, cols1 = 8192;
    int rows2 = 8192, cols2 = 131072;
    uint8_t *data1 = (uint8_t*)malloc(rows1 * cols1 * sizeof(uint8_t));
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            data1[i*cols1 + j] = rand() % 4;
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows1, cols1, (void*)data1, sizeof(uint8_t));
    free(data1);
    uint8_t *data2 = (uint8_t*)malloc(rows2 * cols2 * sizeof(uint8_t));
    for (int i = 0; i < rows2; i++) {
        for (int j = 0; j < cols2; j++) {
            data2[i*cols2 + j] = rand() % 4;
        }
    }
    Matrix* matrix2 = matrix_create_from_row_major_array(rows2, cols2, (void*)data2, sizeof(uint8_t));
    free(data2);
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 2500);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    ASSERT_TRUE(expected_result != NULL, "Expected result matrix should not be NULL");
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    matrix_free(result);
    matrix_free(expected_result);
    return 0;
}