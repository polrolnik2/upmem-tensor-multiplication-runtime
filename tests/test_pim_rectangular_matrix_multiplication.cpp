// Auto-migrated from test_pim_rectangular_matrix_multiplication.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"
#include "host_multiply_matrices.h"
#include "dpu_multiply_matrices.h"

TEST(MatrixTest, pim_rectangular_matrix_multiplication) {
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
    EXPECT_TRUE(matrix1 != NULL) << "Matrix 1 creation failed";
    EXPECT_TRUE(matrix2 != NULL) << "Matrix 2 creation failed";
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2, 4);
    EXPECT_TRUE(result != NULL) << "Result matrix should not be NULL";
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    EXPECT_TRUE(expected_result != NULL) << "Expected result matrix should not be NULL";
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
    EXPECT_TRUE(result != NULL) << "Result matrix should not be NULL";
    EXPECT_TRUE(matrix_compare(result, expected_result)) << "Result matrix should match expected result";
    matrix_free(matrix1);
    matrix_free(matrix2);
}
