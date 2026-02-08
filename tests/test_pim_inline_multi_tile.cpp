// Auto-migrated from test_pim_multiple_tiles_per_row.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"
#include "host_multiply_matrices.h"
#include "dpu_multiply_matrices.h"

TEST(MatrixTest, pim_multiple_tiles_per_row) {
    // Create two sample matrices 4096x8
    uint16_t rows1 = 128, cols1 = 256;
    uint16_t rows2 = 256, cols2 = 128;
    uint8_t data1[128*256], data2[256*128];
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            data1[i*cols1 + j] = (i+j) % 256; // Sample data for matrix 1
            data2[j*cols2 + i] = (255 - (i+j)) % 256; // Sample data for matrix 2
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows1, cols1, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows2, cols2, (void*)data2, sizeof(uint8_t));
    EXPECT_TRUE(matrix1 != NULL) << "Matrix 1 creation failed";
    EXPECT_TRUE(matrix2 != NULL) << "Matrix 2 creation failed";
    pim_matrix_multiplication_frame_t* frame_1 = create_pim_matrix_multiplication_frame(1, 0, matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols, matrix1->rows, matrix2->cols,
                                                                                      sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t));
    pim_matrix_multiplication_frame_t* frame_2 = create_pim_matrix_multiplication_frame(1, frame_1->mem_frame_end, matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols, matrix1->rows, matrix2->cols,
                                                                                      sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t));

    pim_matrix_multiplication_frame_load_first_matrix(frame_1, matrix1);
    pim_matrix_multiplication_frame_load_second_matrix(frame_1, matrix2);
    pim_matrix_multiplication_frame_execute_load(frame_1, frame_2, matrix1, matrix2);
    Matrix * result_1;
    pim_matrix_multiplication_frame_execute_load_retrieve(frame_2, frame_1, matrix2, matrix1, &result_1);
    EXPECT_TRUE(result_1 != NULL) << "Result matrix 1 should not be NULL";
    Matrix * result_2;
    pim_matrix_multiplication_frame_execute_retrieve(frame_1, frame_2, &result_2);
    EXPECT_TRUE(result_2 != NULL) << "Result matrix 2 should not be NULL";
    Matrix* result_3 = pim_matrix_multiplication_frame_get_result(frame_1);
    EXPECT_TRUE(result_3 != NULL) << "Result matrix 3 should not be NULL";
    Matrix* expected_result_1_3 = host_multiply_matrices(matrix1, matrix2);
    Matrix* expected_result_2 = host_multiply_matrices(matrix2, matrix1);
    EXPECT_TRUE(expected_result_1_3 != NULL) << "Expected result matrix 1_3 should not be NULL";
    if (!matrix_compare(result_1, expected_result_1_3)) {
        printf("Expected dimensions: %dx%d, Result dimensions: %dx%d\n",
            expected_result_1_3->rows, expected_result_1_3->cols, result_1->rows, result_1->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result_1_3->rows; i++) {
            for (int j = 0; j < expected_result_1_3->cols; j++) {
                int16_t val;
                matrix_get(expected_result_1_3, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result_1->rows; i++) {
            for (int j = 0; j < result_1->cols; j++) {
                int16_t val;
                matrix_get(result_1, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    if (!matrix_compare(result_2, expected_result_2)) {
        printf("Expected dimensions: %dx%d, Result dimensions: %dx%d\n",
            expected_result_2->rows, expected_result_2->cols, result_2->rows, result_2->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result_2->rows; i++) {
            for (int j = 0; j < expected_result_2->cols; j++) {
                int16_t val;
                matrix_get(expected_result_2, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result_2->rows; i++) {
            for (int j = 0; j < result_2->cols; j++) {
                int16_t val;
                matrix_get(result_2, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    if (!matrix_compare(result_3, expected_result_1_3)) {
        printf("Expected dimensions: %dx%d, Result dimensions: %dx%d\n",
            expected_result_1_3->rows, expected_result_1_3->cols, result_3->rows, result_3->cols);
        printf("Expected matrix:\n");
        for (int i = 0; i < expected_result_1_3->rows; i++) {
            for (int j = 0; j < expected_result_1_3->cols; j++) {
                int16_t val;
                matrix_get(expected_result_1_3, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
        printf("Result matrix:\n");
        for (int i = 0; i < result_3->rows; i++) {
            for (int j = 0; j < result_3->cols; j++) {
                int16_t val;
                matrix_get(result_3, i, j, &val);
                printf("| %d ", val);
            }
            printf("|\n");
        }
    }
    EXPECT_TRUE(result_1 != NULL) << "Result matrix 1 should not be NULL";
    EXPECT_TRUE(matrix_compare(result_1, expected_result_1_3)) << "Result matrix 1 should match expected result";
    EXPECT_TRUE(result_2 != NULL) << "Result matrix 2 should not be NULL";
    EXPECT_TRUE(matrix_compare(result_2, expected_result_2)) << "Result matrix 2 should match expected result";
    EXPECT_TRUE(result_3 != NULL) << "Result matrix 3 should not be NULL";
    EXPECT_TRUE(matrix_compare(result_3, expected_result_1_3)) << "Result matrix 3 should match expected result";
    matrix_free(matrix1);
    matrix_free(matrix2);
    matrix_free(result_1);
    matrix_free(result_2);
    matrix_free(result_3);
    matrix_free(expected_result_1_3);
    matrix_free(expected_result_2);
}
