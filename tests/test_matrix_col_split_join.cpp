// Auto-migrated from test_matrix_col_split_join.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_col_split_join) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    int8_t col1_data[] = {1, 3};
    int8_t col2_data[] = {2, 4};
    int8_t* col1_rows[] = {&col1_data[0], &col1_data[1]};
    int8_t* col2_rows[] = {&col2_data[0], &col2_data[1]};
    Matrix* col_1st = matrix_create_from_2d_array(2, 1, (void**)col1_rows, sizeof(int8_t));
    Matrix* col_2nd = matrix_create_from_2d_array(2, 1, (void**)col2_rows, sizeof(int8_t));
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix** submatrices = matrix_split_by_cols(m, 2);
    EXPECT_TRUE(submatrices != NULL && submatrices[0] != NULL && submatrices[1] != NULL) << "Column split failed";
    EXPECT_TRUE(matrix_compare(submatrices[0], col_1st)) << "First submatrix should match";
    EXPECT_TRUE(matrix_compare(submatrices[1], col_2nd)) << "Second submatrix should match";
    // Join the submatrices back into a single matrix
    Matrix* joined = matrix_join_by_cols(submatrices, 2);
    EXPECT_TRUE(matrix_compare(m, joined)) << "Column join failed";
    matrix_free(col_1st);
    matrix_free(col_2nd);
    matrix_free(m);
    matrix_free(submatrices[0]);
    matrix_free(submatrices[1]);
    free(submatrices);
    matrix_free(joined);
}
