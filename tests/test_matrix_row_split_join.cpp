// Auto-migrated from test_matrix_row_split_join.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_row_split_join) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    void* row_0_ptrs[] = {row0};
    void* row_1_ptrs[] = {row1};
    Matrix* row_1st = matrix_create_from_2d_array(1, 2, row_0_ptrs, sizeof(int8_t));
    Matrix* row_2nd = matrix_create_from_2d_array(1, 2, row_1_ptrs, sizeof(int8_t));
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix** submatrices = matrix_split_by_rows(m, 2);
    EXPECT_TRUE(submatrices != NULL && submatrices[0] != NULL && submatrices[1] != NULL) << "Row split failed";
    Matrix* joined = matrix_join_by_rows(submatrices, 2);
    EXPECT_TRUE(matrix_compare(m, joined)) << "Row join failed";
    EXPECT_TRUE(matrix_compare(submatrices[0], row_1st)) << "First submatrix should match";
    EXPECT_TRUE(matrix_compare(submatrices[1], row_2nd)) << "Second submatrix should match";
    matrix_free(row_1st);
    matrix_free(row_2nd);
    matrix_free(m);
    matrix_free(submatrices[0]);
    matrix_free(submatrices[1]);
    free(submatrices);
    matrix_free(joined);
}
