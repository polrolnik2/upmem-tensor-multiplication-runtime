// Auto-migrated from test_matrix_split_into_0_error.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_split_into_0_error) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix** submatrices = matrix_split_by_rows(m, 0);
    EXPECT_TRUE(submatrices == NULL) << "Split into 0 submatrices should fail";
    submatrices = matrix_split_by_cols(m, 0);
    EXPECT_TRUE(submatrices == NULL) << "Split into 0 submatrices should fail";
    matrix_free(m);
}
