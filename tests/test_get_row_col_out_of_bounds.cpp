// Auto-migrated from test_get_row_col_out_of_bounds.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, get_row_col_out_of_bounds) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    EXPECT_TRUE(matrix_get_row(m, -1) == NULL) << "Get row -1 should fail";
    EXPECT_TRUE(matrix_get_row(m, 2) == NULL) << "Get row 2 should fail";
    EXPECT_TRUE(matrix_get_col(m, -1) == NULL) << "Get col -1 should fail";
    EXPECT_TRUE(matrix_get_col(m, 2) == NULL) << "Get col 2 should fail";
    matrix_free(m);
}
