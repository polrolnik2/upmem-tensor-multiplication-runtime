// Auto-migrated from test_join_unmatching_dimensions_error.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, join_unmatching_dimensions_error) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m1 = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    int8_t row0_2[] = {5, 6, 7};
    int8_t row1_2[] = {8, 9, 10};
    int8_t row2_2[] = {11, 12, 13};
    int8_t* data2[] = {row0_2, row1_2, row2_2};
    Matrix* m2 = matrix_create_from_2d_array(3, 3, (void**)data2, sizeof(int8_t));
    Matrix* matrices[] = {m1, m2};
    Matrix* joined = matrix_join_by_rows(matrices, 2);
    EXPECT_TRUE(joined == NULL) << "Joining matrices with unmatching dimensions should fail";
    Matrix* joined_col = matrix_join_by_cols(matrices, 2);
    EXPECT_TRUE(joined_col == NULL) << "Joining matrices with unmatching dimensions should fail";
    matrix_free(m1);
    matrix_free(m2);
}
