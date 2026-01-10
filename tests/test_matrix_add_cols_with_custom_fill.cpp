// Auto-migrated from test_matrix_add_cols_with_custom_fill.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_add_cols_with_custom_fill) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 1 column with custom fill value
    int8_t fill_value = 7;
    Matrix* extended = matrix_add_cols(m, 1, &fill_value);
    EXPECT_TRUE(extended != NULL) << "Matrix add cols failed";
    EXPECT_EQ(extended->rows, 2) << "Extended matrix should have 2 rows";
    EXPECT_EQ(extended->cols, 3) << "Extended matrix should have 3 cols";
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    EXPECT_EQ(val, 1) << "Original data (0,0)";
    matrix_get(extended, 1, 1, &val);
    EXPECT_EQ(val, 4) << "Original data (1,1)";
    
    // Check new column is filled with custom value
    matrix_get(extended, 0, 2, &val);
    EXPECT_EQ(val, 7) << "New col (0,2) should be filled value";
    matrix_get(extended, 1, 2, &val);
    EXPECT_EQ(val, 7) << "New col (1,2) should be filled value";
    
    matrix_free(m);
    matrix_free(extended);
}
