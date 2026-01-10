// Auto-migrated from test_matrix_add_rows_with_custom_fill.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_add_rows_with_custom_fill) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 1 row with custom fill value
    int8_t fill_value = 9;
    Matrix* extended = matrix_add_rows(m, 1, &fill_value);
    EXPECT_TRUE(extended != NULL) << "Matrix add rows failed";
    EXPECT_EQ(extended->rows, 3) << "Extended matrix should have 3 rows";
    EXPECT_EQ(extended->cols, 2) << "Extended matrix should have 2 cols";
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    EXPECT_EQ(val, 1) << "Original data (0,0)";
    matrix_get(extended, 1, 1, &val);
    EXPECT_EQ(val, 4) << "Original data (1,1)";
    
    // Check new row is filled with custom value
    matrix_get(extended, 2, 0, &val);
    EXPECT_EQ(val, 9) << "New row (2,0) should be filled value";
    matrix_get(extended, 2, 1, &val);
    EXPECT_EQ(val, 9) << "New row (2,1) should be filled value";
    
    matrix_free(m);
    matrix_free(extended);
}
