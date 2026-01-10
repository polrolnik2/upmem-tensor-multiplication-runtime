// Auto-migrated from test_matrix_add_rows_with_zero_fill.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_add_rows_with_zero_fill) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 2 rows with default zero fill
    Matrix* extended = matrix_add_rows(m, 2, NULL);
    EXPECT_TRUE(extended != NULL) << "Matrix add rows failed";
    EXPECT_EQ(extended->rows, 4) << "Extended matrix should have 4 rows";
    EXPECT_EQ(extended->cols, 2) << "Extended matrix should have 2 cols";
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    EXPECT_EQ(val, 1) << "Original data (0,0)";
    matrix_get(extended, 0, 1, &val);
    EXPECT_EQ(val, 2) << "Original data (0,1)";
    matrix_get(extended, 1, 0, &val);
    EXPECT_EQ(val, 3) << "Original data (1,0)";
    matrix_get(extended, 1, 1, &val);
    EXPECT_EQ(val, 4) << "Original data (1,1)";
    
    // Check new rows are zero-filled
    matrix_get(extended, 2, 0, &val);
    EXPECT_EQ(val, 0) << "New row (2,0) should be zero";
    matrix_get(extended, 2, 1, &val);
    EXPECT_EQ(val, 0) << "New row (2,1) should be zero";
    matrix_get(extended, 3, 0, &val);
    EXPECT_EQ(val, 0) << "New row (3,0) should be zero";
    matrix_get(extended, 3, 1, &val);
    EXPECT_EQ(val, 0) << "New row (3,1) should be zero";
    
    matrix_free(m);
    matrix_free(extended);
}
