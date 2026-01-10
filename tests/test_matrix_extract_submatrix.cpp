// Auto-migrated from test_matrix_extract_submatrix.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_extract_submatrix) {
    int8_t row0[] = {1, 2, 3, 4};
    int8_t row1[] = {5, 6, 7, 8};
    int8_t row2[] = {9, 10, 11, 12};
    int8_t* data[] = {row0, row1, row2};
    Matrix* m = matrix_create_from_2d_array(3, 4, (void**)data, sizeof(int8_t));
    
    // Extract 2x2 submatrix from top-left
    Matrix* sub = matrix_extract_submatrix(m, 2, 2);
    EXPECT_TRUE(sub != NULL) << "Matrix extract submatrix failed";
    EXPECT_EQ(sub->rows, 2) << "Extracted matrix should have 2 rows";
    EXPECT_EQ(sub->cols, 2) << "Extracted matrix should have 2 cols";
    
    // Check extracted values
    int8_t val;
    matrix_get(sub, 0, 0, &val);
    EXPECT_EQ(val, 1) << "Extracted (0,0) should be 1";
    matrix_get(sub, 0, 1, &val);
    EXPECT_EQ(val, 2) << "Extracted (0,1) should be 2";
    matrix_get(sub, 1, 0, &val);
    EXPECT_EQ(val, 5) << "Extracted (1,0) should be 5";
    matrix_get(sub, 1, 1, &val);
    EXPECT_EQ(val, 6) << "Extracted (1,1) should be 6";
    
    matrix_free(m);
    matrix_free(sub);
}
