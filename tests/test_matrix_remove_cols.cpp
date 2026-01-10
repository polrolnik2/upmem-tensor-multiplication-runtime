// Auto-migrated from test_matrix_remove_cols.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_remove_cols) {
    int8_t row0[] = {1, 2, 3};
    int8_t row1[] = {4, 5, 6};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 3, (void**)data, sizeof(int8_t));
    
    // Remove 1 column
    Matrix* reduced = matrix_remove_cols(m, 1);
    EXPECT_TRUE(reduced != NULL) << "Matrix remove cols failed";
    EXPECT_EQ(reduced->rows, 2) << "Reduced matrix should have 2 rows";
    EXPECT_EQ(reduced->cols, 2) << "Reduced matrix should have 2 cols";
    
    // Check that only first 2 columns remain
    int8_t val;
    matrix_get(reduced, 0, 0, &val);
    EXPECT_EQ(val, 1) << "First column should be preserved";
    matrix_get(reduced, 0, 1, &val);
    EXPECT_EQ(val, 2) << "Second column should be preserved";
    matrix_get(reduced, 1, 0, &val);
    EXPECT_EQ(val, 4) << "First column second row should be preserved";
    matrix_get(reduced, 1, 1, &val);
    EXPECT_EQ(val, 5) << "Second column second row should be preserved";
    
    matrix_free(m);
    matrix_free(reduced);
}
