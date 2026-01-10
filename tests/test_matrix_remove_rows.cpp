// Auto-migrated from test_matrix_remove_rows.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_remove_rows) {
    int8_t row0[] = {1, 2, 3};
    int8_t row1[] = {4, 5, 6};
    int8_t row2[] = {7, 8, 9};
    int8_t* data[] = {row0, row1, row2};
    Matrix* m = matrix_create_from_2d_array(3, 3, (void**)data, sizeof(int8_t));
    
    // Remove 1 row
    Matrix* reduced = matrix_remove_rows(m, 1);
    EXPECT_TRUE(reduced != NULL) << "Matrix remove rows failed";
    EXPECT_EQ(reduced->rows, 2) << "Reduced matrix should have 2 rows";
    EXPECT_EQ(reduced->cols, 3) << "Reduced matrix should have 3 cols";
    
    // Check that only first 2 rows remain
    int8_t val;
    matrix_get(reduced, 0, 0, &val);
    EXPECT_EQ(val, 1) << "First row should be preserved";
    matrix_get(reduced, 1, 2, &val);
    EXPECT_EQ(val, 6) << "Second row should be preserved";
    
    matrix_free(m);
    matrix_free(reduced);
}
