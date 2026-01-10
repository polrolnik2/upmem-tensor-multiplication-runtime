// Auto-migrated from test_matrix_get_4d_data.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_get_4d_data) {
    int8_t row0[] = {1, 2, 3, 4};
    int8_t row1[] = {5, 6, 7, 8};
    int8_t row2[] = {9, 10, 11, 12};
    int8_t row3[] = {13, 14, 15, 16};
    int8_t* data[] = {row0, row1, row2, row3};
    Matrix* m = matrix_create_from_2d_array(4, 4, (void**)data, sizeof(int8_t));
    
    // Extract 2x2 submatrix from top-left
    int8_t* actual = (int8_t*)matrix_get_data_4d_row_major_tiled(m, 2, 2);
    EXPECT_TRUE(actual != NULL) << "matrix_get_data_4d_row_major_tiled should not return NULL";
    int8_t expected[] = {
        1, 2, 5, 6,
        3, 4, 7, 8,
        9, 10, 13, 14,
        11, 12, 15, 16
    };
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(actual[i], expected[i]) << "4D tiled data does not match expected output";
    }
    
    matrix_free(m);
    free(actual);
}
