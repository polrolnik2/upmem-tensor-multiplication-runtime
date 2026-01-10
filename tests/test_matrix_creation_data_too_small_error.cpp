// Auto-migrated from test_matrix_creation_data_too_small_error.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_creation_data_too_small_error) {
    int8_t data[] = {1, 2}; // Only enough for 1 row of 2 cols
    Matrix* m = matrix_create_from_row_major_array(2, 2, data, sizeof(int8_t));
    EXPECT_TRUE(m == NULL) << "Matrix creation should fail with insufficient data";
    m = matrix_create_from_column_major_array(2, 2, data, sizeof(int8_t));
    EXPECT_TRUE(m == NULL) << "Matrix creation should fail with insufficient data";
    void* data_ptrs[] = {data};
    m = matrix_create_from_2d_array(2, 2, data_ptrs, sizeof(int8_t));
    EXPECT_TRUE(m == NULL) << "Matrix creation should fail with insufficient data";
}
