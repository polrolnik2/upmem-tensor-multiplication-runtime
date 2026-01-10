// Auto-migrated from test_matrix_clone_and_compare.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_clone_and_compare) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m1 = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix* m2 = matrix_clone(m1);
    EXPECT_TRUE(matrix_compare(m1, m2)) << "Matrix clone/compare failed";
    int8_t new_val = 9;
    matrix_set(m2, 0, 0, &new_val);
    EXPECT_TRUE(!matrix_compare(m1, m2)) << "Matrix compare after change";
    matrix_free(m1);
    matrix_free(m2);
}
