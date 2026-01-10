// Auto-migrated from test_matrix_print_and_sprint.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_print_and_sprint) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    char* str = matrix_sprint(m, "%d ");
    EXPECT_TRUE(strstr(str, "1 2") && strstr(str, "3 4")) << "Matrix sprint";
    free(str);
    matrix_free(m);
}
