// Auto-migrated from test_matrix_add_rows_cols_error_cases.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_add_rows_cols_error_cases) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Test error cases
    Matrix* result = matrix_add_rows(NULL, 1, NULL);
    EXPECT_TRUE(result == NULL) << "Add rows with NULL matrix should fail";
    
    result = matrix_add_rows(m, 0, NULL);
    EXPECT_TRUE(matrix_compare(result, m)) << "Add 0 rows should return the same matrix";

    result = matrix_add_rows(m, -1, NULL);
    EXPECT_TRUE(result == NULL) << "Add negative rows should fail";
    
    result = matrix_add_cols(NULL, 1, NULL);
    EXPECT_TRUE(result == NULL) << "Add cols with NULL matrix should fail";
    
    result = matrix_add_cols(m, 0, NULL);
    EXPECT_TRUE(matrix_compare(result, m)) << "Add 0 cols should return the same matrix";

    result = matrix_add_cols(m, -1, NULL);
    EXPECT_TRUE(result == NULL) << "Add negative cols should fail";
    
    matrix_free(m);
}
