// Auto-migrated from test_matrix_remove_error_cases.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_remove_error_cases) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Test error cases
    Matrix* result = matrix_remove_rows(NULL, 1);
    EXPECT_TRUE(result == NULL) << "Remove rows with NULL matrix should fail";
    
    result = matrix_remove_rows(m, -1);
    EXPECT_TRUE(result == NULL) << "Remove negative rows should fail";
    
    result = matrix_remove_rows(m, 3);
    EXPECT_TRUE(result == NULL) << "Remove more rows than exist should fail";
    
    result = matrix_remove_cols(NULL, 1);
    EXPECT_TRUE(result == NULL) << "Remove cols with NULL matrix should fail";
    
    result = matrix_remove_cols(m, -1);
    EXPECT_TRUE(result == NULL) << "Remove negative cols should fail";
    
    result = matrix_remove_cols(m, 3);
    EXPECT_TRUE(result == NULL) << "Remove more cols than exist should fail";
    
    result = matrix_extract_submatrix(NULL, 1, 1);
    EXPECT_TRUE(result == NULL) << "Extract with NULL matrix should fail";
    
    result = matrix_extract_submatrix(m, 0, 1);
    EXPECT_TRUE(result == NULL) << "Extract with zero rows should fail";
    
    result = matrix_extract_submatrix(m, 1, 0);
    EXPECT_TRUE(result == NULL) << "Extract with zero cols should fail";
    
    result = matrix_extract_submatrix(m, 3, 2);
    EXPECT_TRUE(result == NULL) << "Extract more rows than exist should fail";
    
    result = matrix_extract_submatrix(m, 2, 3);
    EXPECT_TRUE(result == NULL) << "Extract more cols than exist should fail";
    
    matrix_free(m);
}
