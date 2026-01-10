// Auto-migrated from test_matrix_add_remove_roundtrip.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_add_remove_roundtrip) {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* original = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add then remove rows
    Matrix* extended_rows = matrix_add_rows(original, 2, NULL);
    Matrix* restored_rows = matrix_remove_rows(extended_rows, 2);
    EXPECT_TRUE(matrix_compare(original, restored_rows)) << "Add/remove rows roundtrip failed";
    
    // Add then remove columns  
    Matrix* extended_cols = matrix_add_cols(original, 2, NULL);
    Matrix* restored_cols = matrix_remove_cols(extended_cols, 2);
    EXPECT_TRUE(matrix_compare(original, restored_cols)) << "Add/remove cols roundtrip failed";
    
    // Extract original dimensions should be equivalent
    Matrix* extracted = matrix_extract_submatrix(extended_rows, 2, 2);
    EXPECT_TRUE(matrix_compare(original, extracted)) << "Extract original dimensions failed";
    
    matrix_free(original);
    matrix_free(extended_rows);
    matrix_free(restored_rows);
    matrix_free(extended_cols);
    matrix_free(restored_cols);
    matrix_free(extracted);
}
