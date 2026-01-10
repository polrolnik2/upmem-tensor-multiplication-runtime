// Auto-migrated from test_matrix_add_rows_cols_different_types.c using migrate_to_gtest.py
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

TEST(MatrixTest, matrix_add_rows_cols_different_types) {
    
    // Test with int16_t
    int16_t row0_16[] = {100, 200};
    int16_t row1_16[] = {300, 400};
    int16_t* data_16[] = {row0_16, row1_16};
    Matrix* m16 = matrix_create_from_2d_array(2, 2, (void**)data_16, sizeof(int16_t));
    
    int16_t fill_16 = 999;
    Matrix* extended_16 = matrix_add_rows(m16, 1, &fill_16);
    EXPECT_TRUE(extended_16 != NULL) << "Matrix add rows int16 failed";
    EXPECT_EQ(extended_16->rows, 3) << "Extended int16 matrix should have 3 rows";
    
    int16_t val_16;
    matrix_get(extended_16, 2, 0, &val_16);
    EXPECT_EQ(val_16, 999) << "New row should have fill value";
    
    matrix_free(m16);
    matrix_free(extended_16);
}
