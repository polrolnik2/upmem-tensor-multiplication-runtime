#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_add_rows_with_custom_fill() {
    printf("Running test_matrix_add_rows_with_custom_fill...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 1 row with custom fill value
    int8_t fill_value = 9;
    Matrix* extended = matrix_add_rows(m, 1, &fill_value);
    ASSERT_TRUE(extended != NULL, "Matrix add rows failed");
    ASSERT_EQ(extended->rows, 3, "Extended matrix should have 3 rows");
    ASSERT_EQ(extended->cols, 2, "Extended matrix should have 2 cols");
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    ASSERT_EQ(val, 1, "Original data (0,0)");
    matrix_get(extended, 1, 1, &val);
    ASSERT_EQ(val, 4, "Original data (1,1)");
    
    // Check new row is filled with custom value
    matrix_get(extended, 2, 0, &val);
    ASSERT_EQ(val, 9, "New row (2,0) should be filled value");
    matrix_get(extended, 2, 1, &val);
    ASSERT_EQ(val, 9, "New row (2,1) should be filled value");
    
    matrix_free(m);
    matrix_free(extended);
    return 0;
}

int main() {
    int result = test_matrix_add_rows_with_custom_fill();
    if (result == 0) {
        printf("[PASS] test_matrix_add_rows_with_custom_fill passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_add_rows_with_custom_fill failed!\n");
        return 1;
    }
}
