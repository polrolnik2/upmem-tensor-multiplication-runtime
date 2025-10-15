#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_add_cols_with_zero_fill() {
    printf("Running test_matrix_add_cols_with_zero_fill...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 2 columns with default zero fill
    Matrix* extended = matrix_add_cols(m, 2, NULL);
    ASSERT_TRUE(extended != NULL, "Matrix add cols failed");
    ASSERT_EQ(extended->rows, 2, "Extended matrix should have 2 rows");
    ASSERT_EQ(extended->cols, 4, "Extended matrix should have 4 cols");
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    ASSERT_EQ(val, 1, "Original data (0,0)");
    matrix_get(extended, 0, 1, &val);
    ASSERT_EQ(val, 2, "Original data (0,1)");
    matrix_get(extended, 1, 0, &val);
    ASSERT_EQ(val, 3, "Original data (1,0)");
    matrix_get(extended, 1, 1, &val);
    ASSERT_EQ(val, 4, "Original data (1,1)");
    
    // Check new columns are zero-filled
    matrix_get(extended, 0, 2, &val);
    ASSERT_EQ(val, 0, "New col (0,2) should be zero");
    matrix_get(extended, 0, 3, &val);
    ASSERT_EQ(val, 0, "New col (0,3) should be zero");
    matrix_get(extended, 1, 2, &val);
    ASSERT_EQ(val, 0, "New col (1,2) should be zero");
    matrix_get(extended, 1, 3, &val);
    ASSERT_EQ(val, 0, "New col (1,3) should be zero");
    
    matrix_free(m);
    matrix_free(extended);
    return 0;
}

int main() {
    int result = test_matrix_add_cols_with_zero_fill();
    if (result == 0) {
        printf("[PASS] test_matrix_add_cols_with_zero_fill passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_add_cols_with_zero_fill failed!\n");
        return 1;
    }
}
