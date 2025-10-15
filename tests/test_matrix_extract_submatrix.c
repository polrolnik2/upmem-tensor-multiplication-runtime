#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_extract_submatrix() {
    printf("Running test_matrix_extract_submatrix...\n");
    int8_t row0[] = {1, 2, 3, 4};
    int8_t row1[] = {5, 6, 7, 8};
    int8_t row2[] = {9, 10, 11, 12};
    int8_t* data[] = {row0, row1, row2};
    Matrix* m = matrix_create_from_2d_array(3, 4, (void**)data, sizeof(int8_t));
    
    // Extract 2x2 submatrix from top-left
    Matrix* sub = matrix_extract_submatrix(m, 2, 2);
    ASSERT_TRUE(sub != NULL, "Matrix extract submatrix failed");
    ASSERT_EQ(sub->rows, 2, "Extracted matrix should have 2 rows");
    ASSERT_EQ(sub->cols, 2, "Extracted matrix should have 2 cols");
    
    // Check extracted values
    int8_t val;
    matrix_get(sub, 0, 0, &val);
    ASSERT_EQ(val, 1, "Extracted (0,0) should be 1");
    matrix_get(sub, 0, 1, &val);
    ASSERT_EQ(val, 2, "Extracted (0,1) should be 2");
    matrix_get(sub, 1, 0, &val);
    ASSERT_EQ(val, 5, "Extracted (1,0) should be 5");
    matrix_get(sub, 1, 1, &val);
    ASSERT_EQ(val, 6, "Extracted (1,1) should be 6");
    
    matrix_free(m);
    matrix_free(sub);
    return 0;
}

int main() {
    int result = test_matrix_extract_submatrix();
    if (result == 0) {
        printf("[PASS] test_matrix_extract_submatrix passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_extract_submatrix failed!\n");
        return 1;
    }
}
