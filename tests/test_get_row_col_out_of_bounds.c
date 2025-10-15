#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_get_row_col_out_of_bounds() {
    printf("Running test_get_row_col_out_of_bounds...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    ASSERT_TRUE(matrix_get_row(m, -1) == NULL, "Get row -1 should fail");
    ASSERT_TRUE(matrix_get_row(m, 2) == NULL, "Get row 2 should fail");
    ASSERT_TRUE(matrix_get_col(m, -1) == NULL, "Get col -1 should fail");
    ASSERT_TRUE(matrix_get_col(m, 2) == NULL, "Get col 2 should fail");
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_get_row_col_out_of_bounds();
    if (result == 0) {
        printf("[PASS] test_get_row_col_out_of_bounds passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_get_row_col_out_of_bounds failed!\n");
        return 1;
    }
}
