#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_split_into_0_error() {
    printf("Running test_matrix_split_into_0_error...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix** submatrices = matrix_split_by_rows(m, 0);
    ASSERT_TRUE(submatrices == NULL, "Split into 0 submatrices should fail");
    submatrices = matrix_split_by_cols(m, 0);
    ASSERT_TRUE(submatrices == NULL, "Split into 0 submatrices should fail");
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_matrix_split_into_0_error();
    if (result == 0) {
        printf("[PASS] test_matrix_split_into_0_error passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_split_into_0_error failed!\n");
        return 1;
    }
}
