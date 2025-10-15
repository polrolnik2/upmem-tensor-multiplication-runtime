#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_split_matrix_indivisible_error() {
    printf("Running test_split_matrix_indivisible_error...\n");
    int8_t row0[] = {1, 2, 3};
    int8_t row1[] = {4, 5, 6};
    int8_t row2[] = {7, 8, 9};
    int8_t* data[] = {row0, row1, row2};
    Matrix* m = matrix_create_from_2d_array(3, 3, (void**)data, sizeof(int8_t));
    Matrix** submatrices = matrix_split_by_rows(m, 2);
    ASSERT_TRUE(submatrices == NULL, "Split into more submatrices than rows should fail");
    submatrices = matrix_split_by_cols(m, 2);
    ASSERT_TRUE(submatrices == NULL, "Split into more submatrices than cols should fail");
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_split_matrix_indivisible_error();
    if (result == 0) {
        printf("[PASS] test_split_matrix_indivisible_error passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_split_matrix_indivisible_error failed!\n");
        return 1;
    }
}
