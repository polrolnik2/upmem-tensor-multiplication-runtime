#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_row_split_join() {
    printf("Running test_matrix_row_split_join...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* row_1st = matrix_create_from_2d_array(1, 2, (void*[]){row0}, sizeof(int8_t));
    Matrix* row_2nd = matrix_create_from_2d_array(1, 2, (void*[]){row1}, sizeof(int8_t));
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix** submatrices = matrix_split_by_rows(m, 2);
    ASSERT_TRUE(submatrices != NULL && submatrices[0] != NULL && submatrices[1] != NULL, "Row split failed");
    Matrix* joined = matrix_join_by_rows(submatrices, 2);
    ASSERT_TRUE(matrix_compare(m, joined), "Row join failed");
    ASSERT_TRUE(matrix_compare(submatrices[0], row_1st), "First submatrix should match");
    ASSERT_TRUE(matrix_compare(submatrices[1], row_2nd), "Second submatrix should match");
    matrix_free(row_1st);
    matrix_free(row_2nd);
    matrix_free(m);
    matrix_free(submatrices[0]);
    matrix_free(submatrices[1]);
    free(submatrices);
    matrix_free(joined);
    return 0;
}

int main() {
    int result = test_matrix_row_split_join();
    if (result == 0) {
        printf("[PASS] test_matrix_row_split_join passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_row_split_join failed!\n");
        return 1;
    }
}
