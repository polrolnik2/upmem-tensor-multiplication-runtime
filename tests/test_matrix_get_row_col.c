#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_get_row_col() {
    printf("Running test_matrix_get_row_col...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    int8_t* row = (int8_t*)matrix_get_row(m, 1);
    ASSERT_EQ(row[0], 3, "Get row 1 col 0");
    ASSERT_EQ(row[1], 4, "Get row 1 col 1");
    int8_t* row0ptr = (int8_t*)matrix_get_row(m, 0);
    ASSERT_EQ(row0ptr[0], 1, "Get row 0 col 0");
    ASSERT_EQ(row0ptr[1], 2, "Get row 0 col 1");
    int8_t* col0 = (int8_t*)matrix_get_col(m, 0);
    ASSERT_EQ(col0[0], 1, "Get col 0 row 0");
    ASSERT_EQ(col0[1], 3, "Get col 0 row 1");
    int8_t* col1 = (int8_t*)matrix_get_col(m, 1);
    ASSERT_EQ(col1[0], 2, "Get col 1 row 0");
    ASSERT_EQ(col1[1], 4, "Get col 1 row 1");
    free(col0);
    free(col1);
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_matrix_get_row_col();
    if (result == 0) {
        printf("[PASS] test_matrix_get_row_col passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_get_row_col failed!\n");
        return 1;
    }
}
