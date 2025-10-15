#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_create_from_column_major_array_and_free() {
    printf("Running test_matrix_create_from_column_major_array_and_free...\n");
    int8_t data[] = {1, 3, 2, 4};
    Matrix* m = matrix_create_from_column_major_array(2, 2, data, sizeof(int8_t));
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    int8_t val;
    matrix_get(m, 0, 0, &val);
    ASSERT_EQ(val, 1, "Matrix get (0,0)");
    matrix_get(m, 0, 1, &val);
    ASSERT_EQ(val, 2, "Matrix get (0,1)");
    matrix_get(m, 1, 0, &val);
    ASSERT_EQ(val, 3, "Matrix get (1,0)");
    matrix_get(m, 1, 1, &val);
    ASSERT_EQ(val, 4, "Matrix get (1,1)");
    int8_t expected_row_major[] = {1, 2, 3, 4};
    int8_t* actual_row_major = (int8_t*)matrix_get_data_row_major(m);
    ASSERT_TRUE(memcmp(expected_row_major, actual_row_major, 4 * sizeof(int8_t)) == 0, "Row major data mismatch");
    free(actual_row_major);
    int8_t expected_col_major[] = {1, 3, 2, 4};
    int8_t* actual_col_major = (int8_t*)matrix_get_data_column_major(m);
    ASSERT_TRUE(memcmp(expected_col_major, actual_col_major, 4 * sizeof(int8_t)) == 0, "Column major data mismatch");
    free(actual_col_major);
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_matrix_create_from_column_major_array_and_free();
    if (result == 0) {
        printf("[PASS] test_matrix_create_from_column_major_array_and_free passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_create_from_column_major_array_and_free failed!\n");
        return 1;
    }
}
