#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_creation_data_too_small_error() {
    printf("Running test_matrix_creation_data_too_small_error...\n");
    int8_t data[] = {1, 2}; // Only enough for 1 row of 2 cols
    Matrix* m = matrix_create_from_row_major_array(2, 2, data, sizeof(int8_t));
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    m = matrix_create_from_column_major_array(2, 2, data, sizeof(int8_t));
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    m = matrix_create_from_2d_array(2, 2, (void*[]){data}, sizeof(int8_t));
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    return 0;
}

int main() {
    int result = test_matrix_creation_data_too_small_error();
    if (result == 0) {
        printf("[PASS] test_matrix_creation_data_too_small_error passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_creation_data_too_small_error failed!\n");
        return 1;
    }
}
