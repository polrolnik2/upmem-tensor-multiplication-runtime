#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_add_rows_cols_error_cases() {
    printf("Running test_matrix_add_rows_cols_error_cases...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Test error cases
    Matrix* result = matrix_add_rows(NULL, 1, NULL);
    ASSERT_TRUE(result == NULL, "Add rows with NULL matrix should fail");
    
    result = matrix_add_rows(m, 0, NULL);
    ASSERT_TRUE(matrix_compare(result, m), "Add 0 rows should return the same matrix");

    result = matrix_add_rows(m, -1, NULL);
    ASSERT_TRUE(result == NULL, "Add negative rows should fail");
    
    result = matrix_add_cols(NULL, 1, NULL);
    ASSERT_TRUE(result == NULL, "Add cols with NULL matrix should fail");
    
    result = matrix_add_cols(m, 0, NULL);
    ASSERT_TRUE(matrix_compare(result, m), "Add 0 cols should return the same matrix");

    result = matrix_add_cols(m, -1, NULL);
    ASSERT_TRUE(result == NULL, "Add negative cols should fail");
    
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_matrix_add_rows_cols_error_cases();
    if (result == 0) {
        printf("[PASS] test_matrix_add_rows_cols_error_cases passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_add_rows_cols_error_cases failed!\n");
        return 1;
    }
}
