#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_remove_error_cases() {
    printf("Running test_matrix_remove_error_cases...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Test error cases
    Matrix* result = matrix_remove_rows(NULL, 1);
    ASSERT_TRUE(result == NULL, "Remove rows with NULL matrix should fail");
    
    result = matrix_remove_rows(m, -1);
    ASSERT_TRUE(result == NULL, "Remove negative rows should fail");
    
    result = matrix_remove_rows(m, 3);
    ASSERT_TRUE(result == NULL, "Remove more rows than exist should fail");
    
    result = matrix_remove_cols(NULL, 1);
    ASSERT_TRUE(result == NULL, "Remove cols with NULL matrix should fail");
    
    result = matrix_remove_cols(m, -1);
    ASSERT_TRUE(result == NULL, "Remove negative cols should fail");
    
    result = matrix_remove_cols(m, 3);
    ASSERT_TRUE(result == NULL, "Remove more cols than exist should fail");
    
    result = matrix_extract_submatrix(NULL, 1, 1);
    ASSERT_TRUE(result == NULL, "Extract with NULL matrix should fail");
    
    result = matrix_extract_submatrix(m, 0, 1);
    ASSERT_TRUE(result == NULL, "Extract with zero rows should fail");
    
    result = matrix_extract_submatrix(m, 1, 0);
    ASSERT_TRUE(result == NULL, "Extract with zero cols should fail");
    
    result = matrix_extract_submatrix(m, 3, 2);
    ASSERT_TRUE(result == NULL, "Extract more rows than exist should fail");
    
    result = matrix_extract_submatrix(m, 2, 3);
    ASSERT_TRUE(result == NULL, "Extract more cols than exist should fail");
    
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_matrix_remove_error_cases();
    if (result == 0) {
        printf("[PASS] test_matrix_remove_error_cases passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_remove_error_cases failed!\n");
        return 1;
    }
}
