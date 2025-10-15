#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_add_remove_roundtrip() {
    printf("Running test_matrix_add_remove_roundtrip...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* original = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add then remove rows
    Matrix* extended_rows = matrix_add_rows(original, 2, NULL);
    Matrix* restored_rows = matrix_remove_rows(extended_rows, 2);
    ASSERT_TRUE(matrix_compare(original, restored_rows), "Add/remove rows roundtrip failed");
    
    // Add then remove columns  
    Matrix* extended_cols = matrix_add_cols(original, 2, NULL);
    Matrix* restored_cols = matrix_remove_cols(extended_cols, 2);
    ASSERT_TRUE(matrix_compare(original, restored_cols), "Add/remove cols roundtrip failed");
    
    // Extract original dimensions should be equivalent
    Matrix* extracted = matrix_extract_submatrix(extended_rows, 2, 2);
    ASSERT_TRUE(matrix_compare(original, extracted), "Extract original dimensions failed");
    
    matrix_free(original);
    matrix_free(extended_rows);
    matrix_free(restored_rows);
    matrix_free(extended_cols);
    matrix_free(restored_cols);
    matrix_free(extracted);
    return 0;
}

int main() {
    int result = test_matrix_add_remove_roundtrip();
    if (result == 0) {
        printf("[PASS] test_matrix_add_remove_roundtrip passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_add_remove_roundtrip failed!\n");
        return 1;
    }
}
