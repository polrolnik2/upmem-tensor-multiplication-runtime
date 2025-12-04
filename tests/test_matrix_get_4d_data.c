#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_get_4d_data() {
    printf("Running test_matrix_get_4d_data...\n");
    int8_t row0[] = {1, 2, 3, 4};
    int8_t row1[] = {5, 6, 7, 8};
    int8_t row2[] = {9, 10, 11, 12};
    int8_t row3[] = {13, 14, 15, 16};
    int8_t* data[] = {row0, row1, row2, row3};
    Matrix* m = matrix_create_from_2d_array(4, 4, (void**)data, sizeof(int8_t));
    
    // Extract 2x2 submatrix from top-left
    int8_t* actual = matrix_get_data_4d_row_major_tiled(m, 2, 2);
    ASSERT_TRUE(actual != NULL, "matrix_get_data_4d_row_major_tiled should not return NULL");
    int8_t expected[] = {
        1, 2, 5, 6,
        3, 4, 7, 8,
        9, 10, 13, 14,
        11, 12, 15, 16
    };
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(actual[i], expected[i], "4D tiled data does not match expected output");
    }
    
    matrix_free(m);
    free(actual);
    return 0;
}

int main() {
    int result = test_matrix_get_4d_data();
    if (result == 0) {
        printf("[PASS] test_matrix_get_4d_data passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_get_4d_data failed!\n");
        return 1;
    }
}
