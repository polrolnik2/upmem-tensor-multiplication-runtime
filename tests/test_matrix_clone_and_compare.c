#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_clone_and_compare() {
    printf("Running test_matrix_clone_and_compare...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m1 = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix* m2 = matrix_clone(m1);
    ASSERT_TRUE(matrix_compare(m1, m2), "Matrix clone/compare failed");
    int8_t new_val = 9;
    matrix_set(m2, 0, 0, &new_val);
    ASSERT_TRUE(!matrix_compare(m1, m2), "Matrix compare after change");
    matrix_free(m1);
    matrix_free(m2);
    return 0;
}

int main() {
    int result = test_matrix_clone_and_compare();
    if (result == 0) {
        printf("[PASS] test_matrix_clone_and_compare passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_clone_and_compare failed!\n");
        return 1;
    }
}
