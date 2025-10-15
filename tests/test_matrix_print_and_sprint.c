#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_print_and_sprint() {
    printf("Running test_matrix_print_and_sprint...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    char* str = matrix_sprint(m, "%d ");
    ASSERT_TRUE(strstr(str, "1 2") && strstr(str, "3 4"), "Matrix sprint");
    free(str);
    matrix_free(m);
    return 0;
}

int main() {
    int result = test_matrix_print_and_sprint();
    if (result == 0) {
        printf("[PASS] test_matrix_print_and_sprint passed!\n");
        return 0;
    } else {
        printf("[FAIL] test_matrix_print_and_sprint failed!\n");
        return 1;
    }
}
