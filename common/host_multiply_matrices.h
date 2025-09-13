#ifndef __HOST_MULTIPLY_MATRICES_H___
#define __HOST_MULTIPLY_MATRICES_H___

Matrix* host_multiply_matrices(const Matrix* matrix1, const Matrix* matrix2) {
    if (!matrix1 || !matrix2 || matrix1->cols != matrix2->rows) return NULL;
    uint16_t * result_data_row_major = malloc(matrix1->rows * matrix2->cols * sizeof(uint16_t));
    if (!result_data_row_major) return NULL;
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix2->cols; j++) {
            uint16_t sum = 0;
            for (int k = 0; k < matrix1->cols; k++) {
                uint8_t val1, val2;
                matrix_get(matrix1, i, k, &val1);
                matrix_get(matrix2, k, j, &val2);
                sum += val1 * val2;
            }
            result_data_row_major[i*matrix2->cols + j] = sum;
        }
    }
    Matrix* result = matrix_create_from_row_major_array(matrix1->rows, matrix2->cols, result_data_row_major, sizeof(uint16_t));
    if (!result) {
        free(result_data_row_major);
        return NULL;
    }
    free(result_data_row_major);
    return result;
}

#endif // __HOST_MULTIPLY_MATRICES_H___