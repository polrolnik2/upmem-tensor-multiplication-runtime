/**
 * @file matrix.c
 * @brief Implementation of the Matrix struct and related functions.
 */
#include "matrix.h"
#include <string.h>

Matrix* matrix_create_from_2d_array(int16_t rows, int16_t cols, void **data, uint32_t element_size) {
    if (rows <= 0 || cols <= 0 || !data || element_size == 0) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->element_size = element_size;
    mat->data = (void**)malloc(rows * sizeof(void*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = malloc(cols * element_size);
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        memcpy(mat->data[r], data[r], cols * element_size);
    }
    return mat;
}

Matrix* matrix_create_from_row_major_array(int16_t rows, int16_t cols, void *data, uint32_t element_size) {
    if (rows <= 0 || cols <= 0 || !data || element_size == 0) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->element_size = element_size;
    mat->data = (void**)malloc(rows * sizeof(void*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = malloc(cols * element_size);
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        memcpy(mat->data[r], (char*)data + r * cols * element_size, cols * element_size);
    }
    return mat;
}

Matrix* matrix_create_from_column_major_array(int16_t rows, int16_t cols, void *data, uint32_t element_size) {
    if (rows <= 0 || cols <= 0 || !data || element_size == 0) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->element_size = element_size;
    mat->data = (void**)malloc(rows * sizeof(void*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = malloc(cols * element_size);
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        for (int c = 0; c < cols; ++c) {
            memcpy((char*)mat->data[r] + c * element_size, 
                   (char*)data + (c * rows + r) * element_size, 
                   element_size);
        }
    }
    return mat;
}

Matrix* matrix_create_from_4d_row_major_tiled_array(int16_t num_row_tiles, int16_t num_col_tiles,
                                                   int16_t tile_rows, int16_t tile_cols,
                                                   void *data, uint32_t element_size) {
    if (num_row_tiles <= 0 || num_col_tiles <= 0 || tile_rows <= 0 || tile_cols <= 0 || !data || element_size == 0) return NULL;
    
    Matrix*** tiles = NULL;
    Matrix** row_matrices = NULL;
    Matrix* result = NULL;
    
    // Create a 2D array of Matrix pointers to hold all tiles
    tiles = (Matrix***)malloc(num_row_tiles * sizeof(Matrix**));
    if (!tiles) goto cleanup;
    
    // Initialize all pointers to NULL for safe cleanup
    for (int16_t i = 0; i < num_row_tiles; i++) {
        tiles[i] = NULL;
    }
    
    for (int16_t i = 0; i < num_row_tiles; i++) {
        tiles[i] = (Matrix**)malloc(num_col_tiles * sizeof(Matrix*));
        if (!tiles[i]) goto cleanup;
        
        // Initialize all tile pointers to NULL for safe cleanup
        for (int16_t j = 0; j < num_col_tiles; j++) {
            tiles[i][j] = NULL;
        }
    }
    
    // Create individual tile matrices from the 4D data
    char* src_data = (char*)data;
    size_t tile_size_bytes = tile_rows * tile_cols * element_size;
    
    for (int16_t tile_row = 0; tile_row < num_row_tiles; tile_row++) {
        for (int16_t tile_col = 0; tile_col < num_col_tiles; tile_col++) {
            // Calculate the offset for this tile in the source data
            size_t tile_offset = (tile_row * num_col_tiles + tile_col) * tile_size_bytes;
            void* tile_data = src_data + tile_offset;
            
            // Create a matrix from this tile's data
            tiles[tile_row][tile_col] = matrix_create_from_row_major_array(tile_rows, tile_cols, tile_data, element_size);
            if (!tiles[tile_row][tile_col]) goto cleanup;
        }
    }
    
    // Join tiles by columns first (for each row of tiles)
    row_matrices = (Matrix**)malloc(num_row_tiles * sizeof(Matrix*));
    if (!row_matrices) goto cleanup;
    
    for (int16_t row = 0; row < num_row_tiles; row++) {
        row_matrices[row] = matrix_join_by_cols(tiles[row], num_col_tiles);
        if (!row_matrices[row]) goto cleanup;
    }
    
    // Join all row matrices to create the final result
    result = matrix_join_by_rows(row_matrices, num_row_tiles);
    if (!result) goto cleanup;
    
    // Clean up intermediate structures but keep the result
    for (int16_t r = 0; r < num_row_tiles; r++) {
        if (tiles[r]) {
            for (int16_t c = 0; c < num_col_tiles; c++) {
                if (tiles[r][c]) matrix_free(tiles[r][c]);
            }
            free(tiles[r]);
        }
        if (row_matrices[r]) matrix_free(row_matrices[r]);
    }
    free(tiles);
    free(row_matrices);
    
    return result;

cleanup:
    if (tiles) {
        for (int16_t r = 0; r < num_row_tiles; r++) {
            if (tiles[r]) {
                for (int16_t c = 0; c < num_col_tiles; c++) {
                    if (tiles[r][c]) matrix_free(tiles[r][c]);
                }
                free(tiles[r]);
            }
        }
        free(tiles);
    }
    
    if (row_matrices) {
        for (int16_t r = 0; r < num_row_tiles; r++) {
            if (row_matrices[r]) matrix_free(row_matrices[r]);
        }
        free(row_matrices);
    }
    
    if (result) matrix_free(result);
    
    return NULL;
}

void matrix_free(Matrix* mat) {
    if (mat) {
        if (mat->data) {
            for (int r = 0; r < mat->rows; ++r) {
                free(mat->data[r]);
            }
            free(mat->data);
        }
        free(mat);
    }
}

void* matrix_get_row(const Matrix* mat, int r) {
    if (!mat || r < 0 || r >= mat->rows) return NULL;
    return mat->data[r];
}

void* matrix_get_col(const Matrix* mat, int c) {
    if (!mat || c < 0 || c >= mat->cols) return NULL;
    void* col = malloc(mat->rows * mat->element_size);
    if (!col) return NULL;
    for (int r = 0; r < mat->rows; ++r) {
        memcpy((char*)col + r * mat->element_size, 
               (char*)mat->data[r] + c * mat->element_size, 
               mat->element_size);
    }
    return col;
}

void* matrix_get_data_row_major(const Matrix* mat) {
    if (!mat) return NULL;
    void* data = malloc(mat->rows * mat->cols * mat->element_size);
    if (!data) return NULL;
    for (int r = 0; r < mat->rows; ++r) {
        memcpy((char*)data + r * mat->cols * mat->element_size, 
               matrix_get_row(mat, r), 
               mat->cols * mat->element_size);
    }
    return data;
}

void* matrix_get_data_column_major(const Matrix* mat) {
    if (!mat) return NULL;
    void* data = malloc(mat->rows * mat->cols * mat->element_size);
    if (!data) return NULL;
    for (int c = 0; c < mat->cols; ++c) {
        void* col_data = matrix_get_col(mat, c);
        if (!col_data) {
            free(data);
            return NULL;
        }
        memcpy((char*)data + c * mat->rows * mat->element_size, 
               col_data, 
               mat->rows * mat->element_size);
        free(col_data);
    }
    return data;
}

Matrix* matrix_clone(const Matrix* mat) {
    if (!mat) return NULL;
    Matrix* copy = (Matrix*)malloc(sizeof(Matrix));
    if (!copy) return NULL;
    copy->rows = mat->rows;
    copy->cols = mat->cols;
    copy->element_size = mat->element_size;
    copy->data = (void**)malloc(mat->rows * sizeof(void*));
    if (!copy->data) {
        free(copy);
        return NULL;
    }
    for (int r = 0; r < mat->rows; ++r) {
        copy->data[r] = malloc(mat->cols * mat->element_size);
        if (!copy->data[r]) {
            for (int i = 0; i < r; ++i) free(copy->data[i]);
            free(copy->data);
            free(copy);
            return NULL;
        }
        memcpy(copy->data[r], mat->data[r], mat->cols * mat->element_size);
    }
    return copy;
}

bool matrix_compare(const Matrix* a, const Matrix* b) {
    if (!a || !b) return false;
    if (a->rows != b->rows || a->cols != b->cols || a->element_size != b->element_size) return false;
    for (int r = 0; r < a->rows; ++r) {
        if (memcmp(a->data[r], b->data[r], a->cols * a->element_size) != 0) return false;
    }
    return true;
}

char* matrix_sprint(const Matrix* mat, const char* format) {
    if (!mat || !format) return NULL;
    
    // Estimate buffer size (rough approximation)
    int max_element_str_len = 20; // Should be enough for most numeric types
    int bufsize = mat->rows * mat->cols * max_element_str_len + mat->rows + 1;
    char* buf = (char*)malloc(bufsize);
    if (!buf) return NULL;
    
    int pos = 0;
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            // Format based on element size
            if (mat->element_size == sizeof(uint8_t)) {
                uint8_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(int16_t)) {
                int16_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(int32_t)) {
                int32_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(float)) {
                float value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(double)) {
                double value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else {
                // For unknown types, format as hex
                pos += snprintf(buf + pos, bufsize - pos, "0x");
                for (uint32_t k = 0; k < mat->element_size; k++) {
                    pos += snprintf(buf + pos, bufsize - pos, "%02x", 
                                  ((unsigned char*)mat->data[r])[c * mat->element_size + k]);
                }
                pos += snprintf(buf + pos, bufsize - pos, " ");
            }
        }
        pos += snprintf(buf + pos, bufsize - pos, "\n");
    }
    buf[pos] = '\0';
    return buf;
}

void matrix_print(const Matrix* mat, const char* format) {
    if (!mat || !format) return;
    
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            // Print based on element size and format
            if (mat->element_size == sizeof(uint8_t)) {
                uint8_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(int16_t)) {
                int16_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(int32_t)) {
                int32_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(float)) {
                float value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(double)) {
                double value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else {
                // For unknown types, print as hex bytes
                printf("0x");
                for (uint32_t k = 0; k < mat->element_size; k++) {
                    printf("%02x", ((unsigned char*)mat->data[r])[c * mat->element_size + k]);
                }
                printf(" ");
            }
        }
        printf("\n");
    }
}

int matrix_get(const Matrix* mat, int r, int c, void* out) {
    if (!mat || !out || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return -1;
    memcpy(out, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
    return 0;
}

int matrix_set(Matrix* mat, int r, int c, const void* value) {
    if (!mat || !value || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return -1;
    memcpy((char*)mat->data[r] + c * mat->element_size, value, mat->element_size);
    return 0;
}

Matrix** matrix_split_by_rows(const Matrix* mat, int num_submatrices) {
    if (!mat || num_submatrices <= 0 || mat->rows < num_submatrices || mat->rows % num_submatrices != 0) return NULL;
    Matrix** submatrices = (Matrix**)malloc(num_submatrices * sizeof(Matrix*));
    if (!submatrices) return NULL;
    int rows_per_submatrix = mat->rows / num_submatrices;
    for (int i = 0; i < num_submatrices; ++i) {
        int start_row = i * rows_per_submatrix;
        int end_row = (i == num_submatrices - 1) ? mat->rows : start_row + rows_per_submatrix;
        int sub_rows = end_row - start_row;
        void **sub_data = (void**)malloc(sub_rows * sizeof(void*));
        if (!sub_data) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
        for (int r = 0; r < sub_rows; ++r) {
            sub_data[r] = malloc(mat->cols * mat->element_size);
            if (!sub_data[r]) {
                for (int j = 0; j < r; ++j) free(sub_data[j]);
                free(sub_data);
                for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
                free(submatrices);
                return NULL;
            }
            memcpy(sub_data[r], matrix_get_row(mat, start_row + r), mat->cols * mat->element_size);
        }
        submatrices[i] = matrix_create_from_2d_array(sub_rows, mat->cols, sub_data, mat->element_size);
        if (!submatrices[i]) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
    }
    return submatrices;
}

Matrix** matrix_split_by_cols(const Matrix* mat, int num_submatrices) {
    if (!mat || num_submatrices <= 0 || mat->cols < num_submatrices || mat->cols % num_submatrices != 0) return NULL;
    Matrix** submatrices = (Matrix**)malloc(num_submatrices * sizeof(Matrix*));
    if (!submatrices) return NULL;
    int cols_per_submatrix = mat->cols / num_submatrices;
    for (int i = 0; i < num_submatrices; ++i) {
        int start_col = i * cols_per_submatrix;
        int end_col = (i == num_submatrices - 1) ? mat->cols : start_col + cols_per_submatrix;
        int sub_cols = end_col - start_col;
        void * sub_data_column_major = malloc(mat->rows * sub_cols * mat->element_size);
        if (!sub_data_column_major) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
        for (int c = 0; c < sub_cols; ++c) {
            void* col_data = matrix_get_col(mat, start_col + c);
            if (!col_data) {
                free(sub_data_column_major);
                for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
                free(submatrices);
                return NULL;
            }
            memcpy((char*)sub_data_column_major + c * mat->rows * mat->element_size, col_data, mat->rows * mat->element_size);
            free(col_data);
        }
        submatrices[i] = matrix_create_from_column_major_array(mat->rows, sub_cols, sub_data_column_major, mat->element_size);
        free(sub_data_column_major);
        if (!submatrices[i]) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
    }
    return submatrices;
}

Matrix* matrix_join_by_rows(Matrix** submatrices, int num_submatrices) {
    if (!submatrices || num_submatrices <= 0) return NULL;
    int total_rows = 0;
    int cols = submatrices[0]->cols;
    for (int i = 0; i < num_submatrices; ++i) {
        if (!submatrices[i] || submatrices[i]->cols != cols) {
            return NULL;
        }
        total_rows += submatrices[i]->rows;
    }

    if (num_submatrices == 1) {
        Matrix* single_matrix = matrix_clone(submatrices[0]);
        return single_matrix;
    }

    // Create temporary 2D array for matrix data
    void **temp_data = (void**)malloc(total_rows * sizeof(void*));
    if (!temp_data) {
        for (int i = 0; i < num_submatrices; ++i) matrix_free(submatrices[i]);
        free(submatrices);
        return NULL;
    }
    int current_row = 0;
    for (int i = 0; i < num_submatrices; ++i) {
        for (int r = 0; r < submatrices[i]->rows; ++r) {
            temp_data[current_row] = malloc(cols * submatrices[i]->element_size);
            if (!temp_data[current_row]) {
                for (int j = 0; j < current_row; ++j) free(temp_data[j]);
                free(temp_data);
                for (int j = 0; j < num_submatrices; ++j) matrix_free(submatrices[j]);
                free(submatrices);
                return NULL;
            }
            memcpy(temp_data[current_row], submatrices[i]->data[r], cols * submatrices[i]->element_size);
            current_row++;
        }
    }

    // Create matrix from temporary data
    Matrix* mat = matrix_create_from_2d_array(total_rows, cols, temp_data, submatrices[0]->element_size);
    free(temp_data);
    return mat;
}

Matrix* matrix_join_by_cols(Matrix** submatrices, int num_submatrices) {
    if (!submatrices || num_submatrices <= 0) return NULL;
    int total_cols = 0;
    int rows = submatrices[0]->rows;
    for (int i = 0; i < num_submatrices; ++i) {
        if (!submatrices[i] || submatrices[i]->rows != rows) {
            return NULL;
        }
        total_cols += submatrices[i]->cols;
    }
    
    if (num_submatrices == 1) {
        Matrix* single_matrix = matrix_clone(submatrices[0]);
        return single_matrix;
    }

    // Create temporary 2D array for matrix data
    void *temp_data_column_major = malloc(rows * total_cols * submatrices[0]->element_size);
    if (!temp_data_column_major) {
        for (int i = 0; i < num_submatrices; ++i) matrix_free(submatrices[i]);
        free(submatrices);
        return NULL;
    }
    int current_col = 0;
    for (int i = 0; i < num_submatrices; ++i) {
        for (int c = 0; c < submatrices[i]->cols; ++c) {
            void* col_data = matrix_get_col(submatrices[i], c);
            if (!col_data) {
                free(temp_data_column_major);
                for (int j = 0; j < num_submatrices; ++j) matrix_free(submatrices[j]);
                free(submatrices);
                return NULL;
            }
            memcpy((char*)temp_data_column_major + current_col * rows * submatrices[0]->element_size, col_data, rows * submatrices[0]->element_size);
            free(col_data);
            current_col++;
        }
    }

    // Create matrix from temporary data
    Matrix* mat = matrix_create_from_column_major_array(rows, total_cols, temp_data_column_major, submatrices[0]->element_size);
    free(temp_data_column_major);
    return mat;
}

Matrix* matrix_add_rows(const Matrix* mat, int16_t num_rows, const void* fill_value) {
    if (!mat || num_rows < 0) return NULL;

    if (num_rows == 0) {
        // If no rows to add, return a clone of the original matrix
        return matrix_clone(mat);
    }
    
    int16_t new_rows = mat->rows + num_rows;
    int16_t cols = mat->cols;
    
    // Create new matrix data structure
    void** new_data = (void**)malloc(new_rows * sizeof(void*));
    if (!new_data) return NULL;
    
    // Copy existing rows
    for (int r = 0; r < mat->rows; ++r) {
        new_data[r] = malloc(cols * mat->element_size);
        if (!new_data[r]) {
            for (int i = 0; i < r; ++i) free(new_data[i]);
            free(new_data);
            return NULL;
        }
        memcpy(new_data[r], mat->data[r], cols * mat->element_size);
    }
    
    // Create fill pattern (default to zero)
    void* fill_pattern = malloc(mat->element_size);
    if (!fill_pattern) {
        for (int i = 0; i < mat->rows; ++i) free(new_data[i]);
        free(new_data);
        return NULL;
    }
    
    if (fill_value) {
        memcpy(fill_pattern, fill_value, mat->element_size);
    } else {
        memset(fill_pattern, 0, mat->element_size);
    }
    
    // Add new rows filled with the pattern
    for (int r = mat->rows; r < new_rows; ++r) {
        new_data[r] = malloc(cols * mat->element_size);
        if (!new_data[r]) {
            for (int i = 0; i < r; ++i) free(new_data[i]);
            free(new_data);
            free(fill_pattern);
            return NULL;
        }
        
        // Fill the row with the fill pattern
        for (int c = 0; c < cols; ++c) {
            memcpy((char*)new_data[r] + c * mat->element_size, fill_pattern, mat->element_size);
        }
    }
    
    free(fill_pattern);
    
    // Create the new matrix
    Matrix* result = matrix_create_from_2d_array(new_rows, cols, new_data, mat->element_size);
    
    // Clean up temporary data
    for (int i = 0; i < new_rows; ++i) free(new_data[i]);
    free(new_data);
    
    return result;
}

Matrix* matrix_add_cols(const Matrix* mat, int16_t num_cols, const void* fill_value) {
    if (!mat || num_cols < 0) return NULL;

    if (num_cols == 0) {
        // If no columns to add, return a clone of the original matrix
        return matrix_clone(mat);
    }
    
    int16_t rows = mat->rows;
    int16_t new_cols = mat->cols + num_cols;
    
    // Create new matrix data structure  
    void** new_data = (void**)malloc(rows * sizeof(void*));
    if (!new_data) return NULL;
    
    // Create fill pattern (default to zero)
    void* fill_pattern = malloc(mat->element_size);
    if (!fill_pattern) {
        free(new_data);
        return NULL;
    }
    
    if (fill_value) {
        memcpy(fill_pattern, fill_value, mat->element_size);
    } else {
        memset(fill_pattern, 0, mat->element_size);
    }
    
    // Create each row with original data + new columns
    for (int r = 0; r < rows; ++r) {
        new_data[r] = malloc(new_cols * mat->element_size);
        if (!new_data[r]) {
            for (int i = 0; i < r; ++i) free(new_data[i]);
            free(new_data);
            free(fill_pattern);
            return NULL;
        }
        
        // Copy existing columns
        memcpy(new_data[r], mat->data[r], mat->cols * mat->element_size);
        
        // Fill new columns with the pattern
        for (int c = mat->cols; c < new_cols; ++c) {
            memcpy((char*)new_data[r] + c * mat->element_size, fill_pattern, mat->element_size);
        }
    }
    
    free(fill_pattern);
    
    // Create the new matrix
    Matrix* result = matrix_create_from_2d_array(rows, new_cols, new_data, mat->element_size);
    
    // Clean up temporary data
    for (int i = 0; i < rows; ++i) free(new_data[i]);
    free(new_data);
    
    return result;
}

Matrix* matrix_remove_rows(const Matrix* mat, int16_t num_rows) {
    if (!mat || num_rows < 0) return NULL;
    
    if (num_rows == 0) {
        // If no rows to remove, return a clone of the original matrix
        return matrix_clone(mat);
    }
    
    if (num_rows >= mat->rows) {
        // Cannot remove more rows than exist
        return NULL;
    }
    
    int16_t new_rows = mat->rows - num_rows;
    int16_t cols = mat->cols;
    
    // Create new matrix data structure
    void** new_data = (void**)malloc(new_rows * sizeof(void*));
    if (!new_data) return NULL;
    
    // Copy existing rows (exclude the last num_rows)
    for (int r = 0; r < new_rows; ++r) {
        new_data[r] = malloc(cols * mat->element_size);
        if (!new_data[r]) {
            for (int i = 0; i < r; ++i) free(new_data[i]);
            free(new_data);
            return NULL;
        }
        memcpy(new_data[r], mat->data[r], cols * mat->element_size);
    }
    
    // Create the new matrix
    Matrix* result = matrix_create_from_2d_array(new_rows, cols, new_data, mat->element_size);
    
    // Clean up temporary data
    for (int i = 0; i < new_rows; ++i) free(new_data[i]);
    free(new_data);
    
    return result;
}

Matrix* matrix_remove_cols(const Matrix* mat, int16_t num_cols) {
    if (!mat || num_cols < 0) return NULL;
    
    if (num_cols == 0) {
        // If no columns to remove, return a clone of the original matrix
        return matrix_clone(mat);
    }
    
    if (num_cols >= mat->cols) {
        // Cannot remove more columns than exist
        return NULL;
    }
    
    int16_t rows = mat->rows;
    int16_t new_cols = mat->cols - num_cols;
    
    // Create new matrix data structure  
    void** new_data = (void**)malloc(rows * sizeof(void*));
    if (!new_data) return NULL;
    
    // Create each row with reduced columns
    for (int r = 0; r < rows; ++r) {
        new_data[r] = malloc(new_cols * mat->element_size);
        if (!new_data[r]) {
            for (int i = 0; i < r; ++i) free(new_data[i]);
            free(new_data);
            return NULL;
        }
        
        // Copy existing columns (exclude the last num_cols)
        memcpy(new_data[r], mat->data[r], new_cols * mat->element_size);
    }
    
    // Create the new matrix
    Matrix* result = matrix_create_from_2d_array(rows, new_cols, new_data, mat->element_size);
    
    // Clean up temporary data
    for (int i = 0; i < rows; ++i) free(new_data[i]);
    free(new_data);
    
    return result;
}

Matrix* matrix_extract_submatrix(const Matrix* mat, int16_t target_rows, int16_t target_cols) {
    if (!mat || target_rows <= 0 || target_cols <= 0) return NULL;
    
    if (target_rows > mat->rows || target_cols > mat->cols) {
        // Cannot extract more rows/columns than exist
        return NULL;
    }
    
    if (target_rows == mat->rows && target_cols == mat->cols) {
        // If extracting the same dimensions, return a clone
        return matrix_clone(mat);
    }
    
    // First remove excess rows from the bottom
    int16_t rows_to_remove = mat->rows - target_rows;
    Matrix* temp_matrix = matrix_remove_rows(mat, rows_to_remove);
    if (!temp_matrix) return NULL;
    
    // Then remove excess columns from the right
    int16_t cols_to_remove = mat->cols - target_cols;
    Matrix* result = matrix_remove_cols(temp_matrix, cols_to_remove);
    
    // Clean up temporary matrix
    matrix_free(temp_matrix);
    
    return result;
}

Matrix * matrix_transpose(const Matrix * mat) {
    if (!mat) return NULL;
    void * data = matrix_get_data_row_major(mat);
    return matrix_create_from_column_major_array(mat->cols, mat->rows, data, mat->element_size);
}
