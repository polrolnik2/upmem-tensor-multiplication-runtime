// Test that reads matrices from files and multiplies on CPU and DPU, comparing results.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"
#include "dpu_multiply_matrices.h"
#include "test_assertions.h"

#include "timer.h"

// Parse a text file with format:
// rows cols\n
// then rows*cols integers in row-major order separated by whitespace.
// Values are parsed as 0..255 and stored as uint8_t. Returns a heap-allocated Matrix*.
static Matrix* read_text_matrix_to_matrix(const char *path) {
	FILE *f = fopen(path, "r");
	if (!f) {
		fprintf(stderr, "Failed to open %s\n", path);
		return NULL;
	}
	uint32_t rows = 0, cols = 0;
	if (fscanf(f, "%u %u", &rows, &cols) != 2) {
		fclose(f);
		fprintf(stderr, "Invalid header in %s (expected 'rows cols')\n", path);
		return NULL;
	}
	// consume rest of the line
	int c;
	while ((c = fgetc(f)) != '\n' && c != EOF) {}

	if (rows == 0 || cols == 0) {
		fclose(f);
		fprintf(stderr, "Invalid dimensions in %s: %u x %u\n", path, rows, cols);
		return NULL;
	}

	uint64_t count = (uint64_t)rows * (uint64_t)cols;
	int8_t *buf = (int8_t*)malloc(count * sizeof(int8_t));
	if (!buf) { fclose(f); return NULL; }
	for (uint64_t i = 0; i < count; i++) {
        long v;
        if (fscanf(f, "%ld", &v) != 1) { free(buf); fclose(f); return NULL; }
        buf[i] = (int8_t)v;
    }
    fclose(f);
    Matrix *mat = matrix_create_from_row_major_array(rows, cols, buf, sizeof(int8_t));
    free(buf);
    return mat;
}

// Parse a text file with format:
// rows cols\n
// then rows*cols integers in row-major order separated by whitespace.
// Values are parsed as 0..255 and stored as uint8_t. Returns a heap-allocated Matrix*.
static Matrix* read_reference_matrix_to_matrix(const char *path) {
	FILE *f = fopen(path, "r");
	if (!f) {
		fprintf(stderr, "Failed to open %s\n", path);
		return NULL;
	}
	uint32_t rows = 0, cols = 0;
	if (fscanf(f, "%u %u", &rows, &cols) != 2) {
		fclose(f);
		fprintf(stderr, "Invalid header in %s (expected 'rows cols')\n", path);
		return NULL;
	}
	// consume rest of the line
	int c;
	while ((c = fgetc(f)) != '\n' && c != EOF) {}

	if (rows == 0 || cols == 0) {
		fclose(f);
		fprintf(stderr, "Invalid dimensions in %s: %u x %u\n", path, rows, cols);
		return NULL;
	}

	uint64_t count = (uint64_t)rows * (uint64_t)cols;
	int16_t *buf = (int16_t*)malloc(count * sizeof(int16_t));
	if (!buf) { fclose(f); return NULL; }
	for (uint64_t i = 0; i < count; i++) {
        long v;
        if (fscanf(f, "%ld", &v) != 1) { free(buf); fclose(f); return NULL; }
        buf[i] = (int16_t)v;
    }
    fclose(f);
    Matrix *mat = matrix_create_from_row_major_array(rows, cols, buf, sizeof(int16_t));
    free(buf);
    return mat;
}

void write_matrix_to_file(const Matrix *m, const char *path) {
	if (!m) return;
	FILE *f = fopen(path, "w");
	if (!f) {
		fprintf(stderr, "Failed to open %s for writing\n", path);
		return;
	}
	fprintf(f, "%u %u\n", m->rows, m->cols);
	for (uint32_t i = 0; i < m->rows; i++) {
		for (uint32_t j = 0; j < m->cols; j++) {
			uint16_t v = 0; matrix_get(m, i, j, &v);
			fprintf(f, "%u ", v);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

static void print_u16_matrix(const Matrix *m, const char *label) {
	if (!m) return;
	printf("%s (%ux%u):\n", label, m->rows, m->cols);
	for (uint32_t i = 0; i < m->rows; i++) {
		for (uint32_t j = 0; j < m->cols; j++) {
			int16_t v = 0; matrix_get(m, i, j, &v);
			printf("%5d ", v);
		}
		printf("\n");
	}
}

static void usage(const char *prog) {
	fprintf(stderr, "Usage: %s <matrixA.txt> <matrixB.txt> <reference_result.txt> [--dpus N]\n", prog);
	fprintf(stderr, "  File format: first line 'rows cols', then rows*cols integers row-major.\n");
	fprintf(stderr, "  Inputs are treated as uint8 (0..255). Reference/result should be uint16 values.\n");
}

int main(int argc, char **argv) {
	if (argc < 3) { usage(argv[0]); return 2; }
	const char *pathA = argv[1];
	const char *pathB = argv[2];
	uint32_t num_dpus = 4; // default
	const char *reference_file = NULL;

	for (int i = 3; i < argc; i++) {
		if (strcmp(argv[i], "--dpus") == 0 && i + 1 < argc) {
			num_dpus = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			usage(argv[0]);
			return 0;
		} else if (strcmp(argv[i], "--reference-file") == 0 && i + 1 < argc) {
			reference_file = argv[++i];
		}
	}

	if (reference_file == NULL) {
		printf("No reference file specified\n");
		return 1;
	}

	printf("Running DPU matrix multiplication test with %u DPUs\n", num_dpus);

	Matrix *A = read_text_matrix_to_matrix(pathA);
	if (!A) {
		fprintf(stderr, "Failed to read matrix A from %s\n", pathA);
		return 1;
	}
	Matrix *B = read_text_matrix_to_matrix(pathB);
	if (!B) {
		fprintf(stderr, "Failed to read matrix B from %s\n", pathB);
		matrix_free(A);
		return 1;
	}

	// Basic dimension check
	if (A->cols != B->rows) {
		fprintf(stderr, "Incompatible dimensions: A is %ux%u, B is %ux%u\n", A->rows, A->cols, B->rows, B->cols);
		matrix_free(A); matrix_free(B);
		return 1;
	}

	// DPU multiplication
	Matrix *dpu_res = dpu_multiply_matrices(A, B, num_dpus);
	ASSERT_TRUE(dpu_res != NULL, "DPU multiply returned NULL");

	Matrix *ref_res = read_reference_matrix_to_matrix(reference_file);
	if (!ref_res) {
		fprintf(stderr, "Failed to read reference result from %s\n", reference_file);
		matrix_free(A); matrix_free(B);
		matrix_free(dpu_res);
		return 1;
	}

	if(!matrix_compare(dpu_res, ref_res)) {
		fprintf(stderr, "[Fail] DPU result does not match reference result from %s\n", reference_file);
		print_u16_matrix(dpu_res, "DPU Result");
		print_u16_matrix(ref_res, "Reference Result");
		matrix_free(ref_res);
		matrix_free(A);
		matrix_free(B);
		matrix_free(dpu_res);
		return 1;
	}

	printf("[Pass] DPU result matches reference result from %s\n", reference_file);
	
	matrix_free(A);
	matrix_free(B);
	matrix_free(dpu_res);
	matrix_free(ref_res);
	return 0;
}

