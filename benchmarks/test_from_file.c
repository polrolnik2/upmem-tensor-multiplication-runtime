// Test that reads matrices from files and multiplies on CPU and DPU, comparing results.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"
#include "host_multiply_matrices.h"
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
	if (elem_bytes == 1) {
		uint8_t *buf = (uint8_t*)malloc(count * sizeof(uint8_t));
		if (!buf) { fclose(f); return NULL; }
		for (uint64_t i = 0; i < count; i++) {
			long v;
			if (fscanf(f, "%ld", &v) != 1) { free(buf); fclose(f); return NULL; }
			if (v < 0) v = 0; if (v > 255) v = 255;
			buf[i] = (uint8_t)v;
		}
		fclose(f);
		Matrix *mat = matrix_create_from_row_major_array(rows, cols, buf, sizeof(uint8_t));
		free(buf);
		return mat;
	} else if (elem_bytes == 2) {
		uint16_t *buf = (uint16_t*)malloc(count * sizeof(uint16_t));
		if (!buf) { fclose(f); return NULL; }
		for (uint64_t i = 0; i < count; i++) {
			long v;
			if (fscanf(f, "%ld", &v) != 1) { free(buf); fclose(f); return NULL; }
			if (v < 0) v = 0; if (v > 65535) v = 65535;
			buf[i] = (uint16_t)v;
		}
		fclose(f);
		Matrix *mat = matrix_create_from_row_major_array(rows, cols, buf, sizeof(uint16_t));
		free(buf);
		return mat;
	} else {
		fclose(f);
		fprintf(stderr, "Unsupported element byte size: %u\n", elem_bytes);
		return NULL;
	}
}

static void print_u16_matrix(const Matrix *m, const char *label) {
	if (!m) return;
	printf("%s (%ux%u):\n", label, m->rows, m->cols);
	for (uint32_t i = 0; i < m->rows; i++) {
		for (uint32_t j = 0; j < m->cols; j++) {
			uint16_t v = 0; matrix_get(m, i, j, &v);
			printf("%5u ", v);
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
	if (argc < 4) { usage(argv[0]); return 2; }
	const char *pathA = argv[1];
	const char *pathB = argv[2];
	const char *pathRef = argv[3];
	uint32_t num_dpus = 4; // default

	for (int i = 4; i < argc; i++) {
		if (strcmp(argv[i], "--dpus") == 0 && i + 1 < argc) {
			num_dpus = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			usage(argv[0]);
			return 0;
		}
	}

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

	// Read reference matrix (expected result) from file (uint16 elements)
	Matrix *ref = read_text_matrix_to_matrix(pathRef, 2);
	if (!ref) {
		fprintf(stderr, "Failed to read reference matrix from %s\n", pathRef);
		matrix_free(A); matrix_free(B);
		return 1;
	}

	// Basic dimensions for result
	if (ref->rows != A->rows || ref->cols != B->cols) {
		fprintf(stderr, "Reference dimensions mismatch: reference is %ux%u, expected %ux%u\n", ref->rows, ref->cols, A->rows, B->cols);
		matrix_free(A); matrix_free(B); matrix_free(ref);
		return 1;
	}

	// DPU multiplication
	Matrix *dpu_res = dpu_multiply_matrices(A, B, num_dpus);
	ASSERT_TRUE(dpu_res != NULL, "DPU multiply returned NULL");

	// Compare DPU result to reference
	if (!matrix_compare(dpu_res, ref)) {
		printf("[FAIL] DPU result differs from reference.\n");
		if (ref->rows <= 16 && ref->cols <= 16) {
			print_u16_matrix(ref, "REFERENCE");
			print_u16_matrix(dpu_res, "DPU");
		}
		matrix_free(A); matrix_free(B); matrix_free(ref); matrix_free(dpu_res);
		return 1;
	}

	printf("[PASS] DPU result matches reference for %ux%u x %ux%u using %u DPUs.\n", A->rows, A->cols, B->rows, B->cols, num_dpus);

	matrix_free(A);
	matrix_free(B);
	matrix_free(ref);
	matrix_free(dpu_res);
	return 0;
}

