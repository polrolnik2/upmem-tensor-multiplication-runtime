// Benchmark for back-to-back matrix multiplication operations.
// Measures:
// 1. Time to perform first multiplication
// 2. Time to load new matrices and get previous result
// 3. Time to perform second multiplication
// 4. Total execution time
//
// Usage: back_to_back_multiplication_benchmark <matrixA1.txt> <matrixB1.txt> <matrixA2.txt> <matrixB2.txt> [--dpus N] [--iterations N]
//
// File format: first line 'rows cols', then rows*cols integers row-major.
// Inputs are treated as int8 (-128..127). Results are uint16.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"
#include "timer.h"

// Parse a text file with format:
// rows cols\n
// then rows*cols integers in row-major order separated by whitespace.
// Values are parsed as int8_t. Returns a heap-allocated Matrix*.
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

static void usage(const char *prog) {
	fprintf(stderr, "Usage: %s <matrixA1.txt> <matrixB1.txt> <matrixA2.txt> <matrixB2.txt> [--dpus N] [--iterations N]\n", prog);
	fprintf(stderr, "  Performs back-to-back multiplications: A1*B1 then A2*B2\n");
	fprintf(stderr, "  Measures time for each multiplication and matrix transfers\n");
	fprintf(stderr, "  File format: first line 'rows cols', then rows*cols integers row-major.\n");
	fprintf(stderr, "  Default: 4 DPUs, 1 iteration\n");
}

int main(int argc, char **argv) {
	if (argc < 5) { usage(argv[0]); return 2; }
	
	const char *pathA1 = argv[1];
	const char *pathB1 = argv[2];
	const char *pathA2 = argv[3];
	const char *pathB2 = argv[4];
	
	uint32_t num_dpus = 4;     // default
	uint32_t iterations = 1;   // default
	
	for (int i = 5; i < argc; i++) {
		if (strcmp(argv[i], "--dpus") == 0 && i + 1 < argc) {
			num_dpus = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
			iterations = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			usage(argv[0]);
			return 0;
		}
	}
	Matrix *A1 = read_text_matrix_to_matrix(pathA1);
	if (!A1) {
		fprintf(stderr, "Failed to read matrix A1 from %s\n", pathA1);
		return 1;
	}
	Matrix *B1 = read_text_matrix_to_matrix(pathB1);
	if (!B1) {
		fprintf(stderr, "Failed to read matrix B1 from %s\n", pathB1);
		matrix_free(A1);
		return 1;
	}
	Matrix *A2 = read_text_matrix_to_matrix(pathA2);
	if (!A2) {
		fprintf(stderr, "Failed to read matrix A2 from %s\n", pathA2);
		matrix_free(A1);
		matrix_free(B1);
		return 1;
	}
	Matrix *B2 = read_text_matrix_to_matrix(pathB2);
	if (!B2) {
		fprintf(stderr, "Failed to read matrix B2 from %s\n", pathB2);
		matrix_free(A1);
		matrix_free(B1);
		matrix_free(A2);
		return 1;
	}

	// Validate dimensions
	if (A1->cols != B1->rows) {
		fprintf(stderr, "Incompatible dimensions for first multiplication: A1 is %ux%u, B1 is %ux%u\n", 
			A1->rows, A1->cols, B1->rows, B1->cols);
		matrix_free(A1); matrix_free(B1); matrix_free(A2); matrix_free(B2);
		return 1;
	}
	if (A2->cols != B2->rows) {
		fprintf(stderr, "Incompatible dimensions for second multiplication: A2 is %ux%u, B2 is %ux%u\n", 
			A2->rows, A2->cols, B2->rows, B2->cols);
		matrix_free(A1); matrix_free(B1); matrix_free(A2); matrix_free(B2);
		return 1;
	}

	// Create multiplication frame
	pim_matrix_multiplication_frame_t* frame = create_pim_matrix_multiplication_frame(
		num_dpus, 0, 
		A1->rows, A1->cols,
		B1->rows, B1->cols,
		A1->rows, B1->cols,
		sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t)
	);
	if (!frame) {
		fprintf(stderr, "Failed to create multiplication frame\n");
		matrix_free(A1); matrix_free(B1); matrix_free(A2); matrix_free(B2);
		return 1;
	}

	Timer total_timer;

	pim_matrix_multiplication_frame_load_first_matrix(frame, A1);
	pim_matrix_multiplication_frame_load_second_matrix(frame, B1);

	startTimer(&total_timer);

	for (uint32_t iter = 0; iter < iterations; iter++) {
		// Phase 1: Load first matrices and execute
		Timer timer;
		pim_matrix_multiplication_frame_execute(frame);
		pim_matrix_multiplication_frame_load_first_matrix(frame, A2);
		pim_matrix_multiplication_frame_load_second_matrix(frame, B2);
		Matrix *result1 = pim_matrix_multiplication_frame_get_result(frame);
		if (!result1) {
			fprintf(stderr, "Failed to get result from first multiplication\n");
			pim_matrix_multiplication_frame_free(frame);
			matrix_free(A1); matrix_free(B1); matrix_free(A2); matrix_free(B2);
			return 1;
		}
		pim_matrix_multiplication_frame_execute(frame);
		pim_matrix_multiplication_frame_load_first_matrix(frame, A2);
		pim_matrix_multiplication_frame_load_second_matrix(frame, B2);
		Matrix *result2 = pim_matrix_multiplication_frame_get_result(frame);
		if (!result2) {
			fprintf(stderr, "Failed to get result from second multiplication\n");
			pim_matrix_multiplication_frame_free(frame);
			matrix_free(A1); matrix_free(B1); matrix_free(A2); matrix_free(B2);
			matrix_free(result1);
			return 1;
		}
        
		// Free results
		matrix_free(result1);
		matrix_free(result2);
	}

	stopTimer(&total_timer);
	double total_time = getElapsedTime(total_timer);

	// Cleanup
	pim_matrix_multiplication_frame_free(frame);
	matrix_free(A1);
	matrix_free(B1);
	matrix_free(A2);
	matrix_free(B2);

	double total_per_iteration = total_time / iterations;

	printf("Total benchmark time:        %.3f ms\n", total_time);
	printf("Average per iteration:       %.3f ms\n", total_per_iteration);

	return 0;
}
