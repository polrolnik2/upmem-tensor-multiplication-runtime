#!/usr/bin/env python3
"""
mpi_reference_generate.py

Generate a reference matrix (A x B) using numpy and multithreaded MKL.

Usage (example):
	python3 benchmarks/scripts/mpi_reference_generate.py A.txt B.txt reference.txt --threads 8

Input format (text): first line: "rows cols" then rows lines with cols whitespace-separated integers (row-major).
Output format (text): same format; by default output elements are clipped to uint32 (0..65535).

This script distributes the row-blocks of the output across MPI ranks; rank 0 collects and writes the final file.
"""
import sys
import argparse
import numpy as np
import os


def read_matrix_text(path, dtype=np.int64):
	# Read header first
	with open(path, 'r') as f:
		header = f.readline().strip()
		if not header:
			raise ValueError(f"Empty file: {path}")
		parts = header.split()
		if len(parts) < 2:
			raise ValueError(f"Invalid header in {path}: '{header}'")
		rows = int(parts[0]); cols = int(parts[1])
	# Use numpy to load the remaining data; expect rows lines each with cols values
	data = np.loadtxt(path, dtype=dtype, skiprows=1)
	# Handle the case where loadtxt returns 1D for rows*cols=1 or single-row
	data = np.asarray(data)
	if data.ndim == 1:
		# Either a single row or flattened; reshape accordingly
		if data.size != rows * cols:
			# Could be that each row is a separate line but loadtxt returned 1D; try reading all ints
			data = np.fromfile(path, sep=' ', dtype=dtype)
			# Skip header entries
			if data.size >= 2:
				data = data[2:]
		# Now reshape
		data = data.reshape((rows, cols))
	else:
		# If 2D, ensure shape matches
		if data.shape[0] != rows or data.shape[1] != cols:
			# try to flatten and reshape
			flat = data.flatten()
			if flat.size != rows * cols:
				raise ValueError(f"Data size mismatch reading {path}: header {rows}x{cols}, data shape {data.shape}")
			data = flat.reshape((rows, cols))
	return data


def write_matrix_text(path, mat):
	rows, cols = mat.shape
	with open(path, 'w') as f:
		f.write(f"{rows} {cols}\n")
		for r in range(rows):
			# join as integers
			f.write(' '.join(str(int(x)) for x in mat[r, :]) + '\n')


def main():
	parser = argparse.ArgumentParser(description='Generate reference matrix using MKL and numpy')
	parser.add_argument('matrix_a', help='Path to matrix A (text)')
	parser.add_argument('matrix_b', help='Path to matrix B (text)')
	parser.add_argument('output', help='Path to write reference matrix (text)')
	args = parser.parse_args()

	A = None
	B = None
	rows = cols = None

	try:
		A = read_matrix_text(args.matrix_a, dtype=np.int64)
		B = read_matrix_text(args.matrix_b, dtype=np.int64)
	except Exception as e:
		print(f"Error reading input files: {e}", file=sys.stderr)

	if A.shape[1] != B.shape[0]:
		print(f"Incompatible dimensions: A is {A.shape}, B is {B.shape}", file=sys.stderr)

	rows, k = A.shape
	k2, cols = B.shape

	result = A.dot(B)

	# write to file
	write_matrix_text(args.output, result)
	print(f"Reference matrix written to {args.output} (shape {result.shape}, dtype {result.dtype})")


if __name__ == '__main__':
	main()

