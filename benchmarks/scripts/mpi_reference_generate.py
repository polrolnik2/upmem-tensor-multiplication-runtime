#!/usr/bin/env python3
"""
mpi_reference_generate.py

Generate a reference matrix (A x B) using numpy and MPI (mpi4py).

Usage (example):
  mpirun -n 4 python3 benchmarks/scripts/mpi_reference_generate.py A.txt B.txt reference.txt

Input format (text): first line: "rows cols" then rows lines with cols whitespace-separated integers (row-major).
Output format (text): same format; by default output elements are clipped to uint16 (0..65535).

This script distributes the row-blocks of the output across MPI ranks; rank 0 collects and writes the final file.
"""
import sys
import argparse
import numpy as np

try:
	from mpi4py import MPI
except Exception as e:
	print("mpi4py is required to run this script. Install it with 'pip install mpi4py'.", file=sys.stderr)
	raise


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
	parser = argparse.ArgumentParser(description='Generate reference matrix using MPI and numpy')
	parser.add_argument('matrix_a', help='Path to matrix A (text)')
	parser.add_argument('matrix_b', help='Path to matrix B (text)')
	parser.add_argument('output', help='Path to write reference matrix (text)')
	parser.add_argument('--out-bytes', type=int, choices=(1,2,4), default=2, help='Bytes per output element (1=uint8,2=uint16,4=uint32)')
	args = parser.parse_args()

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	A = None
	B = None
	rows = cols = None

	if rank == 0:
		try:
			A = read_matrix_text(args.matrix_a, dtype=np.int64)
			B = read_matrix_text(args.matrix_b, dtype=np.int64)
		except Exception as e:
			print(f"Error reading input files: {e}", file=sys.stderr)
			comm.Abort(1)

		if A.shape[1] != B.shape[0]:
			print(f"Incompatible dimensions: A is {A.shape}, B is {B.shape}", file=sys.stderr)
			comm.Abort(1)

		rows, k = A.shape
		k2, cols = B.shape
	else:
		# placeholders
		A = None; B = None
		rows = None; cols = None

	# Broadcast shapes
	rows = comm.bcast(rows, root=0)
	cols = comm.bcast(cols, root=0)

	# Broadcast matrices (pickle-based); for huge matrices consider using Scatterv with buffers
	A = comm.bcast(A, root=0)
	B = comm.bcast(B, root=0)

	# Determine row ranges for each rank
	counts = [rows // size + (1 if i < (rows % size) else 0) for i in range(size)]
	starts = [sum(counts[:i]) for i in range(size)]
	my_start = starts[rank]
	my_count = counts[rank]
	my_end = my_start + my_count

	if my_count == 0:
		local_result = np.empty((0, cols), dtype=np.uint64)
	else:
		# compute local block as uint64 to avoid overflow
		A_block = A[my_start:my_end, :].astype(np.uint64)
		B_cast = B.astype(np.uint64)
		local_result = A_block.dot(B_cast)

	# Gather results at root
	gathered = comm.gather(local_result, root=0)

	if rank == 0:
		# concatenate
		result = np.vstack(gathered)
		# clip and cast to desired output type
		if args.out_bytes == 1:
			out = np.clip(result, 0, 255).astype(np.uint8)
		elif args.out_bytes == 2:
			out = np.clip(result, 0, 65535).astype(np.uint16)
		else:
			out = result.astype(np.uint32)

		# write to file
		write_matrix_text(args.output, out)
		print(f"Reference matrix written to {args.output} (shape {out.shape}, dtype {out.dtype})")


if __name__ == '__main__':
	main()

