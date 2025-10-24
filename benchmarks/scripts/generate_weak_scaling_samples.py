#!/usr/bin/env python3
"""
generate_weak_scaling_samples.py

Create input matrices A and B and their reference product for weak-scaling tests.

This script is intended to pre-generate inputs and reference outputs that the
`benchmarks/scripts/weak_scaling_tests.sh` harness can later use to run tests
against different accelerators.

Output format (text): first line: "rows cols" then rows lines with cols
whitespace-separated integers (row-major). Reference output elements are
clipped to uint16 (0..65535) by default.

Usage examples:
  # generate default classes from the script using 3 instances each
  python3 benchmarks/references/generate_weak_scaling_samples.py --instances 3

  # generate for specific sizes
  python3 benchmarks/references/generate_weak_scaling_samples.py --sizes 256,256,256 512,512,512 --instances 2

Outputs are placed under scratch/references/<class_name>/ with files named
A_<seed>.txt, B_<seedB>.txt, REF_<seed>.txt where seedB = seed+1.
"""

import argparse
import os
import sys
import subprocess
import numpy as np


DEFAULT_SIZES = [
	"12116,12116,12116",
	"17736,17736,17736",
	"21722,21722,21722",
	"25082,25082,25082",
	"28043,28043,28043",
]


def parse_size(s):
	parts = s.split(',')
	if len(parts) != 3:
		raise argparse.ArgumentTypeError(f"Invalid size triplet: {s}")
	return tuple(int(x) for x in parts)


def write_matrix_text(path, mat):
	rows, cols = mat.shape
	with open(path, 'w') as f:
		f.write(f"{rows} {cols}\n")
		# write row by row to avoid huge intermediate strings
		for r in range(rows):
			row = mat[r]
			f.write(' '.join(str(int(x)) for x in row.tolist()) + '\n')


def main():
	parser = argparse.ArgumentParser(description='Generate A/B inputs and reference product matrices')
	parser.add_argument('--sizes', nargs='*', type=parse_size, default=[parse_size(s) for s in DEFAULT_SIZES],
						help='Size triplets M,K,N (A is MxK, B is KxN). Provide multiple as separate args.')
	parser.add_argument('--instances', type=int, default=5, help='Number of randomized instances per class')
	parser.add_argument('--out-dir', default='scratch/references', help='Output directory for generated files')
	parser.add_argument('--min', type=int, default=0, help='Min random value (inclusive) for inputs')
	parser.add_argument('--max', type=int, default=15, help='Max random value (inclusive) for inputs')
	parser.add_argument('--seed', type=int, default=None, help='Optional master seed to make generation reproducible')
	parser.add_argument('--clip-ref', action='store_true', help='Clip reference outputs to uint16 (default behavior)')
	args = parser.parse_args()

	out_root = args.out_dir
	os.makedirs(out_root, exist_ok=True)

	rng = np.random.default_rng(args.seed)

	# locate generator script relative to repo root
	script_dir = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
	gen_script = os.path.join(repo_root, 'scripts', 'generate_random_matrix.py')
	if not os.path.isfile(gen_script):
		print(f"Generator script not found at {gen_script}", file=sys.stderr)
		sys.exit(1)

	for (m, k, n) in args.sizes:
		class_name = f"class_m-{m}_k-{k}_n-{n}"
		class_dir = os.path.join(out_root, class_name)
		os.makedirs(class_dir, exist_ok=True)
		print(f"Generating class {class_name} -> {class_dir}")

		for idx in range(args.instances):
			# generate a seed for this instance
			seedA = int(rng.integers(1, 2**31 - 1))
			seedB = seedA + 1

			A_file = os.path.join(class_dir, f"A_{seedA}.txt")
			B_file = os.path.join(class_dir, f"B_{seedB}.txt")
			REF_file = os.path.join(class_dir, f"REF_{seedA}.txt")

			print(f"  instance {idx+1}/{args.instances}: seedA={seedA} seedB={seedB}")

			# Use the project's generator script to create A and B with deterministic seeds
			print(f"    generating A (seed={seedA}) and B (seed={seedB}) using {gen_script}")
			try:
				subprocess.run([sys.executable, gen_script, str(m), str(k), A_file,
								'--min', str(args.min), '--max', str(args.max),
								'--density', '1.0', '--format', 'text', '--seed', str(seedA)],
							   check=True)

				subprocess.run([sys.executable, gen_script, str(k), str(n), B_file,
								'--min', str(args.min), '--max', str(args.max),
								'--density', '1.0', '--format', 'text', '--seed', str(seedB)],
							   check=True)
			except subprocess.CalledProcessError as e:
				print(f"Generator failed: {e}", file=sys.stderr)
				sys.exit(1)

			# We no longer compute references here; a separate script should compute REF_<seed>.txt
			print(f"    generated A -> {A_file}, B -> {B_file}")

	print(f"All samples generated under {out_root}")


if __name__ == '__main__':
	main()

