#!/usr/bin/env python3
"""
Script to generate a random integer matrix with specified dimensions, value range, and density.
Outputs the matrix to a file in a format compatible with the PIM matrix multiplication benchmarks.
"""

import argparse
import numpy as np
import sys


def generate_random_matrix(rows, cols, min_val, max_val, density):
    """
    Generate a random integer matrix.
    
    Args:
        rows: Number of rows in the matrix
        cols: Number of columns in the matrix
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        density: Fraction of non-zero elements (0.0 to 1.0)
    
    Returns:
        A numpy array containing the random matrix
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError("Density must be between 0.0 and 1.0")
    
    # Generate random integer matrix
    matrix = np.random.randint(min_val, max_val + 1, size=(rows, cols), dtype=np.int32)
    
    # Apply density (sparsity) if less than 1.0
    if density < 1.0:
        # Create a mask for zero elements
        mask = np.random.random((rows, cols)) < density
        matrix = matrix * mask
    
    return matrix


def write_matrix_to_file(matrix, output_file, format_type='text'):
    """
    Write matrix to file.
    
    Args:
        matrix: The numpy array to write
        output_file: Path to output file
        format_type: Format for output ('text', 'binary', 'csv')
    """
    if format_type == 'text':
        # Write in plain text format with dimensions
        with open(output_file, 'w') as f:
            rows, cols = matrix.shape
            f.write(f"{rows} {cols}\n")
            for row in matrix:
                f.write(' '.join(map(str, row)) + '\n')
    
    elif format_type == 'binary':
        # Write in binary format
        with open(output_file, 'wb') as f:
            # Write dimensions as int32
            np.array([matrix.shape[0], matrix.shape[1]], dtype=np.int32).tofile(f)
            # Write matrix data
            matrix.astype(np.int32).tofile(f)
    
    elif format_type == 'csv':
        # Write as CSV
        np.savetxt(output_file, matrix, delimiter=',', fmt='%d')
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a random integer matrix with specified properties',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('rows', type=int, help='Number of rows')
    parser.add_argument('cols', type=int, help='Number of columns')
    parser.add_argument('output_file', type=str, help='Output file path')
    
    parser.add_argument('--min', type=int, default=0, 
                        help='Minimum value for matrix elements')
    parser.add_argument('--max', type=int, default=100,
                        help='Maximum value for matrix elements')
    parser.add_argument('--density', type=float, default=1.0,
                        help='Density of non-zero elements (0.0 to 1.0)')
    parser.add_argument('--format', type=str, default='text',
                        choices=['text', 'binary', 'csv'],
                        help='Output format')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rows <= 0 or args.cols <= 0:
        print("Error: Rows and columns must be positive integers", file=sys.stderr)
        sys.exit(1)
    
    if args.min > args.max:
        print("Error: Minimum value cannot be greater than maximum value", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Generate matrix
    print(f"Generating {args.rows}x{args.cols} matrix...")
    print(f"Value range: [{args.min}, {args.max}]")
    print(f"Density: {args.density}")
    
    matrix = generate_random_matrix(args.rows, args.cols, args.min, args.max, args.density)
    
    # Write to file
    write_matrix_to_file(matrix, args.output_file, args.format)
    
    # Print statistics
    non_zero_count = np.count_nonzero(matrix)
    total_elements = args.rows * args.cols
    actual_density = non_zero_count / total_elements
    
    print(f"Matrix generated successfully!")
    print(f"Output file: {args.output_file}")
    print(f"Format: {args.format}")
    print(f"Non-zero elements: {non_zero_count}/{total_elements} ({actual_density:.2%})")
    print(f"Min value in matrix: {matrix.min()}")
    print(f"Max value in matrix: {matrix.max()}")
    print(f"Mean value: {matrix.mean():.2f}")


if __name__ == '__main__':
    main()
