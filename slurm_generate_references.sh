#!/bin/bash
#SBATCH --job-name=weak_scaling_sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=/home/inf164175/workspace/pim-matmul-benchmarks/logs/slurm_generate_references.out
#SBATCH --error=/home/inf164175/workspace/pim-matmul-benchmarks/logs/slurm_generate_references.err

source ~/miniconda3/bin/activate

conda activate cublas

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch slurm_generate_references.sh <A_matrix_file> <B_matrix_file> <Q_output_file>" >&2
    exit 1
fi

A_MATRIX_FILE="$1"
B_MATRIX_FILE="$2"
Q_OUTPUT_FILE="$3"

REPO_ROOT="/home/inf164175/workspace/pim-matmul-benchmarks/"
PY_SCRIPT="$REPO_ROOT/benchmarks/scripts/cuda_reference_generate.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "ERROR: Python generator not found at $PY_SCRIPT" >&2
    exit 1
fi

echo "Generating reference Q matrix for A='$A_MATRIX_FILE', B='$B_MATRIX_FILE', outputting to '$Q_OUTPUT_FILE'"
conda run -n cublas python3 "$PY_SCRIPT" \
    "$A_MATRIX_FILE" \
    "$B_MATRIX_FILE" \
    "$Q_OUTPUT_FILE"