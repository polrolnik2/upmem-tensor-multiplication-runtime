#!/bin/bash
#SBATCH --job-name=weak_scaling_sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_generate_references.out
#SBATCH --error=logs/slurm_generate_references.err

module load mkl

pip install --user mpi4py

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch slurm_generate_references.sh <A_matrix_file> <B_matrix_file> <Q_output_file>" >&2
    exit 1
fi

A_MATRIX_FILE="$1"
B_MATRIX_FILE="$2"
Q_OUTPUT_FILE="$3"

REPO_ROOT="/mnt/storage_3/home/pawel.polrolniczak/pl0576-01/project_data/pim-matmul-benchmarks"
PY_SCRIPT="$REPO_ROOT/benchmarks/scripts/mpi_reference_generate.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "ERROR: Python generator not found at $PY_SCRIPT" >&2
    exit 1
fi

echo "Generating reference Q matrix for A='$A_MATRIX_FILE', B='$B_MATRIX_FILE', outputting to '$Q_OUTPUT_FILE'"
srun --mpi pmix python3 "$PY_SCRIPT" \
    "$A_MATRIX_FILE" \
    "$B_MATRIX_FILE" \
    "$Q_OUTPUT_FILE"