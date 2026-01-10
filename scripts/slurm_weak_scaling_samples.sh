#!/bin/bash
#SBATCH --job-name=weak_scaling_sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/weak_scaling_sample.%j.out
#SBATCH --error=logs/weak_scaling_sample.%j.err

set -euo pipefail

# Accept parameters either via environment variables (preferred when using sbatch --export)
# or via positional arguments in the order: M K N [SEEDA] [OUT_DIR] [MIN] [MAX]

# Defaults
REPO_ROOT="/mnt/storage_3/home/pawel.polrolniczak/pl0576-01/project_data/pim-matmul-benchmarks"
OUT_DIR_DEFAULT="$REPO_ROOT/scratch/references"
MIN_DEFAULT="-15"
MAX_DEFAULT="15"

# If positional args are provided, use them to set env vars when missing
if [[ $# -ge 3 ]]; then
	export M="${M:-$1}"
	export K="${K:-$2}"
	export N="${N:-$3}"
	if [[ $# -ge 4 ]]; then export SEEDA="${SEEDA:-$4}"; fi
	if [[ $# -ge 5 ]]; then export OUT_DIR="${OUT_DIR:-$5}"; fi
	if [[ $# -ge 6 ]]; then export MIN="${MIN:-$6}"; fi
	if [[ $# -ge 7 ]]; then export MAX="${MAX:-$7}"; fi
fi

# Resolve values with defaults
M="${M:-}"
K="${K:-}"
N="${N:-}"
SEEDA="${SEEDA:-}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
MIN="${MIN:-$MIN_DEFAULT}"
MAX="${MAX:-$MAX_DEFAULT}"

if [[ -z "$M" || -z "$K" || -z "$N" ]]; then
	echo "ERROR: You must provide M, K, N as env vars or positional args." >&2
	echo "Usage: sbatch slurm_weak_scaling_samples.sh M K N [SEEDA] [OUT_DIR] [MIN] [MAX]" >&2
	exit 2
fi

mkdir -p "$REPO_ROOT/logs" "$OUT_DIR"

PY_SCRIPT="$REPO_ROOT/benchmarks/scripts/generate_weak_scaling_samples.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
	echo "ERROR: Python generator not found at $PY_SCRIPT" >&2
	exit 1
fi

SEEDA_ARG=()
if [[ -n "${SEEDA}" ]]; then
	SEEDA_ARG=(--seedA "${SEEDA}")
fi

echo "Running single-sample generation: M=$M K=$K N=$N SEEDA=${SEEDA:-<auto>} MIN=$MIN MAX=$MAX OUT_DIR=$OUT_DIR"

python3 "$PY_SCRIPT" \
	--m "$M" --k "$K" --n "$N" \
	"${SEEDA_ARG[@]}" \
	--min "$MIN" --max "$MAX" \
	--out-dir "$OUT_DIR"

echo "Done."