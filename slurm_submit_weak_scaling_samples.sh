#!/bin/bash
# Orchestrate submission of many weak-scaling sample generation jobs.
# Each job generates exactly one pair of matrices (A and B) for a single (M,K,N) and seedA.
#
# Usage examples:
#   ./slurm_submit_weak_scaling_samples.sh \
#       --instances 2 \
#       --sizes "3965,3965,3965" "11675,11675,11675" \
#       --min -15 --max 15 \
#       --out-dir scratch/references
#
#   # Use a different base seed to vary the generated seeds
#   ./slurm_submit_weak_scaling_samples.sh --instances 5 --base-seed 123456
#
# Notes:
# - Seeds are generated deterministically as seedA = BASE_SEED + job_index*2 + 1
# - The per-job script is slurm_weak_scaling_samples.sh in the repo root.

set -euo pipefail

REPO_ROOT="/mnt/storage_3/home/pawel.polrolniczak/pl0576-01/project_data/pim-matmul-benchmarks"
JOB_SCRIPT="$REPO_ROOT/slurm_weak_scaling_samples.sh"

# Defaults (mirrors DEFAULT_SIZES in the Python script)
SIZES_DEFAULT=(
  "3965,3965,3965"
  "11675,11675,11675"
  "17438,17438,17438"
  "21673,21673,21673"
  "25082,25082,25082"
  "28043,28043,28043"
)
INSTANCES=5
OUT_DIR="$REPO_ROOT/scratch/references"
MIN=-15
MAX=15
BASE_SEED=10001
PARTITION=""
EXTRA_SBATCH_ARGS=()

# Parse args
SIZES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sizes)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SIZES+=("$1")
        shift
      done
      ;;
    --instances)
      INSTANCES="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --min)
      MIN="$2"; shift 2 ;;
    --max)
      MAX="$2"; shift 2 ;;
    --base-seed)
      BASE_SEED="$2"; shift 2 ;;
    --partition)
      PARTITION="$2"; shift 2 ;;
    --sbatch-arg)
      EXTRA_SBATCH_ARGS+=("$2"); shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--sizes M,K,N ...] [--instances N] [--out-dir DIR] [--min MIN] [--max MAX] [--base-seed S] [--partition P] [--sbatch-arg \"--constraint=...\"]";
      exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ${#SIZES[@]} -eq 0 ]]; then
  SIZES=("${SIZES_DEFAULT[@]}")
fi

if [[ ! -x "$JOB_SCRIPT" ]]; then
  # Not executable is fine; ensure it exists
  if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Per-job script not found at $JOB_SCRIPT" >&2
    exit 1
  fi
fi

mkdir -p "$REPO_ROOT/logs" "$OUT_DIR"

job_index=0
submit_count=0
for triplet in "${SIZES[@]}"; do
  IFS=',' read -r M K N <<< "$triplet"
  if [[ -z "$M" || -z "$K" || -z "$N" ]]; then
    echo "Invalid size triplet: $triplet" >&2
    exit 2
  fi
  for ((i=1; i<=INSTANCES; i++)); do
    SEEDA=$(( BASE_SEED + job_index*2 + 1 ))
    args=("--export=ALL,M=$M,K=$K,N=$N,SEEDA=$SEEDA,OUT_DIR=$OUT_DIR,MIN=$MIN,MAX=$MAX")
    if [[ -n "$PARTITION" ]]; then
      args+=("-p" "$PARTITION")
    fi
    if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
      args+=("${EXTRA_SBATCH_ARGS[@]}")
    fi

    echo "Submitting job $((job_index+1)): M=$M K=$K N=$N seedA=$SEEDA"
    sbatch "${args[@]}" "$JOB_SCRIPT"
    submit_count=$((submit_count+1))
    job_index=$((job_index+1))
  done
done

echo "Submitted $submit_count jobs. Outputs will appear under $OUT_DIR and logs under $REPO_ROOT/logs."
