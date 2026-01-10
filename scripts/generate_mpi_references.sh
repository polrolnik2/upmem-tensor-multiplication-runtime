#!/bin/bash

set -euo pipefail

REF_DIR=${1:-scratch/references}

mkdir -p "$REF_DIR/results"

if [ ! -d "$REF_DIR" ]; then
    printf 'Reference directory not found: %s\n' "$REF_DIR" >&2
    exit 1
fi

shopt -s nullglob

for class_dir in "$REF_DIR"/*; do
    [ -d "$class_dir" ] || continue

    # collect A and B matrix files (case-insensitive match for 'A' and 'B' in filename)
    mapfile -t A_files < <(find "$class_dir" -maxdepth 1 -type f -iname '*A*' | sort)
    mapfile -t B_files < <(find "$class_dir" -maxdepth 1 -type f -iname '*B*' | sort)

    lenA=${#A_files[@]}
    lenB=${#B_files[@]}
    n=$(( lenA < lenB ? lenA : lenB ))

    mkdir -p "$REF_DIR/results/$(basename "$class_dir")"
    
    for i in $(seq 0 $((n - 1))); do
        mkdir -p "$REF_DIR/results/$(basename "$class_dir")/pair_$((i + 1))"
        ln -f "${A_files[i]}" "$REF_DIR/results/$(basename "$class_dir")/pair_$((i + 1))/A_matrix.txt"
        ln -f "${B_files[i]}" "$REF_DIR/results/$(basename "$class_dir")/pair_$((i + 1))/B_matrix.txt"
        sbatch -p hgx slurm_generate_references.sh \
            "$REF_DIR/results/$(basename "$class_dir")/pair_$((i + 1))/A_matrix.txt" \
            "$REF_DIR/results/$(basename "$class_dir")/pair_$((i + 1))/B_matrix.txt" \
            "$REF_DIR/results/$(basename "$class_dir")/pair_$((i + 1))/Q.txt"
    done
done