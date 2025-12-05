#!/usr/bin/env bash
set -euo pipefail

# Usage: weak_scaling.sh [DIRECTORY]
# Loops through immediate subdirectories of DIRECTORY (default: current dir)

test1=("/home/unipoznan/workspace/pim-matmul-benchmarks/scratch/references/class_m-11675_k-11675_n-11675" 25)
test2=("/home/unipoznan/workspace/pim-matmul-benchmarks/scratch/references/class_m-17438_k-17438_n-17438" 100)
test3=("/home/unipoznan/workspace/pim-matmul-benchmarks/scratch/references/class_m-21673_k-21673_n-21673" 225)
test4=("/home/unipoznan/workspace/pim-matmul-benchmarks/scratch/references/class_m-25082_k-25082_n-25082" 400)

directory_list=(
    "${test1[*]}"
    "${test2[*]}"
    "${test3[*]}"
    "${test4[*]}"
)

for entry in "${directory_list[@]}"; do
    dir=$(echo $entry | awk '{print $1}')
    dpus=$(echo $entry | awk '{print $2}')
    ls $dir | \
    while read -r pair_dir; do
        full_pair_dir="${dir}/${pair_dir}"
        mkdir -p scratch/runtime_logs/weak_scaling/logs/dpus-"${dpus}"
        echo "Checking directory: $full_pair_dir"
        ./bin/test_from_file "${full_pair_dir}/A_matrix.txt" "${full_pair_dir}/B_matrix.txt" --dpus "${dpus}" --reference-file "${full_pair_dir}/Q.txt" > scratch/runtime_logs/weak_scaling/logs/dpus-"${dpus}"/"${pair_dir}".log 2>&1
    done
done