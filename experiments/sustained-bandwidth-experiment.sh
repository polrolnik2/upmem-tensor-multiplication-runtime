#!/bin/bash

# Check arguments: samples directory, iterations, list of DPUs
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <samples_directory> <iterations> <dpu_values>"
    echo "Example: $0 ../scratch/references 4 100 200 400"
    exit 1
fi

SAMPLES_DIR="$1"
ITERATIONS="$2"
shift 2
DPU_VALUES=($@)

BENCH_BIN="./build/benchmarks/back_to_back_multiplication_benchmark"
if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: benchmark binary not found at $BENCH_BIN."
    exit 1
fi

# Collect subdirectories (each is one pair: A_matrix.txt, B_matrix.txt)
mapfile -t SAMPLE_DIRS < <(find "$SAMPLES_DIR" -maxdepth 1 -mindepth 1 -type d | sort)
if [ "${#SAMPLE_DIRS[@]}" -lt 2 ]; then
    echo "Error: need at least two sample subdirectories in $SAMPLES_DIR"
    exit 1
fi

OUTPUT_FILE="experiment_results.csv"
echo "SampleA,SampleB,DPU,Average_ms" > "$OUTPUT_FILE"

# Iterate over unique unordered pairs of sample subdirectories
for ((i=0; i<${#SAMPLE_DIRS[@]}-1; i++)); do
    for ((j=i+1; j<${#SAMPLE_DIRS[@]}; j++)); do
        PAIR_A="${SAMPLE_DIRS[i]}"
        PAIR_B="${SAMPLE_DIRS[j]}"

        A1="$PAIR_A/A_matrix.txt"
        B1="$PAIR_A/B_matrix.txt"
        A2="$PAIR_B/A_matrix.txt"
        B2="$PAIR_B/B_matrix.txt"

        for f in "$A1" "$B1" "$A2" "$B2"; do
            if [ ! -f "$f" ]; then
                echo "Warning: missing file $f, skipping pair $(basename "$PAIR_A") + $(basename "$PAIR_B")"
                continue 2
            fi
        done

        for DPU in "${DPU_VALUES[@]}"; do
            CMD=("$BENCH_BIN" "$A1" "$B1" "$A2" "$B2" --dpus "$DPU" --iterations "$ITERATIONS")
            OUTPUT=$("${CMD[@]}" 2>&1)

            AVG_MS=$(echo "$OUTPUT" | awk '/Average per iteration/ {print $4; exit}')
            if [ -z "$AVG_MS" ]; then
                echo "Warning: could not parse average from output for $(basename "$PAIR_A") + $(basename "$PAIR_B") at DPU $DPU" >&2
                echo "$OUTPUT" >&2
                continue
            fi

            echo "$(basename "$PAIR_A"),$(basename "$PAIR_B"),$DPU,$AVG_MS" >> "$OUTPUT_FILE"
        done
    done
done

echo "Experiment completed. Results saved to $OUTPUT_FILE."