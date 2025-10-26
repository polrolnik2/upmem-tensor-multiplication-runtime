#!/usr/bin/env bash

# Weak scaling test harness for test_from_file.
# - Compiles the benchmark locally using the repo makefile.
# - For each class (DPU count, matrix sizes), runs multiple randomized instances.
# - Logs seed, command output, and metadata per class.
# - Cleans up generated input files after each run.

set -euo pipefail

# Resolve project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

POSSIBLE_UPMEM_ENVS=(
	"/opt/upmem-2025.1.0-Linux-x86_64/upmem_env.sh"
	"/opt/upmem-2024.3.0-Linux-x86_64/upmem_env.sh"
)

# Configuration: edit these arrays to define classes
# DPU counts to test
DPU_COUNTS=(1 25 100 225 400 625)
# Matrix size triplets: "M,K,N" means A is MxK, B is KxN
SIZE_CLASSES=(
	"3965,3965,3965"
	"11675,11675,11675"
	"17438,17438,17438"
	"21673,21673,21673"
	"25082,25082,25082"
	"28043,28043,28043"
)
# Number of randomized instances per class
INSTANCES=5

# Value range for generated matrices (uint8 inputs)
GEN_MIN=0
GEN_MAX=15

LOG_DIR_HOST="${ROOT}/scratch/runtime_logs/test_from_file"
BIN_HOST="${ROOT}/bin/test_from_file"

mkdir -p "${LOG_DIR_HOST}"

random_seed() {
	# Try to read 4 bytes from urandom; fallback to time + $$ + RANDOM
	if command -v od >/dev/null 2>&1; then
		od -An -N4 -tu4 < /dev/urandom | tr -d ' ' || true
	else
		date +%s%N
	fi
}

source_env_if_available() {
	# Source UPMEM environment if found and not already configured.
	if ! command -v dpu-pkg-config >/dev/null 2>&1; then
		for envfile in "${POSSIBLE_UPMEM_ENVS[@]}"; do
			if [[ -f "${envfile}" ]]; then
				# Use simulator by default if script accepts it
				# shellcheck disable=SC1090
				. "${envfile}" simulator || true
				break
			fi
		done
	fi
	# Source project env if present
	if [[ -f "${ROOT}/source.me" ]]; then
		# shellcheck disable=SC1091
		. "${ROOT}/source.me" || true
	fi
}

compile_locally() {
	echo "[INFO] Compiling test_from_file locally..."
	export PIM_MATMUL_BENCHMARKS_ROOT="${ROOT}"
	if ! command -v dpu-pkg-config >/dev/null 2>&1; then
		echo "[WARN] dpu-pkg-config not found in PATH. Trying to source UPMEM environment..."
		source_env_if_available
	fi
	if ! command -v dpu-pkg-config >/dev/null 2>&1; then
		echo "[ERROR] dpu-pkg-config still not found. Please source the UPMEM SDK env before running."
		exit 1
	fi
	make -C "${ROOT}/benchmarks" compile FILE=test_from_file.c
	ls -l "${BIN_HOST}" || {
		echo "[ERROR] Build completed but binary not found at ${BIN_HOST}"; exit 1; }
}

run_one_instance() {
	local dpus="$1" mkn="$2" seedA="$3" idx="$4"
	local m k n
	IFS=',' read -r m k n <<< "${mkn}"

	local class_log_host="${LOG_DIR_HOST}/class_dpus-${dpus}_m-${m}_k-${k}_n-${n}.log"

	echo "[INFO] Running class dpus=${dpus} size=${m}x${k} * ${k}x${n} (instance ${idx}, seed=${seedA})"

	mkdir -p "${LOG_DIR_HOST}"
	local A_FILE="${LOG_DIR_HOST}/A_${seedA}.txt"
	local B_FILE="${LOG_DIR_HOST}/B_$((seedA+1)).txt"
	local seedB=$((seedA+1))

	{
		echo '---'
		echo "timestamp: $(date -Iseconds)"
		echo "dpus: ${dpus}"
		echo "sizes: A=${m}x${k} B=${k}x${n}"
		echo "seedA: ${seedA}"
		echo "seedB: ${seedB}"
		echo "generate_A_output:"
	} >> "${class_log_host}"

	python3 "${ROOT}/scripts/generate_random_matrix.py" "${m}" "${k}" "${A_FILE}" \
		--min "${GEN_MIN}" --max "${GEN_MAX}" --density 1.0 --format text --seed "${seedA}" \
		>> "${class_log_host}" 2>&1

	{
		echo "generate_B_output:"
	} >> "${class_log_host}"

	python3 "${ROOT}/scripts/generate_random_matrix.py" "${k}" "${n}" "${B_FILE}" \
		--min "${GEN_MIN}" --max "${GEN_MAX}" --density 1.0 --format text --seed "${seedB}" \
		>> "${class_log_host}" 2>&1

	{
		echo 'run_output:'
		echo "cmd: ${BIN_HOST} ${A_FILE} ${B_FILE} --dpus ${dpus}"
	} >> "${class_log_host}"

	set +e
	"${BIN_HOST}" "${A_FILE}" "${B_FILE}" --dpus "${dpus}" >> "${class_log_host}" 2>&1
	rc=$?
	set -e
	if [[ $rc -ne 0 ]]; then
		echo "exit_code: ${rc}" >> "${class_log_host}"
	else
		echo "exit_code: 0" >> "${class_log_host}"
	fi

	rm -f "${A_FILE}" "${B_FILE}"
}

main() {
	source_env_if_available
	compile_locally

	local idx
	for dpus in "${DPU_COUNTS[@]}"; do
		for mkn in "${SIZE_CLASSES[@]}"; do
			for ((idx=1; idx<=INSTANCES; idx++)); do
				seedA=$(random_seed)
				# fallback if empty
				if [[ -z "${seedA}" ]]; then seedA=$(date +%s%N); fi
				run_one_instance "${dpus}" "${mkn}" "${seedA}" "${idx}"
			done
		done
	done

	echo "[INFO] All tests completed. Logs in ${LOG_DIR_HOST}"
}

main "$@"
