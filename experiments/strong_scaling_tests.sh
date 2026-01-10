#!/usr/bin/env bash

# Strong scaling test harness for test_from_file.
# - Compiles the benchmark locally using the repo makefile.
# - For each pair directory under scratch/references, runs the test with
#   several DPU counts and records elapsed times and errors per-DPU.
# - Output:
#   scratch/runtime_logs/strong_scaling/times_dpus-<N>.csv  (pair,elapsed_sec,exit_code)
#   scratch/runtime_logs/strong_scaling/errors_dpus-<N>.log (stderr output per run)

set -euo pipefail

# Resolve project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Resolve CMake build directory (default to 'build' subdirectory of ROOT)
CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR:-${ROOT}/build}"

# DPU counts to test
DPU_COUNTS=(1 25 100 225 400 484)

LOG_DIR_HOST="${ROOT}/scratch/runtime_logs/strong_scaling"
BIN_HOST="${CMAKE_BINARY_DIR}/bin/test_from_file"
REF_DIR_BASE="${ROOT}/scratch/references"

mkdir -p "${LOG_DIR_HOST}"

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
	ls -l "${BIN_HOST}" || { echo "[ERROR] Build completed but binary not found at ${BIN_HOST}"; exit 1; }
}

# Run one pair with specified dpus; append results to per-dpu logs.
run_one_pair() {
	local dpus="$1"
	local pair_dir="$2"
	local pair_name
	pair_name="$(basename "${pair_dir}")"

	local A_FILE="${pair_dir}/A_matrix.txt"
	local B_FILE="${pair_dir}/B_matrix.txt"
	local Q_FILE="${pair_dir}/Q.txt"

	local times_file="${LOG_DIR_HOST}/times_dpus-${dpus}.csv"
	local errors_file="${LOG_DIR_HOST}/errors_dpus-${dpus}.log"

	mkdir -p "${LOG_DIR_HOST}"
	# Ensure times file has header
	if [[ ! -f "${times_file}" ]]; then
		echo "pair,elapsed_sec,exit_code" > "${times_file}"
	fi

	if [[ ! -f "${A_FILE}" || ! -f "${B_FILE}" || ! -f "${Q_FILE}" ]]; then
		echo "[WARN] Missing files in ${pair_dir}; expected A_matrix.txt, B_matrix.txt, Q.txt" >> "${errors_file}"
		echo "${pair_name},,1" >> "${times_file}"
		return
	fi

	echo "[INFO] Running pair=${pair_name} dpus=${dpus}"

	perrun_dir="${LOG_DIR_HOST}/per_run_output/dpus-${dpus}"
	mkdir -p "${perrun_dir}"
	perrun_log="${LOG_DIR_HOST}/per_run_output/dpus-${dpus}-pair-${pair_name}.log"

	# allow the program to fail without causing the script to exit
	set +e
	"${BIN_HOST}" "${A_FILE}" "${B_FILE}" --dpus "${dpus}" --reference-file "${Q_FILE}" 1>${perrun_log} 2>>"${errors_file}"
	rc=$?
	set -e

	# extract the program's stderr block for this run (we wrote a 'command:' header just above)
	# find last occurrence of 'command:' and save following lines (program stderr) to a per-run file
	last_cmd_line=$(grep -n '^command:' "${errors_file}" | tail -n 1 | cut -d: -f1 || true)
	perrun_file="${perrun_dir}/${pair_name}.stderr"
	if [[ -n "${last_cmd_line}" ]]; then
		start_line=$((last_cmd_line + 1))
		sed -n "${start_line},\$p" "${errors_file}" > "${perrun_file}" || true
	else
		: > "${perrun_file}"
	fi

	# Parse a time-like line from the program stderr (if any) and record to a per-dpu CSV
	program_times_file="${LOG_DIR_HOST}/program_times_dpus-${dpus}.csv"
	if [[ ! -f "${program_times_file}" ]]; then
		echo "pair,program_time_line" > "${program_times_file}"
	fi
	# look for common time keywords or units (case-insensitive) and take the first match
	time_line=$(grep -iE 'time|elapsed|ms|msec|sec|s\b' "${perrun_file}" | sed -n '1p' || true)
	# escape double quotes to keep CSV valid
	time_line_esc=$(printf '%s' "${time_line}" | sed 's/"/""/g')
	echo "${pair_name},\"${time_line_esc}\"" >> "${program_times_file}"
}

main() {
	compile_locally

	# iterate dpus then pairs to produce per-dpu CSVs
	for dpus in "${DPU_COUNTS[@]}"; do
		# clear/create errors file for this dpus (so each dpus has a single aggregated log)
		touch "${LOG_DIR_HOST}/errors_dpus-${dpus}.log"
		# iterate pair directories
		for pair_dir in "${REF_DIR_BASE}"/*/; do
			# if no dirs match, skip
			[[ -d "${pair_dir}" ]] || continue
			run_one_pair "${dpus}" "${pair_dir}"
		done
		echo "[INFO] Completed dpus=${dpus}. Times: ${LOG_DIR_HOST}/times_dpus-${dpus}.csv Errors: ${LOG_DIR_HOST}/errors_dpus-${dpus}.log"
	done

	echo "[INFO] Strong-scaling runs completed. Logs in ${LOG_DIR_HOST}"
}

main "$@"
