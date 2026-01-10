# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}"

# Resolve CMake build directory (default to 'build' subdirectory of ROOT)
CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR:-${ROOT}/build}"

mkdir -p "${ROOT}/scratch/runtime_logs/gemv_scaling/logs"

for dpus in 1 2 4 8 16 32 64; do
    log_dir="${ROOT}/scratch/runtime_logs/gemv_scaling/logs/dpus-${dpus}"
    mkdir -p "${log_dir}"
    for run in 1 2 3 4 5; do
        bash "${ROOT}/benchmarks/scripts/gemv_random_test.sh" 1024 8192 "${dpus}" > "${log_dir}/run_${run}.log" 2>&1
    done
done