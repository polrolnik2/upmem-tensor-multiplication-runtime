# Resolve project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Resolve CMake build directory (default to 'build' subdirectory of ROOT)
CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR:-${ROOT}/build}"

m=$1
n=$2
dpus=$3

mkdir -p ${ROOT}/scratch/gemv_random_test

python3 ${ROOT}/scripts/generate_random_matrix.py ${n} ${m} ${ROOT}/scratch/gemv_random_test/A.txt \
  --min 0 \
  --max 50


python3 ${ROOT}/scripts/generate_random_matrix.py ${m} 1 ${ROOT}/scratch/gemv_random_test/B.txt \
  --min 0 \
  --max 50

python3 ${ROOT}/benchmarks/scripts/mpi_reference_generate.py \
    ${ROOT}/scratch/gemv_random_test/A.txt ${ROOT}/scratch/gemv_random_test/B.txt \
    ${ROOT}/scratch/gemv_random_test/C_ref.txt

${CMAKE_BINARY_DIR}/bin/test_from_file ${ROOT}/scratch/gemv_random_test/A.txt ${ROOT}/scratch/gemv_random_test/B.txt --reference-file ${ROOT}/scratch/gemv_random_test/C_ref.txt --dpus ${dpus}