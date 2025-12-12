mkdir -p scratch/runtime_logs/gemv_scaling/logs

for dpus in 1 2 4 8 16 32 64; do
    log_dir="scratch/runtime_logs/gemv_scaling/logs/dpus-${dpus}"
    mkdir -p "${log_dir}"
    for run in 1 2 3 4 5; do
        bash benchmarks/scripts/gemv_random_test.sh 1024 8192 "${dpus}" > "${log_dir}/run_${run}.log" 2>&1
    done
done