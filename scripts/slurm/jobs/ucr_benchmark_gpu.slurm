#!/bin/bash
#SBATCH --job-name=dtwc-ucr-gpu
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/ucr_benchmark_gpu_%j.out
#SBATCH --error=logs/ucr_benchmark_gpu_%j.err
#
# DTWC++ full UCR benchmark — GPU (Lloyd k-medoids, CUDA distance matrix)
#
# Runs all 128 UCR Archive datasets with k = number of classes.
# Distance matrix computed on GPU, clustering on CPU.
# Saves labels, medoids, silhouettes, and distance matrices per dataset.
#
# Usage: sbatch scripts/slurm/jobs/ucr_benchmark_gpu.slurm
#        GPU_TYPE=a100 sbatch --gres=gpu:a100:1 scripts/slurm/jobs/ucr_benchmark_gpu.slurm
#
# Environment variables:
#   GPU_PRECISION  — f32 or f64 (default: f64)
#   DTWC_TIMEOUT   — per-dataset timeout in seconds (default: 1800)

set -uo pipefail

TIMESTAMP_START=$(date -Iseconds)

# ── Portable scratch ───────────────────────────────────────────────────────
WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "${WORKDIR}"

# ── Module load ───────────────────────────────────────────────────────────
module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
module load CUDA/12.4.0 2>/dev/null || module load cuda 2>/dev/null || true

# ── OpenMP ────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ── GPU precision ─────────────────────────────────────────────────────────
GPU_PREC="${GPU_PRECISION:-f64}"

# ── Hardware detection ────────────────────────────────────────────────────
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

# ── Diagnostics header ───────────────────────────────────────────────────
echo "================================================================"
echo "  DTWC++ UCR Benchmark — GPU"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : ${SLURMD_NODENAME}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK}"
echo "  GPU      : ${GPU_NAME} (${GPU_MEMORY})"
echo "  Precision: ${GPU_PREC}"
echo "  CPU      : ${CPU_MODEL}"
echo "  Date     : $(date -Iseconds)"
echo "  WORKDIR  : ${WORKDIR}"
echo "================================================================"
echo ""
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

# ── Paths ─────────────────────────────────────────────────────────────────
DTWC_BIN="${SLURM_SUBMIT_DIR}/build-htc-gpu/bin/dtwc_cl"
if [[ -d "${SLURM_SUBMIT_DIR}/../data/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/../data"
elif [[ -d "${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018"
else
    echo "ERROR: UCR data not found." >&2; exit 1
fi

CSV_FILE="${DATA_DIR}/../UCR_DataSummary.csv"
if [[ ! -f "${CSV_FILE}" ]]; then
    CSV_FILE="${DATA_DIR}/UCR_DataSummary.csv"
fi
if [[ ! -f "${CSV_FILE}" ]]; then
    CSV_FILE="${SLURM_SUBMIT_DIR}/data/benchmark/UCR_DataSummary.csv"
fi

if [[ ! -x "${DTWC_BIN}" ]]; then
    echo "ERROR: Binary not found: ${DTWC_BIN}" >&2; exit 1
fi
if [[ ! -f "${CSV_FILE}" ]]; then
    echo "ERROR: UCR_DataSummary.csv not found." >&2; exit 1
fi

echo "  Binary : ${DTWC_BIN}"
echo "  Data   : ${DATA_DIR}"
echo "  CSV    : ${CSV_FILE}"

# Ensure logs directory exists (SLURM fails silently without it)
mkdir -p "${SLURM_SUBMIT_DIR}/logs" 2>/dev/null || true

# Strip UTF-8 BOM from CSV if present
CSV_CLEAN="${WORKDIR}/UCR_DataSummary_clean.csv"
sed '1s/^\xEF\xBB\xBF//' "${CSV_FILE}" > "${CSV_CLEAN}"

RESULTS_DIR="${WORKDIR}/results"
mkdir -p "${RESULTS_DIR}"

# ── Per-dataset timeout (seconds) ────────────────────────────────────────
DATASET_TIMEOUT="${DTWC_TIMEOUT:-1800}"

# ── Counters ──────────────────────────────────────────────────────────────
TOTAL=0
PASS=0
FAIL=0
SKIP=0
TIMEOUT_COUNT=0

# ── JSON string escaper ──────────────────────────────────────────────────
json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

# ── Parse results from CLI stdout ────────────────────────────────────────
parse_results() {
    local logfile="$1"
    RESULT_COST=$(grep -oP 'Total cost:\s*\K[\d.e+-]+' "$logfile" 2>/dev/null || echo "null")
    RESULT_CONVERGED=$(grep -oP 'Converged:\s*\K\w+' "$logfile" 2>/dev/null || echo "unknown")
    RESULT_ITERATIONS=$(grep -oP 'Iterations:\s*\K\d+' "$logfile" 2>/dev/null || echo "0")
    RESULT_SILHOUETTE=$(grep -oP 'mean=\K-?[\d.]+(?:[eE][+-]?\d+)?' "$logfile" 2>/dev/null || echo "null")
    # GPU-specific: extract GPU distance matrix time
    RESULT_GPU_DTW_MS=$(grep -oP 'GPU distance matrix:.*?in \K[\d.]+(?:[eE][+-]?\d+)?' "$logfile" 2>/dev/null || echo "null")
}

# ── Write timing JSON ────────────────────────────────────────────────────
write_timing() {
    local dataset="$1" n_series="$2" length="$3" k="$4"
    local elapsed="$5" exit_code="$6" status="$7"
    local outfile="${RESULTS_DIR}/timing_${dataset}.json"

    cat > "${outfile}" <<JSONEOF
{
    "dataset": "${dataset}",
    "n_series": ${n_series},
    "series_length": "${length}",
    "k": ${k},
    "method": "kmedoids",
    "device": "cuda",
    "gpu_precision": "${GPU_PREC}",
    "dtype": "float64",
    "cost": ${RESULT_COST:-null},
    "converged": "${RESULT_CONVERGED:-unknown}",
    "iterations": ${RESULT_ITERATIONS:-0},
    "silhouette_mean": ${RESULT_SILHOUETTE:-null},
    "gpu_dtw_ms": ${RESULT_GPU_DTW_MS:-null},
    "elapsed_s": ${elapsed},
    "exit_code": ${exit_code},
    "status": "${status}",
    "job_id": "${SLURM_JOB_ID}",
    "node": "${SLURMD_NODENAME}",
    "cpu": "$(json_escape "${CPU_MODEL}")",
    "cpus": ${SLURM_CPUS_PER_TASK},
    "omp_threads": ${OMP_NUM_THREADS},
    "gpu": "$(json_escape "${GPU_NAME}")",
    "gpu_memory": "$(json_escape "${GPU_MEMORY}")",
    "timestamp": "$(date -Iseconds)"
}
JSONEOF
}

echo ""
echo "── Starting UCR GPU benchmark ────────────────────────────────────"
echo ""

# ── Iterate datasets from CSV ─────────────────────────────────────────────
while IFS=',' read -r id type name train test nclass length rest; do
    # Skip header
    [[ "$id" == "ID" ]] && continue

    # Trim whitespace
    name=$(echo "$name" | xargs)
    train=$(echo "$train" | xargs)
    nclass=$(echo "$nclass" | xargs)
    length=$(echo "$length" | xargs)

    # Validate
    [[ -z "$name" || -z "$nclass" ]] && continue
    [[ "$nclass" =~ ^[0-9]+$ ]] || continue
    [[ "$train" =~ ^[0-9]+$ ]] || train=0

    TOTAL=$((TOTAL + 1))

    # Find dataset directory and TRAIN file
    DATASET_DIR="${DATA_DIR}/${name}"
    TRAIN_FILE="${DATASET_DIR}/${name}_TRAIN.tsv"
    if [[ ! -f "${TRAIN_FILE}" ]]; then
        echo "[SKIP] ${name}: TRAIN file not found"
        SKIP=$((SKIP + 1))
        RESULT_COST="null"; RESULT_CONVERGED="skip"; RESULT_ITERATIONS="0"
        RESULT_SILHOUETTE="null"; RESULT_GPU_DTW_MS="null"
        write_timing "$name" "$train" "$length" "$nclass" "0" "-1" "skip"
        continue
    fi

    echo "── [${TOTAL}] ${name}: N=${train}, L=${length}, k=${nclass} ──"

    # Copy data to scratch
    cp -r "${DATASET_DIR}" "${WORKDIR}/${name}" 2>/dev/null

    # Output directory
    OUT_DIR="${RESULTS_DIR}/${name}"
    mkdir -p "${OUT_DIR}"

    # Run dtwc_cl with GPU + timeout
    LOGFILE="${RESULTS_DIR}/${name}_stdout.log"
    T_START=$(date +%s.%N)
    DTWC_EXIT=0

    timeout "${DATASET_TIMEOUT}" "${DTWC_BIN}" \
        --input "${WORKDIR}/${name}/${name}_TRAIN.tsv" \
        -k "${nclass}" \
        --skip-cols 1 \
        --dtype float64 \
        --method kmedoids \
        --device cuda \
        --gpu-precision "${GPU_PREC}" \
        --max-iter 100 \
        --verbose \
        --name "${name,,}" \
        --output "${OUT_DIR}" \
        > "${LOGFILE}" 2>&1 \
        || DTWC_EXIT=$?

    T_END=$(date +%s.%N)
    ELAPSED=$(echo "${T_END} - ${T_START}" | bc 2>/dev/null || echo "0")
    # Ensure leading zero for sub-second values (bc outputs .123 not 0.123)
    [[ "${ELAPSED}" == .* ]] && ELAPSED="0${ELAPSED}"

    # Determine status
    if [[ "${DTWC_EXIT}" -eq 124 ]]; then
        STATUS="timeout"
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        echo "  [TIMEOUT] ${name} after ${DATASET_TIMEOUT}s"
    elif [[ "${DTWC_EXIT}" -ne 0 ]]; then
        STATUS="error"
        FAIL=$((FAIL + 1))
        echo "  [FAIL] ${name} exit=${DTWC_EXIT} (${ELAPSED}s)"
    else
        STATUS="pass"
        PASS=$((PASS + 1))
        echo "  [PASS] ${name} (${ELAPSED}s)"
    fi

    # Parse results from log
    RESULT_COST="null"; RESULT_CONVERGED="unknown"; RESULT_ITERATIONS="0"
    RESULT_SILHOUETTE="null"; RESULT_GPU_DTW_MS="null"
    parse_results "${LOGFILE}"

    write_timing "$name" "$train" "$length" "$nclass" "$ELAPSED" "$DTWC_EXIT" "$STATUS"

    # Clean scratch copy (keep results)
    rm -rf "${WORKDIR}/${name}"

done < "${CSV_CLEAN}"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  UCR GPU Benchmark Summary"
echo "  GPU: ${GPU_NAME} (${GPU_PREC})"
echo "  Total: ${TOTAL}  Pass: ${PASS}  Fail: ${FAIL}  Skip: ${SKIP}  Timeout: ${TIMEOUT_COUNT}"
echo "  Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "  Date: $(date -Iseconds)"
echo "================================================================"

# ── Write summary JSON ───────────────────────────────────────────────────
cat > "${RESULTS_DIR}/benchmark_meta.json" <<JSONEOF
{
    "benchmark": "ucr_full",
    "device": "cuda",
    "gpu_precision": "${GPU_PREC}",
    "method": "kmedoids",
    "dtype": "float64",
    "job_id": "${SLURM_JOB_ID}",
    "node": "${SLURMD_NODENAME}",
    "cpu": "$(json_escape "${CPU_MODEL}")",
    "cpus": ${SLURM_CPUS_PER_TASK},
    "omp_threads": ${OMP_NUM_THREADS},
    "gpu": "$(json_escape "${GPU_NAME}")",
    "gpu_memory": "$(json_escape "${GPU_MEMORY}")",
    "total": ${TOTAL},
    "pass": ${PASS},
    "fail": ${FAIL},
    "skip": ${SKIP},
    "timeout": ${TIMEOUT_COUNT},
    "dataset_timeout_s": ${DATASET_TIMEOUT},
    "timestamp_start": "${TIMESTAMP_START}",
    "timestamp_end": "$(date -Iseconds)"
}
JSONEOF

# ── Copy results back ───────────────────────────────────────────────────
echo "Copying results to submit directory..."
GPU_TAG=$(echo "${GPU_NAME}" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
FINAL_DIR="${SLURM_SUBMIT_DIR}/results/ucr_benchmark_gpu_${GPU_TAG}_${GPU_PREC}_${SLURM_JOB_ID}"
mkdir -p "${FINAL_DIR}"
cp -r "${RESULTS_DIR}/." "${FINAL_DIR}/"

echo "Results: ${FINAL_DIR##*/}/"
echo "Done."
