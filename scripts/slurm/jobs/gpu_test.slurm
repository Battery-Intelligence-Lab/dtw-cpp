#!/bin/bash
#SBATCH --job-name=dtwc-gpu-test
#SBATCH --partition=short
#SBATCH --clusters=htc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err
#
# DTWC++ GPU test suite — run on SLURM GPU partition
#
# Usage: sbatch scripts/slurm/jobs/gpu_test.slurm
#        Expects build-htc-gpu/ to exist (see scripts/slurm/build-arc.sh htc-gpu)

set -euo pipefail

# ── Portable scratch ───────────────────────────────────────────────────────
WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "${WORKDIR}"

# ── Module load with fallback ──────────────────────────────────────────────
module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
module load CUDA/12.4.0 2>/dev/null || module load cuda 2>/dev/null || true

# ── Diagnostics header ────────────────────────────────────────────────────
echo "================================================================"
echo "  DTWC++ GPU Test"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : ${SLURMD_NODENAME}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK}"
echo "  GPU      : ${SLURM_JOB_GPUS:-unknown}"
echo "  Date     : $(date -Iseconds)"
echo "  WORKDIR  : ${WORKDIR}"
echo "================================================================"
echo ""
echo "── nvidia-smi ──────────────────────────────────────────────────"
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

# ── OpenMP ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# ── Paths ──────────────────────────────────────────────────────────────────
DTWC_BIN="${SLURM_SUBMIT_DIR}/build-htc-gpu/bin/dtwc_cl"
if [[ -d "${SLURM_SUBMIT_DIR}/../data/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/../data"
elif [[ -d "${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018"
else
    echo "ERROR: Test data not found." >&2; exit 1
fi

if [[ ! -x "${DTWC_BIN}" ]]; then
    echo "ERROR: Binary not found: ${DTWC_BIN}" >&2
    exit 1
fi

# ── Copy data to scratch ──────────────────────────────────────────────────
cp -r "${DATA_DIR}/Coffee" "${WORKDIR}/Coffee"

RESULTS_DIR="${WORKDIR}/results"
mkdir -p "${RESULTS_DIR}"

# ── Helper: write timing JSON ─────────────────────────────────────────────
write_timing() {
    local test_name="$1"
    local elapsed="$2"
    local exit_code="$3"
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    local outfile="${RESULTS_DIR}/timing_${test_name}.json"
    cat > "${outfile}" <<JSONEOF
{
    "test": "${test_name}",
    "job_id": "${SLURM_JOB_ID}",
    "node": "${SLURMD_NODENAME}",
    "cpus": ${SLURM_CPUS_PER_TASK},
    "gpu": "${gpu_name}",
    "elapsed_s": ${elapsed},
    "exit_code": ${exit_code},
    "timestamp": "$(date -Iseconds)"
}
JSONEOF
    echo "Timing written: ${outfile}"
}

# ── Test 1: Coffee k=2, f32 precision ────────────────────────────────────
echo ""
echo "── Test 1: Coffee k=2, CUDA f32 ───────────────────────────────"
T_START=$(date +%s.%N)
DTWC_EXIT=0
"${DTWC_BIN}" \
    --input "${WORKDIR}/Coffee/Coffee_TRAIN.tsv" \
    -k2 \
    --skip-cols 1 \
    --dtype float64 \
    --method pam \
    --device cuda \
    --gpu-precision f32 \
    --verbose \
    --name coffee_f32 \
    --output "${RESULTS_DIR}/coffee_k2_f32" \
    || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "coffee_k2_cuda_f32" "${ELAPSED}" "${DTWC_EXIT}"
echo "Coffee k=2 f32 completed in ${ELAPSED}s (exit ${DTWC_EXIT})"

# ── Test 2: Coffee k=2, f64 precision ────────────────────────────────────
echo ""
echo "── Test 2: Coffee k=2, CUDA f64 ───────────────────────────────"
T_START=$(date +%s.%N)
DTWC_EXIT=0
"${DTWC_BIN}" \
    --input "${WORKDIR}/Coffee/Coffee_TRAIN.tsv" \
    -k2 \
    --skip-cols 1 \
    --dtype float64 \
    --method pam \
    --device cuda \
    --gpu-precision f64 \
    --verbose \
    --name coffee_f64 \
    --output "${RESULTS_DIR}/coffee_k2_f64" \
    || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "coffee_k2_cuda_f64" "${ELAPSED}" "${DTWC_EXIT}"
echo "Coffee k=2 f64 completed in ${ELAPSED}s (exit ${DTWC_EXIT})"

# ── Copy results back ────────────────────────────────────────────────────
echo ""
echo "── Copying results to submit directory ─────────────────────────"
mkdir -p "${SLURM_SUBMIT_DIR}/results/gpu_test_${SLURM_JOB_ID}"
cp -r "${RESULTS_DIR}/." "${SLURM_SUBMIT_DIR}/results/gpu_test_${SLURM_JOB_ID}/"

echo ""
echo "================================================================"
echo "  GPU test complete — $(date -Iseconds)"
echo "  Results: results/gpu_test_${SLURM_JOB_ID}/"
echo "================================================================"
