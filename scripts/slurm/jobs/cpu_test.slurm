#!/bin/bash
#SBATCH --job-name=dtwc-cpu-test
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/cpu_test_%j.out
#SBATCH --error=logs/cpu_test_%j.err
#
# DTWC++ CPU test suite — run on SLURM CPU partition
#
# Usage: sbatch scripts/slurm/jobs/cpu_test.slurm
#        Expects build-htc-cpu/ to exist (see scripts/slurm/build-arc.sh htc-cpu)

set -euo pipefail

# ── Portable scratch ───────────────────────────────────────────────────────
WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "${WORKDIR}"

# ── Module load with fallback ──────────────────────────────────────────────
module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true

# ── Diagnostics header ────────────────────────────────────────────────────
echo "================================================================"
echo "  DTWC++ CPU Test"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : ${SLURMD_NODENAME}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK}"
echo "  Date     : $(date -Iseconds)"
echo "  WORKDIR  : ${WORKDIR}"
echo "================================================================"

# ── OpenMP ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# ── Paths ──────────────────────────────────────────────────────────────────
DTWC_BIN="${SLURM_SUBMIT_DIR}/build-htc-cpu/bin/dtwc_cl"
# Data may be at sibling data/ dir (slurm_remote.sh upload layout) or inside src/
if [[ -d "${SLURM_SUBMIT_DIR}/../data/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/../data"
elif [[ -d "${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018"
else
    echo "ERROR: Test data not found. Run 'slurm_remote.sh upload' first." >&2
    exit 1
fi

if [[ ! -x "${DTWC_BIN}" ]]; then
    echo "ERROR: Binary not found: ${DTWC_BIN}" >&2
    exit 1
fi

echo "  Data:    ${DATA_DIR}"

# ── Copy data to scratch ──────────────────────────────────────────────────
cp -r "${DATA_DIR}/Coffee" "${WORKDIR}/Coffee"
cp -r "${DATA_DIR}/Beef"   "${WORKDIR}/Beef"

RESULTS_DIR="${WORKDIR}/results"
mkdir -p "${RESULTS_DIR}"

# ── Helper: write timing JSON ─────────────────────────────────────────────
write_timing() {
    local test_name="$1"
    local elapsed="$2"
    local exit_code="$3"
    local outfile="${RESULTS_DIR}/timing_${test_name}.json"
    cat > "${outfile}" <<JSONEOF
{
    "test": "${test_name}",
    "job_id": "${SLURM_JOB_ID}",
    "node": "${SLURMD_NODENAME}",
    "cpus": ${SLURM_CPUS_PER_TASK},
    "omp_threads": ${OMP_NUM_THREADS},
    "elapsed_s": ${elapsed},
    "exit_code": ${exit_code},
    "timestamp": "$(date -Iseconds)"
}
JSONEOF
    echo "Timing written: ${outfile}"
}

# ── Test 1: Coffee k=2 ───────────────────────────────────────────────────
echo ""
echo "── Test 1: Coffee k=2 ──────────────────────────────────────────"
T_START=$(date +%s.%N)
DTWC_EXIT=0
"${DTWC_BIN}" \
    --input "${WORKDIR}/Coffee/Coffee_TRAIN.tsv" \
    -k2 \
    --skip-cols 1 \
    --dtype float64 \
    --method pam \
    --verbose \
    --name coffee \
    --output "${RESULTS_DIR}/coffee_k2" \
    || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "coffee_k2_cpu" "${ELAPSED}" "${DTWC_EXIT}"
echo "Coffee k=2 completed in ${ELAPSED}s (exit ${DTWC_EXIT})"

# ── Test 2: Beef k=5 ─────────────────────────────────────────────────────
echo ""
echo "── Test 2: Beef k=5 ────────────────────────────────────────────"
T_START=$(date +%s.%N)
DTWC_EXIT=0
"${DTWC_BIN}" \
    --input "${WORKDIR}/Beef/Beef_TRAIN.tsv" \
    -k5 \
    --skip-cols 1 \
    --dtype float64 \
    --method pam \
    --verbose \
    --name beef \
    --output "${RESULTS_DIR}/beef_k5" \
    || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "beef_k5_cpu" "${ELAPSED}" "${DTWC_EXIT}"
echo "Beef k=5 completed in ${ELAPSED}s (exit ${DTWC_EXIT})"

# ── Copy results back ────────────────────────────────────────────────────
echo ""
echo "── Copying results to submit directory ─────────────────────────"
mkdir -p "${SLURM_SUBMIT_DIR}/results/cpu_test_${SLURM_JOB_ID}"
cp -r "${RESULTS_DIR}/." "${SLURM_SUBMIT_DIR}/results/cpu_test_${SLURM_JOB_ID}/"

echo ""
echo "================================================================"
echo "  CPU test complete — $(date -Iseconds)"
echo "  Results: results/cpu_test_${SLURM_JOB_ID}/"
echo "================================================================"
