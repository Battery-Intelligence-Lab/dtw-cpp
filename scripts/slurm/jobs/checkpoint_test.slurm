#!/bin/bash
#SBATCH --job-name=dtwc-ckpt-test
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/checkpoint_test_%j.out
#SBATCH --error=logs/checkpoint_test_%j.err
#
# DTWC++ checkpoint/resume test — verify deterministic restart
#
# Usage: sbatch scripts/slurm/jobs/checkpoint_test.slurm
#        Expects build-htc-cpu/ to exist (see scripts/slurm/build-arc.sh htc-cpu)

set -euo pipefail

# ── Portable scratch ───────────────────────────────────────────────────────
WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "${WORKDIR}"

# ── Module load with fallback ──────────────────────────────────────────────
module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true

# ── Diagnostics header ────────────────────────────────────────────────────
echo "================================================================"
echo "  DTWC++ Checkpoint/Resume Test"
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
DATA_DIR="${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018"

if [[ ! -x "${DTWC_BIN}" ]]; then
    echo "ERROR: Binary not found: ${DTWC_BIN}" >&2
    exit 1
fi

# ── Copy data to scratch ──────────────────────────────────────────────────
cp -r "${DATA_DIR}/Coffee" "${WORKDIR}/Coffee"

RESULTS_DIR="${WORKDIR}/results"
CKPT_DIR="${WORKDIR}/ckpt"
mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}"

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
    "elapsed_s": ${elapsed},
    "exit_code": ${exit_code},
    "timestamp": "$(date -Iseconds)"
}
JSONEOF
    echo "Timing written: ${outfile}"
}

# ── Run 1: Fresh run with checkpoint ─────────────────────────────────────
echo ""
echo "── Run 1: Fresh run with checkpoint ────────────────────────────"
T_START=$(date +%s.%N)
DTWC_EXIT=0
"${DTWC_BIN}" \
    --input "${WORKDIR}/Coffee/Coffee_TRAIN.tsv" \
    --k 2 \
    --skip-cols 1 \
    --dtype float64 \
    --method pam \
    --verbose \
    --name coffee \
    --checkpoint "${CKPT_DIR}" \
    --output "${RESULTS_DIR}/ckpt_run1" \
    || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "checkpoint_run1" "${ELAPSED}" "${DTWC_EXIT}"
echo "Run 1 completed in ${ELAPSED}s (exit ${DTWC_EXIT})"

# ── Run 2: Resume from checkpoint ────────────────────────────────────────
echo ""
echo "── Run 2: Resume from checkpoint ───────────────────────────────"
T_START=$(date +%s.%N)
DTWC_EXIT=0
"${DTWC_BIN}" \
    --input "${WORKDIR}/Coffee/Coffee_TRAIN.tsv" \
    --k 2 \
    --skip-cols 1 \
    --dtype float64 \
    --method pam \
    --verbose \
    --name coffee \
    --resume \
    --checkpoint "${CKPT_DIR}" \
    --output "${RESULTS_DIR}/ckpt_run2" \
    || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "checkpoint_run2" "${ELAPSED}" "${DTWC_EXIT}"
echo "Run 2 completed in ${ELAPSED}s (exit ${DTWC_EXIT})"

# ── Compare labels ───────────────────────────────────────────────────────
echo ""
echo "── Comparing labels from Run 1 and Run 2 ──────────────────────"

# Find the labels CSV files (adapt glob to actual output naming)
LABELS_1=$(find "${RESULTS_DIR}/ckpt_run1" -name "*labels*" -o -name "*assignment*" 2>/dev/null | head -1)
LABELS_2=$(find "${RESULTS_DIR}/ckpt_run2" -name "*labels*" -o -name "*assignment*" 2>/dev/null | head -1)

if [[ -z "${LABELS_1}" || -z "${LABELS_2}" ]]; then
    echo "WARNING: Could not find labels files to compare."
    echo "  Run 1 output: $(ls -la "${RESULTS_DIR}/ckpt_run1/" 2>/dev/null || echo 'missing')"
    echo "  Run 2 output: $(ls -la "${RESULTS_DIR}/ckpt_run2/" 2>/dev/null || echo 'missing')"
    CKPT_RESULT="SKIP"
else
    if diff -q "${LABELS_1}" "${LABELS_2}" > /dev/null 2>&1; then
        echo "PASS: Labels from Run 1 and Run 2 are identical."
        CKPT_RESULT="PASS"
    else
        echo "FAIL: Labels differ between Run 1 and Run 2."
        echo "--- diff ---"
        diff "${LABELS_1}" "${LABELS_2}" || true
        CKPT_RESULT="FAIL"
    fi
fi

# Write checkpoint comparison result
cat > "${RESULTS_DIR}/checkpoint_result.json" <<JSONEOF
{
    "test": "checkpoint_resume_comparison",
    "job_id": "${SLURM_JOB_ID}",
    "result": "${CKPT_RESULT}",
    "labels_run1": "${LABELS_1:-none}",
    "labels_run2": "${LABELS_2:-none}",
    "timestamp": "$(date -Iseconds)"
}
JSONEOF

# ── Copy results back ────────────────────────────────────────────────────
echo ""
echo "── Copying results to submit directory ─────────────────────────"
mkdir -p "${SLURM_SUBMIT_DIR}/results/checkpoint_test_${SLURM_JOB_ID}"
cp -r "${RESULTS_DIR}/." "${SLURM_SUBMIT_DIR}/results/checkpoint_test_${SLURM_JOB_ID}/"
cp -r "${CKPT_DIR}" "${SLURM_SUBMIT_DIR}/results/checkpoint_test_${SLURM_JOB_ID}/ckpt_data"

echo ""
echo "================================================================"
echo "  Checkpoint test complete — ${CKPT_RESULT}"
echo "  Results: results/checkpoint_test_${SLURM_JOB_ID}/"
echo "  $(date -Iseconds)"
echo "================================================================"
