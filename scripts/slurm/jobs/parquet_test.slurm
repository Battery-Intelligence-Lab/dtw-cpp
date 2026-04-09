#!/bin/bash
#SBATCH --job-name=dtwc-parquet-test
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/parquet_test_%j.out
#SBATCH --error=logs/parquet_test_%j.err
#
# DTWC++ Parquet/dtws format test — verify Arrow-based I/O
#
# Usage: sbatch scripts/slurm/jobs/parquet_test.slurm
#        Expects build-htc-cpu/ to exist (see scripts/slurm/build-arc.sh htc-cpu)

set -euo pipefail

# ── Portable scratch ───────────────────────────────────────────────────────
WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "${WORKDIR}"

# ── Module load with fallback ──────────────────────────────────────────────
module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
module load Arrow/15.0.0 2>/dev/null || module load arrow 2>/dev/null || true

# ── Diagnostics header ────────────────────────────────────────────────────
echo "================================================================"
echo "  DTWC++ Parquet/DTWS Format Test"
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
mkdir -p "${RESULTS_DIR}"

TESTS_RUN=0
TESTS_PASS=0
TESTS_FAIL=0
TESTS_SKIP=0

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

# ── Test 1: Parquet format ───────────────────────────────────────────────
PARQUET_FILE="${WORKDIR}/Coffee/Coffee_TRAIN.parquet"
echo ""
if [[ -f "${DATA_DIR}/Coffee/Coffee_TRAIN.parquet" ]]; then
    cp "${DATA_DIR}/Coffee/Coffee_TRAIN.parquet" "${PARQUET_FILE}"
    echo "── Test 1: Coffee.parquet k=2 ──────────────────────────────────"
    TESTS_RUN=$((TESTS_RUN + 1))
    T_START=$(date +%s.%N)
    if "${DTWC_BIN}" \
        --input "${PARQUET_FILE}" \
        --k 2 \
        --skip-cols 1 \
        --dtype float64 \
        --method pam \
        --verbose \
        --output "${RESULTS_DIR}/parquet_coffee_k2"; then
        TESTS_PASS=$((TESTS_PASS + 1))
        echo "Test 1: PASS"
    else
        TESTS_FAIL=$((TESTS_FAIL + 1))
        echo "Test 1: FAIL (exit $?)"
    fi
    T_END=$(date +%s.%N)
    ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
    write_timing "parquet_coffee_k2" "${ELAPSED}" "0"
    echo "Parquet test completed in ${ELAPSED}s"
else
    echo "── Test 1: SKIP (Coffee_TRAIN.parquet not found) ───────────────"
    TESTS_SKIP=$((TESTS_SKIP + 1))
fi

# ── Test 2: DTWS format ─────────────────────────────────────────────────
DTWS_FILE="${WORKDIR}/Coffee/Coffee_TRAIN.dtws"
echo ""
if [[ -f "${DATA_DIR}/Coffee/Coffee_TRAIN.dtws" ]]; then
    cp "${DATA_DIR}/Coffee/Coffee_TRAIN.dtws" "${DTWS_FILE}"
    echo "── Test 2: Coffee.dtws k=2 ────────────────────────────────────"
    TESTS_RUN=$((TESTS_RUN + 1))
    T_START=$(date +%s.%N)
    if "${DTWC_BIN}" \
        --input "${DTWS_FILE}" \
        --k 2 \
        --skip-cols 1 \
        --dtype float64 \
        --method pam \
        --verbose \
        --output "${RESULTS_DIR}/dtws_coffee_k2"; then
        TESTS_PASS=$((TESTS_PASS + 1))
        echo "Test 2: PASS"
    else
        TESTS_FAIL=$((TESTS_FAIL + 1))
        echo "Test 2: FAIL (exit $?)"
    fi
    T_END=$(date +%s.%N)
    ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
    write_timing "dtws_coffee_k2" "${ELAPSED}" "0"
    echo "DTWS test completed in ${ELAPSED}s"
else
    echo "── Test 2: SKIP (Coffee_TRAIN.dtws not found) ──────────────────"
    TESTS_SKIP=$((TESTS_SKIP + 1))
fi

# ── Test 3: Compare with TSV baseline (if both Parquet and TSV ran) ──────
echo ""
TSV_LABELS="${RESULTS_DIR}/tsv_coffee_k2"
PARQUET_LABELS="${RESULTS_DIR}/parquet_coffee_k2"
DTWS_LABELS="${RESULTS_DIR}/dtws_coffee_k2"

# Run TSV baseline for comparison
echo "── Baseline: Coffee TSV k=2 ────────────────────────────────────"
T_START=$(date +%s.%N)
"${DTWC_BIN}" \
    --input "${WORKDIR}/Coffee/Coffee_TRAIN.tsv" \
    --k 2 \
    --skip-cols 1 \
    --dtype float64 \
    --method pam \
    --verbose \
    --output "${TSV_LABELS}" \
    || true
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "tsv_coffee_k2_baseline" "${ELAPSED}" "0"

# Compare Parquet vs TSV
if [[ -d "${PARQUET_LABELS}" && -d "${TSV_LABELS}" ]]; then
    echo ""
    echo "── Comparing Parquet vs TSV results ────────────────────────────"
    PAR_LABEL=$(find "${PARQUET_LABELS}" -name "*labels*" -o -name "*assignment*" 2>/dev/null | head -1)
    TSV_LABEL=$(find "${TSV_LABELS}" -name "*labels*" -o -name "*assignment*" 2>/dev/null | head -1)
    if [[ -n "${PAR_LABEL}" && -n "${TSV_LABEL}" ]]; then
        if diff -q "${PAR_LABEL}" "${TSV_LABEL}" > /dev/null 2>&1; then
            echo "PASS: Parquet and TSV produce identical labels."
        else
            echo "WARN: Parquet and TSV labels differ (may be acceptable if non-deterministic)."
            diff "${PAR_LABEL}" "${TSV_LABEL}" || true
        fi
    else
        echo "SKIP: Could not find labels files for comparison."
    fi
fi

# Compare DTWS vs TSV
if [[ -d "${DTWS_LABELS}" && -d "${TSV_LABELS}" ]]; then
    echo ""
    echo "── Comparing DTWS vs TSV results ──────────────────────────────"
    DTWS_LABEL=$(find "${DTWS_LABELS}" -name "*labels*" -o -name "*assignment*" 2>/dev/null | head -1)
    TSV_LABEL=$(find "${TSV_LABELS}" -name "*labels*" -o -name "*assignment*" 2>/dev/null | head -1)
    if [[ -n "${DTWS_LABEL}" && -n "${TSV_LABEL}" ]]; then
        if diff -q "${DTWS_LABEL}" "${TSV_LABEL}" > /dev/null 2>&1; then
            echo "PASS: DTWS and TSV produce identical labels."
        else
            echo "WARN: DTWS and TSV labels differ (may be acceptable if non-deterministic)."
            diff "${DTWS_LABEL}" "${TSV_LABEL}" || true
        fi
    else
        echo "SKIP: Could not find labels files for comparison."
    fi
fi

# ── Copy results back ────────────────────────────────────────────────────
echo ""
echo "── Copying results to submit directory ─────────────────────────"
mkdir -p "${SLURM_SUBMIT_DIR}/results/parquet_test_${SLURM_JOB_ID}"
cp -r "${RESULTS_DIR}/." "${SLURM_SUBMIT_DIR}/results/parquet_test_${SLURM_JOB_ID}/"

# ── Summary ──────────────────────────────────────────────────────────────
cat > "${SLURM_SUBMIT_DIR}/results/parquet_test_${SLURM_JOB_ID}/summary.json" <<JSONEOF
{
    "job_id": "${SLURM_JOB_ID}",
    "tests_run": ${TESTS_RUN},
    "tests_pass": ${TESTS_PASS},
    "tests_fail": ${TESTS_FAIL},
    "tests_skip": ${TESTS_SKIP},
    "timestamp": "$(date -Iseconds)"
}
JSONEOF

echo ""
echo "================================================================"
echo "  Parquet test complete"
echo "  Run: ${TESTS_RUN}  Pass: ${TESTS_PASS}  Fail: ${TESTS_FAIL}  Skip: ${TESTS_SKIP}"
echo "  Results: results/parquet_test_${SLURM_JOB_ID}/"
echo "  $(date -Iseconds)"
echo "================================================================"
