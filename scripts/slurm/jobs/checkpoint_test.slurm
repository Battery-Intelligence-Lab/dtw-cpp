#!/bin/bash
#SBATCH --job-name=dtwc-ckpt-test
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/checkpoint_test_%j.out
#SBATCH --error=logs/checkpoint_test_%j.err
#
# DTWC++ checkpoint/resume test — verify completed-result replay
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

# ── Run 1: produce a completed result and dense distance checkpoint ──────
RUN1_DIR="${RESULTS_DIR}/ckpt_run1"
RUN2_DIR="${RESULTS_DIR}/ckpt_run2"
RUN1_LOG="${RESULTS_DIR}/checkpoint_run1.log"
RUN2_LOG="${RESULTS_DIR}/checkpoint_run2.log"
mkdir -p "${RUN1_DIR}" "${RUN2_DIR}"

echo ""
echo "── Run 1: Fresh run with checkpoint ────────────────────────────"
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
    --checkpoint "${CKPT_DIR}" \
    --output "${RUN1_DIR}" \
    >"${RUN1_LOG}" 2>&1 \
    || DTWC_EXIT=$?
cat "${RUN1_LOG}"
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "checkpoint_run1" "${ELAPSED}" "${DTWC_EXIT}"
if [[ "${DTWC_EXIT}" -ne 0 ]]; then
    echo "ERROR: Fresh checkpoint run failed with exit ${DTWC_EXIT}." >&2
    exit 1
fi

for artifact in coffee_checkpoint.bin coffee_labels.csv coffee_medoids.csv; do
    if [[ ! -f "${RUN1_DIR}/${artifact}" ]]; then
        echo "ERROR: Fresh run did not create ${artifact}." >&2
        exit 1
    fi
done
cp "${RUN1_DIR}/coffee_labels.csv" "${RESULTS_DIR}/expected_labels.csv"
cp "${RUN1_DIR}/coffee_medoids.csv" "${RESULTS_DIR}/expected_medoids.csv"
cp "${RUN1_DIR}/coffee_checkpoint.bin" "${RUN2_DIR}/coffee_checkpoint.bin"
# Make a byte-identical rewrite observable through whole-second `stat -c %Y`.
touch -d "24 hours ago" "${RUN2_DIR}/coffee_checkpoint.bin"
CHECKPOINT_SHA_BEFORE=$(sha256sum "${RUN2_DIR}/coffee_checkpoint.bin" | awk '{print $1}')
CHECKPOINT_MTIME_BEFORE=$(stat -c %Y "${RUN2_DIR}/coffee_checkpoint.bin")

# ── Run 2: replay only the copied binary result ──────────────────────────
echo ""
echo "── Run 2: Replay completed binary result ───────────────────────"
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
    --resume \
    --output "${RUN2_DIR}" \
    >"${RUN2_LOG}" 2>&1 \
    || DTWC_EXIT=$?
cat "${RUN2_LOG}"
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc)
write_timing "checkpoint_run2" "${ELAPSED}" "${DTWC_EXIT}"
if [[ "${DTWC_EXIT}" -ne 0 ]]; then
    echo "ERROR: Binary-result replay failed with exit ${DTWC_EXIT}." >&2
    exit 1
fi

if ! grep -Fq "Replaying completed result checkpoint:" "${RUN2_LOG}"; then
    echo "ERROR: Replay marker is absent." >&2
    exit 1
fi
if grep -Fq "Running FastPAM" "${RUN2_LOG}"; then
    echo "ERROR: Replay unexpectedly executed FastPAM." >&2
    exit 1
fi

for field in "Total cost:" "Converged:" "Iterations:"; do
    RUN1_FIELD=$(grep -F "${field}" "${RUN1_LOG}" | tail -1)
    RUN2_FIELD=$(grep -F "${field}" "${RUN2_LOG}" | tail -1)
    if [[ -z "${RUN1_FIELD}" || "${RUN1_FIELD}" != "${RUN2_FIELD}" ]]; then
        echo "ERROR: Replay summary differs for ${field}" >&2
        echo "  fresh : ${RUN1_FIELD:-missing}" >&2
        echo "  replay: ${RUN2_FIELD:-missing}" >&2
        exit 1
    fi
done

if ! cmp -s "${RESULTS_DIR}/expected_labels.csv" "${RUN2_DIR}/coffee_labels.csv"; then
    echo "ERROR: Replay labels differ from the completed result." >&2
    exit 1
fi
if ! cmp -s "${RESULTS_DIR}/expected_medoids.csv" "${RUN2_DIR}/coffee_medoids.csv"; then
    echo "ERROR: Replay medoids differ from the completed result." >&2
    exit 1
fi

CHECKPOINT_SHA_AFTER=$(sha256sum "${RUN2_DIR}/coffee_checkpoint.bin" | awk '{print $1}')
CHECKPOINT_MTIME_AFTER=$(stat -c %Y "${RUN2_DIR}/coffee_checkpoint.bin")
if [[ "${CHECKPOINT_SHA_BEFORE}" != "${CHECKPOINT_SHA_AFTER}" \
      || "${CHECKPOINT_MTIME_BEFORE}" != "${CHECKPOINT_MTIME_AFTER}" ]]; then
    echo "ERROR: Replay rewrote its source binary checkpoint." >&2
    exit 1
fi

CKPT_RESULT="PASS"
cat > "${RESULTS_DIR}/checkpoint_result.json" <<JSONEOF
{
    "test": "binary_result_replay",
    "job_id": "${SLURM_JOB_ID}",
    "result": "${CKPT_RESULT}",
    "checkpoint_sha256": "${CHECKPOINT_SHA_AFTER}",
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
