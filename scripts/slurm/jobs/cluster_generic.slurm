#!/bin/bash
#SBATCH --job-name=dtwc-cluster
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/cluster_%j.out
#SBATCH --error=logs/cluster_%j.err
#
# DTWC++ generic clustering job — parametrized via environment variables.
# Driven by `slurm_remote.sh submit-cluster` (which passes them via --export).
#
#   DTWC_INPUT    absolute path to the input file on the cluster (required)
#   DTWC_K        number of clusters (required)
#   DTWC_METHOD   pam | kmedoids | clara | hierarchical   (default: pam)
#   DTWC_DEVICE   cpu | cuda | cuda:N                      (default: cpu)
#   DTWC_BAND     Sakoe-Chiba band (-1 = full)            (default: -1)
#   DTWC_NAME     output filename prefix                   (default: dtwc_job)
#   DTWC_SKIP_COLS / DTWC_DTYPE                            (defaults: 0 / float64)
#   DTWC_N_INIT   deterministic PAM restart count          (default: 1)
#   DTWC_SEED     first restart seed                        (default: CLI default)
#
# For GPU runs add `--gres=gpu:1` at submit time (slurm_remote.sh does this when
# DTWC_DEVICE is cuda). Expects a build-*/bin/dtwc_cl on the cluster.

set -euo pipefail

# ── Parameters (with defaults) ─────────────────────────────────────────────
DTWC_INPUT="${DTWC_INPUT:?DTWC_INPUT not set}"
DTWC_K="${DTWC_K:?DTWC_K not set}"
DTWC_METHOD="${DTWC_METHOD:-pam}"
DTWC_DEVICE="${DTWC_DEVICE:-cpu}"
DTWC_BAND="${DTWC_BAND:--1}"
DTWC_NAME="${DTWC_NAME:-dtwc_job}"
DTWC_SKIP_COLS="${DTWC_SKIP_COLS:-0}"
DTWC_DTYPE="${DTWC_DTYPE:-float64}"
DTWC_N_INIT="${DTWC_N_INIT:-1}"
DTWC_SEED="${DTWC_SEED:-}"

# ── Module load with fallback ──────────────────────────────────────────────
module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
if [[ "${DTWC_DEVICE}" == cuda* ]]; then
    module load CUDA/12.4.0 2>/dev/null || module load cuda 2>/dev/null || true
fi

# ── Locate a built binary (prefer a GPU build when device=cuda) ─────────────
find_bin() {
    local b
    if [[ "${DTWC_DEVICE}" == cuda* ]]; then
        for b in "${SLURM_SUBMIT_DIR}"/build-*gpu*/bin/dtwc_cl "${SLURM_SUBMIT_DIR}"/build-h100/bin/dtwc_cl; do
            [[ -x "${b}" ]] && { echo "${b}"; return 0; }
        done
    fi
    for b in "${SLURM_SUBMIT_DIR}"/build-*/bin/dtwc_cl; do
        [[ -x "${b}" ]] && { echo "${b}"; return 0; }
    done
    return 1
}
DTWC_BIN="$(find_bin)" || { echo "ERROR: no build-*/bin/dtwc_cl found. Run 'slurm_remote.sh build' first." >&2; exit 1; }

if [[ ! -f "${DTWC_INPUT}" ]]; then
    echo "ERROR: input not found: ${DTWC_INPUT}" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "================================================================"
echo "  DTWC++ generic clustering"
echo "  Job ID : ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME:-?}"
echo "  Binary : ${DTWC_BIN}"
echo "  Input  : ${DTWC_INPUT}"
echo "  k=${DTWC_K}  method=${DTWC_METHOD}  device=${DTWC_DEVICE}  band=${DTWC_BAND}"
echo "  n_init=${DTWC_N_INIT}  seed=${DTWC_SEED:-<CLI default>}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "================================================================"

# ── Output lands under results/ so slurm_remote.sh download retrieves it ────
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${DTWC_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

T_START=$(date +%s.%N)
DTWC_EXIT=0
SEED_ARGS=()
if [[ -n "${DTWC_SEED}" ]]; then
    SEED_ARGS=(--seed "${DTWC_SEED}")
fi
"${DTWC_BIN}" \
    --input "${DTWC_INPUT}" \
    -k "${DTWC_K}" \
    --skip-cols "${DTWC_SKIP_COLS}" \
    --dtype "${DTWC_DTYPE}" \
    --method "${DTWC_METHOD}" \
    --band "${DTWC_BAND}" \
    --device "${DTWC_DEVICE}" \
    --n-init "${DTWC_N_INIT}" \
    "${SEED_ARGS[@]}" \
    --name "${DTWC_NAME}" \
    --output "${RESULTS_DIR}" \
    --verbose \
    || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc 2>/dev/null || echo "?")

echo "----------------------------------------------------------------"
echo "  dtwc_cl exit=${DTWC_EXIT}  elapsed=${ELAPSED}s"
echo "  Labels : results/${DTWC_NAME}_${SLURM_JOB_ID}/${DTWC_NAME}_labels.csv"
echo "================================================================"
exit "${DTWC_EXIT}"
