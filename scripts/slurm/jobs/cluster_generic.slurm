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
#   DTWC_MAX_ITER / DTWC_VARIANT                            (100 / standard)
#   DTWC_WDTW_G / DTWC_ADTW_PENALTY / DTWC_MSM_C           (.05 / 1 / 1)
#   DTWC_TWE_NU / DTWC_TWE_LAMBDA                          (.001 / 1)
#   DTWC_MV_MODE / DTWC_MISSING_STRATEGY / DTWC_METRIC     (dependent / error / l1)
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
DTWC_MAX_ITER="${DTWC_MAX_ITER:-100}"
DTWC_VARIANT="${DTWC_VARIANT:-standard}"
DTWC_WDTW_G="${DTWC_WDTW_G:-0.05}"
DTWC_ADTW_PENALTY="${DTWC_ADTW_PENALTY:-1.0}"
DTWC_MSM_C="${DTWC_MSM_C:-1.0}"
DTWC_TWE_NU="${DTWC_TWE_NU:-0.001}"
DTWC_TWE_LAMBDA="${DTWC_TWE_LAMBDA:-1.0}"
DTWC_MV_MODE="${DTWC_MV_MODE:-dependent}"
DTWC_MISSING_STRATEGY="${DTWC_MISSING_STRATEGY:-error}"
DTWC_METRIC="${DTWC_METRIC:-l1}"

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
echo "  max_iter=${DTWC_MAX_ITER}  variant=${DTWC_VARIANT}  metric=${DTWC_METRIC}"
echo "  mv_mode=${DTWC_MV_MODE}  missing_strategy=${DTWC_MISSING_STRATEGY}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "================================================================"

# ── Output lands under results/ so slurm_remote.sh download retrieves it ────
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${DTWC_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

T_START=$(date +%s.%N)
DTWC_EXIT=0
DTWC_ARGS=(
    --input "${DTWC_INPUT}"
    -k "${DTWC_K}"
    --skip-cols "${DTWC_SKIP_COLS}"
    --dtype "${DTWC_DTYPE}"
    --method "${DTWC_METHOD}"
    --band "${DTWC_BAND}"
    --device "${DTWC_DEVICE}"
    --max-iter "${DTWC_MAX_ITER}"
    --variant "${DTWC_VARIANT}"
    --wdtw-g "${DTWC_WDTW_G}"
    --adtw-penalty "${DTWC_ADTW_PENALTY}"
    --msm-c "${DTWC_MSM_C}"
    --twe-nu "${DTWC_TWE_NU}"
    --twe-lambda "${DTWC_TWE_LAMBDA}"
    --mv-mode "${DTWC_MV_MODE}"
    --missing-strategy "${DTWC_MISSING_STRATEGY}"
    --metric "${DTWC_METRIC}"
    --n-init "${DTWC_N_INIT}"
)
if [[ -n "${DTWC_SEED}" ]]; then
    DTWC_ARGS+=(--seed "${DTWC_SEED}")
fi
DTWC_ARGS+=(
    --name "${DTWC_NAME}"
    --output "${RESULTS_DIR}"
    --verbose
)
"${DTWC_BIN}" "${DTWC_ARGS[@]}" || DTWC_EXIT=$?
T_END=$(date +%s.%N)
ELAPSED=$(echo "${T_END} - ${T_START}" | bc 2>/dev/null || echo "?")

echo "----------------------------------------------------------------"
echo "  dtwc_cl exit=${DTWC_EXIT}  elapsed=${ELAPSED}s"
echo "  Labels : results/${DTWC_NAME}_${SLURM_JOB_ID}/${DTWC_NAME}_labels.csv"
echo "================================================================"
exit "${DTWC_EXIT}"
