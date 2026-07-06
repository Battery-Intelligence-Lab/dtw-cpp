#!/usr/bin/env bash
# Generalized SLURM remote helper — upload, build, submit, download.
#
# Requires: ssh, rsync (both available in Git Bash on Windows).
# Reads configuration from .env at the project root.
#
# Usage:
#   bash scripts/slurm/slurm_remote.sh test
#   bash scripts/slurm/slurm_remote.sh upload
#   bash scripts/slurm/slurm_remote.sh build [profile]
#   bash scripts/slurm/slurm_remote.sh submit-cpu
#   bash scripts/slurm/slurm_remote.sh submit-gpu
#   bash scripts/slurm/slurm_remote.sh submit-checkpoint
#   bash scripts/slurm/slurm_remote.sh submit-parquet
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-cpu
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-gpu [a100|l40s|h100]
#   bash scripts/slurm/slurm_remote.sh submit-cluster <input> <k> [method] [device] [band] [name]
#   bash scripts/slurm/slurm_remote.sh status
#   bash scripts/slurm/slurm_remote.sh download
#   bash scripts/slurm/slurm_remote.sh ssh "command"
#   bash scripts/slurm/slurm_remote.sh interactive

set -euo pipefail

# ── Locate project root and load .env ────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env not found at ${ENV_FILE}"
    echo "       Copy scripts/slurm/env.example to .env and edit it."
    exit 1
fi

# Source .env (simple key=value, no shell expansion)
while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"          # strip Windows \r
    line="${line%%#*}"            # strip comments
    line="${line## }"             # trim leading space
    line="${line%% }"             # trim trailing space
    [[ -z "${line}" ]] && continue
    [[ "${line}" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key// /}"              # strip spaces from key
    [[ -z "${key}" ]] && continue
    export "${key}=${value}"
done < "${ENV_FILE}"

# ── Validate required variables ──────────────────────────────────────────
for var in SLURM_USER SLURM_HOST SLURM_REMOTE_BASE; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} is not set in .env"
        exit 1
    fi
done

SSH_TARGET="${SLURM_USER}@${SLURM_HOST}"
REMOTE="${SLURM_REMOTE_BASE}"
PARTITION="${SLURM_PARTITION:-short}"
CLUSTER_FLAG=""
if [[ -n "${SLURM_CLUSTER:-}" ]]; then
    CLUSTER_FLAG="--clusters=${SLURM_CLUSTER}"
fi
GPU_GRES="${SLURM_GPU_GRES:-gpu:1}"
EMAIL_FLAGS=""
if [[ -n "${SLURM_EMAIL:-}" ]]; then
    EMAIL_FLAGS="--mail-type=BEGIN,END,FAIL --mail-user=${SLURM_EMAIL}"
fi

# ── Helper ───────────────────────────────────────────────────────────────
remote() {
    # Run command on the remote host via SSH
    ssh "${SSH_TARGET}" "$@"
}

banner() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════"
}

# ── Commands ─────────────────────────────────────────────────────────────

cmd_test() {
    banner "Testing SSH connection"
    echo "  Target: ${SSH_TARGET}"
    echo ""
    remote "echo '  Hostname: '\$(hostname); echo '  User:     '\$(whoami); echo '  Date:     '\$(date); echo ''; sinfo --summarize 2>/dev/null || echo '  sinfo not available (not on login node?)'"
    echo ""
    echo "  Connection OK."
}

cmd_upload() {
    banner "Uploading source + test data"
    echo "  Local:  ${PROJECT_ROOT}"
    echo "  Remote: ${SSH_TARGET}:${REMOTE}"
    echo ""

    # Create remote directory structure
    remote "mkdir -p ${REMOTE}/src/dtwc ${REMOTE}/src/cmake ${REMOTE}/src/scripts/slurm/jobs ${REMOTE}/data/Coffee ${REMOTE}/data/Beef ${REMOTE}/results ${REMOTE}/logs"

    # Detect transfer tool: rsync (preferred) or scp (fallback)
    local USE_RSYNC=false
    if command -v rsync &>/dev/null; then
        USE_RSYNC=true
    fi

    _upload_dir() {
        local src="$1" dst="$2"
        if ${USE_RSYNC}; then
            rsync -avz --progress "${src}/" "${SSH_TARGET}:${dst}/"
        else
            scp -r "${src}/." "${SSH_TARGET}:${dst}/"
        fi
    }

    _upload_file() {
        local src="$1" dst="$2"
        if ${USE_RSYNC}; then
            rsync -avz --progress "${src}" "${SSH_TARGET}:${dst}"
        else
            scp "${src}" "${SSH_TARGET}:${dst}"
        fi
    }

    # Upload source code (explicit allowlist -- never uploads .env, .git, build/)
    echo "[1/5] Uploading dtwc/ source..."
    _upload_dir "${PROJECT_ROOT}/dtwc" "${REMOTE}/src/dtwc"

    echo ""
    echo "[2/5] Uploading cmake/ + build files..."
    _upload_dir "${PROJECT_ROOT}/cmake" "${REMOTE}/src/cmake"
    _upload_dir "${PROJECT_ROOT}/scripts/slurm" "${REMOTE}/src/scripts/slurm"
    for f in CMakeLists.txt CMakePresets.json VERSION; do
        [[ -f "${PROJECT_ROOT}/${f}" ]] && _upload_file "${PROJECT_ROOT}/${f}" "${REMOTE}/src/${f}"
    done

    # Upload test datasets
    echo ""
    echo "[3/5] Uploading Coffee dataset..."
    local COFFEE="${PROJECT_ROOT}/data/benchmark/UCRArchive_2018/Coffee"
    if [[ -d "${COFFEE}" ]]; then
        _upload_dir "${COFFEE}" "${REMOTE}/data/Coffee"
    else
        echo "  SKIP: ${COFFEE} not found"
    fi

    echo ""
    echo "[4/5] Uploading Beef dataset..."
    local BEEF="${PROJECT_ROOT}/data/benchmark/UCRArchive_2018/Beef"
    if [[ -d "${BEEF}" ]]; then
        _upload_dir "${BEEF}" "${REMOTE}/data/Beef"
    else
        echo "  SKIP: ${BEEF} not found"
    fi

    echo ""
    echo "[5/5] Uploading dummy test data..."
    local DUMMY="${PROJECT_ROOT}/data/dummy"
    if [[ -d "${DUMMY}" ]]; then
        remote "mkdir -p ${REMOTE}/data/dummy"
        _upload_dir "${DUMMY}" "${REMOTE}/data/dummy"
    else
        echo "  SKIP: ${DUMMY} not found"
    fi

    echo ""
    echo "  Upload complete."
}

cmd_build() {
    local PROFILE="${1:-htc-cpu}"
    banner "Building on cluster (profile: ${PROFILE})"

    # Submit a batch build job
    local BUILD_SCRIPT="#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=dtwc-build
#SBATCH --output=${REMOTE}/logs/build_%j.out
#SBATCH --error=${REMOTE}/logs/build_%j.err
${CLUSTER_FLAG:+#SBATCH ${CLUSTER_FLAG}}

module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0 2>/dev/null || true
module load cmake gcc cuda 2>/dev/null || true
cd ${REMOTE}/src
# Disable testing (tests/ not uploaded); Arrow off until include path fix
export DTWC_BUILD_TESTING=OFF
export DTWC_ENABLE_ARROW=OFF
source scripts/slurm/build-arc.sh ${PROFILE}
"

    echo "  Submitting build job..."
    local JOB_ID
    JOB_ID=$(remote "echo '${BUILD_SCRIPT}' | sbatch --parsable")
    echo "  Job ID: ${JOB_ID}"
    echo "  Monitor: ssh ${SSH_TARGET} 'squeue -j ${JOB_ID}'"
    echo "  Log:     ssh ${SSH_TARGET} 'tail -f ${REMOTE}/logs/build_${JOB_ID}.out'"
    echo ""
    echo "  Build submitted. Check status with:"
    echo "    bash scripts/slurm/slurm_remote.sh status"
}

_submit_job() {
    local SLURM_FILE="$1"
    local LABEL="$2"
    local BIN_PATTERN="${3:-}"

    banner "Submitting ${LABEL}"

    # Preflight: check binary exists
    if [[ -n "${BIN_PATTERN}" ]]; then
        local EXISTS
        EXISTS=$(remote "ls ${REMOTE}/src/${BIN_PATTERN} 2>/dev/null | head -1" || true)
        if [[ -z "${EXISTS}" ]]; then
            echo "  ERROR: Binary not found: ${REMOTE}/src/${BIN_PATTERN}"
            echo "         Run 'bash scripts/slurm/slurm_remote.sh build' first."
            exit 1
        fi
        echo "  Binary: ${EXISTS}"
    fi

    # Upload the latest job script
    scp "${PROJECT_ROOT}/${SLURM_FILE}" "${SSH_TARGET}:${REMOTE}/src/${SLURM_FILE}"

    local EXTRA_SBATCH="${4:-}"
    local JOB_ID
    JOB_ID=$(remote "cd ${REMOTE}/src && sbatch --parsable ${CLUSTER_FLAG} ${EMAIL_FLAGS} ${EXTRA_SBATCH} ${SLURM_FILE}")
    echo "  Job ID: ${JOB_ID}"
    echo "  Monitor: bash scripts/slurm/slurm_remote.sh status"
}

cmd_submit_cpu() {
    _submit_job "scripts/slurm/jobs/cpu_test.slurm" "CPU test" "build-*/bin/dtwc_cl"
}

cmd_submit_gpu() {
    _submit_job "scripts/slurm/jobs/gpu_test.slurm" "GPU test" "build-*/bin/dtwc_cl"
}

cmd_submit_checkpoint() {
    _submit_job "scripts/slurm/jobs/checkpoint_test.slurm" "Checkpoint test" "build-*/bin/dtwc_cl"
}

cmd_submit_parquet() {
    _submit_job "scripts/slurm/jobs/parquet_test.slurm" "Parquet test" "build-*/bin/dtwc_cl"
}

cmd_submit_benchmark_cpu() {
    _submit_job "scripts/slurm/jobs/ucr_benchmark_cpu.slurm" "UCR benchmark (CPU)" "build-*/bin/dtwc_cl"
}

cmd_submit_benchmark_gpu() {
    local gpu_type="${1:-}"
    local extra_args=""
    if [[ -n "${gpu_type}" ]]; then
        extra_args="--gres=gpu:${gpu_type}:1"
        echo "  Requesting GPU type: ${gpu_type}"
    fi
    _submit_job "scripts/slurm/jobs/ucr_benchmark_gpu.slurm" "UCR benchmark (GPU${gpu_type:+: ${gpu_type}})" "build-*/bin/dtwc_cl" "${extra_args}"
}

# Generic clustering: submit cluster_generic.slurm on an arbitrary input.
# Args: <input> <k> [method=pam] [device=cpu] [band=-1] [name=dtwc_job] [skip_cols=0] [upload=1]
#   upload=1 : <input> is a local file -> rsync it to the cluster.
#   upload=0 : <input> is a path ON the cluster (pre-staged) -> used as-is, no read/upload.
# Used by the Python device='hpc' offload path (dtwcpp._hpc.cluster_on_hpc).
cmd_submit_cluster() {
    local INPUT="${1:?input required}"
    local K="${2:?number of clusters required}"
    local METHOD="${3:-pam}"
    local DEVICE="${4:-cpu}"
    local BAND="${5:--1}"
    local NAME="${6:-dtwc_job}"
    local SKIP_COLS="${7:-0}"
    local UPLOAD="${8:-1}"

    banner "Submitting clustering job (${NAME}, k=${K}, device=${DEVICE})"

    # Preflight: a build must exist on the cluster
    local EXISTS
    EXISTS=$(remote "ls ${REMOTE}/src/build-*/bin/dtwc_cl 2>/dev/null | head -1" || true)
    if [[ -z "${EXISTS}" ]]; then
        echo "  ERROR: no dtwc_cl build on cluster. Run 'slurm_remote.sh build' first." >&2
        exit 1
    fi

    # Resolve the cluster-side input path (upload a local file, or use as-is)
    local REMOTE_INPUT
    if [[ "${UPLOAD}" == "1" ]]; then
        [[ -f "${INPUT}" ]] || { echo "ERROR: input not found: ${INPUT}"; exit 1; }
        local BASE; BASE="$(basename "${INPUT}")"
        remote "mkdir -p ${REMOTE}/data/userjobs"
        if command -v rsync &>/dev/null; then
            rsync -az "${INPUT}" "${SSH_TARGET}:${REMOTE}/data/userjobs/${BASE}"
        else
            scp "${INPUT}" "${SSH_TARGET}:${REMOTE}/data/userjobs/${BASE}"
        fi
        REMOTE_INPUT="${REMOTE}/data/userjobs/${BASE}"
    else
        REMOTE_INPUT="${INPUT}"          # pre-staged on the cluster
    fi

    # Refresh the generic job script
    scp "${PROJECT_ROOT}/scripts/slurm/jobs/cluster_generic.slurm" \
        "${SSH_TARGET}:${REMOTE}/src/scripts/slurm/jobs/cluster_generic.slurm"

    # GPU runs need a GRES request (the job file is partition-agnostic)
    local GPU_FLAGS=""
    if [[ "${DEVICE}" == cuda* || "${DEVICE}" == gpu ]]; then
        GPU_FLAGS="--gres=${GPU_GRES}"
    fi

    local EXPORTS="ALL,DTWC_INPUT=${REMOTE_INPUT},DTWC_K=${K},DTWC_SKIP_COLS=${SKIP_COLS}"
    EXPORTS+=",DTWC_METHOD=${METHOD},DTWC_DEVICE=${DEVICE},DTWC_BAND=${BAND},DTWC_NAME=${NAME}"

    local JOB_ID
    JOB_ID=$(remote "cd ${REMOTE}/src && sbatch --parsable ${CLUSTER_FLAG} ${EMAIL_FLAGS} ${GPU_FLAGS} --export=${EXPORTS} scripts/slurm/jobs/cluster_generic.slurm")
    echo "  Job ID: ${JOB_ID}"
    echo "  Monitor: bash scripts/slurm/slurm_remote.sh status"
}

cmd_status() {
    banner "SLURM Job Status"
    remote "squeue -u ${SLURM_USER} ${CLUSTER_FLAG} 2>/dev/null || squeue -u ${SLURM_USER}"
}

cmd_download() {
    banner "Downloading results + logs"
    local LOCAL_RESULTS="${PROJECT_ROOT}/results/slurm"
    mkdir -p "${LOCAL_RESULTS}"

    echo "  Remote: ${SSH_TARGET}:${REMOTE}/src/results/"
    echo "  Local:  ${LOCAL_RESULTS}/"
    echo ""

    if command -v rsync &>/dev/null; then
        rsync -avz --progress "${SSH_TARGET}:${REMOTE}/src/results/" "${LOCAL_RESULTS}/"
        echo ""
        echo "  Downloading logs..."
        rsync -avz --progress "${SSH_TARGET}:${REMOTE}/src/logs/" "${LOCAL_RESULTS}/logs/" 2>/dev/null || echo "  No logs found."
    else
        scp -r "${SSH_TARGET}:${REMOTE}/src/results/." "${LOCAL_RESULTS}/"
        echo ""
        echo "  Downloading logs..."
        scp -r "${SSH_TARGET}:${REMOTE}/src/logs/." "${LOCAL_RESULTS}/logs/" 2>/dev/null || echo "  No logs found."
    fi

    echo ""
    echo "  Results downloaded to: ${LOCAL_RESULTS}/"
}

cmd_ssh() {
    remote "cd ${REMOTE}/src 2>/dev/null; $*"
}

cmd_interactive() {
    banner "Interactive Session Guide"
    echo ""
    echo "  1. SSH to cluster:"
    echo "     ssh ${SSH_TARGET}"
    echo ""
    echo "  2. Start interactive session:"
    echo "     srun -p interactive --pty /bin/bash"
    echo ""
    echo "  3. Load modules and build:"
    echo "     module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0"
    echo "     cd ${REMOTE}/src"
    echo "     source scripts/slurm/build-arc.sh htc-gpu"
    echo ""
    echo "  4. Test manually:"
    echo "     ./build-htc-gpu/bin/dtwc_cl -i ../data/Coffee/Coffee_TRAIN.tsv --skip-cols 1 -k 2 -v"
    echo ""
}

# ── Dispatch ─────────────────────────────────────────────────────────────

CMD="${1:-help}"
shift || true

case "${CMD}" in
    test)              cmd_test ;;
    upload)            cmd_upload ;;
    build)             cmd_build "$@" ;;
    submit-cpu)        cmd_submit_cpu ;;
    submit-gpu)        cmd_submit_gpu ;;
    submit-checkpoint) cmd_submit_checkpoint ;;
    submit-parquet)    cmd_submit_parquet ;;
    submit-benchmark-cpu) cmd_submit_benchmark_cpu ;;
    submit-benchmark-gpu) cmd_submit_benchmark_gpu "$@" ;;
    submit-cluster)    cmd_submit_cluster "$@" ;;
    status)            cmd_status ;;
    download)          cmd_download ;;
    ssh)               cmd_ssh "$@" ;;
    interactive)       cmd_interactive ;;
    help|--help|-h)
        echo "Usage: bash scripts/slurm/slurm_remote.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  test              Test SSH connection"
        echo "  upload            Upload source + test data"
        echo "  build [profile]   Submit batch build job (default: htc-cpu)"
        echo "  submit-cpu        Submit CPU test job"
        echo "  submit-gpu        Submit GPU test job"
        echo "  submit-checkpoint Submit checkpoint/resume test"
        echo "  submit-parquet    Submit Parquet I/O test"
        echo "  submit-benchmark-cpu  Submit full UCR benchmark (CPU, ~12h)"
        echo "  submit-benchmark-gpu [type]  Submit full UCR benchmark (GPU, e.g. a100, l40s)"
        echo "  submit-cluster <input> <k> [method] [device] [band] [name]"
        echo "                    Upload an arbitrary input file + cluster it (device='hpc' path)"
        echo "  status            Show SLURM queue"
        echo "  download          Download results + logs"
        echo "  ssh \"command\"     Run arbitrary command on cluster"
        echo "  interactive       Print interactive session guide"
        echo ""
        echo "Configuration: edit .env at project root (see scripts/slurm/env.example)"
        ;;
    *)
        echo "Unknown command: ${CMD}"
        echo "Run 'bash scripts/slurm/slurm_remote.sh help' for usage."
        exit 1
        ;;
esac
