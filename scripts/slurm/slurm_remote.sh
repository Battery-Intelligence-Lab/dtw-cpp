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
#   bash scripts/slurm/slurm_remote.sh build --profile <profile>
#   bash scripts/slurm/slurm_remote.sh submit-cpu
#   bash scripts/slurm/slurm_remote.sh submit-gpu
#   bash scripts/slurm/slurm_remote.sh submit-checkpoint
#   bash scripts/slurm/slurm_remote.sh submit-parquet
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-cpu
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-gpu [a100|l40s|h100]
#   bash scripts/slurm/slurm_remote.sh submit-cluster <input> <k> [method] [device] [band] [name] [skip_cols] [upload] [n_init] [seed] [max_iter] [variant] [variant params...] [mv_mode] [missing_strategy] [metric]
#   bash scripts/slurm/slurm_remote.sh status
#   bash scripts/slurm/slurm_remote.sh download
#   bash scripts/slurm/slurm_remote.sh download-cluster <name> <job-id>
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

# Source .env (simple key=value, no shell expansion). Whitespace surrounding
# keys/values is ignored so the documented `KEY=   # comment` form is empty.
while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"          # strip Windows \r
    line="${line%%#*}"            # strip comments
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    [[ "${line}" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
        echo "ERROR: invalid .env key: ${key}" >&2
        exit 1
    }
    export "${key}=${value}"
done < "${ENV_FILE}"

# ── Validate required variables ──────────────────────────────────────────
for var in SLURM_USER SLURM_HOST SLURM_REMOTE_BASE; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} is not set in .env" >&2
        exit 1
    fi
done

transport_config_error() {
    echo "ERROR: unsafe $1 in .env: $2" >&2
    exit 1
}

# These values cross local argv, rsync/scp's host:path grammar, remote shell
# argv, and Slurm option parsing. Keep the accepted language deliberately
# narrower than those consumers rather than relying on one layer's quoting.
(( ${#SLURM_USER} <= 64 )) \
    && [[ "${SLURM_USER}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] \
    || transport_config_error SLURM_USER "${SLURM_USER}"
(( ${#SLURM_HOST} <= 253 )) \
    && [[ "${SLURM_HOST}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] \
    || transport_config_error SLURM_HOST "${SLURM_HOST}"
(( ${#SLURM_REMOTE_BASE} <= 1024 )) \
    && [[ "${SLURM_REMOTE_BASE}" =~ ^/[A-Za-z0-9_+@%=-][A-Za-z0-9_+@%=.-]*(/[A-Za-z0-9_+@%=-][A-Za-z0-9_+@%=.-]*)*$ ]] \
    || transport_config_error SLURM_REMOTE_BASE "${SLURM_REMOTE_BASE}"

SLURM_PARTITION="${SLURM_PARTITION:-short}"
(( ${#SLURM_PARTITION} <= 64 )) \
    && [[ "${SLURM_PARTITION}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || transport_config_error SLURM_PARTITION "${SLURM_PARTITION}"
if [[ -n "${SLURM_CLUSTER:-}" ]]; then
    (( ${#SLURM_CLUSTER} <= 64 )) \
        && [[ "${SLURM_CLUSTER}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
        || transport_config_error SLURM_CLUSTER "${SLURM_CLUSTER}"
fi
if [[ -n "${SLURM_EMAIL:-}" ]]; then
    (( ${#SLURM_EMAIL} <= 254 )) \
        && [[ "${SLURM_EMAIL}" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}$ ]] \
        || transport_config_error SLURM_EMAIL "${SLURM_EMAIL}"
fi
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:1}"
(( ${#SLURM_GPU_GRES} <= 128 )) \
    && [[ "${SLURM_GPU_GRES}" =~ ^gpu:([A-Za-z][A-Za-z0-9_.-]*:)?[1-9][0-9]*$ ]] \
    || transport_config_error SLURM_GPU_GRES "${SLURM_GPU_GRES}"

SSH_TARGET="${SLURM_USER}@${SLURM_HOST}"
REMOTE="${SLURM_REMOTE_BASE}"
PARTITION="${SLURM_PARTITION}"
CLUSTER_FLAG=""
if [[ -n "${SLURM_CLUSTER:-}" ]]; then
    CLUSTER_FLAG="--clusters=${SLURM_CLUSTER}"
fi
GPU_GRES="${SLURM_GPU_GRES}"

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

decimal_leq() {
    # Compare unsigned decimal strings without signed-shell-integer overflow.
    local VALUE="$1"
    local LIMIT="$2"
    local LEADING_ZEROS
    LEADING_ZEROS="${VALUE%%[!0]*}"
    VALUE="${VALUE#"${LEADING_ZEROS}"}"
    [[ -n "${VALUE}" ]] || VALUE="0"
    (( ${#VALUE} < ${#LIMIT} )) || {
        (( ${#VALUE} == ${#LIMIT} )) \
            && [[ "${VALUE}" == "${LIMIT}" || "${VALUE}" < "${LIMIT}" ]]
    }
}

is_finite_number() {
    [[ "$1" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
        && awk -v value="$1" 'BEGIN {
            numeric = value + 0
            exit !((numeric - numeric) == 0)
        }'
}

shell_join() {
    # OpenSSH hands its arguments to a remote shell as one command string.
    # Quote every argv element independently before crossing that boundary.
    local DESTINATION="$1"
    shift
    local RESULT="" QUOTED ARG
    for ARG in "$@"; do
        printf -v QUOTED '%q' "${ARG}"
        RESULT+="${RESULT:+ }${QUOTED}"
    done
    printf -v "${DESTINATION}" '%s' "${RESULT}"
}

remote_argv() {
    local COMMAND
    shell_join COMMAND "$@"
    remote "${COMMAND}"
}

remote_argv_in_dir() {
    local DIRECTORY="$1"
    shift
    local CD_COMMAND COMMAND
    shell_join CD_COMMAND cd "${DIRECTORY}"
    shell_join COMMAND "$@"
    remote "${CD_COMMAND} && ${COMMAND}"
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
    remote_argv mkdir -p \
        "${REMOTE}/src/dtwc" "${REMOTE}/src/cmake" \
        "${REMOTE}/src/scripts/slurm/jobs" \
        "${REMOTE}/data/Coffee" "${REMOTE}/data/Beef" \
        "${REMOTE}/results" "${REMOTE}/logs"

    # Detect transfer tool: rsync (preferred) or scp (fallback)
    local USE_RSYNC=false
    if command -v rsync &>/dev/null; then
        USE_RSYNC=true
    fi

    _upload_dir() {
        local src="$1" dst="$2"
        if ${USE_RSYNC}; then
            rsync -avz --progress -- "${src}/" "${SSH_TARGET}:${dst}/"
        else
            scp -r -- "${src}/." "${SSH_TARGET}:${dst}/"
        fi
    }

    _upload_file() {
        local src="$1" dst="$2"
        if ${USE_RSYNC}; then
            rsync -avz --progress -- "${src}" "${SSH_TARGET}:${dst}"
        else
            scp -- "${src}" "${SSH_TARGET}:${dst}"
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
        remote_argv mkdir -p "${REMOTE}/data/dummy"
        _upload_dir "${DUMMY}" "${REMOTE}/data/dummy"
    else
        echo "  SKIP: ${DUMMY} not found"
    fi

    echo ""
    echo "  Upload complete."
}

cmd_build() {
    local PROFILE
    case "$#" in
        0) PROFILE="htc-cpu" ;;
        1) PROFILE="$1" ;;
        2)
            [[ "$1" == "--profile" ]] || {
                echo "ERROR: build profile syntax is 'build [--profile] <profile>'" >&2
                exit 1
            }
            PROFILE="$2"
            ;;
        *)
            echo "ERROR: build profile syntax is 'build [--profile] <profile>'" >&2
            exit 1
            ;;
    esac
    case "${PROFILE}" in
        arc|htc-cpu|htc-gpu|htc-v4|h100|grace) ;;
        *)
            echo "ERROR: unsupported build profile '${PROFILE}'; expected arc, htc-cpu, htc-gpu, htc-v4, h100, or grace" >&2
            exit 1
            ;;
    esac
    banner "Building on cluster (profile: ${PROFILE})"

    # The script body is static. Dynamic values cross the boundary as individually
    # quoted sbatch argv/environment entries rather than executable shell text.
    local BUILD_SCRIPT='#!/bin/bash
set -euo pipefail
module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0 2>/dev/null || true
module load cmake gcc cuda 2>/dev/null || true
cd "${DTWC_REMOTE_BASE}/src"
# Disable testing (tests/ not uploaded); Arrow off until include path fix
export DTWC_BUILD_TESTING=OFF
export DTWC_ENABLE_ARROW=OFF
source scripts/slurm/build-arc.sh "${DTWC_BUILD_PROFILE}"
'

    echo "  Submitting build job..."
    local EXPORTS="ALL,DTWC_REMOTE_BASE=${REMOTE},DTWC_BUILD_PROFILE=${PROFILE}"
    local -a SBATCH_ARGS=(
        sbatch --parsable --partition=interactive --time=01:00:00
        --cpus-per-task=8 --mem-per-cpu=4G --job-name=dtwc-build
        "--output=${REMOTE}/logs/build_%j.out"
        "--error=${REMOTE}/logs/build_%j.err"
        "--export=${EXPORTS}"
    )
    [[ -n "${CLUSTER_FLAG}" ]] && SBATCH_ARGS+=("${CLUSTER_FLAG}")
    local SBATCH_COMMAND JOB_ID
    shell_join SBATCH_COMMAND "${SBATCH_ARGS[@]}"
    JOB_ID=$(printf '%s\n' "${BUILD_SCRIPT}" | remote "${SBATCH_COMMAND}")
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
        EXISTS=$(remote_argv find "${REMOTE}/src" \
            -path "${REMOTE}/src/${BIN_PATTERN}" -type f -print -quit || true)
        if [[ -z "${EXISTS}" ]]; then
            echo "  ERROR: Binary not found: ${REMOTE}/src/${BIN_PATTERN}"
            echo "         Run 'bash scripts/slurm/slurm_remote.sh build' first."
            exit 1
        fi
        echo "  Binary: ${EXISTS}"
    fi

    # Upload the latest job script
    scp -- "${PROJECT_ROOT}/${SLURM_FILE}" "${SSH_TARGET}:${REMOTE}/src/${SLURM_FILE}"

    local EXTRA_GRES="${4:-}"
    local -a SBATCH_ARGS=(sbatch --parsable)
    [[ -n "${CLUSTER_FLAG}" ]] && SBATCH_ARGS+=("${CLUSTER_FLAG}")
    if [[ -n "${SLURM_EMAIL:-}" ]]; then
        SBATCH_ARGS+=("--mail-type=BEGIN,END,FAIL" "--mail-user=${SLURM_EMAIL}")
    fi
    [[ -n "${EXTRA_GRES}" ]] && SBATCH_ARGS+=("--gres=${EXTRA_GRES}")
    SBATCH_ARGS+=("${SLURM_FILE}")
    local JOB_ID
    JOB_ID=$(remote_argv_in_dir "${REMOTE}/src" "${SBATCH_ARGS[@]}")
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
    (( $# <= 1 )) || {
        echo "ERROR: benchmark GPU type accepts at most one value" >&2
        exit 1
    }
    local gpu_type="${1:-}"
    case "${gpu_type}" in
        ""|a100|l40s|h100) ;;
        *)
            echo "ERROR: unsupported benchmark GPU type '${gpu_type}'; expected a100, l40s, or h100" >&2
            exit 1
            ;;
    esac
    local gres=""
    if [[ -n "${gpu_type}" ]]; then
        gres="gpu:${gpu_type}:1"
        echo "  Requesting GPU type: ${gpu_type}"
    fi
    _submit_job "scripts/slurm/jobs/ucr_benchmark_gpu.slurm" "UCR benchmark (GPU${gpu_type:+: ${gpu_type}})" "build-*/bin/dtwc_cl" "${gres}"
}

# Generic clustering: submit cluster_generic.slurm on an arbitrary input.
# Args: <input> <k> [method=pam] [device=cpu] [band=-1] [name=dtwc_job]
#       [skip_cols=0] [upload=1] [n_init=1] [seed] [max_iter=100]
#       [variant=standard] [wdtw_g=.05] [adtw_penalty=1] [msm_c=1]
#       [twe_nu=.001] [twe_lambda=1] [mv_mode=dependent]
#       [missing_strategy=error] [metric=l1]
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
    local N_INIT="${9:-1}"
    local SEED="${10:-}"
    local MAX_ITER="${11:-100}"
    local VARIANT="${12:-standard}"
    local WDTW_G="${13:-0.05}"
    local ADTW_PENALTY="${14:-1.0}"
    local MSM_C="${15:-1.0}"
    local TWE_NU="${16:-0.001}"
    local TWE_LAMBDA="${17:-1.0}"
    local MV_MODE="${18:-dependent}"
    local MISSING_STRATEGY="${19:-error}"
    local METRIC="${20:-l1}"

    [[ "${INPUT}" =~ ^[A-Za-z0-9_./:+@%=-]+$ ]] || {
        echo "ERROR: input path contains bytes unsafe for SSH/Slurm export: ${INPUT}" >&2
        exit 1
    }
    [[ "${INPUT}" != -* ]] || {
        echo "ERROR: input path must not start with '-' (transfer option ambiguity): ${INPUT}" >&2
        exit 1
    }
    [[ "${K}" =~ ^[1-9][0-9]*$ ]] \
        && decimal_leq "${K}" "2147483647" || {
        echo "ERROR: n_clusters must fit the dtwc_cl positive int range: ${K}" >&2
        exit 1
    }
    [[ "${METHOD}" =~ ^(auto|pam|onebatch|clara|kmedoids|mip|lrcore|hierarchical|tadpole)$ ]] || {
        echo "ERROR: unsupported method: ${METHOD}" >&2
        exit 1
    }
    [[ "${BAND}" == "-1" || "${BAND}" =~ ^[0-9]+$ ]] || {
        echo "ERROR: band must be -1 or a non-negative integer: ${BAND}" >&2
        exit 1
    }
    if [[ "${BAND}" != "-1" ]] && ! decimal_leq "${BAND}" "2147483647"; then
        echo "ERROR: band exceeds the dtwc_cl int range: ${BAND}" >&2
        exit 1
    fi
    (( ${#NAME} >= 1 && ${#NAME} <= 128 )) \
        && [[ "${NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
        echo "ERROR: job name must be 1-128 ASCII letters, digits, '.', '_', or '-': ${NAME}" >&2
        exit 1
    }
    [[ "${SKIP_COLS}" =~ ^[0-9]+$ ]] \
        && decimal_leq "${SKIP_COLS}" "2147483647" || {
        echo "ERROR: skip_cols must fit the dtwc_cl non-negative int range: ${SKIP_COLS}" >&2
        exit 1
    }
    [[ "${UPLOAD}" == "0" || "${UPLOAD}" == "1" ]] || {
        echo "ERROR: upload must be 0 or 1: ${UPLOAD}" >&2
        exit 1
    }
    if [[ "${UPLOAD}" == "1" && "${INPUT}" == *:* ]]; then
        echo "ERROR: local upload path must not contain ':' (remote-source ambiguity): ${INPUT}" >&2
        exit 1
    fi
    if [[ "${UPLOAD}" == "1" && ! -f "${INPUT}" ]]; then
        echo "ERROR: input not found or not a regular file: ${INPUT}" >&2
        exit 1
    fi
    [[ "${N_INIT}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: n_init must be a positive integer: ${N_INIT}" >&2
        exit 1
    }
    decimal_leq "${N_INIT}" "2147483647" || {
        echo "ERROR: n_init exceeds the dtwc_cl int range: ${N_INIT}" >&2
        exit 1
    }
    [[ -z "${SEED}" || "${SEED}" =~ ^[0-9]+$ ]] || {
        echo "ERROR: seed must be a non-negative integer: ${SEED}" >&2
        exit 1
    }
    if [[ -n "${SEED}" ]] && ! decimal_leq "${SEED}" "4294967295"; then
        echo "ERROR: seed exceeds the dtwc_cl unsigned range: ${SEED}" >&2
        exit 1
    fi
    [[ "${MAX_ITER}" =~ ^[1-9][0-9]*$ ]] \
        && decimal_leq "${MAX_ITER}" "2147483647" || {
        echo "ERROR: max_iter must fit the dtwc_cl positive int range: ${MAX_ITER}" >&2
        exit 1
    }
    [[ "${DEVICE}" =~ ^(cpu|cuda(:[0-9]+)?)$ ]] || {
        echo "ERROR: unsupported remote device: ${DEVICE}" >&2
        exit 1
    }
    if [[ "${DEVICE}" == cuda:* ]] \
        && ! decimal_leq "${DEVICE#cuda:}" "2147483647"; then
        echo "ERROR: remote CUDA ordinal exceeds the dtwc_cl int range: ${DEVICE}" >&2
        exit 1
    fi
    [[ "${VARIANT}" =~ ^(standard|ddtw|wdtw|adtw|msm|twe)$ ]] || {
        echo "ERROR: unsupported variant: ${VARIANT}" >&2
        exit 1
    }
    [[ "${MV_MODE}" =~ ^(dependent|independent)$ ]] || {
        echo "ERROR: unsupported mv_mode: ${MV_MODE}" >&2
        exit 1
    }
    [[ "${MISSING_STRATEGY}" =~ ^(error|zero_cost|arow|interpolate)$ ]] || {
        echo "ERROR: unsupported missing_strategy: ${MISSING_STRATEGY}" >&2
        exit 1
    }
    [[ "${METRIC}" =~ ^(l1|squared_euclidean)$ ]] || {
        echo "ERROR: unsupported metric: ${METRIC}" >&2
        exit 1
    }
    for VALUE in "${WDTW_G}" "${ADTW_PENALTY}" "${MSM_C}" \
                 "${TWE_NU}" "${TWE_LAMBDA}"; do
        is_finite_number "${VALUE}" || {
            echo "ERROR: variant parameters must be finite numbers: ${VALUE}" >&2
            exit 1
        }
    done
    if [[ "${MV_MODE}" == "independent" \
          && ( "${VARIANT}" != "standard" || "${MISSING_STRATEGY}" != "error" ) ]]; then
        echo "ERROR: mv_mode=independent requires variant=standard and missing_strategy=error" >&2
        exit 1
    fi
    if [[ "${VARIANT}" != "standard" && "${MISSING_STRATEGY}" != "error" ]]; then
        echo "ERROR: unsupported variant/missing_strategy combination" >&2
        exit 1
    fi
    if [[ "${DEVICE}" == cuda* ]]; then
        [[ "${VARIANT}" == "standard" ]] || { echo "ERROR: remote CUDA supports variant=standard only" >&2; exit 1; }
        [[ "${MISSING_STRATEGY}" == "error" ]] || { echo "ERROR: remote CUDA does not support missing_strategy" >&2; exit 1; }
        [[ "${MV_MODE}" == "dependent" ]] || { echo "ERROR: remote CUDA does not support mv_mode=independent" >&2; exit 1; }
    elif [[ "${METRIC}" != "l1" ]]; then
        echo "ERROR: metric=${METRIC} is unsupported by the remote CPU CLI" >&2
        exit 1
    fi

    banner "Submitting clustering job (${NAME}, k=${K}, device=${DEVICE})"

    # Preflight: a build must exist on the cluster
    local EXISTS
    EXISTS=$(remote_argv find "${REMOTE}/src" \
        -path "${REMOTE}/src/build-*/bin/dtwc_cl" -type f -print -quit || true)
    if [[ -z "${EXISTS}" ]]; then
        echo "  ERROR: no dtwc_cl build on cluster. Run 'slurm_remote.sh build' first." >&2
        exit 1
    fi

    # Allocate one immutable remote submission directory for both input and job
    # script. Concurrent callers never publish through a shared pathname.
    local REMOTE_JOB_ROOT="${REMOTE}/data/userjobs"
    local MKDIR_COMMAND MKTEMP_COMMAND REMOTE_JOB_DIR REMOTE_JOB_BASENAME
    shell_join MKDIR_COMMAND mkdir -p "${REMOTE_JOB_ROOT}"
    remote "${MKDIR_COMMAND}"
    shell_join MKTEMP_COMMAND mktemp -d \
        "${REMOTE_JOB_ROOT}/${NAME}.XXXXXXXX"
    REMOTE_JOB_DIR="$(remote "${MKTEMP_COMMAND}")"
    REMOTE_JOB_DIR="${REMOTE_JOB_DIR%$'\r'}"
    REMOTE_JOB_BASENAME="${REMOTE_JOB_DIR##*/}"
    local EXPECTED_PREFIX="${NAME}."
    local ALLOCATOR_SUFFIX=""
    if [[ "${REMOTE_JOB_BASENAME}" == "${EXPECTED_PREFIX}"* ]]; then
        ALLOCATOR_SUFFIX="${REMOTE_JOB_BASENAME:${#EXPECTED_PREFIX}}"
    fi
    [[ "${REMOTE_JOB_DIR}" == "${REMOTE_JOB_ROOT}/${REMOTE_JOB_BASENAME}" \
       && "${ALLOCATOR_SUFFIX}" =~ ^[A-Za-z0-9]{8}$ ]] || {
        echo "ERROR: remote submission allocator returned an unsafe path: ${REMOTE_JOB_DIR}" >&2
        exit 1
    }

    # Resolve the cluster-side input path (upload a local file, or use as-is).
    local REMOTE_INPUT
    if [[ "${UPLOAD}" == "1" ]]; then
        local BASE; BASE="$(basename "${INPUT}")"
        if command -v rsync &>/dev/null; then
            rsync -az -- "${INPUT}" "${SSH_TARGET}:${REMOTE_JOB_DIR}/${BASE}"
        else
            scp -- "${INPUT}" "${SSH_TARGET}:${REMOTE_JOB_DIR}/${BASE}"
        fi
        REMOTE_INPUT="${REMOTE_JOB_DIR}/${BASE}"
    else
        REMOTE_INPUT="${INPUT}"          # pre-staged on the cluster
    fi

    # Publish this exact script beside this submission's input. The sbatch path
    # below is immutable for this call rather than shared across callers.
    local REMOTE_JOB_SCRIPT="${REMOTE_JOB_DIR}/cluster_generic.slurm"
    scp -- "${PROJECT_ROOT}/scripts/slurm/jobs/cluster_generic.slurm" \
        "${SSH_TARGET}:${REMOTE_JOB_SCRIPT}"

    # GPU runs need a GRES request (the job file is partition-agnostic)
    local GPU_FLAGS=""
    if [[ "${DEVICE}" == cuda* || "${DEVICE}" == gpu ]]; then
        GPU_FLAGS="--gres=${GPU_GRES}"
    fi

    local EXPORTS="ALL,DTWC_INPUT=${REMOTE_INPUT},DTWC_K=${K},DTWC_SKIP_COLS=${SKIP_COLS}"
    EXPORTS+=",DTWC_METHOD=${METHOD},DTWC_DEVICE=${DEVICE},DTWC_BAND=${BAND},DTWC_NAME=${NAME},DTWC_N_INIT=${N_INIT}"
    EXPORTS+=",DTWC_DTYPE=float64"
    EXPORTS+=",DTWC_MAX_ITER=${MAX_ITER},DTWC_VARIANT=${VARIANT},DTWC_WDTW_G=${WDTW_G},DTWC_ADTW_PENALTY=${ADTW_PENALTY}"
    EXPORTS+=",DTWC_MSM_C=${MSM_C},DTWC_TWE_NU=${TWE_NU},DTWC_TWE_LAMBDA=${TWE_LAMBDA},DTWC_MV_MODE=${MV_MODE}"
    EXPORTS+=",DTWC_MISSING_STRATEGY=${MISSING_STRATEGY},DTWC_METRIC=${METRIC}"
    # Always override an ambient login-shell value inherited through `ALL`.
    # An empty export preserves the CLI as the single source of the default;
    # an explicit value remains byte-for-byte unchanged.
    EXPORTS+=",DTWC_SEED=${SEED}"

    local -a SBATCH_ARGS=(sbatch --parsable)
    [[ -n "${CLUSTER_FLAG}" ]] && SBATCH_ARGS+=("${CLUSTER_FLAG}")
    if [[ -n "${SLURM_EMAIL:-}" ]]; then
        SBATCH_ARGS+=("--mail-type=BEGIN,END,FAIL" "--mail-user=${SLURM_EMAIL}")
    fi
    [[ -n "${GPU_FLAGS}" ]] && SBATCH_ARGS+=("${GPU_FLAGS}")
    SBATCH_ARGS+=("--export=${EXPORTS}" "${REMOTE_JOB_SCRIPT}")

    local JOB_ID
    JOB_ID=$(remote_argv_in_dir "${REMOTE}/src" "${SBATCH_ARGS[@]}")
    echo "  Job ID: ${JOB_ID}"
    echo "  Monitor: bash scripts/slurm/slurm_remote.sh status"
}

cmd_status() {
    banner "SLURM Job Status"
    local -a STATUS_ARGS=(squeue -u "${SLURM_USER}")
    [[ -n "${CLUSTER_FLAG}" ]] && STATUS_ARGS+=("${CLUSTER_FLAG}")
    local STATUS_COMMAND
    shell_join STATUS_COMMAND "${STATUS_ARGS[@]}"
    remote "${STATUS_COMMAND}"
}

cmd_download() {
    banner "Downloading results + logs"
    local LOCAL_RESULTS="${PROJECT_ROOT}/results/slurm"
    mkdir -p "${LOCAL_RESULTS}"

    echo "  Remote: ${SSH_TARGET}:${REMOTE}/src/results/"
    echo "  Local:  ${LOCAL_RESULTS}/"
    echo ""

    if command -v rsync &>/dev/null; then
        rsync -avz --progress -- "${SSH_TARGET}:${REMOTE}/src/results/" "${LOCAL_RESULTS}/"
        echo ""
        echo "  Downloading logs..."
        rsync -avz --progress -- "${SSH_TARGET}:${REMOTE}/src/logs/" "${LOCAL_RESULTS}/logs/" 2>/dev/null || echo "  No logs found."
    else
        scp -r -- "${SSH_TARGET}:${REMOTE}/src/results/." "${LOCAL_RESULTS}/"
        echo ""
        echo "  Downloading logs..."
        scp -r -- "${SSH_TARGET}:${REMOTE}/src/logs/." "${LOCAL_RESULTS}/logs/" 2>/dev/null || echo "  No logs found."
    fi

    echo ""
    echo "  Results downloaded to: ${LOCAL_RESULTS}/"
}

cmd_download_cluster() {
    local NAME="${1:?job name required}"
    local JOB_ID="${2:?job ID required}"
    (( ${#NAME} >= 1 && ${#NAME} <= 128 )) \
        && [[ "${NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
        echo "ERROR: invalid cluster job name: ${NAME}" >&2
        exit 1
    }
    [[ "${JOB_ID}" =~ ^[1-9][0-9]*$ ]] \
        && decimal_leq "${JOB_ID}" "18446744073709551615" || {
        echo "ERROR: job ID must be a positive uint64: ${JOB_ID}" >&2
        exit 1
    }

    local LOCAL_DIR="${PROJECT_ROOT}/results/slurm/${NAME}_${JOB_ID}"
    local LOCAL_TARGET="${LOCAL_DIR}/${NAME}_labels.csv"
    local REMOTE_SOURCE="${REMOTE}/src/results/${NAME}_${JOB_ID}/${NAME}_labels.csv"
    mkdir -p "${LOCAL_DIR}"
    rm -f -- "${LOCAL_TARGET}"
    if command -v rsync &>/dev/null; then
        rsync -az -- "${SSH_TARGET}:${REMOTE_SOURCE}" "${LOCAL_TARGET}"
    else
        scp -- "${SSH_TARGET}:${REMOTE_SOURCE}" "${LOCAL_TARGET}"
    fi
    echo "  Labels downloaded: ${LOCAL_TARGET}"
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
    download-cluster)  cmd_download_cluster "$@" ;;
    ssh)               cmd_ssh "$@" ;;
    interactive)       cmd_interactive ;;
    help|--help|-h)
        echo "Usage: bash scripts/slurm/slurm_remote.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  test              Test SSH connection"
        echo "  upload            Upload source + test data"
        echo "  build [profile] | build --profile <profile>"
        echo "                    Profiles: arc, htc-cpu, htc-gpu, htc-v4, h100, grace"
        echo "  submit-cpu        Submit CPU test job"
        echo "  submit-gpu        Submit GPU test job"
        echo "  submit-checkpoint Submit checkpoint/resume test"
        echo "  submit-parquet    Submit Parquet I/O test"
        echo "  submit-benchmark-cpu  Submit full UCR benchmark (CPU, ~12h)"
        echo "  submit-benchmark-gpu [type]  GPU type: a100, l40s, or h100"
        echo "  submit-cluster <input> <k> [method] [device] [band] [name] [skip_cols] [upload] [n_init] [seed] [max_iter] [variant] [variant params...] [mv_mode] [missing_strategy] [metric]"
        echo "                    Upload an arbitrary input file + cluster it (device='hpc' path)"
        echo "  status            Show SLURM queue"
        echo "  download          Download results + logs"
        echo "  download-cluster <name> <job-id>  Download exact clustering labels"
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
