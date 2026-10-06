#!/usr/bin/env bash
# Generalized SLURM remote helper — upload, build, submit, download.
#
# Requires: ssh, rsync (both available in Git Bash on Windows).
# Reads configuration from .env in the project directory: $DTWC_REPO_ROOT, else
# the working directory. This file ships inside the dtwcpp package, which is how
# device='hpc' finds it from an installed wheel; in a checkout,
# scripts/slurm/slurm_remote.sh forwards here with the checkout as the project.
# upload and the fixed test/benchmark jobs send files from the source checkout
# this copy sits in (its python/dtwcpp/_slurm/), and refuse to run outside one.
#
# Usage:
#   bash scripts/slurm/slurm_remote.sh test
#   bash scripts/slurm/slurm_remote.sh upload
#   bash scripts/slurm/slurm_remote.sh build [profile] [--gpu-device <type>]
#   bash scripts/slurm/slurm_remote.sh build --profile <profile> [--gpu-device <type>]
#   bash scripts/slurm/slurm_remote.sh submit-smoke cpu|gpu|checkpoint|parquet
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-cpu
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-gpu [type]
#   bash scripts/slurm/slurm_remote.sh submit-job <rundir> [--gpu | --gpu-device <type>]
#   bash scripts/slurm/slurm_remote.sh status
#   bash scripts/slurm/slurm_remote.sh download
#   bash scripts/slurm/slurm_remote.sh download-cluster <job-id>
#   bash scripts/slurm/slurm_remote.sh ssh "command"
#   bash scripts/slurm/slurm_remote.sh interactive

set -euo pipefail

# ── Locate project root and load .env ────────────────────────────────────
# The project directory (.env, results/) is not this script's
# directory: the same DTWC_REPO_ROOT-else-working-directory rule as dtwcpp.
# cluster_generic.slurm ships beside this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH='' cd -- "${DTWC_REPO_ROOT:-.}" && pwd)"
# The checkout whose files upload and the fixed jobs send: the one this copy
# sits in, taken from our own location and never from the environment.
SOURCE_ROOT="$(CDPATH='' cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env not found at ${ENV_FILE}"
    echo "       Create it with SLURM_USER, SLURM_HOST and SLURM_REMOTE_BASE"
    echo "       (template: scripts/slurm/env.example in a source checkout)."
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
[[ -z "${SLURM_GPU_GRES:-}" ]] || { echo "ERROR: SLURM_GPU_GRES in .env is no longer read: a run names its GPU with gpu_device= (Python) or --gpu-device (slurm_remote.sh); remove the line" >&2; exit 1; }

SSH_TARGET="${SLURM_USER}@${SLURM_HOST}"
REMOTE="${SLURM_REMOTE_BASE}"
PARTITION="${SLURM_PARTITION}"
CLUSTER_FLAG=""
if [[ -n "${SLURM_CLUSTER:-}" ]]; then
    CLUSTER_FLAG="--clusters=${SLURM_CLUSTER}"
fi

# ── Helper ───────────────────────────────────────────────────────────────
remote() {
    # Run command on the remote host via SSH
    ssh "${SSH_TARGET}" "$@"
}

require_checkout() {
    # An installed package has no source tree to send; say so before any SSH.
    [[ -d "${SOURCE_ROOT}/dtwc" && -d "${SOURCE_ROOT}/scripts/slurm/jobs" ]] || {
        echo "ERROR: '${CMD}' sends files from a DTWC++ source checkout, and this" >&2
        echo "       wrapper is not inside one (${SCRIPT_DIR})." >&2
        echo "       Run it from a clone: bash scripts/slurm/slurm_remote.sh ${CMD}" >&2
        exit 1
    }
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

# GPU_REQUEST becomes the sbatch arguments that ask for a GPU of type $1, or,
# called without an argument, for any GPU at or above the CUDA floor (the '*'
# line), as gpu_devices.txt beside this script lists them; dtwcpp._hpc reads
# the same table. A type below the floor, an empty one, or one the table does
# not name, is refused before any SSH.
gpu_request() {
    local NAME CAPABILITY REQUEST
    if (( $# == 0 )) || [[ "$1" =~ ^[a-z0-9]+$ ]]; then
        while read -r NAME CAPABILITY REQUEST; do
            REQUEST="${REQUEST%$'\r'}"  # a CRLF copy of the table reads as the LF one
            [[ "${NAME}" == "${1-*}" ]] || continue
            [[ "${REQUEST}" != - ]] || {
                echo "ERROR: GPU type '${1-}' has CUDA compute capability ${CAPABILITY}, below the 8.0 DTWC++ needs" >&2
                exit 1
            }
            read -r -a GPU_REQUEST <<< "${REQUEST}"
            return 0
        done < "${SCRIPT_DIR}/gpu_devices.txt"
    fi
    echo "ERROR: unknown GPU type '${1-}'; gpu_devices.txt lists the types" >&2
    exit 1
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
    require_checkout
    banner "Uploading source + test data"
    echo "  Local:  ${SOURCE_ROOT}"
    echo "  Remote: ${SSH_TARGET}:${REMOTE}"
    echo ""

    # Create remote directory structure
    # Jobs submitted from src/ write their logs to src/logs/, which SLURM does not create.
    remote_argv mkdir -p \
        "${REMOTE}/src/dtwc" "${REMOTE}/src/cmake" \
        "${REMOTE}/src/scripts/slurm/jobs" "${REMOTE}/src/logs" \
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
    _upload_dir "${SOURCE_ROOT}/dtwc" "${REMOTE}/src/dtwc"

    echo ""
    echo "[2/5] Uploading cmake/ + build files..."
    _upload_dir "${SOURCE_ROOT}/cmake" "${REMOTE}/src/cmake"
    _upload_dir "${SOURCE_ROOT}/scripts/slurm" "${REMOTE}/src/scripts/slurm"
    for f in CMakeLists.txt CMakePresets.json VERSION; do
        [[ -f "${SOURCE_ROOT}/${f}" ]] && _upload_file "${SOURCE_ROOT}/${f}" "${REMOTE}/src/${f}"
    done

    # Upload test datasets
    echo ""
    echo "[3/5] Uploading Coffee dataset..."
    local COFFEE="${SOURCE_ROOT}/data/benchmark/UCRArchive_2018/Coffee"
    if [[ -d "${COFFEE}" ]]; then
        _upload_dir "${COFFEE}" "${REMOTE}/data/Coffee"
    else
        echo "  SKIP: ${COFFEE} not found"
    fi

    echo ""
    echo "[4/5] Uploading Beef dataset..."
    local BEEF="${SOURCE_ROOT}/data/benchmark/UCRArchive_2018/Beef"
    if [[ -d "${BEEF}" ]]; then
        _upload_dir "${BEEF}" "${REMOTE}/data/Beef"
    else
        echo "  SKIP: ${BEEF} not found"
    fi

    echo ""
    echo "[5/5] Uploading dummy test data..."
    local DUMMY="${SOURCE_ROOT}/data/dummy"
    if [[ -d "${DUMMY}" ]]; then
        remote_argv mkdir -p "${REMOTE}/data/dummy"
        _upload_dir "${DUMMY}" "${REMOTE}/data/dummy"
    else
        echo "  SKIP: ${DUMMY} not found"
    fi

    echo ""
    echo "  Upload complete."
}

build_syntax_error() {
    echo "ERROR: build profile syntax is 'build [--profile] <profile> [--gpu-device <type>]'" >&2
    exit 1
}

cmd_build() {
    local PROFILE="" GPU_DEVICE="" GPU_GIVEN=""
    while (( $# )); do
        case "$1" in
            --profile|--gpu-device)
                (( $# >= 2 )) || build_syntax_error
                if [[ "$1" == --profile ]]; then PROFILE="$2"; else GPU_DEVICE="$2" GPU_GIVEN=1; fi
                shift 2
                ;;
            *)
                [[ -z "${PROFILE}" ]] || build_syntax_error
                PROFILE="$1"
                shift
                ;;
        esac
    done
    PROFILE="${PROFILE:-htc-cpu}"
    case "${PROFILE}" in
        arc|htc-cpu|htc-gpu|htc-v4|h100|grace) ;;
        *)
            echo "ERROR: unsupported build profile '${PROFILE}'; expected arc, htc-cpu, htc-gpu, htc-v4, h100, or grace" >&2
            exit 1
            ;;
    esac
    # Without --gpu-device the build runs on an interactive node, without a GPU,
    # and is portable. With it, the build asks for that GPU as a job does, so
    # build-arc.sh runs on such a node and builds its CUDA architecture into
    # build-<type>, the build that type's jobs run. The CPU code stays portable
    # (DTWC_NATIVE_CPU=OFF): one GPU type's nodes have different CPUs.
    local BUILD_PARTITION="interactive" BUILD_DIR="build-${PROFILE}" NATIVE_CPU="ON"
    local -a GPU_REQUEST=()
    if [[ -n "${GPU_GIVEN}" ]]; then
        [[ "${PROFILE}" == htc-gpu || "${PROFILE}" == h100 ]] || {
            echo "ERROR: --gpu-device needs a GPU profile, htc-gpu or h100, not '${PROFILE}'" >&2
            exit 1
        }
        gpu_request "${GPU_DEVICE}"
        BUILD_PARTITION="${PARTITION}"
        BUILD_DIR="build-${GPU_DEVICE}"
        NATIVE_CPU="OFF"
    fi
    banner "Building on cluster (profile: ${PROFILE}${GPU_DEVICE:+, GPU ${GPU_DEVICE}}, into ${BUILD_DIR})"

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
    local EXPORTS="ALL,DTWC_REMOTE_BASE=${REMOTE},DTWC_BUILD_PROFILE=${PROFILE},DTWC_BUILD_DIR=${BUILD_DIR},DTWC_NATIVE_CPU=${NATIVE_CPU}"
    local -a SBATCH_ARGS=(
        sbatch --parsable "--partition=${BUILD_PARTITION}" --time=01:00:00
        --cpus-per-task=8 --mem-per-cpu=4G --job-name=dtwc-build
        "--output=${REMOTE}/logs/build_%j.out"
        "--error=${REMOTE}/logs/build_%j.err"
        "--export=${EXPORTS}"
    )
    [[ -n "${CLUSTER_FLAG}" ]] && SBATCH_ARGS+=("${CLUSTER_FLAG}")
    SBATCH_ARGS+=(${GPU_REQUEST[@]+"${GPU_REQUEST[@]}"})
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

# _submit_job <job file> <label> <build> [sbatch arguments...]: the job runs
# build-<build>/bin/dtwc_cl, which must be on the cluster.
_submit_job() {
    local SLURM_FILE="$1"
    local LABEL="$2"
    local BIN="${REMOTE}/src/build-$3/bin/dtwc_cl"
    shift 3

    require_checkout
    banner "Submitting ${LABEL}"

    remote_argv test -x "${BIN}" || {
        echo "  ERROR: no dtwc_cl at ${BIN} (or ssh failed). Build it first: bash scripts/slurm/slurm_remote.sh build" >&2
        exit 1
    }
    echo "  Binary: ${BIN}"

    # Upload the latest job script
    scp -- "${SOURCE_ROOT}/${SLURM_FILE}" "${SSH_TARGET}:${REMOTE}/src/${SLURM_FILE}"

    local -a SBATCH_ARGS=(sbatch --parsable)
    [[ -n "${CLUSTER_FLAG}" ]] && SBATCH_ARGS+=("${CLUSTER_FLAG}")
    if [[ -n "${SLURM_EMAIL:-}" ]]; then
        SBATCH_ARGS+=("--mail-type=BEGIN,END,FAIL" "--mail-user=${SLURM_EMAIL}")
    fi
    SBATCH_ARGS+=("$@" "${SLURM_FILE}")
    local JOB_ID
    JOB_ID=$(remote_argv_in_dir "${REMOTE}/src" "${SBATCH_ARGS[@]}")
    echo "  Job ID: ${JOB_ID}"
    echo "  Monitor: bash scripts/slurm/slurm_remote.sh status"
}

# One smoke job, smoke.slurm, in the mode it is given; the GPU mode asks for a
# GPU at or above the CUDA floor, so it never lands on a refused V100.
cmd_submit_smoke() {
    local BUILD="htc-cpu"
    local -a GPU_REQUEST=()
    case "$#:${1:-}" in
        1:cpu|1:checkpoint|1:parquet) ;;
        1:gpu) BUILD="htc-gpu"; gpu_request ;;
        *)
            echo "ERROR: submit-smoke syntax is 'submit-smoke cpu|gpu|checkpoint|parquet'" >&2
            exit 1
            ;;
    esac
    _submit_job "scripts/slurm/jobs/smoke.slurm" "smoke test ($1)" "${BUILD}" \
        "--export=ALL,MODE=$1" ${GPU_REQUEST[@]+"${GPU_REQUEST[@]}"}
}

cmd_submit_benchmark_cpu() {
    _submit_job "scripts/slurm/jobs/ucr_benchmark_cpu.slurm" "UCR benchmark (CPU)" "htc-cpu"
}

cmd_submit_benchmark_gpu() {
    (( $# <= 1 )) || {
        echo "ERROR: benchmark GPU type accepts at most one value" >&2
        exit 1
    }
    local -a GPU_REQUEST=()
    gpu_request ${1+"$1"}
    _submit_job "scripts/slurm/jobs/ucr_benchmark_gpu.slurm" "UCR benchmark (GPU${1:+: $1})" "htc-gpu" "${GPU_REQUEST[@]}"
}

# Run a directory dtwcpp's device='hpc' wrote (job.toml, and input.tsv for
# series sent from memory): upload it to a fresh directory on the cluster and
# submit cluster_generic.slurm there, which runs dtwc_cl --config job.toml.
# --gpu asks for any GPU at the CUDA floor and runs build-htc-gpu;
# --gpu-device <type> asks for that GPU and runs build-<type>; neither, a CPU
# job on build-htc-cpu.
cmd_submit_job() {
    local RUNDIR="${1:-}"
    local BUILD="htc-cpu"
    local -a GPU_REQUEST=()
    case "$#:${2:-}" in
        1:) ;;
        2:--gpu) BUILD="htc-gpu"; gpu_request ;;
        3:--gpu-device) BUILD="$3"; gpu_request "$3" ;;
        *)
            echo "ERROR: submit-job syntax is 'submit-job <rundir> [--gpu | --gpu-device <type>]'" >&2
            exit 1
            ;;
    esac
    # The run directory crosses rsync's argv and its host:path grammar.
    [[ "${RUNDIR}" =~ ^[A-Za-z0-9_./+@%=-]+$ && "${RUNDIR}" != -* ]] || {
        echo "ERROR: run directory must be ASCII letters, digits and _./+@%=-, not starting with '-': ${RUNDIR}" >&2
        exit 1
    }
    [[ -f "${RUNDIR}/job.toml" ]] || {
        echo "ERROR: no job.toml in run directory ${RUNDIR}" >&2
        exit 1
    }

    banner "Submitting clustering job (${RUNDIR})"

    local BIN="${REMOTE}/src/build-${BUILD}/bin/dtwc_cl" BUILD_COMMAND="build ${BUILD}"
    [[ "${BUILD}" == htc-* ]] || BUILD_COMMAND="build htc-gpu --gpu-device ${BUILD}"
    remote_argv test -x "${BIN}" || {
        echo "  ERROR: no dtwc_cl at ${BIN} (or ssh failed). Build it: bash scripts/slurm/slurm_remote.sh ${BUILD_COMMAND}" >&2
        exit 1
    }

    # One fresh remote directory per submission holds the run and its job
    # script, so concurrent callers never publish through a shared pathname.
    local REMOTE_JOB_ROOT="${REMOTE}/data/userjobs"
    local MKDIR_COMMAND MKTEMP_COMMAND REMOTE_JOB_DIR REMOTE_JOB_BASENAME
    shell_join MKDIR_COMMAND mkdir -p "${REMOTE_JOB_ROOT}" "${REMOTE}/src/logs"
    remote "${MKDIR_COMMAND}"
    shell_join MKTEMP_COMMAND mktemp -d "${REMOTE_JOB_ROOT}/job.XXXXXXXX"
    REMOTE_JOB_DIR="$(remote "${MKTEMP_COMMAND}")"
    REMOTE_JOB_DIR="${REMOTE_JOB_DIR%$'\r'}"
    REMOTE_JOB_BASENAME="${REMOTE_JOB_DIR##*/}"
    [[ "${REMOTE_JOB_DIR}" == "${REMOTE_JOB_ROOT}/${REMOTE_JOB_BASENAME}" \
       && "${REMOTE_JOB_BASENAME}" =~ ^job\.[A-Za-z0-9]{8}$ ]] || {
        echo "ERROR: remote submission allocator returned an unsafe path: ${REMOTE_JOB_DIR}" >&2
        exit 1
    }

    if command -v rsync &>/dev/null; then
        rsync -az -- "${RUNDIR}/" "${SSH_TARGET}:${REMOTE_JOB_DIR}/"
    else
        scp -r -- "${RUNDIR}/." "${SSH_TARGET}:${REMOTE_JOB_DIR}/"
    fi
    scp -- "${SCRIPT_DIR}/cluster_generic.slurm" "${SSH_TARGET}:${REMOTE_JOB_DIR}/cluster_generic.slurm"

    # The .env partition, which a GPU build (build --gpu-device) uses too.
    local -a SBATCH_ARGS=(sbatch --parsable "--partition=${PARTITION}")
    [[ -n "${CLUSTER_FLAG}" ]] && SBATCH_ARGS+=("${CLUSTER_FLAG}")
    if [[ -n "${SLURM_EMAIL:-}" ]]; then
        SBATCH_ARGS+=("--mail-type=BEGIN,END,FAIL" "--mail-user=${SLURM_EMAIL}")
    fi
    # ${A[@]+"${A[@]}"}: bash < 4.4 (macOS ships 3.2) calls an empty array unbound under set -u.
    SBATCH_ARGS+=(${GPU_REQUEST[@]+"${GPU_REQUEST[@]}"}
        "--export=ALL,DTWC_JOB=${REMOTE_JOB_DIR},DTWC_BUILD=${BUILD}"
        "${REMOTE_JOB_DIR}/cluster_generic.slurm")

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
    local JOB_ID="${1:?job ID required}"
    [[ "${JOB_ID}" =~ ^[1-9][0-9]*$ ]] \
        && decimal_leq "${JOB_ID}" "18446744073709551615" || {
        echo "ERROR: job ID must be a positive uint64: ${JOB_ID}" >&2
        exit 1
    }

    # A submit-job run writes results/cluster_<job id>/<name>_labels.csv;
    # only the job ID crosses the shell, so a run's name may be any file name.
    local LOCAL_DIR="${PROJECT_ROOT}/results/slurm/cluster_${JOB_ID}"
    local REMOTE_DIR="${REMOTE}/src/results/cluster_${JOB_ID}"
    mkdir -p "${LOCAL_DIR}"
    rm -f -- "${LOCAL_DIR}"/*_labels.csv
    if command -v rsync &>/dev/null; then
        rsync -az --include='*_labels.csv' --exclude='*' -- "${SSH_TARGET}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
    else
        scp -- "${SSH_TARGET}:${REMOTE_DIR}/*_labels.csv" "${LOCAL_DIR}/"
    fi
    echo "  Labels downloaded to: ${LOCAL_DIR}"
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
    submit-smoke)      cmd_submit_smoke "$@" ;;
    submit-benchmark-cpu) cmd_submit_benchmark_cpu ;;
    submit-benchmark-gpu) cmd_submit_benchmark_gpu "$@" ;;
    submit-job)        cmd_submit_job "$@" ;;
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
        echo "  build [profile] | build --profile <profile>   [--gpu-device <type>]"
        echo "                    Profiles: arc, htc-cpu, htc-gpu, htc-v4, h100, grace;"
        echo "                    --gpu-device builds on that GPU's node into build-<type>"
        echo "  submit-smoke cpu|gpu|checkpoint|parquet"
        echo "                    Submit the smoke test in that mode (scripts/slurm/jobs/smoke.slurm)"
        echo "  submit-benchmark-cpu  Submit full UCR benchmark (CPU, ~12h)"
        echo "  submit-benchmark-gpu [type]  GPU type from gpu_devices.txt (default: any of 8.0+)"
        echo "  submit-job <rundir> [--gpu | --gpu-device <type>]"
        echo "                    Upload a run directory (job.toml) and cluster it (device='hpc' path)"
        echo "  status            Show SLURM queue"
        echo "  download          Download results + logs"
        echo "  download-cluster <job-id>  Download a submit-job run's labels"
        echo "  ssh \"command\"     Run arbitrary command on cluster"
        echo "  interactive       Print interactive session guide"
        echo ""
        echo "Configuration: .env in \$DTWC_REPO_ROOT, else the working directory"
        echo "(scripts/slurm/slurm_remote.sh always uses its own checkout;"
        echo " template: scripts/slurm/env.example in a source checkout)"
        ;;
    *)
        echo "Unknown command: ${CMD}"
        echo "Run 'bash scripts/slurm/slurm_remote.sh help' for usage."
        exit 1
        ;;
esac
