#!/usr/bin/env bash
# @file slurm_remote.sh
# @brief Developer SLURM remote submission wrapper for source checkouts on Oxford ARC.
# @author Volkan Kumtepeli
# @date 09 Apr 2026

# Developer entry point for the SLURM wrapper, run from a source checkout.
#
# upload, build, submit-smoke and the fixed benchmark jobs send this checkout's
# files to the cluster, or build and run what upload sent, so they live here and
# not in the wrapper that ships inside the Python package
# (python/dtwcpp/_slurm/slurm_remote.sh, which device='hpc' runs from an
# installed wheel). Every other command runs that wrapper. The project (.env,
# results/) is always this checkout: an ambient DTWC_REPO_ROOT, which steers
# only Python's device='hpc', is overridden.
#
# Usage:
#   bash scripts/slurm/slurm_remote.sh upload
#   bash scripts/slurm/slurm_remote.sh build [profile] [--gpu-device <type>]
#   bash scripts/slurm/slurm_remote.sh build --profile <profile> [--gpu-device <type>]
#   bash scripts/slurm/slurm_remote.sh submit-smoke cpu|gpu|checkpoint|parquet
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-cpu
#   bash scripts/slurm/slurm_remote.sh submit-benchmark-gpu [type]
#   bash scripts/slurm/slurm_remote.sh help
set -euo pipefail
HERE="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The checkout whose files these commands send: the one this script sits in.
SOURCE_ROOT="$(CDPATH='' cd -- "${HERE}/../.." && pwd)"
export DTWC_REPO_ROOT="${SOURCE_ROOT}"
WRAPPER="${SOURCE_ROOT}/python/dtwcpp/_slurm/slurm_remote.sh"

case "${1:-help}" in
    upload|build|submit-smoke|submit-benchmark-cpu|submit-benchmark-gpu) ;;
    help|--help|-h)
        bash "${WRAPPER}" help
        echo ""
        echo "Also here, sending this checkout's files to the cluster:"
        echo "  upload            Upload source + test data"
        echo "  build [profile] | build --profile <profile>   [--gpu-device <type>]"
        echo "                    Profiles: arc, htc-cpu, htc-gpu, htc-v4, h100, grace;"
        echo "                    --gpu-device builds on that GPU's node into build-<type>"
        echo "  submit-smoke cpu|gpu|checkpoint|parquet"
        echo "                    Submit the smoke test in that mode (scripts/slurm/jobs/smoke.slurm)"
        echo "  submit-benchmark-cpu  Submit full UCR benchmark (CPU, ~12h)"
        echo "  submit-benchmark-gpu [type]  GPU type from gpu_devices.txt (default: any of 8.0+)"
        exit 0
        ;;
    *) exec bash "${WRAPPER}" "$@" ;;
esac

# The wrapper's .env, its validation and its helpers; sourced, it stops before its own dispatch.
source "${WRAPPER}"

# ── Commands ─────────────────────────────────────────────────────────────

cmd_upload() {
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
    for f in CMakeLists.txt CMakePresets.json VERSION LICENSE README.md THIRD_PARTY_LICENSES.md; do
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
        GPU_DEVICE="$(printf '%s' "${GPU_DEVICE}" | tr '[:upper:]' '[:lower:]')"  # build-<type>, as dtwcpp._hpc names it
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
if [[ -n "${DTWC_MODULES:-}" ]]; then
    IFS=: read -r -a _DTWC_MOD_LIST <<< "${DTWC_MODULES}"
    _HAS_CMAKE=0
    for _MOD in "${_DTWC_MOD_LIST[@]}"; do
        [[ "${_MOD,,}" == cmake* ]] && _HAS_CMAKE=1
    done
    if (( ! _HAS_CMAKE )); then
        module load CMake/3.27.6 2>/dev/null || module load cmake 2>/dev/null || true
    fi
    for _MOD in "${_DTWC_MOD_LIST[@]}"; do
        [[ -n "${_MOD}" ]] && module load "${_MOD}"
    done
else
    module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0 2>/dev/null || true
    module load cmake gcc cuda 2>/dev/null || true
fi
cd "${DTWC_REMOTE_BASE}/src"
# Disable testing (tests/ not uploaded); Arrow off until include path fix
export DTWC_BUILD_TESTING=OFF
export DTWC_ENABLE_ARROW=OFF
source scripts/slurm/build-arc.sh "${DTWC_BUILD_PROFILE}"
'

    echo "  Submitting build job..."
    local EXPORTS="ALL,DTWC_REMOTE_BASE=${REMOTE},DTWC_BUILD_PROFILE=${PROFILE},DTWC_BUILD_DIR=${BUILD_DIR},DTWC_NATIVE_CPU=${NATIVE_CPU}${MODULES_EXPORT}"
    local -a SBATCH_ARGS=(
        sbatch --parsable "--partition=${BUILD_PARTITION}" --time=01:00:00
        --cpus-per-task=8 --mem-per-cpu=4G --job-name=dtwc-build
        "--output=${REMOTE}/logs/build_%j.out"
        "--error=${REMOTE}/logs/build_%j.err"
        "--export=${EXPORTS}"
    )
    [[ -n "${CLUSTER_FLAG}" ]] && SBATCH_ARGS+=("${CLUSTER_FLAG}")
    [[ -n "${ACCOUNT_FLAG}" ]] && SBATCH_ARGS+=("${ACCOUNT_FLAG}")
    [[ -n "${QOS_FLAG}" ]] && SBATCH_ARGS+=("${QOS_FLAG}")
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
    [[ -n "${ACCOUNT_FLAG}" ]] && SBATCH_ARGS+=("${ACCOUNT_FLAG}")
    [[ -n "${QOS_FLAG}" ]] && SBATCH_ARGS+=("${QOS_FLAG}")
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

# ── Dispatch ─────────────────────────────────────────────────────────────

CMD="$1"
shift

case "${CMD}" in
    upload)               cmd_upload ;;
    build)                cmd_build "$@" ;;
    submit-smoke)         cmd_submit_smoke "$@" ;;
    submit-benchmark-cpu) cmd_submit_benchmark_cpu ;;
    submit-benchmark-gpu) cmd_submit_benchmark_gpu "$@" ;;
esac
