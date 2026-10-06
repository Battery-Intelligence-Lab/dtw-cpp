#!/bin/bash
#SBATCH --job-name=dtwc-smoke
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#
# DTWC++ smoke test on the UCR data `slurm_remote.sh upload` sends. MODE, which
# `slurm_remote.sh submit-smoke <mode>` exports, picks the test:
#   cpu         Coffee k=2 and Beef k=5 on build-htc-cpu
#   gpu         Coffee k=2 on build-htc-gpu, GPU precision fp32 then fp64; the
#               wrapper asks for a GPU of compute capability 8.0 or newer
#   checkpoint  a run saves its distance checkpoint, and a second run resumes from it
#   parquet     Coffee read from Parquet (benchmarks/convert_ucr.py writes it) gives
#               the labels of Coffee read from TSV; needs a build with Arrow
# Any failure fails the job. Results: results/smoke_<mode>_<job id>/.

set -euo pipefail

MODE="${MODE:?MODE not set: cpu, gpu, checkpoint or parquet}"
case "${MODE}" in
    cpu|checkpoint|parquet) BUILD="htc-cpu" ;;
    gpu) BUILD="htc-gpu" ;;
    *) echo "ERROR: unknown MODE '${MODE}': cpu, gpu, checkpoint or parquet" >&2; exit 1 ;;
esac

WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/smoke_${MODE}_${SLURM_JOB_ID}"
DTWC_BIN="${SLURM_SUBMIT_DIR}/build-${BUILD}/bin/dtwc_cl"
mkdir -p "${WORKDIR}" "${RESULTS_DIR}"

module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
[[ "${MODE}" != gpu ]] || module load CUDA/12.4.0 2>/dev/null || module load cuda 2>/dev/null || true
[[ "${MODE}" != parquet ]] || module load Arrow/15.0.0 2>/dev/null || module load arrow 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# The data sits beside src/ (the upload layout) or inside it.
if [[ -d "${SLURM_SUBMIT_DIR}/../data/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/../data"
elif [[ -d "${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018/Coffee" ]]; then
    DATA_DIR="${SLURM_SUBMIT_DIR}/data/benchmark/UCRArchive_2018"
else
    echo "ERROR: Test data not found. Run 'slurm_remote.sh upload' first." >&2
    exit 1
fi
[[ -x "${DTWC_BIN}" ]] || { echo "ERROR: Binary not found: ${DTWC_BIN}" >&2; exit 1; }
cp -r "${DATA_DIR}/Coffee" "${WORKDIR}/Coffee"

echo "================================================================"
echo "  DTWC++ smoke test: ${MODE}"
echo "  Job ID : ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}   CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Binary : ${DTWC_BIN}"
echo "  Date   : $(date -Iseconds)"
echo "================================================================"
[[ "${MODE}" != gpu ]] || nvidia-smi || echo "WARNING: nvidia-smi not available"

# run <output> <dtwc_cl arguments...>: one PAM run into results/<output>, timed.
run() {
    local OUT="$1" T_START
    shift
    T_START=$(date +%s.%N)
    echo ""
    echo "── ${OUT} ──"
    "${DTWC_BIN}" --dtype float64 --method pam --verbose --output "${RESULTS_DIR}/${OUT}" "$@"
    echo "${OUT}: $(echo "$(date +%s.%N) - ${T_START}" | bc)s"
}

COFFEE=(--input "${WORKDIR}/Coffee/Coffee_TRAIN.tsv" --skip-cols 1 -k 2 --name coffee)
case "${MODE}" in
    cpu)
        cp -r "${DATA_DIR}/Beef" "${WORKDIR}/Beef"
        run coffee_k2 "${COFFEE[@]}"
        run beef_k5 --input "${WORKDIR}/Beef/Beef_TRAIN.tsv" --skip-cols 1 -k 5 --name beef
        ;;
    gpu)
        run coffee_k2_f32 "${COFFEE[@]}" --device gpu --gpu-precision f32
        run coffee_k2_f64 "${COFFEE[@]}" --device gpu --gpu-precision f64
        ;;
    checkpoint)
        run run1 "${COFFEE[@]}" --checkpoint "${WORKDIR}/ckpt"
        [[ -s "${WORKDIR}/ckpt/coffee.dtwm" ]] || {
            echo "ERROR: the run saved no distance checkpoint ${WORKDIR}/ckpt/coffee.dtwm" >&2
            exit 1
        }
        run run2 "${COFFEE[@]}" --checkpoint "${WORKDIR}/ckpt" | tee "${RESULTS_DIR}/run2.log"
        grep -q "Resumed from checkpoint" "${RESULTS_DIR}/run2.log" || {
            echo "ERROR: the second run did not resume from ${WORKDIR}/ckpt" >&2
            exit 1
        }
        ;;
    parquet)
        run tsv "${COFFEE[@]}"
        run parquet --input "${DATA_DIR}/Coffee/Coffee_TRAIN.parquet" -k 2 --name coffee
        # The clusters, not the series names, which the two readers give differently.
        diff <(cut -d, -f2 "${RESULTS_DIR}/tsv/coffee_labels.csv") \
             <(cut -d, -f2 "${RESULTS_DIR}/parquet/coffee_labels.csv")
        ;;
esac

echo ""
echo "  Smoke test ${MODE} passed — $(date -Iseconds). Results: ${RESULTS_DIR}"
