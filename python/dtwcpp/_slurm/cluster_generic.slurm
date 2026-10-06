#!/bin/bash
#SBATCH --job-name=dtwc-cluster
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/cluster_%j.out
#SBATCH --error=logs/cluster_%j.err
#
# DTWC++ clustering job: dtwc_cl runs the job.toml of a run directory.
# `slurm_remote.sh submit-job` uploads the directory, publishes this script in
# it, adds the GPU request of a GPU run, and exports
#   DTWC_JOB    the run directory: job.toml, and input.tsv for series sent from memory
#   DTWC_BUILD  the build that runs it, build-${DTWC_BUILD}/bin/dtwc_cl: htc-cpu,
#               htc-gpu, or the GPU type the run names (a100, l40s, ...)
# The output directory, results/cluster_<job id>, is the one command-line value
# (it beats the file); download-cluster fetches the labels from it.

set -euo pipefail

DTWC_JOB="${DTWC_JOB:?DTWC_JOB not set}"
DTWC_BIN="${SLURM_SUBMIT_DIR}/build-${DTWC_BUILD:?DTWC_BUILD not set}/bin/dtwc_cl"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/cluster_${SLURM_JOB_ID}"

module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
[[ "${DTWC_BUILD}" == htc-cpu ]] || module load CUDA/12.4.0 2>/dev/null || module load cuda 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "DTWC++ job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-?}: ${DTWC_BIN} --config ${DTWC_JOB}/job.toml"
cat "${DTWC_JOB}/job.toml"
mkdir -p "${RESULTS_DIR}"
cd "${DTWC_JOB}"  # job.toml names input.tsv relative to its directory
"${DTWC_BIN}" --config "${DTWC_JOB}/job.toml" --output "${RESULTS_DIR}"
