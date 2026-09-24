#!/usr/bin/env bash
# Developer entry point for the SLURM wrapper. The wrapper itself ships inside
# the Python package so that device='hpc' works from an installed wheel; see its
# header for usage. The project (.env, results/) is always this checkout: an
# ambient DTWC_REPO_ROOT, which steers only Python's device='hpc', is overridden.
set -euo pipefail
HERE="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="$(CDPATH='' cd -- "${HERE}/../.." && pwd)"
export DTWC_REPO_ROOT="${CHECKOUT}"
exec bash "${CHECKOUT}/python/dtwcpp/_slurm/slurm_remote.sh" "$@"
