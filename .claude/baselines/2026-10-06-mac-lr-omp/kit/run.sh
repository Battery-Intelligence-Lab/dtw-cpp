#!/bin/bash
# The lr-omp timing kit's one command:   ./run.sh [smoke]
#   ./run.sh                the sweep: N in {100, 200, 400, 800, 1600, 3200}, two regimes (noise, line), HiGHS OFF and ON, 3 repeats,
#                           base b36fad43 and head interleaved; prints the root, the LR phase, Problem::cluster() and dtwc_cl times
#   ./run.sh smoke          N in {100, 200}, 1 repeat, short repeats inside the probe: a check that the kit runs (numbers are not evidence)
# Environment (see run.py): REPEATS, SIZES, REGIMES, FLAVORS, THREADS, HEAD_MIN_N (the head probe's threshold knob), MIN_MS, E2E.
# Needs the machine quiet: nothing else may build or run. First time (or after the product changes): `uv run --no-project python build_kit.py`
# builds the four probes (base / head x HiGHS on / off) from the trees named at the top of build_kit.py.
set -eu
cd "$(dirname "$0")"
if [[ "${1:-}" == "smoke" ]]; then
  export SIZES="${SIZES:-100 200}" REPEATS="${REPEATS:-1}" MIN_MS="${MIN_MS:-20}" RUN_TAG=smoke
fi
uv run --no-project python build_kit.py --if-stale
uv run --no-project python gen_sweep.py data ${SIZES:-}
exec uv run --no-project python run.py
