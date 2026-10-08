#!/bin/sh
# cli_runs.sh <dtwc_cl.exe> <out dir>: the lanes routes through the real CLI (-k 3 -m pam), 16 runs:
# {conformance_series.csv 27 x 16, data/dummy cut to its first 1000 samples 25 x 1000} x {float64, float32}
# x {l1, squared_euclidean} x {unbanded, band 3 | 100}. Each run's four output files go to <out>/<run>/;
# stdout (minus its two wall-clock lines) to <out>/<run>.stdout, exit code appended.
# REPO (default: the XL worktree) supplies the inputs; data/dummy1000 next to this script is cut from its data/dummy.
set -u
CL=$1; OUT=$2
REPO=${REPO:-/c/D/git/wt/XL}
cd "$(dirname "$0")"
if [ ! -d data/dummy1000 ]; then
  mkdir -p data/dummy1000
  for f in "$REPO"/data/dummy/*.csv; do head -n 1001 "$f" > "data/dummy1000/$(basename "$f")"; done
fi
mkdir -p "$OUT"
for ds in conf dummy; do
  if [ $ds = conf ]; then IN=$REPO/tests/conformance/data/conformance_series.csv; EXTRA=""; BAND=3
  else IN=data/dummy1000; EXTRA="--skip-rows 1 --skip-cols 1"; BAND=100; fi
  for dt in float64 float32; do for m in l1 squared_euclidean; do for b in -1 $BAND; do
    R=${ds}_${dt}_${m}_b${b}
    "$CL" -i "$IN" $EXTRA -k 3 -m pam --dtype $dt --metric $m --band $b --name $R -o "$OUT/$R" > "$OUT/$R.raw" 2>&1
    echo "exit $?" >> "$OUT/$R.raw"
    grep -v -E "min:sec" "$OUT/$R.raw" > "$OUT/$R.stdout"; rm "$OUT/$R.raw"
  done; done; done
done
echo "runs: $(ls -d "$OUT"/*/ | wc -l), files: $(find "$OUT" -mindepth 2 -type f | wc -l)"
