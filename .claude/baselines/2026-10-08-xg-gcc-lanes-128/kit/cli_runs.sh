#!/bin/sh
# cli_runs.sh <base|cand>, in WSL: the lanes routes through the real dtwc_cl of ~/xg/repo/build-base or
# ~/xg/repo-cand/build-cand, 64 runs: data/dummy cut to equal lengths (every series's first 1000 samples, and its first
# 5148, the shortest series's length, so the fill's blocks are equal-length: the lanes route) x {float64, float32} x
# {l1, squared_euclidean} x {unbanded, band 100} x k {3, 5} x {pam, kmedoids}, the default seed. OpenMP 8 threads (the
# machine is shared). Each run's files go to out-<v>/<run>/, its stdout and exit code to
# out-<v>/<run>.stdout. The cut data are written once, beside this script, from ~/xg/repo/data/dummy (b3b3a57e).
set -u
V=$1
cd "$(dirname "$0")"
B=$HOME/xg
CL=$B/repo/build-base/bin/dtwc_cl
[ "$V" = cand ] && CL=$B/repo-cand/build-cand/bin/dtwc_cl
for L in 1000 5148; do
  if [ ! -d data/dummy$L ]; then
    mkdir -p data/dummy$L
    for f in "$B"/repo/data/dummy/*.csv; do head -n $((L + 1)) "$f" > "data/dummy$L/$(basename "$f")"; done
  fi
done
OUT=out-$V
rm -rf "$OUT" && mkdir -p "$OUT"
for L in 1000 5148; do for dt in float64 float32; do for m in l1 squared_euclidean; do for b in -1 100; do
  for k in 3 5; do for meth in pam kmedoids; do
    R=dummy${L}_${dt}_${m}_b${b}_k${k}_${meth}
    OMP_NUM_THREADS=8 "$CL" -i data/dummy$L --skip-rows 1 --skip-cols 1 -k $k -m $meth --dtype $dt --metric $m \
      --band $b --name $R -o "$OUT/$R" > "$OUT/$R.stdout" 2>&1
    echo "exit $?" >> "$OUT/$R.stdout"
  done; done
done; done; done; done
echo "$V: $(ls -d "$OUT"/*/ | wc -l) runs, $(find "$OUT" -mindepth 2 -type f | wc -l) files, $(grep -L "^exit 0$" "$OUT"/*.stdout | wc -l) runs without exit 0"
