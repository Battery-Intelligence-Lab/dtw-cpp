#!/usr/bin/env bash
# W7d band: the variable-length banded WDTW fill, base against head, pinned to
# the eight P-cores (logical CPUs 0,1,10,11,12,13,22,23 = mask C03C03) with
# OMP_NUM_THREADS=8. Fill time = the -v clock at "FastPAM converged" minus the
# clock at "Data loaded". Five repetitions per binary and band; the order
# alternates (base first on odd repetitions, head first on even ones).
# Usage: w7d_band.sh <base.exe> <head.exe> <data_dir> <out.csv>
set -u
BASE="$1"; HEAD="$2"; DATA="$3"; OUT="$4"
export OMP_NUM_THREADS=8
TMPD=C:/D/git/wt/tmp/W7d/band
mkdir -p "$TMPD"
win() { cygpath -w "$1"; }
seconds() { # "[M:S.sss min:sec]" -> seconds
  sed -E 's/.*\[([0-9]+):([0-9.]+) min:sec\].*/\1 \2/' | awk '{printf "%.6f", $1 * 60 + $2}'
}
one() { # binary band -> fill milliseconds
  local exe="$1" band="$2" log="$TMPD/run.txt"
  cmd //c "start /b /wait /affinity C03C03 $(win "$exe") -i $(win "$DATA") --skip-rows 1 --skip-cols 1 -k 3 -m pam --variant wdtw --band $band -v -o $(win "$TMPD/out")" > "$log" 2>&1
  local t1 t2
  t1=$(grep 'Data loaded:' "$log" | seconds)
  t2=$(grep 'FastPAM converged' "$log" | seconds)
  awk -v a="$t1" -v b="$t2" 'BEGIN { printf "%.1f", (b - a) * 1000 }'
}
echo "band,rep,binary,fill_ms" > "$OUT"
for band in 50 200; do
  for rep in 1 2 3 4 5; do
    if [ $((rep % 2)) -eq 1 ]; then order="base head"; else order="head base"; fi
    for which in $order; do
      if [ "$which" = base ]; then exe="$BASE"; else exe="$HEAD"; fi
      echo "$band,$rep,$which,$(one "$exe" "$band")" >> "$OUT"
    done
  done
done
cat "$OUT"
