#!/usr/bin/env bash
# W7ef (b): the Soft-DTW and Interpolate fills, base against head, pinned to the
# eight P-cores (logical CPUs 0,1,10,11,12,13,22,23 = mask C03C03) with
# OMP_NUM_THREADS=8. Fill time = the -v clock at "FastPAM converged" minus the
# clock at "Data loaded" (W7d's method). Five repetitions per binary and case;
# the order alternates (base first on odd repetitions, head first on even ones).
# Usage: fill_time.sh <base.exe> <head.exe> <data_dir> <out.csv>
set -u
BASE="$1"; HEAD="$2"; DATA="$3"; OUT="$4"
export OMP_NUM_THREADS=8
TMPD=C:/D/git/wt/tmp/W7ef/fill
mkdir -p "$TMPD"
win() { cygpath -w "$1"; }
seconds() { # "[M:S.sss min:sec]" -> seconds
  sed -E 's/.*\[([0-9]+):([0-9.]+) min:sec\].*/\1 \2/' | awk '{printf "%.6f", $1 * 60 + $2}'
}
one() { # binary case -> fill milliseconds
  local exe="$1" which="$2" log="$TMPD/run.txt" args
  case "$which" in
    softdtw) args="-i $(win "$DATA/softdtw_32x1000") --skip-rows 1 --skip-cols 1 --variant softdtw" ;;
    interp) args="-i $(win "$DATA/interp_1000x64.csv") --missing-strategy interpolate" ;;
    interp_plain) args="-i $(win "$DATA/plain_1000x64.csv") --missing-strategy interpolate" ;;
    zero) args="-i $(win "$DATA/interp_1000x64.csv") --missing-strategy zero_cost" ;;
    standard) args="-i $(win "$DATA/plain_1000x64.csv")" ;;
    adtw) args="-i $(win "$DATA/plain_1000x64.csv") --variant adtw" ;;
    wdtw64) args="-i $(win "$DATA/plain_1000x64.csv") --variant wdtw" ;;
    wdtw32) args="-i $(win "$DATA/plain_1000x64.csv") --variant wdtw --dtype float32" ;;
  esac
  cmd //c "start /b /wait /affinity C03C03 $(win "$exe") $args -k 3 -m pam -v -o $(win "$TMPD/out")" > "$log" 2>&1
  local t1 t2
  t1=$(grep 'Data loaded:' "$log" | seconds)
  t2=$(grep 'FastPAM converged' "$log" | seconds)
  awk -v a="$t1" -v b="$t2" 'BEGIN { printf "%.1f", (b - a) * 1000 }'
}
echo "case,rep,binary,fill_ms" > "$OUT"
for which in ${CASES:-softdtw interp wdtw64 wdtw32}; do
  for rep in 1 2 3 4 5; do
    if [ $((rep % 2)) -eq 1 ]; then order="base head"; else order="head base"; fi
    for b in $order; do
      if [ "$b" = base ]; then exe="$BASE"; else exe="$HEAD"; fi
      echo "$which,$rep,$b,$(one "$exe" "$which")" >> "$OUT"
    done
  done
done
cat "$OUT"
