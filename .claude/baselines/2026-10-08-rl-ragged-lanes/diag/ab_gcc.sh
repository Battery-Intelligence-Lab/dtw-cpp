#!/bin/sh
# diag/ab_gcc.sh <runs> "<fill args>" <name=path> ..., in WSL: diagnostic A/B (not a record): each run of each build
# under taskset -c 2 with OMP_NUM_THREADS=1, the builds rotating; prints each run's timed-fill ms and clock.
R=$1; S=$2; shift 2
i=0
while [ $i -lt "$R" ]; do
  set -- "$@"
  k=0; n=$#
  for b in "$@"; do
    name=${b%%=*}; exe=${b#*=}
    out=$(OMP_NUM_THREADS=1 taskset -c 2 "$exe" fill $S 1 1 2)
    echo "run $i $name $(echo "$out" | sed -E 's/.*ms ([0-9.]+) .*clock ([0-9.]+) GHz.*/ms \1 clock \2/')"
  done
  first=$1; shift; set -- "$@" "$first"
  i=$((i + 1))
done
