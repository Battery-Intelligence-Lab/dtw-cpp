#!/usr/bin/env bash
# W4a: every case, FP32 then FP64; one driver process per case (oracle computed once per case).
# Usage: run_all.sh <dir holding ab.exe>; writes <dir>/results/<prec>_N<N>_L<L>.txt
set -u
cd "$1"
export OMP_NUM_THREADS=8   # host oracle threads
mkdir -p results
cases_f32="1000:16:11 1000:32:11 200:128:11 200:256:11 200:257:11 200:384:11 200:500:11 200:1024:11 200:1100:11 200:1500:11 200:2000:11 200:2048:11 200:2049:11"
# FP64 N reduced where one launch would exceed ~1.5 s (band rule 5); persistent mode stays on.
cases_f64="1000:16:11 1000:32:11 200:128:11 200:256:11 200:257:11 200:384:11 200:500:11 200:1024:7 200:1100:7 140:1500:7 100:2000:7 100:2048:7 100:2049:7"
for prec in f32 f64; do
  eval cases=\$cases_$prec
  for c in $cases; do
    IFS=: read -r N L R <<< "$c"
    ./ab.exe $prec $N $L $R > results/${prec}_N${N}_L${L}.txt 2>&1
    echo "$(date +%H:%M:%S) exit $? $prec N=$N L=$L"
  done
done
