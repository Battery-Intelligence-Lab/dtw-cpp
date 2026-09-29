#!/usr/bin/env bash
# P1 band runs: interleaved base/tip fills (Google Benchmark), the ECG5000 fill (drivers),
# then the pinned single-thread kernels (p1_kernel_ab.exe beside kernel_pinned.bat).
# Raw outputs in $OUT (scratch, not kept); analyse_band.py $OUT prints the tables.
set -u
BIN=C:/D/git/wt/P1/build/bin
OUT=C:/D/git/wt/tmp/P1/bench2
UCR=C:/D/git/dtw-cpp/data/benchmark/UCRArchive_2018/ECG5000/ECG5000_TEST.tsv
mkdir -p "$OUT"
cd "$BIN" || exit 1
typeperf "\Processor(_Total)\% Processor Time" -si 5 -sc 720 -y -o "$OUT/load.csv" > /dev/null 2>&1 &
TP=$!
for r in 0 1 2 3 4 5 6 7 8; do
  if (( r % 2 == 0 )); then order="base tip"; else order="tip base"; fi
  for b in $order; do
    if [ "$b" = base ]; then exe=bench_base_f705329.exe; else exe=bench_dtw_baseline.exe; fi
    ./$exe '--benchmark_filter=^BM_fillDistanceMatrix/(100/1000/-1|50/1000/50)$' \
      --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
      --benchmark_out="$OUT/fill_${b}_r${r}.json" --benchmark_out_format=json > /dev/null 2>&1
    echo "fill r$r $b exit=$? $(date +%T)"
  done
done
for r in 0 1 2 3 4; do
  if (( r % 2 == 0 )); then order="perpair lanes"; else order="lanes perpair"; fi
  for b in $order; do
    if [ "$b" = perpair ]; then exe=ucr_fill_perpair_e34c37f.exe; else exe=ucr_fill_lanes_3c95dc7.exe; fi
    ./$exe "$UCR" -1 2 > "$OUT/ucr_${b}_r${r}.txt" 2>&1
    echo "ucr r$r $b exit=$? $(date +%T)"
  done
done
for cfg in "1000 -1" "1000 100" "140 -1"; do
  set -- $cfg
  cmd //c "C:\D\git\wt\tmp\P1\kernel_pinned.bat $1 $2" > "$OUT/kernel_L$1_b$2.txt" 2>&1
  echo "kernel L=$1 band=$2 exit=$? $(date +%T)"
done
kill $TP 2>/dev/null
echo "BAND RUNS DONE $(date +%T)"
