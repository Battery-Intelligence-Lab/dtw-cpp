#!/usr/bin/env bash
# run_band.sh <rounds-per-process> <pairs> <threads> <cfg...>: alternate base and head processes; log lines tagged.
R=$1; P=$2; T=$3; shift 3
for p in $(seq 1 $P); do
  for who in base head; do
    OMP_NUM_THREADS=$T ./bench_$who.exe "$@" $R 2>&1 | sed "s/^/$who /"
  done
done
