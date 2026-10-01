#!/usr/bin/env bash
# run_ident.sh <exe>: the identity matrix; prints config, table hash, call hash (times stripped).
EXE=$1
# N k L f32 metric odd_eighths band batch
while read -r cfg; do
  echo "== $cfg"
  OMP_NUM_THREADS=${THREADS:-8} ./$EXE.exe $cfg 1 2>&1 | sed -E 's/fill_s=[0-9.]+ //; s/call_s=[0-9.]+ //'
done <<CFGS
400 5 60 0 0 0 -1 -1
400 5 60 0 1 0 -1 -1
400 5 60 1 0 0 -1 -1
400 5 60 1 1 0 -1 -1
400 5 60 0 0 1 -1 -1
400 5 60 0 1 2 -1 -1
400 5 60 1 0 1 -1 -1
400 5 60 1 1 4 -1 -1
400 5 60 0 0 0 6 -1
400 5 60 1 1 0 6 -1
400 5 60 0 0 1 6 -1
400 5 60 0 0 8 -1 -1
400 5 60 0 0 0 -1 13
400 5 60 1 0 0 -1 100
400 5 60 1 0 0 -1 17
37 3 40 0 0 0 -1 -1
37 3 40 1 0 0 -1 -1
70 4 50 0 1 1 -1 -1
70 4 50 1 1 1 -1 9
65 2 30 0 0 0 0 -1
300 5 1 0 0 0 -1 -1
300 5 2 1 0 0 -1 -1
2000 10 200 0 0 0 -1 -1
2000 10 200 1 0 0 -1 -1
CFGS
