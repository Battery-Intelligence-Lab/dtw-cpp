#!/usr/bin/env bash
# Usage: band.sh <base.exe> <head.exe> <out.txt> <mode:L:N>...
# Per case: base then head, back to back, each one warm-up fill + 5 timed fills
# pinned to the 8 P-cores (pinned.bat). Before a case it waits (up to 15 min)
# until no other CUDA process from C:\D\git (another tree's tests or benches)
# is on the GPU, and logs the ones it sees before and after the case; the CPU
# load is logged after the case. A case with a foreign process after it is
# marked CONTENDED. pinned.bat: 2026-09-30-c1-cuda-long-series/pinned.bat with this run's TMP/TEMP.
# long_fill: that folder's build.bat, compiling c3_long_fill.cpp, against base's and head's build-cuda.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
base="$1"; head="$2"; out="$3"; shift 3
pinned="$(cygpath -w "$here/pinned.bat")"
foreign() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -i 'C:\\D\\git' | tr '\n' ';'; }
echo "start $(date +%T)" >> "$out"
for c in "$@"; do
  IFS=: read -r mode L N <<< "$c"
  waited=0
  while [ -n "$(foreign)" ] && [ $waited -lt 900 ]; do sleep 10; waited=$((waited + 10)); done
  echo "gpu before: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader); waited ${waited}s; foreign: [$(foreign)]" >> "$out"
  for which in base head; do
    if [ "$which" = base ]; then exe="$base"; else exe="$head"; fi
    echo "== $which $mode $L $N" >> "$out"
    cmd //c "$pinned $(cygpath -w "$exe") $mode $N $L 5" >> "$out" 2>&1
  done
  after="$(foreign)"
  echo "foreign after: [$after]${after:+ CONTENDED}" >> "$out"
  echo "cpu load $(powershell -NoProfile -Command '(Get-CimInstance Win32_Processor).LoadPercentage' | tr -d '\r')" >> "$out"
done
echo "end $(date +%T)" >> "$out"
