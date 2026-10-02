#!/usr/bin/env bash
# GC band: the registered cases back to back; the load before and after each.
S=C:/Users/engs2321/AppData/Local/Temp/claude/c--D-git-dtw-cpp/79bd8991-b0e9-441b-bc55-95584dcfaf55/scratchpad/gc
EXE=$(cygpath -w $S/bench/clara_assign_bench.exe)
load() {
  local cpu gpu
  cpu=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader)
  echo "load,$1,cpu ${cpu//$'\r'/} %,gpu ${gpu}"
}
case_() {
  load "before $*"
  cmd //c "$(cygpath -w $S/cuda_run.bat) $EXE $*" | grep -v "^RUN_EXIT" | tr -d '\r'
  load "after $*"
}
date "+start %Y-%m-%d %H:%M:%S %Z"
for c in "fill32 1100 500 10 5" "rect32 20000 500 10 5" "fill64 520 500 10 5" "rect64 20000 500 10 5" \
         "cpu 20000 500 10 5" \
         "fill32 600 1024 10 5" "rect32 10000 1024 10 5" "fill32 184 3000 10 5" "rect32 1000 3000 10 5"; do
  case_ $c
done
date "+end %Y-%m-%d %H:%M:%S %Z"
