#!/bin/sh
# arm-lanes timing kit: base (254ecd3b, design-2.0) against head (pb/arm-lanes), interleaved, on this Mac.
#
#   ./run.sh [repeats]   default 5; run it on a quiet machine (no builds or other agents, on power)
#   ./run.sh dry         one repeat at tiny sizes: shows the kit works, measures nothing
#
# lprobe_base / lprobe_head are src/lprobe.cpp linked with each version's own dtwc/core/dtw_lanes.cpp
# (build.sh: dtw_lanes.cpp's compile command, ThinLTO); their four lanes loops equal the linked
# dtwc_cl's instruction for instruction. dtwc_cl_base / dtwc_cl_head are the two CLIs (not run here).
# Each repeat alternates which build goes first. Single thread: the lane function alone, ns/cell and
# cycles/cell at 4.59 GHz (and at the add-chain clock); fills: Problem's brute-force fill loop at 18
# threads, Gcell/s, with a hash of the matrix (base and head must match). summary.py then prints the
# medians over repeats, head/base speed-ups and the bands registered before the run.
set -e
cd "$(dirname "$0")"
R=${1:-5}
CELLS=120e6; FILLR=3; N1=2000; N2=500; N3=1000
if [ "$1" = dry ]; then R=1; CELLS=2e6; FILLR=1; N1=100; N2=40; N3=60; fi
OUT=results_$(date +%Y%m%d_%H%M%S).txt
{
  echo "# arm-lanes kit, $(date), $(sysctl -n machdep.cpu.brand_string), repeats $R, mode ${1:-measure}"
  for r in $(seq 1 "$R"); do
    if [ $((r % 2)) -eq 1 ]; then ORDER="base head"; else ORDER="head base"; fi
    for T in f64 f32; do for D in L1 Sq; do for L in 100 1000; do for B in -1 $((L / 10)); do
      for v in $ORDER; do echo "r$r $v $(./lprobe_$v time $T $D $L $B 5 $CELLS)"; done
    done; done; done; done
    for F in "f64 L1 $N1 100 -1 18 $FILLR 0" "f64 L1 $N2 1000 100 18 $FILLR 0" "f64 L1 $N3 100 -1 18 $FILLR 1"; do
      for v in $ORDER; do echo "r$r $v $(./lprobe_$v fill $F)"; done
    done
  done
} | tee "$OUT"
uv run --no-project python summary.py "$OUT"
