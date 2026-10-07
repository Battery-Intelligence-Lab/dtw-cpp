#!/bin/sh
# pair-2col timing kit: base (e26d5680) against k1 (463d2b52: per-pair kernel 1 two columns per pass) and head
# (00fb9c36: kernels 1 and 2), each linked in three code placements (build.sh: p0 default, p1 every loop on a
# 64-byte boundary, p2 everything 4 bytes later than p0), interleaved, on this Mac.
#
#   ./run.sh [repeats]   default 5; run it on a quiet machine (no builds or other agents, on power)
#   ./run.sh dry         one repeat at tiny sizes: shows the kit works, measures nothing
#
# Single thread, one pair at a time through the fill's per-pair function (std::function over
# dtwBanded + normalize_public_distance): f64/f32 x L1/Sq x {L 100, L 1000} x {unbanded, band L/10}, L 100 at
# band 2 (a narrow band, where kernel 2's per-pass rows weigh most), and the ragged pair 90 x 110 unbanded and at
# band 20 (the narrowest band that fits it); 40 M cells per measurement, 5 per invocation; ns/cell and
# cycles/cell (at 4.59 GHz and at the add-chain clock, measured around each measurement).
# Fills at 18 threads through Problem's fill loop and schedule, f64 L1, Gcell/s and a matrix hash: ragged
# N 1000 L 90-110 unbanded and band 10 (the per-pair kernels), equal N 2000 L 100 unbanded and N 500 L 1000
# band 100 (the lanes, which the unit leaves alone: a placement control).
# Each repeat rotates which build runs first and the order of the placements. A probe that fails leaves a
# FAILED line, and summary.py then gives no verdict. summary.py prints the medians, the speed-ups against base
# per placement, the checksums and matrix hashes, and the bands registered before the run.
set -e
cd "$(dirname "$0")"
R=${1:-5}
CELLS=40e6; REPS=5; FILLR=3; NR=1000; NE=2000; NB=500
if [ "$1" = dry ]; then R=1; CELLS=1e6; REPS=2; FILLR=1; NR=60; NE=40; NB=20; fi
for v in base k1 head; do for p in p0 p1 p2; do [ -x "pprobe_${v}_$p" ] || { echo "missing pprobe_${v}_$p: ./build_all.sh"; exit 1; }; done; done
probe() { # repeat version placement probe-arguments...
  r=$1; v=$2; p=$3; shift 3
  if out=$("./pprobe_${v}_$p" "$@" 2>&1); then echo "r$r $v $p $out"; else echo "r$r $v $p FAILED $*: $out"; fi
}
OUT=results_$(date +%Y%m%d_%H%M%S).txt
{
  echo "# pair-2col kit, $(date), $(sysctl -n machdep.cpu.brand_string), repeats $R, mode ${1:-measure}"
  for r in $(seq 1 "$R"); do
    case $((r % 3)) in
      1) ORDER="base k1 head"; PLACES="p0 p1 p2" ;;
      2) ORDER="k1 head base"; PLACES="p1 p2 p0" ;;
      0) ORDER="head base k1"; PLACES="p2 p0 p1" ;;
    esac
    for p in $PLACES; do
      for T in f64 f32; do for D in L1 Sq; do
        for S in "100 100 -1" "100 100 10" "100 100 2" "1000 1000 -1" "1000 1000 100" "90 110 -1" "90 110 20"; do
          for v in $ORDER; do probe "$r" "$v" "$p" time $T $D $S $REPS $CELLS; done
        done
      done; done
      for F in "f64 L1 $NR 100 -1 18 $FILLR 1" "f64 L1 $NR 100 10 18 $FILLR 1" "f64 L1 $NE 100 -1 18 $FILLR 0" "f64 L1 $NB 1000 100 18 $FILLR 0"; do
        for v in $ORDER; do probe "$r" "$v" "$p" fill $F; done
      done
    done
  done
} | tee "$OUT"
uv run --no-project python summary.py "$OUT"
