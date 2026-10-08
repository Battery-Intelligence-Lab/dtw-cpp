#!/bin/sh
# Orchestrator's second attempt (2026-10-08) on the one shape of the B3 run outside its band: cl, f32 L1, N 1000,
# lengths 90-110, unbanded, 24 threads (median per-round ratio 0.940; rounds 1.078, 0.940, 0.939). Under cl B3's fill
# sends these blocks to the one-pair route as base does (same lane calls, same matrix), so the question is whether
# the 0.940 is a real slowdown or the 24-thread noise of this box.
# REGISTERED BEFORE THE RUN: 20 pairs, each two probe processes (base and B3, the first alternating), each probe one
# untimed fill and 3 timed; ratio = base ms / B3 ms per pair. Median ratio >= 0.97: noise, the shape passes on this
# attempt; < 0.97: a real cl slowdown, FALSIFIED stands and the cause is looked for.
cd "$(dirname "$0")/.."
out=orch/ab_cl_f32.txt
echo "# ab_cl_f32 $(date '+%Y-%m-%d %H:%M:%S') 20 pairs f32 L1 1000 90 110 -1 0 3" >> $out
for k in $(seq 0 19); do
  if [ $((k % 2)) -eq 0 ]; then order="base B3"; else order="B3 base"; fi
  for v in $order; do
    echo "pair=$k build=msvc_$v $(./bin/msvc_$v/rprobe.exe fill f32 L1 1000 90 110 -1 0 3 12 2>&1 | grep '^fill')" >> $out
  done
done
echo "# done $(date '+%H:%M:%S')" >> $out
