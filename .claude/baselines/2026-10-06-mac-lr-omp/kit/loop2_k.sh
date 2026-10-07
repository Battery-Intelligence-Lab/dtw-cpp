#!/bin/bash
# Phase 2: does forking loop 2 pay at a larger k? Noise regime, N = 1600 and 3200, k = 20 and 64 (the root runs its 4000 iterations there).
cd "$(dirname "$0")" || exit 1
for k in 20 64; do
  K_NOISE=$k REGIMES=noise SIZES="1600 3200" uv run --no-project python loop2_ab.py >> runs/loop2_k.log 2>&1
done
echo "exit=$?" >> runs/loop2_k.log
