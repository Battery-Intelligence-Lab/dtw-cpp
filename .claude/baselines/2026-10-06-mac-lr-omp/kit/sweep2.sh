#!/bin/bash
# Phase 2, sweep 2: the final tree (kParallelMinN = 280, loop 2 serial), the object's own threshold (HEAD_MIN_N unset), end to end included.
# The grid is the registered one plus the boundary cells of the threshold (240 serial, 280 forks, 320).
cd "$(dirname "$0")" || exit 1
RUN_TAG=sweep2 SIZES="100 200 240 280 320 400 800 1600 3200" ./run.sh > runs/sweep2.log 2>&1
echo "exit=$?" >> runs/sweep2.log
