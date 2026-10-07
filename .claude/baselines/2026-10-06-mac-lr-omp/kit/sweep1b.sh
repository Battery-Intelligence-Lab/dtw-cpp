#!/bin/bash
# Phase 2, sweep 1b: the loop-2-serial design (HEAD_LOOP2=0), every N forks loop 1 (HEAD_MIN_N=1), HiGHS-OFF, a finer grid around the crossover.
cd "$(dirname "$0")" || exit 1
RUN_TAG=sweep1b HEAD_LOOP2=0 HEAD_MIN_N=1 FLAVORS=off SIZES="200 240 280 320 360 400" E2E=0 ./run.sh > runs/sweep1b.log 2>&1
echo "exit=$?" >> runs/sweep1b.log
