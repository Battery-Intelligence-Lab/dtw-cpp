#!/bin/bash
# Phase 2, sweep 1: every N forks (HEAD_MIN_N=1), the crossover. E2E off: the head dtwc_cl has the placeholder 1 anyway, and sweep 2 reads that column.
cd "$(dirname "$0")" || exit 1
RUN_TAG=sweep1 HEAD_MIN_N=1 E2E=0 ./run.sh > runs/sweep1.log 2>&1
echo "exit=$?" >> runs/sweep1.log
