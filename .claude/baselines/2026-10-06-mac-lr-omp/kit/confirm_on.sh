#!/bin/bash
# Phase 2: the HiGHS-ON cells around the threshold again with 9 repeats (A/B), and the same cells base against itself (A/A, the harness's own bias).
cd "$(dirname "$0")" || exit 1
RUN_TAG=confirm-ab FLAVORS=on SIZES="100 200 240 280 400" REPEATS=9 E2E=0 ./run.sh > runs/confirm_ab.log 2>&1
echo "exit=$?" >> runs/confirm_ab.log
AA=1 RUN_TAG=confirm-aa FLAVORS=on SIZES="100 200 240 280 400" REPEATS=9 E2E=0 ./run.sh > runs/confirm_aa.log 2>&1
echo "exit=$?" >> runs/confirm_aa.log
