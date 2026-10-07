#!/bin/bash
# Phase 2, sweep 2 again for the HiGHS-ON cells only, base and head loading one libhighs (the first sweep 2 compared two separately built dylibs: see diag_lib.py).
cd "$(dirname "$0")" || exit 1
RUN_TAG=sweep2on FLAVORS=on SIZES="100 200 240 280 320 400 800 1600 3200" ./run.sh > runs/sweep2on.log 2>&1
echo "exit=$?" >> runs/sweep2on.log
