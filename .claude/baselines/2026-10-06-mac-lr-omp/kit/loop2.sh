#!/bin/bash
# Phase 2: loop 2 forked or serial (head against head), HiGHS-OFF root; see loop2_ab.py.
cd "$(dirname "$0")" || exit 1
uv run --no-project python loop2_ab.py > runs/loop2.log 2>&1
echo "exit=$?" >> runs/loop2.log
