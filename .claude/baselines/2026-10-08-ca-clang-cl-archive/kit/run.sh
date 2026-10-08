#!/bin/sh
# CA, one command, on a quiet machine only: sh /c/D/git/wt/tmp/CA/kit/run.sh
# A load record, 3 interleaved rounds of the release archive's dtwc_cl.exe (clang-cl) and dtwc_cl_msvc.exe (MSVC)
# on the CLI's distance-matrix fill (archive/ holds the unpacked dtwc-2.0.0rc1-Windows-AMD64.zip; data/ the
# fixed-seed CSVs), a load record, then the report against the registered band (tables.md). A previous raw.txt is
# moved aside first, so the report reads this run only. About 5 minutes (estimated from CW's smoke rates: the
# pinned MSVC fills of (a) and (b) take about 16 s and 22 s each).
set -e
cd /c/D/git/wt/tmp/CA/kit
if [ -f raw.txt ]; then mv raw.txt "raw.$(date '+%Y%m%d-%H%M%S').txt"; fi
sh load.sh "before the first round"
uv run --no-project python driver.py run
sh load.sh "after the last round"
uv run --no-project python driver.py report
