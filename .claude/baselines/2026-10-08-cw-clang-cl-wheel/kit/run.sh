#!/bin/sh
# CW, one command (on a quiet machine only): a load record, 3 interleaved rounds of the MSVC wheel
# (C:/D/git/wt/venv/cw-msvc, design-2.0 as CI builds it today) and the clang-cl wheel (cw-clang, pb/CW) on WM's
# shapes (raw.txt), a load record, the report against the registered band (tables.md). A previous raw.txt is
# moved aside first, so the report reads this run only.
set -e
cd /c/D/git/wt/tmp/CW/kit
if [ -f raw.txt ]; then mv raw.txt "raw.$(date '+%Y%m%d-%H%M%S').txt"; fi
sh load.sh "before the first round"
uv run --no-project python driver.py run
sh load.sh "after the last round"
uv run --no-project python driver.py report
