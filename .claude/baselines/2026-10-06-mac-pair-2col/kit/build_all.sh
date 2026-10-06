#!/bin/sh
# Builds the nine probes (base, k1, head x p0, p1, p2) and shows where their per-pair loops landed.
set -e
cd "$(dirname "$0")"
for v in base k1 head; do ./build.sh "$v"; done
uv run --no-project python placement.py pprobe_*_p? > placement.txt
echo "placement report: placement.txt"
