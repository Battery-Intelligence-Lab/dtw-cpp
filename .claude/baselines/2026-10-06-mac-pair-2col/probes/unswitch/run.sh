#!/bin/sh
# Compiles and links k.cpp against a dtwc/ tree at -O3, -O2, -Os (the library's FP flags, no LTO) and maps
# the DP loops (>= 10 instructions) to their functions.
# usage: run.sh <dtwc dir> <tag>
cd "$(dirname "$0")" || exit 1
FP="-fno-finite-math-only -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
for O in O3 O2 Os; do
  /usr/bin/clang++ -$O -DNDEBUG -std=c++20 -arch arm64 -march=native $FP -I"$1" k.cpp -o "k_$2_$O" || exit 1
  objdump -d --no-show-raw-insn "k_$2_$O" > "k_$2_$O.dis"
  echo "== $2 -$O"
  uv run --no-project python funcs.py "k_$2_$O.dis"
done
