#!/bin/sh
# Builds kbench with the compile command of dtwc/core/dtw_lanes.cpp (build/compile_commands.json),
# minus the -D/-I for HiGHS/llfio it does not need, plus this directory for the v_*.hpp copies.
# $1: native (build/, MEX) or default (the wheel: no -march), $2: extra flags
set -e
cd "$(dirname "$0")"
ARCH=-march=native
[ "$1" = default ] && ARCH=
OUT=kbench_${1:-native}${3:+_$3}
FLAGS="-O3 -DNDEBUG -std=c++20 -flto=thin -arch arm64 -fPIC $ARCH -fno-finite-math-only -Xclang -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
/usr/bin/clang++ -DDTWC_HAS_OPENMP -I/Users/engs2321/git/dtw-cpp/dtwc/. -I. -isystem /opt/homebrew/opt/libomp/include $FLAGS $2 -c kbench.cpp -o $OUT.o
/usr/bin/clang++ -O3 -DNDEBUG -flto=thin -arch arm64 $2 $OUT.o -L/opt/homebrew/opt/libomp/lib -lomp -Wl,-rpath,/opt/homebrew/opt/libomp/lib -o $OUT
echo built $OUT
