#!/usr/bin/env bash
# build.sh <tree_root> <out.exe>: the harness plus that tree's dtw_dispatch.cpp, library flags, ThinLTO.
set -eu
T="$1"; OUT="$2"
FLAGS="-DDTWC_HAS_OPENMP -I$T/dtwc/. -I$T/dtwc/extern -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -march=native -fno-finite-math-only -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
clang++ $FLAGS -fuse-ld=lld -o "$OUT" C:/D/git/wt/tmp/W7d/harness/harness.cpp "$T/dtwc/core/dtw_dispatch.cpp"
