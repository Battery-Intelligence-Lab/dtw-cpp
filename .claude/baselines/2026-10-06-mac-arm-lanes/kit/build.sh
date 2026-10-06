#!/bin/sh
# Builds lprobe_<tag> from src/lprobe.cpp and src/<tag>/core/dtw_lanes.cpp (that version's sources:
# src/base = 254ecd3b, src/head = pb/arm-lanes), with the compile command of dtwc/core/dtw_lanes.cpp
# in build/compile_commands.json (AppleClang, -march=native, the library's FP flags, ThinLTO).
# usage: ./build.sh base|head
set -e
cd "$(dirname "$0")"
T=$1
R=src/$T
FLAGS="-O3 -DNDEBUG -std=c++20 -flto=thin -arch arm64 -fPIC -march=native -fno-finite-math-only -Xclang -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
INC="-DDTWC_HAS_OPENMP -I$R -isystem /opt/homebrew/opt/libomp/include"
mkdir -p obj
/usr/bin/clang++ $INC $FLAGS -c "$R/core/dtw_lanes.cpp" -o "obj/dtw_lanes_$T.o"
/usr/bin/clang++ $INC $FLAGS -c src/lprobe.cpp -o "obj/lprobe_$T.o"
/usr/bin/clang++ -O3 -DNDEBUG -flto=thin -arch arm64 "obj/dtw_lanes_$T.o" "obj/lprobe_$T.o" \
  -L/opt/homebrew/opt/libomp/lib -lomp -Wl,-rpath,/opt/homebrew/opt/libomp/lib -o "lprobe_$T"
echo "built lprobe_$T"
