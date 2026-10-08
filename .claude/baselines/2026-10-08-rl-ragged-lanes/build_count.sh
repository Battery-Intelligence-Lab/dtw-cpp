#!/bin/sh
# build_count.sh <v>: bin/clang_<v>_count/rprobe.exe, the clang counterpart of build_count.bat (kit only): the shipped
# src/<v>/dtwc/core/dtw_lanes.cpp with its clang-tree flags (-fno-lto, as CMake compiles it) plus -include
# count/count_lanes.hpp; rprobe.cpp with -DRL_COUNT; linked as build_clang.sh links, the counting object first.
set -e
cd "$(dirname "$0")"
V=$1
T=C:/D/git/wt/RL/build
CXX="/c/Program Files/LLVM/bin/clang++.exe"
O=bin/clang_${V}_count
FP="-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
mkdir -p $O
"$CXX" -DDTWC_ENABLE_GUROBI -DDTWC_ENABLE_HIGHS -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -I. \
  -Isrc/$V/dtwc/. -Isrc/$V/dtwc/extern -Isrc/$V/dtwc/mip/. -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang \
  --dependent-lib=msvcrt -march=native -fno-finite-math-only -Werror=switch -fopenmp $FP -fno-lto \
  -include count/count_lanes.hpp -c src/$V/dtwc/core/dtw_lanes.cpp -o $O/dtw_lanes.obj
"$CXX" -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -DRL_COUNT -I. -Isrc/$V/dtwc/. -O3 -DNDEBUG \
  -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -march=native -fopenmp $FP -c rprobe.cpp \
  -o $O/rprobe.obj
"$CXX" -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -Xlinker /subsystem:console -fuse-ld=lld-link \
  $O/rprobe.obj $O/dtw_lanes.obj lib/clang_$V/dtwc_core.lib $T/GurobiCXX.lib C:/gurobi1301/win64/lib/gurobi130.lib \
  $T/bin/highs.lib "C:/Program Files/LLVM/lib/libomp.lib" -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 \
  -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames -o $O/rprobe.exe
echo "built $O/rprobe.exe"
