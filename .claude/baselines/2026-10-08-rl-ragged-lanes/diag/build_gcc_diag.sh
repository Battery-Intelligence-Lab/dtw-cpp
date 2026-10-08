#!/bin/sh
# diag/build_gcc_diag.sh <name> <v> <lanes dir>, in WSL: ~/rl/diag/<name>/rprobe, a diagnostic GCC probe (kit only):
# rprobe.cpp against src/<v>/dtwc and ~/rl/lib/gcc_<v>/libdtwc_core.a, with <lanes dir>/dtwc/core/dtw_lanes.cpp
# compiled by dtwc_core's flags (LTO) and linked first, so the library's own dtw_lanes object is not pulled in.
set -e
KIT=$(cd "$(dirname "$0")/.." && pwd)
N=$1; V=$2; L=$KIT/$3/dtwc
O=$HOME/rl/diag/$N
FP="-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -fno-signaling-nans"
LTO="-flto=auto -fno-fat-lto-objects"
mkdir -p "$O"
g++ -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -I$L/. -I$L/extern -I$L/mip/. -O3 -DNDEBUG -std=c++20 -fPIC $LTO \
  -march=x86-64-v3 -fno-finite-math-only -Werror=switch -fopenmp $FP -c "$L/core/dtw_lanes.cpp" -o "$O/dtw_lanes.o"
g++ -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -I$KIT/src/$V/dtwc/. -O3 -DNDEBUG -std=c++20 \
  $LTO -march=x86-64-v3 -fopenmp $FP -c "$KIT/rprobe.cpp" -o "$O/rprobe.o"
g++ -O3 -DNDEBUG $LTO "$O/dtw_lanes.o" "$O/rprobe.o" "$HOME/rl/lib/gcc_$V/libdtwc_core.a" \
  /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.a -ldl -o "$O/rprobe"
echo "built $O/rprobe"
