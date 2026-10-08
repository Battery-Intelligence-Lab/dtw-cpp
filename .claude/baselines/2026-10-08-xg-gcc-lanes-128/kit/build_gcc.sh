#!/bin/sh
# build_gcc.sh base|cand, run in WSL (wsl -e sh <kit>/build_gcc.sh base): ~/xg/bin/gcc_<v>/xprobe from xprobe.cpp and
# that version's src/<v>/dtwc/core/{dtw_lanes,dtw_dispatch}.cpp, with GCC 13.3 and the flags CMake gives these files
# in WSL (b3b3a57e configured Release with DTWC_ARCH_LEVEL=v3, as release-artifacts.yml; a wheel's
# DTWC_BUILD_PYTHON=ON picks v3 too): dtwc_core's flags for the two library files, dtwc_cl.cpp's for the probe, linked
# with LTO as dtwc_cl is. The project's -D/-I flags are dropped: no header these files include tests a project macro.
# Also writes asm/gcc_<v>_dtw_lanes.s (-S -masm=intel, the same flags without LTO, since an LTO object holds no
# code) and asm/gcc_<v>_xprobe.dis (objdump of the LTO-linked probe: the code that runs).
set -e
cd "$(dirname "$0")"
V=$1
O=$HOME/xg/bin/gcc_$V
FP="-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -fno-signaling-nans"
LTO="-flto=auto -fno-fat-lto-objects"
CORE="-O3 -DNDEBUG -std=c++20 -fPIC -march=x86-64-v3 -fno-finite-math-only -Werror=switch -fopenmp $FP"
CLI="-O3 -DNDEBUG -std=c++20 -march=x86-64-v3 -fopenmp $FP"
INC="-Isrc/$V/dtwc"
mkdir -p "$O" asm
g++ $INC $CORE $LTO -c src/$V/dtwc/core/dtw_lanes.cpp -o "$O/dtw_lanes.o"
g++ $INC $CORE $LTO -c src/$V/dtwc/core/dtw_dispatch.cpp -o "$O/dtw_dispatch.o"
g++ $INC $CLI $LTO -c xprobe.cpp -o "$O/xprobe.o"
g++ -O3 -DNDEBUG $LTO "$O/dtw_lanes.o" "$O/dtw_dispatch.o" "$O/xprobe.o" -o "$O/xprobe" \
  /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.a -ldl
g++ $INC $CORE -S -masm=intel src/$V/dtwc/core/dtw_lanes.cpp -o asm/gcc_${V}_dtw_lanes.s
objdump -d -M intel --no-show-raw-insn "$O/xprobe" > asm/gcc_${V}_xprobe.dis
echo "built $O/xprobe ($(g++ --version | head -1))"
