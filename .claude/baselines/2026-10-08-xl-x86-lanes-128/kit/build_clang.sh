#!/bin/sh
# build_clang.sh base|cand: bin/clang_<v>/xprobe.exe from xprobe.cpp and that version's
# src/<v>/dtwc/core/{dtw_lanes,dtw_dispatch}.cpp, with the compile commands of the XL worktree's build/
# (clang++ 21.1.8, GNU-style driver, MSVC ABI, as the main checkout's build/ configures it): dtw_lanes.cpp with
# -fno-lto, as dtwc/core/CMakeLists.txt sets for clang on the MSVC ABI, the other two ThinLTO; linked as
# dtwc_cl.exe is (lld-link, ThinLTO, LLVM's libomp). The project's -D/-I flags are dropped: no header these files
# include tests a project macro. Also writes asm/clang_<v>_dtw_lanes.s (-S -masm=intel, the same flags).
set -e
cd "$(dirname "$0")"
V=$1
CXX="/c/Program Files/LLVM/bin/clang++.exe"
CFLAGS="-O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -march=native -fno-finite-math-only -Werror=switch -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
LFLAGS="-O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -Xlinker /subsystem:console -fuse-ld=lld-link"
INC="-Isrc/$V/dtwc"
O=bin/clang_$V
mkdir -p $O asm
"$CXX" $INC $CFLAGS -fno-lto -c src/$V/dtwc/core/dtw_lanes.cpp -o $O/dtw_lanes.obj
"$CXX" $INC $CFLAGS -fno-lto -S -masm=intel src/$V/dtwc/core/dtw_lanes.cpp -o asm/clang_${V}_dtw_lanes.s
"$CXX" $INC $CFLAGS -c src/$V/dtwc/core/dtw_dispatch.cpp -o $O/dtw_dispatch.obj
"$CXX" $INC $CFLAGS -c xprobe.cpp -o $O/xprobe.obj
"$CXX" $LFLAGS $O/dtw_lanes.obj $O/dtw_dispatch.obj $O/xprobe.obj "C:/Program Files/LLVM/lib/libomp.lib" -o $O/xprobe.exe
echo "built $O/xprobe.exe"
