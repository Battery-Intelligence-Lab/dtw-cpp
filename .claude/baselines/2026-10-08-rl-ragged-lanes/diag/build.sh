#!/bin/sh
# diag/build.sh <name> <v> <lanes v> <pad> [extra flags for the lanes object]: diag/bin/<name>/rprobe.exe, a diagnostic
# probe (kit only): rprobe.cpp against src/<v>/dtwc and lib/clang_<v>/dtwc_core.lib, linked as build_clang.sh links,
# with src/<lanes v>/dtwc/core/dtw_lanes.cpp compiled by the tree's flags (+ extra) and placed first in the link
# (the library's own dtw_lanes object is then not pulled in), after a <pad>-byte padding object (diag/pad.s), so the
# lanes code moves by <pad> bytes.
set -e
cd "$(dirname "$0")/.."
N=$1; V=$2; LV=$3; PAD=$4; shift 4
T=C:/D/git/wt/RL/build
CXX="/c/Program Files/LLVM/bin/clang++.exe"
O=diag/bin/$N
FP="-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
mkdir -p $O
"$CXX" -DDTWC_ENABLE_GUROBI -DDTWC_ENABLE_HIGHS -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" \
  -Isrc/$LV/dtwc/. -Isrc/$LV/dtwc/extern -Isrc/$LV/dtwc/mip/. -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang \
  --dependent-lib=msvcrt -march=native -fno-finite-math-only -Werror=switch -fopenmp $FP -fno-lto "$@" \
  -c src/$LV/dtwc/core/dtw_lanes.cpp -o $O/dtw_lanes.obj
printf '\t.text\n\t.globl\trl_pad\n\t.p2align 6\nrl_pad:\n\t.fill %s, 1, 0x90\n' "$PAD" > $O/pad.s
[ "$PAD" -gt 0 ] || printf '\t.text\n' > $O/pad.s
"$CXX" -c $O/pad.s -o $O/pad.obj
"$CXX" -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -Isrc/$V/dtwc/. -O3 -DNDEBUG -std=c++20 \
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -march=native -fopenmp $FP -c rprobe.cpp -o $O/rprobe.obj
"$CXX" -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -Xlinker /subsystem:console -fuse-ld=lld-link \
  $O/pad.obj $O/dtw_lanes.obj $O/rprobe.obj lib/clang_$V/dtwc_core.lib $T/GurobiCXX.lib C:/gurobi1301/win64/lib/gurobi130.lib \
  $T/bin/highs.lib "C:/Program Files/LLVM/lib/libomp.lib" -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 \
  -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames -o $O/rprobe.exe
echo "$N: $(uv run --no-project python tools/exe_loops.py $O/rprobe.exe | grep 'f64 L1')"
