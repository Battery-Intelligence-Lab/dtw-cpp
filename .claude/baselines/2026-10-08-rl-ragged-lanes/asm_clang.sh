#!/bin/sh
# asm_clang.sh <v> <dtwc dir>: asm/clang_<v>_dtw_lanes.s, dtw_lanes.cpp of <dtwc dir> compiled -S -masm=intel with the
# flags of the RL clang tree's compile_commands.json for that file (clang++ 21.1.8, -O3 -march=native, the FP subset,
# -fno-lto as dtwc/core/CMakeLists.txt sets it on the MSVC ABI), the -I path pointed at <dtwc dir>; then the
# innermost min loops of each lanes kernel (tools/loops.py --all) into asm/clang_<v>_loops.txt.
set -e
cd "$(dirname "$0")"
V=$1; D=$2
CXX="/c/Program Files/LLVM/bin/clang++.exe"
mkdir -p asm
"$CXX" -DDTWC_ENABLE_GUROBI -DDTWC_ENABLE_HIGHS -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" \
  -I$D/. -I$D/extern -I$D/mip/. -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -march=native \
  -fno-finite-math-only -Werror=switch -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math \
  -fno-signed-zeros -fno-rounding-math -fno-lto -S -masm=intel $D/core/dtw_lanes.cpp -o asm/clang_${V}_dtw_lanes.s
uv run --no-project python tools/loops.py asm/clang_${V}_dtw_lanes.s --all > asm/clang_${V}_loops.txt
echo "asm/clang_${V}_dtw_lanes.s: $(wc -l < asm/clang_${V}_dtw_lanes.s) lines"
