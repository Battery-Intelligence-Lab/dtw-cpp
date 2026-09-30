#!/usr/bin/env bash
# Usage: build_cpu.sh <worktree> <out.exe> -- long_fill against <worktree>/build's clang dtwc++.lib
# (compile and link lines: build's bench_cuda_dtw target, from compile_commands.json and build.ninja).
set -eu
wt="$1"; out="$2"; here="$(cd "$(dirname "$0")" && pwd)"
cd "$wt/build"
clang++ -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML '-DDTWC_VERSION_STRING="2.0.0rc1"' \
  -I"$wt/dtwc" -I"$wt/tests" -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt \
  -flto=thin -march=native -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math \
  -fassociative-math -fno-signed-zeros -fno-rounding-math -c "$here/long_fill.cpp" -o "$out.obj"
clang++ -nostartfiles -nostdlib -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin \
  -Xlinker /subsystem:console -fuse-ld=lld-link "$out.obj" -o "$out" bin/dtwc++.lib GurobiCXX.lib \
  C:/gurobi1301/win64/lib/gurobi130.lib bin/highs.lib "C:/Program Files/LLVM/lib/libomp.lib" \
  -lshlwapi.lib -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid \
  -lcomdlg32 -ladvapi32 -loldnames
echo BUILD_OK
