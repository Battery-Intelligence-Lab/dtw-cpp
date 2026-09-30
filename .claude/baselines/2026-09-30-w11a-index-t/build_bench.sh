#!/usr/bin/env bash
# Build the scratch PAM swap benchmark against a W11a build tree, with the flags
# the faster_pam unit test uses (ThinLTO, as the shipped binaries link).
# usage: build.sh <build_dir> <out_exe>
set -eu
B=$1
OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$B"
"C:/Program Files/LLVM/bin/clang++.exe" -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML \
  -I"${REPO:-C:/D/git/wt/W11a}/dtwc/." -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt \
  -flto=thin -march=native -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math \
  -fassociative-math -fno-signed-zeros -fno-rounding-math \
  -c "$HERE/bench_pam_swap.cpp" -o "$HERE/bench_pam_swap.obj"
"C:/Program Files/LLVM/bin/clang++.exe" -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin \
  -Xlinker /subsystem:console -fuse-ld=lld-link "$HERE/bench_pam_swap.obj" -o "$OUT" \
  bin/dtwc++.lib GurobiCXX.lib C:/gurobi1301/win64/lib/gurobi130.lib bin/highs.lib \
  "C:/Program Files/LLVM/lib/libomp.lib" -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 \
  -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames
echo "built $OUT"
