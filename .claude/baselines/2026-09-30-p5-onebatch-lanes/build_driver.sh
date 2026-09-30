#!/usr/bin/env bash
# build_driver.sh <name>: compile p5_bench.cpp against the P5 worktree's current build, to <scratch>/<name>.exe
set -euo pipefail
NAME="$1"
SRC=C:/D/git/wt/tmp/P5-scratch/p5_bench.cpp
CXX="C:/Program Files/LLVM/bin/clang++.exe"
cd C:/D/git/wt/P5/build
"$CXX" -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML '-DDTWC_VERSION_STRING="2.0.0rc1"' \
  -DNTKERNEL_ERROR_CATEGORY_INLINE -DNTKERNEL_ERROR_CATEGORY_STATIC -D_WIN32_WINNT=0x601 \
  -IC:/D/git/wt/P5/dtwc/. -IC:/D/cpm-cache/llfio/46e0/include \
  -IC:/D/cpm-cache/llfio/46e0/include/llfio/ntkernel-error-category/include \
  -isystem C:/D/cpm-cache/cli11/ac32/include -isystem C:/D/git/wt/P5/build/install/include \
  -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -march=native -fopenmp \
  -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math \
  -o "C:/D/git/wt/tmp/P5-scratch/$NAME.obj" -c "$SRC"
"$CXX" -nostartfiles -nostdlib -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin \
  -Xlinker /subsystem:console -fuse-ld=lld-link "C:/D/git/wt/tmp/P5-scratch/$NAME.obj" -o "C:/D/git/wt/tmp/P5-scratch/$NAME.exe" \
  bin/dtwc++.lib GurobiCXX.lib C:/gurobi1301/win64/lib/gurobi130.lib bin/highs.lib \
  "C:/Program Files/LLVM/lib/libomp.lib" -lshlwapi.lib -lkernel32 -luser32 -lgdi32 -lwinspool \
  -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames
rm -f "C:/D/git/wt/tmp/P5-scratch/$NAME.obj"
ls -la "C:/D/git/wt/tmp/P5-scratch/$NAME.exe"
