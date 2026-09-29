#!/usr/bin/env bash
# Build p1_fill_ucr.exe against the library currently in C:/D/git/wt/P1/build (the
# benchmark TU's compile flags and link line from that build), into build/bin/<name>.exe.
#   build_driver.sh <name>
set -euo pipefail
NAME="$1"
SRC="$(cd "$(dirname "$0")" && pwd)/p1_fill_ucr.cpp"
CXX="C:/Program Files/LLVM/bin/clang++.exe"
cd C:/D/git/wt/P1/build
"$CXX" -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML '-DDTWC_VERSION_STRING="2.0.0rc1"' \
  -DNTKERNEL_ERROR_CATEGORY_INLINE -DNTKERNEL_ERROR_CATEGORY_STATIC -D_WIN32_WINNT=0x601 \
  -IC:/D/git/wt/P1/dtwc/. -IC:/D/cpm-cache/llfio/46e0/include \
  -IC:/D/cpm-cache/llfio/46e0/include/llfio/ntkernel-error-category/include \
  -isystem C:/D/cpm-cache/cli11/ac32/include -isystem C:/D/git/wt/P1/build/install/include \
  -O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -march=native -fopenmp \
  -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math \
  -o "bin/$NAME.obj" -c "$SRC"
"$CXX" -nostartfiles -nostdlib -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin \
  -Xlinker /subsystem:console -fuse-ld=lld-link "bin/$NAME.obj" -o "bin/$NAME.exe" \
  bin/dtwc++.lib GurobiCXX.lib C:/gurobi1301/win64/lib/gurobi130.lib bin/highs.lib \
  "C:/Program Files/LLVM/lib/libomp.lib" -lshlwapi.lib -lkernel32 -luser32 -lgdi32 -lwinspool \
  -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames
rm -f "bin/$NAME.obj"
ls -la "bin/$NAME.exe"
