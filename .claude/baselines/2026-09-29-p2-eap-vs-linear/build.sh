#!/bin/sh
# Flags: the dtwc/core/dtw.cpp command in C:/D/git/dtw-cpp/build/compile_commands.json (design-2.0 @ 59750c4)
# minus -flto=thin and minus the dynamic-CRT trio (-D_DLL -D_MT -Xclang --dependent-lib=msvcrt).
CXX="C:/Program Files/LLVM/bin/clang++.exe"
R=C:/D/git/dtw-cpp
FLAGS="-DDTWC_ENABLE_GUROBI -DDTWC_ENABLE_HIGHS -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" \
 -DNTKERNEL_ERROR_CATEGORY_INLINE -DNTKERNEL_ERROR_CATEGORY_STATIC -DQUICKCPPLIB_REQUIRE_CXX_STANDARD=201402L \
 -DQUICKCPPLIB_USE_SYSTEM_BYTE_LITE=1 -DQUICKCPPLIB_USE_SYSTEM_SPAN_LITE=1 -D_WIN32_WINNT=0x601 \
 -I$R/dtwc/. -I$R/dtwc/extern -I$R/dtwc/mip/. \
 -O3 -DNDEBUG -std=c++20 -march=native -fno-finite-math-only -fno-math-errno -fno-trapping-math -freciprocal-math \
 -fassociative-math -fno-signed-zeros -fno-rounding-math -fopenmp"
cd "$(dirname "$0")"
"$CXX" $FLAGS -fno-color-diagnostics p2_eap_vs_linear.cpp -o p2_eap_vs_linear.exe || exit 1
"$CXX" $FLAGS -fno-color-diagnostics -S -masm=intel p2_eap_vs_linear.cpp -o p2_eap_vs_linear.s || exit 1
echo built
