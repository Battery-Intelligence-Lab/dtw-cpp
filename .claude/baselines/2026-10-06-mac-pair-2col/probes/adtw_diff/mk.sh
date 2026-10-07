#!/bin/sh
# Copies the base header into namespace base_core (internal calls qualified, so ADL on dtwc::core Cost types
# cannot reach the other copy) and builds diff.cpp against it and a candidate tree's headers, with the library's
# compile flags; then run ./diff.
# usage: mk.sh <candidate dtwc dir> <base dtwc dir>       (e.g. the kit's src/head and src/base)
set -e
cd "$(dirname "$0")"
sed -e 's/^namespace dtwc::core {/namespace base_core {/' -e 's|^} // namespace dtwc::core|} // namespace base_core|' \
    -e 's/return dtw_kernel_linear<T>(/return ::base_core::dtw_kernel_linear<T>(/g' \
    -e 's/return dtw_kernel_banded<T>(/return ::base_core::dtw_kernel_banded<T>(/g' "$2/core/dtw_kernel.hpp" > base_kernel.hpp
FLAGS="-O3 -DNDEBUG -std=c++20 -flto=thin -arch arm64 -fPIC -march=native -fno-finite-math-only -Xclang -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
/usr/bin/clang++ -DDTWC_HAS_OPENMP -I"$1" -I. -isystem /opt/homebrew/opt/libomp/include $FLAGS diff.cpp -L/opt/homebrew/opt/libomp/lib -lomp -Wl,-rpath,/opt/homebrew/opt/libomp/lib -o diff
echo built diff
