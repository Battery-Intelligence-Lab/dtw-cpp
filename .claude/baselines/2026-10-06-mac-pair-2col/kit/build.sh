#!/bin/sh
# Builds pprobe_<version>_<placement> from src/pprobe.cpp and src/<version>/ (that commit's dtwc/: base =
# e26d5680, k1 = kernel 1 two columns per pass, head = kernels 1 and 2), with the compile command of
# dtwc/core/dtw_dispatch.cpp and dtw_lanes.cpp in build/compile_commands.json (AppleClang, -march=native, the
# library's FP flags, ThinLTO). The three placements differ only at link time, where ThinLTO generates the code:
#   p0  the default layout, behind a 4-byte function (_kit_pad: ret) that kit.order puts first in __text
#   p1  as p0, every loop aligned to 64 bytes (-Wl,-mllvm,-align-loops=64)
#   p2  as p0 behind an 8-byte _kit_pad (nop; ret): every function, so every loop, 4 bytes later
# placement.sh shows where each binary's per-pair loops landed.
# usage: ./build.sh base|k1|head
set -e
cd "$(dirname "$0")"
V=$1
R=src/$V
[ -f "$R/core/dtw_kernel.hpp" ] || { echo "no sources in $R"; exit 1; }
FLAGS="-O3 -DNDEBUG -std=c++20 -flto=thin -arch arm64 -fPIC -march=native -fno-finite-math-only -Xclang -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math"
INC="-DDTWC_HAS_OPENMP -I$R -I$R/extern -isystem /opt/homebrew/opt/libomp/include"
mkdir -p obj
/usr/bin/clang++ $INC $FLAGS -c "$R/core/dtw_lanes.cpp" -o "obj/dtw_lanes_$V.o"
/usr/bin/clang++ $INC $FLAGS -c src/pprobe.cpp -o "obj/pprobe_$V.o"
/usr/bin/clang -arch arm64 -c pad/pad4.s -o obj/pad4.o
/usr/bin/clang -arch arm64 -c pad/pad8.s -o obj/pad8.o
LIBS="-L/opt/homebrew/opt/libomp/lib -lomp -Wl,-rpath,/opt/homebrew/opt/libomp/lib"
link() { # pad object, output, extra flags
  /usr/bin/clang++ -O3 -DNDEBUG -flto=thin -arch arm64 -Wl,-order_file,pad/kit.order $3 \
    "$1" "obj/dtw_lanes_$V.o" "obj/pprobe_$V.o" $LIBS -o "$2"
}
link obj/pad4.o "pprobe_${V}_p0" ""
link obj/pad4.o "pprobe_${V}_p1" "-Wl,-mllvm,-align-loops=64"
link obj/pad8.o "pprobe_${V}_p2" ""
echo "built pprobe_${V}_p0 pprobe_${V}_p1 pprobe_${V}_p2"
