#!/bin/sh
# build_clang.sh base|A|B|B2: bin/clang_<v>/rprobe.exe from rprobe.cpp against src/<v>/dtwc (headers) and
# lib/clang_<v>/dtwc_core.lib (that version's library, saved from the RL clang tree C:/D/git/wt/RL/build: clang++
# 21.1.8, GNU-style driver, MSVC ABI, configured as the main checkout's build/). Compiled with dtwc_cl.cpp's flags
# from the tree's compile_commands.json and linked as dtwc_cl.exe is (lld-link, ThinLTO; HiGHS, Gurobi and libomp
# from the tree and the system: the unit changes neither). So `fill` runs that version's Problem::fill_distance_matrix,
# and the sweeps its block and per-pair functions, as dtwc_cl.exe does.
set -e
cd "$(dirname "$0")"
V=$1
T=C:/D/git/wt/RL/build
CXX="/c/Program Files/LLVM/bin/clang++.exe"
O=bin/clang_$V
mkdir -p $O
"$CXX" -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -Isrc/$V/dtwc/. -O3 -DNDEBUG -std=c++20 \
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -march=native -fopenmp -fno-math-errno -fno-trapping-math \
  -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -c rprobe.cpp -o $O/rprobe.obj
"$CXX" -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -flto=thin -Xlinker /subsystem:console -fuse-ld=lld-link \
  $O/rprobe.obj lib/clang_$V/dtwc_core.lib $T/GurobiCXX.lib C:/gurobi1301/win64/lib/gurobi130.lib $T/bin/highs.lib \
  "C:/Program Files/LLVM/lib/libomp.lib" -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid \
  -lcomdlg32 -ladvapi32 -loldnames -o $O/rprobe.exe
echo "built $O/rprobe.exe"
