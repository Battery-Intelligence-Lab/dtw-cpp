#!/bin/sh
# build_gcc.sh base|A|B, in WSL: ~/rl/bin/gcc_<v>/rprobe from ../rprobe.cpp against the kit's src/<v>/dtwc (headers)
# and ~/rl/lib/gcc_<v>/libdtwc_core.a (that version's library, saved from ~/rl/repo-base/build-base for base and from
# ~/rl/repo-cand/build-cand when it held A, then B; see trees.sh), with dtwc_cl.cpp's flags from the tree's
# compile_commands.json (GCC 13.3, Release, DTWC_ARCH_LEVEL=v3: -march=x86-64-v3, the FP subset, -fno-signaling-nans,
# LTO) and linked as dtwc_cl is (LTO, libgomp). Also asm/gcc_<v>_dtw_lanes.s (dtw_lanes.cpp with dtwc_core's flags,
# -S -masm=intel, LTO off: an LTO object holds no code), its min loops (tools/loops.py), and asm/gcc_<v>_rprobe.dis
# (objdump of the linked probe: the code that runs).
set -e
KIT=$(cd "$(dirname "$0")/.." && pwd)
V=$1
D=$KIT/src/$V/dtwc
O=$HOME/rl/bin/gcc_$V
FP="-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -fno-signaling-nans"
LTO="-flto=auto -fno-fat-lto-objects"
mkdir -p "$O" "$KIT/asm"
g++ -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -I$D/. -O3 -DNDEBUG -std=c++20 $LTO \
  -march=x86-64-v3 -fopenmp $FP -c "$KIT/rprobe.cpp" -o "$O/rprobe.o"
g++ -O3 -DNDEBUG $LTO "$O/rprobe.o" "$HOME/rl/lib/gcc_$V/libdtwc_core.a" /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so \
  /usr/lib/x86_64-linux-gnu/libpthread.a -ldl -o "$O/rprobe"
g++ -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -I$D/. -I$D/extern -I$D/mip/. -O3 -DNDEBUG -std=c++20 -fPIC \
  -march=x86-64-v3 -fno-finite-math-only -Werror=switch -fopenmp $FP -S -masm=intel "$D/core/dtw_lanes.cpp" \
  -o "$KIT/asm/gcc_${V}_dtw_lanes.s"
python3 "$KIT/tools/loops.py" "$KIT/asm/gcc_${V}_dtw_lanes.s" --all > "$KIT/asm/gcc_${V}_loops.txt"
objdump -d -M intel --no-show-raw-insn -C "$O/rprobe" > "$KIT/asm/gcc_${V}_rprobe.dis"
echo "built $O/rprobe ($(g++ --version | head -1))"
