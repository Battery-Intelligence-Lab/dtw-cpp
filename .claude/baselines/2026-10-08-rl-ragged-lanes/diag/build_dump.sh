#!/bin/sh
# build_dump.sh: diag/dump against each WSL tree (GCC, dtwc_cl.cpp's flags, LTO), run both, diff.
set -e
KIT=$(cd "$(dirname "$0")/.." && pwd)
FP="-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -fno-signaling-nans"
for v in base cand; do
  S=$HOME/rl/repo-$v; T=$S/build-$v
  g++ -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -I$S/dtwc/. -O3 -DNDEBUG -std=c++20 -flto=auto -fno-fat-lto-objects -march=x86-64-v3 -fopenmp $FP \
    -c "$KIT/diag/dump.cpp" -o $HOME/rl/dump_$v.o
  g++ -O3 -flto=auto -fno-fat-lto-objects $HOME/rl/dump_$v.o $T/bin/libdtwc_core.a /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so -o $HOME/rl/dump_$v
  $HOME/rl/dump_$v > $HOME/rl/dump_$v.txt
done
diff $HOME/rl/dump_base.txt $HOME/rl/dump_cand.txt | grep '^<' | awk '{print "kind", $2, "n", $3, "band", $4}' > "$KIT/diag/gcc_f64sq_differ.txt" || true
wc -l < "$KIT/diag/gcc_f64sq_differ.txt"
