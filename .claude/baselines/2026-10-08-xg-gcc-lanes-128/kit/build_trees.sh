#!/bin/sh
# build_trees.sh, in WSL: the repo at b3b3a57e built twice with GCC 13.3, Release, DTWC_ARCH_LEVEL=v3 (the Linux
# release CLI's level, so its flags; HiGHS OFF: no MIP here, the flags of our code are the same), tests ON:
# ~/xg/repo/build-base (design-2.0's header) and ~/xg/repo-cand/build-cand (the candidate's one line), each only the
# targets the clustering check runs. Under taskset -c 0-7: the machine is shared, and GCC's LTO partitions follow the
# affinity. Logs beside each tree.
set -e
B=$HOME/xg
KIT=$(cd "$(dirname "$0")/.." && pwd)
if [ ! -d "$B/repo-cand" ]; then
  mkdir -p "$B/repo-cand"
  (cd "$B/repo" && tar --exclude=./build-gcc --exclude='./build-base*' -cf - .) | (cd "$B/repo-cand" && tar -xf -)
  sed -i 's/^#if defined(__aarch64__) || (defined(__x86_64__) \&\& defined(__clang__))$/#if defined(__aarch64__) || defined(__x86_64__)/' \
    "$B/repo-cand/dtwc/core/dtw_kernel.hpp"
fi
echo "base dtwc/ vs the kit's src/base: $(diff -r "$B/repo/dtwc" "$KIT/src/base/dtwc" | wc -l) diff lines"
echo "cand dtwc/ vs the kit's src/cand: $(diff -r "$B/repo-cand/dtwc" "$KIT/src/cand/dtwc" | wc -l) diff lines"
diff -r "$B/repo/dtwc" "$B/repo-cand/dtwc" || true
for v in base cand; do
  S=$B/repo; [ $v = cand ] && S=$B/repo-cand
  T=$S/build-$v
  taskset -c 0-7 cmake -S "$S" -B "$T" -G Ninja -DCMAKE_BUILD_TYPE=Release -DDTWC_ARCH_LEVEL=v3 \
    -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_LLFIO=ON -DDTWC_BUILD_TESTING=ON \
    -DCPM_SOURCE_CACHE="$HOME/cpm-cache" > "$T.configure.log" 2>&1
  start=$(date +%s)
  taskset -c 0-7 cmake --build "$T" -j 8 --target dtwc_cl cpp_conformance unit_test_dtw_kernel_lanes \
    unit_test_one_batch_pam > "$T.build.log" 2>&1
  echo "$v: configured and built in $(( $(date +%s) - start )) s; $(grep -c -i 'warning' "$T.build.log") warning lines"
done
