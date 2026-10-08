#!/bin/sh
# trees.sh [base|cand|both], in WSL: the repo at e691e48c (base, from ../src/base_e691e48c.tar) and with the unit's
# patch (cand: ../src/rl.patch applied to a fresh copy, then rsync --checksum onto ~/rl/repo-cand, so a rebuild
# recompiles only what the patch changed), each built with GCC 13.3, Release, DTWC_ARCH_LEVEL=v3 (the Linux release
# CLI's level and flags, as XG's clust/build_trees.sh; HiGHS and Gurobi OFF: no MIP here, our code's flags are the
# same), tests ON. Base builds the library and dtwc_cl; cand everything (the full suite runs there). Under
# taskset -c 0-7 (the machine is shared; GCC's LTO partitions follow the affinity). Logs beside each tree.
set -e
KIT=$(cd "$(dirname "$0")/.." && pwd)
B=$HOME/rl
mkdir -p "$B"
which="${1:-both}"
if [ ! -d "$B/repo-base" ]; then
  mkdir -p "$B/repo-base" && tar -xf "$KIT/src/base_e691e48c.tar" -C "$B/repo-base"
fi
rm -rf "$B/stage" && mkdir -p "$B/stage" && tar -xf "$KIT/src/base_e691e48c.tar" -C "$B/stage"
(cd "$B/stage" && patch -p1 --quiet < "$KIT/src/rl.patch")
mkdir -p "$B/repo-cand"
rsync -a --checksum "$B/stage/" "$B/repo-cand/"
echo "base vs cand dtwc/ and tests/: $(diff -r "$B/repo-base/dtwc" "$B/repo-cand/dtwc" | grep -c '^[<>]') + $(diff -r "$B/repo-base/tests" "$B/repo-cand/tests" | grep -c '^[<>]') differing lines"
for v in base cand; do
  [ "$which" = both ] || [ "$which" = $v ] || continue
  S=$B/repo-$v; T=$S/build-$v
  if [ ! -f "$T/CMakeCache.txt" ]; then
    taskset -c 0-7 cmake -S "$S" -B "$T" -G Ninja -DCMAKE_BUILD_TYPE=Release -DDTWC_ARCH_LEVEL=v3 \
      -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_LLFIO=ON -DDTWC_BUILD_TESTING=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCPM_SOURCE_CACHE="$HOME/cpm-cache" > "$T.configure.log" 2>&1
  fi
  start=$(date +%s)
  if [ $v = base ]; then targets="--target dtwc_core dtwc_cl"; else targets=""; fi
  taskset -c 0-7 cmake --build "$T" -j 8 $targets > "$T.build.log" 2>&1
  echo "$v: built in $(( $(date +%s) - start )) s; warning lines: $(grep -c -i 'warning' "$T.build.log")"
done
