#!/bin/sh
# Exports each version's dtwc/ into src/<version>/ (read only; worktrees share the object store, so any checkout
# of the repository has the three commits).
# usage: ./prepare_src.sh <base commit> <k1 commit> <head commit>      e.g. e26d5680 463d2b52 00fb9c36
#        DTWC_REPO=<checkout> ./prepare_src.sh ...                      (default /Users/engs2321/git/dtw-cpp)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=${DTWC_REPO:-/Users/engs2321/git/dtw-cpp}
for pair in "base $1" "k1 $2" "head $3"; do
  set -- $pair
  rm -rf "$HERE/src/$1"
  mkdir -p "$HERE/src/$1"
  git -C "$REPO" archive "$2" dtwc | tar -x -C "$HERE/src/$1" --strip-components=1
  echo "src/$1 = $2, dtw_kernel.hpp $(shasum "$HERE/src/$1/core/dtw_kernel.hpp" | cut -c1-12)"
done
