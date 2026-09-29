#!/usr/bin/env bash
# W4a: wavefront_3buf.inc = dtw_wavefront_kernel with DOUBLE_BUF_MAX = 0, renamed (the kernel
# as it would be with the double-buffer mode deleted). Usage: make_clone.sh <repo> <out dir>
set -eu
src="$1/dtwc/cuda/cuda_dtw.cu"
out="$2"
start=$(( $(grep -n '^__global__ void dtw_wavefront_kernel(' "$src" | cut -d: -f1) - 1 ))
end=$(awk -v s="$start" 'NR > s && /^}$/ {print NR; exit}' "$src")
sed -n "${start},${end}p" "$src" > "$out/wavefront_orig.inc"
sed -e 's/^__global__ void dtw_wavefront_kernel(/__global__ void dtw_wavefront_kernel_3buf(/' \
    -e 's|constexpr int DOUBLE_BUF_MAX = 2048;|constexpr int DOUBLE_BUF_MAX = 0; // W4a clone: the double-buffer mode never runs|' \
    "$out/wavefront_orig.inc" > "$out/wavefront_3buf.inc"
# Exactly two lines differ: the name and the constant.
test "$(diff "$out/wavefront_orig.inc" "$out/wavefront_3buf.inc" | grep -c '^>')" = 2
echo "clone of lines $start-$end written to $out/wavefront_3buf.inc"
