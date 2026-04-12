#!/usr/bin/env bash
# Google Benchmark wrapper that strips the hostname from the JSON output.
#
# Runs the given benchmark binary with --benchmark_out=<file> (if not already
# supplied), then blanks the auto-filled "host_name" field. CPU specs, GPU
# specs (for bench_metal_dtw / bench_cuda_dtw, injected via AddCustomContext),
# caches, library version, etc. are preserved — only the hostname is removed.
#
# Usage:
#   scripts/run_bench.sh <bench_binary> [extra benchmark args...]
#   scripts/run_bench.sh <bench_binary> --benchmark_out=path.json [args...]
#
# Examples:
#   scripts/run_bench.sh ./build/bin/bench_metal_dtw \
#       --benchmark_filter=BM_metal_distanceMatrix/100/1000 \
#       --benchmark_out=benchmarks/results/mac_m2max/metal_latest.json
#
#   # Auto-generates a timestamped file under benchmarks/results/_autorun/
#   scripts/run_bench.sh ./build/bin/bench_cuda_dtw
#
# If --benchmark_format is not set, defaults to json (since we're post-processing).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bench_binary> [extra benchmark args...]" >&2
  exit 64
fi

bin="$1"; shift

if [[ ! -x "$bin" ]]; then
  echo "error: '$bin' is not executable" >&2
  exit 66
fi

# Resolve repo root relative to this script so default output paths work
# regardless of the caller's cwd.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"

# Detect whether the user already passed --benchmark_out; if not, generate one.
out_file=""
args=("$@")
saw_out=0
saw_format=0
for a in "${args[@]}"; do
  case "$a" in
    --benchmark_out=*)    out_file="${a#--benchmark_out=}"; saw_out=1 ;;
    --benchmark_out)      saw_out=1 ;;
    --benchmark_format=*) saw_format=1 ;;
  esac
done

if [[ $saw_out -eq 0 ]]; then
  default_dir="$repo_root/benchmarks/results/_autorun"
  mkdir -p "$default_dir"
  ts="$(date +%Y%m%d_%H%M%S)"
  out_file="$default_dir/$(basename "$bin")_${ts}.json"
  args+=("--benchmark_out=$out_file")
fi

if [[ $saw_format -eq 0 ]]; then
  args+=("--benchmark_format=json")
fi

echo "[run_bench] $bin ${args[*]}"
"$bin" "${args[@]}"

# Strip the hostname — keep all other context (CPU specs, GPU specs, caches).
# Works on macOS/BSD sed and GNU sed alike by omitting -i's argument pattern.
if [[ -f "$out_file" ]]; then
  if sed --version >/dev/null 2>&1; then
    # GNU sed
    sed -i -E 's/"host_name": "[^"]*"/"host_name": ""/' "$out_file"
  else
    # BSD sed (macOS)
    sed -i '' -E 's/"host_name": "[^"]*"/"host_name": ""/' "$out_file"
  fi
  echo "[run_bench] wrote $out_file (host_name stripped)"
else
  echo "[run_bench] warning: expected output file '$out_file' not found" >&2
fi
