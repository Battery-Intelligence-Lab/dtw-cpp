#!/usr/bin/env bash
# Build DTWC++ on Oxford ARC SLURM clusters
#
# Usage:   source scripts/slurm/build-arc.sh [profile]
# Profiles: arc       — arc cluster (Cascade Lake + Turin), CPU only, AVX-512
#           htc-cpu   — htc cluster CPU-only, AVX2 portable (covers Broadwell→Turin)
#           htc-gpu   — htc cluster GPU build, compute capability 8.0+ (A100, RTX A6000, L40S), AVX2 portable
#           htc-v4    — htc cluster, AVX-512 only (excludes Broadwell/Rome nodes)
#           h100      — htc H100 nodes, AVX-512, sm_90 only (fastest compile)
#           grace     — htc-g057 Grace Hopper (AArch64), no CUDA yet
#
# htc-gpu and h100 run on a GPU node build for that node: native CUDA architecture and native CPU
# tuning, the most specialised binary, which then runs only on that node type. On any other node, or
# on a GPU below compute capability 8.0 (the floor; the library refuses such a GPU), the profile's
# portable lists apply.
#
# Prerequisites: module load CMake/3.27.6  (or any >= 3.26)
#                module load GCC/13.2.0    (or any C++20-capable GCC/Clang)
#                module load CUDA/12.4.0   (for GPU builds)
#                module load Arrow/15.0.0  (optional — CPM fallback if missing)

set -euo pipefail

PROFILE="${1:-arc}"
BUILD_DIR="build-${PROFILE}"
NPROC=$(nproc)

# ── Common flags ────────────────────────────────────────────────────────────
# Override defaults via environment: DTWC_BUILD_TESTING=OFF, DTWC_ENABLE_ARROW=OFF, etc.
CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE="${DTWC_BUILD_TYPE:-Release}"
    -DDTWC_BUILD_TESTING="${DTWC_BUILD_TESTING:-ON}"
    -DDTWC_ENABLE_ARROW="${DTWC_ENABLE_ARROW:-ON}"
)

case "${PROFILE}" in

    # ════════════════════════════════════════════════════════════════════════
    # arc cluster: 262× Cascade Lake + 10× AMD Turin — all support AVX-512
    # ════════════════════════════════════════════════════════════════════════
    arc)
        echo "═══ Profile: arc (Cascade Lake + Turin, AVX-512, CPU only) ═══"
        CMAKE_ARGS=(
            "${CMAKE_COMMON[@]}"
            -DDTWC_ARCH_LEVEL=v4           # AVX-512 — safe on all arc nodes
            -DDTWC_ENABLE_CUDA=OFF
        )
        ;;

    # ════════════════════════════════════════════════════════════════════════
    # htc CPU-only: heterogeneous — Broadwell through Turin
    # Must use v3 (AVX2+FMA) for portability across ALL htc CPU nodes
    # ════════════════════════════════════════════════════════════════════════
    htc-cpu)
        echo "═══ Profile: htc-cpu (all CPU nodes, AVX2 portable) ═══"
        CMAKE_ARGS=(
            "${CMAKE_COMMON[@]}"
            -DDTWC_ARCH_LEVEL=v3           # AVX2+FMA — safe for Broadwell, Rome, Genoa, Turin
            -DDTWC_ENABLE_CUDA=OFF
        )
        ;;

    # ════════════════════════════════════════════════════════════════════════
    # htc GPU: compute capability 8.0 and newer, portable CPU (AVX2)
    # GPUs: A100(80), RTXA6000(86), L40S(89). H100 (90): use the h100 profile.
    # P100, V100, RTX8000 and Titan RTX are below the floor and are refused at run time.
    # ════════════════════════════════════════════════════════════════════════
    htc-gpu)
        echo "═══ Profile: htc-gpu (A100, RTX A6000, L40S, AVX2 portable) ═══"
        CMAKE_ARGS=(
            "${CMAKE_COMMON[@]}"
            -DDTWC_ARCH_LEVEL=v3           # Portable across all htc CPU nodes
            -DDTWC_ENABLE_CUDA=ON
            -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
        )
        ;;

    # ════════════════════════════════════════════════════════════════════════
    # htc AVX-512: excludes Broadwell (htc-g045-049) and Rome (htc-g019)
    # ════════════════════════════════════════════════════════════════════════
    htc-v4)
        echo "═══ Profile: htc-v4 (AVX-512, excludes Broadwell/Rome) ═══"
        CMAKE_ARGS=(
            "${CMAKE_COMMON[@]}"
            -DDTWC_ARCH_LEVEL=v4           # AVX-512 — all nodes except Broadwell/Rome
            -DDTWC_ENABLE_CUDA=OFF
        )
        ;;

    # ════════════════════════════════════════════════════════════════════════
    # H100-only: fastest compile, maximum performance
    # Nodes: htc-g[053-055,058-060] — 4-8× H100, 80-96GB HBM3
    # CPU: Sapphire/Emerald Rapids — full AVX-512
    # ════════════════════════════════════════════════════════════════════════
    h100)
        echo "═══ Profile: h100 (H100 nodes, AVX-512, sm_90 only) ═══"
        CMAKE_ARGS=(
            "${CMAKE_COMMON[@]}"
            -DDTWC_ARCH_LEVEL=v4           # AVX-512
            -DDTWC_ENABLE_CUDA=ON
            -DCMAKE_CUDA_ARCHITECTURES=90  # H100 only — fastest compile
        )
        ;;

    # ════════════════════════════════════════════════════════════════════════
    # Grace Hopper: AArch64 (ARM) — htc-g057, 72 cores, 580GB + 96GB GPU
    # No CUDA support yet (kernel needs AArch64 port)
    # ════════════════════════════════════════════════════════════════════════
    grace)
        echo "═══ Profile: grace (Grace Hopper AArch64, CPU only) ═══"
        CMAKE_ARGS=(
            "${CMAKE_COMMON[@]}"
            -DDTWC_ENABLE_NATIVE_ARCH=ON   # Let -march=native pick up NEON/SVE
            -DDTWC_ENABLE_CUDA=OFF         # CUDA kernel not yet ported to AArch64
        )
        ;;

    *)
        echo "Unknown profile: ${PROFILE}"
        echo "Usage: source scripts/slurm/build-arc.sh [arc|htc-cpu|htc-gpu|htc-v4|h100|grace]"
        return 1 2>/dev/null || exit 1
        ;;
esac

# True when this node has GPUs and every one is at or above the compute capability 8.0 floor.
gpu_node_supported() {
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
        awk '$1 < 8.0 { bad = 1 } END { exit !(NR > 0 && !bad) }'
}

if [[ "${PROFILE}" == htc-gpu || "${PROFILE}" == h100 ]] && gpu_node_supported; then
    echo "GPU node: native CUDA architecture and CPU tuning"
    # An empty ARCH_LEVEL also clears a v3/v4 cached by an earlier portable build in this directory.
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES=native -DDTWC_ENABLE_NATIVE_ARCH=ON -DDTWC_ARCH_LEVEL=)
fi

echo "Build directory: ${BUILD_DIR}"
echo "CMake args: ${CMAKE_ARGS[*]}"
echo ""

cmake -S . -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j "${NPROC}"

echo ""
echo "═══ Build complete. Run tests: ═══"
echo "  ctest --test-dir ${BUILD_DIR} -C Release -j ${NPROC}"
