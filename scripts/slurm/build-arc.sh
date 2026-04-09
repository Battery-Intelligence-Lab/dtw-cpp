#!/usr/bin/env bash
# Build DTWC++ on Oxford ARC SLURM clusters
#
# Usage:   source scripts/slurm/build-arc.sh [profile]
# Profiles: arc       — arc cluster (Cascade Lake + Turin), CPU only, AVX-512
#           htc-cpu   — htc cluster CPU-only, AVX2 portable (covers Broadwell→Turin)
#           htc-gpu   — htc cluster GPU build, all GPU archs, AVX2 portable
#           htc-v4    — htc cluster, AVX-512 only (excludes Broadwell/Rome nodes)
#           h100      — htc H100 nodes, AVX-512, sm_90 only (fastest compile)
#           grace     — htc-g057 Grace Hopper (AArch64), no CUDA yet
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
            -DDTWC_ENABLE_MPI=ON           # HDR100/NDR400 interconnect available
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
    # htc GPU: all GPU architectures, portable CPU (AVX2)
    # GPUs: P100(60), V100(70), RTX8000/TitanRTX(75), A100(80),
    #       RTXA6000(86), L40S(89), H100/GH200(90)
    # ════════════════════════════════════════════════════════════════════════
    htc-gpu)
        echo "═══ Profile: htc-gpu (all GPUs, AVX2 portable) ═══"
        CMAKE_ARGS=(
            "${CMAKE_COMMON[@]}"
            -DDTWC_ARCH_LEVEL=v3           # Portable across all htc CPU nodes
            -DDTWC_ENABLE_CUDA=ON
            -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90"
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

echo "Build directory: ${BUILD_DIR}"
echo "CMake args: ${CMAKE_ARGS[*]}"
echo ""

cmake -S . -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j "${NPROC}"

echo ""
echo "═══ Build complete. Run tests: ═══"
echo "  ctest --test-dir ${BUILD_DIR} -C Release -j ${NPROC}"
