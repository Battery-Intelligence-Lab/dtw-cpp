/**
 * @file highway_targets.hpp
 * @brief Google Highway boilerplate for multi-target SIMD compilation.
 *
 * @details This header sets up the Highway dynamic dispatch infrastructure.
 *          Each .cpp file that uses Highway includes this header, then defines
 *          its SIMD kernels inside the HWY_NAMESPACE block. Highway compiles
 *          each kernel for every supported ISA and selects the best at runtime.
 *
 *          Usage pattern in .cpp files:
 *            #undef HWY_TARGET_INCLUDE
 *            #define HWY_TARGET_INCLUDE "dtwc/simd/my_simd.cpp"
 *            #include "dtwc/simd/highway_targets.hpp"
 *            // This re-includes the .cpp once per target ISA.
 *
 *          Then inside the .cpp:
 *            HWY_BEFORE_NAMESPACE();
 *            namespace dtwc::simd::HWY_NAMESPACE {
 *              // ... SIMD code using hn:: aliases ...
 *            }
 *            HWY_AFTER_NAMESPACE();
 *
 * @note NO #pragma once — this header is intentionally re-included once per
 *       ISA target by Highway's foreach_target.h mechanism.
 *
 * **Why Highway instead of raw intrinsics?**
 * - Single binary with runtime ISA dispatch: no separate SSE4/AVX2/AVX-512 build
 *   variants are needed. One binary works optimally on any node in an HPC cluster.
 * - Portability: the same kernels compile for ARM NEON, RISC-V V, SVE, etc. —
 *   future-proofing for non-x86 HPC (AWS Graviton, NVIDIA Grace, etc.).
 * - Safety: the Highway API avoids the sharp-edge surface of raw intrinsic names
 *   (which differ across ISA families) and handles alignment rules automatically.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

// Highway's foreach_target.h handles the multi-ISA compilation loop.
// It re-includes HWY_TARGET_INCLUDE once per target (AVX-512, AVX2, SSE4, NEON, etc.)
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"         // IWYU pragma: keep
