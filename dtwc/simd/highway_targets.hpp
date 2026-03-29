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
 * @date 29 Mar 2026
 */

// Highway's foreach_target.h handles the multi-ISA compilation loop.
// It re-includes HWY_TARGET_INCLUDE once per target (AVX-512, AVX2, SSE4, NEON, etc.)
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"         // IWYU pragma: keep
