/**
 * @file KernelOverride.hpp
 * @brief KernelOverride enum — force a specific GPU DTW kernel path.
 *
 * @details Both CUDA and Metal backends auto-select kernels based on
 *          `max_L` and `band`. This enum lets advanced users override
 *          that choice for benchmarking or to work around a heuristic
 *          miss on atypical workloads. Backends silently fall back to
 *          `Auto` when the requested path is unsupported (e.g. BandedRow
 *          on CUDA, which only exists on Metal).
 *
 * @date 2026-04-12
 */

#pragma once

namespace dtwc {

enum class KernelOverride {
  Auto,             ///< Backend's own heuristic (default).
  Wavefront,        ///< Anti-diagonal wavefront, threadgroup/shared memory.
  WavefrontGlobal,  ///< Anti-diagonal wavefront, device/global memory.
  BandedRow,        ///< Row-major banded kernel (Metal-only).
  RegTile           ///< Register-tile kernel (unbanded, max_L small).
};

} // namespace dtwc
