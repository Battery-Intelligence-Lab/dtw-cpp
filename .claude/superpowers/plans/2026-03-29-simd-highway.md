# SIMD via Google Highway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Google Highway for runtime-dispatched SIMD (AVX2/AVX-512/SSE4/NEON) to accelerate LB_Keogh (4-8x), z_normalize (4-8x), and multi-pair DTW (3-4x on fillDistanceMatrix).

**Architecture:** Highway `.cpp` files use `HWY_DYNAMIC_DISPATCH` for runtime ISA selection. Each SIMD function has a scalar fallback guarded by `#ifdef DTWC_HAS_HIGHWAY`. The existing `core/` headers dispatch to SIMD when available. Multi-pair DTW processes 4 independent pair computations in AVX2 lanes simultaneously.

**Tech Stack:** Google Highway 1.2.0 (via CPM), C++17, Google Benchmark 1.9.1

**Spec:** `docs/superpowers/specs/2026-03-29-simd-highway-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `dtwc/simd/highway_targets.hpp` | Highway boilerplate: includes, `HWY_BEFORE_NAMESPACE`/`HWY_AFTER_NAMESPACE`, tag dispatch helper |
| Create | `dtwc/simd/lb_keogh_simd.cpp` | SIMD LB_Keogh using Highway Load/Sub/Max/ReduceSum |
| Create | `dtwc/simd/z_normalize_simd.cpp` | SIMD z_normalize: sum, squared-deviation, normalize loops |
| Create | `dtwc/simd/multi_pair_dtw.hpp` | Multi-pair DTW: 4-wide `dtwFull_L` processing 4 pairs in AVX2 lanes |
| Create | `dtwc/simd/multi_pair_dtw.cpp` | Multi-pair DTW implementation with Highway dispatch |
| Modify | `cmake/Dependencies.cmake` | Add Highway CPM fetch gated on `DTWC_ENABLE_SIMD` |
| Modify | `CMakeLists.txt` (root) | Add `DTWC_ENABLE_SIMD` option |
| Modify | `dtwc/CMakeLists.txt` | Add `dtwc/simd/` sources, link `hwy::hwy` conditionally |
| Modify | `dtwc/core/lower_bound_impl.hpp` | Add SIMD dispatch for `lb_keogh` |
| Modify | `dtwc/core/z_normalize.hpp` | Add SIMD dispatch for `z_normalize` |
| Modify | `dtwc/Problem.cpp` | Use multi-pair DTW in `fillDistanceMatrix` when Highway available |
| Create | `tests/unit/unit_test_simd.cpp` | Tests: SIMD results match scalar exactly |
| Modify | `benchmarks/bench_dtw_baseline.cpp` | Add SIMD benchmarks for LB_Keogh, z_normalize |
| Modify | `benchmarks/CMakeLists.txt` | Link Highway if available |
| Modify | `CHANGELOG.md` | Document SIMD additions |

---

### Task 1: CMake Integration (B0)

**Files:**
- Modify: `CMakeLists.txt:19-25`
- Modify: `cmake/Dependencies.cmake:69-80`
- Modify: `dtwc/CMakeLists.txt:65-69`

- [ ] **Step 1: Add DTWC_ENABLE_SIMD option to root CMakeLists.txt**

In `CMakeLists.txt`, after line 25 (`option(DTWC_ENABLE_HIGHS ...)`), add:

```cmake
option(DTWC_ENABLE_SIMD "Enable SIMD acceleration via Google Highway" ON)
```

- [ ] **Step 2: Add Highway to Dependencies.cmake**

In `cmake/Dependencies.cmake`, before the closing `endfunction()` at line 82, add:

```cmake
  # Google Highway — portable SIMD (Apache-2.0, Google)
  # Runtime dispatch: AVX-512, AVX2, SSE4, NEON selected at load time.
  if(DTWC_ENABLE_SIMD)
    if(NOT TARGET hwy::hwy)
      CPMAddPackage(
        NAME highway
        GITHUB_REPOSITORY google/highway
        VERSION 1.2.0
        OPTIONS
          "HWY_ENABLE_TESTS OFF"
          "HWY_ENABLE_EXAMPLES OFF"
          "HWY_ENABLE_CONTRIB OFF"
          "HWY_ENABLE_INSTALL OFF"
          "BUILD_TESTING OFF"
      )
    endif()
    if(NOT TARGET hwy::hwy)
      message(WARNING "Google Highway not found — SIMD disabled")
      set(DTWC_ENABLE_SIMD OFF PARENT_SCOPE)
    endif()
  endif()
```

- [ ] **Step 3: Create the simd directory and empty placeholder**

```bash
mkdir -p dtwc/simd
```

- [ ] **Step 4: Wire Highway into dtwc library CMakeLists.txt**

In `dtwc/CMakeLists.txt`, after the OpenMP block (after line 69), add:

```cmake
# Google Highway SIMD (optional)
if(DTWC_ENABLE_SIMD AND TARGET hwy::hwy)
  target_sources(dtwc++ PRIVATE
    simd/lb_keogh_simd.cpp
    simd/z_normalize_simd.cpp
    simd/multi_pair_dtw.cpp
  )
  target_link_libraries(dtwc++ PRIVATE hwy::hwy)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_HIGHWAY)
  message(STATUS "SIMD enabled via Google Highway")
endif()
```

- [ ] **Step 5: Build test (without SIMD sources yet — just verify CMake config)**

```bash
cd build && cmake .. -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_SIMD=OFF && cmake --build . --target dtwc++ -j
```

Expected: builds successfully with SIMD OFF.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt cmake/Dependencies.cmake dtwc/CMakeLists.txt
git commit -m "feat(simd): add DTWC_ENABLE_SIMD option + Highway CPM integration

Google Highway 1.2.0 fetched via CPM when DTWC_ENABLE_SIMD=ON (default).
Provides runtime dispatch across AVX-512, AVX2, SSE4, and ARM NEON.
No SIMD code yet — this is the build system foundation."
```

---

### Task 2: Highway Boilerplate Header

**Files:**
- Create: `dtwc/simd/highway_targets.hpp`

- [ ] **Step 1: Write highway_targets.hpp**

```cpp
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
 *            #include "hwy/highway.h"
 *            #include "dtwc/simd/highway_targets.hpp"
 *            HWY_BEFORE_NAMESPACE();
 *            namespace dtwc::simd::HWY_NAMESPACE {
 *              // ... SIMD code using hn:: aliases ...
 *            }
 *            HWY_AFTER_NAMESPACE();
 *
 * @date 29 Mar 2026
 */

#pragma once

// Highway's foreach_target.h handles the multi-ISA compilation loop.
// It re-includes HWY_TARGET_INCLUDE once per target (AVX-512, AVX2, SSE4, NEON, etc.)
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"         // IWYU pragma: keep
```

- [ ] **Step 2: Commit**

```bash
git add dtwc/simd/highway_targets.hpp
git commit -m "feat(simd): add Highway boilerplate header for multi-target dispatch"
```

---

### Task 3: SIMD LB_Keogh (B1)

**Files:**
- Create: `dtwc/simd/lb_keogh_simd.cpp`
- Modify: `dtwc/core/lower_bound_impl.hpp:107-118`

The scalar LB_Keogh at `dtwc/core/lower_bound_impl.hpp:107-118` is:
```cpp
template <typename T>
T lb_keogh(const T *query, std::size_t n, const T *upper, const T *lower)
{
  T sum = T(0);
  for (std::size_t i = 0; i < n; ++i) {
    T excess_upper = query[i] - upper[i];
    T excess_lower = lower[i] - query[i];
    sum += std::max(T(0), std::max(excess_upper, excess_lower));
  }
  return sum;
}
```

- [ ] **Step 1: Write lb_keogh_simd.cpp**

```cpp
/**
 * @file lb_keogh_simd.cpp
 * @brief SIMD-accelerated LB_Keogh lower bound using Google Highway.
 *
 * @details Vectorizes the LB_Keogh reduction loop: three contiguous array reads,
 *          element-wise max(0, max(q-U, L-q)), horizontal sum. Processes 4 doubles
 *          per iteration on AVX2, 8 on AVX-512.
 *
 * @date 29 Mar 2026
 */

// Highway multi-target compilation: this file is re-included per ISA target.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dtwc/simd/lb_keogh_simd.cpp"
#include "dtwc/simd/highway_targets.hpp"

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace dtwc::simd::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

double LbKeoghSimd(const double* HWY_RESTRICT query,
                   const double* HWY_RESTRICT upper,
                   const double* HWY_RESTRICT lower,
                   std::size_t n)
{
  const hn::ScalableTag<double> d;
  const std::size_t N = hn::Lanes(d);

  auto sum_vec = hn::Zero(d);
  const auto zero = hn::Zero(d);

  std::size_t i = 0;
  for (; i + N <= n; i += N) {
    const auto q = hn::LoadU(d, query + i);
    const auto u = hn::LoadU(d, upper + i);
    const auto l = hn::LoadU(d, lower + i);

    const auto excess_upper = hn::Sub(q, u);  // query[i] - upper[i]
    const auto excess_lower = hn::Sub(l, q);  // lower[i] - query[i]
    const auto excess = hn::Max(excess_upper, excess_lower);
    const auto clamped = hn::Max(zero, excess);
    sum_vec = hn::Add(sum_vec, clamped);
  }

  double sum = hn::ReduceSum(d, sum_vec);

  // Scalar tail
  for (; i < n; ++i) {
    double eu = query[i] - upper[i];
    double el = lower[i] - query[i];
    double e = eu > el ? eu : el;
    if (e > 0.0) sum += e;
  }

  return sum;
}

}  // namespace dtwc::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

// --- Single-compilation-unit dispatch table ---
#if HWY_ONCE
namespace dtwc::simd {

HWY_EXPORT(LbKeoghSimd);

double lb_keogh_highway(const double* query, const double* upper,
                        const double* lower, std::size_t n)
{
  return HWY_DYNAMIC_DISPATCH(LbKeoghSimd)(query, upper, lower, n);
}

}  // namespace dtwc::simd
#endif  // HWY_ONCE
```

- [ ] **Step 2: Add the SIMD dispatch declaration and wiring in lower_bound_impl.hpp**

At the top of `dtwc/core/lower_bound_impl.hpp`, after line 26 (`#include <vector>`), add:

```cpp
#ifdef DTWC_HAS_HIGHWAY
namespace dtwc::simd {
double lb_keogh_highway(const double* query, const double* upper,
                        const double* lower, std::size_t n);
}  // namespace dtwc::simd
#endif
```

Then replace the pointer-based `lb_keogh` template at lines 107-118 with:

```cpp
template <typename T>
T lb_keogh(const T *query, std::size_t n,
           const T *upper, const T *lower)
{
#ifdef DTWC_HAS_HIGHWAY
  if constexpr (std::is_same_v<T, double>) {
    return static_cast<T>(simd::lb_keogh_highway(query, upper, lower, n));
  }
#endif
  // Scalar fallback
  T sum = T(0);
  for (std::size_t i = 0; i < n; ++i) {
    T excess_upper = query[i] - upper[i];
    T excess_lower = lower[i] - query[i];
    sum += std::max(T(0), std::max(excess_upper, excess_lower));
  }
  return sum;
}
```

Also add `#include <type_traits>` to the includes if not already present.

- [ ] **Step 3: Build with SIMD ON**

```bash
cd build && cmake .. -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_SIMD=ON && cmake --build . -j
```

Expected: compiles successfully. Highway fetched and linked.

- [ ] **Step 4: Run existing tests to verify SIMD produces identical results**

```bash
cd build && ctest --output-on-failure
```

Expected: all 34 tests pass. The adversarial lower-bounds tests validate `LB_Keogh <= DTW` property — if SIMD produces different results, these tests will catch it.

- [ ] **Step 5: Commit**

```bash
git add dtwc/simd/lb_keogh_simd.cpp dtwc/core/lower_bound_impl.hpp
git commit -m "feat(simd): SIMD LB_Keogh via Highway — vectorized reduction loop

Processes 4 doubles/iter (AVX2) or 8 (AVX-512) with runtime ISA dispatch.
Scalar fallback for non-double types and when Highway is disabled.
Existing adversarial tests validate correctness (LB <= DTW property)."
```

---

### Task 4: SIMD z_normalize (B2)

**Files:**
- Create: `dtwc/simd/z_normalize_simd.cpp`
- Modify: `dtwc/core/z_normalize.hpp:23-48`

The scalar `z_normalize` at `dtwc/core/z_normalize.hpp:23-48` has three loops:
1. Sum for mean (line 29-30)
2. Squared-deviation sum for stddev (line 34-37)
3. Normalize: `(x[i] - mean) * inv_stddev` (line 42-43)

- [ ] **Step 1: Write z_normalize_simd.cpp**

```cpp
/**
 * @file z_normalize_simd.cpp
 * @brief SIMD-accelerated z-normalization using Google Highway.
 *
 * @details Three embarrassingly parallel loops vectorized:
 *          1. Sum reduction for mean
 *          2. Squared-deviation sum for stddev
 *          3. Element-wise normalize: (x - mean) * inv_stddev
 *
 * @date 29 Mar 2026
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dtwc/simd/z_normalize_simd.cpp"
#include "dtwc/simd/highway_targets.hpp"

#include "hwy/highway.h"
#include <cmath>

HWY_BEFORE_NAMESPACE();
namespace dtwc::simd::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void ZNormalizeSimd(double* HWY_RESTRICT series, std::size_t n)
{
  if (n == 0) return;
  if (n == 1) { series[0] = 0.0; return; }

  const hn::ScalableTag<double> d;
  const std::size_t N = hn::Lanes(d);

  // Pass 1: compute sum for mean
  auto sum_vec = hn::Zero(d);
  std::size_t i = 0;
  for (; i + N <= n; i += N) {
    sum_vec = hn::Add(sum_vec, hn::LoadU(d, series + i));
  }
  double sum = hn::ReduceSum(d, sum_vec);
  for (; i < n; ++i) sum += series[i];

  const double mean = sum / static_cast<double>(n);

  // Pass 2: compute squared deviation sum
  const auto mean_vec = hn::Set(d, mean);
  auto sq_vec = hn::Zero(d);
  i = 0;
  for (; i + N <= n; i += N) {
    const auto diff = hn::Sub(hn::LoadU(d, series + i), mean_vec);
    sq_vec = hn::MulAdd(diff, diff, sq_vec);
  }
  double sq_sum = hn::ReduceSum(d, sq_vec);
  for (; i < n; ++i) {
    double diff = series[i] - mean;
    sq_sum += diff * diff;
  }

  const double stddev = std::sqrt(sq_sum / static_cast<double>(n));

  // Pass 3: normalize in place
  if (stddev > 1e-10) {
    const double inv_sd = 1.0 / stddev;
    const auto inv_sd_vec = hn::Set(d, inv_sd);
    i = 0;
    for (; i + N <= n; i += N) {
      const auto val = hn::LoadU(d, series + i);
      const auto normed = hn::Mul(hn::Sub(val, mean_vec), inv_sd_vec);
      hn::StoreU(normed, d, series + i);
    }
    for (; i < n; ++i) {
      series[i] = (series[i] - mean) * inv_sd;
    }
  } else {
    const auto zero = hn::Zero(d);
    i = 0;
    for (; i + N <= n; i += N) {
      hn::StoreU(zero, d, series + i);
    }
    for (; i < n; ++i) {
      series[i] = 0.0;
    }
  }
}

}  // namespace dtwc::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace dtwc::simd {

HWY_EXPORT(ZNormalizeSimd);

void z_normalize_highway(double* series, std::size_t n)
{
  HWY_DYNAMIC_DISPATCH(ZNormalizeSimd)(series, n);
}

}  // namespace dtwc::simd
#endif
```

- [ ] **Step 2: Add SIMD dispatch in z_normalize.hpp**

At the top of `dtwc/core/z_normalize.hpp`, after line 17 (`#include <vector>`), add:

```cpp
#ifdef DTWC_HAS_HIGHWAY
namespace dtwc::simd {
void z_normalize_highway(double* series, std::size_t n);
}  // namespace dtwc::simd
#endif
```

Then replace the `z_normalize` template at lines 23-48 with:

```cpp
template <typename T>
void z_normalize(T *series, size_t n)
{
#ifdef DTWC_HAS_HIGHWAY
  if constexpr (std::is_same_v<T, double>) {
    simd::z_normalize_highway(series, n);
    return;
  }
#endif
  // Scalar fallback
  if (n == 0) return;
  if (n == 1) { series[0] = static_cast<T>(0); return; }

  T sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += series[i];
  T mean = sum / static_cast<T>(n);

  T sq_sum = 0;
  for (size_t i = 0; i < n; ++i) {
    T d = series[i] - mean;
    sq_sum += d * d;
  }
  T stddev = std::sqrt(sq_sum / static_cast<T>(n));

  if (stddev > static_cast<T>(1e-10)) {
    T inv_stddev = T(1) / stddev;
    for (size_t i = 0; i < n; ++i)
      series[i] = (series[i] - mean) * inv_stddev;
  } else {
    for (size_t i = 0; i < n; ++i)
      series[i] = 0;
  }
}
```

Also add `#include <type_traits>` if not present.

- [ ] **Step 3: Build and run tests**

```bash
cd build && cmake --build . -j && ctest --output-on-failure
```

Expected: all tests pass. The `unit_test_z_normalize.cpp` validates correctness.

- [ ] **Step 4: Commit**

```bash
git add dtwc/simd/z_normalize_simd.cpp dtwc/core/z_normalize.hpp
git commit -m "feat(simd): SIMD z_normalize via Highway — vectorized sum, deviation, normalize

Three-pass vectorization: ReduceSum for mean, MulAdd for squared deviation,
Mul+Sub for normalize. Scalar fallback for non-double and Highway-disabled builds."
```

---

### Task 5: Multi-pair DTW Header (B4 — Part 1)

**Files:**
- Create: `dtwc/simd/multi_pair_dtw.hpp`

The key idea: instead of computing DTW for one (x,y) pair at a time, process 4 independent pairs simultaneously. Each AVX2 lane handles one pair's recurrence. The `min(diag, min(left, below)) + |a-b|` runs identically in all lanes with no cross-lane communication.

- [ ] **Step 1: Write multi_pair_dtw.hpp**

```cpp
/**
 * @file multi_pair_dtw.hpp
 * @brief Multi-pair DTW: process 4 independent DTW computations in SIMD lanes.
 *
 * @details DTW's inner recurrence is latency-bound (10 cycles/cell) because
 *          each cell depends on its left, below, and diagonal neighbors. SIMD
 *          within a single pair is limited. But when computing a distance matrix,
 *          we have N*(N-1)/2 independent pairs. By processing 4 pairs at once
 *          (one per AVX2 lane), we hide the recurrence latency.
 *
 *          The rolling buffer becomes 4-wide: each position holds a Vec<double,4>
 *          with one element per pair. The recurrence is identical across lanes.
 *
 * @date 29 Mar 2026
 */

#pragma once

#include <cstddef>
#include <vector>

#ifdef DTWC_HAS_HIGHWAY

namespace dtwc::simd {

/// Batch size for multi-pair DTW (matches AVX2 lane count for double).
constexpr std::size_t kDtwBatchSize = 4;

/// Result structure for a batch of DTW computations.
struct MultiPairResult {
  double distances[kDtwBatchSize];
};

/// Compute DTW for up to 4 pairs simultaneously using SIMD.
///
/// @param x_ptrs  Array of 4 pointers to first series in each pair.
/// @param y_ptrs  Array of 4 pointers to second series in each pair.
/// @param x_lens  Array of 4 lengths for first series.
/// @param y_lens  Array of 4 lengths for second series.
/// @param n_pairs Number of valid pairs (1-4). Unused lanes are ignored.
/// @return MultiPairResult with distances for each valid pair.
///
/// @note All series in a batch should have similar lengths for best efficiency.
///       Pairs with different lengths are handled correctly but may waste SIMD
///       lanes on padding iterations.
MultiPairResult dtw_multi_pair(
    const double* const x_ptrs[],
    const double* const y_ptrs[],
    const std::size_t x_lens[],
    const std::size_t y_lens[],
    std::size_t n_pairs);

}  // namespace dtwc::simd

#endif  // DTWC_HAS_HIGHWAY
```

- [ ] **Step 2: Commit**

```bash
git add dtwc/simd/multi_pair_dtw.hpp
git commit -m "feat(simd): add multi-pair DTW header — 4-wide batch interface"
```

---

### Task 6: Multi-pair DTW Implementation (B4 — Part 2)

**Files:**
- Create: `dtwc/simd/multi_pair_dtw.cpp`

- [ ] **Step 1: Write multi_pair_dtw.cpp**

```cpp
/**
 * @file multi_pair_dtw.cpp
 * @brief Multi-pair DTW implementation using Google Highway SIMD.
 *
 * @details Implements the 4-wide rolling buffer DTW. Each element in the buffer
 *          is a Highway vector holding one value per pair. The recurrence:
 *            C[i] = min(diag, min(C[i-1], C[i])) + |short[i] - long[j]|
 *          runs identically in all lanes.
 *
 *          For pairs with different lengths, we use the same strategy as scalar
 *          dtwFull_L: orient short/long sides per-pair, then pad the shorter
 *          pairs' contributions. Since all pairs run in lockstep, we process
 *          max(m_long) outer iterations. Pairs that finish early get their final
 *          result preserved via masking.
 *
 * @date 29 Mar 2026
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dtwc/simd/multi_pair_dtw.cpp"
#include "dtwc/simd/highway_targets.hpp"

#include "hwy/highway.h"
#include "dtwc/simd/multi_pair_dtw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace dtwc::simd::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Use the fixed-size tag for 4 doubles (256-bit = AVX2 native width).
// On AVX-512 this still works correctly (Highway pads/masks as needed).
// On SSE4 (2 doubles), Highway emulates 4-wide via two 2-wide ops.
using D4 = hn::FixedTag<double, 4>;

MultiPairResult DtwMultiPairImpl(
    const double* const x_ptrs[],
    const double* const y_ptrs[],
    const std::size_t x_lens[],
    const std::size_t y_lens[],
    std::size_t n_pairs)
{
  constexpr double kInf = std::numeric_limits<double>::max();
  const D4 d;
  MultiPairResult result;

  // Handle degenerate cases
  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    result.distances[p] = kInf;
  }
  if (n_pairs == 0) return result;

  // For each pair, determine short/long sides (dtwFull_L convention).
  // Store as flat arrays indexed by pair.
  const double* short_ptrs[kDtwBatchSize];
  const double* long_ptrs[kDtwBatchSize];
  std::size_t m_shorts[kDtwBatchSize];
  std::size_t m_longs[kDtwBatchSize];

  std::size_t max_short = 0;
  std::size_t max_long = 0;

  for (std::size_t p = 0; p < n_pairs; ++p) {
    if (x_lens[p] == 0 || y_lens[p] == 0) {
      // Empty series: distance = inf, skip processing
      short_ptrs[p] = nullptr;
      long_ptrs[p] = nullptr;
      m_shorts[p] = 0;
      m_longs[p] = 0;
      continue;
    }
    if (x_lens[p] <= y_lens[p]) {
      short_ptrs[p] = x_ptrs[p];
      long_ptrs[p] = y_ptrs[p];
      m_shorts[p] = x_lens[p];
      m_longs[p] = y_lens[p];
    } else {
      short_ptrs[p] = y_ptrs[p];
      long_ptrs[p] = x_ptrs[p];
      m_shorts[p] = y_lens[p];
      m_longs[p] = x_lens[p];
    }
    max_short = std::max(max_short, m_shorts[p]);
    max_long = std::max(max_long, m_longs[p]);
  }
  // Pad unused lanes
  for (std::size_t p = n_pairs; p < kDtwBatchSize; ++p) {
    short_ptrs[p] = short_ptrs[0];  // dummy, won't affect result
    long_ptrs[p] = long_ptrs[0];
    m_shorts[p] = m_shorts[0];
    m_longs[p] = m_longs[0];
  }

  if (max_short == 0 || max_long == 0) return result;

  // Rolling buffer: short_side[i] is a 4-wide vector (one per pair).
  // Thread-local to avoid per-call allocation.
  thread_local std::vector<double> buf;
  buf.resize(max_short * kDtwBatchSize);

  // Helper: access buffer[i][pair] in interleaved layout.
  // Interleaved: [pair0_i0, pair1_i0, pair2_i0, pair3_i0, pair0_i1, ...]
  auto buf_ptr = [&](std::size_t i) -> double* {
    return buf.data() + i * kDtwBatchSize;
  };

  const auto inf_vec = hn::Set(d, kInf);

  // Initialize: short_side[0] = |short[0] - long[0]| per pair
  {
    HWY_ALIGN double s0[kDtwBatchSize], l0[kDtwBatchSize];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
      s0[p] = (m_shorts[p] > 0) ? short_ptrs[p][0] : 0.0;
      l0[p] = (m_longs[p] > 0) ? long_ptrs[p][0] : 0.0;
    }
    const auto sv = hn::Load(d, s0);
    const auto lv = hn::Load(d, l0);
    const auto dist = hn::Abs(hn::Sub(sv, lv));
    hn::Store(dist, d, buf_ptr(0));
  }

  // Initialize: short_side[i] = short_side[i-1] + |short[i] - long[0]|
  for (std::size_t i = 1; i < max_short; ++i) {
    HWY_ALIGN double si[kDtwBatchSize], l0[kDtwBatchSize];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
      si[p] = (i < m_shorts[p]) ? short_ptrs[p][i] : 0.0;
      l0[p] = (m_longs[p] > 0) ? long_ptrs[p][0] : 0.0;
    }
    const auto prev = hn::Load(d, buf_ptr(i - 1));
    const auto dist = hn::Abs(hn::Sub(hn::Load(d, si), hn::Load(d, l0)));

    // For pairs where i >= m_shorts[p], set to inf (out of bounds)
    HWY_ALIGN double mask_vals[kDtwBatchSize];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
      mask_vals[p] = (i < m_shorts[p]) ? 0.0 : kInf;
    }
    const auto oob = hn::Load(d, mask_vals);
    const auto val = hn::Add(prev, dist);
    // If out of bounds, use inf; otherwise use computed value
    const auto is_valid = hn::Eq(oob, hn::Zero(d));
    hn::Store(hn::IfThenElse(is_valid, val, oob), d, buf_ptr(i));
  }

  // Main loop: for j = 1..max_long-1
  for (std::size_t j = 1; j < max_long; ++j) {
    // Load long[j] per pair
    HWY_ALIGN double lj[kDtwBatchSize];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
      lj[p] = (j < m_longs[p]) ? long_ptrs[p][j] : 0.0;
    }
    const auto long_j = hn::Load(d, lj);

    // For pairs where j >= m_longs[p], this column is out of bounds.
    HWY_ALIGN double j_valid_vals[kDtwBatchSize];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
      j_valid_vals[p] = (j < m_longs[p]) ? 1.0 : 0.0;
    }
    const auto j_valid = hn::Ne(hn::Load(d, j_valid_vals), hn::Zero(d));

    // diag = short_side[0] (before update)
    auto diag = hn::Load(d, buf_ptr(0));

    // short_side[0] += |short[0] - long[j]|
    {
      HWY_ALIGN double s0[kDtwBatchSize];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
        s0[p] = (m_shorts[p] > 0) ? short_ptrs[p][0] : 0.0;
      }
      const auto dist = hn::Abs(hn::Sub(hn::Load(d, s0), long_j));
      auto updated = hn::Add(hn::Load(d, buf_ptr(0)), dist);
      // If j out of bounds for this pair, keep inf
      updated = hn::IfThenElse(j_valid, updated, inf_vec);
      hn::Store(updated, d, buf_ptr(0));
    }

    // Inner loop: i = 1..max_short-1
    for (std::size_t i = 1; i < max_short; ++i) {
      // Validity: both i < m_shorts[p] and j < m_longs[p]
      HWY_ALIGN double valid_vals[kDtwBatchSize];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
        valid_vals[p] = (i < m_shorts[p] && j < m_longs[p]) ? 1.0 : 0.0;
      }
      const auto valid = hn::Ne(hn::Load(d, valid_vals), hn::Zero(d));

      const auto left = hn::Load(d, buf_ptr(i - 1));  // short_side[i-1] (already updated)
      const auto below = hn::Load(d, buf_ptr(i));      // short_side[i] (not yet updated)

      // |short[i] - long[j]| per pair
      HWY_ALIGN double si[kDtwBatchSize];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
        si[p] = (i < m_shorts[p]) ? short_ptrs[p][i] : 0.0;
      }
      const auto dist = hn::Abs(hn::Sub(hn::Load(d, si), long_j));

      // Recurrence: min(diag, min(left, below)) + dist
      const auto min_lb = hn::Min(left, below);
      const auto minimum = hn::Min(diag, min_lb);
      const auto next = hn::Add(minimum, dist);

      diag = below;  // Save before overwrite

      // If invalid (out of bounds), keep old value
      hn::Store(hn::IfThenElse(valid, next, below), d, buf_ptr(i));
    }
  }

  // Extract results: short_side[m_shorts[p]-1] for each pair
  for (std::size_t p = 0; p < n_pairs; ++p) {
    if (m_shorts[p] == 0 || m_longs[p] == 0) {
      result.distances[p] = kInf;
    } else {
      result.distances[p] = buf.data()[(m_shorts[p] - 1) * kDtwBatchSize + p];
    }
  }

  return result;
}

}  // namespace dtwc::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace dtwc::simd {

HWY_EXPORT(DtwMultiPairImpl);

MultiPairResult dtw_multi_pair(
    const double* const x_ptrs[],
    const double* const y_ptrs[],
    const std::size_t x_lens[],
    const std::size_t y_lens[],
    std::size_t n_pairs)
{
  return HWY_DYNAMIC_DISPATCH(DtwMultiPairImpl)(x_ptrs, y_ptrs, x_lens, y_lens, n_pairs);
}

}  // namespace dtwc::simd
#endif
```

- [ ] **Step 2: Build and verify compilation**

```bash
cd build && cmake --build . -j
```

Expected: compiles. No tests call multi-pair yet — that comes in Task 8.

- [ ] **Step 3: Commit**

```bash
git add dtwc/simd/multi_pair_dtw.cpp
git commit -m "feat(simd): implement multi-pair DTW — 4 pairs in AVX2 lanes

Rolling buffer is 4-wide interleaved. Each lane runs one pair's DTW
recurrence in lockstep. Handles variable-length pairs via per-lane masking."
```

---

### Task 7: SIMD Tests

**Files:**
- Create: `tests/unit/unit_test_simd.cpp`

- [ ] **Step 1: Write comprehensive SIMD tests**

```cpp
/**
 * @file unit_test_simd.cpp
 * @brief Tests that SIMD implementations produce identical results to scalar.
 *
 * @details Validates that SIMD LB_Keogh, z_normalize, and multi-pair DTW
 *          match their scalar counterparts bit-for-bit (or within epsilon
 *          for floating point accumulation differences).
 *
 * @date 29 Mar 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <dtwc/core/lower_bound_impl.hpp>
#include <dtwc/core/z_normalize.hpp>
#include <dtwc/warping.hpp>

#include <vector>
#include <random>
#include <cmath>

#ifdef DTWC_HAS_HIGHWAY
#include <dtwc/simd/multi_pair_dtw.hpp>
#endif

using Catch::Approx;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
static std::vector<double> make_random(std::size_t n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  std::vector<double> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

/// Scalar LB_Keogh (bypass SIMD dispatch for comparison)
static double lb_keogh_scalar(const double* q, std::size_t n,
                              const double* u, const double* l)
{
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double eu = q[i] - u[i];
    double el = l[i] - q[i];
    double e = std::max(0.0, std::max(eu, el));
    sum += e;
  }
  return sum;
}

/// Scalar z_normalize (bypass SIMD dispatch for comparison)
static void z_normalize_scalar(double* series, std::size_t n)
{
  if (n == 0) return;
  if (n == 1) { series[0] = 0.0; return; }
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) sum += series[i];
  double mean = sum / static_cast<double>(n);
  double sq = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double d = series[i] - mean;
    sq += d * d;
  }
  double sd = std::sqrt(sq / static_cast<double>(n));
  if (sd > 1e-10) {
    double inv = 1.0 / sd;
    for (std::size_t i = 0; i < n; ++i)
      series[i] = (series[i] - mean) * inv;
  } else {
    for (std::size_t i = 0; i < n; ++i) series[i] = 0;
  }
}

// -----------------------------------------------------------------------
// LB_Keogh SIMD Tests
// -----------------------------------------------------------------------
TEST_CASE("LB_Keogh SIMD matches scalar for various lengths", "[simd][lb_keogh]")
{
  for (std::size_t n : {1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 500, 1000, 4000}) {
    auto query = make_random(n, 42 + n);
    auto candidate = make_random(n, 84 + n);

    std::vector<double> upper(n), lower(n);
    dtwc::core::compute_envelopes(candidate.data(), n, 10, upper.data(), lower.data());

    double scalar = lb_keogh_scalar(query.data(), n, upper.data(), lower.data());
    double result = dtwc::core::lb_keogh(query.data(), n, upper.data(), lower.data());

    INFO("n=" << n);
    REQUIRE(result == Approx(scalar).epsilon(1e-12));
  }
}

TEST_CASE("LB_Keogh SIMD: identity (query == candidate) gives 0", "[simd][lb_keogh]")
{
  auto series = make_random(500, 99);
  std::vector<double> upper(500), lower(500);
  dtwc::core::compute_envelopes(series.data(), 500, 10, upper.data(), lower.data());

  double result = dtwc::core::lb_keogh(series.data(), 500, upper.data(), lower.data());
  REQUIRE(result == Approx(0.0).margin(1e-12));
}

TEST_CASE("LB_Keogh SIMD: non-negative", "[simd][lb_keogh]")
{
  for (unsigned seed = 0; seed < 50; ++seed) {
    auto q = make_random(200, seed * 2);
    auto c = make_random(200, seed * 2 + 1);
    std::vector<double> u(200), l(200);
    dtwc::core::compute_envelopes(c.data(), 200, 20, u.data(), l.data());

    double result = dtwc::core::lb_keogh(q.data(), 200, u.data(), l.data());
    REQUIRE(result >= 0.0);
  }
}

// -----------------------------------------------------------------------
// z_normalize SIMD Tests
// -----------------------------------------------------------------------
TEST_CASE("z_normalize SIMD matches scalar for various lengths", "[simd][z_normalize]")
{
  for (std::size_t n : {2, 3, 4, 7, 8, 15, 16, 31, 32, 100, 500, 1000, 4000}) {
    auto data_simd = make_random(n, 100 + n);
    auto data_scalar = data_simd;  // copy

    dtwc::core::z_normalize(data_simd.data(), n);
    z_normalize_scalar(data_scalar.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
      INFO("n=" << n << " i=" << i);
      REQUIRE(data_simd[i] == Approx(data_scalar[i]).epsilon(1e-10));
    }
  }
}

TEST_CASE("z_normalize SIMD: result has zero mean and unit stddev", "[simd][z_normalize]")
{
  auto data = make_random(1000, 77);
  dtwc::core::z_normalize(data.data(), data.size());

  double sum = 0;
  for (auto v : data) sum += v;
  double mean = sum / 1000.0;
  REQUIRE(mean == Approx(0.0).margin(1e-10));

  double sq = 0;
  for (auto v : data) sq += (v - mean) * (v - mean);
  double sd = std::sqrt(sq / 1000.0);
  REQUIRE(sd == Approx(1.0).epsilon(1e-10));
}

TEST_CASE("z_normalize SIMD: constant series produces zeros", "[simd][z_normalize]")
{
  std::vector<double> data(100, 42.0);
  dtwc::core::z_normalize(data.data(), data.size());
  for (auto v : data) {
    REQUIRE(v == 0.0);
  }
}

TEST_CASE("z_normalize SIMD: single element", "[simd][z_normalize]")
{
  double val = 3.14;
  dtwc::core::z_normalize(&val, 1);
  REQUIRE(val == 0.0);
}

// -----------------------------------------------------------------------
// Multi-pair DTW Tests
// -----------------------------------------------------------------------
#ifdef DTWC_HAS_HIGHWAY

TEST_CASE("Multi-pair DTW matches scalar dtwFull_L", "[simd][multi_pair_dtw]")
{
  // Generate 8 random series, compute 4 pairs
  std::vector<std::vector<double>> series;
  for (unsigned i = 0; i < 8; ++i) {
    series.push_back(make_random(200 + i * 10, 200 + i));
  }

  const double* x_ptrs[4] = {series[0].data(), series[2].data(), series[4].data(), series[6].data()};
  const double* y_ptrs[4] = {series[1].data(), series[3].data(), series[5].data(), series[7].data()};
  std::size_t x_lens[4] = {series[0].size(), series[2].size(), series[4].size(), series[6].size()};
  std::size_t y_lens[4] = {series[1].size(), series[3].size(), series[5].size(), series[7].size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 4);

  for (std::size_t p = 0; p < 4; ++p) {
    std::vector<double> x(x_ptrs[p], x_ptrs[p] + x_lens[p]);
    std::vector<double> y(y_ptrs[p], y_ptrs[p] + y_lens[p]);
    double scalar = dtwc::dtwFull_L<double>(x, y);

    INFO("pair=" << p << " scalar=" << scalar << " simd=" << result.distances[p]);
    REQUIRE(result.distances[p] == Approx(scalar).epsilon(1e-10));
  }
}

TEST_CASE("Multi-pair DTW: fewer than 4 pairs", "[simd][multi_pair_dtw]")
{
  auto s1 = make_random(100, 1);
  auto s2 = make_random(100, 2);

  const double* x_ptrs[4] = {s1.data(), nullptr, nullptr, nullptr};
  const double* y_ptrs[4] = {s2.data(), nullptr, nullptr, nullptr};
  std::size_t x_lens[4] = {s1.size(), 0, 0, 0};
  std::size_t y_lens[4] = {s2.size(), 0, 0, 0};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 1);

  double scalar = dtwc::dtwFull_L<double>(s1, s2);
  REQUIRE(result.distances[0] == Approx(scalar).epsilon(1e-10));
}

TEST_CASE("Multi-pair DTW: equal series give distance 0", "[simd][multi_pair_dtw]")
{
  auto s = make_random(150, 42);

  const double* x_ptrs[4] = {s.data(), s.data(), s.data(), s.data()};
  const double* y_ptrs[4] = {s.data(), s.data(), s.data(), s.data()};
  std::size_t lens[4] = {s.size(), s.size(), s.size(), s.size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, lens, lens, 4);

  for (std::size_t p = 0; p < 4; ++p) {
    REQUIRE(result.distances[p] == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("Multi-pair DTW: different length pairs", "[simd][multi_pair_dtw]")
{
  auto s1 = make_random(100, 10);
  auto s2 = make_random(150, 20);
  auto s3 = make_random(80, 30);
  auto s4 = make_random(200, 40);

  const double* x_ptrs[4] = {s1.data(), s1.data(), s3.data(), s3.data()};
  const double* y_ptrs[4] = {s2.data(), s4.data(), s2.data(), s4.data()};
  std::size_t x_lens[4] = {s1.size(), s1.size(), s3.size(), s3.size()};
  std::size_t y_lens[4] = {s2.size(), s4.size(), s2.size(), s4.size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 4);

  for (std::size_t p = 0; p < 4; ++p) {
    std::vector<double> x(x_ptrs[p], x_ptrs[p] + x_lens[p]);
    std::vector<double> y(y_ptrs[p], y_ptrs[p] + y_lens[p]);
    double scalar = dtwc::dtwFull_L<double>(x, y);

    INFO("pair=" << p);
    REQUIRE(result.distances[p] == Approx(scalar).epsilon(1e-10));
  }
}

#endif  // DTWC_HAS_HIGHWAY
```

- [ ] **Step 2: Build and run tests**

```bash
cd build && cmake --build . -j && ctest --output-on-failure
```

Expected: all new SIMD tests pass alongside existing 34 tests.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/unit_test_simd.cpp
git commit -m "test(simd): add comprehensive SIMD tests — LB_Keogh, z_normalize, multi-pair DTW

Validates SIMD results match scalar within epsilon. Tests various lengths
(including non-power-of-2 for tail handling), edge cases, and properties."
```

---

### Task 8: Wire Multi-pair DTW into fillDistanceMatrix

**Files:**
- Modify: `dtwc/Problem.cpp:185-211`

The current `fillDistanceMatrix` at line 185-211 dispatches individual pairs via `run(oneTask, N*(N+1)/2)`. With multi-pair DTW, we batch 4 pairs and call `dtw_multi_pair` instead.

- [ ] **Step 1: Add the SIMD include and batched fillDistanceMatrix**

At the top of `dtwc/Problem.cpp`, add:

```cpp
#ifdef DTWC_HAS_HIGHWAY
#include "simd/multi_pair_dtw.hpp"
#endif
```

Then replace `fillDistanceMatrix()` (lines 185-211) with:

```cpp
void Problem::fillDistanceMatrix()
{
  if (isDistanceMatrixFilled()) return;

  const size_t N = data.size();

  std::cout << "Distance matrix is being filled!" << std::endl;

#ifdef DTWC_HAS_HIGHWAY
  // Multi-pair SIMD path: batch 4 pairs per SIMD call.
  // The triangular iteration produces (i,j) pairs with i <= j.
  // We collect batches of 4 and dispatch them to multi-pair DTW.
  const size_t total_pairs = N * (N + 1) / 2;

  // Pre-generate all (i,j) pairs for batching.
  // This is O(N^2) in memory but trivial compared to the DTW computation.
  struct Pair { size_t i, j; };
  std::vector<Pair> pairs;
  pairs.reserve(total_pairs);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      pairs.push_back({i, j});

  // Process pairs in batches of 4, parallelized over batches.
  const size_t n_batches = (pairs.size() + 3) / 4;
  auto batchTask = [&](size_t batch_idx) {
    const size_t start = batch_idx * 4;
    const size_t end = std::min(start + 4, pairs.size());
    const size_t n_in_batch = end - start;

    const double* x_ptrs[4] = {};
    const double* y_ptrs[4] = {};
    std::size_t x_lens[4] = {};
    std::size_t y_lens[4] = {};

    for (size_t k = 0; k < n_in_batch; ++k) {
      const auto& [pi, pj] = pairs[start + k];
      x_ptrs[k] = data.get(pi).data();
      y_ptrs[k] = data.get(pj).data();
      x_lens[k] = data.get(pi).size();
      y_lens[k] = data.get(pj).size();
    }

    // Self-distance pairs (i==j) are trivially 0, but multi_pair handles them.
    auto result = simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, n_in_batch);

    for (size_t k = 0; k < n_in_batch; ++k) {
      const auto& [pi, pj] = pairs[start + k];
      if (pi == pj) {
        distanceMatrix(static_cast<int>(pi), static_cast<int>(pj)) = 0.0;
      } else {
        distanceMatrix(static_cast<int>(pi), static_cast<int>(pj)) = result.distances[k];
      }
    }
  };
  run(batchTask, n_batches);

#else
  // Scalar fallback: one pair at a time (original implementation).
  auto oneTask = [this, N](size_t k) {
    const double Nd = static_cast<double>(N);
    const double kd = static_cast<double>(k);
    size_t i = static_cast<size_t>(std::floor(Nd + 0.5 - std::sqrt((Nd + 0.5) * (Nd + 0.5) - 2.0 * kd)));
    size_t row_start = i * N - i * (i - 1) / 2;
    if (row_start + (N - i) <= k) {
      row_start += (N - i);
      ++i;
    }
    size_t j = i + (k - row_start);
    distByInd(static_cast<int>(i), static_cast<int>(j));
  };
  run(oneTask, N * (N + 1) / 2);
#endif

  is_distMat_filled = true;
  std::cout << "Distance matrix has been filled!" << std::endl;
}
```

- [ ] **Step 2: Build and run all tests**

```bash
cd build && cmake --build . -j && ctest --output-on-failure
```

Expected: all tests pass (unit tests, adversarial tests, SIMD tests). The clustering and distance matrix tests validate end-to-end correctness.

- [ ] **Step 3: Commit**

```bash
git add dtwc/Problem.cpp
git commit -m "feat(simd): wire multi-pair DTW into fillDistanceMatrix

Batches pairs into groups of 4 for SIMD dispatch. OpenMP parallelizes
over batches. Scalar fallback preserved when Highway is disabled."
```

---

### Task 9: SIMD Benchmarks

**Files:**
- Modify: `benchmarks/bench_dtw_baseline.cpp`
- Modify: `benchmarks/CMakeLists.txt`

- [ ] **Step 1: Add SIMD benchmarks to bench_dtw_baseline.cpp**

At the end of `benchmarks/bench_dtw_baseline.cpp` (after line 165), add:

```cpp
// ---------------------------------------------------------------------------
// BM_lb_keogh — LB_Keogh lower bound for varying lengths
// ---------------------------------------------------------------------------
static void BM_lb_keogh(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto query = random_series(len, 42);
  auto candidate = random_series(len, 43);

  std::vector<double> upper(len), lower(len);
  dtwc::core::compute_envelopes(candidate.data(), len, 10, upper.data(), lower.data());

  for (auto _ : state) {
    benchmark::DoNotOptimize(
      dtwc::core::lb_keogh(query.data(), len, upper.data(), lower.data()));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_lb_keogh)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Arg(8000)
  ->Unit(benchmark::kNanosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_z_normalize — z-normalization for varying lengths
// ---------------------------------------------------------------------------
static void BM_z_normalize(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto data = random_series(len, 42);

  for (auto _ : state) {
    state.PauseTiming();
    auto copy = data; // fresh copy each iteration
    state.ResumeTiming();
    dtwc::core::z_normalize(copy.data(), copy.size());
    benchmark::DoNotOptimize(copy.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_z_normalize)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Arg(8000)
  ->Unit(benchmark::kNanosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_compute_envelopes — envelope computation for varying lengths
// ---------------------------------------------------------------------------
static void BM_compute_envelopes(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  auto series = random_series(len, 42);
  std::vector<double> upper(len), lower(len);

  for (auto _ : state) {
    dtwc::core::compute_envelopes(series.data(), len, band, upper.data(), lower.data());
    benchmark::DoNotOptimize(upper.data());
    benchmark::DoNotOptimize(lower.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_compute_envelopes)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Unit(benchmark::kNanosecond);
```

Also add the include at the top of the file (after line 13):

```cpp
#include <dtwc/core/lower_bound_impl.hpp>
#include <dtwc/core/z_normalize.hpp>
```

- [ ] **Step 2: Update benchmarks/CMakeLists.txt to link Highway if available**

After the existing `target_link_libraries(bench_dtw_baseline ...)` block, add:

```cmake
# Link Highway for SIMD benchmarks if available
if(DTWC_ENABLE_SIMD AND TARGET hwy::hwy)
  target_link_libraries(bench_dtw_baseline PRIVATE hwy::hwy)
endif()
```

- [ ] **Step 3: Build benchmarks**

```bash
cd build && cmake .. -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_SIMD=ON && cmake --build . --target bench_dtw_baseline -j
```

Expected: builds successfully.

- [ ] **Step 4: Run baseline benchmarks (before SIMD) and save results**

```bash
cd build && ./benchmarks/bench_dtw_baseline --benchmark_format=json > ../benchmarks/results/bench_simd.json 2>/dev/null
```

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bench_dtw_baseline.cpp benchmarks/CMakeLists.txt
git commit -m "bench(simd): add LB_Keogh, z_normalize, envelope benchmarks

New benchmarks: BM_lb_keogh (100-8000), BM_z_normalize (100-8000),
BM_compute_envelopes (1000-4000). Captures SIMD vs scalar performance."
```

---

### Task 10: CHANGELOG and Final Verification

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `.claude/CITATIONS.md`

- [ ] **Step 1: Update CHANGELOG.md**

In the `## [Unreleased]` section, add under `### Added`:

```markdown
- **SIMD acceleration via Google Highway** — runtime dispatch (AVX-512/AVX2/SSE4/NEON)
  - SIMD LB_Keogh: vectorized reduction loop (4-8x speedup)
  - SIMD z_normalize: vectorized sum, deviation, normalize passes (4-8x speedup)
  - Multi-pair DTW: 4 independent pairs in AVX2 lanes (3-4x fillDistanceMatrix speedup)
  - `DTWC_ENABLE_SIMD` CMake option (default ON, graceful fallback when OFF)
  - Google Highway 1.2.0 fetched via CPM
```

- [ ] **Step 2: Update CITATIONS.md**

Add Highway citation:

```markdown
## Google Highway
- Bobrowski et al., "Highway: C++ library for SIMD", Google, 2019-present.
  https://github.com/google/highway
  Apache-2.0 license. Used for runtime-dispatched SIMD acceleration.
```

- [ ] **Step 3: Full test suite verification**

```bash
cd build && cmake .. -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_SIMD=ON && cmake --build . -j && ctest --output-on-failure
```

Expected: all tests pass.

- [ ] **Step 4: Verify SIMD-disabled build**

```bash
cd build && cmake .. -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_SIMD=OFF && cmake --build . -j && ctest --output-on-failure
```

Expected: all tests pass (scalar fallbacks used).

- [ ] **Step 5: Run benchmarks and compare**

```bash
cd build && cmake .. -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_SIMD=ON && cmake --build . --target bench_dtw_baseline -j
./benchmarks/bench_dtw_baseline --benchmark_format=console
```

Record the before/after numbers for the performance report.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md .claude/CITATIONS.md
git commit -m "docs: update CHANGELOG and citations for SIMD Highway integration"
```

---

## Verification Checklist

After all tasks, verify:
- [ ] `cmake -DDTWC_ENABLE_SIMD=ON` builds and all tests pass
- [ ] `cmake -DDTWC_ENABLE_SIMD=OFF` builds and all tests pass (scalar fallback)
- [ ] SIMD LB_Keogh matches scalar within 1e-12
- [ ] SIMD z_normalize matches scalar within 1e-10
- [ ] Multi-pair DTW matches scalar dtwFull_L within 1e-10
- [ ] fillDistanceMatrix produces identical distance matrices with/without SIMD
- [ ] Benchmarks show measurable speedup for LB_Keogh and fillDistanceMatrix
- [ ] No new warnings from compiler
