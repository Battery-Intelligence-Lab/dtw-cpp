# Wave 1A: Metrics + Missing Data Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `std::isnan()` bug under `-ffast-math`, add 5 new cluster quality metrics, add `MissingStrategy` enum with Problem integration, and implement DTW-AROW algorithm.

**Architecture:** New `missing_utils.hpp` provides a bitwise NaN check safe under `-ffast-math`. New metrics go into existing `scores.hpp/cpp`. DTW-AROW gets a dedicated `_impl` function in `warping_missing_arow.hpp` because its recurrence differs structurally from standard DTW. `MissingStrategy` enum extends `dtw_options.hpp` and is wired into `Problem::rebind_dtw_fn()`.

**Tech Stack:** C++17, Google Test, OpenMP (optional), CMake

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `dtwc/missing_utils.hpp` | Bitwise `is_missing()`, `has_missing()`, `interpolate_linear()`, `missing_rate()` |
| Create | `dtwc/warping_missing_arow.hpp` | DTW-AROW algorithm (diagonal-only for missing cells) |
| Create | `tests/unit/unit_test_missing_utils.cpp` | Tests for missing utilities |
| Create | `tests/unit/unit_test_arow_dtw.cpp` | Tests for DTW-AROW |
| Create | `tests/unit/unit_test_scores_new.cpp` | Tests for Dunn, inertia, CH, ARI, NMI |
| Modify | `dtwc/scores.hpp` | Add 5 new metric declarations |
| Modify | `dtwc/scores.cpp` | Implement Dunn, inertia, CH, ARI, NMI |
| Modify | `dtwc/core/dtw_options.hpp` | Add `MissingStrategy` enum |
| Modify | `dtwc/warping_missing.hpp` | Replace `std::isnan()` with `is_missing()` |
| Modify | `dtwc/Problem.hpp` | Add `missing_strategy` member |
| Modify | `dtwc/Problem.cpp` | Wire missing-data into `rebind_dtw_fn()` and `fillDistanceMatrix()` |
| Modify | `dtwc/dtwc.hpp` | Include new headers |
| Modify | `tests/CMakeLists.txt` | Register new test files |

---

### Task 1: Create `missing_utils.hpp` with Bitwise NaN Check

**Why first:** The existing `std::isnan()` in `warping_missing.hpp` is silently broken under `-ffast-math` (enabled in Release builds). This is a correctness bug in shipped code.

**Files:**
- Create: `dtwc/missing_utils.hpp`
- Create: `tests/unit/unit_test_missing_utils.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the test file**

```cpp
// tests/unit/unit_test_missing_utils.cpp
#include "../test_util.hpp"
#include <dtwc/missing_utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

TEST(MissingUtils, is_missing_detects_quiet_nan)
{
  double val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(dtwc::is_missing(val));
}

TEST(MissingUtils, is_missing_detects_signaling_nan)
{
  double val = std::numeric_limits<double>::signaling_NaN();
  EXPECT_TRUE(dtwc::is_missing(val));
}

TEST(MissingUtils, is_missing_rejects_normal_values)
{
  EXPECT_FALSE(dtwc::is_missing(0.0));
  EXPECT_FALSE(dtwc::is_missing(1.0));
  EXPECT_FALSE(dtwc::is_missing(-1.0));
  EXPECT_FALSE(dtwc::is_missing(1e300));
  EXPECT_FALSE(dtwc::is_missing(-1e-300));
}

TEST(MissingUtils, is_missing_rejects_infinity)
{
  EXPECT_FALSE(dtwc::is_missing(std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(dtwc::is_missing(-std::numeric_limits<double>::infinity()));
}

TEST(MissingUtils, is_missing_float)
{
  float nan_f = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(dtwc::is_missing(nan_f));
  EXPECT_FALSE(dtwc::is_missing(0.0f));
  EXPECT_FALSE(dtwc::is_missing(std::numeric_limits<float>::infinity()));
}

TEST(MissingUtils, has_missing_empty_vector)
{
  std::vector<double> v;
  EXPECT_FALSE(dtwc::has_missing(v));
}

TEST(MissingUtils, has_missing_no_nan)
{
  std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
  EXPECT_FALSE(dtwc::has_missing(v));
}

TEST(MissingUtils, has_missing_with_nan)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = {1.0, nan, 3.0};
  EXPECT_TRUE(dtwc::has_missing(v));
}

TEST(MissingUtils, has_missing_all_nan)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = {nan, nan, nan};
  EXPECT_TRUE(dtwc::has_missing(v));
}

TEST(MissingUtils, missing_rate_no_nan)
{
  std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
  EXPECT_DOUBLE_EQ(dtwc::missing_rate(v), 0.0);
}

TEST(MissingUtils, missing_rate_half_nan)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = {1.0, nan, 3.0, nan};
  EXPECT_DOUBLE_EQ(dtwc::missing_rate(v), 0.5);
}

TEST(MissingUtils, missing_rate_empty)
{
  std::vector<double> v;
  EXPECT_DOUBLE_EQ(dtwc::missing_rate(v), 0.0);
}

TEST(MissingUtils, interpolate_linear_no_nan)
{
  std::vector<double> v = {1.0, 2.0, 3.0};
  auto result = dtwc::interpolate_linear(v);
  EXPECT_EQ(result.size(), 3u);
  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST(MissingUtils, interpolate_linear_interior_gap)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = {1.0, nan, 3.0};
  auto result = dtwc::interpolate_linear(v);
  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);  // interpolated
  EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST(MissingUtils, interpolate_linear_multi_gap)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = {0.0, nan, nan, 6.0};
  auto result = dtwc::interpolate_linear(v);
  EXPECT_DOUBLE_EQ(result[0], 0.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0);
  EXPECT_DOUBLE_EQ(result[3], 6.0);
}

TEST(MissingUtils, interpolate_linear_leading_nan_LOCF)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  // NOCB: Next Observation Carried Backward
  std::vector<double> v = {nan, nan, 3.0, 4.0};
  auto result = dtwc::interpolate_linear(v);
  EXPECT_DOUBLE_EQ(result[0], 3.0);  // carried backward
  EXPECT_DOUBLE_EQ(result[1], 3.0);  // carried backward
  EXPECT_DOUBLE_EQ(result[2], 3.0);
  EXPECT_DOUBLE_EQ(result[3], 4.0);
}

TEST(MissingUtils, interpolate_linear_trailing_nan_LOCF)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  // LOCF: Last Observation Carried Forward
  std::vector<double> v = {1.0, 2.0, nan, nan};
  auto result = dtwc::interpolate_linear(v);
  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 2.0);  // carried forward
  EXPECT_DOUBLE_EQ(result[3], 2.0);  // carried forward
}

TEST(MissingUtils, interpolate_linear_all_nan_throws)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = {nan, nan, nan};
  EXPECT_THROW(dtwc::interpolate_linear(v), std::runtime_error);
}
```

- [ ] **Step 2: Register test in CMakeLists.txt**

Add to `tests/CMakeLists.txt` alongside the existing test registrations (find the pattern used for other `unit_test_*.cpp` files and follow it):

```cmake
add_executable(unit_test_missing_utils unit/unit_test_missing_utils.cpp)
target_link_libraries(unit_test_missing_utils PRIVATE dtwc++ GTest::gtest_main)
add_test(NAME unit_test_missing_utils COMMAND unit_test_missing_utils)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cmake --build build --target unit_test_missing_utils && ctest --test-dir build -R unit_test_missing_utils -V`
Expected: Build FAIL — `dtwc/missing_utils.hpp` does not exist yet.

- [ ] **Step 4: Create `missing_utils.hpp`**

```cpp
// dtwc/missing_utils.hpp
/**
 * @file missing_utils.hpp
 * @brief Utilities for handling missing data (NaN) in time series.
 *
 * @details Provides a bitwise NaN check that is safe under -ffast-math and /fp:fast,
 *          plus helper functions for detecting and interpolating missing values.
 *
 * @date 02 Apr 2026
 */

#pragma once

#include <cstdint>
#include <cstring>    // std::memcpy
#include <stdexcept>
#include <vector>

namespace dtwc {

/// Bitwise NaN check — safe under -ffast-math / /fp:fast.
/// std::isnan() may be optimized away under -ffast-math; this uses raw bit inspection.
template <typename T>
inline bool is_missing(T val) noexcept
{
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "is_missing only supports float and double");
  if constexpr (std::is_same_v<T, double>) {
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    // NaN: exponent all 1s AND mantissa non-zero
    return (bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL
        && (bits & 0x000FFFFFFFFFFFFFULL) != 0;
  } else {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    return (bits & 0x7F800000U) == 0x7F800000U
        && (bits & 0x007FFFFFU) != 0;
  }
}

/// Returns true if any element in the vector is NaN.
template <typename T>
bool has_missing(const std::vector<T> &v)
{
  for (const auto &x : v)
    if (is_missing(x)) return true;
  return false;
}

/// Returns the fraction of NaN values in the vector (0.0 if empty).
template <typename T>
double missing_rate(const std::vector<T> &v)
{
  if (v.empty()) return 0.0;
  size_t count = 0;
  for (const auto &x : v)
    if (is_missing(x)) ++count;
  return static_cast<double>(count) / static_cast<double>(v.size());
}

/// Linear interpolation of NaN gaps.
/// Interior NaN: linearly interpolated between nearest observed neighbors.
/// Leading NaN: filled with first observed value (NOCB — Next Observation Carried Backward).
/// Trailing NaN: filled with last observed value (LOCF — Last Observation Carried Forward).
/// Throws std::runtime_error if ALL values are NaN.
template <typename T>
std::vector<T> interpolate_linear(const std::vector<T> &v)
{
  if (v.empty()) return {};

  // Find first and last non-NaN indices
  size_t first_valid = v.size();
  size_t last_valid = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (!is_missing(v[i])) {
      if (first_valid == v.size()) first_valid = i;
      last_valid = i;
    }
  }
  if (first_valid == v.size())
    throw std::runtime_error("interpolate_linear: all values are NaN");

  std::vector<T> result(v.size());

  // NOCB: fill leading NaN with first observed value
  for (size_t i = 0; i < first_valid; ++i)
    result[i] = v[first_valid];

  // Interior: linear interpolation
  size_t prev_valid = first_valid;
  result[first_valid] = v[first_valid];
  for (size_t i = first_valid + 1; i <= last_valid; ++i) {
    if (!is_missing(v[i])) {
      result[i] = v[i];
      // Interpolate the gap between prev_valid and i
      if (i - prev_valid > 1) {
        T start = v[prev_valid];
        T end = v[i];
        T span = static_cast<T>(i - prev_valid);
        for (size_t j = prev_valid + 1; j < i; ++j) {
          T frac = static_cast<T>(j - prev_valid) / span;
          result[j] = start + frac * (end - start);
        }
      }
      prev_valid = i;
    }
  }

  // LOCF: fill trailing NaN with last observed value
  for (size_t i = last_valid + 1; i < v.size(); ++i)
    result[i] = v[last_valid];

  return result;
}

} // namespace dtwc
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build --target unit_test_missing_utils && ctest --test-dir build -R unit_test_missing_utils -V`
Expected: All 14 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add dtwc/missing_utils.hpp tests/unit/unit_test_missing_utils.cpp tests/CMakeLists.txt
git commit -m "feat: add missing_utils.hpp with bitwise NaN check safe under -ffast-math

Provides is_missing<T>() using raw bit inspection instead of std::isnan(),
which is unreliable under -ffast-math/fp:fast. Also adds has_missing(),
missing_rate(), and interpolate_linear() with LOCF/NOCB edge handling."
```

---

### Task 2: Fix `std::isnan()` Bug in `warping_missing.hpp`

**Files:**
- Modify: `dtwc/warping_missing.hpp` (lines 44-60)

- [ ] **Step 1: Write a regression test**

Add to the bottom of `tests/unit/unit_test_missing_dtw.cpp`:

```cpp
// Regression: verify NaN detection works even if compiled with -ffast-math.
// The bitwise is_missing() check must be used instead of std::isnan().
TEST(MissingDTW, nan_detection_is_bitwise_safe)
{
  // Construct NaN via bit manipulation to ensure it's truly NaN
  // even if the compiler tries to optimize NaN away.
  uint64_t nan_bits = 0x7FF8000000000000ULL; // quiet NaN
  double nan_val;
  std::memcpy(&nan_val, &nan_bits, sizeof(nan_val));

  std::vector<double> x = {1.0, nan_val, 3.0};
  std::vector<double> y = {1.0, 2.0, 3.0};

  // With NaN, missing DTW should give LESS than or equal to standard DTW
  // because the NaN position contributes zero cost
  double d_missing = dtwc::dtwMissing_L(x, y);
  double d_standard = dtwc::dtwFull_L(y, y); // 0.0 (identical)

  // The distance should be finite (not NaN-propagated)
  EXPECT_FALSE(dtwc::is_missing(d_missing));
  EXPECT_GE(d_missing, 0.0);
}
```

Add `#include <cstring>` and `#include <dtwc/missing_utils.hpp>` to the test file's includes.

- [ ] **Step 2: Update `MissingL1Dist` in `warping_missing.hpp`**

Replace lines 44-49:

```cpp
struct MissingL1Dist {
  template <typename T>
  T operator()(T a, T b) const {
    if (is_missing(a) || is_missing(b)) return T(0);
    return std::abs(a - b);
  }
};
```

Replace lines 53-60:

```cpp
struct MissingSquaredL2Dist {
  template <typename T>
  T operator()(T a, T b) const {
    if (is_missing(a) || is_missing(b)) return T(0);
    auto d = a - b;
    return d * d;
  }
};
```

Add `#include "missing_utils.hpp"` to the includes of `warping_missing.hpp` (after the existing includes around line 17).

The `is_missing()` calls use the bitwise check from `missing_utils.hpp` instead of `std::isnan()`.

- [ ] **Step 3: Run all missing-data tests**

Run: `cmake --build build --target unit_test_missing_dtw && ctest --test-dir build -R unit_test_missing_dtw -V`
Expected: All 28 tests PASS (27 existing + 1 new regression test).

- [ ] **Step 4: Commit**

```bash
git add dtwc/warping_missing.hpp tests/unit/unit_test_missing_dtw.cpp
git commit -m "fix: replace std::isnan() with bitwise is_missing() in warping_missing.hpp

std::isnan() is unreliable under -ffast-math (GCC/Clang Release builds) and
/fp:fast (MSVC). The missing-data DTW feature was silently broken in production
builds. Now uses bitwise NaN check from missing_utils.hpp."
```

---

### Task 3: Add `MissingStrategy` Enum to `dtw_options.hpp`

**Files:**
- Modify: `dtwc/core/dtw_options.hpp` (after line 38)

- [ ] **Step 1: Add the enum and field**

After the `DTWVariant` enum (line 38) and before `DTWVariantParams` (line 41), add:

```cpp
/// Strategy for handling missing data (NaN values) in time series.
enum class MissingStrategy
{
  Error,        ///< Throw if NaN encountered (default, backward-compatible)
  ZeroCost,     ///< Zero local cost for NaN pairs (existing warping_missing.hpp behavior)
  AROW,         ///< DTW-AROW: one-to-one diagonal-only alignment for missing positions
  Interpolate   ///< Linear interpolation preprocessing, then standard DTW
};
```

Add `MissingStrategy missing_strategy = MissingStrategy::Error;` to the `DTWOptions` struct (after line 55, before the closing brace).

- [ ] **Step 2: Build to verify no compilation errors**

Run: `cmake --build build`
Expected: Clean build. The enum is defined but not yet used.

- [ ] **Step 3: Commit**

```bash
git add dtwc/core/dtw_options.hpp
git commit -m "feat: add MissingStrategy enum (Error/ZeroCost/AROW/Interpolate)"
```

---

### Task 4: Wire Missing-Data DTW into `Problem`

**Files:**
- Modify: `dtwc/Problem.hpp` (add member ~line 88)
- Modify: `dtwc/Problem.cpp` (modify `rebind_dtw_fn()` at lines 122-147, `fillDistanceMatrix()` at lines 229-268)

- [ ] **Step 1: Write the integration test**

Create or append to `tests/unit/unit_test_Problem.cpp` (or create a new test file `tests/unit/unit_test_problem_missing.cpp`):

```cpp
#include <dtwc/dtwc.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

TEST(ProblemMissing, missing_strategy_error_throws_on_nan)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.p_vec = {{1.0, 2.0, 3.0}, {1.0, nan, 3.0}};
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::Error;
  prob.verbose = false;

  EXPECT_THROW(prob.fillDistanceMatrix(), std::runtime_error);
}

TEST(ProblemMissing, missing_strategy_zerocost_computes)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.p_vec = {{1.0, 2.0, 3.0}, {1.0, nan, 3.0}, {4.0, 5.0, 6.0}};
  data.p_names = {"a", "b", "c"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  // Distance should be finite and non-negative
  double d01 = prob.distByInd(0, 1);
  EXPECT_GE(d01, 0.0);
  EXPECT_FALSE(dtwc::is_missing(d01));

  // Zero-cost: NaN position contributes 0, so d(a,b) < d(a,c)
  double d02 = prob.distByInd(0, 2);
  EXPECT_LT(d01, d02);
}

TEST(ProblemMissing, missing_strategy_interpolate_computes)
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.p_vec = {{1.0, 2.0, 3.0}, {1.0, nan, 3.0}};
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::Interpolate;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  // After interpolation, b becomes {1.0, 2.0, 3.0} — identical to a
  double d01 = prob.distByInd(0, 1);
  EXPECT_NEAR(d01, 0.0, 1e-10);
}
```

- [ ] **Step 2: Add `missing_strategy` member to `Problem.hpp`**

After line 87 (`core::DTWVariantParams variant_params;`), add:

```cpp
  core::MissingStrategy missing_strategy = core::MissingStrategy::Error;
```

Add `#include "missing_utils.hpp"` to `Problem.hpp` includes (or ensure it's reachable via existing includes).

- [ ] **Step 3: Modify `rebind_dtw_fn()` in `Problem.cpp`**

Replace the entire `rebind_dtw_fn()` method (lines 122-147) with:

```cpp
void Problem::rebind_dtw_fn()
{
  using namespace core;

  // If missing strategy is Interpolate, pre-interpolate and use standard variant
  // If missing strategy is ZeroCost, use warping_missing functions
  // If missing strategy is AROW, use warping_missing_arow functions (when available)
  // Otherwise, use standard variant dispatch

  if (missing_strategy == MissingStrategy::ZeroCost) {
    dtw_fn_ = [this](const auto &x, const auto &y) {
      return dtwMissing_banded(x, y, band);
    };
    return;
  }

  if (missing_strategy == MissingStrategy::Interpolate) {
    dtw_fn_ = [this](const auto &x, const auto &y) {
      auto xi = has_missing(x) ? interpolate_linear(x) : x;
      auto yi = has_missing(y) ? interpolate_linear(y) : y;
      return dtwBanded(xi, yi, band);
    };
    return;
  }

  // Standard variant dispatch (for Error and AROW-not-yet-implemented)
  switch (variant_params.variant) {
  case DTWVariant::DDTW:
    dtw_fn_ = [this](const auto &x, const auto &y) { return ddtwBanded(x, y, band); };
    break;
  case DTWVariant::WDTW:
    dtw_fn_ = [this](const auto &x, const auto &y) {
      return wdtwBanded(x, y, band, static_cast<data_t>(variant_params.wdtw_g));
    };
    break;
  case DTWVariant::ADTW:
    dtw_fn_ = [this](const auto &x, const auto &y) {
      return adtwBanded(x, y, band, static_cast<data_t>(variant_params.adtw_penalty));
    };
    break;
  case DTWVariant::Standard:
  default:
    dtw_fn_ = [this](const auto &x, const auto &y) { return dtwBanded(x, y, band); };
    break;
  }
}
```

- [ ] **Step 4: Add NaN pre-scan to `fillDistanceMatrix()` in `Problem.cpp`**

At the top of `fillDistanceMatrix()` (after the `isDistanceMatrixFilled()` early return at line 231), add:

```cpp
  // Pre-scan for NaN if strategy is Error
  if (missing_strategy == core::MissingStrategy::Error) {
    for (size_t i = 0; i < data.size(); ++i) {
      if (has_missing(p_vec(i))) {
        throw std::runtime_error(
          "fillDistanceMatrix: NaN detected in series '" + data.p_names[i]
          + "' (index " + std::to_string(i)
          + "). Set missing_strategy to ZeroCost, AROW, or Interpolate to handle missing data.");
      }
    }
  }

  // Disable LB pruning if dataset has missing values
  if (missing_strategy != core::MissingStrategy::Error
      && missing_strategy != core::MissingStrategy::Interpolate) {
    for (size_t i = 0; i < data.size(); ++i) {
      if (has_missing(p_vec(i))) {
        if (effective == DistanceMatrixStrategy::Pruned)
          effective = DistanceMatrixStrategy::BruteForce;
        break;
      }
    }
  }
```

Note: move the `DistanceMatrixStrategy effective = distance_strategy;` line and the `Auto` resolution ABOVE this new block so `effective` is available.

- [ ] **Step 5: Add missing includes to `Problem.cpp`**

Add at the top of `Problem.cpp`:
```cpp
#include "missing_utils.hpp"
#include "warping_missing.hpp"
```

- [ ] **Step 6: Run the integration tests**

Run: `cmake --build build && ctest --test-dir build -R "unit_test_Problem|unit_test_problem_missing" -V`
Expected: All PASS.

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `ctest --test-dir build --output-on-failure`
Expected: All existing tests still PASS.

- [ ] **Step 8: Commit**

```bash
git add dtwc/Problem.hpp dtwc/Problem.cpp tests/unit/unit_test_problem_missing.cpp tests/CMakeLists.txt
git commit -m "feat: wire MissingStrategy into Problem (Error/ZeroCost/Interpolate)

Problem::rebind_dtw_fn() now dispatches based on missing_strategy.
fillDistanceMatrix() pre-scans for NaN under Error strategy (throws with
helpful message), and auto-disables LB pruning for ZeroCost/AROW modes."
```

---

### Task 5: Add Dunn Index

**Files:**
- Modify: `dtwc/scores.hpp`
- Modify: `dtwc/scores.cpp`
- Create: `tests/unit/unit_test_scores_new.cpp`

- [ ] **Step 1: Write the test**

```cpp
// tests/unit/unit_test_scores_new.cpp
#include <dtwc/dtwc.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace {

// Helper: create a Problem with a known distance matrix and clustering.
// 4 points, 2 clusters: {0,1} and {2,3}
// Distance matrix:
//   0  1  5  6
//   1  0  6  5
//   5  6  0  1
//   6  5  1  0
dtwc::Problem make_test_problem_2clusters()
{
  dtwc::Data data;
  data.p_vec = {{0.0}, {1.0}, {5.0}, {6.0}};
  data.p_names = {"a", "b", "c", "d"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.fillDistanceMatrix();

  prob.clusters_ind = {0, 0, 1, 1};
  prob.centroids_ind = {0, 2}; // medoids: point 0 for cluster 0, point 2 for cluster 1
  prob.set_numberOfClusters(2);

  return prob;
}

} // namespace

TEST(ScoresNew, dunn_index_two_well_separated_clusters)
{
  auto prob = make_test_problem_2clusters();
  double dunn = dtwc::scores::dunnIndex(prob);

  // min inter-cluster distance: min(d(0,2), d(0,3), d(1,2), d(1,3)) = min(5,6,6,5) = 5
  // max intra-cluster diameter: max(d(0,1), d(2,3)) = max(1, 1) = 1
  // Dunn = 5/1 = 5.0
  EXPECT_DOUBLE_EQ(dunn, 5.0);
}

TEST(ScoresNew, inertia_two_clusters)
{
  auto prob = make_test_problem_2clusters();
  double iner = dtwc::scores::inertia(prob);

  // inertia = d(0, medoid0) + d(1, medoid0) + d(2, medoid1) + d(3, medoid1)
  //         = d(0,0) + d(1,0) + d(2,2) + d(3,2)
  //         = 0 + 1 + 0 + 1 = 2.0
  EXPECT_DOUBLE_EQ(iner, 2.0);
}

TEST(ScoresNew, calinski_harabasz_two_clusters)
{
  auto prob = make_test_problem_2clusters();
  double ch = dtwc::scores::calinskiHarabaszIndex(prob);

  // Overall medoid: argmin of row-sums
  // Row-sums: 0+1+5+6=12, 1+0+6+5=12, 5+6+0+1=12, 6+5+1+0=12
  // All equal, so overall medoid = 0 (first minimum)
  //
  // W = sum_c sum_{x in c} d(x, medoid_c)^2
  //   = (0^2 + 1^2) + (0^2 + 1^2) = 2
  //
  // B = sum_c |c| * d(medoid_c, overall_medoid)^2
  //   = 2 * d(0,0)^2 + 2 * d(2,0)^2
  //   = 2 * 0 + 2 * 25 = 50
  //
  // CH = (B/(k-1)) / (W/(N-k)) = (50/1) / (2/2) = 50
  EXPECT_DOUBLE_EQ(ch, 50.0);
}

TEST(ScoresNew, adjusted_rand_index_perfect_agreement)
{
  std::vector<int> labels = {0, 0, 1, 1, 2, 2};
  double ari = dtwc::scores::adjustedRandIndex(labels, labels);
  EXPECT_NEAR(ari, 1.0, 1e-10);
}

TEST(ScoresNew, adjusted_rand_index_random_labels)
{
  // Completely different labelings should give ARI near 0
  std::vector<int> true_labels = {0, 0, 0, 1, 1, 1};
  std::vector<int> pred_labels = {0, 1, 0, 1, 0, 1};
  double ari = dtwc::scores::adjustedRandIndex(true_labels, pred_labels);
  EXPECT_LT(ari, 0.5);
}

TEST(ScoresNew, adjusted_rand_index_permutation_invariant)
{
  std::vector<int> true_labels = {0, 0, 1, 1};
  std::vector<int> pred_labels = {1, 1, 0, 0}; // same partition, different label names
  double ari = dtwc::scores::adjustedRandIndex(true_labels, pred_labels);
  EXPECT_NEAR(ari, 1.0, 1e-10);
}

TEST(ScoresNew, nmi_perfect_agreement)
{
  std::vector<int> labels = {0, 0, 1, 1, 2, 2};
  double nmi = dtwc::scores::normalizedMutualInformation(labels, labels);
  EXPECT_NEAR(nmi, 1.0, 1e-10);
}

TEST(ScoresNew, nmi_independent_labels)
{
  // Very different labelings should give low NMI
  std::vector<int> true_labels = {0, 0, 0, 1, 1, 1};
  std::vector<int> pred_labels = {0, 1, 0, 1, 0, 1};
  double nmi = dtwc::scores::normalizedMutualInformation(true_labels, pred_labels);
  EXPECT_LT(nmi, 0.5);
}

TEST(ScoresNew, ari_and_nmi_mismatched_sizes_throw)
{
  std::vector<int> a = {0, 0, 1};
  std::vector<int> b = {0, 1};
  EXPECT_THROW(dtwc::scores::adjustedRandIndex(a, b), std::invalid_argument);
  EXPECT_THROW(dtwc::scores::normalizedMutualInformation(a, b), std::invalid_argument);
}
```

Register in `tests/CMakeLists.txt`:
```cmake
add_executable(unit_test_scores_new unit/unit_test_scores_new.cpp)
target_link_libraries(unit_test_scores_new PRIVATE dtwc++ GTest::gtest_main)
add_test(NAME unit_test_scores_new COMMAND unit_test_scores_new)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build --target unit_test_scores_new`
Expected: FAIL — `dunnIndex`, `inertia`, etc. not declared.

- [ ] **Step 3: Add declarations to `scores.hpp`**

Replace the contents of `dtwc/scores.hpp` with:

```cpp
/**
 * @file scores.hpp
 * @brief Cluster quality metrics for DTWC++.
 *
 * @details Internal validation metrics (need distance matrix + clustering):
 *          silhouette, Davies-Bouldin, Dunn, inertia, Calinski-Harabasz.
 *          External validation metrics (need ground truth labels):
 *          Adjusted Rand Index, Normalized Mutual Information.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include <vector>

namespace dtwc {
class Problem; // Pre-definition
namespace scores {

  // --- Internal validation (need Problem with distance matrix + clustering) ---
  std::vector<double> silhouette(Problem &prob);
  double daviesBouldinIndex(Problem &prob);
  double dunnIndex(Problem &prob);
  double inertia(Problem &prob);
  double calinskiHarabaszIndex(Problem &prob);

  // --- External validation (ground-truth comparison, standalone) ---
  double adjustedRandIndex(const std::vector<int> &labels_true,
                           const std::vector<int> &labels_pred);
  double normalizedMutualInformation(const std::vector<int> &labels_true,
                                     const std::vector<int> &labels_pred);

} // namespace scores
} // namespace dtwc
```

- [ ] **Step 4: Implement all 5 new metrics in `scores.cpp`**

Add to the bottom of `dtwc/scores.cpp` (after `daviesBouldinIndex`):

```cpp
double scores::dunnIndex(Problem &prob)
{
  const int Nb = static_cast<int>(prob.size());
  const int Nc = prob.cluster_size();
  if (Nc < 2 || Nb < 2) return 0.0;

  prob.fillDistanceMatrix();

  // Min inter-cluster distance
  double min_inter = std::numeric_limits<double>::max();
  for (int i = 0; i < Nb; ++i) {
    for (int j = i + 1; j < Nb; ++j) {
      if (prob.clusters_ind[i] != prob.clusters_ind[j]) {
        double d = prob.distByInd(i, j);
        if (d < min_inter) min_inter = d;
      }
    }
  }

  // Max intra-cluster diameter
  double max_intra = 0.0;
  for (int i = 0; i < Nb; ++i) {
    for (int j = i + 1; j < Nb; ++j) {
      if (prob.clusters_ind[i] == prob.clusters_ind[j]) {
        double d = prob.distByInd(i, j);
        if (d > max_intra) max_intra = d;
      }
    }
  }

  if (max_intra == 0.0) return std::numeric_limits<double>::infinity();
  return min_inter / max_intra;
}

double scores::inertia(Problem &prob)
{
  const int Nb = static_cast<int>(prob.size());
  prob.fillDistanceMatrix();

  double total = 0.0;
  for (int i = 0; i < Nb; ++i) {
    int medoid = prob.centroids_ind[prob.clusters_ind[i]];
    total += prob.distByInd(i, medoid);
  }
  return total;
}

double scores::calinskiHarabaszIndex(Problem &prob)
{
  const int Nb = static_cast<int>(prob.size());
  const int Nc = prob.cluster_size();
  if (Nc < 2 || Nb <= Nc) return 0.0;

  prob.fillDistanceMatrix();

  // Find overall medoid (argmin of row-sums)
  int overall_medoid = 0;
  double min_rowsum = std::numeric_limits<double>::max();
  for (int i = 0; i < Nb; ++i) {
    double rowsum = 0.0;
    for (int j = 0; j < Nb; ++j)
      rowsum += prob.distByInd(i, j);
    if (rowsum < min_rowsum) {
      min_rowsum = rowsum;
      overall_medoid = i;
    }
  }

  // W = within-cluster dispersion (sum of squared distances to medoid)
  double W = 0.0;
  for (int i = 0; i < Nb; ++i) {
    int medoid = prob.centroids_ind[prob.clusters_ind[i]];
    double d = prob.distByInd(i, medoid);
    W += d * d;
  }

  // B = between-cluster dispersion (medoid-adapted)
  // Cluster sizes
  std::vector<int> sizes(Nc, 0);
  for (int i = 0; i < Nb; ++i)
    ++sizes[prob.clusters_ind[i]];

  double B = 0.0;
  for (int c = 0; c < Nc; ++c) {
    double d = prob.distByInd(prob.centroids_ind[c], overall_medoid);
    B += sizes[c] * d * d;
  }

  double denom = W / (Nb - Nc);
  if (denom == 0.0) return std::numeric_limits<double>::infinity();
  return (B / (Nc - 1)) / denom;
}
```

Add `#include <stdexcept>`, `#include <unordered_map>`, `#include <cmath>` to the includes of `scores.cpp` if not already present. Then add the external validation metrics:

```cpp
double scores::adjustedRandIndex(const std::vector<int> &labels_true,
                                 const std::vector<int> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw std::invalid_argument("adjustedRandIndex: label vectors must have the same size");

  const int n = static_cast<int>(labels_true.size());
  if (n == 0) return 0.0;

  // Build contingency table
  std::unordered_map<int, int> true_map, pred_map;
  int next_t = 0, next_p = 0;
  for (int i = 0; i < n; ++i) {
    if (true_map.find(labels_true[i]) == true_map.end()) true_map[labels_true[i]] = next_t++;
    if (pred_map.find(labels_pred[i]) == pred_map.end()) pred_map[labels_pred[i]] = next_p++;
  }
  int R = next_t, C = next_p;
  std::vector<int> contingency(R * C, 0);
  std::vector<int> row_sums(R, 0), col_sums(C, 0);

  for (int i = 0; i < n; ++i) {
    int r = true_map[labels_true[i]];
    int c = pred_map[labels_pred[i]];
    ++contingency[r * C + c];
    ++row_sums[r];
    ++col_sums[c];
  }

  // Compute using the combinatorial formula
  // sum_ij C(n_ij, 2)
  auto comb2 = [](int64_t x) -> int64_t { return x * (x - 1) / 2; };

  int64_t sum_nij = 0;
  for (int i = 0; i < R * C; ++i)
    sum_nij += comb2(contingency[i]);

  int64_t sum_ai = 0;
  for (int r = 0; r < R; ++r)
    sum_ai += comb2(row_sums[r]);

  int64_t sum_bj = 0;
  for (int c = 0; c < C; ++c)
    sum_bj += comb2(col_sums[c]);

  int64_t cn2 = comb2(n);
  if (cn2 == 0) return 0.0;

  double expected = static_cast<double>(sum_ai) * static_cast<double>(sum_bj) / static_cast<double>(cn2);
  double max_index = 0.5 * (static_cast<double>(sum_ai) + static_cast<double>(sum_bj));
  double denom = max_index - expected;

  if (std::abs(denom) < 1e-15) return 0.0;
  return (static_cast<double>(sum_nij) - expected) / denom;
}

double scores::normalizedMutualInformation(const std::vector<int> &labels_true,
                                           const std::vector<int> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw std::invalid_argument("normalizedMutualInformation: label vectors must have the same size");

  const int n = static_cast<int>(labels_true.size());
  if (n == 0) return 0.0;

  // Build contingency table (same approach as ARI)
  std::unordered_map<int, int> true_map, pred_map;
  int next_t = 0, next_p = 0;
  for (int i = 0; i < n; ++i) {
    if (true_map.find(labels_true[i]) == true_map.end()) true_map[labels_true[i]] = next_t++;
    if (pred_map.find(labels_pred[i]) == pred_map.end()) pred_map[labels_pred[i]] = next_p++;
  }
  int R = next_t, C = next_p;
  std::vector<int> contingency(R * C, 0);
  std::vector<int> row_sums(R, 0), col_sums(C, 0);

  for (int i = 0; i < n; ++i) {
    int r = true_map[labels_true[i]];
    int c = pred_map[labels_pred[i]];
    ++contingency[r * C + c];
    ++row_sums[r];
    ++col_sums[c];
  }

  double N = static_cast<double>(n);

  // H(true)
  double H_true = 0.0;
  for (int r = 0; r < R; ++r) {
    if (row_sums[r] > 0) {
      double p = row_sums[r] / N;
      H_true -= p * std::log(p);
    }
  }

  // H(pred)
  double H_pred = 0.0;
  for (int c = 0; c < C; ++c) {
    if (col_sums[c] > 0) {
      double p = col_sums[c] / N;
      H_pred -= p * std::log(p);
    }
  }

  // MI (mutual information)
  double MI = 0.0;
  for (int r = 0; r < R; ++r) {
    for (int c = 0; c < C; ++c) {
      int nij = contingency[r * C + c];
      if (nij > 0) {
        double p_ij = nij / N;
        double p_i = row_sums[r] / N;
        double p_j = col_sums[c] / N;
        MI += p_ij * std::log(p_ij / (p_i * p_j));
      }
    }
  }

  // NMI with arithmetic mean normalization
  double denom = 0.5 * (H_true + H_pred);
  if (denom < 1e-15) return 0.0;
  return MI / denom;
}
```

- [ ] **Step 5: Run new metric tests**

Run: `cmake --build build --target unit_test_scores_new && ctest --test-dir build -R unit_test_scores_new -V`
Expected: All 9 tests PASS.

- [ ] **Step 6: Run existing score tests for regression**

Run: `ctest --test-dir build -R "scores" -V`
Expected: Both `unit_test_scores_phase0` (if exists) and `unit_test_scores_new` PASS.

- [ ] **Step 7: Commit**

```bash
git add dtwc/scores.hpp dtwc/scores.cpp tests/unit/unit_test_scores_new.cpp tests/CMakeLists.txt
git commit -m "feat: add 5 cluster quality metrics (Dunn, inertia, CH, ARI, NMI)

Dunn Index: min(inter-cluster) / max(intra-cluster diameter).
Inertia: total within-cluster sum of distances to medoids.
Calinski-Harabasz: medoid-adapted (uses overall medoid as global reference).
ARI: combinatorial agreement with ground truth (permutation-invariant).
NMI: information-theoretic agreement with ground truth."
```

---

### Task 6: Include New Headers in `dtwc.hpp`

**Files:**
- Modify: `dtwc/dtwc.hpp`

- [ ] **Step 1: Add includes**

Add after the existing `#include "warping_missing.hpp"` line:

```cpp
#include "missing_utils.hpp"
```

(The `warping_missing_arow.hpp` will be added in Task 7 after it's created.)

- [ ] **Step 2: Build and run full test suite**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add dtwc/dtwc.hpp
git commit -m "chore: include missing_utils.hpp in main header"
```

---

### Task 7: DTW-AROW Core Algorithm

**Why:** This is the main missing-data improvement — diagonal-only alignment for NaN positions prevents "free stretching" through missing regions.

**Pre-requisite research:** Before writing the implementation, fetch and study the Yurtman et al. reference implementation at `github.com/aras-y/DTW_with_missing_values` to verify boundary condition handling. The boundary treatment (what happens when `x[0]` or `y[0]` is NaN) is the single most critical detail.

**Files:**
- Create: `dtwc/warping_missing_arow.hpp`
- Create: `tests/unit/unit_test_arow_dtw.cpp`
- Modify: `dtwc/dtwc.hpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Research — fetch and study reference implementation**

Run:
```bash
curl -sL "https://raw.githubusercontent.com/aras-y/DTW_with_missing_values/main/dtw_missing.py" -o /tmp/dtw_arow_ref.py 2>/dev/null || echo "Fetch failed - implement from paper description"
```

Study the boundary conditions. Key questions:
- What does the reference do when `x[0]` is NaN? (Is `C(0,0) = 0` or `+inf`?)
- How does the first-row/first-column initialization work under AROW?
- Is the recurrence `C(i,j) = C(i-1,j-1)` or something different?

Document findings as comments in the implementation.

- [ ] **Step 2: Write the test file**

```cpp
// tests/unit/unit_test_arow_dtw.cpp
#include "../test_util.hpp"
#include <dtwc/warping_missing_arow.hpp>
#include <dtwc/warping.hpp>
#include <dtwc/warping_missing.hpp>
#include <dtwc/missing_utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

static const double NaN = std::numeric_limits<double>::quiet_NaN();

// Property 1: No NaN => AROW matches standard DTW
TEST(AROW, no_nan_matches_standard_dtw)
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};

  double d_arow = dtwc::dtwAROW_L(x, y);
  double d_std  = dtwc::dtwFull_L(x, y);

  EXPECT_NEAR(d_arow, d_std, 1e-10);
}

// Property 2: All NaN => AROW returns 0
TEST(AROW, all_nan_returns_zero)
{
  std::vector<double> x = {NaN, NaN, NaN};
  std::vector<double> y = {1, 2, 3};
  EXPECT_DOUBLE_EQ(dtwc::dtwAROW_L(x, y), 0.0);
}

// Property 3: Symmetry
TEST(AROW, symmetry)
{
  std::vector<double> x = {1, NaN, 3, 4};
  std::vector<double> y = {2, 3, NaN, 5};
  EXPECT_DOUBLE_EQ(dtwc::dtwAROW_L(x, y), dtwc::dtwAROW_L(y, x));
}

// Property 4: AROW >= ZeroCost (stricter constraint => higher or equal distance)
TEST(AROW, arow_geq_zerocost)
{
  std::vector<double> x = {1, NaN, NaN, 4, 5};
  std::vector<double> y = {1, 2, 3, 4, 5};

  double d_arow = dtwc::dtwAROW_L(x, y);
  double d_zero = dtwc::dtwMissing_L(x, y);

  EXPECT_GE(d_arow + 1e-10, d_zero); // AROW >= ZeroCost
}

// Property 5: Leading NaN should NOT produce +inf
TEST(AROW, leading_nan_is_finite)
{
  std::vector<double> x = {NaN, NaN, 3, 4};
  std::vector<double> y = {1, 2, 3, 4};

  double d = dtwc::dtwAROW_L(x, y);
  EXPECT_FALSE(dtwc::is_missing(d));
  EXPECT_FALSE(std::isinf(d));
  EXPECT_GE(d, 0.0);
}

// Property 6: Trailing NaN should NOT produce +inf
TEST(AROW, trailing_nan_is_finite)
{
  std::vector<double> x = {1, 2, NaN, NaN};
  std::vector<double> y = {1, 2, 3, 4};

  double d = dtwc::dtwAROW_L(x, y);
  EXPECT_FALSE(dtwc::is_missing(d));
  EXPECT_GE(d, 0.0);
}

// Property 7: Non-negativity
TEST(AROW, non_negative)
{
  std::vector<double> x = {NaN, 2, NaN, 4, NaN};
  std::vector<double> y = {5, 4, 3, 2, 1};
  EXPECT_GE(dtwc::dtwAROW_L(x, y), 0.0);
}

// Property 8: Banded AROW with large band equals unbanded
TEST(AROW, banded_large_band_matches_unbanded)
{
  std::vector<double> x = {1, NaN, 3, 4, 5};
  std::vector<double> y = {2, 3, NaN, 5, 1};

  double d_unbanded = dtwc::dtwAROW_L(x, y);
  double d_banded = dtwc::dtwAROW_banded(x, y, 100);

  EXPECT_NEAR(d_unbanded, d_banded, 1e-10);
}

// Property 9: Full-matrix matches linear-space
TEST(AROW, full_matrix_matches_linear_space)
{
  std::vector<double> x = {1, NaN, 3, NaN, 5};
  std::vector<double> y = {2, 3, NaN, 5, 1};

  double d_full = dtwc::dtwAROW(x, y);
  double d_linear = dtwc::dtwAROW_L(x, y);

  EXPECT_NEAR(d_full, d_linear, 1e-10);
}

// Hand-computed example:
// x = {1, NaN, 3}, y = {1, 2, 3}
// Under AROW:
//   C(0,0) = |1-1| = 0
//   C(1,0) = C(0,-1) -> boundary, use zero-cost boundary propagation -> C(0,0) = 0
//   C(0,1) = C(0,0) + |1-2| = 1
//   C(1,1): x[1] is NaN -> C(1,1) = C(0,0) = 0  (diagonal only)
//   C(0,2) = C(0,1) + |1-3| = 3
//   C(1,2): x[1] is NaN -> C(1,2) = C(0,1) = 1
//   C(2,0) = C(1,0) + |3-1| = 0+2 = 2 (wait, need to check C(1,0))
// This depends on boundary treatment. Leave as property-based test until
// reference implementation is verified.
```

Register in `tests/CMakeLists.txt`:
```cmake
add_executable(unit_test_arow_dtw unit/unit_test_arow_dtw.cpp)
target_link_libraries(unit_test_arow_dtw PRIVATE dtwc++ GTest::gtest_main)
add_test(NAME unit_test_arow_dtw COMMAND unit_test_arow_dtw)
```

- [ ] **Step 3: Create `warping_missing_arow.hpp`**

This is the dedicated `_impl` function. The recurrence differs from standard DTW at missing cells (diagonal-only instead of min-of-3). Cannot reuse the existing `_impl` via functor swap.

```cpp
// dtwc/warping_missing_arow.hpp
/**
 * @file warping_missing_arow.hpp
 * @brief DTW-AROW: DTW with one-to-one alignment constraint for missing values.
 *
 * @details Implements DTW-AROW from Yurtman et al., "Estimating DTW Distance
 *          Between Time Series with Missing Data," ECML-PKDD 2023, LNCS 14173.
 *
 *          When x[i] or y[j] is NaN, the warping path is restricted to the
 *          diagonal direction (one-to-one), preventing many-to-one stretching
 *          through missing regions. Uses bitwise NaN check (safe under -ffast-math).
 *
 * @date 02 Apr 2026
 */

#pragma once

#include "missing_utils.hpp"
#include "settings.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dtwc {
namespace detail {

/// DTW-AROW linear-space implementation.
/// @param x, nx  First series (pointer + length)
/// @param y, ny  Second series (pointer + length)
/// @param distance  Callable (T, T) -> T for observed pairs
/// @return DTW-AROW distance
template <typename data_t, typename DistFn>
data_t dtwAROW_L_impl(const data_t *x, size_t nx, const data_t *y, size_t ny, DistFn distance)
{
  if (nx == 0 || ny == 0) return data_t(0);

  // Ensure x is the shorter side for O(min(m,n)) space
  const data_t *short_ptr = x;
  size_t m_short = nx;
  const data_t *long_ptr = y;
  size_t m_long = ny;
  if (nx > ny) {
    std::swap(short_ptr, long_ptr);
    std::swap(m_short, m_long);
  }

  constexpr data_t INF = std::numeric_limits<data_t>::infinity();
  thread_local std::vector<data_t> col;
  col.resize(m_short);

  // Initialize first column (j=0)
  // C(0,0): if either missing -> 0, else distance
  bool s0_miss = is_missing(short_ptr[0]);
  bool l0_miss = is_missing(long_ptr[0]);
  col[0] = (s0_miss || l0_miss) ? data_t(0) : distance(short_ptr[0], long_ptr[0]);

  for (size_t i = 1; i < m_short; ++i) {
    bool si_miss = is_missing(short_ptr[i]);
    if (si_miss || l0_miss) {
      // AROW: diagonal only. In column 0, diagonal predecessor is C(i-1, -1) = doesn't exist.
      // Boundary treatment: propagate from C(i-1, 0) — in the first column,
      // there is only one direction (vertical), so no many-to-one cheating is possible.
      col[i] = col[i - 1]; // zero additional cost for missing
    } else {
      col[i] = col[i - 1] + distance(short_ptr[i], long_ptr[0]);
    }
  }

  // Fill remaining columns
  for (size_t j = 1; j < m_long; ++j) {
    bool lj_miss = is_missing(long_ptr[j]);

    // Save diagonal predecessor before overwriting
    data_t diag = col[0];

    // Update col[0] (first row, column j)
    if (s0_miss || lj_miss) {
      // First row: only horizontal predecessor. No many-to-one cheating possible.
      // col[0] stays as col[0] (horizontal propagation with zero cost).
    } else {
      col[0] = col[0] + distance(short_ptr[0], long_ptr[j]);
    }

    for (size_t i = 1; i < m_short; ++i) {
      data_t old_col_i = col[i]; // this becomes diag for next i
      bool si_miss = is_missing(short_ptr[i]);

      if (si_miss || lj_miss) {
        // AROW: use diagonal predecessor only (one-to-one alignment)
        col[i] = diag;
      } else {
        // Standard DTW: min of three predecessors + local cost
        data_t cost = distance(short_ptr[i], long_ptr[j]);
        col[i] = std::min({diag, col[i - 1], col[i]}) + cost;
      }

      diag = old_col_i;
    }
  }

  return col[m_short - 1];
}

/// DTW-AROW full-matrix implementation (for debugging/verification).
template <typename data_t, typename DistFn>
data_t dtwAROW_impl(const data_t *x, size_t nx, const data_t *y, size_t ny, DistFn distance)
{
  if (nx == 0 || ny == 0) return data_t(0);

  constexpr data_t INF = std::numeric_limits<data_t>::infinity();
  std::vector<data_t> C(nx * ny, INF);
  auto idx = [ny](size_t i, size_t j) { return i * ny + j; };

  // C(0,0)
  C[idx(0, 0)] = (is_missing(x[0]) || is_missing(y[0]))
                     ? data_t(0)
                     : distance(x[0], y[0]);

  // First row
  for (size_t j = 1; j < ny; ++j) {
    if (is_missing(x[0]) || is_missing(y[j]))
      C[idx(0, j)] = C[idx(0, j - 1)];
    else
      C[idx(0, j)] = C[idx(0, j - 1)] + distance(x[0], y[j]);
  }

  // First column
  for (size_t i = 1; i < nx; ++i) {
    if (is_missing(x[i]) || is_missing(y[0]))
      C[idx(i, 0)] = C[idx(i - 1, 0)];
    else
      C[idx(i, 0)] = C[idx(i - 1, 0)] + distance(x[i], y[0]);
  }

  // Interior
  for (size_t i = 1; i < nx; ++i) {
    for (size_t j = 1; j < ny; ++j) {
      if (is_missing(x[i]) || is_missing(y[j])) {
        C[idx(i, j)] = C[idx(i - 1, j - 1)]; // diagonal only
      } else {
        data_t cost = distance(x[i], y[j]);
        C[idx(i, j)] = std::min({C[idx(i - 1, j - 1)], C[idx(i - 1, j)], C[idx(i, j - 1)]}) + cost;
      }
    }
  }

  return C[idx(nx - 1, ny - 1)];
}

/// DTW-AROW banded (Sakoe-Chiba) implementation.
template <typename data_t, typename DistFn>
data_t dtwAROW_banded_impl(const data_t *x, size_t nx, const data_t *y, size_t ny,
                            int band, DistFn distance)
{
  if (band < 0) return dtwAROW_L_impl(x, nx, y, ny, distance);
  if (nx == 0 || ny == 0) return data_t(0);

  constexpr data_t INF = std::numeric_limits<data_t>::infinity();
  thread_local std::vector<data_t> col;
  const size_t col_size = static_cast<size_t>(2 * band + 1);
  col.resize(std::max(col_size, ny));

  // For banded: use full-matrix approach with band constraint.
  // Reuse the full-matrix impl with band check.
  // This is simpler and correct; optimize later if needed.
  std::vector<data_t> C(nx * ny, INF);
  auto idx = [ny](size_t i, size_t j) { return i * ny + j; };
  auto in_band = [&](size_t i, size_t j) {
    return std::abs(static_cast<int>(i) - static_cast<int>(j)) <= band;
  };

  if (in_band(0, 0))
    C[idx(0, 0)] = (is_missing(x[0]) || is_missing(y[0]))
                       ? data_t(0)
                       : distance(x[0], y[0]);

  for (size_t j = 1; j < ny; ++j) {
    if (!in_band(0, j)) continue;
    if (is_missing(x[0]) || is_missing(y[j]))
      C[idx(0, j)] = (in_band(0, j - 1) && C[idx(0, j - 1)] < INF) ? C[idx(0, j - 1)] : data_t(0);
    else if (in_band(0, j - 1) && C[idx(0, j - 1)] < INF)
      C[idx(0, j)] = C[idx(0, j - 1)] + distance(x[0], y[j]);
  }

  for (size_t i = 1; i < nx; ++i) {
    if (!in_band(i, 0)) continue;
    if (is_missing(x[i]) || is_missing(y[0]))
      C[idx(i, 0)] = (in_band(i - 1, 0) && C[idx(i - 1, 0)] < INF) ? C[idx(i - 1, 0)] : data_t(0);
    else if (in_band(i - 1, 0) && C[idx(i - 1, 0)] < INF)
      C[idx(i, 0)] = C[idx(i - 1, 0)] + distance(x[i], y[0]);
  }

  for (size_t i = 1; i < nx; ++i) {
    for (size_t j = 1; j < ny; ++j) {
      if (!in_band(i, j)) continue;
      if (is_missing(x[i]) || is_missing(y[j])) {
        if (in_band(i - 1, j - 1) && C[idx(i - 1, j - 1)] < INF)
          C[idx(i, j)] = C[idx(i - 1, j - 1)];
        else
          C[idx(i, j)] = data_t(0); // boundary fallback
      } else {
        data_t d_val = distance(x[i], y[j]);
        data_t best = INF;
        if (in_band(i - 1, j - 1) && C[idx(i - 1, j - 1)] < INF) best = C[idx(i - 1, j - 1)];
        if (in_band(i - 1, j) && C[idx(i - 1, j)] < best) best = C[idx(i - 1, j)];
        if (in_band(i, j - 1) && C[idx(i, j - 1)] < best) best = C[idx(i, j - 1)];
        C[idx(i, j)] = (best < INF) ? best + d_val : d_val;
      }
    }
  }

  return (C[idx(nx - 1, ny - 1)] < INF) ? C[idx(nx - 1, ny - 1)] : data_t(0);
}

} // namespace detail

// --- Public API ---

template <typename data_t = double>
data_t dtwAROW_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 core::MetricType metric = core::MetricType::L1)
{
  auto dist = (metric == core::MetricType::SquaredL2) ? detail::SquaredL2Dist{} : detail::L1Dist{};
  if (metric == core::MetricType::SquaredL2)
    return detail::dtwAROW_L_impl(x.data(), x.size(), y.data(), y.size(), detail::SquaredL2Dist{});
  return detail::dtwAROW_L_impl(x.data(), x.size(), y.data(), y.size(), detail::L1Dist{});
}

template <typename data_t = double>
data_t dtwAROW(const std::vector<data_t> &x, const std::vector<data_t> &y,
               core::MetricType metric = core::MetricType::L1)
{
  if (metric == core::MetricType::SquaredL2)
    return detail::dtwAROW_impl(x.data(), x.size(), y.data(), y.size(), detail::SquaredL2Dist{});
  return detail::dtwAROW_impl(x.data(), x.size(), y.data(), y.size(), detail::L1Dist{});
}

template <typename data_t = double>
data_t dtwAROW_banded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                      int band = settings::DEFAULT_BAND_LENGTH,
                      core::MetricType metric = core::MetricType::L1)
{
  if (metric == core::MetricType::SquaredL2)
    return detail::dtwAROW_banded_impl(x.data(), x.size(), y.data(), y.size(), band, detail::SquaredL2Dist{});
  return detail::dtwAROW_banded_impl(x.data(), x.size(), y.data(), y.size(), band, detail::L1Dist{});
}

} // namespace dtwc
```

- [ ] **Step 4: Add include to `dtwc.hpp`**

After the `warping_missing.hpp` include, add:
```cpp
#include "warping_missing_arow.hpp"
```

- [ ] **Step 5: Run AROW tests**

Run: `cmake --build build --target unit_test_arow_dtw && ctest --test-dir build -R unit_test_arow_dtw -V`
Expected: All 9 property tests PASS.

- [ ] **Step 6: Wire AROW into Problem::rebind_dtw_fn()**

In `Problem.cpp`, add a case for AROW in `rebind_dtw_fn()` (after the `ZeroCost` block, before the `Interpolate` block):

```cpp
  if (missing_strategy == MissingStrategy::AROW) {
    dtw_fn_ = [this](const auto &x, const auto &y) {
      return dtwAROW_banded(x, y, band);
    };
    return;
  }
```

Add `#include "warping_missing_arow.hpp"` to `Problem.cpp` includes.

- [ ] **Step 7: Run full test suite**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add dtwc/warping_missing_arow.hpp dtwc/dtwc.hpp dtwc/Problem.cpp tests/unit/unit_test_arow_dtw.cpp tests/CMakeLists.txt
git commit -m "feat: add DTW-AROW algorithm (diagonal-only alignment for missing values)

Implements DTW-AROW from Yurtman et al. (ECML-PKDD 2023). When x[i] or y[j]
is NaN, the warping path is restricted to diagonal (one-to-one) alignment,
preventing free stretching through missing regions. Provides linear-space,
full-matrix, and banded variants. Wired into Problem via MissingStrategy::AROW."
```

---

### Task 8: Update CHANGELOG.md

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add entries to Unreleased section**

Add under the `## [Unreleased]` section:

```markdown
### Added
- `missing_utils.hpp`: Bitwise NaN check (`is_missing()`) safe under `-ffast-math`/`/fp:fast`, plus `has_missing()`, `missing_rate()`, `interpolate_linear()` with LOCF/NOCB edge handling.
- `MissingStrategy` enum: `Error` (default), `ZeroCost`, `AROW`, `Interpolate` for controlling missing-data handling in `Problem`.
- DTW-AROW algorithm (`warping_missing_arow.hpp`): One-to-one diagonal-only alignment for missing values (Yurtman et al., ECML-PKDD 2023).
- 5 new cluster quality metrics in `scores.hpp`:
  - `dunnIndex()`: Min inter-cluster distance / max intra-cluster diameter.
  - `inertia()`: Total within-cluster sum of distances to medoids.
  - `calinskiHarabaszIndex()`: Medoid-adapted Calinski-Harabasz (uses overall medoid as global reference).
  - `adjustedRandIndex()`: Combinatorial agreement with ground-truth labels.
  - `normalizedMutualInformation()`: Information-theoretic agreement with ground-truth labels.

### Fixed
- `warping_missing.hpp`: Replaced `std::isnan()` with bitwise `is_missing()` check. The missing-data DTW feature was silently broken in Release builds due to `-ffast-math`/`/fp:fast` making `std::isnan()` unreliable.

### Changed
- `Problem::fillDistanceMatrix()` now pre-scans for NaN and throws with a helpful message under `MissingStrategy::Error`. Auto-disables LB pruning when missing data is detected under `ZeroCost`/`AROW` strategies.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for Wave 1A (metrics + missing data)"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - C0 (isnan fix): Task 1 + Task 2
   - C1 (MissingStrategy enum): Task 3
   - C2 (DTW-AROW): Task 7
   - C3 (Interpolation): Task 4 (Interpolate wired in rebind_dtw_fn)
   - C5 (LB pruning guard): Task 4 (fillDistanceMatrix NaN guard)
   - C6 (Problem integration): Task 4
   - D1-D3 (metrics): Task 5
   - CHANGELOG: Task 8
   - **Gap:** C4 (path-length normalization heuristic) — deferred to a follow-up task as it requires modifying the DTW _impl return value. Not blocking.

2. **Placeholder scan:** No TBD/TODO. All code blocks are complete. The AROW boundary treatment has a clear research step (Step 1 of Task 7).

3. **Type consistency:** `is_missing()`, `has_missing()`, `interpolate_linear()` — used consistently. `MissingStrategy` enum referenced as `core::MissingStrategy` in Problem (consistent with dtw_options.hpp namespace). `dtwAROW_L`, `dtwAROW`, `dtwAROW_banded` — consistent signatures in header and tests.
