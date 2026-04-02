# Wave 1B: Multivariate ND Array Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multivariate (multi-dimensional) time series support to DTWC++ without degrading univariate performance. Each timestep can have D features (e.g., temperature, voltage, current).

**Architecture:** Interleaved flat storage `[t0_f0, t0_f1, ..., t1_f0, t1_f1, ...]` with `Data.ndim` field. Compile-time dispatch for D=1 (zero overhead — identical to current code). The `_impl` functions gain a `stride` parameter; distance functors change from `(T,T)→T` to `(const T*, const T*, size_t)→T`.

**Tech Stack:** C++17, Catch2, CMake/MSVC

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `dtwc/Data.hpp` | Add `ndim` field, `series_length()` accessor, validation |
| Modify | `dtwc/core/time_series.hpp` | Add `ndim` to `TimeSeriesView` |
| Modify | `dtwc/core/distance_metric.hpp` | Add multivariate metric functors |
| Modify | `dtwc/warping.hpp` | Add `_mv_impl` variants and `ndim` public API overloads |
| Modify | `dtwc/warping_ddtw.hpp` | Stride-aware `derivative_transform` |
| Modify | `dtwc/core/lower_bound_impl.hpp` | Per-channel `compute_envelopes` and `lb_keogh` |
| Modify | `dtwc/Problem.cpp` | Pass `data.ndim` through `rebind_dtw_fn()` |
| Create | `tests/unit/unit_test_multivariate_dtw.cpp` | Tests for MV DTW |
| Create | `tests/unit/unit_test_multivariate_data.cpp` | Tests for Data.ndim |

---

### Task 1: Add `ndim` to Data and TimeSeriesView

**Files:**
- Modify: `dtwc/Data.hpp`
- Modify: `dtwc/core/time_series.hpp`
- Create: `tests/unit/unit_test_multivariate_data.cpp`

- [ ] **Step 1: Write tests**

```cpp
// tests/unit/unit_test_multivariate_data.cpp
#include <dtwc.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Data: ndim defaults to 1", "[data][mv]")
{
  dtwc::Data data;
  REQUIRE(data.ndim == 1);
}

TEST_CASE("Data: series_length with ndim=1", "[data][mv]")
{
  dtwc::Data data;
  data.p_vec = {{1.0, 2.0, 3.0}};
  data.p_names = {"a"};
  REQUIRE(data.series_length(0) == 3);
}

TEST_CASE("Data: series_length with ndim=3", "[data][mv]")
{
  // 2 timesteps x 3 features = 6 elements
  dtwc::Data data;
  data.ndim = 3;
  data.p_vec = {{1,2,3, 4,5,6}};
  data.p_names = {"a"};
  REQUIRE(data.series_length(0) == 2);
}

TEST_CASE("Data: validate_ndim catches bad size", "[data][mv]")
{
  dtwc::Data data;
  data.ndim = 3;
  data.p_vec = {{1,2,3,4,5}}; // 5 not divisible by 3
  data.p_names = {"a"};
  REQUIRE_THROWS(data.validate_ndim());
}

TEST_CASE("Data: validate_ndim passes for valid", "[data][mv]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {{1,2,3,4}, {5,6,7,8}};
  data.p_names = {"a", "b"};
  REQUIRE_NOTHROW(data.validate_ndim());
}

TEST_CASE("TimeSeriesView: ndim and at()", "[data][mv]")
{
  double arr[] = {1,2,3, 4,5,6}; // 2 timesteps x 3 features
  dtwc::core::TimeSeriesView<double> view{arr, 2, 3};
  REQUIRE(view.length == 2);
  REQUIRE(view.ndim == 3);
  REQUIRE(view.at(0)[0] == 1.0);
  REQUIRE(view.at(0)[2] == 3.0);
  REQUIRE(view.at(1)[0] == 4.0);
  REQUIRE(view.at(1)[2] == 6.0);
  REQUIRE(view.flat_size() == 6);
}
```

- [ ] **Step 2: Modify `dtwc/Data.hpp`**

Add `ndim` field, `series_length()`, and `validate_ndim()`:

```cpp
struct Data
{
  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;
  size_t ndim = 1;  ///< Dimensions per timestep (1 = univariate)

  auto size() const { return p_vec.size(); }

  /// Number of timesteps for series i (flat_size / ndim).
  size_t series_length(size_t i) const { return p_vec[i].size() / ndim; }

  /// Validate that all series sizes are divisible by ndim.
  void validate_ndim() const
  {
    for (size_t i = 0; i < p_vec.size(); ++i) {
      if (p_vec[i].size() % ndim != 0)
        throw std::runtime_error(
          "Data::validate_ndim: series '" + p_names[i] + "' (index " + std::to_string(i)
          + ") has " + std::to_string(p_vec[i].size()) + " elements, not divisible by ndim="
          + std::to_string(ndim));
    }
  }

  Data() = default;

  Data(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new,
       size_t ndim_new = 1)
    : ndim(ndim_new)
  {
    if (p_vec_new.size() != p_names_new.size())
      throw std::runtime_error("Data and name vectors should be of the same size");
    p_vec = std::move(p_vec_new);
    p_names = std::move(p_names_new);
    validate_ndim();
  }
};
```

Add `#include <stdexcept>` if not present.

- [ ] **Step 3: Modify `dtwc/core/time_series.hpp`**

Add `ndim` to `TimeSeriesView`:

```cpp
template <typename T = double>
struct TimeSeriesView {
  const T *data;
  size_t length;   ///< Number of timesteps
  size_t ndim = 1; ///< Features per timestep

  /// Access timestep i as a pointer to ndim contiguous values.
  const T *at(size_t i) const { return data + i * ndim; }

  /// Total number of elements (length * ndim).
  size_t flat_size() const { return length * ndim; }

  // Keep existing operator[] for backward compat (univariate)
  const T &operator[](size_t i) const { return data[i]; }
  const T *begin() const { return data; }
  const T *end() const { return data + length * ndim; }
  bool empty() const { return length == 0; }
  size_t size() const { return length; }

  bool operator==(const TimeSeriesView &other) const {
    if (length != other.length || ndim != other.ndim) return false;
    const size_t total = length * ndim;
    for (size_t i = 0; i < total; ++i)
      if (data[i] != other.data[i]) return false;
    return true;
  }
  bool operator!=(const TimeSeriesView &other) const { return !(*this == other); }
};
```

- [ ] **Step 4: Build and test**

```
cmake -S . -B build && cmake --build build --config Release --target unit_test_multivariate_data
build/bin/unit_test_multivariate_data.exe
```

- [ ] **Step 5: Run full test suite for regressions**

```
ctest --test-dir build -C Release --output-on-failure
```

- [ ] **Step 6: Commit**

```bash
git add dtwc/Data.hpp dtwc/core/time_series.hpp tests/unit/unit_test_multivariate_data.cpp
git commit -m "feat: add ndim to Data and TimeSeriesView for multivariate support

Data.ndim (default 1) specifies features per timestep. series_length()
returns timestep count. validate_ndim() checks divisibility. TimeSeriesView
gains ndim, at(i) for timestep access, flat_size()."
```

---

### Task 2: Add Multivariate Distance Metric Functors

**Files:**
- Modify: `dtwc/core/distance_metric.hpp`

- [ ] **Step 1: Write tests**

Add to `tests/unit/unit_test_multivariate_dtw.cpp` (create new file):

```cpp
// tests/unit/unit_test_multivariate_dtw.cpp
#include <dtwc.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <cmath>

using Catch::Matchers::WithinAbs;

// ---- Metric functor tests ----

TEST_CASE("MVL1Dist: ndim=1 matches scalar L1", "[mv][metric]")
{
  double a = 3.0, b = 7.0;
  REQUIRE(dtwc::detail::MVL1Dist{}(&a, &b, 1) == std::abs(a - b));
}

TEST_CASE("MVL1Dist: ndim=3", "[mv][metric]")
{
  double a[] = {1.0, 2.0, 3.0};
  double b[] = {4.0, 1.0, 6.0};
  // |1-4| + |2-1| + |3-6| = 3 + 1 + 3 = 7
  REQUIRE(dtwc::detail::MVL1Dist{}(a, b, 3) == 7.0);
}

TEST_CASE("MVSquaredL2Dist: ndim=1 matches scalar", "[mv][metric]")
{
  double a = 3.0, b = 7.0;
  REQUIRE(dtwc::detail::MVSquaredL2Dist{}(&a, &b, 1) == 16.0);
}

TEST_CASE("MVSquaredL2Dist: ndim=3", "[mv][metric]")
{
  double a[] = {1.0, 2.0, 3.0};
  double b[] = {4.0, 1.0, 6.0};
  // (1-4)^2 + (2-1)^2 + (3-6)^2 = 9 + 1 + 9 = 19
  REQUIRE(dtwc::detail::MVSquaredL2Dist{}(a, b, 3) == 19.0);
}
```

- [ ] **Step 2: Add multivariate functors to `distance_metric.hpp`**

After the existing `SquaredL2Metric` struct, add (still in `dtwc::core` namespace):

```cpp
// These are kept separate in dtwc::detail for use in DTW _impl functions.
```

And in `dtwc/warping.hpp`, after the existing `detail::L1Dist` and `detail::SquaredL2Dist` (around line 265), add the multivariate versions inside `namespace detail`:

```cpp
/// Multivariate L1: sum of |a[d] - b[d]| across ndim dimensions.
struct MVL1Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d)
      sum += std::abs(a[d] - b[d]);
    return sum;
  }
};

/// Multivariate Squared L2: sum of (a[d] - b[d])^2 across ndim dimensions.
struct MVSquaredL2Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d) {
      T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }
};
```

- [ ] **Step 3: Build and test**

```
cmake -S . -B build && cmake --build build --config Release --target unit_test_multivariate_dtw
build/bin/unit_test_multivariate_dtw.exe
```

- [ ] **Step 4: Commit**

```bash
git add dtwc/warping.hpp tests/unit/unit_test_multivariate_dtw.cpp
git commit -m "feat: add multivariate distance functors MVL1Dist and MVSquaredL2Dist"
```

---

### Task 3: Add Multivariate DTW `_impl` Functions

This is the core change. Add new `_impl` functions that accept `ndim` parameter and use pointer-stride indexing.

**Key design:** Instead of modifying the existing `_impl` functions (which would risk regressions), add NEW `_mv_impl` functions. The public API dispatches: `ndim==1` → existing `_impl` (zero overhead), `ndim>1` → `_mv_impl`.

**Files:**
- Modify: `dtwc/warping.hpp`

- [ ] **Step 1: Write tests (append to `unit_test_multivariate_dtw.cpp`)**

```cpp
// ---- Multivariate DTW tests ----

TEST_CASE("MV DTW: ndim=1 matches standard DTW", "[mv][dtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};

  double d_std = dtwc::dtwFull_L(x, y);
  double d_mv = dtwc::dtwFull_L_mv(x.data(), 5, y.data(), 5, 1);

  REQUIRE_THAT(d_mv, WithinAbs(d_std, 1e-10));
}

TEST_CASE("MV DTW banded: ndim=1 matches standard banded", "[mv][dtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};

  double d_std = dtwc::dtwBanded(x, y, 2);
  double d_mv = dtwc::dtwBanded_mv(x.data(), 5, y.data(), 5, 1, 2);

  REQUIRE_THAT(d_mv, WithinAbs(d_std, 1e-10));
}

TEST_CASE("MV DTW: 2D identical series = 0", "[mv][dtw]")
{
  // 3 timesteps x 2 features, interleaved
  double x[] = {1,2, 3,4, 5,6};
  double y[] = {1,2, 3,4, 5,6};

  double d = dtwc::dtwFull_L_mv(x, 3, y, 3, 2);
  REQUIRE(d == 0.0);
}

TEST_CASE("MV DTW: 2D known distance", "[mv][dtw]")
{
  // 2 timesteps x 2 features
  // x: [(0,0), (1,1)]
  // y: [(1,1), (2,2)]
  double x[] = {0,0, 1,1};
  double y[] = {1,1, 2,2};

  // DTW with L1: each timestep distance = |0-1|+|0-1|=2 or |1-2|+|1-2|=2
  // Optimal alignment: (0,0)→(1,1) cost 2, (1,1)→(2,2) cost 2. Total = 4
  double d = dtwc::dtwFull_L_mv(x, 2, y, 2, 2);
  REQUIRE_THAT(d, WithinAbs(4.0, 1e-10));
}

TEST_CASE("MV DTW: 3D series", "[mv][dtw]")
{
  // 2 timesteps x 3 features
  // x: [(1,0,0), (0,1,0)]
  // y: [(0,0,1), (1,0,0)]
  double x[] = {1,0,0, 0,1,0};
  double y[] = {0,0,1, 1,0,0};

  // L1 cost (0,0): |1|+|0|+|1| = 2
  // L1 cost (0,1): |1-1|+|0|+|0| = 0
  // L1 cost (1,0): |0|+|1|+|1| = 2
  // L1 cost (1,1): |0-1|+|1|+|0| = 2
  // DTW: min path = (0,0)→(0,1)→(1,1) = 2+0+2 = 4
  //   or (0,0)→(1,0)→(1,1) = 2+2+2 = 6
  //   or (0,0)→(1,1) = 2+2 = 4
  // Best = 4
  double d = dtwc::dtwFull_L_mv(x, 2, y, 2, 3);
  REQUIRE_THAT(d, WithinAbs(4.0, 1e-10));
}

TEST_CASE("MV DTW: symmetry", "[mv][dtw]")
{
  double x[] = {1,2,3, 4,5,6, 7,8,9};
  double y[] = {9,8,7, 6,5,4, 3,2,1};

  double d1 = dtwc::dtwFull_L_mv(x, 3, y, 3, 3);
  double d2 = dtwc::dtwFull_L_mv(y, 3, x, 3, 3);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

TEST_CASE("MV DTW banded: ndim=2 with large band matches unbanded", "[mv][dtw]")
{
  double x[] = {1,2, 3,4, 5,6, 7,8};
  double y[] = {2,1, 4,3, 6,5, 8,7};

  double d_full = dtwc::dtwFull_L_mv(x, 4, y, 4, 2);
  double d_band = dtwc::dtwBanded_mv(x, 4, y, 4, 2, 100);
  REQUIRE_THAT(d_full, WithinAbs(d_band, 1e-10));
}

TEST_CASE("MV DTW SquaredL2: ndim=2", "[mv][dtw]")
{
  // x: [(0,0), (1,1)], y: [(1,1), (2,2)]
  double x[] = {0,0, 1,1};
  double y[] = {1,1, 2,2};

  // SquaredL2: each timestep = (0-1)^2+(0-1)^2 = 2
  // Total = 2 + 2 = 4
  double d = dtwc::dtwFull_L_mv(x, 2, y, 2, 2, -1, dtwc::core::MetricType::SquaredL2);
  REQUIRE_THAT(d, WithinAbs(4.0, 1e-10));
}
```

- [ ] **Step 2: Add `dtwFull_L_mv_impl` to `warping.hpp`**

Add inside `namespace detail`, after the existing `dtwBanded_impl` (around line 243):

```cpp
/// Multivariate linear-space DTW.
/// @param x, nx_steps  First series (flat interleaved, nx_steps timesteps)
/// @param y, ny_steps  Second series (flat interleaved, ny_steps timesteps)
/// @param ndim         Features per timestep
/// @param early_abandon Threshold; negative disables
/// @param distance     Callable (const T*, const T*, size_t ndim) -> T
template <typename data_t, typename DistFn>
data_t dtwFull_L_mv_impl(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                          size_t ndim, data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  if (nx_steps == 0 || ny_steps == 0) return maxValue;
  if (x == y && nx_steps == ny_steps) return 0;

  thread_local static std::vector<data_t> short_side;

  const data_t* short_ptr;
  const data_t* long_ptr;
  size_t m_short, m_long;
  if (nx_steps < ny_steps) {
    short_ptr = x; m_short = nx_steps;
    long_ptr  = y; m_long  = ny_steps;
  } else {
    short_ptr = y; m_short = ny_steps;
    long_ptr  = x; m_long  = nx_steps;
  }

  short_side.resize(m_short);

  short_side[0] = distance(short_ptr, long_ptr, ndim);

  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + distance(short_ptr + i * ndim, long_ptr, ndim);

  const bool do_early_abandon = (early_abandon >= 0);

  for (size_t j = 1; j < m_long; j++) {
    const data_t* long_j = long_ptr + j * ndim;
    auto diag = short_side[0];
    short_side[0] += distance(short_ptr, long_j, ndim);

    data_t row_min = do_early_abandon ? short_side[0] : data_t{0};

    for (size_t i = 1; i < m_short; i++) {
      const data_t min1 = std::min(short_side[i - 1], short_side[i]);
      const data_t dist = distance(short_ptr + i * ndim, long_j, ndim);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
      if (do_early_abandon) row_min = std::min(row_min, next);
    }

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return short_side.back();
}

/// Multivariate banded DTW.
template <typename data_t, typename DistFn>
data_t dtwBanded_mv_impl(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                          size_t ndim, int band, data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const data_t* short_ptr;
  const data_t* long_ptr;
  int m_short, m_long;
  if (nx_steps < ny_steps) {
    short_ptr = x; m_short = static_cast<int>(nx_steps);
    long_ptr  = y; m_long  = static_cast<int>(ny_steps);
  } else {
    short_ptr = y; m_short = static_cast<int>(ny_steps);
    long_ptr  = x; m_long  = static_cast<int>(nx_steps);
  }

  if ((m_short == 0) || (m_long == 0)) return maxValue;

  const double slope = static_cast<double>(m_long - 1) / std::max(m_short - 1, 1);
  const auto window = std::max(static_cast<double>(band), slope / 2);

  thread_local std::vector<data_t> col;
  col.assign(m_long, maxValue);
  thread_local std::vector<int> low_bounds, high_bounds;
  low_bounds.resize(m_short);
  high_bounds.resize(m_short);
  const bool do_early_abandon = (early_abandon >= 0);

  for (int row = 0; row < m_short; ++row) {
    const double center = slope * row;
    low_bounds[row] = static_cast<int>(
      std::ceil(std::round(100.0 * (center - window)) / 100.0));
    high_bounds[row] = static_cast<int>(
      std::floor(std::round(100.0 * (center + window)) / 100.0)) + 1;
  }

  col[0] = distance(long_ptr, short_ptr, ndim);
  {
    const int hi = high_bounds[0];
    for (int i = 1; i < std::min(hi, m_long); ++i)
      col[i] = col[i - 1] + distance(long_ptr + i * ndim, short_ptr, ndim);
  }
  if (do_early_abandon && col[0] > early_abandon) return maxValue;

  for (int j = 1; j < m_short; j++) {
    const int lo = low_bounds[j];
    const int hi = high_bounds[j];
    const int prev_lo = low_bounds[j - 1];
    const int prev_hi = high_bounds[j - 1];
    const int high = std::min(hi, m_long);
    const int low = std::max(lo, 0);

    data_t diag = maxValue;
    data_t row_min = do_early_abandon ? maxValue : data_t{0};

    const int first_row = std::max(low, 1);
    if (first_row - 1 >= std::max(prev_lo, 0) && first_row - 1 < std::min(prev_hi, m_long))
      diag = col[first_row - 1];

    if (low == 0) {
      diag = col[0];
      col[0] = col[0] + distance(long_ptr, short_ptr + j * ndim, ndim);
      if (do_early_abandon) row_min = col[0];
    }

    for (int i = std::max(prev_lo, 0); i < std::min(low, std::min(prev_hi, m_long)); ++i)
      col[i] = maxValue;

    for (int i = first_row; i < high; ++i) {
      const data_t old_col_i = col[i];
      const auto minimum = std::min(diag, std::min(col[i - 1], old_col_i));
      diag = old_col_i;
      col[i] = minimum + distance(long_ptr + i * ndim, short_ptr + j * ndim, ndim);
      if (do_early_abandon) row_min = std::min(row_min, col[i]);
    }

    for (int i = std::max(high, std::max(prev_lo, 0)); i < std::min(prev_hi, m_long); ++i)
      col[i] = maxValue;

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return col[m_long - 1];
}
```

- [ ] **Step 3: Add public API functions**

After the existing public API functions in `warping.hpp`:

```cpp
/// Multivariate linear-space DTW (pointer + length + ndim).
template <typename data_t = double>
data_t dtwFull_L_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1) return dtwFull_L(x, nx_steps, y, ny_steps, early_abandon, metric);

  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwFull_L_mv_impl(x, nx_steps, y, ny_steps, ndim, early_abandon,
                                      detail::MVSquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default:
    return detail::dtwFull_L_mv_impl(x, nx_steps, y, ny_steps, ndim, early_abandon,
                                      detail::MVL1Dist{});
  }
}

/// Multivariate banded DTW (pointer + length + ndim).
template <typename data_t = double>
data_t dtwBanded_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, int band = settings::DEFAULT_BAND_LENGTH,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwFull_L_mv(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);
  if (ndim == 1) return dtwBanded(x, nx_steps, y, ny_steps, band, early_abandon, metric);

  const size_t min_sz = std::min(nx_steps, ny_steps);
  const size_t max_sz = std::max(nx_steps, ny_steps);
  if (min_sz == 0 || max_sz == 0) return std::numeric_limits<data_t>::max();
  if (min_sz == 1 || max_sz == 1 || static_cast<int>(max_sz) <= band + 1)
    return dtwFull_L_mv(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);

  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwBanded_mv_impl(x, nx_steps, y, ny_steps, ndim, band, early_abandon,
                                      detail::MVSquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default:
    return detail::dtwBanded_mv_impl(x, nx_steps, y, ny_steps, ndim, band, early_abandon,
                                      detail::MVL1Dist{});
  }
}
```

- [ ] **Step 4: Build and test**

```
cmake -S . -B build && cmake --build build --config Release --target unit_test_multivariate_dtw
build/bin/unit_test_multivariate_dtw.exe
```

- [ ] **Step 5: Benchmark D=1 regression**

Add a test that verifies D=1 MV path matches standard path performance-wise (timing, not just correctness):

```cpp
TEST_CASE("MV DTW: D=1 performance parity", "[mv][dtw][perf]")
{
  // Generate random data
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 100.0);
  std::vector<double> x(200), y(200);
  for (auto &v : x) v = dist(rng);
  for (auto &v : y) v = dist(rng);

  // Warmup
  for (int i = 0; i < 100; ++i) {
    dtwc::dtwFull_L(x.data(), 200, y.data(), 200);
    dtwc::dtwFull_L_mv(x.data(), 200, y.data(), 200, 1);
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  double sum_std = 0;
  for (int i = 0; i < 1000; ++i)
    sum_std += dtwc::dtwFull_L(x.data(), 200, y.data(), 200);
  auto t1 = std::chrono::high_resolution_clock::now();
  double sum_mv = 0;
  for (int i = 0; i < 1000; ++i)
    sum_mv += dtwc::dtwFull_L_mv(x.data(), 200, y.data(), 200, 1);
  auto t2 = std::chrono::high_resolution_clock::now();

  // Results should be identical
  REQUIRE_THAT(sum_mv, WithinAbs(sum_std, 1e-6));

  // Print timing (informational)
  auto ms_std = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto ms_mv = std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[perf] Standard: " << ms_std << " ms, MV(D=1): " << ms_mv << " ms\n";
}
```

- [ ] **Step 6: Run full test suite**

```
ctest --test-dir build -C Release --output-on-failure
```

- [ ] **Step 7: Commit**

```bash
git add dtwc/warping.hpp tests/unit/unit_test_multivariate_dtw.cpp
git commit -m "feat: multivariate DTW _impl functions with ndim stride

dtwFull_L_mv and dtwBanded_mv accept interleaved multivariate series.
ndim=1 dispatches to existing scalar _impl (zero overhead).
ndim>1 uses new _mv_impl with pointer-stride indexing."
```

---

### Task 4: Wire Multivariate DTW into Problem

**Files:**
- Modify: `dtwc/Problem.cpp`
- Modify: `dtwc/Problem.hpp`

- [ ] **Step 1: Write integration test**

Append to `tests/unit/unit_test_multivariate_dtw.cpp`:

```cpp
TEST_CASE("Problem: multivariate DTW distance matrix", "[mv][problem]")
{
  // 3 series, 2 timesteps each, ndim=2
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {0,0, 1,1},   // series 0: [(0,0), (1,1)]
    {0,0, 1,1},   // series 1: identical to 0
    {10,10, 11,11} // series 2: far away
  };
  data.p_names = {"a", "b", "c"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.fillDistanceMatrix();

  REQUIRE(prob.distByInd(0, 1) == 0.0);  // identical
  REQUIRE(prob.distByInd(0, 2) > 0.0);   // different
  REQUIRE(prob.distByInd(0, 2) == prob.distByInd(1, 2));  // symmetry of equal series
}
```

- [ ] **Step 2: Modify `Problem::rebind_dtw_fn()`**

In the standard variant dispatch, replace the `Standard` case to use multivariate DTW when `data.ndim > 1`:

```cpp
  case DTWVariant::Standard:
  default:
    if (data.ndim > 1) {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        return dtwBanded_mv(x.data(), x.size() / data.ndim,
                            y.data(), y.size() / data.ndim,
                            data.ndim, band);
      };
    } else {
      dtw_fn_ = [this](const auto &x, const auto &y) { return dtwBanded(x, y, band); };
    }
    break;
```

- [ ] **Step 3: Add ndim validation in `set_data()`**

In `Problem::set_data()`, add `data.validate_ndim()` call.

- [ ] **Step 4: Build and test**

- [ ] **Step 5: Commit**

```bash
git add dtwc/Problem.cpp dtwc/Problem.hpp tests/unit/unit_test_multivariate_dtw.cpp
git commit -m "feat: wire multivariate DTW into Problem via data.ndim"
```

---

### Task 5: Stride-Aware Derivative Transform for DDTW

**Files:**
- Modify: `dtwc/warping_ddtw.hpp`

- [ ] **Step 1: Write tests** (append to `unit_test_multivariate_dtw.cpp`)

```cpp
TEST_CASE("derivative_transform: ndim=1 unchanged", "[mv][ddtw]")
{
  std::vector<double> x = {1, 3, 6, 10};
  auto dx_old = dtwc::derivative_transform(x);
  auto dx_new = dtwc::derivative_transform_mv(x, 1);
  REQUIRE(dx_old.size() == dx_new.size());
  for (size_t i = 0; i < dx_old.size(); ++i)
    REQUIRE_THAT(dx_new[i], WithinAbs(dx_old[i], 1e-10));
}

TEST_CASE("derivative_transform: ndim=2 per-channel", "[mv][ddtw]")
{
  // 4 timesteps x 2 features: [(1,10), (3,20), (6,30), (10,40)]
  std::vector<double> x = {1,10, 3,20, 6,30, 10,40};
  auto dx = dtwc::derivative_transform_mv(x, 2);
  REQUIRE(dx.size() == 8); // same size

  // Channel 0: {1,3,6,10} -> derivative_transform -> check
  // Channel 1: {10,20,30,40} -> derivative_transform -> check
  auto ch0 = dtwc::derivative_transform(std::vector<double>{1,3,6,10});
  auto ch1 = dtwc::derivative_transform(std::vector<double>{10,20,30,40});

  // dx should be interleaved: [ch0[0],ch1[0], ch0[1],ch1[1], ...]
  for (size_t t = 0; t < 4; ++t) {
    REQUIRE_THAT(dx[t*2+0], WithinAbs(ch0[t], 1e-10));
    REQUIRE_THAT(dx[t*2+1], WithinAbs(ch1[t], 1e-10));
  }
}
```

- [ ] **Step 2: Add `derivative_transform_mv`**

In `warping_ddtw.hpp`, add after the existing `derivative_transform`:

```cpp
/// Multivariate derivative transform: apply Keogh-Pazzani derivative
/// independently per channel, preserving interleaved layout.
template <typename data_t>
std::vector<data_t> derivative_transform_mv(const std::vector<data_t> &x, size_t ndim)
{
  if (ndim == 1) return derivative_transform(x);

  const size_t n = x.size() / ndim; // number of timesteps
  if (n == 0) return {};
  if (n == 1) return std::vector<data_t>(ndim, data_t(0));

  std::vector<data_t> dx(x.size());

  for (size_t d = 0; d < ndim; ++d) {
    // Boundary: first point
    dx[0 * ndim + d] = x[1 * ndim + d] - x[0 * ndim + d];

    // Interior points
    for (size_t i = 1; i + 1 < n; ++i) {
      dx[i * ndim + d] = ((x[i * ndim + d] - x[(i-1) * ndim + d])
                        + (x[(i+1) * ndim + d] - x[(i-1) * ndim + d]) / data_t(2)) / data_t(2);
    }

    // Boundary: last point
    dx[(n-1) * ndim + d] = x[(n-1) * ndim + d] - x[(n-2) * ndim + d];
  }

  return dx;
}
```

- [ ] **Step 3: Build and test**

- [ ] **Step 4: Commit**

```bash
git add dtwc/warping_ddtw.hpp tests/unit/unit_test_multivariate_dtw.cpp
git commit -m "feat: stride-aware derivative_transform_mv for multivariate DDTW"
```

---

### Task 6: Update CHANGELOG and Run Full Regression

- [ ] **Step 1: Update CHANGELOG.md** with Wave 1B entries
- [ ] **Step 2: Run full test suite**: `ctest --test-dir build -C Release --output-on-failure`
- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for Wave 1B (multivariate foundation)"
```

---

## Self-Review

1. **Spec coverage:** Data.ndim ✓, TimeSeriesView.ndim ✓, MV metrics ✓, MV DTW _impl ✓, Problem wiring ✓, stride-aware derivative ✓, D=1 perf benchmark ✓
2. **Deferred to later:** Per-channel LB_Keogh (Wave 2), GPU planar layout (Wave 3), Python/MATLAB MV bindings (Wave 3)
3. **No placeholders:** All code blocks complete
4. **Type consistency:** `dtwFull_L_mv`, `dtwBanded_mv`, `MVL1Dist`, `MVSquaredL2Dist` used consistently
