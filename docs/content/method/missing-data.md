---
title: Missing Data
weight: 7
---

# Missing Data

DTW-C++ supports time series with missing values, represented as NaN. Missing data is common in real-world sensor recordings, medical time series, and IoT data streams. The library provides multiple strategies for handling NaN values during DTW computation.

## Missing Strategy

The handling of NaN values is controlled by the `MissingStrategy` enum (defined in `core/dtw_options.hpp`):

| Strategy | Behavior | Use case |
|----------|----------|----------|
| `Error` | Throws a `std::runtime_error` if NaN is encountered (default); the distance functions throw its subclass `InvalidInput`, naming the series and position | Strict mode; data should be clean |
| `ZeroCost` | NaN pairs contribute zero cost to the warping path | Tolerant alignment through gaps |
| `AROW` | Diagonal-only alignment when NaN is encountered | Prevents free stretching through gaps |
| `Interpolate` | Linear interpolation preprocessing, then standard DTW | Smooth gap filling before comparison |

NaN is the only missing-value marker; ±inf never is. The checked entry points — `dtwc::distance::*` (including `distance::missing` and `distance::arow`), Python's `dtwcpp.distance.dtw` and MATLAB's `dtwc.distance.dtw` — reject ±inf under every strategy, raising `InvalidInput` naming the series and the 0-based position, for example `distance::missing: x[2] is +inf`. The `dtwMissing_*` and `dtwAROW_*` wrappers below are the unchecked per-pair layer (see [DTW](../dtw/)): pass them finite values and NaN only.

## ZeroCost DTW

**Reference:** Yurtman, Soenen, Meert & Blockeel, "Estimating DTW Distance Between Time Series with Missing Data," ECML-PKDD 2023, LNCS 14173.

When either $$x[i]$$ or $$y[j]$$ is NaN, the pointwise cost is set to zero. The warping path can pass through missing regions without penalty. The recurrence is identical to standard DTW:

$$C(i,j) = \text{cost}(x[i], y[j]) + \min\big(C(i-1,j-1),\; C(i-1,j),\; C(i,j-1)\big)$$

where:

$$\text{cost}(a, b) = \begin{cases} 0 & \text{if } a \text{ or } b \text{ is NaN} \\ d(a, b) & \text{otherwise} \end{cases}$$

### C++ API

```cpp
#include <dtwc/warping_missing.hpp>

std::vector<double> x = {1.0, NAN, 3.0, 4.0};
std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

// Checked: ±inf raises InvalidInput (dtwc/distance.hpp)
double dist_checked = dtwc::distance::missing(x, y, /*band=*/-1);

// Linear-space (O(min(m,n)) memory)
double dist = dtwc::dtwMissing_L(x, y);

// With early abandon threshold
double dist_ea = dtwc::dtwMissing_L(x, y, /*early_abandon=*/5.0);

// Banded
double dist_b = dtwc::dtwMissing_banded(x, y, /*band=*/3);

// With squared-L2 metric
double dist_sq = dtwc::dtwMissing_L(x, y, -1.0, dtwc::core::MetricType::SquaredL2);
```

### Pointer + length overloads

Zero-copy overloads are available for binding and performance use:

```cpp
double dist = dtwc::dtwMissing_L(x_ptr, nx, y_ptr, ny);
double dist_b = dtwc::dtwMissing_banded(x_ptr, nx, y_ptr, ny, /*band=*/5);
```

---

## DTW-AROW

**Reference:** Yurtman, Soenen, Meert & Blockeel, ECML-PKDD 2023 (same paper as ZeroCost).

DTW-AROW (Adaptive Restriction of Warping) restricts the warping path to the **diagonal direction only** when NaN is encountered, enforcing one-to-one alignment through missing regions. This prevents the "free stretching" problem where standard DTW might exploit missing values by matching many observed points to a single missing point at zero cost.

The recurrence is:

$$C(i,j) = \begin{cases} C(i-1,j-1) & \text{if } x[i] \text{ or } y[j] \text{ is NaN (diagonal only, zero cost)} \\ \min(C(i-1,j-1),\; C(i-1,j),\; C(i,j-1)) + d(x[i], y[j]) & \text{otherwise} \end{cases}$$

### C++ API

```cpp
#include <dtwc/warping_missing_arow.hpp>

std::vector<double> x = {1.0, NAN, NAN, 4.0, 5.0};
std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

// Checked: ±inf raises InvalidInput (dtwc/distance.hpp)
double dist_checked = dtwc::distance::arow(x, y, /*band=*/-1);

// Linear-space (O(min(m,n)) memory)
double dist = dtwc::dtwAROW_L(x, y);

// Banded
double dist_b = dtwc::dtwAROW_banded(x, y, /*band=*/3);

// With squared-L2 metric
double dist_sq = dtwc::dtwAROW_L(x, y, dtwc::core::MetricType::SquaredL2);
```

---

## Interpolation

The `interpolate_linear_into()` function fills NaN gaps before DTW computation, into a buffer the caller keeps; a
series without NaN comes back as itself, without a copy:

- **Interior NaN:** linearly interpolated between the nearest observed neighbors on each side.
- **Leading NaN:** filled with the first observed value (Next Observation Carried Backward, NOCB).
- **Trailing NaN:** filled with the last observed value (Last Observation Carried Forward, LOCF).
- **All-NaN input:** throws `dtwc::InvalidInput` (Python `dtwcpp.InvalidInput`, a `ValueError`).

```cpp
#include <dtwc/base/missing_utils.hpp>

std::vector<double> v = {NAN, 1.0, NAN, NAN, 4.0, NAN};
std::vector<double> buffer;
std::span<const double> filled = dtwc::interpolate_linear_into(v, buffer);
// filled views buffer: {1.0, 1.0, 2.0, 3.0, 4.0, 4.0}
```

When `MissingStrategy::Interpolate` is set on a `Problem`, interpolation is applied automatically as a preprocessing step before standard DTW.

---

## Bitwise NaN Check

The `is_missing()` function uses raw bit inspection to detect NaN values, making it safe under aggressive floating-point optimization flags (`-ffast-math`, `/fp:fast`) where `std::isnan()` may be optimized away by the compiler.

```cpp
#include <dtwc/base/missing_utils.hpp>

double val = NAN;
bool m = dtwc::is_missing(val);  // true (safe under -ffast-math)
bool s = std::isnan(val);         // may be false under -ffast-math!
```

Additional utility:

```cpp
bool any_nan = dtwc::has_missing(series);      // true if any NaN present
```

---

## Multivariate Missing Data

Missing data support extends to multivariate time series with per-channel NaN handling. If channel $$d$$ at timestep $$i$$ is NaN, only that channel is skipped; other channels at the same timestep still contribute normally.

```cpp
#include <dtwc/warping_missing.hpp>

// Interleaved layout: [x1_ch1, x1_ch2, x2_ch1, x2_ch2, ...]
double dist = dtwc::dtwMissing_L_mv(x_ptr, nx_steps, y_ptr, ny_steps, /*ndim=*/3);
double dist_b = dtwc::dtwMissing_banded_mv(x_ptr, nx_steps, y_ptr, ny_steps, /*ndim=*/3, /*band=*/5);
```

When `ndim == 1`, the multivariate functions dispatch to the faster scalar implementations.

---

## Python API

```python
import numpy as np
import dtwcpp

x = np.array([1.0, np.nan, 3.0, 4.0])
y = np.array([1.0, 2.0, 3.0, 4.0])

# ZeroCost DTW
dist = dtwcpp.distance.dtw(x, y, missing_strategy="zero_cost")

# DTW-AROW
dist_arow = dtwcpp.distance.dtw(x, y, missing_strategy="arow")

# With squared Euclidean metric
dist_sq = dtwcpp.distance.dtw(x, y, missing_strategy="zero_cost", metric="squared_euclidean")

# Via Problem (for clustering with missing data)
prob = dtwcpp.Problem()
prob.set_data(series_with_nans, names)
prob.missing_strategy = dtwcpp.MissingStrategy.ZeroCost  # or AROW, Interpolate
```

## CLI

```bash
dtwc_cl --input data.tsv --n-clusters 3 \
  --missing-strategy zero_cost --variant standard
```

Accepted values are `error` (default), `zero_cost`, `arow`, and `interpolate`.
The same `missing-strategy` key is available in TOML configuration.
Non-error missing handling cannot be combined with a non-standard DTW variant,
because that combination would otherwise substitute the missing-data recurrence
for the requested variant.

