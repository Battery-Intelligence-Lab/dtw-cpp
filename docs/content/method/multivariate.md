---
title: Multivariate Time Series
weight: 8
---

# Multivariate Time Series

DTW-C++ supports multi-dimensional (multivariate) time series, where each timestep has $$D$$ features. This is common in sensor fusion, motion capture, multi-channel physiological signals, and other domains where multiple measurements are recorded simultaneously.

## Data Layout

Multivariate series use an **interleaved** (row-major) memory layout. For a series with $$n$$ timesteps and $$D$$ dimensions, the flat buffer contains $$n \times D$$ elements:

```
[x1_dim1, x1_dim2, ..., x1_dimD, x2_dim1, x2_dim2, ..., x2_dimD, ...]
```

The value at timestep $$t$$ and dimension $$d$$ is at index $$t \times D + d$$.

### Data struct

The `Data` struct stores multivariate series via its `ndim` field:

```cpp
dtwc::Data data;
data.ndim = 3;  // 3-dimensional time series

// Each series in p_vec has flat_size = n_timesteps * ndim elements
data.p_vec.push_back({1.0, 2.0, 3.0,   // timestep 0: [1, 2, 3]
                      4.0, 5.0, 6.0,   // timestep 1: [4, 5, 6]
                      7.0, 8.0, 9.0}); // timestep 2: [7, 8, 9]
data.p_names.push_back("series_0");

data.validate_ndim();  // ensures all series sizes are divisible by ndim
```

The `series_length(i)` method returns the number of timesteps (not the flat size):

```cpp
size_t n_steps = data.series_length(0);  // 3 (not 9)
```

### TimeSeriesView

The `TimeSeriesView` struct provides a lightweight, non-owning reference to contiguous time series data:

```cpp
dtwc::core::TimeSeriesView<double> view;
view.data = buffer_ptr;
view.length = n_timesteps;
view.ndim = 3;

// Access timestep i (returns pointer to ndim elements)
const double* step_i = view.at(i);

// Total scalar count
size_t flat = view.flat_size();  // length * ndim
```

---

## Multivariate DTW Functions

All DTW variants have `_mv` counterparts that operate on interleaved multivariate data. These functions take raw pointers and explicit timestep counts:

### Standard DTW

```cpp
#include <dtwc/warping.hpp>

double dist = dtwc::dtwFull_L_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim);
double dist_b = dtwc::dtwBanded_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim, band);
```

### WDTW

```cpp
#include <dtwc/warping_wdtw.hpp>

double dist = dtwc::wdtwFull_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim, /*g=*/0.05);
double dist_b = dtwc::wdtwBanded_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim, band, /*g=*/0.05);
```

### ADTW

```cpp
#include <dtwc/warping_adtw.hpp>

double dist = dtwc::adtwFull_L_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim, /*penalty=*/0.1);
double dist_b = dtwc::adtwBanded_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim, band, /*penalty=*/0.1);
```

### Missing Data DTW

```cpp
#include <dtwc/warping_missing.hpp>

double dist = dtwc::dtwMissing_L_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim);
double dist_b = dtwc::dtwMissing_banded_mv(x_ptr, nx_steps, y_ptr, ny_steps, ndim, band);
```

### DDTW (Derivative Transform)

```cpp
#include <dtwc/warping_ddtw.hpp>

// Transform multivariate series, preserving interleaved layout
auto dx = dtwc::derivative_transform_mv(flat_data, ndim);
// Then use standard multivariate DTW on the derivative series
```

---

## Zero Overhead for Univariate

When `ndim == 1`, all `_mv` functions dispatch to the existing scalar code paths, avoiding any overhead from the multivariate inner loop. You can safely use the multivariate API for mixed workloads without performance penalty on univariate data.

---

## Multivariate Distance Functors

The inner loop of multivariate DTW uses specialized distance functors defined in `warping.hpp`:

| Functor | Formula | Header |
|---------|---------|--------|
| `detail::MVL1Dist` | $$\sum_{d=0}^{D-1} \lvert a[d] - b[d] \rvert$$ | `warping.hpp` |
| `detail::MVSquaredL2Dist` | $$\sum_{d=0}^{D-1} (a[d] - b[d])^2$$ | `warping.hpp` |
| `detail::MissingMVL1Dist` | L1 with per-channel NaN skipping | `warping_missing.hpp` |
| `detail::MissingMVSquaredL2Dist` | Squared-L2 with per-channel NaN skipping | `warping_missing.hpp` |

These are invoked automatically based on the `MetricType` selection.

---

## Per-Channel LB_Keogh

DTW-C++ provides multivariate lower-bound pruning via per-channel envelopes, which are valid lower bounds for dependent multivariate DTW (where all channels share a single warping path).

```cpp
#include <dtwc/core/lower_bound_impl.hpp>

// Compute upper and lower envelopes for each channel independently
std::vector<double> upper(n_steps * ndim), lower(n_steps * ndim);
dtwc::core::compute_envelopes_mv(series_ptr, n_steps, ndim, band,
                                  upper.data(), lower.data());

// Compute multivariate LB_Keogh (L1 variant)
double lb = dtwc::core::lb_keogh_mv(query_ptr, n_steps, ndim,
                                      upper.data(), lower.data());

// Squared-L2 variant
double lb_sq = dtwc::core::lb_keogh_mv_squared(query_ptr, n_steps, ndim,
                                                 upper.data(), lower.data());
```

The lower bound satisfies $$\text{LB\_Keogh}(q, s) \le \text{DTW}(q, s)$$ for any query $$q$$ and candidate $$s$$ under the given band constraint. This allows pruning candidate pairs without computing the full DTW, which is especially valuable for large multivariate datasets.

---

## Auto-Dispatch in Problem

When using the `Problem` class, setting `data.ndim > 1` causes `Problem::rebind_dtw_fn()` to automatically select the appropriate multivariate DTW function. No manual function selection is needed:

```cpp
dtwc::Data data;
data.ndim = 3;
// ... populate data.p_vec with interleaved series ...

dtwc::Problem prob;
prob.set_data(std::move(data));
prob.set_numberOfClusters(5);
prob.cluster();  // automatically uses multivariate DTW
```

---

## Python API

```python
import dtwcpp

# Create Data with ndim > 1
data = dtwcpp.Data()
data.ndim = 3

# Each series is a flat list: [t0_ch0, t0_ch1, t0_ch2, t1_ch0, ...]
series_0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2 timesteps, 3 channels
data.p_vec = [series_0, series_1, ...]
data.p_names = ["s0", "s1", ...]
data.validate_ndim()  # check all sizes are divisible by ndim

# Use with Problem for clustering
prob = dtwcpp.Problem()
prob.set_data(data.p_vec, data.p_names)
# Problem auto-dispatches to multivariate DTW when ndim > 1
```

The `Data.series_length(i)` method returns the number of timesteps:

```python
n_steps = data.series_length(0)  # flat_size / ndim
```
