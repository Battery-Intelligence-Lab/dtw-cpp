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

## Modes and implemented routes

`DTWVariantParams::mv_mode` selects one of two meanings:

- `MVMode::Dependent` (the default) uses one warping path shared by all
  channels. Each cell combines the channel costs.
- `MVMode::Independent` computes a separate univariate DTW for each channel
  and sums the results. The `Problem` route supports this mode only for
  Standard DTW with `MissingStrategy::Error`.

For the same band, $$\mathrm{DTW_I} \le \mathrm{DTW_D}$$ follows for additive
L1 and squared-L2 channel costs because each independent channel can choose its
own path. That ordering does not extend to Euclidean `MetricType::L2`, whose
square root couples the channel costs.

The current channel-aware routes are:

| Configuration | Implemented route |
|---------------|-------------------|
| Standard, dependent | `dtwFull_L_mv` / `dtwBanded_mv` |
| Standard, independent | `dtw_independent_mv` |
| DDTW, dependent | per-channel derivative transform followed by Standard multivariate DTW |
| WDTW, dependent | `wdtwFull_mv` / `wdtwBanded_mv` |
| ADTW, dependent | `adtwFull_L_mv` / `adtwBanded_mv` |
| ZeroCost missing data, dependent | `dtwMissing_L_mv` / `dtwMissing_banded_mv` |
| AROW missing data, dependent | the multivariate AROW cost in the `Problem` resolver |

MSM and TWE reject `ndim > 1`. Soft-DTW and the Interpolate missing strategy
do not have channel-aware multivariate routes; do not use those combinations
as multivariate distances.

### Direct C++ calls

The direct functions take interleaved pointers plus explicit timestep counts:

```cpp
#include <dtwc/warping.hpp>
#include <dtwc/warping_adtw.hpp>
#include <dtwc/warping_missing.hpp>
#include <dtwc/warping_wdtw.hpp>

double standard = dtwc::dtwBanded_mv(
    x_ptr, nx_steps, y_ptr, ny_steps, ndim, band);
double weighted = dtwc::wdtwBanded_mv(
    x_ptr, nx_steps, y_ptr, ny_steps, ndim, band, /*g=*/0.05);
double amerced = dtwc::adtwBanded_mv(
    x_ptr, nx_steps, y_ptr, ny_steps, ndim, band, /*penalty=*/0.1);
double zero_cost_missing = dtwc::dtwMissing_banded_mv(
    x_ptr, nx_steps, y_ptr, ny_steps, ndim, band);
```

DDTW applies `derivative_transform_mv_inplace` to the interleaved series and
then calls Standard multivariate DTW. When `ndim == 1`, the `_mv` wrappers
delegate to their scalar implementations.

## Multivariate point costs

The Standard wrappers select these live functors from `warping.hpp`:

| Functor | Pointwise formula |
|---------|-------------------|
| `detail::MVL1Dist` | $$\sum_d \lvert a_d-b_d\rvert$$ |
| `detail::MVSquaredL2Dist` | $$\sum_d (a_d-b_d)^2$$ |
| `detail::MVL2Dist` | $$\sqrt{\sum_d (a_d-b_d)^2}$$ |

The Euclidean implementation is `MVL2Dist`.
The missing-data wrappers use the index-based
`SpanMVNanAwareL1Cost`, `SpanMVNanAwareSquaredL2Cost`, and
`SpanMVNanAwareL2Cost` implementations. The similarly named legacy missing
functors remain direct-call compatibility helpers, not the wrapper dispatch.

## Per-channel LB_Keogh primitives

`compute_envelopes_mv`, `lb_keogh_mv`, and `lb_keogh_mv_squared` are
low-level primitives. They are not wired into `Problem`'s automatic
distance-matrix route.

```cpp
#include <dtwc/core/lower_bound_impl.hpp>

std::vector<double> upper(n_steps * ndim), lower(n_steps * ndim);
dtwc::core::compute_envelopes_mv(
    series_ptr, n_steps, ndim, band, upper.data(), lower.data());

double lb_l1 = dtwc::core::lb_keogh_mv(
    query_ptr, n_steps, ndim, upper.data(), lower.data());
double lb_squared = dtwc::core::lb_keogh_mv_squared(
    query_ptr, n_steps, ndim, upper.data(), lower.data());
```

The implemented bound is scoped to equal-length dependent DTW with additive
L1 or squared-L2 costs. The envelope window must cover the DTW window; an
unbanded DTW therefore needs a full-width admissible envelope. A thresholded
search can reject a candidate when the bound exceeds its cutoff, but an exact
distance matrix still has to compute and store every requested finite
distance.

## `Problem` dispatch

`Problem` validates `Data::ndim` and routes supported combinations:

```cpp
dtwc::Data data(
    {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
     {1.1, 2.1, 3.1, 4.1, 5.1, 6.1},
     {8.0, 7.0, 6.0, 5.0, 4.0, 3.0},
     {8.2, 7.1, 6.2, 5.1, 4.2, 3.1}},
    {"a", "b", "c", "d"},
    3);

dtwc::Problem prob("multivariate");
prob.set_data(std::move(data));
prob.set_n_clusters(2);
prob.cluster();
```

To request independent mode, copy `prob.variant_params`, set its `mv_mode` to
`MVMode::Independent`, and pass it to `prob.set_variant(...)`.

## Python API

The `Data` object must carry `ndim`; passing only two vectors to
`Problem.set_data(series, names)` uses its default `ndim=1`.
Pass the constructed object with `prob.set_data(data)`.

```python
import dtwcpp

series = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
    [8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
    [8.2, 7.1, 6.2, 5.1, 4.2, 3.1],
]
data = dtwcpp.Data(series, ["a", "b", "c", "d"], 3)
data.validate_ndim()

prob = dtwcpp.Problem("multivariate")
prob.set_data(data)
prob.set_n_clusters(2)
result = dtwcpp.fast_pam(prob, n_clusters=2)

n_steps = data.series_length(0)  # 2, because each timestep has 3 channels
```
