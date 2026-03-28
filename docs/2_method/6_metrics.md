---
layout: default
title: Distance Metrics
nav_order: 6
---

# Distance Metrics

The DTW algorithm requires a **pointwise distance metric** to compute the cost of aligning element $$x_i$$ of one time series with element $$y_j$$ of another. The choice of metric affects both the meaning of the resulting DTW distance and the applicability of lower-bound pruning techniques.

DTW-C++ provides several built-in metrics in the `dtwc::core` namespace.

## L1 (Manhattan)

The L1 metric computes the absolute difference between two values:

$$
d_{\text{L1}}(a, b) = |a - b|
$$

This is the default metric used by DTW-C++. It is robust to outliers compared to squared metrics and is compatible with LB_Keogh lower-bound pruning.

**Implementation:** `L1Metric`

## Squared L2

The squared L2 metric computes the squared difference:

$$
d_{\text{sqL2}}(a, b) = (a - b)^2
$$

This is the metric used in the classical DTW formulation. It penalises large deviations more heavily than L1 and is compatible with LB_Keogh lower-bound pruning.

Using squared L2 rather than L2 avoids computing a square root for every cell in the cost matrix, which is more efficient. When comparing DTW distances computed with this metric, note that the resulting distance is the sum of squared differences along the warping path (not the Euclidean distance).

**Implementation:** `SquaredL2Metric`

## L2 (Euclidean)

The L2 metric computes the Euclidean distance between two values:

$$
d_{\text{L2}}(a, b) = \sqrt{(a - b)^2} = |a - b|
$$

For scalar (univariate) time series, the L2 metric is equivalent to the L1 metric. The L2 metric becomes distinct from L1 in the multivariate case, where the per-cell cost is the Euclidean norm of the difference vector. For univariate series, prefer `SquaredL2Metric` for efficiency.

**Implementation:** `L2Metric`

## LB_Keogh compatibility

[LB_Keogh](https://dl.acm.org/doi/10.5555/1367985.1367993) is a lower-bound technique that can prune unnecessary DTW computations when building the distance matrix. If the lower bound between two series exceeds a known best distance, the full DTW computation can be skipped.

Not all metrics are compatible with LB_Keogh pruning. The following table summarises compatibility:

| Metric | LB_Keogh compatible | Notes |
|--------|---------------------|-------|
| L1 (Manhattan) | Yes | Default metric |
| Squared L2 | Yes | Classical DTW metric |
| L2 (Euclidean) | Yes | Equivalent to L1 for univariate |
| Cosine (planned) | No | Requires vector normalisation |
| Huber (planned) | No | Non-quadratic penalty |

Compatibility is encoded at compile time via the `lb_keogh_valid<Metric>` trait, allowing the distance matrix builder to automatically enable or disable LB_Keogh pruning based on the chosen metric.

## Choosing a metric

* **For most applications**, use `SquaredL2Metric` (the classical DTW choice) or `L1Metric` (more robust to outliers).
* **When LB_Keogh pruning is important** (large datasets), ensure the chosen metric is compatible.
* **When outlier robustness matters**, prefer L1 over squared L2, as the squared term amplifies large deviations.

## Adding a custom metric

A metric is any callable that satisfies:

```cpp
T operator()(T a, T b) const;
```

To add a new metric, define a struct with this call operator. If the metric is compatible with LB_Keogh, specialise the `lb_keogh_valid` trait:

```cpp
template <>
inline constexpr bool lb_keogh_valid<MyMetric> = true;
```
