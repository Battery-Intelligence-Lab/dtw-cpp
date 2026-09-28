---
title: Distance Metrics
weight: 5
---

# Distance Metrics

DTWC++ has three runtime pointwise metric selectors in C++:
`MetricType::L1`, `MetricType::L2`, and `MetricType::SquaredL2`. Huber is not
implemented.

## Available metrics

| Metric | Scalar cost | Multivariate cost | Notes |
|---|---:|---:|---|
| L1 (default) | $$|x_i-y_j|$$ | $$\sum_c |x_{i,c}-y_{j,c}|$$ | Absolute-difference sum. |
| L2 | $$\sqrt{(x_i-y_j)^2}$$ | $$\sqrt{\sum_c (x_{i,c}-y_{j,c})^2}$$ | For one channel this equals L1; it differs for multivariate input. |
| Squared L2 | $$(x_i-y_j)^2$$ | $$\sum_c (x_{i,c}-y_{j,c})^2$$ | Emphasizes large pointwise deviations; the accumulated result is not itself a metric distance. |

The C++ runtime API selects among all three. `Problem::set_metric` makes the
metric part of a `Problem`'s distance semantics — its CPU and GPU fills, lazy
lookups, mmap cache and checkpoint identities, and FastCLARA's samples. A metric
other than L1 is implemented for Standard DTW with `MissingStrategy::Error`
(univariate or multivariate); with another variant or a missing-data strategy
it raises `InvalidInput`, because the `Problem` passes the metric to the
Standard kernels only. The CLI's `--metric` is the same `Problem`
setting: `l1` or `squared_euclidean` on `--device cpu` and `--device gpu` alike.
A combination the kernels do not implement fails before computation rather than
silently using another metric.

## Lower bounds and pruning

Lower bounds can avoid exact distance work only when the consumer needs a
threshold or nearest-neighbour decision rather than every exact pair value.
TADPole uses this property: a bound relative to its cutoff can classify some
pairs without computing DTW. An exact full matrix needs every pair, so its
fill uses no lower bound.

### LB_Keogh

LB_Keogh constructs an envelope around one series and measures how far the
other lies outside it. For a fixed DTW radius `w`, the envelope radius `r`
must satisfy `r >= w`; a wider envelope is valid but weaker. Full DTW requires
an envelope that repeats the candidate's global minimum and maximum; a
negative band passed to the low-level envelope helpers requests exactly that
envelope (FX-13).

The admissibility statements in this section require finite input samples and
ordered finite envelope bounds. Missing-value policies and non-finite data are
outside the D2 proof.

The bound is L1 and has amplitude units `U`; TADPole applies it only to
univariate L1 Standard DTW.

For a feasible fixed window (`w >= |n-m|`), a directional bound may sum only
the first `min(n,m)` rows and remains admissible: every included row can be
charged to one distinct path cell. The symmetric bound is the maximum of the
two directional bounds, not their sum. The complete proof and executable
oracle are in the
[D2 derivation](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/derivations/02-envelopes-lb-keogh.md).

## Choosing a metric

- Use **L1** for the default CPU/CLI path and absolute-deviation costs.
- Use **L2** when multivariate Euclidean point costs are required through the
  C++ runtime API.
- Use **Squared L2** when large pointwise deviations should receive a quadratic
  penalty, or for the supported CUDA CLI path.

Metric choice and elastic-distance choice are separate. MSM and TWE are
standalone elastic metrics with their own recurrences and parameters; they are
documented under [DTW variants](../dtw-variants/).
