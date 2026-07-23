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

The C++ runtime API selects among all three. The 2.0 CLI surface is narrower:
CPU accepts `--metric l1`; CUDA accepts `l1` or `squared_euclidean`. Unsupported
device/metric combinations fail before computation rather than silently using
another metric.

## Lower bounds and pruning

Lower bounds can avoid exact distance work only when the consumer needs a
threshold or nearest-neighbour decision rather than every exact pair value.
TADPole uses this property: a bound relative to its cutoff can classify some
pairs without computing DTW.

The legacy `DistanceMatrixStrategy::Pruned` route still produces an exact full
matrix. Its early-abandon kernel returns a sentinel, so an abandoned pair is
then recomputed to recover the exact value. That route is correct but is not
documented as an acceleration; the registered LB-cascade experiment found the
extra partial-plus-full work to be a pessimization on its fixture.

### LB_Keogh

LB_Keogh constructs an envelope around one series and measures how far the
other lies outside it. The implementation requires a Sakoe–Chiba band and has
valid L1, L2, and Squared L2 specializations.

### LB_Kim

LB_Kim is an O(1) bound derived from endpoint and range summaries. It is valid
for the same three pointwise metrics and is generally looser than LB_Keogh.

### LB_Enhanced and LB_Webb

The C++ lower-bound library also implements LB_Enhanced and LB_Webb for
supported metrics. Their validity and the registered relationship
`LB_Webb >= LB_Keogh` are exercised by the adversarial lower-bound suite.

## Choosing a metric

- Use **L1** for the default CPU/CLI path and absolute-deviation costs.
- Use **L2** when multivariate Euclidean point costs are required through the
  C++ runtime API.
- Use **Squared L2** when large pointwise deviations should receive a quadratic
  penalty, or for the supported CUDA CLI path.

Metric choice and elastic-distance choice are separate. MSM and TWE are
standalone elastic metrics with their own recurrences and parameters; they are
documented under [DTW variants](../dtw-variants/).
