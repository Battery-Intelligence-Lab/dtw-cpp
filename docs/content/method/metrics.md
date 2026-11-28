---
title: Distance Metrics
weight: 5
---

# Distance Metrics

DTWC++ has three pointwise metrics, `MetricType::L1` (`l1`), `MetricType::L2`
(`euclidean`) and `MetricType::SquaredL2` (`squared_euclidean`); the names in
brackets are the ones the CLI, Python and MATLAB take. Huber is not
implemented.

## Available metrics

| Metric | Scalar cost | Multivariate cost | Notes |
|---|---:|---:|---|
| L1 (default) | $|x_i-y_j|$ | $\sum_c |x_{i,c}-y_{j,c}|$ | Absolute-difference sum. |
| L2 | $\sqrt{(x_i-y_j)^2}$ | $\sqrt{\sum_c (x_{i,c}-y_{j,c})^2}$ | For one channel this equals L1; it differs for multivariate input, except in independent mode, which runs a univariate DTW on each channel. |
| Squared L2 | $(x_i-y_j)^2$ | $\sum_c (x_{i,c}-y_{j,c})^2$ | Emphasizes large pointwise deviations; the accumulated result is not itself a metric distance. |

Every front end selects among all three. `Problem::set_metric` makes the
metric part of a `Problem`'s distance semantics — its CPU and GPU fills, its
bound distance function, mmap cache and checkpoint identities, and FastCLARA's
samples. A metric other than L1 is implemented for Standard DTW, with or
without a missing-data strategy, and for DDTW (univariate or multivariate; L2
with AROW univariate only). WDTW, ADTW, Soft-DTW, MSM and TWE compute L1, so
another metric with them raises `InvalidInput`. `dtwc::distance::dtw` and a
`Problem` apply this one rule (`core::validate`). The CLI's `--metric` is the
same `Problem` setting: `l1`, `euclidean` or `squared_euclidean` on `--device cpu`
and `--device gpu` alike (the GPU runs Standard DTW on univariate series only,
where `euclidean` is the L1 kernel). A combination the
kernels do not implement fails before computation rather than silently using
another metric.

## Lower bounds and pruning

Lower bounds can avoid exact distance work only when the consumer needs a
threshold or nearest-neighbour decision rather than every exact pair value.
TADPole uses this property: a bound relative to its cutoff can classify some
pairs without computing DTW. An exact full matrix needs every pair, so its
fill uses no lower bound.

### LB_Keogh

LB_Keogh constructs an envelope around candidate series $c$ and measures how far query series $a$ lies outside it (`dtwc/core/lower_bound_impl.hpp`). For a fixed DTW radius $w$, the envelope radius $r$ must satisfy $r \ge w$; a wider envelope is valid but weaker. Full DTW requires an envelope that repeats the candidate's global minimum and maximum ($r \ge m$); a negative band passed to `compute_envelopes` / `compute_envelope` builds that global envelope:

$$U_i = \max_{j \in [\max(0, i-r),\; \min(m-1, i+r)]} c_j, \qquad L_i = \min_{j \in [\max(0, i-r),\; \min(m-1, i+r)]} c_j.$$

Over the first $n_{\mathrm{cmp}} = \min(|a|, |c|)$ positions, the directional lower bound sums each excursion outside $[L_i, U_i]$:

$$\mathrm{LB\_Keogh}(a; L, U) = \sum_{i=0}^{n_{\mathrm{cmp}}-1} \delta(a_i; L_i, U_i), \qquad \delta(a; L, U) = \begin{cases} a - U, & a > U, \\ L - a, & a < L, \\ 0, & L \le a \le U. \end{cases}$$

The admissibility statements in this section require finite input samples and
ordered finite envelope bounds. Missing-value policies and non-finite data are
outside the D2 proof.

The bound is L1 and has amplitude units `U`; TADPole applies it only to
univariate L1 Standard DTW.

For a feasible fixed window ($w \ge |n-m|$), summing over the first $\min(n,m)$ rows remains admissible because every included row can be charged to one distinct path cell. The symmetric bound is the maximum of the two directional bounds, not their sum:

$$\mathrm{LB\_Keogh}_{\mathrm{sym}}(x, y) = \max\big(\mathrm{LB\_Keogh}(x; L_y, U_y),\; \mathrm{LB\_Keogh}(y; L_x, U_x)\big).$$

For threshold tests (`core::lb_keogh_at_least` and `core::lb_keogh_greater`), DTWC++ evaluates $\mathrm{LB\_Keogh}(x; L_y, U_y)$ first and short-circuits before running the second pass whenever the first direction already reaches (or strictly exceeds) the threshold. The complete proof and executable oracle are in the
[D2 derivation](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/derivations/02-envelopes-lb-keogh.md).

### LB_Kim (two-endpoint $O(1)$ bound)

Before scanning linear envelopes, TADPole checks the two-endpoint $O(1)$ L1 bound (`lb_kim_l1` in `dtwc/algorithms/tadpole.cpp`) on equal-length pairs because endpoint preservation forces $x_0$ to match $y_0$ and $x_{n-1}$ to match $y_{n-1}$:

$$\mathrm{LB\_Kim}_{\mathrm{L1}}(x, y) = \begin{cases} |x_0 - y_0| + |x_{n-1} - y_{n-1}|, & n \ge 2, \\ |x_0 - y_0|, & n = 1. \end{cases}$$

Kim et al.'s full four-feature bound is not used, and stronger bounds such as `LB_Improved` and `BGLB` are not implemented in DTWC++.

## Choosing a metric

- Use **L1** for the default CPU/CLI path and absolute-deviation costs.
- Use **L2** (`--metric euclidean`) when multivariate Euclidean point costs are
  required; on univariate series it is L1.
- Use **Squared L2** (`--metric squared_euclidean`) when large pointwise deviations
  should receive a quadratic penalty; the CPU and the GPU take it alike.

Metric choice and elastic-distance choice are separate. MSM and TWE are
standalone elastic metrics with their own recurrences and parameters; they are
documented under [DTW variants](../dtw-variants/).
