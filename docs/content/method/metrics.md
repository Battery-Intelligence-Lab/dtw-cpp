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
other lies outside it. For a fixed DTW radius `w`, the envelope radius `r`
must satisfy `r >= w`; a wider envelope is valid but weaker. Full DTW requires
an envelope that repeats the candidate's global minimum and maximum. Passing a
negative band directly to the current low-level envelope helper instead
constructs a radius-zero envelope, which is not generally admissible for full
DTW (F46).

The admissibility statements in this section require finite input samples and
ordered finite envelope bounds. Missing-value policies and non-finite data are
outside the D2 proof.

The L1 (and scalar L2) bound has amplitude units `U`; the unrooted squared-L2
bound squares each excess and has units `U^2`. The implementation provides
separate L1 and squared-excess primitives. The scalar L2 selector has the same
point cost as L1; this statement does not extend to multivariate Euclidean L2.

For a feasible fixed window (`w >= |n-m|`), a directional bound may sum only
the first `min(n,m)` rows and remains admissible: every included row can be
charged to one distinct path cell. The symmetric bound is the maximum of the
two directional bounds, not their sum. The complete proof and executable
oracle are in the
[D2 derivation](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/derivations/02-envelopes-lb-keogh.md).

### LB_Kim

LB_Kim is an O(1) bound derived from endpoint and range summaries. It is valid
for the current scalar L1 cost (and scalar L2, which is identical to L1).
Although the public trait currently advertises squared-L2 compatibility, the
implementation still returns raw absolute feature differences. That result is
not an admissible squared-L2 bound; the mismatch is tracked as F47. LB_Kim is
generally looser than LB_Keogh in its valid regime.

### LB_Enhanced and LB_Webb

The C++ lower-bound library implements LB_Enhanced and retains `lb_webb` and
the `Webb` strategy as historical public names. The local Webb implementation
is the paper's all-index `LB_Webb_NoLR` bridge and corrections plus a separate
conservative trailing-flag cap. It is not full Algorithm 2, which includes
`MinLRPaths`, and no universal ordering between the local variant and full
Webb is claimed.

Inside the confirmed finite, nonempty, equal-length scalar L1 or unrooted
squared-L2 domain, the envelope, lower bound, and DTW must use the same
saturated window. The local directional Webb result is at least the
matching-direction LB_Keogh; taking the maximum of both directions therefore
dominates symmetric Keogh. The tail-cap proof is a different statement:
the production result is no greater than exact-predicate NoLR and remains
admissible.

For effective `V=1`, directional LB_Enhanced dominates matching-direction
Keogh. For effective `V>=2`, neither dominates: the D3 exact oracle contains
strict witnesses in both directions, so the `Enhanced` cascade evaluates
their maximum. Custom point costs, nonfinite inputs, mutable envelope
shape/provenance, and last-ULP threshold decisions are outside this confirmed
contract. The complete assumptions, proofs, witnesses, and code-conformance
map are in the
[D3 derivation](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/derivations/03-lb-enhanced-webb.md).

## Choosing a metric

- Use **L1** for the default CPU/CLI path and absolute-deviation costs.
- Use **L2** when multivariate Euclidean point costs are required through the
  C++ runtime API.
- Use **Squared L2** when large pointwise deviations should receive a quadratic
  penalty, or for the supported CUDA CLI path.

Metric choice and elastic-distance choice are separate. MSM and TWE are
standalone elastic metrics with their own recurrences and parameters; they are
documented under [DTW variants](../dtw-variants/).
