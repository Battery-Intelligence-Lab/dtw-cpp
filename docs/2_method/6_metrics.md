---
layout: default
title: Distance Metrics
nav_order: 6
---

# Distance Metrics

DTW-C++ supports multiple pointwise distance metrics for use within the DTW computation. The pointwise metric determines how the cost of aligning two individual time series points is calculated.

## Available Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| L1 (default) | $$\|x_i - y_j\|$$ | Absolute difference. Robust to outliers. |
| L2 | $$\sqrt{(x_i - y_j)^2}$$ | Euclidean distance between points. |
| Squared L2 | $$(x_i - y_j)^2$$ | Squared Euclidean. Emphasizes large differences. |
| Huber | See below | Quadratic for small errors, linear for large. |

### Huber Metric

The Huber metric provides a compromise between L1 and L2:

$$
d_\delta(x_i, y_j) = \begin{cases}
\frac{1}{2}(x_i - y_j)^2 & \text{if } |x_i - y_j| \le \delta \\
\delta \left(|x_i - y_j| - \frac{1}{2}\delta\right) & \text{otherwise}
\end{cases}
$$

where $$\delta$$ is the Huber threshold parameter.

## Lower Bound Pruning

Lower bounds allow skipping full DTW computations when the lower bound already exceeds a known upper bound, significantly accelerating distance matrix construction.

### LB_Keogh

LB_Keogh computes a lower bound on the DTW distance by constructing an envelope around one series and measuring how much the other series falls outside that envelope. It requires a Sakoe-Chiba band constraint.

#### LB_Keogh Compatibility by Metric

| Metric | LB_Keogh Supported | Notes |
|--------|-------------------|-------|
| L1 | Yes | Envelope-based bound holds for L1. |
| L2 | Yes | Envelope-based bound holds for L2. |
| Squared L2 | Yes | Envelope-based bound holds for squared L2. |
| Huber | No | The Huber metric does not satisfy the envelope-based lower bound property in general. Because the Huber function transitions between quadratic and linear regimes depending on the magnitude of each pointwise difference, the envelope-based bound can overestimate the true DTW cost in the linear regime, violating the lower bound guarantee. |

### LB_Kim

LB_Kim is a simpler O(1) lower bound based on comparing the first, last, minimum, and maximum values of the two series. It requires a monotone pointwise metric (valid for L1, L2, Squared L2) and typically provides a looser bound than LB_Keogh.

## Choosing a Metric

- **L1 (default):** Good general-purpose choice. More robust to outliers than L2 or squared L2.
- **L2:** Standard Euclidean pointwise distance. Common in the literature.
- **Squared L2:** Penalizes large misalignments more heavily. Useful when large deviations are especially undesirable.
- **Huber:** Best when data contains occasional outliers but you still want sensitivity to moderate differences. Note that LB_Keogh pruning is not available with the Huber metric.
