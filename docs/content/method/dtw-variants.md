---
title: DTW Variants
weight: 6
---

# DTW Variants

DTW-C++ supports multiple Dynamic Time Warping variants beyond standard DTW. Each variant modifies the distance computation in a different way, targeting different aspects of time series comparison: shape sensitivity, warping regularization, or differentiability.

## DDTW (Derivative DTW)

**Reference:** Keogh & Pazzani, "Derivative Dynamic Time Warping," SIAM SDM, 2001.

DDTW applies a derivative transform to both series as a preprocessing step, then runs standard DTW on the derivative series. This makes the comparison sensitive to **shape** (local slopes) rather than raw amplitude.

The derivative formula for interior points is:

$$x'[i] = \frac{(x[i] - x[i-1]) + (x[i+1] - x[i-1]) / 2}{2}, \quad 1 \le i \le n-2$$

Boundary cases use simple forward/backward differences:

$$x'[0] = x[1] - x[0], \quad x'[n-1] = x[n-1] - x[n-2]$$

### C++ API

```cpp
#include <dtwc/warping_ddtw.hpp>

std::vector<double> x = {1.0, 2.0, 3.0, 2.0, 1.0};
std::vector<double> y = {1.0, 1.5, 3.0, 2.5, 1.0};

// Full (unconstrained) DDTW
double dist = dtwc::ddtwFull_L(x, y);

// Banded DDTW with Sakoe-Chiba band
double dist_banded = dtwc::ddtwBanded(x, y, /*band=*/2);

// Multivariate derivative transform (interleaved layout)
auto dx = dtwc::derivative_transform_mv(flat_data, /*ndim=*/3);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.ddtw(x, y, band=-1)  # full DDTW
dist = dtwcpp.distance.ddtw(x, y, band=5)   # banded DDTW
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant ddtw
```

---

## WDTW (Weighted DTW)

**Reference:** Jeong, Jeong & Omitaomu, "Weighted dynamic time warping for time series classification," Pattern Recognition, 44(9), 2231--2240, 2011.

WDTW multiplies each pointwise distance by a weight that depends on the index deviation $$|i - j|$$ between matched points. Larger deviations receive heavier penalties, discouraging excessive warping. The weight vector uses a logistic function:

$$w(d) = \frac{w_{\max}}{1 + \exp(-g \cdot (d - m/2))}$$

where $$d = |i - j|$$, $$m$$ is the maximum possible deviation, and $$g$$ controls steepness. The modified recurrence is:

$$C(i,j) = w(|i-j|) \cdot d(x[i], y[j]) + \min(C(i-1,j-1),\; C(i-1,j),\; C(i,j-1))$$

**Important:** WDTW modifies the DTW recurrence itself, not just the pointwise metric. It cannot be implemented as a simple metric swap.

### C++ API

```cpp
#include <dtwc/warping_wdtw.hpp>

// With precomputed weights
auto weights = dtwc::wdtw_weights<double>(/*max_dev=*/99, /*g=*/0.05);
double dist = dtwc::wdtwFull(x, y, weights);
double dist_banded = dtwc::wdtwBanded(x, y, weights, /*band=*/10);

// Convenience: pass g parameter directly
double dist2 = dtwc::wdtwFull(x, y, /*g=*/0.05);
double dist3 = dtwc::wdtwBanded(x, y, /*band=*/10, /*g=*/0.05);

// Multivariate (pointer + timestep count interface)
double dist_mv = dtwc::wdtwFull_mv(x_ptr, nx, y_ptr, ny, ndim, /*g=*/0.05);
double dist_mv_b = dtwc::wdtwBanded_mv(x_ptr, nx, y_ptr, ny, ndim, band, /*g=*/0.05);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.wdtw(x, y, band=-1, g=0.05)
dist = dtwcpp.distance.wdtw(x, y, band=10, g=0.1)
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant wdtw --wdtw-g 0.05
```

---

## ADTW (Amerced DTW)

**Reference:** Herrmann & Shifaz, "Amercing: An intuitive and effective constraint for dynamic time warping," Pattern Recognition, 137, 109301, 2023.

ADTW adds a fixed penalty for non-diagonal (horizontal/vertical) warping steps, discouraging time stretching and compression. The recurrence becomes:

$$C(i,j) = d(x[i], y[j]) + \min\big(C(i-1,j-1),\; C(i-1,j) + \omega,\; C(i,j-1) + \omega\big)$$

where $$\omega$$ is the penalty. A diagonal step (no stretching) incurs no extra cost; horizontal or vertical steps (one-to-many alignment) incur the penalty $$\omega$$.

**Important:** Like WDTW, ADTW modifies the recurrence, not just the metric.

### C++ API

```cpp
#include <dtwc/warping_adtw.hpp>

// Full ADTW (linear-space, O(min(m,n)) memory)
double dist = dtwc::adtwFull_L(x, y, /*penalty=*/0.1);

// Banded ADTW
double dist_banded = dtwc::adtwBanded(x, y, /*band=*/10, /*penalty=*/0.1);

// Multivariate
double dist_mv = dtwc::adtwFull_L_mv(x_ptr, nx, y_ptr, ny, ndim, /*penalty=*/0.1);
double dist_mv_b = dtwc::adtwBanded_mv(x_ptr, nx, y_ptr, ny, ndim, band, /*penalty=*/0.1);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.adtw(x, y, band=-1, penalty=0.1)
dist = dtwcpp.distance.adtw(x, y, band=10, penalty=0.5)
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant adtw --adtw-penalty 0.1
```

---

## Soft-DTW

**Reference:** Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series," ICML, 2017.

Soft-DTW replaces the hard $$\min$$ in the DTW recurrence with a differentiable softmin operator, making the distance differentiable with respect to the input series. This is useful for gradient-based optimization (e.g., training neural networks on time series).

The softmin is defined as:

$$\text{softmin}_\gamma(a, b, c) = -\gamma \log\big(e^{-a/\gamma} + e^{-b/\gamma} + e^{-c/\gamma}\big)$$

The recurrence becomes:

$$C(i,j) = d(x[i], y[j]) + \text{softmin}_\gamma\big(C(i-1,j-1),\; C(i-1,j),\; C(i,j-1)\big)$$

As $$\gamma \to 0$$, Soft-DTW converges to standard DTW.

**Note:** Soft-DTW can return **negative** values for identical series when $$\gamma > 0$$. It also requires the full cost matrix (no rolling-buffer optimization), so memory usage is $$O(m \times n)$$.

### C++ API

```cpp
#include <dtwc/soft_dtw.hpp>

// Compute Soft-DTW distance
double dist = dtwc::soft_dtw(x, y, /*gamma=*/1.0);

// Compute gradient w.r.t. x (for backpropagation)
std::vector<double> grad = dtwc::soft_dtw_gradient(x, y, /*gamma=*/1.0);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.soft_dtw(x, y, gamma=1.0)
grad = dtwcpp.soft_dtw_gradient(x, y, gamma=1.0)
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant softdtw --sdtw-gamma 1.0
```

---

## Comparison Table

| Variant | What it modifies | Key parameter | Best for | Memory |
|---------|-----------------|---------------|----------|--------|
| **Standard DTW** | -- | band width | General-purpose alignment | $$O(\min(m,n))$$ |
| **DDTW** | Preprocessing (derivative) | -- | Shape-sensitive comparison | $$O(\min(m,n))$$ |
| **WDTW** | Recurrence (weighted cost) | $$g$$ (steepness) | Penalizing large phase shifts | $$O(\min(m,n))$$ |
| **ADTW** | Recurrence (penalty) | $$\omega$$ (penalty) | Discouraging stretching/compression | $$O(\min(m,n))$$ |
| **Soft-DTW** | Recurrence (softmin) | $$\gamma$$ (smoothness) | Differentiable loss, gradient-based learning | $$O(m \times n)$$ |

## Selecting a Variant via Problem

When using the `Problem` class for clustering, you can select the variant through `DTWVariantParams`:

```cpp
#include <dtwc/Problem.hpp>

dtwc::Problem prob;
prob.set_data(std::move(data));

// Use ADTW for clustering
dtwc::core::DTWVariantParams params;
params.variant = dtwc::core::DTWVariant::ADTW;
params.adtw_penalty = 0.1;
prob.set_variant(params);

prob.cluster();
```

In Python:

```python
import dtwcpp

prob = dtwcpp.Problem()
prob.set_data(series, names)
prob.variant_params.variant = dtwcpp.DTWVariant.ADTW
prob.variant_params.adtw_penalty = 0.1
prob.set_variant(dtwcpp.DTWVariant.ADTW)
```

The `Problem::rebind_dtw_fn()` method automatically selects the correct function (including multivariate variants when `data.ndim > 1`).

