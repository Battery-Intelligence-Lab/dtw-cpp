---
title: DTW Variants
weight: 6
---

# DTW Variants

DTWC++ exposes six alternatives to standard DTW. DDTW, WDTW, ADTW, and
Soft-DTW alter DTW preprocessing or recurrence; MSM and TWE are standalone
elastic metrics with their own dynamic programs.

## DDTW (Derivative DTW)

**Reference:** Keogh & Pazzani, "Derivative Dynamic Time Warping," SIAM SDM, 2001.

DDTW applies a derivative transform to both series as a preprocessing step, then runs standard DTW on the derivative series. This makes the comparison sensitive to **shape** (local slopes) rather than raw amplitude.

The derivative formula for interior points is:

$$x'[i] = \frac{(x[i] - x[i-1]) + (x[i+1] - x[i-1]) / 2}{2}, \quad 1 \le i \le n-2$$

Boundary cases use simple forward/backward differences:

$$x'[0] = x[1] - x[0], \quad x'[n-1] = x[n-1] - x[n-2]$$

### C++ API

```cpp
#include <warping_ddtw.hpp>

std::vector<double> x = {1.0, 2.0, 3.0, 2.0, 1.0};
std::vector<double> y = {1.0, 1.5, 3.0, 2.5, 1.0};

// Full (unconstrained) DDTW: band -1, the default
double dist = dtwc::ddtwBanded(x, y);

// Banded DDTW with Sakoe-Chiba band
double dist_banded = dtwc::ddtwBanded(x, y, /*band=*/2);

// Multivariate derivative transform (interleaved layout)
auto dx = dtwc::derivative_transform_mv(flat_data, /*ndim=*/3);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.dtw(x, y, variant="ddtw")          # full DDTW
dist = dtwcpp.distance.dtw(x, y, variant="ddtw", band=5)  # banded DDTW
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

Each pair has its own $$m$$, its longer series' length less one, so the weights near the diagonal are about $$e^{-g m/2}$$ and the WDTW distances of long series are tiny; a pair of different lengths must leave the diagonal by the difference, where the weights are far larger, so distances spread over many orders of magnitude (on the sample data, `data/dummy`, from about 1e-95 to 1e-8 at the default $$g = 0.05$$).

With float32 data (`--dtype float32` for `dtwc_cl`, `dtype="float32"` in Python) the weights are float32 too, and for long series the smallest ones underflow to zero: once $$g \cdot (n-1)/2$$ exceeds about 88, where $$n$$ is the longer series' length (about 3550 steps at the default $$g = 0.05$$), the weights near the diagonal vanish, and the distance of every pair whose warping path can keep to them, series of equal length among them, collapses to 0. Use float64 data for long series with WDTW.

### C++ API

```cpp
#include <warping_wdtw.hpp>

// With precomputed weights
auto weights = dtwc::wdtw_weights<double>(/*max_dev=*/99, /*g=*/0.05);
double dist = dtwc::wdtwBanded(x, y, weights);  // band -1, the default: full
double dist_banded = dtwc::wdtwBanded(x, y, weights, /*band=*/10);

// Convenience: pass g parameter directly
double dist2 = dtwc::wdtwBanded(x, y, /*band=*/-1, /*g=*/0.05);
double dist3 = dtwc::wdtwBanded(x, y, /*band=*/10, /*g=*/0.05);

// Multivariate (pointer + timestep count interface)
double dist_mv = dtwc::wdtwBanded_mv(x_ptr, nx, y_ptr, ny, ndim, /*band=*/-1, /*g=*/0.05);
double dist_mv_b = dtwc::wdtwBanded_mv(x_ptr, nx, y_ptr, ny, ndim, band, /*g=*/0.05);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.dtw(x, y, variant="wdtw", wdtw_g=0.05)
dist = dtwcpp.distance.dtw(x, y, variant="wdtw", band=10, wdtw_g=0.1)
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant wdtw --wdtw-g 0.05
```

---

## ADTW (Amerced DTW)

**Reference:** Herrmann & Webb, "Amercing: An intuitive and effective constraint for dynamic time warping," Pattern Recognition, 137, 109333, 2023.

ADTW adds a fixed penalty for non-diagonal (horizontal/vertical) warping steps, discouraging time stretching and compression. The recurrence becomes:

$$C(i,j) = d(x[i], y[j]) + \min\big(C(i-1,j-1),\; C(i-1,j) + \omega,\; C(i,j-1) + \omega\big)$$

where $$\omega$$ is the penalty. A diagonal step (no stretching) incurs no extra cost; horizontal or vertical steps (one-to-many alignment) incur the penalty $$\omega$$.

**Important:** Like WDTW, ADTW modifies the recurrence, not just the metric.

### C++ API

```cpp
#include <warping_adtw.hpp>

// Full ADTW (band -1; linear-space, O(min(m,n)) memory)
double dist = dtwc::adtwBanded(x, y, /*band=*/-1, /*penalty=*/0.1);

// Banded ADTW
double dist_banded = dtwc::adtwBanded(x, y, /*band=*/10, /*penalty=*/0.1);

// Multivariate
double dist_mv = dtwc::adtwBanded_mv(x_ptr, nx, y_ptr, ny, ndim, /*band=*/-1, /*penalty=*/0.1);
double dist_mv_b = dtwc::adtwBanded_mv(x_ptr, nx, y_ptr, ny, ndim, band, /*penalty=*/0.1);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.dtw(x, y, variant="adtw", adtw_penalty=0.1)
dist = dtwcpp.distance.dtw(x, y, variant="adtw", band=10, adtw_penalty=0.5)
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

**Note:** Soft-DTW can return **negative** values for identical series when $$\gamma > 0$$. The distance keeps one rolling column, $$O(\min(m, n))$$ memory, as standard DTW does; `soft_dtw_gradient` keeps two $$m \times n$$ matrices, which its backward pass reads.
Soft-DTW is univariate and unbanded: a `Problem`'s band is ignored, and multivariate
input is refused with `InvalidInput`.

**What clustering uses.** The raw value of similar series is negative, below the 0 between a series
and itself that clustering needs: a medoid would be nearer to another series than to itself. Clustering
under `softdtw` therefore uses the Soft-DTW divergence (Blondel, Mensch & Vert, "Differentiable
Divergences Between Time Series," AISTATS, 2021):

$$D_\gamma(x, y) = \text{sdtw}_\gamma(x, y) - \tfrac{1}{2}\big(\text{sdtw}_\gamma(x, x) + \text{sdtw}_\gamma(y, y)\big)$$

For this Soft-DTW, univariate with the absolute-value cost, the paper proves $$D_\gamma \ge 0$$, with
$$D_\gamma(x, y) = 0$$ only for identical series, at every $$\gamma > 0$$. A `Problem`'s matrix holds
it, as do a result's `distance_matrix` and `<name>_distance_matrix.csv`, its zero diagonal exact;
every clustering method and score reads it, and `DTWClustering`'s `transform`, `predict` and `score`
compute it. The distance functions (`soft_dtw`, `distance.dtw`, `cdist`), `soft_dtw_gradient` and the
Soft-DTW barycentre keep the raw value. A computed divergence below 0 is rounding and reads 0; as a
small difference of larger raw values, in float32 it can read 0 for two distinct but very close series too.

### C++ API

```cpp
#include <soft_dtw.hpp>

// Compute Soft-DTW distance
double dist = dtwc::soft_dtw(x, y, /*gamma=*/1.0);

// Compute gradient w.r.t. x (for backpropagation)
std::vector<double> grad = dtwc::soft_dtw_gradient(x, y, /*gamma=*/1.0);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.dtw(x, y, variant="softdtw", sdtw_gamma=1.0)
grad = dtwcpp.soft_dtw_gradient(x, y, gamma=1.0)
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant softdtw --sdtw-gamma 1.0
```

---

## MSM (Move-Split-Merge)

**Reference:** Stefan, Athitsos & Das, "The Move-Split-Merge Metric for Time
Series," IEEE TKDE, 2013.

MSM is a metric edit distance whose move, split, and merge operations are
controlled by a positive cost $$c$$. DTWC++ implements the exact univariate,
unbanded recurrence with $$O(\min(m,n))$$ rolling scratch; multivariate input is
rejected (`InvalidInput`) and the `Problem` band is ignored.

```cpp
#include <distance.hpp>

double dist = dtwc::distance::msm(x, y, /*c=*/1.0);
```

```bash
dtwc_cl -i data.csv -k 3 --variant msm --msm-c 1.0
```

---

## TWE (Time Warp Edit)

**Reference:** Marteau, "Time Warp Edit Distance with Stiffness Adjustment for
Time Series Matching," IEEE TPAMI, 2009.

TWE combines edit operations with a positive stiffness $$\nu$$ and positive
edit penalty $$\lambda$$. DTWC++ implements the exact univariate, unbanded
recurrence with $$O(\min(m,n))$$ rolling scratch; multivariate input is rejected
(`InvalidInput`) and the `Problem` band is ignored.

```cpp
#include <distance.hpp>

double dist = dtwc::distance::twe(x, y, /*nu=*/0.001, /*lambda=*/1.0);
```

```bash
dtwc_cl -i data.csv -k 3 --variant twe --twe-nu 0.001 --twe-lambda 1.0
```

---

## Comparison Table

| Variant | What it modifies | Key parameter | Best for | Memory |
|---------|-----------------|---------------|----------|--------|
| **Standard DTW** | -- | band width | General-purpose alignment | $$O(\min(m,n))$$ |
| **DDTW** | Preprocessing (derivative) | -- | Shape-sensitive comparison | $$O(\min(m,n))$$ |
| **WDTW** | Recurrence (weighted cost) | $$g$$ (steepness) | Penalizing large phase shifts | $$O(\min(m,n))$$ |
| **ADTW** | Recurrence (penalty) | $$\omega$$ (penalty) | Discouraging stretching/compression | $$O(\min(m,n))$$ |
| **Soft-DTW** | Recurrence (softmin) | $$\gamma$$ (smoothness) | Differentiable loss, gradient-based learning | $$O(\min(m,n))$$; the gradient $$O(m \times n)$$ |
| **MSM** | Move/split/merge edit recurrence | $$c$$ | Metric elastic comparison | $$O(\min(m,n))$$ |
| **TWE** | Stiff edit-distance recurrence | $$\nu,\lambda$$ | Metric elastic comparison with edit penalties | $$O(\min(m,n))$$ |

## Selecting a Variant via Problem

When using the `Problem` class for clustering, you can select the variant through `DTWVariantParams`:

```cpp
#include <Problem.hpp>

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
params = dtwcpp.DTWVariantParams()  # prob.variant_params is read-only: set a whole one
params.variant = dtwcpp.DTWVariant.ADTW
params.adtw_penalty = 0.1
prob.variant_params = params  # or prob.set_variant_params(params)
```

The `Problem::rebind_dtw_fn()` method automatically selects the correct function (including multivariate variants when `Problem::data().ndim > 1`).

