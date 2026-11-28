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

### MATLAB API

```matlab
d = dtwc.distance.dtw(x, y, 'Variant', 'ddtw');
d_banded = dtwc.distance.dtw(x, y, 'Variant', 'ddtw', 'Band', 5);
```

---

## WDTW (Weighted DTW)

**Reference:** Jeong, Jeong & Omitaomu, "Weighted dynamic time warping for time series classification," Pattern Recognition, 44(9), 2231--2240, 2011.

WDTW multiplies each pointwise distance by a weight that depends on the index deviation $|i - j|$ between matched points. Larger deviations receive heavier penalties, discouraging excessive warping. The weight vector uses a logistic function:

$$w(d) = \frac{w_{\max}}{1 + \exp(-g \cdot (d - m/2))}$$

where $d = |i - j|$, $m$ is the maximum possible deviation, and $g$ controls steepness. The modified recurrence is:

$$C(i,j) = w(|i-j|) \cdot d(x[i], y[j]) + \min(C(i-1,j-1),\; C(i-1,j),\; C(i,j-1))$$

**Important:** WDTW modifies the DTW recurrence itself, not just the pointwise metric. It cannot be implemented as a simple metric swap.

Each pair has its own $m$, its longer series' length less one, so the weights near the diagonal are about $e^{-g m/2}$ and the WDTW distances of long series are tiny; a pair of different lengths must leave the diagonal by the difference, where the weights are far larger, so distances spread over many orders of magnitude (on the sample data, `data/dummy`, from about 1e-95 to 1e-8 at the default $g = 0.05$).

With float32 data (`--dtype float32` for `dtwc_cl`, `dtype="float32"` in Python) the weights are float32 too, and for long series the smallest ones underflow to zero: once $g \cdot (n-1)/2$ exceeds about 88, where $n$ is the longer series' length (about 3550 steps at the default $g = 0.05$), the weights near the diagonal vanish, and the distance of every pair whose warping path can keep to them, series of equal length among them, collapses to 0. Use float64 data for long series with WDTW.

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

### MATLAB API

```matlab
d = dtwc.distance.dtw(x, y, 'Variant', 'wdtw', 'WdtwG', 0.05);
d_banded = dtwc.distance.dtw(x, y, 'Variant', 'wdtw', 'Band', 10, 'WdtwG', 0.1);
```

---

## ADTW (Amerced DTW)

**Reference:** Herrmann & Webb, "Amercing: An intuitive and effective constraint for dynamic time warping," Pattern Recognition, 137, 109333, 2023.

ADTW adds a fixed penalty for non-diagonal (horizontal/vertical) warping steps, discouraging time stretching and compression. The recurrence becomes:

$$C(i,j) = d(x[i], y[j]) + \min\big(C(i-1,j-1),\; C(i-1,j) + \omega,\; C(i,j-1) + \omega\big)$$

where $\omega$ is the penalty. A diagonal step (no stretching) incurs no extra cost; horizontal or vertical steps (one-to-many alignment) incur the penalty $\omega$.

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

### MATLAB API

```matlab
d = dtwc.distance.dtw(x, y, 'Variant', 'adtw', 'AdtwPenalty', 0.1);
d_banded = dtwc.distance.dtw(x, y, 'Variant', 'adtw', 'Band', 10, 'AdtwPenalty', 0.5);
```

---

## Soft-DTW

**Reference:** Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series," ICML, 2017.

Soft-DTW replaces the hard $\min$ in the DTW recurrence with a differentiable softmin operator, making the distance differentiable with respect to the input series. This is useful for gradient-based optimization (e.g., training neural networks on time series).

The softmin is defined as:

$$\text{softmin}_\gamma(a, b, c) = -\gamma \log\big(e^{-a/\gamma} + e^{-b/\gamma} + e^{-c/\gamma}\big)$$

The recurrence becomes:

$$C(i,j) = d(x[i], y[j]) + \text{softmin}_\gamma\big(C(i-1,j-1),\; C(i-1,j),\; C(i,j-1)\big)$$

As $\gamma \to 0$, Soft-DTW converges to standard DTW.

**Note:** Soft-DTW can return **negative** values for identical series when $\gamma > 0$. The distance keeps one rolling column, $O(\min(m, n))$ memory, as standard DTW does; `soft_dtw_gradient` keeps two $m \times n$ matrices, which its backward pass reads.
Soft-DTW is univariate and unbanded: a `Problem`'s band is ignored, and multivariate
input is refused with `InvalidInput`.

**Gradient (`soft_dtw_gradient`).** Using zero-based indices $0 \le i < m$, $0 \le j < n$ and pointwise L1 cost $d(x_i, y_j) = |x_i - y_j|$, the backward pass stores the forward cost matrix $C$ and computes the adjoint matrix $E_{i,j} = \partial C_{m-1,n-1} / \partial C_{i,j}$ in reverse order starting from $E_{m-1,n-1} = 1$. From each interior cell $(i,j)$ ($i \ge 1, j \ge 1$), let $M = \min(C_{i-1,j},\, C_{i,j-1},\, C_{i-1,j-1})$; $E_{i,j}$ distributes to each predecessor $q \in \{(i-1,j),\,(i,j-1),\,(i-1,j-1)\}$ with the softmin Jacobian weight

$$w_{i,j \to q} = \frac{\exp\!\big(-(C_q - M)/\gamma\big)}{\sum_{r} \exp\!\big(-(C_r - M)/\gamma\big)}, \qquad E_q \mathrel{+}= E_{i,j}\, w_{i,j \to q}.$$

Along the first row and column, each cell has a single predecessor with weight $1$ ($E_{0,j-1} \mathrel{+}= E_{0,j}$ and $E_{i-1,0} \mathrel{+}= E_{i,0}$). The L1 subgradient with respect to $x_i$ is then

$$\frac{\partial\,\text{sdtw}_\gamma(x, y)}{\partial x_i} = \sum_{j=0}^{n-1} E_{i,j}\,\operatorname{sign}(x_i - y_j),$$

where $\operatorname{sign}(0) = 0$. Note that `dtw_barycenter` and `barycenter_kmeans` under `BarycenterMethod::SoftDTW` use squared local costs $(x_i - y_j)^2$ instead (yielding pointwise derivative $2(x_i - y_j)$) so that barycentric averaging remains smooth.

**What clustering uses.** The raw value of similar series is negative, below the 0 between a series
and itself that clustering needs: a medoid would be nearer to another series than to itself. Clustering
under `softdtw` therefore uses the Soft-DTW divergence (Blondel, Mensch & Vert, "Differentiable
Divergences Between Time Series," AISTATS, 2021):

$$D_\gamma(x, y) = \text{sdtw}_\gamma(x, y) - \tfrac{1}{2}\big(\text{sdtw}_\gamma(x, x) + \text{sdtw}_\gamma(y, y)\big)$$

For this Soft-DTW, univariate with the absolute-value cost, the paper proves $D_\gamma \ge 0$, with
$D_\gamma(x, y) = 0$ only for identical series, at every $\gamma > 0$. A `Problem`'s matrix holds
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

### MATLAB API

```matlab
d = dtwc.distance.dtw(x, y, 'Variant', 'softdtw', 'SdtwGamma', 1.0);
g = dtwc.soft_dtw_gradient(x, y, 'Gamma', 1.0);
```

---

## MSM (Move-Split-Merge)

**Reference:** Stefan, Athitsos & Das, "The Move-Split-Merge Metric for Time
Series," IEEE TKDE, 2013.

MSM is a metric edit distance whose move, split, and merge operations are
controlled by a positive cost $c$. The split/merge transition cost of placing value $\nu$ beside neighbours $a$ and $b$ is

$$C(\nu, a, b) = \begin{cases} c, & \text{if } a \le \nu \le b \text{ or } a \ge \nu \ge b, \\ c + \min(|\nu - a|,\, |\nu - b|), & \text{otherwise.} \end{cases}$$

Using zero-based series indices $0 \le i < m$, $0 \le j < n$, the recurrence is:

$$M_{0,0} = |x_0 - y_0|, \qquad M_{i,0} = M_{i-1,0} + C(x_i, x_{i-1}, y_0), \qquad M_{0,j} = M_{0,j-1} + C(y_j, x_0, y_{j-1}),$$

$$M_{i,j} = \min\big(M_{i-1,j-1} + |x_i - y_j|,\; M_{i-1,j} + C(x_i, x_{i-1}, y_j),\; M_{i,j-1} + C(y_j, x_i, y_{j-1})\big).$$

DTWC++ implements the exact univariate, unbanded recurrence with $O(\min(m,n))$ rolling scratch; multivariate input is rejected (`InvalidInput`) and the `Problem` band is ignored.

### C++ API

```cpp
#include <distance.hpp>

double dist = dtwc::distance::msm(x, y, /*c=*/1.0);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.dtw(x, y, variant="msm", msm_c=1.0)
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant msm --msm-c 1.0
```

### MATLAB API

```matlab
d = dtwc.distance.dtw(x, y, 'Variant', 'msm', 'MsmC', 1.0);
```

---

## TWE (Time Warp Edit)

**Reference:** Marteau, "Time Warp Edit Distance with Stiffness Adjustment for
Time Series Matching," IEEE TPAMI, 2009.

TWE combines edit operations with a positive stiffness $\nu$ and positive
edit penalty $\lambda$. Padding both series at the front with zero ($\hat{x}_0 = 0$, $\hat{x}_i = x_{i-1}$ for $1 \le i \le m$, and similarly $\hat{y}_0 = 0$, $\hat{y}_j = y_{j-1}$ for $1 \le j \le n$), the padded table starts with $D_{0,0} = 0$ and $D_{i,0} = D_{0,j} = +\infty$ for $i, j > 0$. Consequently $D_{1,1} = |x_0 - y_0|$, $D_{1,j} = D_{1,j-1} + |\hat{y}_{j-1} - \hat{y}_j| + \nu + \lambda$ ($j \ge 2$), $D_{i,1} = D_{i-1,1} + |\hat{x}_{i-1} - \hat{x}_i| + \nu + \lambda$ ($i \ge 2$), and for $i, j \ge 2$:

$$D_{i,j} = \min\begin{cases} D_{i-1,j} + |\hat{x}_{i-1} - \hat{x}_i| + \nu + \lambda, \\ D_{i,j-1} + |\hat{y}_{j-1} - \hat{y}_j| + \nu + \lambda, \\ D_{i-1,j-1} + |\hat{x}_i - \hat{y}_j| + |\hat{x}_{i-1} - \hat{y}_{j-1}| + 2\nu|i - j|. \end{cases}$$

DTWC++ implements the exact univariate, unbanded recurrence with $O(\min(m,n))$ rolling scratch; multivariate input is rejected (`InvalidInput`) and the `Problem` band is ignored.

### C++ API

```cpp
#include <distance.hpp>

double dist = dtwc::distance::twe(x, y, /*nu=*/0.001, /*lambda=*/1.0);
```

### Python API

```python
import dtwcpp

dist = dtwcpp.distance.dtw(x, y, variant="twe", twe_nu=0.001, twe_lambda=1.0)
```

### CLI

```bash
dtwc_cl -i data.csv -k 3 --variant twe --twe-nu 0.001 --twe-lambda 1.0
```

### MATLAB API

```matlab
d = dtwc.distance.dtw(x, y, 'Variant', 'twe', 'TweNu', 0.001, 'TweLambda', 1.0);
```

---

## Comparison Table

| Variant | What it modifies | Key parameter | Best for | Memory |
|---------|-----------------|---------------|----------|--------|
| **Standard DTW** | -- | band width | General-purpose alignment | $O(\min(m,n))$ |
| **DDTW** | Preprocessing (derivative) | -- | Shape-sensitive comparison | $O(\min(m,n))$ |
| **WDTW** | Recurrence (weighted cost) | $g$ (steepness) | Penalizing large phase shifts | $O(\min(m,n))$ |
| **ADTW** | Recurrence (penalty) | $\omega$ (penalty) | Discouraging stretching/compression | $O(\min(m,n))$ |
| **Soft-DTW** | Recurrence (softmin) | $\gamma$ (smoothness) | Differentiable loss, gradient-based learning | $O(\min(m,n))$; the gradient $O(m \times n)$ |
| **MSM** | Move/split/merge edit recurrence | $c$ | Metric elastic comparison | $O(\min(m,n))$ |
| **TWE** | Stiff edit-distance recurrence | $\nu,\lambda$ | Metric elastic comparison with edit penalties | $O(\min(m,n))$ |

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

