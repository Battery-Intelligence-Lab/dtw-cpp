# DTW with Missing Data: Literature Review and Implementation Plan

## 1. Types of Missing Data in Time Series

### 1.1 Missing Completely at Random (MCAR)

The probability of a value being missing is independent of both observed and unobserved values. Example: a sensor randomly drops packets due to network congestion unrelated to the measured phenomenon.

**Impact on DTW:** Simplest case. Both imputation and modified-DTW approaches work well. The expected DTW distance with imputed values is an unbiased estimator of the true distance, provided the imputation method is unbiased.

### 1.2 Missing at Random (MAR)

The probability of missingness depends on observed values but not on the unobserved values themselves. Example: a battery monitoring system that records voltage less frequently during stable operation.

**Impact on DTW:** Simple imputation introduces bias because the missing pattern correlates with observable features. Model-based imputation (EM, multiple imputation) is needed. Modified DTW with zero cost for missing positions may underestimate distance because missing regions correlate with specific signal characteristics.

### 1.3 Missing Not at Random (MNAR)

The probability of missingness depends on the unobserved value itself. Example: a temperature sensor that stops reporting when exceeding its operating range.

**Impact on DTW:** Most challenging. Both imputation and skip-based approaches are biased. The missingness pattern itself carries distance information. No purely algorithmic approach can fully compensate; domain knowledge is needed.

### 1.4 Structural Missing Data

Missingness from data collection design rather than random processes:

- **Sensors offline:** Entire segments missing due to device failure or maintenance
- **Different sampling rates:** Two sensors at different frequencies (1 Hz vs 10 Hz)
- **Partial overlap:** Series that only overlap temporally for a portion of their duration

**Impact on DTW:** The most common real-world scenario for DTWC++. Missing segments can be large contiguous blocks, making local interpolation ineffective. Open-end DTW (Tormene et al., 2009) and subsequence DTW are designed for this case.

### 1.5 Irregular Sampling

Time series sampled at non-uniform intervals. While not strictly "missing" data, standard DTW assumes consecutive indices correspond to uniform time steps.

**Impact on DTW:** Two approaches: (a) resample to uniform grid (interpolation artifacts), or (b) time-aware DTW that accounts for timestamps. Shares implementation concerns with missing data in terms of representing "gaps."

---

## 2. Approaches to DTW with Missing Data

### 2A. Imputation-Based Approaches (Pre-processing)

Fill in missing values before computing standard DTW. The DTW core remains unmodified.

#### Linear Interpolation

For a gap between observed values at positions t_a and t_b:
`x(t_i) = x(t_a) + (x(t_b) - x(t_a)) * (t_i - t_a) / (t_b - t_a)`

- **Pros:** Simple, fast O(n), preserves endpoints, works well for smooth signals with small gaps
- **Cons:** Introduces artificial smoothness that reduces DTW distance; systematically underestimates distances. Fails for large gaps or oscillatory signals. Cannot handle leading/trailing missing values.

#### Spline Interpolation (Cubic)

Piecewise cubic polynomials through observed data, preserving continuity up to second derivative.

- **Pros:** Better preserves local curvature, less systematic bias in DTW distance
- **Cons:** Can overshoot/undershoot for large gaps (Runge phenomenon). Slightly more expensive. Still cannot extrapolate at boundaries.

#### Last Observation Carried Forward (LOCF)

Fill gaps with the nearest observed value.

- **Pros:** Extremely simple, O(n). Natural for step-like signals.
- **Cons:** Creates artificial flat segments that attract DTW warping paths. Very poor for continuous signals. Introduces zero-derivative regions that interact badly with derivative DTW.

#### Mean/Median Imputation

Replace missing values with the series mean or median.

- **Pros:** Simple, fast. Preserves overall level.
- **Cons:** Destroys local temporal structure. Creates discontinuities. Severely biases DTW distances. Should generally be avoided for time series.

#### Model-Based Imputation (EM, GP, Matrix Completion)

Use a statistical model to estimate conditional distribution of missing values given observed data.

- **Pros:** Unbiased under MAR. GP provides uncertainty estimates. Matrix completion leverages cross-series structure.
- **Cons:** Expensive (GP: O(n^3) exact). Requires model/kernel choice. Overkill for MCAR with small gaps. Adds library dependencies.

#### Multiple Imputation

Generate M imputed datasets, compute DTW on each, combine results.

- **Pros:** Confidence intervals on DTW distances. Properly propagates uncertainty. Principled under MAR.
- **Cons:** Multiplies computation by M (typically 5-20). Combining DTW distances is non-trivial. Prohibitive for large-scale clustering.

**Summary:** Linear interpolation should be the default imputation utility (simple, fast, adequate for MCAR with small gaps). Provide it as a pre-processing option but keep it separate from DTW core.

### 2B. Modified DTW Algorithms

#### Zero-Cost DTW (Skip Missing)

The simplest modification: when either x[i] or y[j] is missing, set the local distance to zero.

Standard recurrence:

```
C(i, j) = d(x[i], y[j]) + min{ C(i-1, j), C(i, j-1), C(i-1, j-1) }
```

Modified recurrence:

```
if is_missing(x[i]) or is_missing(y[j]):
    C(i, j) = 0 + min{ C(i-1, j), C(i, j-1), C(i-1, j-1) }
else:
    C(i, j) = d(x[i], y[j]) + min{ C(i-1, j), C(i, j-1), C(i-1, j-1) }
```

- **Complexity:** Same as standard DTW: O(mn) time, O(mn) or O(min(m,n)) space
- **Concern:** Systematically underestimates distance when many values are missing. Path-length normalization is essential.

#### DTW-AROW (Yurtman et al., 2023)

"Additional Restrictions on Warping." The most relevant paper for this use case. Modifies DTW to handle missing values with additional warping restrictions to prevent the path from "cheating" through missing regions.

Core idea: missing positions get zero local cost but the warping path restrictions ensure meaningful alignment. Path-length normalization accounts for varying numbers of non-missing comparisons.

Open-source implementation available: https://github.com/aras-y/DTW_with_missing_values

#### DTW-CAI (Yurtman et al., 2023)

"Clustering, Averaging, Imputation." A global method using the full dataset:

1. Cluster time series with partial DTW-AROW distances
2. Compute cluster averages (DBA or similar)
3. Impute missing values using nearest cluster average
4. Recompute DTW distances on imputed data
5. Iterate

More accurate than DTW-AROW alone but requires the full dataset and is iterative. Cost: O(N * mn * K_iter) where N is dataset size.

#### Weighted DTW with Missing Data

Assign per-position weights based on confidence:

```
C(i, j) = w(i, j) * d(x[i], y[j]) + min{ C(i-1, j), C(i, j-1), C(i-1, j-1) }
```

where w(i,j) = 0 if either value is missing, w(i,j) = 1 otherwise. Advanced: use imputation confidence as weights.

#### Open-End DTW (Tormene et al., 2009)

Modifies boundary conditions to allow alignment to end before reaching the end of the reference series:

```
DTW_open_end = min over j: C(m, j) / normalization(j)
```

Designed for prefix matching of truncated series. Demonstrated kappa=0.898 vs 0.447 for global DTW on partial series classification.

- **Pros:** Natural for structurally incomplete series (partial observations)
- **Cons:** Only addresses trailing truncation, not arbitrary gaps within the series

#### Partial DTW

Only align observed subsequences. Break each series into observed segments and compute DTW on overlapping portions.

- **Pros:** Only uses real data; no imputation bias
- **Cons:** Complex when multiple gaps exist. May compare incomparable portions.

### 2C. Probabilistic Approaches

#### Uncertainty-DTW (Wang and Koniusz, 2022)

Models heteroscedastic aleatoric uncertainty at each time step:

```
d_uncertainty(i, j) = (x[i] - y[j])^2 / (sigma_x[i]^2 + sigma_y[j]^2)
                     + log(sigma_x[i]^2 + sigma_y[j]^2)
```

Missing values are represented as positions with very large variance, naturally down-weighting their contribution.

- **Pros:** Principled statistical framework. Handles varying confidence levels (not just binary missing/present).
- **Cons:** Requires variance estimates. Based on soft-DTW (differentiable), not classical DP. More expensive.

#### Gaussian Process DTW (Kazlauskaite et al., 2019)

Model each time series as a GP realization. The GP provides posterior distribution over function values at missing points. Compute expected DTW under the posterior.

- **Pros:** Theoretically elegant. Full posterior over DTW distances.
- **Cons:** Very expensive (O(n^3) per series). Impractical for large-scale clustering.

### 2D. Embedding-Based Approaches

#### Time Series Cluster Kernel (TCK) (Mikalsen et al., 2018)

Ensemble of Gaussian mixture models with informative priors that naturally handle missing data. The kernel between two series is the proportion of ensemble members assigning them to the same cluster.

- **Pros:** Native missing data handling. Produces a valid kernel (PSD).
- **Cons:** Not a DTW distance; fundamentally different approach. Requires hyperparameter tuning. Less interpretable.

---

## 3. Theoretical Analysis

### 3.1 Metric Properties

Standard DTW already fails to be a proper metric (Marteau, 2009; Herrmann, 2023). It is at best a symmetric premetric (non-negative, symmetric, but fails triangle inequality).

With missing data, the situation worsens:

| Property | Standard DTW | DTW with Zero-Cost Missing | DTW with Imputation |
| --- | --- | --- | --- |
| Non-negativity: d(x,y) >= 0 | Yes | Yes | Yes |
| Identity: d(x,x) = 0 | Yes | Yes | Yes |
| Indiscernibles: d(x,y)=0 => x=y | No | No (worse: more false zeros) | No |
| Symmetry: d(x,y) = d(y,x) | Yes | Yes (is_missing OR is symmetric) | Yes |
| Triangle inequality | No | No (worse) | No |

### 3.2 Implications for k-Medoids Clustering

**Good news:** k-medoids (PAM) does NOT require a metric. It only requires d(x,y) >= 0 with d(x,x) = 0 (Jiang et al., 2021).

PAM convergence is guaranteed as long as:

- The cost function is bounded below by zero
- Each swap step strictly decreases cost or terminates
- The number of possible medoid configurations is finite

All three hold for DTW with missing data. **PAM convergence is unaffected.**

However, clustering **quality** may degrade: noisy distance estimates from high missingness may not reflect true structure. A minimum coverage threshold is recommended.

### 3.3 Impact on Lower Bounds

**LB_Keogh** requires all values present for envelope construction. With missing data:

- Envelope construction must skip or interpolate missing values
- The lower bound guarantee (LB_Keogh <= DTW) may not hold with inconsistent handling
- With consistent zero-cost treatment, the bound holds but becomes very loose

**LB_Kim** uses features (first, last, min, max). Can be adapted by computing features over observed values only.

**Recommendation:** Disable lower bound pruning when either series has missing data, at least initially.

### 3.4 Computational Complexity

| Approach | Time | Space | Notes |
| --- | --- | --- | --- |
| Imputation + standard DTW | O(n) + O(mn) | O(min(m,n)) | Imputation negligible |
| Zero-cost DTW | O(mn) | O(min(m,n)) | Same as standard DTW |
| DTW-AROW | O(mn) | O(mn) | Same DP structure |
| DTW-CAI | O(N * mn * K) | O(N^2 + mn) | N=dataset, K=iterations |
| Uncertainty-DTW | O(mn) | O(mn) | Per-pair, soft-DTW variant |
| GP + DTW | O(n^3 + mn) | O(n^2 + mn) | GP dominates |
| Multiple imputation | O(M * mn) | O(M * min(m,n)) | M=number of imputations |

---

## 4. Key Papers (Annotated Bibliography)

### Core DTW with Missing Data

1. **Yurtman, A., Soenen, J., Meert, W., Blockeel, H.** (2023). "Estimating Dynamic Time Warping Distance Between Time Series with Missing Data." ECML PKDD 2023, LNAI 14169, pp. 224-240. **[Most relevant paper]** Proposes DTW-AROW and DTW-CAI. Open-source code available. GitHub: https://github.com/aras-y/DTW_with_missing_values

2. **Tormene, P., Giorgino, T., Quaglini, S., Stefanelli, M.** (2009). "Matching Incomplete Time Series with Dynamic Time Warping." Artificial Intelligence in Medicine, 45, 11-34. **[Foundational]** Open-end DTW for truncated series.

3. **Phan, T.T.H., Poisson-Caillault, E., Lefebvre, A., Bigand, A.** (2017). "Dynamic Time Warping-Based Imputation for Univariate Time Series Data." Pattern Recognition Letters, 100, 1-7. DTW-based imputation (DTWBI): uses DTW to find similar sub-sequences for imputing large gaps.

### Uncertainty and Probabilistic DTW

4. **Wang, L., Koniusz, P.** (2022). "Uncertainty-DTW for Time Series and Sequences." ECCV 2022. arXiv:2211.00005. Per-frame aleatoric uncertainty in differentiable DTW. Missing values modeled as high-variance positions.

5. **Kazlauskaite, I. et al.** (2019). "Gaussian Process Latent Variable Alignment Learning." AISTATS 2019. GP-based alignment with uncertainty.

### DTW Metric Properties

6. **Marteau, P.F.** (2009). "On the Metric Properties of Dynamic Time Warping." IEEE ICPR. Shows DTW violates triangle inequality.

7. **Herrmann, M.** (2023). "Semi-Metrification of the Dynamic Time Warping Distance." arXiv:1808.09964. Converting DTW to a semi-metric.

### Kernel and Embedding Approaches

8. **Mikalsen, K.O. et al.** (2018). "Time Series Cluster Kernel for Learning Similarities Between Multivariate Time Series with Missing Data." Pattern Recognition, 76, 569-581. TCK: ensemble-based kernel with native missing data handling.

### k-Medoids Theory

9. **Jiang, H., Jang, J., Kpotufe, S.** (2021). "On the Consistency of Metric and Non-Metric K-medoids." AISTATS 2021. k-medoids consistency even for non-metric dissimilarities.

### DTW-Based Imputation

10. **Kim, H.J., Dunn, K.P., Bhowmik, P.** (2005). "KNN-DTW Based Missing Value Imputation for Microarray Time Series Data." k-nearest neighbors with DTW distance for imputation.

### Lower Bounds

11. **Keogh, E., Ratanamahatana, C.A.** (2005). "Exact Indexing of Dynamic Time Warping." Knowledge and Information Systems, 7, 358-386. Defines LB_Keogh.

12. **Webb, G.I.** (2021). "Tight Lower Bounds for Dynamic Time Warping." arXiv:2102.07076. Tighter bounds than LB_Keogh.

### Time Series Clustering

13. **Holder, C., Middlehurst, M., Bagnall, A.** (2024). "A Review and Evaluation of Elastic Distance Functions for Time Series Clustering." Knowledge and Information Systems, 66, 765-809. Comparison of 9 elastic distances with k-medoids.

---

## 5. Implementation Plan for DTWC++

### 5.1 Representing Missing Values: IEEE 754 NaN

```cpp
#include <cmath>
#include <limits>

namespace dtwc {
    constexpr data_t MISSING = std::numeric_limits<data_t>::quiet_NaN();

    inline bool is_missing(data_t v) noexcept {
#ifdef __FAST_MATH__
        // Bitwise NaN check for -ffast-math compatibility
        union { data_t f; uint64_t i; } u = {v};
        return (u.i & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL
            && (u.i & 0x000FFFFFFFFFFFFFULL) != 0;
#else
        return std::isnan(v);
#endif
    }
}
```

**Why NaN over alternatives:**

| Representation | Pros | Cons |
| --- | --- | --- |
| NaN | IEEE standard, self-propagating, portable `std::isnan()` | NaN != NaN, `-ffast-math` breaks `isnan` |
| Sentinel (-999.0) | Simple equality check | Collides with real data |
| `std::vector<bool>` mask | No value-space collision | Doubles memory, sync issues |
| `std::optional<data_t>` | Type-safe, C++17 | 2x memory, breaks vectorization |

### 5.2 DTWOptions and MissingStrategy

```cpp
namespace dtwc {

enum class MissingStrategy {
    Error,           // Throw if NaN encountered (backward-compatible default)
    ZeroCost,        // Zero local cost for missing pairs
    ZeroCostNorm,    // Zero cost + path-length normalization
    Interpolate,     // Linear interpolation before DTW
    Skip             // Skip missing indices entirely (reindex observed only)
};

struct DTWOptions {
    int band = -1;                                    // Sakoe-Chiba band (-1 = full)
    MissingStrategy missing = MissingStrategy::Error;  // Missing data handling
    double min_coverage = 0.0;                         // Min fraction of non-missing pairs
};

} // namespace dtwc
```

### 5.3 Modified DP Recurrence

Core modification to `dtwFull` in `dtwc/warping.hpp`:

```cpp
// Inner loop modification:
for (int j = 1; j < my; j++) {
    for (int i = 1; i < mx; i++) {
        const auto minimum = std::min({ C(i-1, j), C(i, j-1), C(i-1, j-1) });

        if (is_missing(x[i]) || is_missing(y[j])) {
            C(i, j) = minimum;  // Zero local cost
        } else {
            C(i, j) = minimum + distance(x[i], y[j]);
        }
    }
}
```

Boundary conditions need the same treatment:

```cpp
// First column:
for (int i = 1; i < mx; i++) {
    data_t cost = (is_missing(x[i]) || is_missing(y[0])) ? 0 : distance(x[i], y[0]);
    C(i, 0) = C(i-1, 0) + cost;
}

// First row:
for (int j = 1; j < my; j++) {
    data_t cost = (is_missing(x[0]) || is_missing(y[j])) ? 0 : distance(x[0], y[j]);
    C(0, j) = C(0, j-1) + cost;
}
```

### 5.4 Path-Length Normalization

When missing values are present, raw accumulated cost is not comparable across pairs with different missing patterns. Two approaches:

**Approach A: Approximate normalization** (recommended for dtwFull_L):

```cpp
int missing_in_x = std::count_if(x.begin(), x.end(), is_missing);
int missing_in_y = std::count_if(y.begin(), y.end(), is_missing);
double coverage = 1.0 - (double)(missing_in_x + missing_in_y) / (mx + my);
return (coverage > min_coverage) ? C(mx-1, my-1) / coverage : maxValue;
```

**Approach B: Exact path tracking** (for dtwFull with backtracking):

Maintain a parallel count matrix tracking non-missing comparisons along the optimal path. Doubles memory but gives exact normalization.

### 5.5 Impact on Existing DTW Variants

**Banded DTW (`dtwBanded`):** The missing data modification is identical to full DTW -- add `is_missing` check inside the inner loop. Band constraint and missing data are orthogonal. However, if the band excludes regions where non-missing data could align (while within-band positions are missing), consider recommending wider bands with high missingness.

**Memory-efficient DTW (`dtwFull_L`):** Same modification works. Challenge: path-length normalization requires backtracking (unavailable). Use approximate normalization (Approach A above).

**Thread safety:** Current `thread_local` pattern for scratch buffers works fine. The NaN check is a pure function. Additional count vectors can also be `thread_local`.

### 5.6 Distance Matrix Properties

**Symmetry preserved:** `is_missing(x[i]) || is_missing(y[j])` is symmetric in x and y. The DP produces the same cost regardless of which series is "row" vs "column." The upper-triangular fill optimization in `Problem::fillDistanceMatrix()` remains correct.

**MIP formulation unaffected:** The MIP in `mip_Gurobi.cpp` and `mip_Highs.cpp` takes the distance matrix as input. It requires only non-negativity and symmetry -- no triangle inequality needed. **No changes to MIP code required.** Only the distance matrix entries change (via modified DTW).

**Caveat:** If coverage-based normalization produces non-uniform scaling, the MIP may assign points with high missing rates to nearby medoids (lower apparent distances). A minimum coverage threshold with maxValue fallback prevents meaningless assignments.

### 5.7 Data Loading

The existing `readTimeSeriesCSV` in `dtwc/fileOperations.hpp` has a catch block that skips non-numeric values. Modify to insert NaN instead:

```cpp
// Change from skipping to inserting MISSING:
} catch (const std::exception &) {
    series.push_back(MISSING);
}
```

### 5.8 User API

```cpp
std::vector<double> x = {1.0, 2.0, NAN, NAN, 5.0, 6.0};
std::vector<double> y = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};

DTWOptions opts;
opts.missing = MissingStrategy::ZeroCostNorm;
opts.min_coverage = 0.5;

double d = dtw(x, y, opts);
```

Default `MissingStrategy::Error` ensures backward compatibility. Users must explicitly opt in.

---

## 6. Phased Implementation Plan

### Phase 1: Foundation (PR 1)

- Add `is_missing()` utility in new `dtwc/missing.hpp`
- Add `MissingStrategy` enum and `DTWOptions` struct
- Add `bool has_missing(const std::vector<data_t>&)` utility
- Unit tests for missing value detection

### Phase 2: Core DTW Modifications (PR 2)

- Modify `dtwFull` to accept `DTWOptions` and handle `ZeroCost`
- Modify `dtwFull_L` similarly
- Modify `dtwBanded` similarly
- Add unified `dtw()` dispatcher based on options
- Comprehensive unit tests with known expected values

### Phase 3: Normalization (PR 3)

- Implement `ZeroCostNorm` with path-length normalization
- Implement `min_coverage` threshold
- Tests comparing normalized vs unnormalized

### Phase 4: Imputation Utilities (PR 4)

- `interpolate_linear(std::vector<data_t>&)` -- in-place
- `interpolate_spline(std::vector<data_t>&)` -- in-place
- `MissingStrategy::Interpolate` dispatches interpolation before DTW
- Imputation correctness tests

### Phase 5: Problem Class Integration (PR 5)

- Add `DTWOptions` member to `Problem`
- Modify `distByInd()` to pass options through
- Coverage-based distance masking (maxValue for low-coverage pairs)
- Integration tests: clustering on data with missing values

---

## 7. Connections to Other DTW Extensions

### 7.1 Distance Metrics (L1, L2, Cosine)

Missing data handling is **orthogonal** to metric choice. The `is_missing()` check wraps the metric call:

```cpp
if (is_missing(x[i]) || is_missing(y[j])) {
    cost = 0;
} else {
    cost = metric(x[i], y[j]);  // Any metric
}
```

Metric abstraction and missing data can be implemented in parallel without conflicts.

### 7.2 Derivative DTW

Missing values propagate through derivatives: if x[i] or x[i+1] is NaN, then x'[i] = NaN. This correctly expands the missing region by one sample on each side. Then apply zero-cost DTW on the derivative series.

### 7.3 Weighted DTW

Fully compatible. Missing overrides weight:

```cpp
cost = is_missing(x[i]) || is_missing(y[j]) ? 0 : w(i,j) * metric(x[i], y[j]);
```

Advanced: use weight to encode confidence near gap boundaries.

### 7.4 Multi-Dimensional DTW

Per-dimension missing data requires summing only over observed dimensions with normalization:

```cpp
data_t sum = 0; int count = 0;
for (size_t d = 0; d < dim; d++) {
    if (!is_missing(xi[d]) && !is_missing(yj[d])) {
        sum += metric(xi[d], yj[d]);
        count++;
    }
}
return (count > 0) ? sum * dim / count : 0;
```

Future concern -- DTWC++ currently handles only univariate series.

### 7.5 Subsequence DTW

Missing data at the tail is equivalent to a shorter query. Missing values within the search region use zero-cost handling as for full DTW. No special interaction.

---

## 8. Recommendation Summary

| Approach | Recommended? | When to Use |
| --- | --- | --- |
| Zero-Cost DTW + Normalization | **Primary (Tier 1)** | Default for all missing data scenarios |
| Linear Interpolation | **Secondary (Tier 2)** | Pre-processing utility, user's choice |
| DTW-AROW | Reference | Equivalent to our zero-cost approach |
| DTW-CAI | Future | When dataset-level imputation is needed |
| Uncertainty-DTW | Future/Research | When per-point confidence is available |
| GP-DTW | Not recommended | Too expensive for clustering workloads |
| Multiple Imputation | Not recommended | Too expensive, combining DTW is non-trivial |

**For DTWC++:** Implement Zero-Cost DTW with normalization as the core strategy (equivalent to simplified DTW-AROW), plus linear interpolation as an optional pre-processing utility. This approach has minimal code changes, preserves all existing optimizations (banding, memory-efficient), works with any distance metric, and preserves distance matrix symmetry needed by both PAM and MIP clustering.
