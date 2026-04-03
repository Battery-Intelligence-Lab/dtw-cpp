---
title: Cluster Quality Scores
weight: 9
---

# Cluster Quality Scores

DTW-C++ provides both **internal** and **external** cluster quality evaluation metrics. Internal metrics assess clustering quality using only the data and assignments (no ground truth needed). External metrics compare predicted labels against known ground-truth labels.

All scoring functions are in the `dtwc::scores` namespace (header: `scores.hpp`).

---

## Internal Metrics

Internal metrics evaluate clustering quality based on the distance matrix and cluster assignments. They are useful for model selection (choosing the best $$k$$) and comparing different clustering algorithms.

### Silhouette Score

```cpp
std::vector<double> dtwc::scores::silhouette(Problem &prob);
```

Computes the silhouette coefficient for each data point. For point $$i$$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\; b(i))}$$

where $$a(i)$$ is the mean distance from $$i$$ to other points in the same cluster, and $$b(i)$$ is the mean distance from $$i$$ to points in the nearest neighboring cluster.

- **Range:** $$[-1, 1]$$
- **Interpretation:** Values near 1 indicate well-clustered points; values near 0 indicate points on cluster boundaries; negative values indicate possible misassignment.
- **Returns:** A vector of per-point silhouette scores. The overall silhouette score is the mean of all values.

### Davies-Bouldin Index

```cpp
double dtwc::scores::daviesBouldinIndex(Problem &prob);
```

Measures the average similarity between each cluster and its most similar cluster. For each cluster $$i$$, it computes the ratio of within-cluster scatter to between-cluster separation, then averages the worst-case ratios:

$$\text{DBI} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$$

where $$\sigma_i$$ is the average distance of points in cluster $$i$$ to its medoid $$c_i$$.

- **Range:** $$[0, \infty)$$
- **Interpretation:** **Lower is better.** A value of 0 would mean perfectly separated clusters.

### Dunn Index

```cpp
double dtwc::scores::dunnIndex(Problem &prob);
```

Ratio of the minimum inter-cluster distance to the maximum intra-cluster diameter:

$$\text{DI} = \frac{\min_{i \ne j}\; d_{\text{inter}}(i, j)}{\max_i\; d_{\text{intra}}(i)}$$

- **Range:** $$(0, \infty)$$
- **Interpretation:** **Higher is better.** Large values indicate compact, well-separated clusters.

### Inertia

```cpp
double dtwc::scores::inertia(Problem &prob);
```

Total within-cluster distance: the sum of distances from each point to its assigned medoid.

$$\text{Inertia} = \sum_{i=1}^{n} d(x_i, c_{l(i)})$$

where $$c_{l(i)}$$ is the medoid of the cluster to which point $$x_i$$ is assigned.

- **Range:** $$[0, \infty)$$
- **Interpretation:** **Lower is better.** Useful for the elbow method when varying $$k$$.

### Calinski-Harabasz Index

```cpp
double dtwc::scores::calinskiHarabaszIndex(Problem &prob);
```

A medoid-adapted version of the Calinski-Harabasz (Variance Ratio) criterion. It measures the ratio of between-cluster dispersion to within-cluster dispersion, adjusted for the number of clusters and data points:

$$\text{CH} = \frac{\text{SS}_B / (k - 1)}{\text{SS}_W / (n - k)}$$

- **Range:** $$(0, \infty)$$
- **Interpretation:** **Higher is better.** Useful for selecting the optimal number of clusters.

---

## External Metrics

External metrics require ground-truth labels. They measure agreement between the predicted clustering and a known reference partition.

### Adjusted Rand Index

```cpp
double dtwc::scores::adjustedRandIndex(
    const std::vector<int> &labels_true,
    const std::vector<int> &labels_pred);
```

The Adjusted Rand Index (ARI) is a chance-corrected measure of agreement between two partitions. It counts pairs of points that are in the same or different clusters in both partitions, adjusted for the expected value under random labeling.

- **Range:** $$[-1, 1]$$ (in practice, $$[-0.5, 1]$$)
- **Interpretation:** 1.0 = perfect agreement; 0.0 = random labeling; negative values indicate worse-than-random agreement.

### Normalized Mutual Information

```cpp
double dtwc::scores::normalizedMutualInformation(
    const std::vector<int> &labels_true,
    const std::vector<int> &labels_pred);
```

Normalized Mutual Information (NMI) is an information-theoretic measure of the mutual dependence between two labelings, normalized to $$[0, 1]$$:

$$\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$$

where $$I(U; V)$$ is the mutual information and $$H(\cdot)$$ is the entropy.

- **Range:** $$[0, 1]$$
- **Interpretation:** 1.0 = perfect agreement; 0.0 = independent labelings.

---

## Choosing a Metric

| Metric | Ground truth needed? | Best for |
|--------|---------------------|----------|
| **Silhouette** | No | General cluster quality assessment; works with any $$k$$ |
| **Davies-Bouldin** | No | Comparing clusterings; penalizes overlapping clusters |
| **Dunn Index** | No | Detecting compact, well-separated clusters |
| **Inertia** | No | Elbow method for selecting $$k$$ |
| **Calinski-Harabasz** | No | Selecting optimal $$k$$; fast to compute |
| **Adjusted Rand Index** | Yes | Benchmark evaluation; robust to cluster count mismatch |
| **NMI** | Yes | Benchmark evaluation; information-theoretic |

For **model selection** (choosing $$k$$), silhouette and Calinski-Harabasz are the most widely used. For **benchmark evaluation** against ground truth, ARI is standard in the time series clustering literature.

---

## C++ Example

```cpp
#include <dtwc/Problem.hpp>
#include <dtwc/scores.hpp>

dtwc::Problem prob;
prob.set_data(std::move(data));
prob.set_numberOfClusters(3);
prob.cluster();

// Internal metrics
auto sil = dtwc::scores::silhouette(prob);
double dbi = dtwc::scores::daviesBouldinIndex(prob);
double di  = dtwc::scores::dunnIndex(prob);
double ine = dtwc::scores::inertia(prob);
double ch  = dtwc::scores::calinskiHarabaszIndex(prob);

// Mean silhouette score
double mean_sil = 0;
for (double s : sil) mean_sil += s;
mean_sil /= sil.size();

// External metrics (if ground truth is available)
std::vector<int> true_labels = {0, 0, 1, 1, 2, 2};
std::vector<int> pred_labels = prob.cluster_labels;  // from clustering

double ari = dtwc::scores::adjustedRandIndex(true_labels, pred_labels);
double nmi = dtwc::scores::normalizedMutualInformation(true_labels, pred_labels);
```

## Python Example

```python
import dtwcpp

prob = dtwcpp.Problem()
prob.set_data(series, names)
prob.set_number_of_clusters(3)
result = dtwcpp.fast_pam(prob, n_clusters=3)

# Internal metrics (results are stored back in prob)
sil = dtwcpp.silhouette(prob)           # list of per-point scores
dbi = dtwcpp.davies_bouldin_index(prob)  # lower is better
di  = dtwcpp.dunn_index(prob)            # higher is better
ine = dtwcpp.inertia(prob)               # lower is better
ch  = dtwcpp.calinski_harabasz_index(prob)  # higher is better

mean_silhouette = sum(sil) / len(sil)
print(f"Silhouette: {mean_silhouette:.3f}")
print(f"Davies-Bouldin: {dbi:.3f}")
print(f"Dunn: {di:.3f}")
print(f"Inertia: {ine:.3f}")
print(f"Calinski-Harabasz: {ch:.3f}")

# External metrics (no Problem needed, just label vectors)
true_labels = [0, 0, 1, 1, 2, 2]
pred_labels = list(result.labels)

ari = dtwcpp.adjusted_rand_index(true_labels, pred_labels)
nmi = dtwcpp.normalized_mutual_information(true_labels, pred_labels)
print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
```
