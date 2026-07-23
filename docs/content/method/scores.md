---
title: Cluster Quality Scores
weight: 9
---

# Cluster Quality Scores

DTW-C++ exposes internal metrics for an already-clustered `Problem` and
external metrics for two label vectors. The canonical C++ functions are in
`dtwc::scores` in `scores.hpp`; Python exports the same snake_case names from
`dtwcpp`.

## Internal metrics

Internal metrics read the assignments, medoids, and distance matrix stored in
the `Problem`.

### Silhouette

```cpp
std::vector<double> dtwc::scores::silhouette(Problem &prob);
```

For point $$i$$,

$$s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))},$$

where $$a(i)$$ is the mean dissimilarity to the other members of its cluster
and $$b(i)$$ is the smallest mean dissimilarity to another non-empty cluster.
Ordinary finite results lie in $$[-1,1]$$.

The function returns one value per series. An unclustered problem returns a
vector filled with `-1` after printing a diagnostic. A singleton cluster
member receives `0`. If a non-singleton point has both means equal to zero,
the implemented `0/0` expression can return NaN; callers that aggregate the
vector must decide how to handle non-finite values.

### Davies-Bouldin

```cpp
double dtwc::scores::davies_bouldin(Problem &prob);
```

For cluster medoids $$c_i$$ and mean within-cluster scatter $$S_i$$,

$$\mathrm{DBI} = \frac{1}{k}\sum_i
  \max_{j\ne i}\frac{S_i+S_j}{d(c_i,c_j)}.$$

This implementation requires at least two clusters. A pair with
zero medoid separation is skipped rather than divided by zero. Lower finite
values indicate smaller within-cluster scatter relative to medoid separation.

### Dunn

```cpp
double dtwc::scores::dunn(Problem &prob);
```

The implementation uses all pairwise series dissimilarities:

$$\mathrm{Dunn} =
  \frac{\min\{d(x_i,x_j): \ell_i\ne\ell_j\}}
       {\max\{d(x_i,x_j): \ell_i=\ell_j,\ i<j\}}.$$

It requires at least two clusters. If the maximum intra-cluster diameter is
zero, it returns positive infinity. Higher finite values indicate greater
separation relative to cluster diameter.

### Inertia

```cpp
double dtwc::scores::inertia(Problem &prob);
```

Inertia is the sum of the stored, unsquared dissimilarities from each series
to its assigned medoid:

$$\mathrm{Inertia} = \sum_i d(x_i,c_{\ell_i}).$$

The result is nonnegative only when the selected dissimilarity itself is
nonnegative. Lower values mean a smaller total medoid-assignment cost for the
same dissimilarity contract.

### Calinski-Harabasz

```cpp
double dtwc::scores::calinski_harabasz(Problem &prob);
```

The medoid-adapted calculation uses squared stored distances. Its within term
is the sum of squared distances to each cluster medoid; its between term is
the cluster-size-weighted sum of squared distances from cluster medoids to an
overall medoid:

$$\mathrm{CH} = \frac{B/(k-1)}{W/(N-k)}.$$

It requires $$k>1$$ and $$N>k$$. If the within term $$W$$ is zero, the function
returns positive infinity.

## External metrics

Both external functions require label vectors of equal length.

### Adjusted Rand

```cpp
double dtwc::scores::adjusted_rand(
    const std::vector<int> &labels_true,
    const std::vector<int> &labels_pred);
```

The Adjusted Rand score compares the pair-count contingency table after its
chance correction. Identical non-degenerate partitions return `1`; a
denominator-zero case also returns `1` in the current implementation.

### Normalized mutual information

```cpp
double dtwc::scores::normalized_mutual_info(
    const std::vector<int> &labels_true,
    const std::vector<int> &labels_pred);
```

The implemented arithmetic-mean normalization is

$$\mathrm{NMI}(U,V)=\frac{2I(U;V)}{H(U)+H(V)}.$$

An empty input returns `0`. For a non-empty input whose two marginal entropies
sum to zero, the function returns `1`.

## C++ example

```cpp
#include <dtwc/Problem.hpp>
#include <dtwc/scores.hpp>

dtwc::Problem prob;
prob.set_data(std::move(data));
prob.set_n_clusters(3);
prob.cluster();

auto sil = dtwc::scores::silhouette(prob);
double dbi = dtwc::scores::davies_bouldin(prob);
double di = dtwc::scores::dunn(prob);
double ine = dtwc::scores::inertia(prob);
double ch = dtwc::scores::calinski_harabasz(prob);

std::vector<int> true_labels{0, 0, 1, 1, 2, 2};
const auto &pred_labels = prob.labels();
double ari = dtwc::scores::adjusted_rand(true_labels, pred_labels);
double nmi = dtwc::scores::normalized_mutual_info(true_labels, pred_labels);
```

## Python example

```python
import dtwcpp

prob = dtwcpp.Problem("scores")
prob.set_data(series, names)
prob.set_n_clusters(3)
result = dtwcpp.fast_pam(prob, n_clusters=3)

sil = dtwcpp.silhouette(prob)
dbi = dtwcpp.davies_bouldin(prob)
di = dtwcpp.dunn(prob)
ine = dtwcpp.inertia(prob)
ch = dtwcpp.calinski_harabasz(prob)

true_labels = [0, 0, 1, 1, 2, 2]
pred_labels = list(result.labels)
ari = dtwcpp.adjusted_rand(true_labels, pred_labels)
nmi = dtwcpp.normalized_mutual_info(true_labels, pred_labels)
```
