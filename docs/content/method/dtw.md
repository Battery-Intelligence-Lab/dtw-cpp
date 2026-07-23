---
title: Dynamic Time Warping
weight: 1
---

# Dynamic time warping

[Dynamic time warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) is a technique for manipulating time series data to enable comparisons between datasets, using local warping (stretching or compressing along the time axis) of the elements within each time series to find an optimal alignment between series. Unlike traditional distance measures such as Euclidean distances, the local warping in DTW can capture similarities that linear alignment methods might miss. By emphasising _shape_ similarity rather than strict temporal alignment, DTW is particularly useful in scenarios where the exact timing of occurrences is less important for the analysis.

## The DTW algorithm

Consider a time series to be a vector of some arbitrary length. Consider that we have $$p$$ such vectors in total, each possibly differing in length. To find a subset of $$k$$ clusters from within the total set of $$p$$ vectors, where each cluster contains similar vectors, we must first make $${p \choose 2} = \frac{p(p-1)}{2}$$ pairwise comparisons between all vectors within the total set and find the `similarity' between each pair. In this case, the similarity is defined as the DTW distance between a pair of vectors. Consider two time series $$x$$ and $$y$$ of differing lengths $$n$$ and $$m$$ respectively,

$$
x=(x_1, x_2, ..., x_n)
$$

$$
y=(y_1, y_2, ..., y_m).
$$

The DTW distance is the sum of the pointwise distance between each point and its matched point(s) in the other vector. In DTW-C++, the default pointwise metric is the absolute difference (L1 metric), i.e., $$|x_i - y_j|$$. The following constraints must be met:

1. The first and last elements of each series must be matched.
2. Only unidirectional forward movement through relative time is allowed, i.e., if $$x_1$$ is mapped to $$y_2$$ then $$x_2$$ may not be mapped to
    $$y_1$$ (this ensures monotonicity). 
3. Each point is mapped to at least one other point, i.e., there are no jumps in time (this ensures continuity).

Finding the optimal warping arrangement is an optimisation problem that can be solved using dynamic programming, which splits the problem into easier sub-problems and solves each of them recursively, storing intermediate solutions until the final solution is reached. To understand the memory-efficient method used in DTW-C++, it is useful to first examine the full-cost matrix solution. Using one-based series indices and a padded boundary, define

$$
C_{0,0}=0,\qquad
C_{i,0}=+\infty\ (i>0),\qquad
C_{0,j}=+\infty\ (j>0).
$$

For $$1\le i\le n$$ and $$1\le j\le m$$, each element is the minimum cumulative cost of a path ending at the pair $$x_i,y_j$$:

$$
C_{i,j} = |x_i-y_j|+\min \begin{cases}
    C_{i-1,j-1}\\
    C_{i-1,j}\\
    C_{i,j-1}.
    \end{cases}
$$

The three predecessors correspond to a diagonal match, advancing only in $$x$$, or advancing only in $$y$$. Every admissible path reaches $$(i,j)$$ through exactly one of them, so an optimal path must contain an optimal predecessor path; otherwise replacing its prefix would lower the total cost.

> **Cost convention:** The formula above uses the absolute difference (L1 local cost), which is the default in DTW-C++. The alternative squared-L2 local cost is $$(x_i-y_j)^2$$. If amplitudes have unit $$U$$, the accumulated results have units $$U$$ and $$U^2$$ respectively; DTWC++ does not take a final square root. Warping means that both accumulated forms are dissimilarities, not a metric: distinct repeated-value sequences can have zero cost, and the triangle inequality can fail.

The final element $$C_{n,m}$$ is the DTW dissimilarity between the two series. Below is an example of the cost matrix $$C$$ and the warping path through it.

As an example, below are two time series with DTW pairwise alignment between elements. On the right is the cost matrix $$C$$ for the two time series, showing the warping path and final DTW cost at element $$C_{14,13}$$.

<img src="/method/dtw_image.png" alt="Two time series with DTW pairwise alignment between each element, showing one-to-many mapping properties of DTW (left). Cost matrix $$C$$ for the two time series, showing the warping path and final DTW cost at $$C_{14,13}$$ (right)." caption="Two time series with DTW pairwise alignment between each element, showing one-to-many mapping properties of DTW (left). Cost matrix $$C$$ for the two time series, showing the warping path and final DTW cost at $$C_{14,13}$$ (right).">

For the clustering problem, only the final cost for each pairwise comparison is required; the actual warping path (or mapping of each point in one time series to the other) is superfluous for clustering. The memory complexity of the cost matrix $$C$$ is $$O(nm)$$, so as the length of the time series increases, the memory required increases greatly. Therefore, significant reductions in memory can be made by not storing the entire $$C$$ matrix. When the warping path is not required, only a vector containing the previous row for the current step of the dynamic programming sub-problem is required (i.e., the previous three values $$c_{i-1,j-1}$$, $$c_{i-1,j}$$, $$c_{i,j-1}$$).

In DTW-C++, the DTW distance $$C_{x,y}$$ is found for each pairwise comparison. Pairwise distances are then stored in a separate symmetric matrix, $$D^{p\times p}$$, where ($$p$$) is the total number of time series in the clustering exercise. In other words, the element $$d_{i,j}$$ gives the distance between time series ($$i$$) and ($$j$$).

## Z-Normalization

Before computing DTW distances, it is often beneficial to **z-normalize** each time series to have zero mean and unit standard deviation:

$$
\hat{x}_i = \frac{x_i - \mu_x}{\sigma_x}
$$

where $$\mu_x = \frac{1}{n}\sum_{i=1}^{n} x_i$$ and $$\sigma_x = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_x)^2}$$.

### Why z-normalize?

Z-normalization provides **amplitude invariance**: two time series with the same shape but different scales or offsets will have a small DTW distance after normalization. Without normalization, a constant vertical shift or scaling difference between series can dominate the DTW cost, masking genuine shape similarity.

For example, consider two temperature sensor readings that follow the same daily pattern but are offset by a calibration difference. Without z-normalization, DTW would report a large distance driven by the offset. After z-normalization, the distance reflects only the shape difference.

### When to use z-normalization

* **Recommended** when comparing time series from different sources, sensors, or scales where amplitude differences are not meaningful.
* **Recommended** for the UCR time series benchmark and most classification/clustering tasks in the literature.
* **Not recommended** when absolute amplitude carries important information (e.g., comparing actual energy consumption values where magnitude matters).

### Interaction with DTW

Z-normalization is applied as a **preprocessing step** before DTW computation. The normalization does not affect the DTW algorithm itself -- it only transforms the input data. The warping path found by DTW on z-normalized data may differ from the path on raw data, as the pointwise distances change.

DTW-C++ provides two functions for z-normalization:

* `z_normalize(series)` -- normalizes the series in place
* `z_normalized(series)` -- returns a new normalized copy, leaving the original unchanged

Both functions handle the edge case of constant series ($$\sigma_x = 0$$) by returning a zero series.

## Warping Window

For longer time series it is possible to reduce the calculation domain by using a fixed Sakoe–Chiba warping window. On DTWC++'s CPU routes, a cell is admissible exactly when

$$|i-j| \le w$$

for a non-negative integer half-width $$w$$. For example, with two length-100 series and $$w=10$$, $$x_1$$ may be matched only to $$y_1,\ldots,y_{11}$$. Setting $$w=0$$ forces strictly diagonal alignment for equal-length series, while $$w=1$$ permits a shift of one position.

For non-empty series, because the terminal cell is $$(n,m)$$, an endpoint-preserving path exists if and only if

$$w \ge |n-m|.$$

Below this threshold the CPU routes return the finite no-path sentinel `numeric_limits<T>::max()`. Empty input and an exceeded early-abandon cutoff use the same value, so a mere `isfinite(result)` check does not prove that a path was evaluated. A negative CPU API band requests unconstrained DTW.

Widening the window only adds admissible paths. Therefore the exact banded value is non-increasing in $$w$$ and is always at least the full-DTW value. A narrower band can reduce work, but it can also increase the dissimilarity or eliminate every path; choose it from the timing variation allowed by the application. The fixed-window definition and endpoint conditions come from [Sakoe and Chiba (1978)](https://doi.org/10.1109/TASSP.1978.1163055).

Cross-backend parity remains open under finding F12. CUDA source currently uses an endpoint-scaled corridor rather than the fixed window above. Metal source uses fixed geometry, but its double-returning no-path route widens `FLT_MAX` rather than returning the CPU `DBL_MAX` sentinel. Until executable backend gates close F12, the exact geometry and sentinel in this section describe the CPU routes.
