---
title: MATLAB Bindings
weight: 7
---

# MATLAB Bindings

DTWC++ provides MATLAB bindings through a MEX interface, wrapped in a clean `+dtwc` package that mirrors the Python API.

## Requirements

- MATLAB R2018a or later (C++ MEX API with `mex.hpp`)
- A C++17 compiler supported by your MATLAB version
- CMake 3.15+

## Building

Configure the project with the MATLAB flag enabled:

```bash
mkdir build && cd build
cmake .. -DDTWC_BUILD_MATLAB=ON
cmake --build . --config Release
```

This produces the `dtwc_mex` MEX file. Ensure it is on your MATLAB path along with the `+dtwc` package directory.

## DTW distance

Compute the DTW distance between two time series using the `dtwc.distance`
namespace:

```matlab
x = [1 2 3 4 5];
y = [2 4 6 3 1];

d = dtwc.distance.dtw(x, y);
fprintf('DTW distance: %.4f\n', d);

% Banded DTW (Sakoe-Chiba constraint)
d_banded = dtwc.distance.dtw(x, y, 'Band', 2);
fprintf('DTW distance (band=2): %.4f\n', d_banded);

% Variant dispatch convenience
d_soft = dtwc.distance.dtw(x, y, 'Variant', 'soft_dtw', 'Gamma', 1.0);
fprintf('Soft-DTW distance: %.4f\n', d_soft);
```

Pairwise distances now live under `dtwc.distance.*`. The old root-level
helpers such as `dtwc.dtw_distance(...)` were removed in this breaking
release.

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | numeric vector | required | First time series |
| `y` | numeric vector | required | Second time series |
| `Band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |

## Distance matrix

Compute the full NxN pairwise DTW distance matrix:

```matlab
rng(42);
X = randn(50, 100);  % 50 series of length 100

D = dtwc.compute_distance_matrix(X, 'Band', 5);
fprintf('Distance matrix: %dx%d\n', size(D));
fprintf('Symmetric: %d\n', issymmetric(D));
```

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | double matrix (N x L) | required | Each row is a time series of length L |
| `Band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |

The returned matrix `D` is symmetric with zeros on the diagonal.

## DTWClustering class

`dtwc.DTWClustering` is a handle class for k-medoids clustering with DTW distance, implementing FastPAM.

### Basic usage

```matlab
clust = dtwc.DTWClustering('NClusters', 3, 'Band', 10);
labels = clust.fit_predict(X);

fprintf('Cluster labels:\n');
disp(labels);
fprintf('Total cost: %.2f\n', clust.TotalCost);
fprintf('Medoid indices: ');
disp(clust.MedoidIndices);
```

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NClusters` | int | `3` | Number of clusters |
| `Band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |
| `Metric` | char | `'l1'` | Pointwise distance metric |
| `MaxIter` | int | `100` | Maximum clustering iterations |
| `NInit` | int | `1` | Number of random restarts (best result kept) |
| `Variant` | char | `'standard'` | DTW variant: `'standard'`, `'ddtw'`, `'wdtw'`, `'adtw'`, `'softdtw'` |
| `WdtwG` | double | `0.05` | WDTW steepness parameter |
| `AdtwPenalty` | double | `1.0` | ADTW non-diagonal step penalty |
| `MissingStrategy` | char | `'error'` | NaN handling: `'error'`, `'zero_cost'`, `'arow'`, `'interpolate'` |

### Methods

- **`fit(X)`** -- Run k-medoids clustering on the data matrix `X` (N x L). Returns the object with updated properties.
- **`fit_predict(X)`** -- Fit and return cluster labels.
- **`predict(X)`** -- Assign new data to nearest medoids (requires prior `fit`).

### Read-only properties (set after fit)

- `Labels` -- `int32` row vector of cluster assignments (**1-based**)
- `MedoidIndices` -- `int32` row vector of medoid indices (**1-based**)
- `TotalCost` -- sum of intra-cluster DTW distances

### Indexing note

All indices returned by the MATLAB bindings are **1-based**, consistent with MATLAB conventions. The C++ core uses 0-based indexing internally; the conversion is handled automatically.

## Complete example

This example reproduces the workflow from `examples/example_quickstart.m`:

```matlab
%% 1. Pairwise DTW distance
x = sin(linspace(0, 2*pi, 100));
y = cos(linspace(0, 2*pi, 100));

d = dtwc.distance.dtw(x, y);
fprintf('DTW distance (sin vs cos): %.4f\n', d);

d_banded = dtwc.distance.dtw(x, y, 'Band', 10);
fprintf('DTW distance (band=10):    %.4f\n', d_banded);

%% 2. Distance matrix
rng(42);
N = 20;
L = 100;
data = randn(N, L);

dm = dtwc.compute_distance_matrix(data);
fprintf('\nDistance matrix: %dx%d\n', size(dm));
fprintf('Min non-zero: %.4f\n', min(dm(dm > 0)));
fprintf('Max:          %.4f\n', max(dm(:)));

%% 3. Clustering
clust = dtwc.DTWClustering('NClusters', 3, 'Band', 10);
labels = clust.fit_predict(data);

fprintf('\nCluster labels (1-based):\n');
disp(labels);

fprintf('Cluster sizes: ');
for k = 1:3
    fprintf('%d ', sum(labels == k));
end
fprintf('\n');
fprintf('Total cost: %.2f\n', clust.TotalCost);
fprintf('Medoid indices: ');
disp(clust.MedoidIndices);
```

## API correspondence with Python

The MATLAB and Python APIs are designed to mirror each other where reasonable:

| Python | MATLAB | Notes |
|--------|--------|-------|
| `dtwcpp.distance.dtw(x, y)` | `dtwc.distance.dtw(x, y)` | Preferred namespace |
| `dtwcpp.compute_distance_matrix(X)` | `dtwc.compute_distance_matrix(X)` | |
| `DTWClustering(n_clusters=3)` | `DTWClustering('NClusters', 3)` | Name-value pairs |
| `clf.fit_predict(X)` | `clust.fit_predict(X)` | |
| `clf.labels_` | `clust.Labels` | 0-based vs 1-based |
| `clf.medoid_indices_` | `clust.MedoidIndices` | 0-based vs 1-based |
| `clf.inertia_` | `clust.TotalCost` | |


