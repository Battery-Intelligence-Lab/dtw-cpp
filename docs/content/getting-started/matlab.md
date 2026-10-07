---
title: MATLAB Bindings
weight: 7
---

# MATLAB Bindings

DTWC++ provides MATLAB bindings through a MEX interface, wrapped in a clean `+dtwc` package that mirrors the Python API.

## Requirements

- MATLAB R2018a or later (the C MEX API, interleaved complex); R2019a or later to read Parquet
- A C++20 compiler supported by your MATLAB version
- CMake 3.26+

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

% Any variant, by the names dtwc_cl takes
d_soft = dtwc.distance.dtw(x, y, 'Variant', 'softdtw', 'SdtwGamma', 1.0);
fprintf('Soft-DTW distance: %.4f\n', d_soft);
```

The distance of two series is `dtwc.distance.dtw`, for every variant. The old
root-level helpers such as `dtwc.dtw_distance(...)` were removed in this
breaking release.

Its settings are name-value pairs, the `dtwc_cl` keys in CamelCase; C++ reads
and checks them, so an unknown name, a parameter outside its domain or a
combination no kernel implements raises `dtwc:invalidArgument`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | numeric vector | required | First time series |
| `y` | numeric vector | required | Second time series |
| `Variant` | char | `'standard'` | `'standard'`, `'ddtw'`, `'wdtw'`, `'adtw'`, `'softdtw'`, `'msm'`, `'twe'` |
| `Band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |
| `Metric` | char | `'l1'` | `'l1'` or `'squared_euclidean'` (Standard DTW and DDTW) |
| `MissingStrategy` | char | `'error'` | `'error'`, `'zero_cost'`, `'arow'` or `'interpolate'` (Standard DTW; NaN is then a missing value) |
| `WdtwG` | double | `0.05` | WDTW steepness |
| `AdtwPenalty` | double | `1.0` | ADTW non-diagonal step penalty |
| `SdtwGamma` | double | `1.0` | Soft-DTW smoothing |
| `MsmC` | double | `1.0` | MSM split/merge cost |
| `TweNu`, `TweLambda` | double | `0.001`, `1.0` | TWE stiffness and edit penalty |

## Clustering a dataset

The Tier-1 flow is Python's: set the device once, load a dataset, cluster it,
read the result.

```matlab
dtwc.device('cpu');                                   % or 'gpu', 'gpu:N'
data = dtwc.load('cycles.csv', 'SkipCols', 1);        % nothing is read yet
res = dtwc.cluster(data, 3, 'Band', 10, 'MaxIter', 50);
res.labels                                            % 1-based cluster of each series
res.medoids                                           % 1-based medoid of each cluster
res.score('silhouette')                               % the mean silhouette
res.save('out');                                      % the four CSVs dtwc_cl writes
```

`dtwc.cluster(data, k, Name, Value, ...)` takes these `dtwc_cl` keys, by
Python's words in CamelCase: `Method`, `Band`, `Metric`,
`Variant` and its parameters (`WdtwG`, `AdtwPenalty`, `SdtwGamma`, `MsmC`,
`TweNu`, `TweLambda`), `MvMode`, `MissingStrategy`, `MaxIter`, `NInit`, `Seed`,
`SampleSize`, `NSamples`, `BatchSize`, `Linkage`, `Dc`, `Solver` and the MIP
settings (`MipGap`, `TimeLimit`, `NoWarmStart`, `NumericFocus`, `MipFocus`,
`VerboseSolver`, `LrMaxNodes`), `GpuPrecision`, `Device`, `Name` and `Verbose`.
C++ reads and checks them before a series is read; a key not given takes
`dtwc_cl`'s default, so `Method` is `'auto'` (PAM on a GPU and for up to 5,000
series on the CPU, CLARA above), and an unknown key raises `dtwc:invalidArgument`
naming the valid ones. `Device` defaults to `dtwc.device()`; naming one sets
that run's device only.

## Loading data

A file goes through `dtwc.load`, which reads it as `dtwc_cl` and Python do; series
you have already read go in as they are, to `dtwc.load`, `dtwc.cluster`,
`DTWClustering.fit`, `compute_distance_matrix` or `Problem.set_data`:

```matlab
% A file or a folder: read by the same reader as dtwc_cl and Python
data = dtwc.load('cycles.csv', 'SkipRows', 1, 'SkipCols', 1);
data = dtwc.load('cycles/');                 % one series per file
data = dtwc.load('cycles.parquet');          % a series per row, or per file: see below

% Already read: a numeric matrix (one series per row) or a cell (any lengths)
X = readmatrix('cycles.csv', 'NumHeaderLines', 1);
res = dtwc.cluster(X(:, 2:end), 3);
res = dtwc.cluster({x1, x2, x3, x4}, 2);
```

| Key | Default | Meaning |
|-----|---------|---------|
| `SkipCols` | `0` | leading fields of each line or columns of a Parquet file (a file), or values of each series (in memory) |
| `SkipRows` | `0` | leading lines of a file, rows of a Parquet file, or leading series in memory |
| `Delimiter` | `''` | the field delimiter of text; `''` infers it from the extension |
| `Name` | `''` | the run's name: the file's name without its extension, the folder's name, or `'dataset'` |

CSV/TSV text and folders of it are read by the C++ reader in the MEX, the one
`dtwc_cl` and Python use. Parquet (`.parquet`, `.pq`, or a folder of them) is read
by MATLAB's `parquetread` (R2019a or later; the MEX links no Arrow) by the C++
reader's rule ([Supported data](../supported-data/)): a list column, or a row of
several Float32/Float64 columns, is one series per row, named by the file's first
string column, else `series_<i>`; the file's only Float32/Float64 column is one
series, named by its file. MATLAB
has no Arrow IPC reader, so an `.arrow`, `.ipc` or `.feather` file raises
`dtwc:invalidArgument`: read it elsewhere and pass the series in memory.

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
| `X` | numeric matrix (N x L) or cell of vectors | required | One series per row, or per cell (any lengths) |
| `Band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |

The returned matrix `D` is symmetric with zeros on the diagonal.

## DTWClustering class

`dtwc.DTWClustering` is Python's `dtwcpp.DTWClustering`: k-medoids clustering
with DTW distance, FastPAM by default. It is a value class, so `fit` returns the
fitted object.

### Basic usage

```matlab
clust = dtwc.DTWClustering('NClusters', 3, 'Band', 10);
clust = clust.fit(X);

fprintf('Total cost: %.2f\n', clust.Inertia);
fprintf('Medoid indices: ');
disp(clust.MedoidIndices);
labels = clust.predict(Y);       % the nearest medoid of each series of Y
```

### Properties

The settable properties are Python's parameters in CamelCase. One left empty
takes the C++ default, and C++ checks every value when `fit` runs: a value no run
can take (`MaxIter` 0, an unknown `Metric`) raises `dtwc:invalidArgument`.

| Property | Default | Description |
|----------|---------|-------------|
| `NClusters` | `3` | Number of clusters |
| `Method` | `'pam'` | Any `dtwc.cluster` method |
| `Variant`, `Band`, `Metric`, `MissingStrategy`, `WdtwG`, `AdtwPenalty`, `MsmC`, `TweNu`, `TweLambda` | empty | The distance, as `dtwc.distance.dtw` takes it |
| `MaxIter`, `NInit`, `MvMode`, `BatchSize`, `Device` | empty | As the `dtwc.cluster` keys; `Device` empty is `dtwc.device()` |
| `RandomState` | empty | Seed of the first restart (restart i uses `RandomState + i - 1`); empty is `dtwc.default_random_seed()` |

### Methods

- **`fit(X)`** -- Cluster `X` (an N x L numeric matrix, one series per row, or a cell of numeric vectors): one `Problem`, `NInit` seeded restarts on one distance matrix. Returns the fitted object.
- **`fit_predict(X)`** -- Fit and return the cluster labels.
- **`transform(X)`** -- The DTW distance of each series of `X` to each medoid (M x k).
- **`predict(X)`** -- The cluster of the nearest medoid of each series of `X`.
- **`score(X)`** -- Minus the total distance of the series of `X` to their nearest medoids. Neither `predict` nor `score` refits.

### Read-only properties (set after fit)

- `Labels` -- `double` row vector of cluster assignments (**1-based**)
- `MedoidIndices` -- `double` row vector of medoid indices (**1-based**)
- `Inertia` -- sum of the DTW distances of the series to their medoids
- `ClusterCenters` -- the medoid series, a cell

### Indexing note

All indices returned by the MATLAB bindings are **1-based**, consistent with MATLAB conventions. The C++ core uses 0-based indexing internally; the conversion is handled automatically.

## Complete example

This example reproduces the workflow from `examples/matlab/example_quickstart.m`:

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
clust = clust.fit(data);              % a value class: fit returns the fitted estimator
labels = clust.Labels;

fprintf('\nCluster labels (1-based):\n');
disp(labels);

fprintf('Cluster sizes: ');
for k = 1:3
    fprintf('%d ', sum(labels == k));
end
fprintf('\n');
fprintf('Total cost: %.2f\n', clust.Inertia);
fprintf('Medoid indices: ');
disp(clust.MedoidIndices);
```

## API correspondence with Python

The MATLAB and Python APIs are designed to mirror each other: the same words,
snake_case in Python and CamelCase in MATLAB's name-value keys and properties.

| Python | MATLAB | Notes |
|--------|--------|-------|
| `dtwcpp.load(path, skip_cols=1)` | `dtwc.load(path, 'SkipCols', 1)` | |
| `dtwcpp.cluster(data, k=3, max_iter=50)` | `dtwc.cluster(data, 3, 'MaxIter', 50)` | Name-value pairs |
| `dtwcpp.distance.dtw(x, y)` | `dtwc.distance.dtw(x, y)` | Preferred namespace |
| `dtwcpp.compute_distance_matrix(X)` | `dtwc.compute_distance_matrix(X)` | |
| `DTWClustering(n_clusters=3)` | `DTWClustering('NClusters', 3)` | Name-value pairs |
| `clf.fit_predict(X)` | `clust.fit_predict(X)` | |
| `clf.labels_` | `clust.Labels` | 0-based vs 1-based |
| `clf.medoid_indices_` | `clust.MedoidIndices` | 0-based vs 1-based |
| `clf.inertia_` | `clust.Inertia` | |
| `prob.band`, `prob.max_iter` | `prob.Band`, `prob.MaxIter` | Read-only in MATLAB: `set_band`, `set_max_iter` |
| `Problem("p", device="gpu")` | `dtwc.Problem('p', 'Device', 'gpu')` | a `Problem`'s device; it does not follow `dtwc.device()` |
| `prob.set_device("cpu")` | `prob.set_device('cpu')` | same names as `dtwc.device()`; `'hpc'` is Python's alone and is `dtwc:deviceError` here |
