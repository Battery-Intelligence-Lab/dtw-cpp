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

Settings are passed as CamelCase name-value pairs matching the `dtwc_cl` keys; C++ validates them and raises `dtwc:invalidArgument` for unknown names, out-of-range parameters, or unsupported combinations:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | numeric vector | required | First time series |
| `y` | numeric vector | required | Second time series |
| `Variant` | char | `'standard'` | `'standard'`, `'ddtw'`, `'wdtw'`, `'adtw'`, `'softdtw'`, `'msm'`, `'twe'` |
| `Band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |
| `Metric` | char | `'l1'` | `'l1'`, `'euclidean'` or `'squared_euclidean'` (Standard DTW and DDTW) |
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
dtwc.device('cpu');                                   % or 'gpu', 'gpu:N', 'cpu+gpu'
data = dtwc.load('cycles.csv', 'SkipCols', 1);        % nothing is read yet
res = dtwc.cluster(data, 3, 'Band', 10, 'MaxIter', 50);
res.labels                                            % 1-based cluster of each series
res.medoids                                           % 1-based medoid of each cluster
res.score('silhouette')                               % the mean silhouette
res.save('out');                                      % the CSVs dtwc_cl writes, then the run report
res.Report                                            % the run report's text
```

`dtwc.cluster` runs `dtwc::run` in memory unless `'Output'` names a directory (`res.save` writes the CSV outputs and `<name>_report.txt`; see [Tier 1](../../api/tier-1/)). It accepts the `dtwc_cl` options ([CLI reference](../cli/)) in CamelCase: `MatrixDtype`, `RamLimit`, `Method`, `Band`, `Metric`, `Variant`, `WdtwG`, `AdtwPenalty`, `SdtwGamma`, `MsmC`, `TweNu`, `TweLambda`, `MvMode`, `MissingStrategy`, `MaxIter`, `NInit`, `Seed`, `SampleSize`, `NSamples`, `BatchSize`, `Linkage`, `Dc`, `Solver`, `MipGap`, `TimeLimit`, `NoWarmStart`, `NumericFocus`, `MipFocus`, `VerboseSolver`, `LrMaxNodes`, `GpuPrecision`, `GpuShare`, `Device`, `Fallback` (a `Device` `'gpu'` that cannot be honoured runs on the CPU, saying so on stderr and in `res.Report`, and `res.device` is `'cpu'`, instead of `dtwc:deviceError`; this run only), `Name`, `Verbose`, `NoReport` (no run report: `res.Report` is `''`), plus `Dtype` when loading a path or in-memory series directly, and the run's files: `Output`, the directory it writes its result files and run report into (`''`, the default, writes none), `Checkpoint`, the directory it saves its distance matrix in and resumes from, and `CheckpointInterval`, the rows between saves (it needs a `Checkpoint`). A name given twice takes its last value, so `dtwc.cluster(data, 3, opts{:}, 'Band', 2)` overrides the `Band` kept in `opts`. C++ reads and checks the keys before a series is read: omitted keys use `dtwc_cl` defaults (`Method` = `'auto'`, `Device` = `dtwc.device()`, naming one sets that run's device only), and an unknown key raises `dtwc:invalidArgument` naming the valid ones. Streamed Arrow IPC runs under `RamLimit` hold no series (`score`, `distance_matrix`, and `plot` raise `dtwc:invalidArgument`; `save` writes labels, medoids, and the report).

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
| `Dtype` | `'float64'` | the precision the series are stored in: `'float32'` keeps them `single`, half the memory, each value the double read rounded; `single` data alone stays `double` |
| `Ndim` | `1` | the features per time step of a multivariate series: each series holds its steps interleaved, `[t0f0 t0f1 t1f0 ...]`, so its length is a multiple of `Ndim`; an Arrow IPC file gives its own in its metadata, and one given above 1 that differs is refused when the file is read; an `Ndim` above 1 makes `dtwc.cluster` read a path into memory, so an Arrow IPC file, which carries its own, streams under `RamLimit` only if `Ndim` is left out |

CSV/TSV text, folders of it and Arrow IPC (`.arrow`, `.ipc`, `.feather`: mapped, its
series views of the file; decoded or read into memory where it cannot be viewed in place,
as a compressed file) are read by the C++ reader in the MEX, the one `dtwc_cl` and
Python use.
Parquet (`.parquet`, `.pq`, or a folder of them) is read
by MATLAB's `parquetread` (R2019a or later; the MEX links no Arrow) by the C++
reader's rule ([Supported data](../supported-data/)): a list column, or a row of
several Float32/Float64 columns, is one series per row, named by the file's first
string column, else `series_<i>`; the file's only Float32/Float64 column is one
series, named by its file.

`dtwc.load` reads nothing, and a `Dataset` of a path keeps no series: `dtwc.cluster`
hands a text file, a folder or an Arrow IPC file to C++, which reads it at every
call. Parquet is the exception, and so is a `Dataset` of `Ndim` above 1:
`Dataset.as_series()` reads it at the first `dtwc.cluster` call (`parquetread` for
Parquet) and keeps the series, which are clustered in the Dataset's `Ndim`. To read any file once for several
runs, take its series and cluster those (`'Dtype'` again for `'float32'`):

```matlab
[series, names] = data.as_series();
res = dtwc.cluster(series, 3);
```

The clustering is the path's, with two differences. The result files name series in
memory `0`, `1`, ..., not by `names`. And `dtwc.cluster` takes no `Ndim` keyword (give it to
`dtwc.load`), so the third output of `as_series()`, the `Ndim`, does not reach it: an Arrow IPC
file whose schema says `ndim` above 1 clusters as multivariate from its path, and as
univariate series (each row's interleaved values) from `as_series()`.

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
| `MatrixDtype` | char | `'float64'` | `'float64'` or `'float32'`: the matrix's value type and the class of `D` (`double` or `single`); `'float32'` takes half the memory, each distance rounded to a single |

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
| `MaxIter`, `NInit`, `MvMode`, `BatchSize`, `Device`, `Fallback` | empty | As the `dtwc.cluster` keys; `Device` empty is `dtwc.device()` |
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
| `Problem("p", device="gpu")` | `dtwc.Problem('p', 'Device', 'gpu')` | a `Problem`'s device; it does not follow `dtwc.device()`; `'Fallback', true` (Python `fallback=True`) computes a call the GPU cannot honour on the CPU, saying so |
| `prob.set_gpu_share(0.5)` | `prob.set_gpu_share(0.5)` | the share of the work device `cpu+gpu` splits; `-1` measures it |
| `prob.set_device("cpu")` | `prob.set_device('cpu')` | same names as `dtwc.device()` (and `'Fallback', true`: `prob.set_device("gpu", fallback=True)`); `'hpc'` is an option of Python's `cluster()` and `dtwcpp.device()` only, and is `dtwc:deviceError` here |
| `prob.read_distance_matrix(path, not_computed="negative")` | `prob.read_distance_matrix(path, 'NotComputed', 'negative')` | a v1.0.0 file's `-1` (any value below 0) marks a pair to compute; the default, `'nan'`, an empty field or NaN |
