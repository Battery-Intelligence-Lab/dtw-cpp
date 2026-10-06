---
title: Python API
weight: 6
---

# Python API

The `dtwcpp` package provides fast DTW distance computation and time-series clustering from Python, backed by the C++ core library.

## Installation

```bash
pip install dtwcpp
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install dtwcpp
```

Exact MIP clustering (`method="mip"`) solves with highspy, which the `mip` extra installs
(the wheel links no MIP solver; without highspy, `method="mip"` raises `SolverError` naming it):

```bash
pip install dtwcpp[mip]
```

For GPU support, install the CUDA-enabled build:

```bash
pip install dtwcpp[cuda]
```

## Quick start

### DTW distance between two series

```python
import dtwcpp

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 3.0, 1.0]

d = dtwcpp.distance.dtw(x, y)
print(f"DTW distance: {d}")

# Banded DTW (Sakoe-Chiba constraint)
d_banded = dtwcpp.distance.dtw(x, y, band=2)
print(f"DTW distance (band=2): {d_banded}")

# Any variant, by the names dtwc_cl takes
d_soft = dtwcpp.distance.dtw(x, y, variant="softdtw", sdtw_gamma=1.0)
print(f"Soft-DTW distance: {d_soft}")
```

Both plain Python lists and NumPy arrays are accepted. Lists are automatically converted to `float64` arrays.

The distance of two series is `dtwcpp.distance.dtw`, for every variant. The
old root-level distance helpers such as `dtwcpp.dtw_distance(...)` were removed
in this breaking release.

### Distance matrix

Compute the full pairwise DTW distance matrix for a collection of series:

```python
import numpy as np
import dtwcpp

series = [np.sin(np.linspace(0, 2 * np.pi, 100) + phase)
          for phase in np.linspace(0, np.pi, 20)]

D = dtwcpp.compute_distance_matrix(series, band=10)
print(f"Distance matrix shape: {D.shape}")  # (20, 20)
```

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series` | list of list/array | required | Input time series |
| `band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |
| `metric` | str | `"l1"` | `"l1"` or `"squared_euclidean"` |
| `device` | str | `None` | `"cpu"`, `"gpu"`, `"gpu:N"` (`"cuda"` / `"cuda:N"` are aliases); `None` uses `dtwcpp.device()` |

## DTWClustering class

`DTWClustering` provides an sklearn-compatible interface for k-medoids clustering with DTW distance. Its fit is
`dtwcpp.cluster`'s C++ run, FastPAM (Schubert & Rousseeuw, 2021) unless `method` says otherwise, with `n_init` seeded
restarts on one distance matrix.

```python
import numpy as np
import dtwcpp

rng = np.random.RandomState(42)
group_a = rng.randn(10, 50)
group_b = rng.randn(10, 50) + 5
group_c = rng.randn(10, 50) + 10
X = np.vstack([group_a, group_b, group_c])

clf = dtwcpp.DTWClustering(n_clusters=3, band=10)
labels = clf.fit_predict(X)

print(f"Labels:         {labels}")
print(f"Inertia:        {clf.inertia_:.2f}")
print(f"Medoid indices: {clf.medoid_indices_}")
print(f"Iterations:     {clf.n_iter_}")
```

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | `3` | Number of clusters |
| `method` | str | `"pam"` | Any `dtwcpp.cluster` method; with `metric="precomputed"` one that reads the matrix (`pam`, `kmedoids`, `mip`, `lrcore`, `hierarchical`) |
| `variant` | str | `"standard"` | DTW variant: `"standard"`, `"ddtw"`, `"wdtw"`, `"adtw"`, `"msm"`, `"twe"` |
| `band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |
| `max_iter` | int | `100` | Maximum iterations of the method (at least 1) |
| `n_init` | int | `1` | Seeded restarts `random_state + i`, the best kept (at least 1) |
| `wdtw_g` | float | `0.05` | WDTW logistic weight steepness (only for `variant="wdtw"`) |
| `adtw_penalty` | float | `1.0` | ADTW non-diagonal step penalty (only for `variant="adtw"`) |
| `msm_c` | float | `1.0` | MSM split/merge cost (only for `variant="msm"`) |
| `twe_nu` | float | `0.001` | TWE stiffness (only for `variant="twe"`) |
| `twe_lambda` | float | `1.0` | TWE edit penalty (only for `variant="twe"`) |
| `mv_mode` | str | `"dependent"` | Multivariate mode: `"dependent"` or `"independent"` |
| `missing_strategy` | str | `"error"` | NaN handling: `"error"`, `"zero_cost"`, `"arow"`, `"interpolate"` |
| `metric` | str | `"l1"` | Pointwise metric: `"l1"` or `"squared_euclidean"`; `"precomputed"`: `X` is an N x N distance matrix for `fit`, M x N for `predict`/`transform`/`score` |
| `batch_size` | int | `-1` | OneBatchPAM's batch size (`-1`: automatic) |
| `random_state` | int | `None` | The seed; `None` is `dtwcpp.DEFAULT_RANDOM_SEED` (42) |
| `device` | str | `None` | Local `"cpu"`/`"gpu"`/`"cuda:N"`, or whole-job `"hpc"` offload |

The estimator validates the complete distance contract before computing.
Squared Euclidean cost is available for Standard dependent DTW with
`missing_strategy="error"`; the non-Standard variants retain their intrinsic L1
cost. Non-error missing handling requires Standard, dependent, L1 DTW, and
independent multivariate mode requires Standard, Error, L1 DTW. Local CUDA and
Metal execution support Standard, dependent, Error mode with either metric.
`fit()` and `predict()` use the same validated recurrence and parameters.

### Methods

- **`fit(X)`** -- Fit clustering on `X`: a 2-D array, a list of 1-D arrays (any lengths), a pandas DataFrame or an Arrow array, one series per row. Returns `self`.
- **`predict(X)`** -- Assign each series in `X` to the nearest medoid.
- **`transform(X)`** -- Distances from each series in `X` to each medoid.
- **`fit_predict(X)`** -- Fit and return cluster labels.
- **`score(X)`** -- Negative total distance of `X` to its nearest medoids (for sklearn grid search); nothing is refitted.

### Attributes (set after `fit`)

- `labels_` -- cluster labels (ndarray of shape `(n_samples,)`)
- `medoid_indices_` -- indices of medoid series
- `cluster_centers_` -- list of medoid time-series arrays
- `inertia_` -- total within-cluster cost
- `n_iter_` -- number of FastPAM iterations

### Predict on new data

```python
new_series = rng.randn(3, 50) + 5  # should match group_b
predicted = clf.predict(new_series)
print(f"Predicted labels: {predicted}")
```

## DTW functions

`dtwcpp.distance.dtw(x, y, ...)` computes every variant. Its settings are named as
the `dtwc_cl` keys, and C++ reads and checks them: an unknown name, a parameter
outside its domain or a combination no kernel implements raises `InvalidInput`.
Lists and NumPy arrays are accepted.

```python
d = dtwcpp.distance.dtw(x, y, band=-1, metric="l1")                   # Standard DTW
d = dtwcpp.distance.dtw(x, y, variant="ddtw")                         # Derivative DTW
d = dtwcpp.distance.dtw(x, y, variant="wdtw", wdtw_g=0.05)            # Weighted DTW
d = dtwcpp.distance.dtw(x, y, variant="adtw", adtw_penalty=1.0)       # Amerced DTW
d = dtwcpp.distance.dtw(x, y, variant="softdtw", sdtw_gamma=1.0)      # Soft-DTW
d = dtwcpp.distance.dtw(x, y, variant="msm", msm_c=1.0)               # Move-Split-Merge
d = dtwcpp.distance.dtw(x, y, variant="twe", twe_nu=0.001, twe_lambda=1.0)  # Time Warp Edit
```

`metric` is `"l1"` or `"squared_euclidean"` for Standard DTW and DDTW; the other
variants compute an L1 cost and refuse another metric.

### DTW with missing data

Under a missing-data strategy (Standard DTW) NaN is a missing value:

```python
x_missing = [1.0, float('nan'), 3.0, 4.0, 5.0]
d = dtwcpp.distance.dtw(x_missing, y, missing_strategy="zero_cost")    # NaN pairs cost 0
d = dtwcpp.distance.dtw(x_missing, y, missing_strategy="arow")         # diagonal-only at NaN
d = dtwcpp.distance.dtw(x_missing, y, missing_strategy="interpolate")  # gaps filled linearly
```

## Clustering functions

### FastPAM

```python
import dtwcpp

prob = dtwcpp.Problem("my_clustering")
prob.set_data(series, names)
prob.band = 10
prob.set_n_clusters(3)

result = dtwcpp.fast_pam(prob, n_clusters=3, max_iter=100)
print(result.labels, result.medoid_indices, result.total_cost)
```

### FastCLARA

Scalable subsampling-based clustering:

```python
from dtwcpp import fast_clara

result = fast_clara(prob, n_clusters=3, sample_size=-1, n_samples=5, seed=42)
```

### Hierarchical clustering

Build a dendrogram, then cut it at the desired number of clusters:

```python
from dtwcpp import build_dendrogram, cut_dendrogram, HierarchicalOptions, Linkage

hier_opts = HierarchicalOptions()
hier_opts.linkage = Linkage.Average  # Single, Complete, or Average

dend = build_dendrogram(prob, hier_opts)
result = cut_dendrogram(dend, prob, n_clusters=3)
```

## Threading

A `Problem` instance must not be used concurrently from multiple Python
threads. The GIL is released during C++ work so that other threads can run, but
two threads calling methods on the same `Problem` race on its lazily-filled
distance cache. Use one `Problem` per thread, or call `fill_distance_matrix()`
first and only read afterwards.

## Quality scores

All scoring functions operate on a `Problem` object that has been clustered (distance matrix computed and labels assigned).

```python
from dtwcpp import (
    silhouette,
    davies_bouldin,
    dunn,
    inertia,
    calinski_harabasz,
    adjusted_rand,
    normalized_mutual_info,
)

sil = silhouette(prob)                   # per-point silhouette values
dbi = davies_bouldin(prob)               # lower is better
di  = dunn(prob)                         # higher is better
ine = inertia(prob)                      # total within-cluster cost
chi = calinski_harabasz(prob)            # higher is better

# External validation (requires ground-truth labels)
ari = adjusted_rand(labels_true, labels_pred)
nmi = normalized_mutual_info(labels_true, labels_pred)
```

## GPU acceleration

If DTWC++ was built with a GPU backend (CUDA on NVIDIA, Metal on macOS), GPU-accelerated
distance matrix computation is available.

### Check availability

```python
print(dtwcpp.gpu_available())  # True if this build's backend finds a GPU
print(dtwcpp.gpu_info())       # e.g. "CUDA: <device name and properties>", or why there is none
```

`dtwcpp.test.gpu()` runs a small distance matrix on the GPU and checks it against the CPU.

### Use GPU for distance matrix

Pass `device="cuda"` to `compute_distance_matrix` or `DTWClustering`:

```python
# Direct distance matrix computation on GPU
D = dtwcpp.compute_distance_matrix(series, band=10, device="cuda")

# Multi-GPU: select a specific device
D = dtwcpp.compute_distance_matrix(series, band=10, device="cuda:1")

# Clustering with GPU-accelerated distance matrix
clf = dtwcpp.DTWClustering(n_clusters=3, band=10, device="cuda")
labels = clf.fit_predict(X)
```

`cuda` is a spelling of `gpu`: it runs on this build's GPU (CUDA, else Metal),
and with no GPU the request raises `dtwcpp.DeviceError`. Select `device="cpu"`
explicitly if CPU execution is wanted.

**Note:** GPU clustering supports Standard, dependent DTW with
`missing_strategy="error"`; both L1 and squared-Euclidean metrics are available.
Other variants and missing-data recurrences require CPU computation.

A `Problem` takes the device once and does not follow the process-wide
`dtwcpp.device()`. It reads the name with the same C++ grammar as
`dtwcpp.device()` and the per-call `device=` arguments above — `cpu`, `gpu`,
`gpu:N`, with `cuda` / `cuda:N` as aliases of `gpu` / `gpu:N`, so on a Metal
build `device="cuda"` selects Metal. `hpc` is a `cluster()` option, not a
`Problem` device (`InvalidInput`):

```python
prob = dtwcpp.Problem("my_clustering", device="gpu")   # CUDA, else Metal
prob.set_data(series, names)
prob.fill_distance_matrix()        # on the GPU
prob.set_device("cpu")             # back to the CPU
```

On a GPU, a request its kernels do not implement — for example
`prob.set_variant(dtwcpp.DTWVariant.WDTW)` — raises `dtwcpp.DeviceError` naming
the setting when distances are computed; it never runs on the CPU instead.

## I/O utilities

The `dtwcpp.io` module provides functions for saving and loading time-series datasets.

### CSV (always available)

```python
from dtwcpp import save_dataset_csv, load_dataset_csv

# Save: each row is one series, columns are time steps
save_dataset_csv(data, "timeseries.csv", names=["s0", "s1", "s2"])

# Load: returns (data, names)
data, names = load_dataset_csv("timeseries.csv")
```

### HDF5 (requires h5py)

HDF5 provides gzip-compressed storage and can also store the distance matrix and metadata.

```python
from dtwcpp import save_dataset_hdf5, load_dataset_hdf5

save_dataset_hdf5(
    data, "timeseries.h5",
    names=["s0", "s1", "s2"],
    distance_matrix=D,
    metadata={"band": 10, "variant": "standard"},
)

result = load_dataset_hdf5("timeseries.h5")
# result["series"], result["names"], result["distmat"], result["metadata"]
```

Install the dependency: `uv add h5py`

### Parquet (requires pyarrow)

Parquet provides Snappy-compressed columnar storage:

```python
from dtwcpp import save_dataset_parquet, load_dataset_parquet

save_dataset_parquet(data, "timeseries.parquet")
data, names = load_dataset_parquet("timeseries.parquet")
```

Install the dependency: `uv add pyarrow`

## Checkpointing

For long-running computations, save and resume distance-matrix state:

```python
from dtwcpp import save_checkpoint, load_checkpoint, CheckpointOptions

# Save current state
save_checkpoint(prob, "./checkpoints")

# Resume later
loaded = load_checkpoint(prob, "./checkpoints")
if loaded:
    print("Resumed from checkpoint")
```

See [Checkpointing](../checkpointing/) for full details.

## Utility functions

```python
# Derivative transform (used internally by DDTW)
transformed = dtwcpp.derivative_transform(series)

# Z-normalization (zero mean, unit variance)
normalized = dtwcpp.z_normalize(series)
```

