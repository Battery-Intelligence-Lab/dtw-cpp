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

# Variant dispatch convenience
d_soft = dtwcpp.distance.dtw(x, y, variant="soft_dtw", gamma=1.0)
print(f"Soft-DTW distance: {d_soft}")
```

Both plain Python lists and NumPy arrays are accepted. Lists are automatically converted to `float64` arrays.

Pairwise distances live under `dtwcpp.distance.*`. The old root-level
distance helpers such as `dtwcpp.dtw_distance(...)` were removed in this
breaking release.

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
| `use_pruning` | bool | `True` | Enable LB_Keogh pruning (CPU only) |
| `device` | str | `"cpu"` | `"cpu"`, `"cuda"`, or `"cuda:N"` |

## DTWClustering class

`DTWClustering` provides an sklearn-compatible interface for k-medoids clustering with DTW distance. It implements FastPAM (Schubert & Rousseeuw, 2021).

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
| `variant` | str | `"standard"` | DTW variant: `"standard"`, `"ddtw"`, `"wdtw"`, `"adtw"` |
| `band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |
| `max_iter` | int | `100` | Maximum FastPAM iterations |
| `n_init` | int | `1` | Number of random restarts (best result kept) |
| `wdtw_g` | float | `0.05` | WDTW logistic weight steepness (only for `variant="wdtw"`) |
| `adtw_penalty` | float | `1.0` | ADTW non-diagonal step penalty (only for `variant="adtw"`) |
| `missing_strategy` | str | `"error"` | NaN handling: `"error"`, `"zero_cost"`, `"arow"`, `"interpolate"` |
| `device` | str | `"cpu"` | Distance matrix device: `"cpu"`, `"cuda"`, `"cuda:N"` |

### Methods

- **`fit(X)`** -- Fit clustering on `X` (2D array or list of 1D arrays). Returns `self`.
- **`predict(X)`** -- Assign each series in `X` to the nearest medoid.
- **`fit_predict(X)`** -- Fit and return cluster labels.
- **`score(X)`** -- Return negative inertia (for sklearn grid search compatibility).

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

All DTW functions accept lists or NumPy arrays.

### Standard DTW

```python
d = dtwcpp.distance.dtw(x, y, band=-1, metric="l1")
```

### Derivative DTW (DDTW)

Applies a derivative transform before computing standard DTW:

```python
d = dtwcpp.distance.ddtw(list(x), list(y), band=-1)
```

### Weighted DTW (WDTW)

Applies logistic weights based on the warping step index:

```python
d = dtwcpp.distance.wdtw(list(x), list(y), band=-1, g=0.05)
```

### Amerced DTW (ADTW)

Adds a penalty for non-diagonal warping steps:

```python
d = dtwcpp.distance.adtw(list(x), list(y), band=-1, penalty=1.0)
```

### Soft-DTW

A differentiable relaxation of DTW using softmin:

```python
d = dtwcpp.distance.soft_dtw(list(x), list(y), gamma=1.0)
```

### DTW with missing data

NaN values contribute zero cost:

```python
x_missing = [1.0, float('nan'), 3.0, 4.0, 5.0]
d = dtwcpp.distance.missing(x_missing, y, band=-1, metric="l1")
```

### DTW-AROW (diagonal-only alignment for missing values)

When a value is NaN, the warping path is restricted to the diagonal direction only:

```python
d = dtwcpp.distance.arow(x_missing, y, band=-1, metric="l1")
```

## Clustering functions

### FastPAM

```python
import dtwcpp

prob = dtwcpp.Problem("my_clustering")
prob.set_data(series, names)
prob.band = 10
prob.set_number_of_clusters(3)

result = dtwcpp.fast_pam(prob, n_clusters=3, max_iter=100)
print(result.labels, result.medoid_indices, result.total_cost)
```

### FastCLARA

Scalable subsampling-based clustering:

```python
from dtwcpp import fast_clara

result = fast_clara(prob, n_clusters=3, sample_size=-1, n_samples=5, seed=42)
```

### CLARANS

```python
from dtwcpp import clarans, CLARANSOptions

opts = CLARANSOptions()
result = clarans(prob, opts)
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

## Quality scores

All scoring functions operate on a `Problem` object that has been clustered (distance matrix computed and labels assigned).

```python
from dtwcpp import (
    silhouette,
    davies_bouldin_index,
    dunn_index,
    inertia,
    calinski_harabasz_index,
    adjusted_rand_index,
    normalized_mutual_information,
)

sil = silhouette(prob)                         # per-point silhouette values
dbi = davies_bouldin_index(prob)               # lower is better
di  = dunn_index(prob)                         # higher is better
ine = inertia(prob)                            # total within-cluster cost
chi = calinski_harabasz_index(prob)            # higher is better

# External validation (requires ground-truth labels)
ari = adjusted_rand_index(labels_true, labels_pred)
nmi = normalized_mutual_information(labels_true, labels_pred)
```

## GPU acceleration

If DTWC++ was built with CUDA support, GPU-accelerated distance matrix computation is available.

### Check availability

```python
print(dtwcpp.CUDA_AVAILABLE)      # True if compiled with CUDA
print(dtwcpp.cuda_available())    # True if a CUDA GPU is detected
print(dtwcpp.cuda_device_info())  # Device name and properties
```

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

If CUDA is not available, a warning is issued and computation falls back to CPU automatically.

**Note:** GPU mode currently only supports `variant="standard"`. Other DTW variants require CPU computation.

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

For long-running computations, save and resume distance matrix state:

```python
from dtwcpp import save_checkpoint, load_checkpoint, CheckpointOptions

# Save current state
save_checkpoint(prob, "./checkpoints")

# Resume later
loaded = load_checkpoint(prob, "./checkpoints")
if loaded:
    print("Resumed from checkpoint")
```

See [Checkpointing](checkpointing/) for full details.

## Utility functions

```python
# Derivative transform (used internally by DDTW)
transformed = dtwcpp.derivative_transform(series)

# Z-normalization (zero mean, unit variance)
normalized = dtwcpp.z_normalize(series)
```

