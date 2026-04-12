---
description: "Get help with DTWC++ — algorithm selection, DTW variants, parameter tuning, data formats. Read-only reference."
allowed-tools:
  - Read
  - Glob
  - Grep
---

# DTWC++ Help

You are answering a user's question about the DTWC++ library (Dynamic Time Warping Clustering). `$ARGUMENTS` contains the user's question.

## Step 0: Route the question

Match the question to one of these sections. If no match, read source files (`python/dtwcpp/__init__.py`, `dtwc/dtwc_cl.cpp`) to synthesize an answer.

## Algorithm selection

| When | Use | Why |
|------|-----|-----|
| N ≤ 5000 | **FastPAM1** | Optimal k-medoids via swap; fast and exact for small N |
| 5000 < N ≤ 50000 | **FastCLARA** | Samples subsets; scales linearly; near-optimal |
| Need dendrogram | **Hierarchical** | Agglomerative; produces full tree |
| Need provable optimum | **MIP** (Gurobi/HiGHS) | Integer programming; expensive but exact |
| N > 50000 | **CLARANS** | Randomized k-medoids; best scaling |
| Very large, fits memory | **FastCLARA with chunking** | Use `--ram-limit` |

Python: `fast_pam()`, `fast_clara()`, `build_dendrogram()` + `cut_dendrogram()`, `clarans()`.
CLI: `--method pam|clara|hierarchical|mip|clarans`.

## DTW variants

| Variant | When | Key params |
|---------|------|-----------|
| **Standard** | General-purpose | `band` (Sakoe-Chiba) |
| **DDTW** | Shape matching, ignore amplitude | — |
| **WDTW** | Penalize time offsets | `wdtw_g` ∈ [0.01, 0.5] |
| **ADTW** | Penalize non-diagonal warping | `adtw_penalty` ∈ [0.1, 10] |
| **Soft-DTW** | Differentiable (gradient-based) | `sdtw_gamma` > 0 |
| **AROW** | Missing (NaN) data, diagonal-only | — |
| **Missing (zero-cost)** | Missing data, lenient | — |

CLI: `--variant standard|ddtw|wdtw|adtw|softdtw`, `--wdtw-g 0.05`, `--adtw-penalty 1.0`, `--sdtw-gamma 1.0`.

## Parameter tuning

- **`band`**: Start at `series_length / 10`. Narrower → faster but more constrained. Tune via silhouette.
- **`k` (clusters)**: No ground truth? Try k=2..10, pick highest mean silhouette.
- **`--dtype`**: `float32` is 2× faster, uses 2× less memory, max DTW error ≈ 0.003% (acceptable for clustering).

## Data formats

| Format | When | Extension |
|--------|------|-----------|
| CSV | Small, human-readable | `.csv` |
| Parquet | Compressed, recommended for N > 10k | `.parquet` |
| Arrow IPC | Fastest load (zero-copy mmap) | `.arrow`, `.ipc` |
| HDF5 | With metadata | `.h5`, `.hdf5` |
| `.dtws` | Internal distance matrix cache | `.dtws` |

Python I/O: `dtwcpp.load_dataset_csv`, `load_dataset_parquet`, `load_dataset_hdf5`.
Convert: `dtwc-convert input.csv output.parquet`.

## Evaluation metrics

**Internal** (no ground truth):
- `silhouette()` — mean ∈ [-1, 1]; > 0.5 good
- `davies_bouldin_index()` — lower better
- `calinski_harabasz_index()` — higher better
- `dunn_index()` — higher better
- `inertia()` — within-cluster dispersion

**External** (require ground truth):
- `adjusted_rand_index()` — 1.0 perfect, 0 random
- `normalized_mutual_information()` — 1.0 perfect

## Python API quick reference

```python
import dtwcpp as dc

# Load data
data = dc.load_dataset_csv("data.csv")

# Simple sklearn-style
clustering = dc.DTWClustering(n_clusters=3, method="pam")
clustering.fit(data)
labels = clustering.labels_

# Advanced Problem API
prob = dc.Problem(data)
prob.set_method(dc.Method.Kmedoids)
prob.set_number_of_clusters(3)
prob.cluster()

# Raw DTW distance
d = dc.dtw_distance(x, y, band=10)
```

## CLI quick reference

```bash
dtwc_cl --input data.parquet --method clara --k 5 \
        --variant wdtw --wdtw-g 0.05 --band 10 \
        --output-dir results/
```

Run `dtwc_cl --help` for full flag reference.

## Fallback: When question doesn't match

1. Use `Grep` to search for keywords in `docs/content/` and `python/dtwcpp/__init__.py`.
2. Read the specific source if the question references a function/flag.
3. Include a runnable Python snippet or CLI command in your answer.

## Related commands

- `/cluster` — run clustering
- `/distance` — compute distances
- `/evaluate` — score clustering quality
- `/troubleshoot` — diagnose problems
