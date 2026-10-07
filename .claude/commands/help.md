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
| N ≤ 5000 | **FasterPAM** | Local-optimum k-medoids via eager swap; fast for small N |
| 5000 < N ≤ 50000 | **FastCLARA** | Samples subsets; scales linearly; near-optimal |
| Need dendrogram | **Hierarchical** | Agglomerative; produces full tree |
| Need provable optimum | **MIP** (Gurobi/HiGHS) | Integer programming; expensive but exact |
| N > 50000 | **OneBatchPAM** | One fixed N×m distance batch; O(Nm) distances |
| Series exceed RAM | **FastCLARA, streamed** | `--ram-limit` on one list-per-row Parquet file |

Python: `fast_pam()`, `fast_clara()`, `one_batch_pam()`, `build_dendrogram()` + `cut_dendrogram()`.
CLI: `--method pam|clara|onebatch|hierarchical|mip|lrcore`.

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
- **`--dtype`**: `float32` stores the series and runs DTW in Float32: half the series memory (the distance matrix stays float64); distances can differ slightly from `float64`.

## Data formats

| Format | When | Extension |
|--------|------|-----------|
| CSV | Small, human-readable | `.csv` |
| Parquet | Compressed, recommended for N > 10k | `.parquet` |
| Arrow IPC | Fastest load (memory-mapped) | `.arrow`, `.ipc` |
| HDF5 | With metadata | `.h5`, `.hdf5` |
| `.dtwm` | Distance matrix of `--checkpoint`, not time series | `.dtwm` |

Python I/O: `dtwcpp.load_dataset_csv`, `load_dataset_parquet`, `load_dataset_hdf5`.
Convert to Arrow IPC: `dtwc-convert input.csv -o output.arrow`.

## Evaluation metrics

**Internal** (no ground truth):
- `silhouette()` — mean ∈ [-1, 1]; > 0.5 good
- `davies_bouldin()` — lower better
- `calinski_harabasz()` — higher better
- `dunn()` — higher better
- `inertia()` — within-cluster dispersion

**External** (require ground truth):
- `adjusted_rand()` — 1.0 perfect, 0 random
- `normalized_mutual_info()` — 1.0 perfect

## Python API quick reference

```python
import dtwcpp as dc

# Load data
data = dc.load("data.csv").as_data()

# Simple sklearn-style
clustering = dc.DTWClustering(n_clusters=3, method="pam")
clustering.fit(data.p_vec)
labels = clustering.labels_

# Advanced Problem API
prob = dc.Problem("data")
prob.set_data(data)
prob.set_method(dc.Method.Kmedoids)
prob.set_n_clusters(3)
prob.cluster()

# Raw DTW distance
d = dc.distance.dtw(x, y, band=10)
```

## CLI quick reference

```bash
dtwc_cl --input data.parquet --method clara -k 5 \
        --variant wdtw --wdtw-g 0.05 --band 10 \
        --output results/
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
