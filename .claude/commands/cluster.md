---
description: "Cluster time series using DTW. Handles data loading, method selection, clustering, evaluation, and output."
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
---

# DTWC++ Cluster

Run the full clustering pipeline for the user. `$ARGUMENTS` contains the request (e.g., `data.csv -k 3 --variant wdtw`).

## Step 0: Parse intent

Extract from `$ARGUMENTS`:
- **Input path** (required). If missing, ask the user.
- **k** (number of clusters). If missing, suggest silhouette-based search for k=2..10.
- **Variant**: standard / ddtw / wdtw / adtw / softdtw (default: standard)
- **Method**: auto (default, pick by N), pam, clara, hierarchical, mip
- **Band**: default `max(5, series_length/10)`
- **Output dir**: default `./clustering_output/`
- **Special**: missing data? GPU? very large (> 50k series)?

## Step 1: Detect environment

```bash
python3 -c "import dtwcpp; print('python_ok')" 2>/dev/null && echo "Python: available"
which dtwc_cl && echo "CLI: available"
```

Prefer Python (better error messages). Fall back to CLI for very large batch jobs.

## Step 2: Inspect data

Generate a short Python snippet to characterize the input:
```python
import numpy as np, dtwcpp as dc
data = dc.load_dataset_csv("INPUT")  # or load_dataset_parquet, load_dataset_hdf5
print(f"Series: {data.size}, length: {data.max_length}, ndim: {data.ndim}")
# Check for NaN
has_nan = any(np.any(np.isnan(data[i])) for i in range(min(data.size, 100)))
print(f"NaN detected: {has_nan}")
```

If NaN detected, default `missing_strategy="arow"`.

## Step 3: Pick method automatically

| N | Method | Rationale |
|---|--------|-----------|
| ≤ 5000 | `fast_pam` | Exact k-medoids, fastest for small N |
| 5000–50000 | `fast_clara` | Subsample-based, scales linearly |
| > 50000 | `fast_clara` + `--ram-limit` | Chunked to fit memory |
| MIP requested | `mip` (Gurobi preferred, HiGHS fallback) | Provable optimum |

Announce the chosen method and reasoning in one sentence.

## Step 4: Generate and run clustering script

Write to a temporary file (`cluster_run.py`) and execute. Example template:

```python
#!/usr/bin/env python3
import numpy as np
import dtwcpp as dc
from pathlib import Path
import json

# Load
data = dc.load_dataset_csv("INPUT_PATH")
print(f"Loaded {data.size} series, max length {data.max_length}")

# Problem setup
prob = dc.Problem(data)
prob.set_distance_type(dc.DistanceType.DTW)
prob.set_variant(dc.DTWVariant.STANDARD)  # or user choice
prob.set_band(BAND)
prob.set_method(dc.Method.Kmedoids)
prob.set_number_of_clusters(K)
# Missing data?
# prob.set_missing_strategy(dc.MissingStrategy.AROW)

# Run
import time
t0 = time.time()
prob.cluster()
elapsed = time.time() - t0

# Collect
labels = np.array(prob.clusters_ind)
medoids = np.array(prob.centroids_ind)
cost = prob.get_cost()

# Internal metrics
sil = dc.silhouette(prob)
dbi = dc.davies_bouldin_index(prob)
ch = dc.calinski_harabasz_index(prob)

print(f"\nClustering complete in {elapsed:.2f}s")
print(f"Cost: {cost:.4f}")
print(f"Mean silhouette: {sil.mean():.3f}")
print(f"Davies-Bouldin:  {dbi:.3f} (lower better)")
print(f"Calinski-Harabasz: {ch:.3f} (higher better)")

# Cluster sizes
unique, counts = np.unique(labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f"  Cluster {c}: {n} series (medoid = series {medoids[c]})")

# Save
outdir = Path("OUTPUT_DIR")
outdir.mkdir(exist_ok=True)
np.savetxt(outdir / "labels.csv", labels, fmt="%d", header="cluster_label", comments="")
np.savetxt(outdir / "medoids.csv", medoids, fmt="%d", header="medoid_index", comments="")
np.savetxt(outdir / "silhouette.csv", sil, fmt="%.6f", header="silhouette", comments="")
with open(outdir / "summary.json", "w") as f:
    json.dump({
        "n": int(data.size), "k": int(K), "cost": float(cost),
        "mean_silhouette": float(sil.mean()), "davies_bouldin": float(dbi),
        "calinski_harabasz": float(ch), "elapsed_sec": elapsed,
    }, f, indent=2)
print(f"\nOutputs in {outdir}/")
```

Run: `python3 cluster_run.py 2>&1`.

## Step 5: Report results

Print a summary table:

```
╔══════════════ Clustering Complete ══════════════╗
║  N: XXX   k: X   Method: XXX   Variant: XXX
║  Time:    X.XX s
║  Cost:    X.XXXX
║  Silhouette (mean): X.XXX
║  Davies-Bouldin:    X.XXX (lower better)
║  Cluster sizes:     [X, X, X, ...]
║  Outputs:           OUTPUT_DIR/
╚════════════════════════════════════════════════╝
```

Suggest next actions:
- "Run `/evaluate` to compute more metrics or compare against ground truth"
- "Run `/visualize` to plot clusters and silhouette"

## Fallback to CLI

If Python fails or user requests:
```bash
dtwc_cl \
  --input INPUT_PATH \
  --method auto \
  --k K \
  --variant VARIANT \
  --band BAND \
  --output-dir OUTPUT_DIR \
  --verbose
```

Parse stdout for cost and timing; report the same summary.

## Error handling

- **OOM / bad_alloc**: retry with `--method clara --ram-limit 8G`
- **NaN in distances**: retry with `--missing-strategy arow`
- **CUDA fails**: retry on CPU
- **No MIP solver**: fall back to `--method pam`

## Related

- `/distance` — just compute distances
- `/evaluate` — score an existing clustering
- `/visualize` — plot results
- `/help` — algorithm reference
