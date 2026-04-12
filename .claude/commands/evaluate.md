---
description: "Evaluate clustering quality — Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI."
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# DTWC++ Evaluate

Evaluate an existing clustering. `$ARGUMENTS` contains: data path, labels path, optionally ground-truth labels.

## Step 0: Parse inputs

Required:
- **Data path** — original series (CSV / Parquet / HDF5)
- **Labels path** — predicted cluster labels (one int per series)

Optional:
- **Ground-truth labels** — for ARI / NMI
- **Medoids path** — if not provided, will re-derive via intra-cluster min-cost

## Step 1: Load and reconstruct

```python
import numpy as np
import dtwcpp as dc
from pathlib import Path

data = dc.load_dataset_csv("DATA_PATH")
labels = np.loadtxt("LABELS_PATH", dtype=int, skiprows=1)
assert len(labels) == data.size, f"Label count {len(labels)} != series count {data.size}"

k = int(labels.max() + 1)
print(f"N = {data.size}, k = {k}")

# Rebuild Problem with labels
prob = dc.Problem(data)
prob.set_number_of_clusters(k)
# Derive medoids: for each cluster, pick the series with lowest total intra-cluster distance
# (or skip if user provided medoids)
dm = dc.compute_distance_matrix(data, band=BAND)
arr = np.asarray(dm)
medoids = []
for c in range(k):
    members = np.where(labels == c)[0]
    sub = arr[np.ix_(members, members)]
    medoids.append(int(members[sub.sum(axis=1).argmin()]))
prob.set_clusters_and_medoids(list(labels), medoids)
```

## Step 2: Internal metrics

```python
sil = dc.silhouette(prob)          # per-point
dbi = dc.davies_bouldin_index(prob)
ch  = dc.calinski_harabasz_index(prob)
dun = dc.dunn_index(prob)
inr = dc.inertia(prob)

print(f"\nInternal metrics:")
print(f"  Silhouette (mean): {sil.mean():>8.4f}  (> 0.5 good, < 0.25 weak)")
print(f"  Silhouette (min):  {sil.min():>8.4f}")
print(f"  Davies-Bouldin:    {dbi:>8.4f}  (lower better)")
print(f"  Calinski-Harabasz: {ch:>8.4f}  (higher better)")
print(f"  Dunn index:        {dun:>8.4f}  (higher better)")
print(f"  Inertia:           {inr:>8.4f}  (within-cluster dispersion)")
```

## Step 3: External metrics (if ground truth provided)

```python
if gt_path:
    gt = np.loadtxt(gt_path, dtype=int, skiprows=1)
    ari = dc.adjusted_rand_index(list(gt), list(labels))
    nmi = dc.normalized_mutual_information(list(gt), list(labels))
    print(f"\nExternal metrics (vs ground truth):")
    print(f"  Adjusted Rand Index: {ari:.4f}  (1.0 perfect, 0 random)")
    print(f"  Normalized MI:       {nmi:.4f}  (1.0 perfect)")
```

## Step 4: Cluster balance analysis

```python
unique, counts = np.unique(labels, return_counts=True)
balance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
print(f"\nCluster balance:")
for c, n in zip(unique, counts):
    print(f"  Cluster {c}: {n} series ({100*n/len(labels):.1f}%)")
print(f"  Max/min ratio: {balance_ratio:.2f}  (<3 balanced, >10 skewed)")
```

## Step 5: Diagnosis and recommendations

Apply these heuristics:

| Observation | Recommendation |
|-------------|----------------|
| Mean silhouette < 0.25 | Try different k, variant, or check data quality |
| Davies-Bouldin > 2.0 | Clusters overlap; try smaller k or different variant |
| ARI < 0.3 (with ground truth) | Wrong variant or method; try DDTW if shapes matter |
| Balance ratio > 10 | Very skewed — may indicate outliers or wrong k |
| Silhouette.min() very negative | Check if outliers are dominating |

## Step 6: Save report

```python
report = {
    "n": int(data.size), "k": int(k),
    "mean_silhouette": float(sil.mean()),
    "davies_bouldin": float(dbi),
    "calinski_harabasz": float(ch),
    "dunn_index": float(dun),
    "inertia": float(inr),
    "cluster_sizes": counts.tolist(),
}
if gt_path:
    report["ari"] = float(ari); report["nmi"] = float(nmi)

import json
with open("evaluation.json", "w") as f:
    json.dump(report, f, indent=2)
```

## Related

- `/cluster` — to generate labels
- `/visualize silhouette` — plot silhouette per-point
- `/help` — metric reference
