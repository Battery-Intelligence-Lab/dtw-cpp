---
description: "Plot clustering results — clusters, silhouette, distance matrix heatmap, warping path."
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
---

# DTWC++ Visualize

Generate matplotlib plots. `$ARGUMENTS` specifies what to plot.

## Plot types

| Type | Description |
|------|-------------|
| `clusters` | All series, colored by cluster, medoid bold |
| `silhouette` | Per-point silhouette, sorted within cluster |
| `distance-matrix` | Heatmap reordered by cluster |
| `medoids` | Medoid series overlaid |
| `warping-path` | Two series with alignment path |
| `elbow` | Cost vs k for k=2..10 |

## Step 1: Check dependencies

```bash
python3 -c "import matplotlib, numpy" 2>/dev/null && echo "matplotlib: ok" || echo "install matplotlib: uv pip install matplotlib"
```

## Step 2: Generate plot

### Clusters plot

```python
import numpy as np, matplotlib.pyplot as plt, dtwcpp as dc

data = dc.load_dataset_csv("DATA_PATH")
labels = np.loadtxt("LABELS_PATH", dtype=int, skiprows=1)
medoids = np.loadtxt("MEDOIDS_PATH", dtype=int, skiprows=1) if MEDOIDS_PATH else None

k = int(labels.max() + 1)
fig, axes = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)
if k == 1: axes = [axes]
cmap = plt.cm.tab10

for c in range(k):
    ax = axes[c]
    members = np.where(labels == c)[0]
    for i in members:
        s = np.asarray(data[i])
        ax.plot(s, color=cmap(c), alpha=0.3, linewidth=0.8)
    if medoids is not None:
        m = np.asarray(data[medoids[c]])
        ax.plot(m, color="black", linewidth=2.5, label=f"medoid (series {medoids[c]})")
    ax.set_title(f"Cluster {c} ({len(members)} series)")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("clusters.png", dpi=150)
print("Saved: clusters.png")
```

### Silhouette plot

```python
sil = np.loadtxt("SILHOUETTE_PATH", skiprows=1)
labels = np.loadtxt("LABELS_PATH", dtype=int, skiprows=1)
k = int(labels.max() + 1)

fig, ax = plt.subplots(figsize=(8, 6))
y0 = 10
for c in range(k):
    mask = labels == c
    s = np.sort(sil[mask])
    y1 = y0 + len(s)
    ax.fill_betweenx(np.arange(y0, y1), 0, s,
                      facecolor=plt.cm.tab10(c), edgecolor=plt.cm.tab10(c), alpha=0.7)
    ax.text(-0.05, (y0+y1)/2, f"C{c}", ha="right", va="center")
    y0 = y1 + 10

ax.axvline(sil.mean(), color="red", linestyle="--", label=f"mean={sil.mean():.3f}")
ax.set_xlabel("Silhouette coefficient")
ax.set_yticks([])
ax.legend()
plt.tight_layout()
plt.savefig("silhouette.png", dpi=150)
print("Saved: silhouette.png")
```

### Distance matrix heatmap

```python
import seaborn as sns  # or use plt.imshow if seaborn unavailable
dm = np.loadtxt("DISTANCES_PATH", delimiter=",")
labels = np.loadtxt("LABELS_PATH", dtype=int, skiprows=1)

# Reorder rows/cols by cluster
order = np.argsort(labels)
dm_ord = dm[np.ix_(order, order)]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(dm_ord, cmap="viridis", aspect="auto")
plt.colorbar(im, ax=ax, label="DTW distance")
# Cluster boundaries
bounds = np.cumsum(np.bincount(labels))[:-1]
for b in bounds:
    ax.axhline(b, color="white", linewidth=0.5)
    ax.axvline(b, color="white", linewidth=0.5)
ax.set_title("DTW distance matrix (reordered by cluster)")
plt.tight_layout()
plt.savefig("distance_matrix.png", dpi=150)
print("Saved: distance_matrix.png")
```

### Warping path (two series)

```python
x = np.asarray(data[I])
y = np.asarray(data[J])
# Compute path
path = dc.dtw_path(x, y)  # returns list of (i, j) tuples
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, label=f"series {I}", linewidth=2)
ax.plot(y, label=f"series {J}", linewidth=2)
# Draw alignment lines
for (i, j) in path[::max(1, len(path)//50)]:  # subsample
    ax.plot([i, j], [x[i], y[j]], color="gray", alpha=0.3, linewidth=0.5)
ax.legend()
ax.set_title(f"DTW alignment: distance = {dc.dtw_distance(x, y):.3f}")
plt.tight_layout()
plt.savefig("warping_path.png", dpi=150)
```

### Elbow plot

```python
costs = []
for k in range(2, 11):
    prob = dc.Problem(data)
    prob.set_method(dc.Method.Kmedoids)
    prob.set_number_of_clusters(k)
    prob.cluster()
    costs.append((k, prob.get_cost()))
ks, cs = zip(*costs)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ks, cs, "o-", markersize=8)
ax.set_xlabel("k"); ax.set_ylabel("Total cost")
ax.set_title("Elbow method — pick k at inflection")
plt.tight_layout()
plt.savefig("elbow.png", dpi=150)
```

## Step 3: Report

Print the output file path and a one-line summary of what's in the plot.

## Fallback

If matplotlib is unavailable, print an ASCII summary:
```
Cluster 0: ████████ (24 series)
Cluster 1: ██████   (18 series)
Cluster 2: ████████████ (36 series)
```

## Related

- `/cluster` — generate labels and medoids
- `/evaluate` — compute silhouette scores for plotting
