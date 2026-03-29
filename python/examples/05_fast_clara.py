"""
DTWC++ FastCLARA — Scalable k-medoids for large datasets.

Demonstrates: fast_clara() for datasets too large for full PAM,
CLARAOptions, comparison with FastPAM.

FastCLARA runs FastPAM on random subsamples and picks the best
medoids, avoiding the O(N^2) distance matrix of full PAM.

References:
  Kaufman & Rousseeuw (1990). "Finding Groups in Data."
  Schubert & Rousseeuw (2021). JMLR 22(1), 4653-4688.
"""

import numpy as np
import dtwcpp

# --- 1. Generate 200 series in 5 well-separated clusters ---
rng = np.random.default_rng(42)
n_per_cluster = 40
length = 50
n_clusters = 5

groups = []
for k in range(n_clusters):
    center = k * 20  # well-separated
    for _ in range(n_per_cluster):
        groups.append(center + rng.standard_normal(length) * 2)

X = np.array(groups)
true_labels = np.repeat(np.arange(n_clusters), n_per_cluster)

print(f"Dataset: {X.shape[0]} series of length {X.shape[1]}")
print(f"True cluster sizes: {np.bincount(true_labels)}")
print()

# --- 2. FastCLARA via Problem + fast_clara ---
series = [list(row) for row in X]
names = [str(i) for i in range(len(series))]

prob = dtwcpp.Problem("clara_example")
prob.set_data(series, names)
prob.band = -1  # full DTW

result = dtwcpp.fast_clara(
    prob,
    n_clusters=n_clusters,
    sample_size=80,    # subsample size (default auto: 40 + 2*k)
    n_samples=5,       # number of subsamples
    max_iter=100,
    seed=42,
)

print(f"FastCLARA result: {result}")
print(f"Cluster sizes: {np.bincount(result.labels)}")
print(f"Medoid indices: {result.medoid_indices}")
print()

# --- 3. Check clustering accuracy ---
from itertools import permutations

best_acc = 0
for perm in permutations(range(n_clusters)):
    remapped = np.array([perm[l] for l in result.labels])
    acc = np.mean(remapped == true_labels)
    best_acc = max(best_acc, acc)

print(f"Cluster recovery accuracy: {best_acc:.0%}")
print()

# --- 4. sklearn-like API with DTWClustering ---
clf = dtwcpp.DTWClustering(n_clusters=n_clusters, band=-1)
labels = clf.fit_predict(X)
print(f"DTWClustering (FastPAM) cluster sizes: {np.bincount(labels)}")
print(f"DTWClustering inertia: {clf.inertia_:.2f}")
