"""
@file 03_clustering_evaluation.py
@brief DTWC++ Clustering Evaluation — Compare clustering quality across DTW variants.
@details Demonstrates: DTWClustering with different variants, silhouette scores, DBI.
@author Volkan Kumtepeli
"""

import numpy as np
import dtwcpp

# Generate synthetic data: 3 well-separated clusters
rng = np.random.RandomState(42)
n_per_cluster = 10
length = 30

clusters = []
for center in [0, 50, 100]:
    for _ in range(n_per_cluster):
        series = center + rng.randn(length) * 2
        clusters.append(series)

X = np.array(clusters)
true_labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)

print("=== Clustering Evaluation ===")
print(f"Data: {X.shape[0]} series of length {X.shape[1]}")
print(f"True clusters: {np.bincount(true_labels)}")
print()

# Compare DTW variants for clustering
variants = ["standard", "ddtw", "wdtw", "adtw"]

for variant in variants:
    kwargs = {}
    if variant == "wdtw":
        kwargs["wdtw_g"] = 0.1
    elif variant == "adtw":
        kwargs["adtw_penalty"] = 1.0

    clf = dtwcpp.DTWClustering(
        n_clusters=3, variant=variant, band=-1, max_iter=100, **kwargs
    )
    labels = clf.fit_predict(X)

    # Check cluster recovery accuracy
    # (labels may be permuted — check if partition matches)
    from itertools import permutations
    best_acc = 0
    for perm in permutations(range(3)):
        remapped = np.array([perm[l] for l in labels])
        acc = np.mean(remapped == true_labels)
        best_acc = max(best_acc, acc)

    print(f"{variant:10s} — cost: {clf.inertia_:8.2f}, "
          f"iters: {clf.n_iter_:2d}, "
          f"accuracy: {best_acc:.0%}")

# --- Detailed evaluation with Problem class ---
print("\n=== Detailed Scoring (Standard DTW) ===")

prob = dtwcpp.Problem("eval")
prob.set_data(X.tolist(), [f"s{i}" for i in range(len(X))])
prob.band = -1
prob.fill_distance_matrix()

result = dtwcpp.fast_pam(prob, n_clusters=3)
print(f"FastPAM result: {result}")

# Apply labels to Problem for scoring
prob.set_number_of_clusters(3)
prob.clusters_ind = result.labels
prob.centroids_ind = result.medoid_indices

sil = dtwcpp.silhouette(prob)
sil_mean = np.mean(sil)
print(f"Mean silhouette: {sil_mean:.3f}")

dbi = dtwcpp.davies_bouldin_index(prob)
print(f"Davies-Bouldin Index: {dbi:.3f}")

# --- Distance matrix visualization ---
print(f"\nDistance matrix size: {prob.size} × {prob.size}")
print(f"Max distance: {prob.max_distance():.2f}")
print(f"Medoid indices: {result.medoid_indices}")
print(f"Medoid names: {[prob.get_name(i) for i in result.medoid_indices] if hasattr(prob, 'get_name') else 'N/A'}")
