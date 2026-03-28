"""
DTWC++ Quick Start — DTW distance and clustering in 10 lines.

Install: pip install dtwcpp
"""

import numpy as np
import dtwcpp

# --- 1. Compute DTW distance between two time series ---
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 3.0, 1.0]

d = dtwcpp.dtw_distance(x, y)
print(f"DTW distance: {d}")

# Banded DTW (Sakoe-Chiba constraint) — much faster for long series
d_banded = dtwcpp.dtw_distance(x, y, band=2)
print(f"DTW distance (band=2): {d_banded}")

# --- 2. Cluster time series with sklearn-like API ---
rng = np.random.RandomState(42)

# Create 3 groups of 10 series each (length 50)
group_a = rng.randn(10, 50)              # centered at 0
group_b = rng.randn(10, 50) + 5          # centered at 5
group_c = rng.randn(10, 50) + 10         # centered at 10
X = np.vstack([group_a, group_b, group_c])

clf = dtwcpp.DTWClustering(n_clusters=3, band=10)
labels = clf.fit_predict(X)

print(f"\nCluster labels: {labels}")
print(f"Inertia (total cost): {clf.inertia_:.2f}")
print(f"Medoid indices: {clf.medoid_indices_}")
print(f"Iterations: {clf.n_iter_}")

# Predict cluster for new data
new_series = rng.randn(3, 50) + 5  # should be assigned to group_b's cluster
predicted = clf.predict(new_series)
print(f"Predicted labels for new data: {predicted}")
