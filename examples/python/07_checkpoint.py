"""
@file 07_checkpoint.py
@brief DTWC++ Checkpointing — Save and resume distance matrix computation.
@details
Demonstrates: save_checkpoint(), load_checkpoint() for long-running
computations that may need to survive crashes or interruptions.

Checkpoint format:
  - distances.csv: the NxN matrix (NaN for uncomputed entries)
  - metadata.txt:  key=value metadata (n, band, pairs_computed, timestamp)
@author Volkan Kumtepeli
"""

import numpy as np
import tempfile
import os
import dtwcpp

# --- 1. Create a problem and compute distances ---
rng = np.random.default_rng(42)
n_series = 15
length = 50
series = [list(rng.uniform(-1, 1, length)) for _ in range(n_series)]
names = [f"s{i}" for i in range(n_series)]

prob = dtwcpp.Problem("checkpoint_demo")
prob.set_data(series, names)
prob.band = -1
prob.fill_distance_matrix()

print(f"Computed distance matrix: {prob.size} x {prob.size}")
print(f"Max distance: {prob.max_distance():.4f}")
print()

# --- 2. Save checkpoint ---
ckpt_dir = os.path.join(tempfile.mkdtemp(), "dtwcpp_checkpoint")
dtwcpp.save_checkpoint(prob, ckpt_dir)
print(f"Checkpoint saved to: {ckpt_dir}")

# List saved files
for fname in os.listdir(ckpt_dir):
    fpath = os.path.join(ckpt_dir, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname}: {size_kb:.1f} KB")
print()

# --- 3. Resume from checkpoint in a new Problem ---
prob2 = dtwcpp.Problem("resumed")
prob2.set_data(series, names)  # must set data first (to know matrix size)
prob2.band = -1

loaded = dtwcpp.load_checkpoint(prob2, ckpt_dir)
if loaded:
    print("Checkpoint loaded successfully!")
    print(f"Distance matrix filled: {prob2.is_distance_matrix_filled()}")

    # Verify distances match
    match = True
    for i in range(n_series):
        for j in range(i + 1, n_series):
            d1 = prob.dist_by_ind(i, j)
            d2 = prob2.dist_by_ind(i, j)
            if abs(d1 - d2) > 1e-12:
                match = False
                break
    print(f"All distances match original: {match}")
else:
    print("Checkpoint not found or incompatible.")

print()

# --- 4. Use resumed matrix for clustering (no recomputation) ---
result = dtwcpp.fast_pam(prob2, n_clusters=3)
print(f"Clustering from checkpoint: {result}")
print(f"Cluster sizes: {np.bincount(result.labels)}")
