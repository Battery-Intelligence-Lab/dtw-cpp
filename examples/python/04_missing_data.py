"""
@file 04_missing_data.py
@brief DTWC++ Missing Data â€” DTW with NaN-aware distance.
@details
Demonstrates: dtwcpp.distance.missing(), comparison with standard DTW.

When real-world sensors fail or data is incomplete, some time points
are missing. dtwcpp.distance.missing() treats NaN values as missing
and contributes zero cost for those pairs, rather than propagating NaN
or requiring imputation.

Reference: Yurtman et al. (2023), ECML-PKDD.
@author Volkan Kumtepeli
"""

import numpy as np
import dtwcpp

# --- 1. Simple example with one missing value ---
x = [1.0, 2.0, float("nan"), 4.0, 5.0]
y = [1.5, 2.5, 3.5, 4.5, 5.5]

d_missing = dtwcpp.distance.missing(x, y)
print(f"DTW with missing data:  {d_missing:.4f}")

# Standard DTW on the same data (NaN propagates, giving inf or NaN)
d_standard = dtwcpp.distance.dtw(x, y)
print(f"Standard DTW (with NaN): {d_standard}")
print()

# --- 2. Multiple missing values in both series ---
x2 = [1.0, float("nan"), 3.0, float("nan"), 5.0, 6.0]
y2 = [float("nan"), 2.0, 3.0, 4.0, float("nan"), 6.0]

d2 = dtwcpp.distance.missing(x2, y2)
print(f"Both series have gaps:  {d2:.4f}")

# Compare to clean data (no NaN)
x2_clean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y2_clean = [1.5, 2.0, 3.0, 4.0, 5.5, 6.0]
d2_clean = dtwcpp.distance.dtw(x2_clean, y2_clean)
print(f"Clean data DTW:         {d2_clean:.4f}")
print()

# --- 3. Banded DTW with missing data ---
rng = np.random.default_rng(42)
length = 100
x3 = rng.standard_normal(length).tolist()
y3 = rng.standard_normal(length).tolist()

# Introduce 10% missing values at random positions
for idx in rng.choice(length, size=10, replace=False):
    x3[idx] = float("nan")
for idx in rng.choice(length, size=10, replace=False):
    y3[idx] = float("nan")

d_full = dtwcpp.distance.missing(x3, y3, band=-1)
d_banded = dtwcpp.distance.missing(x3, y3, band=10)

print(f"Long series (n={length}, 10% missing each):")
print(f"  Full DTW:    {d_full:.4f}")
print(f"  Banded (10): {d_banded:.4f}")

# --- 4. Squared Euclidean metric ---
d_sqeuc = dtwcpp.distance.missing(x, y, metric="squared_euclidean")
print(f"\nSquared Euclidean metric: {d_sqeuc:.4f}")

