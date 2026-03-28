"""
DTWC++ DTW Variants — Compare all 5 DTW distance measures.

Demonstrates: Standard DTW, DDTW, WDTW, ADTW, Soft-DTW.
"""

import numpy as np
import dtwcpp

# Create two time series
np.random.seed(42)
t = np.linspace(0, 4 * np.pi, 200)
x = np.sin(t).tolist()
y = np.sin(t + 0.5).tolist()  # phase-shifted version

print("=== DTW Variant Comparison ===")
print(f"Series length: {len(x)}")
print()

# Standard DTW
d_std = dtwcpp.dtw_distance(x, y, band=-1)
print(f"Standard DTW:     {d_std:.4f}")

# Banded DTW (Sakoe-Chiba)
d_banded = dtwcpp.dtw_distance(x, y, band=20)
print(f"Banded DTW (b=20): {d_banded:.4f}")

# DDTW — Derivative DTW (shape-based, ignores amplitude)
d_ddtw = dtwcpp.ddtw_distance(x, y, band=-1)
print(f"DDTW (derivative): {d_ddtw:.4f}")

# WDTW — Weighted DTW (penalizes off-diagonal alignment)
d_wdtw_lo = dtwcpp.wdtw_distance(x, y, band=-1, g=0.01)
d_wdtw_hi = dtwcpp.wdtw_distance(x, y, band=-1, g=0.5)
print(f"WDTW (g=0.01):    {d_wdtw_lo:.4f}  (lenient)")
print(f"WDTW (g=0.50):    {d_wdtw_hi:.4f}  (strict)")

# ADTW — Amerced DTW (penalizes non-diagonal steps)
d_adtw_lo = dtwcpp.adtw_distance(x, y, band=-1, penalty=0.1)
d_adtw_hi = dtwcpp.adtw_distance(x, y, band=-1, penalty=5.0)
print(f"ADTW (p=0.1):     {d_adtw_lo:.4f}  (lenient)")
print(f"ADTW (p=5.0):     {d_adtw_hi:.4f}  (strict)")

# Soft-DTW — Differentiable (for gradient-based optimization)
d_soft_lo = dtwcpp.soft_dtw_distance(x, y, gamma=0.01)
d_soft_hi = dtwcpp.soft_dtw_distance(x, y, gamma=10.0)
print(f"Soft-DTW (g=0.01): {d_soft_lo:.4f}  (~ hard DTW)")
print(f"Soft-DTW (g=10):  {d_soft_hi:.4f}  (smooth)")

# Soft-DTW gradient — unique to Soft-DTW, enables optimization
grad = dtwcpp.soft_dtw_gradient(x, y, gamma=1.0)
print(f"\nSoft-DTW gradient norm: {np.linalg.norm(grad):.4f}")
print(f"Gradient shape: {len(grad)} (same as input series)")

# --- Preprocessing: derivative transform ---
dx = dtwcpp.derivative_transform(x)
print(f"\nDerivative transform: first 5 values = {[f'{v:.3f}' for v in dx[:5]]}")

# --- Z-normalization ---
z = dtwcpp.z_normalize([10, 20, 30, 40, 50])
print(f"Z-normalize([10..50]) = [{', '.join(f'{v:.3f}' for v in z)}]")
print(f"  Mean = {np.mean(z):.6f}, Std = {np.std(z):.6f}")
