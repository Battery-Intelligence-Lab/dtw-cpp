---
description: "Compute DTW distances — two series or full pairwise matrix. Compare variants across user's data."
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# DTWC++ Distance

Compute DTW distance(s). `$ARGUMENTS` describes: two series, or a dataset (pairwise matrix).

## Step 0: Mode detection

- **Two-series mode**: args contain two file paths / inline arrays / "compare series X Y in file Z"
- **Matrix mode**: args contain a single dataset path and user wants all-pairs distances

## Step 1: Parse variant and params

Every variant is `dc.distance.dtw(x, y, variant=..., ...)`; the keywords are the
`dtwc_cl` names.

| Variant | Params |
|---------|--------|
| standard | `band`, `metric` (`l1` or `squared_euclidean`) |
| ddtw | `band`, `metric` |
| wdtw | `band`, `wdtw_g` (default 0.05) |
| adtw | `band`, `adtw_penalty` (default 1.0) |
| softdtw | `sdtw_gamma` (default 1.0) |
| msm | `msm_c` (default 1.0) |
| twe | `twe_nu` (default 0.001), `twe_lambda` (default 1.0) |

Also ask: `device` (cpu/cuda), `metric` (l1/squared_euclidean).

## Step 2a: Two-series mode

Write and run:

```python
import numpy as np
import dtwcpp as dc

x = np.asarray(X, dtype=np.float64)
y = np.asarray(Y, dtype=np.float64)

print(f"Series X: len={len(x)}, Y: len={len(y)}")

# Compare variants
variants = {
    "Standard DTW":       dc.distance.dtw(x, y, band=BAND),
    "DDTW":               dc.distance.dtw(x, y, variant="ddtw", band=BAND),
    "WDTW (g=0.05)":      dc.distance.dtw(x, y, variant="wdtw", wdtw_g=0.05, band=BAND),
    "ADTW (penalty=1.0)": dc.distance.dtw(x, y, variant="adtw", adtw_penalty=1.0, band=BAND),
    "Soft-DTW (γ=1.0)":   dc.distance.dtw(x, y, variant="softdtw", sdtw_gamma=1.0),
}

print(f"\n{'Variant':<24}  Distance")
print("-" * 40)
for name, d in variants.items():
    print(f"{name:<24}  {d:.6f}")

# Soft-DTW gradient (differentiability indicator)
grad = dc.soft_dtw_gradient(x, y, gamma=1.0)
print(f"\nSoft-DTW ‖∇_x‖ = {np.linalg.norm(grad):.4f}")
```

Report the variants table and highlight the chosen one.

## Step 2b: Matrix mode

```python
import numpy as np
import dtwcpp as dc
import time

data = dc.load("INPUT").as_data()  # CSV/TSV, Parquet or Arrow IPC
print(f"Loaded {data.size} series")

# Compute pairwise: a Problem's matrix, set with the dc.distance.dtw keywords
t0 = time.time()
prob = dc.Problem("distances", device="cpu")  # or "cuda"
prob.set_distance(variant="standard", band=BAND)
prob.set_data(data)
dm = prob.distance_matrix()
elapsed = time.time() - t0

print(f"Computed {data.size}x{data.size} matrix in {elapsed:.2f}s")
print(f"Shape: {dm.shape if hasattr(dm, 'shape') else data.size}")

# Matrix stats
arr = np.asarray(dm)
off_diag = arr[~np.eye(arr.shape[0], dtype=bool)]
print(f"Min non-zero: {off_diag.min():.4f}")
print(f"Max:          {off_diag.max():.4f}")
print(f"Mean:         {off_diag.mean():.4f}")
print(f"Symmetric:    {np.allclose(arr, arr.T)}")

# Save if requested
# np.savetxt("distances.csv", arr, delimiter=",")
```

## Step 3: Output format

Two-series: variant table + chosen distance.
Matrix: shape / min / max / mean / save path / wall time.

Suggest next actions:
- "Run `/cluster` to group by similarity"
- "Run `/visualize distance-matrix DISTANCES.csv` to see heatmap"

## Error handling

- Series too short (`len < 2`): warn and abort
- `gamma <= 0` for soft-DTW: clamp to `1e-3` or error
- Mismatched dtypes: coerce to `float64`

## Related

- `/cluster` — full clustering pipeline
- `/visualize` — plot matrix
- `/help` — variant reference
