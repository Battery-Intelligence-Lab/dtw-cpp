"""
@file 08_parquet_io.py
@brief DTWC++ Parquet / Arrow IPC I/O — save, convert, load, and cluster.
@details
Demonstrates:
  - Saving time series to Parquet (pyarrow)
  - Converting Parquet -> Arrow IPC via dtwc-convert
  - Loading Arrow IPC data for clustering
  - Float32 vs float64 precision comparison

Requires: pyarrow  (uv add pyarrow)
@author Volkan Kumtepeli
"""

import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import dtwcpp
from dtwcpp.io import save_dataset_parquet, load_dataset_parquet

# --- 1. Create a small dataset and save it as Parquet ---
rng = np.random.default_rng(42)
n_series, length, n_clusters = 60, 40, 3

groups = [rng.standard_normal((20, length)) + k * 15 for k in range(n_clusters)]
X = np.vstack(groups)                         # (60, 40) float64
names = [f"series_{i}" for i in range(len(X))]

with tempfile.TemporaryDirectory() as tmp:
    parquet_path = Path(tmp) / "timeseries.parquet"
    arrow_path   = Path(tmp) / "timeseries.arrow"

    save_dataset_parquet(X, parquet_path, names=names)
    print(f"Saved {len(X)} series to {parquet_path.name} "
          f"({parquet_path.stat().st_size / 1024:.1f} KB)")

    # --- 2. Round-trip check via load_dataset_parquet ---
    X_loaded, col_names = load_dataset_parquet(parquet_path)
    assert X_loaded.shape == X.shape, "Shape mismatch after Parquet round-trip"
    np.testing.assert_allclose(X_loaded, X, rtol=1e-12)
    print(f"Parquet round-trip OK: shape={X_loaded.shape}, columns={col_names[:3]}...")
    print()

    # --- 3. Convert Parquet -> Arrow IPC with dtwc-convert ---
    ret = subprocess.run(
        [sys.executable, "-m", "dtwcpp.convert",
         str(parquet_path), "-o", str(arrow_path)],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print("dtwc-convert failed:", ret.stderr)
        sys.exit(1)
    print(ret.stdout.strip())
    print(f"Arrow IPC size: {arrow_path.stat().st_size / 1024:.1f} KB")
    print()

    # --- 4. Load Arrow IPC and cluster (float64 path) ---
    import pyarrow.ipc as ipc
    with ipc.open_file(str(arrow_path)) as reader:
        batch = reader.get_batch(0)
    series_f64 = [np.array(batch["data"][i].as_py()) for i in range(batch.num_rows)]
    series_names = [str(batch["name"][i].as_py()) for i in range(batch.num_rows)]

    prob64 = dtwcpp.Problem("parquet_f64")
    prob64.set_data([list(s) for s in series_f64], series_names)
    prob64.band = 10

    result64 = dtwcpp.fast_pam(prob64, n_clusters=n_clusters)
    print(f"Float64 clustering — cost: {result64.total_cost:.4f}, "
          f"sizes: {np.bincount(result64.labels)}")

    # --- 5. Float32 vs float64 comparison ---
    # Cast to float32; expect similar cost with 2x memory saving.
    series_f32 = [s.astype(np.float32) for s in series_f64]

    prob32 = dtwcpp.Problem("parquet_f32")
    # Pass float32 lists directly — dtwcpp accepts float32 arrays via set_data.
    prob32.set_data([list(s) for s in series_f32], series_names)
    prob32.band = 10

    result32 = dtwcpp.fast_pam(prob32, n_clusters=n_clusters)
    print(f"Float32 clustering — cost: {result32.total_cost:.4f}, "
          f"sizes: {np.bincount(result32.labels)}")

    rel_err = abs(result32.total_cost - result64.total_cost) / (result64.total_cost + 1e-12)
    print(f"Relative cost difference (f32 vs f64): {rel_err:.2e}")
