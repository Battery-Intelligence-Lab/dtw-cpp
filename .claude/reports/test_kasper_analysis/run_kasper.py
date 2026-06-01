"""Kasper parquet sample — DTWC++ characterization + clustering.

Private analysis script — NOT for repo distribution.
Targets installed wheel v1.0.0 (snake_case API, no hierarchical, no CH-index).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import dtwcpp
from dtwcpp import (
    Problem,
    fast_pam,
    fast_clara,
    silhouette,
    davies_bouldin_index,
    compute_distance_matrix,
)

OUT = Path(".claude/reports/test_kasper_analysis")
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("data/test_kasper/sequence_example.parquet")


def load_data():
    table = pq.read_table(str(SRC))
    df = table.to_pandas()
    series = [list(map(float, s)) for s in df["sequence"]]
    names = [f"ride_{int(r)}" for r in df["ride_number"]]
    lengths = np.array([len(s) for s in series])
    rides = df["ride_number"].to_numpy()
    return series, names, lengths, rides


def build_problem(series, names, band):
    p = Problem("kasper")
    p.set_data(series, names)
    p.band = band
    return p


def pam_run(series, names, band, k):
    p = build_problem(series, names, band)
    t0 = time.perf_counter()
    res = fast_pam(p, n_clusters=k, max_iter=200)
    t_pam = time.perf_counter() - t0

    if not p.is_distance_matrix_filled():
        t0 = time.perf_counter()
        p.fill_distance_matrix()
        t_dm = time.perf_counter() - t0
    else:
        t_dm = 0.0

    try:
        sil = silhouette(p)
        mean_sil = float(np.mean(sil))
    except Exception as e:
        sil = []
        mean_sil = float("nan")
        print(f"  silhouette failed: {e}")
    try:
        db = float(davies_bouldin_index(p))
    except Exception as e:
        db = float("nan")
        print(f"  db_index failed: {e}")

    sizes = np.bincount(np.asarray(res.labels, dtype=int), minlength=k).tolist()
    return {
        "k": k,
        "band": band,
        "fast_pam_sec": t_pam,
        "post_distmat_sec": t_dm,
        "cost": float(res.total_cost),
        "iterations": int(res.iterations),
        "converged": bool(res.converged),
        "labels": [int(x) for x in res.labels],
        "medoids": [int(x) for x in res.medoid_indices],
        "sizes": sizes,
        "mean_silhouette": mean_sil,
        "silhouette_per_series": [float(x) for x in sil],
        "davies_bouldin": db,
    }


def clara_run(series, names, band, k, sample_size, n_samples=5, seed=42):
    p = build_problem(series, names, band)
    t0 = time.perf_counter()
    res = fast_clara(p, n_clusters=k, sample_size=sample_size,
                    n_samples=n_samples, max_iter=200, seed=seed)
    t_clara = time.perf_counter() - t0

    if not p.is_distance_matrix_filled():
        p.fill_distance_matrix()
    try:
        mean_sil = float(np.mean(silhouette(p)))
    except Exception:
        mean_sil = float("nan")
    sizes = np.bincount(np.asarray(res.labels, dtype=int), minlength=k).tolist()
    return {
        "method": "fast_clara",
        "k": k,
        "band": band,
        "sample_size": sample_size,
        "n_samples": n_samples,
        "time_sec": t_clara,
        "cost": float(res.total_cost),
        "mean_silhouette": mean_sil,
        "sizes": sizes,
        "labels": [int(x) for x in res.labels],
        "medoids": [int(x) for x in res.medoid_indices],
    }


def main():
    series, names, lengths, rides = load_data()
    N = len(series)
    print(f"Loaded {N} series, length range [{lengths.min()}, {lengths.max()}], "
          f"mean {lengths.mean():.1f}")

    # System info
    summary = {
        "dataset": {
            "path": str(SRC),
            "n_series": int(N),
            "length_min": int(lengths.min()),
            "length_max": int(lengths.max()),
            "length_mean": float(lengths.mean()),
            "length_median": float(np.median(lengths)),
            "length_std": float(lengths.std()),
            "ride_number_range": [int(rides.min()), int(rides.max())],
        },
        "wheel_version": dtwcpp.__version__,
        "pam_runs": [],
        "clara_runs": [],
    }

    # --- FastPAM (band, k) sweep ---
    bands = [50, 100, 200]
    k_values = [2, 3, 4, 5, 6, 8]
    for band in bands:
        print(f"\n--- band={band} ---")
        for k in k_values:
            r = pam_run(series, names, band, k)
            summary["pam_runs"].append(r)
            print(f"  k={k:2d}  cost={r['cost']:11.3f}  sil={r['mean_silhouette']:+.4f}  "
                  f"db={r['davies_bouldin']:.4f}  iters={r['iterations']}  "
                  f"sizes={r['sizes']}  t_pam={r['fast_pam_sec']:.2f}s "
                  f"t_dm={r['post_distmat_sec']:.2f}s")

    # --- CLARA cross-check ---
    print("\n--- FastCLARA cross-check (band=100) ---")
    for k in [3, 5]:
        ss = min(40 + 2 * k, N)
        cr = clara_run(series, names, 100, k, sample_size=ss, n_samples=5, seed=42)
        summary["clara_runs"].append(cr)
        print(f"  CLARA k={k}: cost={cr['cost']:.3f} sil={cr['mean_silhouette']:+.4f} "
              f"sizes={cr['sizes']} t={cr['time_sec']:.2f}s")

    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT / 'results.json'}")

    # Headline labels file
    head = next(r for r in summary["pam_runs"] if r["band"] == 100 and r["k"] == 3)
    np.savetxt(OUT / "headline_pam_band100_k3_labels.csv",
               np.column_stack([rides, head["labels"]]),
               fmt="%d", delimiter=",",
               header="ride_number,cluster", comments="")
    print(f"Wrote {OUT / 'headline_pam_band100_k3_labels.csv'}")


if __name__ == "__main__":
    main()
