"""Re-run FastPAM after per-series z-normalization (addresses library reviewer finding 1).

The independent reviewer warned: raw L1 on values [-734, +1046] with no normalization
will cluster by magnitude, not shape. This script applies dtwcpp.z_normalize per ride
and re-runs FastPAM at band=100 and band=-1 (full DTW).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dtwcpp import (
    Problem,
    fast_pam,
    silhouette,
    davies_bouldin_index,
    z_normalize,
)

OUT = Path(".claude/reports/test_kasper_analysis")
SRC = Path("data/test_kasper/sequence_example.parquet")


def main():
    df = pq.read_table(str(SRC)).to_pandas()
    raw_series = [list(map(float, s)) for s in df["sequence"]]
    names = [f"ride_{int(r)}" for r in df["ride_number"]]
    rides = df["ride_number"].to_numpy()

    # Apply z-normalization per series
    znorm_series = [list(z_normalize(s)) for s in raw_series]

    out: dict = {"znorm_pam_runs": []}

    for band in [100, -1]:
        print(f"\n--- z-norm, band={band} ---")
        for k in [2, 3, 4, 5, 6, 8]:
            p = Problem(f"znorm_k{k}_b{band}")
            p.set_data(znorm_series, names)
            p.band = band

            t0 = time.perf_counter()
            res = fast_pam(p, n_clusters=k, max_iter=200)
            t_pam = time.perf_counter() - t0

            # Wire result for scoring
            p.set_number_of_clusters(k)
            p.clusters_ind = list(map(int, res.labels))
            p.centroids_ind = list(map(int, res.medoid_indices))
            if not p.is_distance_matrix_filled():
                p.fill_distance_matrix()
            ms = float(np.mean(silhouette(p)))
            dbi = float(davies_bouldin_index(p))

            sizes = np.bincount(np.asarray(res.labels, dtype=int), minlength=k).tolist()
            entry = {
                "band": band, "k": k,
                "cost": float(res.total_cost),
                "iterations": int(res.iterations),
                "converged": bool(res.converged),
                "mean_silhouette": ms,
                "davies_bouldin": dbi,
                "sizes": sizes,
                "labels": [int(x) for x in res.labels],
                "medoids": [int(x) for x in res.medoid_indices],
                "fast_pam_sec": t_pam,
            }
            out["znorm_pam_runs"].append(entry)
            print(f"  k={k:2d}  cost={entry['cost']:11.3f}  sil={ms:+.4f}  "
                  f"db={dbi:.4f}  sizes={sizes}  t_pam={t_pam:.2f}s")

    with open(OUT / "znorm_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT / 'znorm_results.json'}")


if __name__ == "__main__":
    main()
