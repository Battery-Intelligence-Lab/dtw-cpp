"""Re-score silhouette + DBI by writing PAM labels/medoids back into Problem.

Reads `results.json` produced by run_kasper.py, re-runs each PAM config
(cheaply — distance matrix is cached on the Problem after fill), assigns
the labels into prob.clusters_ind/centroids_ind, then computes scores.
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
    fast_clara,
    silhouette,
    davies_bouldin,
)

OUT = Path(".claude/reports/test_kasper_analysis")
SRC = Path("data/test_kasper/sequence_example.parquet")


def load_data():
    df = pq.read_table(str(SRC)).to_pandas()
    series = [list(map(float, s)) for s in df["sequence"]]
    names = [f"ride_{int(r)}" for r in df["ride_number"]]
    return series, names, df["ride_number"].to_numpy()


def score(prob: Problem, labels, medoids, k):
    # Wire result back into prob state so silhouette/DBI work.
    prob.set_n_clusters(k)
    prob.clusters_ind = list(map(int, labels))
    prob.centroids_ind = list(map(int, medoids))
    if not prob.is_distance_matrix_filled():
        prob.fill_distance_matrix()
    sil = silhouette(prob)
    db = davies_bouldin(prob)
    return float(np.mean(sil)), float(db), [float(s) for s in sil]


def main():
    with open(OUT / "results.json") as f:
        summary = json.load(f)
    series, names, rides = load_data()

    # Re-score PAM runs
    new_pam = []
    for r in summary["pam_runs"]:
        band = r["band"]; k = r["k"]
        p = Problem(f"k{k}_b{band}")
        p.set_data(series, names)
        p.band = band
        res = fast_pam(p, n_clusters=k, max_iter=200)
        ms, dbi, sil = score(p, res.labels, res.medoid_indices, k)
        r["mean_silhouette"] = ms
        r["davies_bouldin"] = dbi
        r["silhouette_per_series"] = sil
        new_pam.append(r)
        print(f"  band={band:3d} k={k:2d}  cost={r['cost']:11.3f}  "
              f"sil={ms:+.4f}  db={dbi:.4f}  sizes={r['sizes']}")
    summary["pam_runs"] = new_pam

    # Re-score CLARA runs
    for r in summary["clara_runs"]:
        from dtwcpp import fast_clara
        band = r["band"]; k = r["k"]
        p = Problem(f"clara_k{k}_b{band}")
        p.set_data(series, names)
        p.band = band
        res = fast_clara(p, n_clusters=k, sample_size=r["sample_size"],
                         n_samples=r["n_samples"], max_iter=200, seed=42)
        ms, dbi, _sil = score(p, res.labels, res.medoid_indices, k)
        r["mean_silhouette"] = ms
        r["davies_bouldin"] = dbi
        print(f"  CLARA band={band} k={k}: cost={r['cost']:.3f} sil={ms:+.4f} db={dbi:.4f} sizes={r['sizes']}")

    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote rescored {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
