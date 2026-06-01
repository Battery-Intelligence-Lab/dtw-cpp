"""Run clustering on preprocessed Kasper sequences (full pipeline applied).

Uses the prototype `preprocess.power_signal` pipeline. Runs FastPAM and
FastCLARA over k=2..10 with full DTW (post-decimation lengths are ~3x smaller,
so band restriction is unnecessary). Also runs the domain reviewer's three
sanity checks per result.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import preprocess as pp

from dtwcpp import (
    Problem,
    fast_pam,
    fast_clara,
    silhouette,
    davies_bouldin_index,
)

OUT = Path(".claude/reports/test_kasper_analysis")
SRC = Path("data/test_kasper/sequence_example.parquet")


def load_raw():
    df = pq.read_table(str(SRC)).to_pandas()
    raw = [np.asarray(s, dtype=np.float64) for s in df["sequence"]]
    names = [f"ride_{int(r)}" for r in df["ride_number"]]
    rides = df["ride_number"].to_numpy()
    return raw, names, rides


def apply_pipeline(raw):
    processed = [pp.power_signal(s) for s in raw]
    lens = np.array([len(s) for s in processed])
    return processed, lens


def medoid_idle_fraction(raw_series, medoid_idx):
    s = raw_series[medoid_idx]
    return float(np.mean(np.abs(s) < 10.0))


def length_correlation(rides_lengths: np.ndarray, labels: np.ndarray) -> float:
    """Within-vs-between cluster ride_length variance ratio.

    Returns var_within / var_between (the per-cluster mean weighted).
    Domain reviewer: ratio >= 0.5 means clustering is NOT just by length.
    """
    rl = np.asarray(rides_lengths)
    lab = np.asarray(labels)
    total = rl.var()
    if total == 0:
        return 1.0
    # Within: weighted average of within-cluster variances
    clusters = np.unique(lab)
    within_var_sum = 0.0
    between_var_sum = 0.0
    overall_mean = rl.mean()
    for c in clusters:
        m = lab == c
        n = m.sum()
        within_var_sum += n * rl[m].var()
        between_var_sum += n * (rl[m].mean() - overall_mean) ** 2
    within = within_var_sum / len(rl)
    between = between_var_sum / len(rl)
    return float(within / (between + 1e-12))


def score_and_check(prob, res, k, raw_series, rides_lengths):
    prob.set_number_of_clusters(k)
    prob.clusters_ind = list(map(int, res.labels))
    prob.centroids_ind = list(map(int, res.medoid_indices))
    if not prob.is_distance_matrix_filled():
        prob.fill_distance_matrix()
    mean_sil = float(np.mean(silhouette(prob)))
    dbi = float(davies_bouldin_index(prob))
    sizes = np.bincount(np.asarray(res.labels, dtype=int), minlength=k).tolist()
    medoid_idles = [medoid_idle_fraction(raw_series, m) for m in res.medoid_indices]
    lvar = length_correlation(rides_lengths, np.asarray(res.labels))
    sanity = {
        "min_cluster_size": int(min(sizes)),
        "any_singleton": bool(min(sizes) <= 3),
        "max_medoid_idle": float(max(medoid_idles)),
        "any_idle_medoid": bool(max(medoid_idles) > 0.5),
        "within_between_length_var_ratio": lvar,
        "length_dominated": bool(lvar < 0.5),
    }
    return mean_sil, dbi, sizes, medoid_idles, sanity


def main():
    raw, names, rides = load_raw()
    rides_lengths = np.array([len(s) for s in raw])

    print("Applying preprocessing pipeline...")
    t0 = time.perf_counter()
    processed, post_lens = apply_pipeline(raw)
    print(f"  done in {time.perf_counter()-t0:.2f}s. "
          f"L_post: min={post_lens.min()} mean={post_lens.mean():.0f} max={post_lens.max()}")

    # Convert to list[list[float]] for the C++ bindings
    series = [s.tolist() for s in processed]

    summary = {
        "pipeline": "strip_idle(>=10) | decimate_zoh | sg(5,2) | diff | z_norm",
        "post_length_min": int(post_lens.min()),
        "post_length_mean": float(post_lens.mean()),
        "post_length_max": int(post_lens.max()),
        "pam_runs": [],
        "clara_runs": [],
    }

    print("\n--- FastPAM, full DTW, on preprocessed data ---")
    for k in [2, 3, 4, 5, 6, 8, 10]:
        p = Problem(f"pre_pam_k{k}")
        p.set_data(series, names)
        p.band = -1   # full DTW; post-decimation lengths make this tractable
        t0 = time.perf_counter()
        res = fast_pam(p, n_clusters=k, max_iter=200)
        t_pam = time.perf_counter() - t0

        sil, dbi, sizes, medoid_idles, sanity = score_and_check(
            p, res, k, raw, rides_lengths)
        entry = {
            "k": k, "cost": float(res.total_cost),
            "iterations": int(res.iterations),
            "converged": bool(res.converged),
            "fast_pam_sec": t_pam,
            "sizes": sizes, "medoid_indices": [int(x) for x in res.medoid_indices],
            "medoid_idle_fractions": medoid_idles,
            "mean_silhouette": sil, "davies_bouldin": dbi,
            "sanity": sanity,
            "labels": [int(x) for x in res.labels],
        }
        summary["pam_runs"].append(entry)
        flag = ""
        if sanity["any_singleton"]: flag += "[SINGLETON]"
        if sanity["any_idle_medoid"]: flag += "[IDLE-MEDOID]"
        if sanity["length_dominated"]: flag += "[LEN-DOM]"
        print(f"  k={k:2d}  cost={entry['cost']:11.3f}  sil={sil:+.4f}  db={dbi:.4f}  "
              f"sizes={sizes}  t={t_pam:.1f}s  {flag}")

    print("\n--- FastCLARA cross-check ---")
    for k in [3, 5]:
        ss = min(40 + 2 * k, len(series))
        p = Problem(f"pre_clara_k{k}")
        p.set_data(series, names)
        p.band = -1
        t0 = time.perf_counter()
        res = fast_clara(p, n_clusters=k, sample_size=ss,
                         n_samples=5, max_iter=200, seed=42)
        t_c = time.perf_counter() - t0
        sil, dbi, sizes, medoid_idles, sanity = score_and_check(
            p, res, k, raw, rides_lengths)
        entry = {
            "k": k, "sample_size": ss, "cost": float(res.total_cost),
            "time_sec": t_c, "sizes": sizes,
            "medoid_idle_fractions": medoid_idles,
            "mean_silhouette": sil, "davies_bouldin": dbi,
            "sanity": sanity,
        }
        summary["clara_runs"].append(entry)
        print(f"  CLARA k={k}: cost={entry['cost']:.3f} sil={sil:+.4f} db={dbi:.4f} "
              f"sizes={sizes} t={t_c:.1f}s")

    with open(OUT / "preprocessed_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT / 'preprocessed_results.json'}")

    # Headline labels for the best k by silhouette
    best = max(summary["pam_runs"], key=lambda r: r["mean_silhouette"])
    fn = OUT / f"preprocessed_pam_k{best['k']}_labels.csv"
    np.savetxt(fn,
               np.column_stack([rides, best["labels"]]),
               fmt="%d", delimiter=",",
               header="ride_number,cluster", comments="")
    print(f"Wrote headline labels (best k={best['k']}, sil={best['mean_silhouette']:+.4f}) to {fn}")


if __name__ == "__main__":
    main()
