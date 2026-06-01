"""Extended clustering with wheel 2.0.0: hierarchical + CH + dunn + new scores.

Tests on three feature-spaces:
  (a) Raw signal, L1 DTW, band=100
  (b) Preprocessed via dtwcpp.preprocess.power_signal, L1 DTW, full
  (c) 7-feature summary kMeans baseline (sklearn)
And cross-checks via ARI: do the three methods agree at k=2?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import dtwcpp
from dtwcpp import (
    Problem, fast_pam, silhouette, davies_bouldin_index,
    calinski_harabasz_index, dunn_index, adjusted_rand_index,
    build_dendrogram, cut_dendrogram, HierarchicalOptions, Linkage,
)
import dtwcpp.preprocess as pp

OUT = Path(".claude/reports/test_kasper_analysis")
SRC = Path("data/test_kasper/sequence_example.parquet")


def setup_problem(series, names, band):
    p = Problem("ext")
    p.set_data(series, names)
    p.band = band
    return p


def score_full(prob, labels, medoids, k):
    prob.set_number_of_clusters(k)
    prob.clusters_ind = list(map(int, labels))
    prob.centroids_ind = list(map(int, medoids)) if medoids else list(range(k))
    if not prob.is_distance_matrix_filled():
        prob.fill_distance_matrix()
    sil = float(np.mean(silhouette(prob)))
    db  = float(davies_bouldin_index(prob))
    try:
        ch = float(calinski_harabasz_index(prob))
    except Exception as e:
        ch = float('nan')
    try:
        dunn = float(dunn_index(prob))
    except Exception as e:
        dunn = float('nan')
    return sil, db, ch, dunn


def main():
    df = pq.read_table(str(SRC)).to_pandas()
    raw = [list(map(float, s)) for s in df["sequence"]]
    names = [f"ride_{int(r)}" for r in df["ride_number"]]

    # --- (a) raw L1 DTW, band=100, PAM + hierarchical, k=2..6 ---
    summary = {"raw_band100": {}, "preprocessed": {}, "features": {}, "ari": {}}
    print("\n=== (a) Raw, band=100 ===")
    for k in [2, 3, 4, 5, 6]:
        p = setup_problem(raw, names, 100)
        t0 = time.perf_counter()
        res = fast_pam(p, n_clusters=k, max_iter=200)
        t_pam = time.perf_counter() - t0
        sil, db, ch, dunn = score_full(p, res.labels, res.medoid_indices, k)
        sizes = np.bincount(np.asarray(res.labels, int), minlength=k).tolist()
        summary["raw_band100"][f"pam_k{k}"] = {
            "method": "fast_pam", "k": k, "cost": float(res.total_cost),
            "sil": sil, "db": db, "ch": ch, "dunn": dunn,
            "sizes": sizes, "labels": [int(x) for x in res.labels],
            "time_sec": t_pam,
        }
        print(f"  PAM   k={k}: sil={sil:+.4f} db={db:.3f} ch={ch:.2f} dunn={dunn:.3f} sizes={sizes} t={t_pam:.1f}s")

    # Hierarchical (average) on raw band=100
    p_hier = setup_problem(raw, names, 100)
    p_hier.fill_distance_matrix()
    hopts = HierarchicalOptions()
    hopts.linkage = Linkage.Average
    t0 = time.perf_counter()
    dend = build_dendrogram(p_hier, hopts)
    t_dend = time.perf_counter() - t0
    for k in [2, 3, 4, 5]:
        res = cut_dendrogram(dend, p_hier, k)
        # Hierarchical doesn't return medoids; set centroids to first member per cluster
        sil, db, ch, dunn = score_full(p_hier, res.labels, [], k)
        sizes = np.bincount(np.asarray(res.labels, int), minlength=k).tolist()
        summary["raw_band100"][f"hier_avg_k{k}"] = {
            "method": "hierarchical_average", "k": k,
            "sil": sil, "db": db, "ch": ch, "dunn": dunn,
            "sizes": sizes, "labels": [int(x) for x in res.labels],
        }
        print(f"  HIER  k={k}: sil={sil:+.4f} db={db:.3f} ch={ch:.2f} dunn={dunn:.3f} sizes={sizes}")
    print(f"  (dendrogram built in {t_dend:.1f}s)")

    # --- (b) preprocessed via power_signal, full DTW ---
    print("\n=== (b) Preprocessed (power_signal), full DTW ===")
    processed = [pp.power_signal(s).tolist() for s in raw]
    for k in [2, 3, 4, 5, 6]:
        p = setup_problem(processed, names, -1)
        t0 = time.perf_counter()
        res = fast_pam(p, n_clusters=k, max_iter=200)
        t_pam = time.perf_counter() - t0
        sil, db, ch, dunn = score_full(p, res.labels, res.medoid_indices, k)
        sizes = np.bincount(np.asarray(res.labels, int), minlength=k).tolist()
        summary["preprocessed"][f"pam_k{k}"] = {
            "method": "fast_pam", "k": k, "cost": float(res.total_cost),
            "sil": sil, "db": db, "ch": ch, "dunn": dunn,
            "sizes": sizes, "labels": [int(x) for x in res.labels],
            "time_sec": t_pam,
        }
        print(f"  PAM   k={k}: sil={sil:+.4f} db={db:.3f} ch={ch:.2f} dunn={dunn:.3f} sizes={sizes} t={t_pam:.1f}s")

    # --- (c) feature kMeans ---
    print("\n=== (c) Feature kMeans baseline ===")
    seqs = [np.asarray(s) for s in raw]
    feats = np.column_stack([
        [s.mean() for s in seqs],
        [s.std() for s in seqs],
        [s.max() for s in seqs],
        [s.min() for s in seqs],
        [np.mean(np.abs(s) < 10) for s in seqs],
        [len(s) for s in seqs],
        [np.abs(s).sum() for s in seqs],
    ])
    feats_z = (feats - feats.mean(0)) / feats.std(0)
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(feats_z)
        sil_f = float(silhouette_score(feats_z, km.labels_))
        db_f = float(davies_bouldin_score(feats_z, km.labels_))
        ch_f = float(calinski_harabasz_score(feats_z, km.labels_))
        sizes = np.bincount(km.labels_, minlength=k).tolist()
        summary["features"][f"kmeans_k{k}"] = {
            "method": "feature_kmeans", "k": k, "inertia": float(km.inertia_),
            "sil": sil_f, "db": db_f, "ch": ch_f,
            "sizes": sizes, "labels": [int(x) for x in km.labels_],
        }
        print(f"  KMEANS k={k}: sil={sil_f:+.4f} db={db_f:.3f} ch={ch_f:.2f} sizes={sizes}")

    # --- (d) ARI cross-method agreement at k=2..4 ---
    print("\n=== (d) ARI agreement (DTW-raw vs feature-kMeans vs preprocessed) ===")
    p_ref = setup_problem(raw, names, 100)
    p_ref.fill_distance_matrix()
    for k in [2, 3, 4]:
        l_raw  = np.array(summary["raw_band100"][f"pam_k{k}"]["labels"])
        l_pre  = np.array(summary["preprocessed"][f"pam_k{k}"]["labels"])
        l_feat = np.array(summary["features"][f"kmeans_k{k}"]["labels"])
        ari_raw_feat = float(adjusted_rand_index(l_raw.tolist(), l_feat.tolist()))
        ari_raw_pre  = float(adjusted_rand_index(l_raw.tolist(), l_pre.tolist()))
        ari_pre_feat = float(adjusted_rand_index(l_pre.tolist(), l_feat.tolist()))
        summary["ari"][f"k{k}"] = {
            "raw_vs_features": ari_raw_feat,
            "raw_vs_preprocessed": ari_raw_pre,
            "preprocessed_vs_features": ari_pre_feat,
        }
        print(f"  k={k}: raw<>feat={ari_raw_feat:+.4f}  raw<>pre={ari_raw_pre:+.4f}  pre<>feat={ari_pre_feat:+.4f}")

    with open(OUT / "extended_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT / 'extended_results.json'}")


if __name__ == "__main__":
    main()
