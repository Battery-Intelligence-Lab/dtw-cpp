"""
@file 09_device_clustering.py
@brief PyTorch-style device clustering demo: cpu / gpu / hpc in one switch.
@details
    Pipeline:  pick device -> load data -> full DTW distance matrix
               -> k-medoids (FastPAM) -> timing -> 2D cluster plot.

    device="cpu"  runs locally on CPU.
    device="gpu"  runs locally on the GPU (CUDA); auto-falls back to CPU
                  with a warning if no GPU / not compiled with CUDA.
    device="hpc"  offloads to a SLURM cluster (Oxford ARC) via the tested
                  scripts/slurm/slurm_remote.sh wrapper.

    Works on Windows and macOS. Requires: numpy, matplotlib.
    HPC mode additionally needs ssh + rsync (Git Bash on Windows) and a
    configured .env at the repo root (see scripts/slurm/env.example).

@author Volkan Kumtepeli
"""
import time
import numpy as np
import dtwcpp


# ─────────────────────────────────────────────────────────────────────────
# Helpers (keep the demo body short)
# ─────────────────────────────────────────────────────────────────────────
def resolve_device(name):
    """Map friendly names to dtwcpp backends. 'gpu' -> 'cuda' (CPU fallback)."""
    name = name.lower()
    if name == "cpu":
        return "cpu"
    if name == "gpu":
        return "cuda"          # dtwcpp warns + falls back to CPU if no GPU
    raise ValueError("device must be 'cpu', 'gpu', or 'hpc'")


def load_data(source="synthetic", seed=0):
    """Return (X as list-of-lists, true_labels).

    source="synthetic": 3 well-separated shape groups (offline, deterministic).
    source=<path.tsv/.csv>: UCR format — column 0 = class label, rest = series.
    """
    if source not in ("synthetic",) and (source.endswith(".tsv") or source.endswith(".csv")):
        delim = "\t" if source.endswith(".tsv") else ","
        arr = np.loadtxt(source, delimiter=delim)
        y = arr[:, 0].astype(int)
        X = arr[:, 1:]
        return [list(r) for r in X], y

    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, 50)
    groups, y = [], []
    for cid, shape in enumerate((np.sin(t), np.sign(np.sin(t)), t / t.max())):
        for _ in range(20):                       # 20 series per group
            shift = rng.randint(-3, 4)             # small time warp
            s = np.roll(shape, shift) + 0.15 * rng.randn(t.size)
            groups.append(list(s))
            y.append(cid)
    return groups, np.array(y)


def plot_clusters_2d(D, labels, medoids, dev, png="clusters_2d.png"):
    """Classical MDS (PCoA) of the DTW distance matrix -> 2D scatter by cluster."""
    import matplotlib
    import matplotlib.pyplot as plt

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n           # centering matrix
    B = -0.5 * J @ (D ** 2) @ J                    # double-centered Gram matrix
    w, V = np.linalg.eigh(B)
    top = np.argsort(w)[::-1][:2]                  # top-2 eigenpairs
    XY = V[:, top] * np.sqrt(np.maximum(w[top], 0.0))

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(XY[:, 0], XY[:, 1], c=labels, cmap="tab10", s=45,
                    edgecolor="white", linewidth=0.5)
    ax.scatter(XY[medoids, 0], XY[medoids, 1], marker="*", s=320,
               c="black", label="medoids", zorder=3)
    ax.set_title(f"DTW k-medoids  (device={dev}, {n} series)")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2"); ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"plot saved -> {png}")
    if "agg" not in matplotlib.get_backend().lower():   # interactive backend only
        plt.show()


def cluster_on_hpc(X, k, repo_root):
    """Offload the whole clustering to SLURM via dtwcpp's native device='hpc' path.

    Requires a configured .env at the repo root + ssh/rsync (Git Bash on Windows)
    and a one-time cluster build: `bash scripts/slurm/slurm_remote.sh build htc-cpu`.
    NOTE: cluster submission cannot be verified on a dev laptop — run on ARC.
    """
    import os
    os.environ.setdefault("DTWC_REPO_ROOT", str(repo_root))
    t0 = time.perf_counter()
    labels = dtwcpp.DTWClustering(n_clusters=k, device="hpc").fit_predict(X)
    print(f"[device=hpc] {len(X)} series, k={k}  ->  {time.perf_counter() - t0:6.1f} s"
          f"   (ran on SLURM)")
    print("cluster sizes:", np.bincount(np.asarray(labels)))
    return labels


# ─────────────────────────────────────────────────────────────────────────
# Demo  (the ~12-line core)
# ─────────────────────────────────────────────────────────────────────────
def main(device="gpu", k=3, data="synthetic"):
    from pathlib import Path
    X, y_true = load_data(data)                                   # list-of-lists, true labels
    names = [str(i) for i in range(len(X))]

    if device == "hpc":                                           # whole job -> SLURM
        return cluster_on_hpc(X, k, repo_root=Path(__file__).resolve().parents[2])

    dev = resolve_device(device)                                  # gpu -> cuda (+CPU fallback)
    t0 = time.perf_counter()
    D = dtwcpp.compute_distance_matrix(X, band=-1, device=dev)    # FULL DTW matrix, on device
    prob = dtwcpp.Problem("device_demo")
    prob.set_data(X, names)
    prob.set_distance_matrix_from_numpy(D)
    res = dtwcpp.fast_pam(prob, k, 100)                           # k-medoids (FastPAM)
    dt = time.perf_counter() - t0

    print(f"[device={dev}] {len(X)} series, k={k}  ->  {dt * 1e3:7.1f} ms"
          f"   (cost={res.total_cost:.2f}, iters={res.iterations})")
    plot_clusters_2d(D, np.array(res.labels), np.array(res.medoid_indices), dev)


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "gpu"             # cpu | gpu | hpc
    main(device=dev)
