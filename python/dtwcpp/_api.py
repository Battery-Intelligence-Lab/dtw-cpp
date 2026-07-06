"""
@file _api.py
@brief Unified high-level interface: device() -> load() -> cluster() -> result.plot().
@details
    A PyTorch-style flow where the device is set once (globally) and a single
    cluster() call does the right thing for that device:

        import dtwcpp as dtwc
        dtwc.device("hpc")                 # cpu | gpu | hpc  (set once)
        data = dtwc.load("Crop_TRAIN.tsv") # lazy handle (no local read on hpc)
        res  = dtwc.cluster(data, k=3)     # local for cpu/gpu; offloaded for hpc
        print(res.summary())
        res.plot()

    All device resolution lives in the library (dtwcpp._parse_device /
    _resolve_device); callers never map 'gpu'->'cuda' themselves.
@author Volkan Kumtepeli
"""
import os
import time

import numpy as np


class Dataset:
    """A lazy handle to time-series data — a local array or a file path.

    The data is only materialized (read into memory) when a *local* backend
    needs it. On ``device="hpc"`` a path is passed straight to the cluster and
    never read locally, so it scales beyond what fits on the calling machine.
    """

    def __init__(self, source, *, skip_cols=0, delimiter=None, name=None):
        self.source = source
        self.skip_cols = skip_cols
        self.delimiter = delimiter
        if name is not None:
            self.name = name
        elif self.is_path:
            self.name = os.path.splitext(os.path.basename(str(source)))[0]
        else:
            self.name = "dataset"
        self._series = None

    @property
    def is_path(self):
        return isinstance(self.source, (str, os.PathLike))

    def as_series(self):
        """Materialize to a list of 1-D series (reads the file if a path). Cached."""
        if self._series is None:
            if self.is_path:
                delim = self.delimiter or (
                    "\t" if str(self.source).endswith((".tsv", ".txt")) else ",")
                arr = np.loadtxt(self.source, delimiter=delim)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                arr = arr[:, self.skip_cols:]
                self._series = [list(row) for row in arr]
            else:
                arr = np.asarray(self.source, dtype=float)
                self._series = [list(row) for row in arr]
        return self._series


def load(source, *, skip_cols=0, delimiter=None, name=None):
    """Wrap a path or array in a lazy :class:`Dataset` handle (does not read it)."""
    if isinstance(source, Dataset):
        return source
    return Dataset(source, skip_cols=skip_cols, delimiter=delimiter, name=name)


class ClusterResult:
    """Outcome of :func:`cluster` — labels plus timing, and a 2D plot."""

    def __init__(self, labels, *, device, elapsed_s, k, n_series,
                 medoid_indices=None, distance_matrix=None, cost=None, name="dataset"):
        self.labels = np.asarray(labels)
        self.device = device
        self.elapsed_s = elapsed_s
        self.k = k
        self.n_series = n_series
        self.medoid_indices = None if medoid_indices is None else np.asarray(medoid_indices)
        self.distance_matrix = distance_matrix
        self.cost = cost
        self.name = name

    def summary(self):
        extra = f", cost={self.cost:.2f}" if self.cost is not None else ""
        return (f"[device={self.device}] {self.n_series} series, k={self.k}  ->  "
                f"{self.elapsed_s * 1e3:7.1f} ms{extra}")

    def plot(self, png="clusters_2d.png", show=True):
        """Classical-MDS 2D scatter of the distance matrix, coloured by cluster.

        Needs a local distance matrix (cpu/gpu runs). On an hpc run only labels
        come back, so this prints the cluster sizes and returns ``None``.
        """
        if self.distance_matrix is None:
            sizes = np.bincount(self.labels).tolist()
            print(f"no local distance matrix to plot ({self.device} run); "
                  f"labels in result.labels, cluster sizes={sizes}")
            return None

        import matplotlib
        import matplotlib.pyplot as plt

        D = np.asarray(self.distance_matrix, dtype=float)
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n          # centering matrix
        B = -0.5 * J @ (D ** 2) @ J                    # double-centered Gram matrix
        w, V = np.linalg.eigh(B)
        top = np.argsort(w)[::-1][:2]                  # top-2 eigenpairs
        XY = V[:, top] * np.sqrt(np.maximum(w[top], 0.0))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(XY[:, 0], XY[:, 1], c=self.labels, cmap="tab10", s=45,
                   edgecolor="white", linewidth=0.5)
        if self.medoid_indices is not None:
            ax.scatter(XY[self.medoid_indices, 0], XY[self.medoid_indices, 1],
                       marker="*", s=320, c="black", label="medoids", zorder=3)
            ax.legend(loc="best")
        ax.set_title(f"DTW k-medoids  (device={self.device}, {n} series)")
        ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        print(f"plot saved -> {png}")
        if show and "agg" not in matplotlib.get_backend().lower():
            plt.show()
        return png


def cluster(data, k, *, method="pam", band=-1, device=None, max_iter=100):
    """Cluster a dataset with DTW k-medoids, honoring the global/over-ridden device.

    ``data`` may be a :class:`Dataset`, a path, or an array. ``device=None`` uses
    the global default (see :func:`dtwcpp.device`). For ``"hpc"`` the work is
    offloaded to a SLURM cluster and the data is never read locally.
    """
    from dtwcpp import get_device, _resolve_device
    data = load(data)
    eff = device if device is not None else get_device()
    backend, _ = _resolve_device(eff)

    t0 = time.perf_counter()
    if backend == "hpc":
        from dtwcpp import _hpc
        source = data.source if data.is_path else data.as_series()
        labels = _hpc.cluster_on_hpc(source, k, method=method, band=band,
                                     skip_cols=data.skip_cols, name=f"dtwc_{data.name}")
        return ClusterResult(labels, device="hpc", elapsed_s=time.perf_counter() - t0,
                             k=k, n_series=len(labels), name=data.name)

    # Local cpu / gpu: full distance matrix on the device, then FastPAM.
    from dtwcpp import compute_distance_matrix, Problem, fast_pam
    series = data.as_series()
    names = [str(i) for i in range(len(series))]
    D = compute_distance_matrix(series, band=band, device=eff)
    prob = Problem(data.name)
    prob.set_data(series, names)
    prob.set_distance_matrix_from_numpy(D)
    res = fast_pam(prob, k, max_iter)
    return ClusterResult(res.labels, device=("cuda" if backend == "cuda" else "cpu"),
                         elapsed_s=time.perf_counter() - t0, k=k, n_series=len(series),
                         medoid_indices=res.medoid_indices, distance_matrix=D,
                         cost=res.total_cost, name=data.name)


def plot(result, **kwargs):
    """Top-level alias for ``result.plot(...)``."""
    return result.plot(**kwargs)
