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


# score() names accepted by Result.score (api-contract-2.0.md §1.4). Each resolves
# to the Tier-2 scores::* function; "silhouette" returns the MEAN silhouette.
_SCORE_NAMES = ("silhouette", "davies_bouldin", "dunn", "calinski_harabasz", "inertia")


class Result:
    """Outcome of :func:`cluster` — the canonical 2.0 ``Result`` (api-contract §1.4).

    Members: ``labels``, ``medoids``, ``score(name)``, ``save(dir)``, ``plot()``,
    plus ``cost`` and ``device``. ``distance_matrix`` is ``None`` for the
    matrix-free OneBatchPAM, CLARA, and TADPole methods. ``ClusterResult`` is a
    deprecated alias name.
    """

    def __init__(self, labels, *, device, elapsed_s, k, n_series,
                 medoid_indices=None, distance_matrix=None, cost=None, name="dataset",
                 series_names=None):
        self.labels = np.asarray(labels)
        self.device = device
        self.elapsed_s = elapsed_s
        self.k = k
        self.n_series = n_series
        self.medoids = None if medoid_indices is None else np.asarray(medoid_indices)
        self.distance_matrix = distance_matrix
        self.cost = cost
        self.name = name
        self._series_names = series_names

    @property
    def medoid_indices(self):
        """Deprecated alias for :attr:`medoids` (kept one cycle, api-contract §4)."""
        import warnings
        warnings.warn("Result.medoid_indices is deprecated; use Result.medoids",
                      DeprecationWarning, stacklevel=2)
        return self.medoids

    def summary(self):
        extra = f", cost={self.cost:.2f}" if self.cost is not None else ""
        return (f"[device={self.device}] {self.n_series} series, k={self.k}  ->  "
                f"{self.elapsed_s * 1e3:7.1f} ms{extra}")

    def _scoring_problem(self):
        """Rebuild a Tier-2 Problem carrying this result's distance matrix + labels.

        Scores read state from a Problem (its distance matrix + clusters_ind +
        centroids_ind), so score()/save() reconstruct a minimal one. Dummy series
        stand in — the scores use only the distance matrix and the labels.
        """
        import dtwcpp
        if self.distance_matrix is None:
            raise dtwcpp.InvalidInput(
                "no local distance matrix available to score (an hpc run returns "
                "labels only; score()/save(silhouettes) need cpu/gpu output)."
            )
        D = np.asarray(self.distance_matrix, dtype=float)
        n = D.shape[0]
        names = list(self._series_names) if self._series_names is not None \
            else [str(i) for i in range(n)]
        prob = dtwcpp.Problem(self.name)
        prob.set_data([[0.0] for _ in range(n)], names)
        prob.set_distance_matrix(D)
        prob.set_n_clusters(int(self.k))
        prob.clusters_ind = [int(x) for x in self.labels]
        if self.medoids is not None:
            prob.centroids_ind = [int(x) for x in self.medoids]
        return prob

    def score(self, name):
        """Return a clustering-quality score by name (api-contract §1.4).

        ``name`` is one of ``"silhouette"`` (the MEAN silhouette),
        ``"davies_bouldin"``, ``"dunn"``, ``"calinski_harabasz"``, ``"inertia"``.
        An unknown name raises :class:`dtwcpp.InvalidInput`.
        """
        import dtwcpp
        key = str(name).strip().lower()
        if key not in _SCORE_NAMES:
            raise dtwcpp.InvalidInput(
                f"unknown score {name!r}. Valid names: {', '.join(_SCORE_NAMES)}."
            )
        prob = self._scoring_problem()
        if key == "silhouette":
            return float(np.mean(dtwcpp.silhouette(prob)))
        if key == "davies_bouldin":
            return float(dtwcpp.davies_bouldin(prob))
        if key == "dunn":
            return float(dtwcpp.dunn(prob))
        if key == "calinski_harabasz":
            return float(dtwcpp.calinski_harabasz(prob))
        return float(dtwcpp.inertia(prob))

    def save(self, directory):
        """Write the four human-readable result CSVs into ``directory`` (§1.4/§7).

        Emits ``<name>_labels.csv`` (``name,cluster``), ``<name>_medoids.csv``
        (``cluster,medoid_index,medoid_name``), ``<name>_distance_matrix.csv`` and
        ``<name>_silhouettes.csv`` (``name,cluster,silhouette``). Byte-identity
        with the CLI output is the Phase 2.4 conformance fixture's contract.
        """
        import dtwcpp
        os.makedirs(directory, exist_ok=True)
        n = len(self.labels)
        names = list(self._series_names) if self._series_names is not None \
            else [str(i) for i in range(n)]
        base = os.path.join(directory, self.name)

        with open(base + "_labels.csv", "w", newline="") as f:
            f.write("name,cluster\n")
            for nm, lab in zip(names, self.labels):
                f.write(f"{nm},{int(lab)}\n")

        with open(base + "_medoids.csv", "w", newline="") as f:
            f.write("cluster,medoid_index,medoid_name\n")
            if self.medoids is not None:
                for c, m in enumerate(self.medoids):
                    f.write(f"{c},{int(m)},{names[int(m)]}\n")

        if self.distance_matrix is not None:
            np.savetxt(base + "_distance_matrix.csv",
                       np.asarray(self.distance_matrix, dtype=float), delimiter=",")
            prob = self._scoring_problem()
            sil = dtwcpp.silhouette(prob)
            with open(base + "_silhouettes.csv", "w", newline="") as f:
                f.write("name,cluster,silhouette\n")
                for nm, lab, s in zip(names, self.labels, sil):
                    f.write(f"{nm},{int(lab)},{s}\n")
        return directory

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
        if self.medoids is not None:
            ax.scatter(XY[self.medoids, 0], XY[self.medoids, 1],
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


# Deprecated alias name for the Tier-1 result (kept one cycle, api-contract §4).
ClusterResult = Result


# Canonical clustering methods. This is exactly the dtwc_cl CLI vocabulary
# (dtwc_cl.cpp: "auto, pam, clara, kmedoids, mip, hierarchical"), so the SAME
# name is valid on cpu/gpu (dispatched here) and on hpc (forwarded to
# dtwc_cl --method). "hclust" is the CLI's alias for "hierarchical".
_METHODS = ("auto", "pam", "onebatch", "clara", "kmedoids", "mip",
            "lrcore", "hierarchical", "tadpole")
_AUTO_PAM_SERIES_LIMIT = 5000


def _normalize_method(method):
    """Lower-case, alias-resolve, and validate a clustering method name.

    Raises ``ValueError`` on anything outside the documented set. This enforces
    the 2.0 no-silent-fallback rule: an unrecognised method must NEVER quietly
    run a different algorithm. Before Task 0.14 the local path ran FastPAM for
    *every* value of ``method`` (the argument was accepted but ignored).
    """
    m = str(method).strip().lower()
    if m == "hclust":
        m = "hierarchical"
    if m == "lr":
        m = "lrcore"
    if m == "obp":
        m = "onebatch"
    if m not in _METHODS:
        raise ValueError(
            f"unknown method: {method!r}. Expected one of: {', '.join(_METHODS)} "
            f"(aliases: 'obp', 'lr', 'hclust')."
        )
    return m


def _resolve_tier1_method(method, n, backend):
    """Resolve ``auto`` after the execution backend is known.

    HPC keeps ``auto`` because the remote CLI owns data materialisation and the
    final size-dependent decision. Explicit method requests are never changed.
    """
    if method != "auto" or backend == "hpc":
        return method
    if backend in ("cuda", "metal"):
        return "pam"
    return "pam" if n <= _AUTO_PAM_SERIES_LIMIT else "clara"


def _run_local_method(prob, method, k, max_iter):
    """Dispatch the local (cpu/gpu) clustering call for a validated ``method``.

    ``prob`` already has its distance matrix loaded. Returns
    ``(labels, medoid_indices, cost)``. Each documented method routes to its OWN
    algorithm — Task 0.14: before this, the local path ran FastPAM for EVERY
    method value. The kmedoids/mip/hierarchical branches mirror the verified CLI
    dispatch (dtwc_cl.cpp:874-926).
    """
    import dtwcpp

    if method == "pam":
        res = dtwcpp.fast_pam_seeded(
            prob, k, dtwcpp.DEFAULT_RANDOM_SEED, max_iter
        )
        return res.labels, res.medoid_indices, res.total_cost
    if method == "onebatch":
        res = dtwcpp.one_batch_pam(
            prob, k, max_iter=max_iter, seed=dtwcpp.DEFAULT_RANDOM_SEED
        )
        return res.labels, res.medoid_indices, res.total_cost
    if method == "clara":
        res = dtwcpp.fast_clara(
            prob, k, max_iter=max_iter, seed=dtwcpp.DEFAULT_RANDOM_SEED
        )
        return res.labels, res.medoid_indices, res.total_cost
    if method == "hierarchical":
        dend = dtwcpp.build_dendrogram(prob)
        res = dtwcpp.cut_dendrogram(dend, prob, k)
        return res.labels, res.medoid_indices, res.total_cost

    # kmedoids (Lloyd) and mip run through Problem.cluster() and read back state.
    prob.set_n_clusters(k)
    if method == "mip":
        prob.method = dtwcpp.Method.MIP
    elif method == "lrcore":
        prob.method = dtwcpp.Method.LRCore
    elif method == "tadpole":
        prob.method = dtwcpp.Method.TADPole
    else:
        # Tier-1 Lloyd owns an invocation-local default. Set it explicitly at
        # this boundary so the standard initializer cannot inherit whatever
        # state the deliberately mutable, unseeded Tier-2 engine last consumed.
        # This does not replace init_fun, so custom Tier-2 callbacks retain
        # their legacy invocation semantics.
        prob.set_random_seed(dtwcpp.DEFAULT_RANDOM_SEED)
        prob.method = dtwcpp.Method.Kmedoids
    prob.cluster()
    return prob.clusters_ind, prob.centroids_ind, prob.find_total_cost()


def cluster(data, k, *, method="pam", band=-1, device=None, max_iter=100):
    """Cluster a dataset with DTW, honoring the global/over-ridden device.

    ``data`` may be a :class:`Dataset`, a path, or an array. ``device=None`` uses
    the global default (see :func:`dtwcpp.device`). For ``"hpc"`` the work is
    offloaded to a SLURM cluster and the data is never read locally.

    ``method`` selects the clustering algorithm: one of ``"pam"`` (default),
    ``"onebatch"``, ``"clara"``, ``"kmedoids"``, ``"mip"``, ``"lrcore"``,
    ``"tadpole"``, ``"hierarchical"``, or ``"auto"``.
    Locally, ``"auto"`` selects PAM for GPU execution and for CPU datasets up
    to 5,000 series, otherwise CPU CLARA. HPC forwards ``"auto"`` so the remote
    process can resolve it after materialising the dataset.
    An unrecognised method raises ``ValueError`` — it is NEVER silently ignored.
    """
    from dtwcpp import get_device, _resolve_device
    data = load(data)
    method = _normalize_method(method)          # validate BEFORE any backend work
    eff = device if device is not None else get_device()
    backend, _ = _resolve_device(eff)

    t0 = time.perf_counter()
    if backend == "hpc":
        from dtwcpp import _hpc
        source = data.source if data.is_path else data.as_series()
        labels = _hpc.cluster_on_hpc(source, k, method=method, band=band,
                                     skip_cols=data.skip_cols, name=f"dtwc_{data.name}")
        return Result(labels, device="hpc", elapsed_s=time.perf_counter() - t0,
                      k=k, n_series=len(labels), name=data.name)

    # Matrix-free methods must not accidentally pay the N^2 cost in this Tier-1
    # wrapper. They currently execute on CPU; an explicit GPU request is rejected
    # loudly instead of silently defeating their scaling contract.
    from dtwcpp import compute_distance_matrix, Problem
    series = data.as_series()
    method = _resolve_tier1_method(method, len(series), backend)
    names = [str(i) for i in range(len(series))]
    prob = Problem(data.name)
    # Configure the band before set_data() refreshes/rebinds the Problem's DTW
    # callable.  Matrix-free methods query distances from Problem directly, so
    # relying only on compute_distance_matrix(..., band=...) silently made them
    # unbanded.
    prob.set_band(band)
    prob.set_data(series, names)
    # CLARA is matrix-free too: it computes only sample and assignment
    # distances. Treating it as a matrix method here defeated its O(Ns)
    # scaling contract by materialising N^2 distances before dispatch.
    matrix_free = method in ("onebatch", "clara", "tadpole")
    if matrix_free and backend in ("cuda", "metal"):
        from dtwcpp import DeviceError
        raise DeviceError(
            f"method='{method}' uses its own matrix-free CPU distance schedule; "
            "CUDA execution is not implemented for that schedule. Use device='cpu'."
        )
    D = None if matrix_free else compute_distance_matrix(
        series, band=band, device=eff)
    if D is not None:
        prob.set_distance_matrix(D)
    labels, medoid_indices, cost = _run_local_method(prob, method, k, max_iter)
    return Result(labels, device=backend,
                  elapsed_s=time.perf_counter() - t0, k=k, n_series=len(series),
                  medoid_indices=medoid_indices, distance_matrix=D,
                  cost=cost, name=data.name, series_names=names)


def plot(result, **kwargs):
    """Top-level alias for ``result.plot(...)``."""
    return result.plot(**kwargs)
