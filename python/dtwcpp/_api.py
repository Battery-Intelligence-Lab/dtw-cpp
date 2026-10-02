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

    A local cluster() is C++'s: the keywords become a dtwcpp Config, which C++
    applies to a Problem holding the series, and Problem.cluster() runs the
    method. Python reads and writes the files (dtwcpp.io).
@author Volkan Kumtepeli
"""
import os
import time

import numpy as np


def _float_rows(source):
    """In-memory source -> ``list[list[float]]``, variable lengths preserved.

    C++ ``dtwc::load(series_type)`` takes a ``vector<vector<double>>``, so the
    rows need not be the same length. A rectangular source keeps the NumPy fast
    path; a ragged one (which ``np.asarray(..., dtype=float)`` rejects) is
    converted row by row.
    """
    if isinstance(source, np.ndarray):
        return [list(row) for row in np.asarray(source, dtype=float)]
    try:
        rectangular = np.asarray(source, dtype=float)
    except (ValueError, TypeError):
        return [[float(value) for value in row] for row in source]
    return [list(row) for row in rectangular]


class Dataset:
    """A lazy handle to time-series data — a local array or a file path.

    The data is only materialized (read into memory) when a *local* backend
    needs it. On ``device="hpc"`` a path is passed straight to the cluster and
    never read locally, so it scales beyond what fits on the calling machine.
    """

    def __init__(self, source, *, skip_cols=0, skip_rows=0, delimiter=None,
                 name=None):
        from dtwcpp import InvalidInput
        # Checked where the handle is made, as C++ dtwc::load does, so a bad
        # count fails before any file is read or job submitted.
        for key, value in (("skip_cols", skip_cols), ("skip_rows", skip_rows)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"load: {key} must be an integer")
            if value < 0:
                raise InvalidInput(f"load: {key} must be non-negative.")
        self.source = source
        self.skip_cols = int(skip_cols)
        self.skip_rows = int(skip_rows)
        self.delimiter = delimiter
        if name is not None:
            self.name = name
        elif self.is_path:
            self.name = os.path.splitext(os.path.basename(str(source)))[0]
        else:
            self.name = "dataset"
        self._data = None

    @property
    def is_path(self):
        return isinstance(self.source, (str, os.PathLike))

    def _as_data(self):
        """Read the source once, caching the owning C++ ``dtwc::Data``.

        A path is read by :mod:`dtwcpp.io` as dtwc_cl reads it: text with the
        C++ reader's rules (``skip_cols`` drops leading FIELDS before numeric
        parsing, ``skip_rows`` leading LINES, variable-length rows are kept,
        a batch file names its series 1, 2, ... and a folder by file stem),
        Parquet and Arrow IPC through the installed pyarrow. ``skip_rows``
        drops leading SERIES of an in-memory source — one memory row is one
        file line — which is named by its ordinal, as
        ``Dataset::materialize_local`` names it (api.cpp).
        """
        if self._data is None:
            from dtwcpp import _dtwcpp_core, io
            if self.is_path:
                self._data = io._read_data(self.source, self.skip_cols,
                                           self.skip_rows, self.delimiter)
            else:
                from dtwcpp import InvalidInput
                rows = _float_rows(self.source)[self.skip_rows:]
                if self.skip_cols > 0:
                    for row in rows:
                        if self.skip_cols > len(row):
                            raise InvalidInput(
                                "load: skip_cols exceeds an in-memory series "
                                "length.")
                    rows = [row[self.skip_cols:] for row in rows]
                self._data = _dtwcpp_core.Data(
                    rows, [str(i) for i in range(len(rows))])
        return self._data

    def as_data(self):
        """The owning C++ :class:`dtwcpp.Data` (reads the file if a path). Cached."""
        return self._as_data()

    def as_series(self):
        """A fresh ``list`` view of the series — built from :meth:`as_data` on demand.

        Not cached: the C++ ``Data`` is the retained representation, so asking
        for Python floats is an explicit, transient request.
        """
        return self._as_data().p_vec

    def series_names(self):
        """The loader's name for each series — what C++ ``series_name(i)`` returns."""
        return self._as_data().p_names


def load(source, *, skip_cols=0, skip_rows=0, delimiter=None, name=None):
    """Wrap a path or array in a lazy :class:`Dataset` handle (does not read it)."""
    if isinstance(source, Dataset):
        return source
    return Dataset(source, skip_cols=skip_cols, skip_rows=skip_rows,
                   delimiter=delimiter, name=name)


class Result:
    """Outcome of :func:`cluster` — the canonical 2.0 ``Result`` (api-contract §1.4).

    ``labels``, ``medoids`` and ``cost`` are values; ``score(name)``,
    ``save(dir)`` and ``distance_matrix`` read the clustered ``Problem`` the
    result keeps, which fills its matrix on demand after a matrix-free
    OneBatchPAM/CLARA/TADPole run. An ``hpc`` run returns labels only.
    """

    def __init__(self, labels, *, device, elapsed_s, k, n_series,
                 medoid_indices=None, cost=None, name="dataset",
                 series_names=None, problem=None):
        self.labels = np.asarray(labels)
        self.device = device
        self.elapsed_s = elapsed_s
        self.k = k
        self.n_series = n_series
        self.medoids = None if medoid_indices is None else np.asarray(medoid_indices)
        self.cost = cost
        self.name = name
        self._series_names = series_names
        self._problem = problem
        self._distance_matrix = None

    def _clustered(self):
        """The Problem this result was clustered on; an hpc run has none."""
        if self._problem is None:
            import dtwcpp
            raise dtwcpp.InvalidInput(
                "no local distance matrix available to score (an hpc run "
                "returns labels only; score()/save(silhouettes) need "
                "cpu/gpu output).")
        return self._problem

    @property
    def distance_matrix(self):
        """Dense N x N distances, filled on demand — C++ ``Result::distance_matrix()``.

        A matrix-free ``onebatch``/``clara``/``tadpole`` run leaves the matrix
        unmaterialised, so the O(N) schedule survives until this property is
        read; reading it is an explicit request for the full N x N and fills
        the retained ``Problem``, exactly as ``score()``/``save()`` do. An
        ``hpc`` run has no local ``Problem`` and stays ``None``.
        """
        if self._distance_matrix is None and self._problem is not None:
            self._distance_matrix = self._problem.distance_matrix()
        return self._distance_matrix

    def summary(self):
        extra = f", cost={self.cost:.2f}" if self.cost is not None else ""
        return (f"[device={self.device}] {self.n_series} series, k={self.k}  ->  "
                f"{self.elapsed_s * 1e3:7.1f} ms{extra}")

    def _names(self, n=None):
        """Series names for output — the loader's, as C++ ``series_name(i)`` is."""
        if self._series_names is not None:
            return list(self._series_names)
        return [str(i) for i in range(len(self.labels) if n is None else n)]

    def score(self, name):
        """The clustering-quality score ``name`` names, computed by C++.

        ``"silhouette"`` (the MEAN silhouette), ``"davies_bouldin"``,
        ``"dunn"``, ``"calinski_harabasz"`` or ``"inertia"``; any other name
        raises :class:`dtwcpp.InvalidInput`.
        """
        from dtwcpp import _dtwcpp_core
        return _dtwcpp_core.score(self._clustered(), name)

    def save(self, directory):
        """Write the four result CSVs into ``directory``, as dtwc_cl writes them.

        ``<name>_labels.csv`` (``name,cluster``), ``<name>_medoids.csv``
        (``cluster,medoid_index,medoid_name``), ``<name>_distance_matrix.csv``
        and ``<name>_silhouettes.csv`` (``name,cluster,silhouette``), byte for
        byte the CLI's files (``dtwcpp.io`` holds the formats). An ``hpc``
        result writes the labels and an empty medoid table only.

        With one cluster the silhouette is undefined and no silhouettes file is
        written, silently, as C++ ``Result::save`` and the CLI do. A partition
        with fewer than two realised clusters prints ``Warning: silhouettes
        skipped: ...`` on ``stderr`` instead of failing a clustering that
        succeeded. ``score("silhouette")`` still raises ``UndefinedScore``.
        """
        import sys

        import dtwcpp
        from dtwcpp import io
        # As C++ Result::save: the matrix is filled before any file is written.
        matrix = self.distance_matrix
        names = self._names()
        base = os.path.join(directory, self.name)
        io._write_text(base + "_labels.csv", "name,cluster\n" + "".join(
            f"{nm},{int(lab)}\n" for nm, lab in zip(names, self.labels)))
        medoids = [] if self.medoids is None else self.medoids
        io._write_text(base + "_medoids.csv", "cluster,medoid_index,medoid_name\n" + "".join(
            f"{c},{int(m)},{names[int(m)]}\n" for c, m in enumerate(medoids)))
        if matrix is None:  # an hpc result: labels only
            return directory
        io._write_matrix_csv(matrix, base + "_distance_matrix.csv")
        if len(medoids) < 2:
            return directory
        try:
            silhouettes = dtwcpp.silhouette(self._problem)
        except dtwcpp.UndefinedScore as e:
            print(f"Warning: silhouettes skipped: {e}", file=sys.stderr)
            return directory
        io._write_text(base + "_silhouettes.csv", "name,cluster,silhouette\n" + "".join(
            f"{nm},{int(lab)},{s:.8g}\n" for nm, lab, s in zip(names, self.labels, silhouettes)))
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


# The keywords the SLURM transport carries to the remote dtwc_cl.
_HPC_KEYS = frozenset({"method", "band", "max_iter", "n_init", "seed", "variant",
                       "wdtw_g", "adtw_penalty", "msm_c", "twe_nu", "twe_lambda",
                       "mv_mode", "missing_strategy", "metric", "name"})


def _config(k, keys):
    """A C++ Config from cluster()'s keywords: C++ reads and checks each value."""
    from dtwcpp import InvalidInput, _dtwcpp_core
    config = _dtwcpp_core.Config()
    config.n_clusters = k
    for key, value in keys.items():
        if key.startswith("_") or not hasattr(_dtwcpp_core.Config, key):
            valid = sorted(name for name in dir(_dtwcpp_core.Config) if not name.startswith("_"))
            raise InvalidInput(f"cluster: unknown key '{key}'. Valid keys: device, "
                               + ", ".join(valid) + ".")
        setattr(config, key, value)
    return config


def cluster(data, k, **keys):
    """Cluster a dataset into ``k`` clusters with DTW, on the global or given device.

    ``data`` may be a :class:`Dataset`, a path, or an array. ``keys`` are the
    keys of a dtwc_cl run — its long names in snake_case: ``method``, ``band``,
    ``metric``, ``variant`` and its parameters (``wdtw_g``, ``adtw_penalty``,
    ``sdtw_gamma``, ``msm_c``, ``twe_nu``, ``twe_lambda``), ``mv_mode``,
    ``missing_strategy``, ``max_iter``, ``n_init``, ``seed``, ``sample_size``,
    ``n_samples``, ``batch_size``, ``linkage``, ``dc``, ``solver`` and the MIP
    settings, ``gpu_precision``, ``name``, ``verbose`` — read and checked by
    C++; an unknown key raises :class:`dtwcpp.InvalidInput`. ``method`` is
    ``"auto"`` unless given: PAM on a GPU and for up to 5,000 series on the
    CPU, CLARA above. ``device=None`` uses the global default (see
    :func:`dtwcpp.device`); ``"hpc"`` offloads the run to a SLURM cluster and
    never reads the data locally.
    """
    import dtwcpp
    from dtwcpp import InvalidInput, _dtwcpp_core, _hpc_remote_device, _resolve_device
    device = keys.pop("device", None)
    config = _config(k, keys)
    eff = device if device is not None else dtwcpp.device()
    backend, _ = _resolve_device(eff)
    data = load(data)

    t0 = time.perf_counter()
    if backend == "hpc":
        from dtwcpp import _hpc
        # The SLURM wrapper takes a fixed positional argument list with no
        # skip_rows slot, so the remote CLI cannot receive it. Refuse loudly
        # instead of clustering the header rows the caller asked to drop.
        if data.skip_rows:
            raise InvalidInput(
                "cluster: skip_rows is not carried by the HPC transport; strip "
                "the header rows before staging, or use device='cpu'/'gpu'."
            )
        # Nor is delimiter: the remote reader takes it from the file extension.
        if data.is_path and data.delimiter:
            raise InvalidInput(
                "cluster: delimiter is not carried by the HPC transport; the "
                "remote reader takes it from the file extension (.csv comma, "
                ".tsv/.txt tab). Drop delimiter= for a file whose extension "
                "matches, or use device='cpu'/'gpu'."
            )
        dropped = sorted(set(keys) - _HPC_KEYS)
        if dropped:
            raise InvalidInput(
                f"cluster: {', '.join(dropped)} is not carried by the HPC "
                "transport; drop it, or use device='cpu'/'gpu'.")
        options = {key: getattr(config, key) for key in keys if key != "name"}
        options["method"] = config.method
        source = data.source if data.is_path else data.as_series()
        # as_series() has already dropped an in-memory source's skip_cols.
        labels = _hpc.cluster_on_hpc(source, config.n_clusters,
                                     device=_hpc_remote_device(eff),
                                     skip_cols=data.skip_cols if data.is_path else 0,
                                     name=f"dtwc_{config.name or data.name}",
                                     **options)
        return Result(labels, device="hpc", elapsed_s=time.perf_counter() - t0,
                      k=k, n_series=len(labels), name=config.name or data.name)

    # As dtwc::run: the settings reach the Problem, and are checked, before
    # the series are read.
    config.device = eff
    prob = _dtwcpp_core.Problem(config.name or data.name)
    _dtwcpp_core.apply(config, prob)
    prob.set_data(data.as_data())
    result = prob.cluster()
    return Result(result.labels, device=backend,
                  elapsed_s=time.perf_counter() - t0, k=config.n_clusters,
                  n_series=prob.size, medoid_indices=result.medoid_indices,
                  cost=result.total_cost, name=prob.name,
                  series_names=data.series_names(), problem=prob)


def plot(result, **kwargs):
    """Top-level alias for ``result.plot(...)``."""
    return result.plot(**kwargs)
