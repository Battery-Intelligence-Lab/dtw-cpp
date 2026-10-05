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


class _NotSeries(TypeError, ValueError):
    """Input that is not real-valued series: a TypeError, and a ValueError, which
    scikit-learn's estimator checks require of DTWClustering for complex and 1-D
    input."""


_FORMS = ("series go in as a 2-D array (one series per row), a list of 1-D series "
          "(any lengths), a pandas DataFrame or an Arrow array")


def _float64(values):
    """``values`` as float64, refusing complex values: numpy casts a complex
    array to its real part with only a ComplexWarning. A list goes to numpy as
    it is, which refuses a complex number in it and reads a None as NaN."""
    if not isinstance(values, (list, tuple)):
        values = np.asarray(values)  # through __array__: no copy, no array function
        if values.dtype.kind == "c":
            raise _NotSeries("Complex data not supported: " + _FORMS + " of real numbers.")
    return np.asarray(values, dtype=np.float64)


def _series(source):
    """Already-read series as C++ takes them: one 1-D float64 array per series,
    and their names. The one conversion behind cluster(), load(),
    Problem.set_data, compute_distance_matrix and DTWClustering.

    A 2-D array holds one series per row and a list or tuple one per element, of
    any lengths; both are named by their ordinals. A pandas DataFrame holds one
    series per row, named by its index (read through ``to_numpy``: pandas is not
    imported), and an Arrow array or stream (pyarrow, polars, DuckDB) is read by
    the compiled-in nanoarrow and named as it names them. Complex values, an
    array of another dimension and an empty series are refused; NaN and inf are
    C++'s to judge, by the missing-data strategy.
    """
    from dtwcpp import InvalidInput
    from dtwcpp._dtwcpp_core import data_from_arrow_c_array
    names = None
    if type(source).__module__.split(".")[0] == "pandas" and hasattr(source, "columns"):
        names = [str(label) for label in source.index]
        source = source.to_numpy()
    elif hasattr(source, "__arrow_c_array__") or hasattr(source, "__arrow_c_stream__"):
        data = data_from_arrow_c_array(source)
        return [np.asarray(row, dtype=np.float64) for row in data.p_vec], list(data.p_names)
    elif type(source).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse input is not supported; provide a dense array.")
    if isinstance(source, (list, tuple)):
        rows = [_float64(row) for row in source]
        for i, row in enumerate(rows):
            if row.ndim != 1:
                raise _NotSeries(f"{_FORMS}; got a list whose element {i} is {row.ndim}-D.")
            if row.size == 0:
                raise InvalidInput(f"series {i} is empty; every series needs at least one value.")
    else:
        array = np.asarray(source)
        if array.ndim != 2:  # "Reshape your data", as scikit-learn's checks expect
            raise _NotSeries(f"{_FORMS}; got a {array.ndim}-D array of shape {array.shape}. "
                             "Reshape your data to (n_series, n_timesteps).")
        array = _float64(array)
        if array.shape[0] and not array.shape[1]:  # scikit-learn's words, which its checks match
            raise InvalidInput(f"every series needs at least one value: 0 feature(s) "
                               f"(shape={array.shape}) while a minimum of 1 is required.")
        rows = list(array)
    return rows, names or [str(i) for i in range(len(rows))]


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
        from dtwcpp import _dtwcpp_core
        self.source = source
        self.skip_cols = int(skip_cols)
        self.skip_rows = int(skip_rows)
        self.delimiter = delimiter
        # dtwc_cl's rule, so a run of "data/" is named "data" in every language.
        self.name = name if name is not None else _dtwcpp_core._default_name(
            os.fspath(source) if self.is_path else "")
        self._data = None

    @property
    def is_path(self):
        return isinstance(self.source, (str, os.PathLike))

    def _as_data(self):
        """Read the source once, caching the owning C++ ``dtwc::Data``.

        A path is read as dtwc_cl reads it: text by the C++ reader
        (``skip_cols`` drops leading FIELDS before numeric parsing, ``skip_rows``
        leading LINES, variable-length rows are kept, a batch file names its
        series 1, 2, ... and a folder by file stem), Parquet and Arrow IPC
        through the installed pyarrow. ``skip_rows`` drops leading SERIES of an
        in-memory source — one memory row is one file line — named as
        :func:`_series` names them.
        """
        if self._data is None:
            from dtwcpp import _dtwcpp_core, io
            if self.is_path:
                self._data = io._read_data(self.source, self.skip_cols,
                                           self.skip_rows, self.delimiter)
            else:
                from dtwcpp import InvalidInput
                rows, names = _series(self.source)
                rows, names = rows[self.skip_rows:], names[self.skip_rows:]
                if self.skip_cols > 0:
                    for row in rows:
                        if self.skip_cols > len(row):
                            raise InvalidInput(
                                "load: skip_cols exceeds an in-memory series "
                                "length.")
                    rows = [row[self.skip_cols:] for row in rows]
                self._data = _dtwcpp_core.Data([row.tolist() for row in rows], names)
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
                 medoid_indices=None, cost=None, name="dataset", problem=None):
        self.labels = np.asarray(labels)
        self.device = device
        self.elapsed_s = elapsed_s
        self.k = k
        self.n_series = n_series
        self.medoids = None if medoid_indices is None else np.asarray(medoid_indices)
        self.cost = cost
        self.name = name
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

    def score(self, name):
        """The clustering-quality score ``name`` names, computed by C++.

        ``"silhouette"`` (the MEAN silhouette), ``"davies_bouldin"``,
        ``"dunn"``, ``"calinski_harabasz"`` or ``"inertia"``; any other name
        raises :class:`dtwcpp.InvalidInput`.
        """
        from dtwcpp import _dtwcpp_core
        return _dtwcpp_core.score(self._clustered(), name)

    def save(self, directory):
        """Write the four result CSVs into ``directory`` with the C++ writer
        ``dtwc_cl`` and C++ ``Result::save`` use: ``<name>_labels.csv``
        (``name,cluster``), ``<name>_medoids.csv``
        (``cluster,medoid_index,medoid_name``), ``<name>_distance_matrix.csv``
        and ``<name>_silhouettes.csv`` (``name,cluster,silhouette``).

        With one cluster the silhouette is undefined and no silhouettes file is
        written; a partition with fewer than two realised clusters prints
        ``Warning: silhouettes skipped: ...`` on ``stderr``. An ``hpc`` result
        holds labels only: the remote ``dtwc_cl`` wrote its files.
        """
        from dtwcpp import InvalidInput, _dtwcpp_core
        if self._problem is None:
            raise InvalidInput(
                "save: this hpc result holds the labels only; dtwc_cl wrote the "
                "result files on the cluster.")
        _dtwcpp_core._write_result_files(self._problem, os.fspath(directory))
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


def _set_key(config, key, value, name=None):
    """Hand C++ one Config key. A value must be of the field's kind: the
    binding's casters would read True or "3" as an integer and truncate a NumPy
    float; C++ then checks the value itself."""
    from dtwcpp import InvalidInput, _dtwcpp_core
    if key.startswith("_") or not hasattr(_dtwcpp_core.Config, key):
        valid = sorted(n for n in dir(_dtwcpp_core.Config)
                       if not n.startswith("_") and n != "n_clusters")  # k names it
        raise InvalidInput(f"cluster: unknown key '{key}'. Valid keys: "
                           + ", ".join(valid) + ".")
    current = getattr(config, key)
    flag = isinstance(value, (bool, np.bool_))
    if isinstance(current, bool):
        kind, ok, value = "a bool", flag, bool(value) if flag else value
    elif isinstance(current, int):
        ok = not flag and isinstance(value, (int, np.integer))
        kind, value = "an integer", int(value) if ok else value
    elif isinstance(current, float):
        ok = not flag and isinstance(value, (int, float, np.integer, np.floating))
        kind, value = "a number", float(value) if ok else value
    else:
        kind, ok = "a string", isinstance(value, str)
    if not ok:
        raise TypeError(f"{name or key} must be {kind}, got {type(value).__name__}")
    setattr(config, key, value)


def _config(k, keys):
    """A C++ Config from cluster()'s keywords: C++ reads and checks each value."""
    from dtwcpp import InvalidInput, _dtwcpp_core
    if "n_clusters" in keys:
        raise InvalidInput("cluster: k is the number of clusters; drop n_clusters.")
    config = _dtwcpp_core.Config()
    _set_key(config, "n_clusters", k, name="k")
    for key, value in keys.items():
        _set_key(config, key, value)
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
                  cost=result.total_cost, name=prob.name, problem=prob)


def plot(result, **kwargs):
    """Top-level alias for ``result.plot(...)``."""
    return result.plot(**kwargs)
