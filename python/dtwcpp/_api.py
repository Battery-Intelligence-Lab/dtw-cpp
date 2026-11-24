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

    A local cluster() is C++'s dtwc::run, the run dtwc_cl makes: the keywords
    become a dtwcpp Config, and the run reads a path itself (text, a folder or
    Arrow IPC, which its memory plan may stream) or takes series in memory
    (arrays, and Parquet, which pyarrow reads: dtwcpp.io). Result wraps the
    run's C++ dtwc::Result.
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


def _series(source, skip_rows=0):
    """Already-read series as C++ takes them: one 1-D float64 array per series,
    and their names. The one conversion behind cluster(), load(),
    Problem.set_data, compute_distance_matrix and DTWClustering.

    A 2-D array holds one series per row and a list or tuple one per element, of
    any lengths; both are named by their ordinals. The ``skip_rows`` leading series
    a :class:`Dataset` drops go first, unread, and the ordinals count from 0 after
    them, as C++ ``dtwc::load`` treats series in memory. A pandas DataFrame holds one
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
        return ([np.asarray(data.series(i), dtype=np.float64) for i in range(skip_rows, data.size)],
                list(data.p_names)[skip_rows:])
    elif type(source).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse input is not supported; provide a dense array.")
    if isinstance(source, (list, tuple)):
        rows = [_float64(row) for row in source[skip_rows:]]
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
        array = _float64(array[skip_rows:])
        if array.shape[0] and not array.shape[1]:  # scikit-learn's words, which its checks match
            raise InvalidInput(f"every series needs at least one value: 0 feature(s) "
                               f"(shape={array.shape}) while a minimum of 1 is required.")
        rows = list(array)
    return rows, names[skip_rows:] if names is not None else [str(i) for i in range(len(rows))]


def _to_data(source, *, skip_rows=0, skip_cols=0, names=None, ndim=1, dtype="float64"):
    """Construct a C++ :class:`dtwcpp.Data` directly from in-memory series without
    intermediate Python list-of-floats boxing."""
    from dtwcpp import InvalidInput, _dtwcpp_core

    if isinstance(ndim, (bool, np.bool_)) or not isinstance(ndim, (int, np.integer)):
        raise TypeError(f"ndim must be an integer, got {type(ndim).__name__}")
    if ndim < 1:
        raise InvalidInput("ndim must be at least 1.")
    ndim = int(ndim)
    dtype = _dtwcpp_core._dtype(dtype)

    rows = None
    df_names = None
    if isinstance(source, _dtwcpp_core.Data):
        if (skip_rows == 0 and skip_cols == 0 and names is None
                and source.ndim == ndim and source.is_f32() == (dtype == "float32")):
            return source
        names = None if names is None else list(names)  # a bad names fails before the series are copied
        rows = [np.asarray(source.series(i), dtype=np.float64) for i in range(skip_rows, source.size)]
        default_names = list(source.p_names)[skip_rows:]
    elif type(source).__module__.split(".")[0] == "pandas" and hasattr(source, "columns"):
        df_names = [str(label) for label in source.index]
        source = source.to_numpy()
    elif hasattr(source, "__arrow_c_array__") or hasattr(source, "__arrow_c_stream__"):
        if skip_rows == 0 and skip_cols == 0 and names is None and ndim == 1:
            return _dtwcpp_core.data_from_arrow_c_array(source, dtype)
        rows, default_names = _series(source, skip_rows)
    elif type(source).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse input is not supported; provide a dense array.")
    elif isinstance(source, (list, tuple)):
        rows, default_names = _series(source, skip_rows)

    if rows is not None:
        if skip_cols > 0:
            for row in rows:
                if skip_cols > len(row):
                    raise InvalidInput("load: skip_cols exceeds an in-memory series length.")
            rows = [np.ascontiguousarray(row[skip_cols:], dtype=np.float64) for row in rows]
        else:
            rows = [np.ascontiguousarray(row, dtype=np.float64) for row in rows]
        eff_names = list(names) if names is not None else default_names
        return _dtwcpp_core.Data.from_numpy_list(rows, eff_names, ndim, dtype)

    array = np.asarray(source)
    if array.dtype.kind == "c":
        raise _NotSeries("Complex data not supported: " + _FORMS + " of real numbers.")
    if ndim > 1 and array.ndim == 3:
        if array.shape[2] != ndim:
            raise InvalidInput(
                f"3-D multivariate array shape[2] ({array.shape[2]}) must match ndim ({ndim})."
            )
    elif array.ndim != 2:
        raise _NotSeries(
            f"{_FORMS}; got a {array.ndim}-D array of shape {array.shape}. "
            "Reshape your data to (n_series, n_timesteps)."
        )

    sliced = array[skip_rows:]
    if not (sliced.ndim == 2 and sliced.dtype == np.float32):
        sliced = _float64(sliced)
    if sliced.shape[0] and not sliced.shape[1]:
        raise InvalidInput(
            f"every series needs at least one value: 0 feature(s) "
            f"(shape={sliced.shape}) while a minimum of 1 is required."
        )
    if skip_cols > 0:
        if sliced.shape[0] and skip_cols > sliced.shape[1]:
            raise InvalidInput("load: skip_cols exceeds an in-memory series length.")
        sliced = sliced[:, skip_cols:]
    arr_c = np.ascontiguousarray(sliced)
    eff_names = list(names) if names is not None else (
        df_names[skip_rows:] if df_names is not None else [str(i) for i in range(arr_c.shape[0])]
    )
    if arr_c.ndim == 3:
        return _dtwcpp_core.Data.from_numpy_3d(arr_c, eff_names, dtype)
    if arr_c.dtype == np.float32:
        return _dtwcpp_core.Data.from_numpy_2d_f32(arr_c, eff_names, ndim, dtype)
    return _dtwcpp_core.Data.from_numpy_2d(arr_c, eff_names, ndim, dtype)


class Dataset:
    """A lazy handle to time-series data — a local array or a file path.

    The data is only materialized (read into memory) when a *local* backend
    needs it. On ``device="hpc"`` a path is passed straight to the cluster and
    never read locally, so it scales beyond what fits on the calling machine.
    """

    def __init__(self, source, *, skip_cols=0, skip_rows=0, delimiter=None,
                 name=None, dtype="float64"):
        from dtwcpp import InvalidInput
        # Checked where the handle is made, as C++ dtwc::load does, so a bad
        # count fails before any file is read or job submitted.
        for key, value in (("skip_cols", skip_cols), ("skip_rows", skip_rows)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"load: {key} must be an integer")
            if value < 0:
                raise InvalidInput(f"load: {key} must be non-negative.")
        if not isinstance(dtype, str):
            raise TypeError(f"load: dtype must be a string such as 'float32', got {type(dtype).__name__}")
        from dtwcpp import _dtwcpp_core
        self.source = source
        self.skip_cols = int(skip_cols)
        self.skip_rows = int(skip_rows)
        self.delimiter = delimiter
        # The precision the series are stored in, by the C++ table --dtype reads:
        # "float32" halves their memory; float32 input alone does not choose it.
        self.dtype = _dtwcpp_core._dtype(dtype)
        # dtwc_cl's rule, so a run of "data/" is named "data" in every language.
        self.name = name if name is not None else _dtwcpp_core._default_name(
            os.fspath(source) if self.is_path else "")
        self._ndim = source.ndim if isinstance(source, _dtwcpp_core.Data) else 1  # a Data carries its own
        self._data = None

    @property
    def is_path(self):
        return isinstance(self.source, (str, os.PathLike))

    def as_data(self):
        """The C++ :class:`dtwcpp.Data` of the source, read once and cached.

        A path is read as dtwc_cl reads it: text by the C++ reader
        (``skip_cols`` drops leading FIELDS before numeric parsing, ``skip_rows``
        leading LINES, variable-length rows are kept, a batch file names its
        series 1, 2, ... and a folder by file stem), Arrow IPC by the C++ reader
        too, in place (the series are views of the file, which the Data keeps
        open), and Parquet (``skip_cols`` and ``skip_rows`` drop leading columns
        and rows) through the installed pyarrow. ``skip_rows`` drops leading SERIES of an
        in-memory source — one memory row is one file line — and the series left
        are named as :func:`_series` names them: ordinals count from 0, as in C++.
        The series are stored in ``dtype``, each value rounded once to it.
        """
        if self._data is None:
            from dtwcpp import io
            if self.is_path:
                data = io._read_data(self.source, self.skip_cols,
                                     self.skip_rows, self.delimiter, self.dtype)
                if self._ndim > 1 and data.ndim not in (1, self._ndim):  # an Arrow IPC file's metadata ndim
                    from dtwcpp import InvalidInput
                    raise InvalidInput(f"load: the Arrow IPC file's metadata gives ndim={data.ndim} and "
                                       f"ndim={self._ndim} was given; pass ndim={data.ndim}, or leave ndim out.")
                if self._ndim > 1 and data.ndim == 1:
                    data.ndim = self._ndim
                    data.validate_ndim()
                self._data = data
            else:
                self._data = _to_data(
                    self.source,
                    skip_rows=self.skip_rows,
                    skip_cols=self.skip_cols,
                    ndim=self._ndim,
                    dtype=self.dtype,
                )
        return self._data

    def as_series(self):
        """A fresh ``list`` view of the series — built from :meth:`as_data` on demand.

        Not cached: the C++ ``Data`` is the retained representation, so asking
        for Python floats is an explicit, transient request.
        """
        data = self.as_data()
        return [data.series(i) for i in range(data.size)]

    def series_names(self):
        """The loader's name for each series — what C++ ``series_name(i)`` returns."""
        return self.as_data().p_names


def load(source, *, skip_cols=0, skip_rows=0, delimiter=None, name=None, dtype="float64"):
    """Wrap a path or array in a lazy :class:`Dataset` handle (does not read it).
    ``dtype="float32"`` stores the series as float32, half the memory: a file's
    reader rounds each value, never holding a float64 copy (float32 input data
    alone does not choose it). A Dataset passes through with the options it was
    made with, as :func:`load_data` takes it: a keyword other than its default
    that differs from the Dataset's raises :class:`dtwcpp.InvalidInput` naming
    both values, one equal to it runs, and ``name`` is refused. ``load`` cannot
    tell a keyword at its default from one not given; ``load_data`` and
    :func:`cluster` can."""
    if isinstance(source, Dataset):
        return load_data(source, **{key: value for key, value, default in (
            ("skip_cols", skip_cols, 0), ("skip_rows", skip_rows, 0),
            ("delimiter", delimiter, None), ("name", name, None),
            ("dtype", dtype, "float64")) if value != default})
    return Dataset(source, skip_cols=skip_cols, skip_rows=skip_rows,
                   delimiter=delimiter, name=name, dtype=dtype)


class Result:
    """Outcome of :func:`cluster` — the canonical 2.0 ``Result``.

    ``labels``, ``medoids``, ``cost`` and ``report`` (the run report's text,
    ``""`` under ``no_report=True``) are values; ``score(name)``, ``save(dir)``
    and ``distance_matrix`` read the run's C++ ``dtwc::Result``, which the
    result keeps and which fills its matrix on demand after a matrix-free
    OneBatchPAM/CLARA/TADPole run. A run that streamed an Arrow IPC file under
    ``ram_limit`` holds no series: its ``score`` and ``distance_matrix`` raise
    :class:`dtwcpp.InvalidInput`. An ``hpc`` run populates ``labels``,
    ``medoids`` (when written), and ``report`` (when downloaded); its
    ``distance_matrix`` is ``None``.
    """

    def __init__(self, labels, *, device, elapsed_s, k, n_series,
                 medoid_indices=None, cost=None, name="dataset", result=None,
                 report_text=None, hpc_result_dir=None):
        self.labels = np.asarray(labels)
        self.device = device
        self.elapsed_s = elapsed_s
        self.k = k
        self.n_series = n_series
        self.medoids = None if medoid_indices is None else np.asarray(medoid_indices)
        self.cost = cost
        self.name = name
        self.report = result.report if result is not None else report_text
        self._result = result
        self._hpc_result_dir = hpc_result_dir
        self._distance_matrix = None

    def _clustered(self):
        """The run's C++ Result (``_dtwcpp_core._Result``); an hpc run has none."""
        if self._result is None:
            import dtwcpp
            raise dtwcpp.InvalidInput(
                "no local distance matrix available to score (an hpc run "
                "returns labels only; score()/save(silhouettes) need "
                "cpu/gpu output).")
        return self._result

    @property
    def distance_matrix(self):
        """Dense N x N distances, filled on demand — C++ ``Result::distance_matrix()``,
        at the width the run stored them: float32 under ``matrix_dtype="float32"``.

        A matrix-free ``onebatch``/``clara``/``tadpole`` run leaves the matrix
        unmaterialised, so the O(N) schedule survives until this property is
        read; reading it is an explicit request for the full N x N and fills
        it, exactly as ``score()``/``save()`` do. A streamed run holds no
        series and raises :class:`dtwcpp.InvalidInput`; an ``hpc`` run has no
        local matrix and stays ``None``.
        """
        if self._distance_matrix is None and self._result is not None:
            self._distance_matrix = self._result._distance_matrix()
        return self._distance_matrix

    def summary(self):
        extra = f", cost={self.cost:.2f}" if self.cost is not None else ""
        return (f"[device={self.device}] {self.n_series} series, k={self.k}  ->  "
                f"{self.elapsed_s * 1e3:7.1f} ms{extra}")

    def score(self, name):
        """The clustering-quality score ``name`` names, computed by C++.

        ``"silhouette"`` (the MEAN silhouette), ``"davies_bouldin"``,
        ``"dunn"``, ``"calinski_harabasz"`` or ``"inertia"``; any other name
        raises :class:`dtwcpp.InvalidInput`, as does a streamed run, which
        holds no series.
        """
        return self._clustered().score(name)

    def save(self, directory):
        """Write the result files into ``directory`` as C++ ``Result::save``
        does, with the writer ``dtwc_cl`` uses: ``<name>_labels.csv``
        (``name,cluster``), ``<name>_medoids.csv``
        (``cluster,medoid_index,medoid_name``), ``<name>_distance_matrix.csv``
        and ``<name>_silhouettes.csv`` (``name,cluster,silhouette``), then the
        run report, ``<name>_report.txt``, last (none under ``no_report=True``).
        A streamed run holds no series: it writes its labels and medoids, with
        the reader's names, as ``dtwc_cl`` does, and its report.

        With one cluster the silhouette is undefined and no silhouettes file is
        written; a partition with fewer than two realised clusters prints
        ``Warning: silhouettes skipped: ...`` on ``stderr``. An ``hpc`` result
        copies the files it downloaded (labels, medoids, report) into ``directory``.
        """
        from dtwcpp import InvalidInput
        if self._result is None:
            if self._hpc_result_dir is not None and os.path.isdir(self._hpc_result_dir):
                import shutil
                out_dir = os.fspath(directory)
                os.makedirs(out_dir, exist_ok=True)
                prefix = f"{self.name}_"
                for fname in os.listdir(self._hpc_result_dir):
                    if fname.startswith(prefix):
                        shutil.copy2(
                            os.path.join(self._hpc_result_dir, fname),
                            os.path.join(out_dir, fname),
                        )
                return directory
            raise InvalidInput(
                "save: this hpc result holds the labels only; dtwc_cl wrote the "
                "result files on the cluster.")
        self._result.save(os.fspath(directory))
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
        else:
            plt.close(fig)  # not shown: the saved file is what plot gives back
        return png


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
    if key == "ram_limit":  # bytes, or a size such as "8G" that C++ reads
        ok = not flag and isinstance(value, (int, np.integer, str))
        kind = "a byte count or a size such as '8G'"
        if ok and not isinstance(value, str):
            value = int(value)
    elif isinstance(current, bool):
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


def _run_local(dataset, config, device, backend, *, output="", checkpoint="", checkpoint_interval=0):
    """The local run of :func:`cluster` and :func:`do_clustering`: C++ ``dtwc::run`` on ``device``."""
    import dtwcpp
    from dtwcpp import _dtwcpp_core
    t0 = time.perf_counter()
    # C++ checks every value before any series is read, as dtwc::run does, so
    # that a bad one costs no read and reads the same for every input. A path
    # C++ reads goes as the path, with the Dataset's reader options (its memory
    # plan may stream an Arrow IPC file); Parquet, which pyarrow reads, and
    # series in memory go as the Data, their skips applied (run refuses a
    # reader option beside series in memory). The run is named as the Dataset
    # is, else as dtwc_cl names it.
    config.device = device
    _dtwcpp_core.apply(config, _dtwcpp_core.Problem())
    if dataset.is_path and dataset._ndim == 1 and not _dtwcpp_core._parquet_files(os.fspath(dataset.source)):
        source = os.fspath(dataset.source)
        reader = {"skip_cols": dataset.skip_cols, "skip_rows": dataset.skip_rows,
                  "delimiter": dataset.delimiter or ""}
    else:
        source, reader = dataset.as_data(), {}
    config.name = config.name or dataset.name or _dtwcpp_core._default_name(
        source if isinstance(source, str) else "")
    # An extension that links no HiGHS (the wheel) hands MIP on HiGHS and
    # LR-core's kelley root to dtwcpp._mip.solve, on highspy.
    result = _dtwcpp_core._run(config, source, dtype=dataset.dtype,
                               external=not dtwcpp.HIGHS_AVAILABLE, output=output,
                               checkpoint=checkpoint, checkpoint_interval=checkpoint_interval,
                               **reader)
    return Result(result.labels, device=backend,
                  elapsed_s=time.perf_counter() - t0, k=config.n_clusters,
                  n_series=len(result.labels), medoid_indices=result.medoids,
                  cost=result.cost, name=config.name, result=result)


def cluster(data, k, **keys):
    """Cluster a dataset into ``k`` clusters with DTW, on the global or given device.

    A local run is C++ ``dtwc::run``, the run ``dtwc_cl`` makes, without its
    files: C++ reads a path itself (CSV/TSV, a folder or an Arrow IPC file),
    pyarrow reads Parquet, and series in memory go as they are. ``data`` may be
    a :class:`Dataset`, a path, or an array; ``dtype`` is
    :func:`load`'s, for a path or an array cluster() loads (``"float32"`` stores
    the series at half the memory), and a :class:`Dataset` keeps its own: one
    given that differs raises :class:`dtwcpp.InvalidInput`, as in
    :func:`load_data`. ``keys`` are the
    keys of a dtwc_cl run — its long names in snake_case: ``method``, ``band``,
    ``metric``, ``variant`` and its parameters (``wdtw_g``, ``adtw_penalty``,
    ``sdtw_gamma``, ``msm_c``, ``twe_nu``, ``twe_lambda``), ``mv_mode``,
    ``missing_strategy``, ``max_iter``, ``n_init``, ``seed``, ``sample_size``,
    ``n_samples``, ``batch_size``, ``linkage``, ``dc``, ``solver`` and the MIP
    settings, ``gpu_precision``, ``gpu_share`` (``device="cpu+gpu"``: the GPU's
    share of the work it splits with the CPU; -1, the default, measures it, a
    value in [0, 1] forces it so a run repeats; another device refuses a
    forced one), ``matrix_dtype`` (``"float32"`` stores the
    distance matrix at half the memory, each distance rounded to a float),
    ``name``, ``verbose`` (``dtwc_cl -v``'s lines, the memory plan among
    them), ``no_report`` (no run report: ``Result.report`` is ``""``),
    ``ram_limit`` (the memory the run may use, bytes or a size such as
    ``"8G"``; 0, the default, plans with 90 % of the memory available; a limit
    that cannot hold the run raises :class:`dtwcpp.InvalidInput`, and an Arrow
    IPC file whose series it cannot hold streams through CLARA) — read and
    checked by C++; an unknown key raises :class:`dtwcpp.InvalidInput`.
    ``method`` is ``"auto"`` unless given: PAM on a GPU and for up to 5,000
    series on the CPU, CLARA above, and CLARA too where PAM's distance matrix
    does not fit the memory plan. ``device=None`` uses the global default (see
    :func:`dtwcpp.device`). ``"hpc"`` and ``"hpc:gpu"`` run the clustering on a
    SLURM cluster as one job.toml: the keys given and the data (a path is read
    on the cluster, never here); the result holds the labels. ``gpu_device``
    names the GPU of an ``"hpc:gpu"`` run (``"a100"``, ``"a6000"``, ``"l40s"``,
    ``"h100"``); without it the run takes any GPU of compute capability 8.0 or
    newer.
    """
    import dtwcpp
    from dtwcpp import InvalidInput, _dtwcpp_core, _resolve_device
    device = keys.pop("device", None)
    gpu_device = keys.pop("gpu_device", None)
    given = {"dtype": keys.pop("dtype")} if "dtype" in keys else {}  # load_data's rule beside a Dataset
    config = _config(k, keys)
    eff = device if device is not None else dtwcpp.device()
    backend, _ = _resolve_device(eff)
    if gpu_device is not None and eff.strip().lower() != "hpc:gpu":
        raise InvalidInput(f"cluster: gpu_device names the GPU of a device='hpc:gpu' "
                           f"run, and the device is '{eff}'.")
    data = load_data(data, **given)

    if backend == "hpc":
        t0 = time.perf_counter()
        from dtwcpp import _hpc
        # As a local run, C++ checks every value before anything is written or
        # sent; the device stays the CPU here, since the GPU is the cluster's.
        _dtwcpp_core.apply(config, _dtwcpp_core.Problem(config.name or data.name))
        details = _hpc.cluster_on_hpc(data, config, keys, device=eff,
                                      gpu_device=gpu_device, return_details=True)
        labels = details["labels"]
        return Result(labels, device="hpc", elapsed_s=time.perf_counter() - t0,
                      k=k, n_series=len(labels),
                      medoid_indices=details.get("medoids"),
                      name=config.name or data.name,
                      report_text=details.get("report"),
                      hpc_result_dir=details.get("result_dir"))
    return _run_local(data, config, eff, backend)


def plot(result, **kwargs):
    """Top-level alias for ``result.plot(...)``."""
    return result.plot(**kwargs)


# ---------------------------------------------------------------------------
# CasADi + PyTorch device= workflow: settings -> load_data -> do_clustering -> write_data
# ---------------------------------------------------------------------------
class Settings:
    """Validated workflow configuration shared across :func:`load_data`,
    :func:`do_clustering`, and :func:`write_data`.

    Supports both crash-safe persistent output + ``.dtwm`` checkpointing by
    default (``output="./results"``) and pure in-memory execution
    (``output=""``), plus PyTorch-style ``device="cpu" | "gpu" | "cpu+gpu" |
    "hpc" | "hpc:gpu"`` switching.
    """

    _WORKFLOW_ATTRS = frozenset({
        "input", "skip_cols", "skip_rows", "delimiter", "dtype", "ndim",
        "output", "checkpoint", "checkpoint_interval", "device",
        "gpu_device", "upload_input",
    })

    def __init__(self, **kwargs):
        from dtwcpp import _dtwcpp_core

        object.__setattr__(self, "_ready", False)
        object.__setattr__(self, "input", "")
        object.__setattr__(self, "skip_cols", 0)
        object.__setattr__(self, "skip_rows", 0)
        object.__setattr__(self, "delimiter", None)
        object.__setattr__(self, "dtype", "float64")
        object.__setattr__(self, "ndim", 1)
        object.__setattr__(self, "output", "./results")
        object.__setattr__(self, "checkpoint", "")
        object.__setattr__(self, "checkpoint_interval", 0)
        object.__setattr__(self, "device", None)
        object.__setattr__(self, "gpu_device", None)
        object.__setattr__(self, "upload_input", False)
        object.__setattr__(self, "_config", _dtwcpp_core.Config())
        object.__setattr__(self, "_clustering_keys", [])
        self._update(**kwargs)
        object.__setattr__(self, "_ready", True)

    def __setattr__(self, name, value):
        if name.startswith("_") or not self.__dict__.get("_ready", False):
            object.__setattr__(self, name, value)
            return
        if (
            name != "k"
            and name not in self._WORKFLOW_ATTRS
            and not hasattr(type(self._config), name)
        ):
            raise AttributeError(f"Unknown settings attribute {name!r}")
        self._update(**{name: value})

    def _update(self, **kwargs):
        import dtwcpp
        from dtwcpp import InvalidInput, _dtwcpp_core, _resolve_device

        if "k" in kwargs and "n_clusters" in kwargs and kwargs["k"] != kwargs["n_clusters"]:
            raise InvalidInput("settings: k and n_clusters disagree; pass only one.")
        if "k" in kwargs:
            kwargs["n_clusters"] = kwargs.pop("k")

        touched_config = []
        prev_config = {}
        prev_attrs = dict(self.__dict__)  # a refused update leaves the Settings as it was
        try:
            for key, value in kwargs.items():
                if key in ("input", "output", "checkpoint"):
                    object.__setattr__(self, key, "" if value is None else os.fspath(value))
                elif key in ("skip_cols", "skip_rows", "checkpoint_interval"):
                    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                        raise TypeError(f"settings: {key} must be an integer")
                    if value < 0:
                        raise InvalidInput(f"settings: {key} must be non-negative.")
                    object.__setattr__(self, key, int(value))
                elif key == "ndim":
                    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                        raise TypeError("settings: ndim must be an integer")
                    if value < 1:
                        raise InvalidInput("settings: ndim must be at least 1.")
                    object.__setattr__(self, "ndim", int(value))
                elif key == "delimiter":
                    if value is not None and (not isinstance(value, str) or len(value) > 1):
                        raise InvalidInput("settings: delimiter must be None or a single character.")
                    object.__setattr__(self, "delimiter", value)
                elif key == "dtype":
                    if not isinstance(value, str):
                        raise TypeError(f"settings: dtype must be a string, got {type(value).__name__}")
                    object.__setattr__(self, "dtype", _dtwcpp_core._dtype(value))
                elif key == "device":
                    if value is not None:
                        _resolve_device(value)
                    object.__setattr__(self, "device", value)
                elif key == "gpu_device":
                    if value is not None and not isinstance(value, str):
                        raise TypeError("settings: gpu_device must be a string or None")
                    object.__setattr__(self, "gpu_device", value)
                elif key == "upload_input":
                    if not isinstance(value, (bool, np.bool_)):
                        raise TypeError("settings: upload_input must be a bool")
                    object.__setattr__(self, "upload_input", bool(value))
                else:
                    if hasattr(self._config, key) and key not in prev_config:
                        prev_config[key] = getattr(self._config, key)
                    _set_key(self._config, key, value, name="k" if key == "n_clusters" else key)
                    if key == "n_clusters" and self._config.n_clusters < 1:
                        raise InvalidInput("settings: k (n_clusters) must be at least 1.")
                    touched_config.append(key)

            if touched_config or "device" in kwargs:
                saved_k, saved_device = self._config.n_clusters, self._config.device
                try:
                    if saved_k < 1:
                        self._config.n_clusters = 1
                    # Checked on the device do_clustering runs on, the global one when none is set; hpc on the CPU.
                    device = self.device if self.device is not None else dtwcpp.device()
                    if _resolve_device(device)[0] != "hpc":
                        self._config.device = device
                    _dtwcpp_core.apply(self._config, _dtwcpp_core.Problem())
                finally:
                    if saved_k < 1:
                        self._config.n_clusters = saved_k
                    self._config.device = saved_device
                for key in touched_config:
                    if key != "n_clusters" and key not in self._clustering_keys:
                        self._clustering_keys.append(key)
        except Exception:
            self.__dict__.update(prev_attrs)
            for key, old_val in prev_config.items():
                setattr(self._config, key, old_val)
            raise

    @property
    def k(self):
        return self._config.n_clusters

    @k.setter
    def k(self, value):
        self._update(k=value)

    def __getattr__(self, name):
        if not name.startswith("_") and "_config" in self.__dict__ and hasattr(type(self._config), name):
            return getattr(self._config, name)
        raise AttributeError(f"'Settings' object has no attribute {name!r}")

    def copy(self, **overrides):
        """Return a new :class:`Settings` with ``overrides`` applied."""
        cloned = Settings()
        for attr in self._WORKFLOW_ATTRS:
            object.__setattr__(cloned, attr, getattr(self, attr))
        if self._config.n_clusters > 0:
            _set_key(cloned._config, "n_clusters", self._config.n_clusters, name="k")
        for key in self._clustering_keys:
            setattr(cloned._config, key, getattr(self._config, key))
        object.__setattr__(cloned, "_clustering_keys", list(self._clustering_keys))
        if overrides:
            cloned._update(**overrides)
        return cloned

    def to_dict(self):
        """Return all workflow settings as a dictionary."""
        out = {
            "input": self.input,
            "k": self._config.n_clusters,
            "method": self._config.method,
            "band": self._config.band,
            "metric": self._config.metric,
            "variant": self._config.variant,
            "device": self.device,
            "gpu_device": self.gpu_device,
            "output": self.output,
            "checkpoint": self.checkpoint,
            "checkpoint_interval": self.checkpoint_interval,
            "dtype": self.dtype,
            "ndim": self.ndim,
            "skip_cols": self.skip_cols,
            "skip_rows": self.skip_rows,
            "delimiter": self.delimiter,
            "upload_input": self.upload_input,
        }
        for key in self._clustering_keys:
            out[key] = getattr(self._config, key)
        return out

    def __repr__(self):
        dev = self.device if self.device is not None else "default"
        return (f"Settings(k={self.k}, method={self._config.method!r}, "
                f"device={dev!r}, output={self.output!r})")


def settings(base=None, **kwargs):
    """Build or refine a CasADi-style :class:`Settings` object.

    By default ``output="./results"`` writes result files automatically in
    :func:`do_clustering`; pass ``output=""`` for pure in-memory clustering.
    """
    if base is not None:
        if not isinstance(base, Settings):
            raise TypeError(f"settings: base must be a Settings instance, got {type(base).__name__}")
        return base.copy(**kwargs)
    return Settings(**kwargs)


_READER_OPTIONS = ("skip_cols", "skip_rows", "delimiter", "dtype", "ndim")  # a Dataset is read with its own


def load_data(source=None, settings=None, **kwargs):
    """Create a lazy :class:`Dataset` configured from ``settings`` and/or ``kwargs``.

    On ``device="hpc"`` file paths stay lazy and are read on the SLURM cluster;
    locally, NumPy arrays and Arrow buffers materialize directly into C++
    ``Data`` without Python list conversion. A :class:`Dataset` passes through
    with the reader options it was made with (``skip_cols``, ``skip_rows``,
    ``delimiter``, ``dtype``, ``ndim``; once it has read an Arrow IPC file, the
    file's ``ndim``): one of them given here, or one the settings give other
    than its default, that differs from the Dataset's raises
    :class:`dtwcpp.InvalidInput`. A :class:`Data` keeps
    its own ``ndim``: one given, or one the settings give other than 1, that differs is refused too.
    """
    from dtwcpp import InvalidInput, _dtwcpp_core

    s = settings if isinstance(settings, Settings) else Settings()
    if settings is not None and not isinstance(settings, Settings):
        raise TypeError(f"load_data: settings must be a Settings instance, got {type(settings).__name__}")

    if source is None:
        if not s.input:
            raise InvalidInput("load_data: pass source or set input=... in settings().")
        source = s.input

    if isinstance(source, Dataset):
        unexpected = sorted(set(kwargs) - set(_READER_OPTIONS))
        if unexpected:
            raise InvalidInput(
                f"load_data: Dataset keeps the options it was made with; unexpected {unexpected}."
            )
        # It is read with its own reader options: one given here, or one the settings give other than its
        # default, must be the same.
        s = s.copy(**kwargs)  # the options given, checked as settings() checks them
        made_ndim = source._ndim if source._data is None else source._data.ndim
        for key, given, made, default in (
                ("skip_cols", s.skip_cols, source.skip_cols, 0), ("skip_rows", s.skip_rows, source.skip_rows, 0),
                ("delimiter", s.delimiter or None, source.delimiter or None, None),
                ("dtype", s.dtype, source.dtype, "float64"), ("ndim", s.ndim, made_ndim, 1)):
            if (key in kwargs or given != default) and given != made:
                raise InvalidInput(
                    f"load_data: the Dataset was made with {key}={made!r} and the settings give {key}={given!r}; "
                    f"a Dataset keeps the options it was made with, so pass {key} to load_data, or leave it out "
                    "of the settings.")
        return source

    ndim_given = "ndim" in kwargs
    skip_cols = kwargs.pop("skip_cols", s.skip_cols)
    skip_rows = kwargs.pop("skip_rows", s.skip_rows)
    delimiter = kwargs.pop("delimiter", s.delimiter)
    name = kwargs.pop("name", s._config.name or None)
    dtype = kwargs.pop("dtype", s.dtype)
    ndim = kwargs.pop("ndim", s.ndim)
    if kwargs:
        raise InvalidInput(f"load_data: unknown option(s): {', '.join(sorted(kwargs))}.")

    if isinstance(ndim, (bool, np.bool_)) or not isinstance(ndim, (int, np.integer)):
        raise TypeError("load_data: ndim must be an integer")
    if ndim < 1:
        raise InvalidInput("load_data: ndim must be at least 1.")
    if isinstance(source, _dtwcpp_core.Data) and (ndim_given or ndim != 1) and ndim != source.ndim:
        raise InvalidInput(f"load_data: Data already carries ndim={source.ndim}, which disagrees with ndim={ndim}.")

    ds = Dataset(source, skip_cols=skip_cols, skip_rows=skip_rows,
                 delimiter=delimiter, name=name, dtype=dtype)
    ds._ndim = int(ndim) if ndim != 1 else ds._ndim  # a Data keeps its own ndim (a given one equals it)
    return ds


def do_clustering(data=None, settings=None, **overrides):
    """Execute clustering using ``settings`` (with optional ``overrides``).

    Supports automatic result file persistence (``output="./results"`` by
    default), crash-safe ``.dtwm`` checkpoints (``checkpoint=...``), pure
    in-memory execution (``output=""``), and transparent ``device="cpu" | "gpu"
    | "cpu+gpu" | "hpc" | "hpc:gpu"`` execution.
    """
    import dtwcpp
    from dtwcpp import InvalidInput, _dtwcpp_core, _resolve_device

    if isinstance(data, Settings) and settings is None:
        settings, data = data, None
    if settings is not None and not isinstance(settings, Settings):
        raise TypeError(f"do_clustering: settings must be a Settings instance, got {type(settings).__name__}")

    eff = settings.copy(**overrides) if settings is not None else Settings(**overrides)
    if eff.k < 1:
        raise InvalidInput("do_clustering: k (n_clusters) must be at least 1.")

    eff_dev = eff.device if eff.device is not None else dtwcpp.device()
    backend, _ = _resolve_device(eff_dev)
    if eff.gpu_device is not None and eff_dev.strip().lower() != "hpc:gpu":
        raise InvalidInput(
            f"do_clustering: gpu_device names the GPU of a device='hpc:gpu' "
            f"run, and the device is '{eff_dev}'."
        )

    dataset = load_data(data, eff, **{key: overrides[key] for key in _READER_OPTIONS if key in overrides})
    config = eff._config

    if backend == "hpc":
        t0 = time.perf_counter()
        from dtwcpp import _hpc
        if eff.checkpoint or eff.checkpoint_interval:  # the run is the cluster's: a local path names nothing there
            raise InvalidInput("do_clustering: device='hpc' runs on the cluster, which would ignore a local "
                               "checkpoint; leave checkpoint empty and checkpoint_interval at 0.")
        _dtwcpp_core.apply(config, _dtwcpp_core.Problem(config.name or dataset.name))
        details = _hpc.cluster_on_hpc(
            dataset,
            config,
            eff._clustering_keys,
            device=eff_dev,
            gpu_device=eff.gpu_device,
            upload_input=eff.upload_input,
            return_details=True,
        )
        labels = details["labels"]
        res = Result(
            labels,
            device="hpc",
            elapsed_s=time.perf_counter() - t0,
            k=eff.k,
            n_series=len(labels),
            medoid_indices=details.get("medoids"),
            name=config.name or dataset.name,
            report_text=details.get("report"),
            hpc_result_dir=details.get("result_dir"),
        )
        if eff.output:
            res.save(eff.output)
        return res

    return _run_local(dataset, config, eff_dev, backend, output=eff.output,
                      checkpoint=eff.checkpoint, checkpoint_interval=eff.checkpoint_interval)


def write_data(result, directory="./results"):
    """Write ``result``'s files (labels, medoids, distance matrix, silhouettes,
    and run report) into ``directory``."""
    if not isinstance(result, Result):
        raise TypeError(f"write_data: expected a Result, got {type(result).__name__}")
    return result.save(directory)
