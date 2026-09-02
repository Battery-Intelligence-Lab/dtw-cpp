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
        self.source = source
        self.skip_cols = skip_cols
        self.skip_rows = skip_rows
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

        The handle holds the C++ object, not a ``list[list[float]]``: a path is
        parsed by the C++ ``DataLoader`` — the reader C++ and the CLI use — and
        the result is handed straight to ``Problem.set_data(Data)``, so a
        Tier-1 run creates no Python floats at all. ``skip_cols`` therefore
        drops leading FIELDS before numeric parsing (an id column may be text),
        variable-length rows are preserved, and the names are the loader's own
        (file stem per file for a folder, 1-based row number for a batch file).
        ``skip_rows`` drops leading FILE LINES for a path source and leading
        SERIES for an in-memory one — one memory row is one file line — and an
        in-memory source is named by its ordinal, as
        ``Dataset::materialize_local`` names it (api.cpp).
        """
        if self._data is None:
            from dtwcpp import _dtwcpp_core
            if self.is_path:
                self._data = _dtwcpp_core._read_data(
                    os.fspath(self.source), self.skip_cols, self.skip_rows,
                    self.delimiter or "")
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


# score() names accepted by Result.score (api-contract-2.0.md §1.4). Each resolves
# to the Tier-2 scores::* function; "silhouette" returns the MEAN silhouette.
_SCORE_NAMES = ("silhouette", "davies_bouldin", "dunn", "calinski_harabasz", "inertia")


class Result:
    """Outcome of :func:`cluster` — the canonical 2.0 ``Result`` (api-contract §1.4).

    Members: ``labels``, ``medoids``, ``score(name)``, ``save(dir)``, ``plot()``,
    plus ``cost`` and ``device``. ``distance_matrix`` fills on demand from the
    retained ``Problem`` after a matrix-free OneBatchPAM/CLARA/TADPole run, and
    is ``None`` only for an ``hpc`` run. ``ClusterResult`` is a deprecated alias
    name.
    """

    def __init__(self, labels, *, device, elapsed_s, k, n_series,
                 medoid_indices=None, distance_matrix=None, cost=None, name="dataset",
                 series_names=None, problem=None):
        self.labels = np.asarray(labels)
        self.device = device
        self.elapsed_s = elapsed_s
        self.k = k
        self.n_series = n_series
        self.medoids = None if medoid_indices is None else np.asarray(medoid_indices)
        self._distance_matrix = distance_matrix
        self.cost = cost
        self.name = name
        self._series_names = series_names
        # C++ Result owns the Problem it clustered (api.cpp), so score() and
        # save() can fill a matrix-free run's distance matrix on demand.
        self._problem = problem

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

    def _names(self, n=None):
        """Series names for output — the loader's, as C++ ``series_name(i)`` is."""
        if self._series_names is not None:
            return list(self._series_names)
        return [str(i) for i in range(len(self.labels) if n is None else n)]

    def _scoring_problem(self):
        """Rebuild a Tier-2 Problem carrying this result's distance matrix + labels.

        Scores read state from a Problem (its distance matrix + clusters_ind +
        centroids_ind), so score()/save() reconstruct a minimal one. Dummy series
        stand in — the scores use only the distance matrix and the labels.

        A matrix-free run kept its own Problem instead of a matrix; filling it
        here is what C++ ``Result::score()``/``save()`` do, so the O(N)
        schedule survives until a score is explicitly asked for.
        """
        import dtwcpp
        if self._distance_matrix is None:
            if self._problem is None:
                raise dtwcpp.InvalidInput(
                    "no local distance matrix available to score (an hpc run "
                    "returns labels only; score()/save(silhouettes) need "
                    "cpu/gpu output)."
                )
            prob = self._problem
            prob.fill_distance_matrix()
            prob.set_n_clusters(int(self.k))
            prob.clusters_ind = [int(x) for x in self.labels]
            if self.medoids is not None:
                prob.centroids_ind = [int(x) for x in self.medoids]
            return prob
        D = np.asarray(self._distance_matrix, dtype=float)
        n = D.shape[0]
        names = self._names(n)
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
        with the CLI output is the Phase 2.4 conformance fixture's contract: the
        names are the loader's (UTF-8, as C++ emits them), the line ending is
        the platform one C++'s text-mode ``ofstream`` writes, and the
        silhouettes carry C++'s ``setprecision(8)``.

        With fewer than two realised clusters the silhouette is undefined: this
        prints ``Warning: silhouettes skipped: ...`` to ``stderr`` and skips
        ``<name>_silhouettes.csv`` instead of failing a clustering that
        succeeded — the same stream and the same text as C++ ``Result::save``,
        so a ``-W error`` caller is not broken by a successful save. Asking for
        the number — ``score("silhouette")`` — still raises ``UndefinedScore``.
        """
        import sys

        import dtwcpp
        os.makedirs(directory, exist_ok=True)
        names = self._names()
        base = os.path.join(directory, self.name)

        # C++ writes the loader's bytes through a text-mode ofstream: UTF-8
        # payload, platform line ending. open(..., "w") alone would use the
        # locale encoding and mangle a non-ASCII series name.
        def _text(path):
            return open(path, "w", encoding="utf-8")

        with _text(base + "_labels.csv") as f:
            f.write("name,cluster\n")
            for nm, lab in zip(names, self.labels):
                f.write(f"{nm},{int(lab)}\n")

        with _text(base + "_medoids.csv") as f:
            f.write("cluster,medoid_index,medoid_name\n")
            if self.medoids is not None:
                for c, m in enumerate(self.medoids):
                    f.write(f"{c},{int(m)},{names[int(m)]}\n")

        if self._distance_matrix is not None or self._problem is not None:
            prob = self._scoring_problem()
            matrix = np.asarray(
                self._distance_matrix if self._distance_matrix is not None
                else prob.distance_matrix(), dtype=float)
            # core::detail::preflight_distance_matrix_csv: +/-inf is refused
            # BEFORE the file is opened, row-major, naming the first offender.
            if np.isinf(matrix).any():
                i, j = (int(x) for x in np.argwhere(np.isinf(matrix))[0])
                raise dtwcpp.InvalidInput(
                    f"distance-matrix CSV: computed non-finite value at row {i}, "
                    f"column {j}.")
            # core::detail::distance_matrix_csv_token: to_chars(general,
            # max_digits10) into a stream opened in BINARY mode, so LF and
            # 17 significant digits regardless of platform; a NaN cell is an
            # uncomputed distance and is written as an EMPTY field.
            with open(base + "_distance_matrix.csv", "wb") as f:
                for row in matrix:
                    line = ",".join(
                        "" if value != value else f"{value:.17g}"
                        for value in row) + "\n"
                    f.write(line.encode("ascii"))
            try:
                sil = dtwcpp.silhouette(prob)
            except dtwcpp.UndefinedScore as e:
                print(f"Warning: silhouettes skipped: {e}", file=sys.stderr)
                return directory
            with _text(base + "_silhouettes.csv") as f:
                f.write("name,cluster,silhouette\n")
                for nm, lab, s in zip(names, self.labels, sil):
                    f.write(f"{nm},{int(lab)},{s:.8g}\n")
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


# Canonical clustering methods. This is exactly the dtwc_cl CLI vocabulary
# (dtwc_cl.cpp: "auto, pam, clara, kmedoids, mip, hierarchical"), so the SAME
# name is valid on cpu/gpu (dispatched here) and on hpc (forwarded to
# dtwc_cl --method). "hclust" is the CLI's alias for "hierarchical".
_METHODS = ("auto", "pam", "onebatch", "clara", "kmedoids", "mip",
            "lrcore", "hierarchical", "tadpole")
_AUTO_PAM_SERIES_LIMIT = 5000
_CPP_INT_MAX = (1 << 31) - 1


def _normalize_tier1_int(name, value, *, minimum):
    """Normalize one public integer to the signed C++ ``int`` domain."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= _CPP_INT_MAX:
        raise ValueError(
            f"{name} must be in [{minimum}, {_CPP_INT_MAX}] for the C++ "
            "Tier-1 API"
        )
    return value


def _validate_common(data, k, max_iter):
    """Mirror C++ ``validate_common`` before any Python backend side effect.

    A raw array or path has the implicit ``load(..., skip_cols=0)`` value.  An
    existing :class:`Dataset` can carry a caller-supplied value, so inspect it
    without materializing or mutating the handle.  The order matches C++:
    ``k``, ``max_iter``, then ``skip_cols`` and ``skip_rows``.
    """
    k = _normalize_tier1_int("k", k, minimum=1)
    max_iter = _normalize_tier1_int("max_iter", max_iter, minimum=1)
    handle = data if isinstance(data, Dataset) else None
    skip_cols = _normalize_tier1_int(
        "skip_cols", handle.skip_cols if handle else 0, minimum=0)
    skip_rows = _normalize_tier1_int(
        "skip_rows", handle.skip_rows if handle else 0, minimum=0)
    return k, max_iter, skip_cols, skip_rows


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
    # These branches do not receive max_iter as a direct function argument;
    # configure the shared Problem before dispatch instead.
    prob.set_max_iter(max_iter)
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
    # Match C++ validate_common before lazy loading, device lookup/resolution,
    # remote submission, or local construction/compute.  Besides preventing
    # partial side effects, normalization keeps NumPy integers from reaching
    # nanobind or the CLI with backend-dependent conversion behavior.
    k, max_iter, skip_cols, skip_rows = _validate_common(data, k, max_iter)
    method = _normalize_method(method)

    import dtwcpp
    from dtwcpp import _resolve_device
    eff = device if device is not None else dtwcpp.device()
    backend, _ = _resolve_device(eff)
    data = load(data)

    t0 = time.perf_counter()
    if backend == "hpc":
        from dtwcpp import InvalidInput, _hpc
        # The SLURM wrapper takes a fixed positional argument list with no
        # skip_rows slot, so the remote CLI cannot receive it. Refuse loudly
        # instead of clustering the header rows the caller asked to drop.
        if skip_rows:
            raise InvalidInput(
                "cluster: skip_rows is not carried by the HPC transport; strip "
                "the header rows before staging, or use device='cpu'/'gpu'."
            )
        source = data.source if data.is_path else data.as_series()
        labels = _hpc.cluster_on_hpc(source, k, method=method, band=band,
                                     skip_cols=skip_cols, name=f"dtwc_{data.name}",
                                     max_iter=max_iter)
        return Result(labels, device="hpc", elapsed_s=time.perf_counter() - t0,
                      k=k, n_series=len(labels), name=data.name)

    # Matrix-free methods must not accidentally pay the N^2 cost in this Tier-1
    # wrapper. They currently execute on CPU; an explicit GPU request is rejected
    # loudly instead of silently defeating their scaling contract.
    from dtwcpp import compute_distance_matrix, InvalidInput, Problem
    # The owning C++ Data, never a list[list[float]]: it is moved into the
    # Problem, so a Tier-1 run holds one copy of the payload instead of the
    # ~4x PyFloat+list expansion the old as_series() route retained.
    series_data = data.as_data()
    n_series = series_data.size
    # Same guards, same messages, same order as C++ cluster() (api.cpp).
    if not n_series:
        raise InvalidInput("cluster: dataset is empty.")
    if k > n_series:
        raise InvalidInput("cluster: k must not exceed the number of series.")
    method = _resolve_tier1_method(method, n_series, backend)
    # The loader's names, so Tier-1 output carries what C++ series_name(i) does.
    names = data.series_names()
    prob = Problem(data.name)
    # Configure the band before set_data() refreshes/rebinds the Problem's DTW
    # callable.  Matrix-free methods query distances from Problem directly, so
    # relying only on compute_distance_matrix(..., band=...) silently made them
    # unbanded.
    prob.set_band(band)
    prob.set_data(series_data)
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
    if matrix_free:
        D = None
    elif backend == "cpu":
        # Fill through the Problem the CLI itself uses: no Python list of the
        # series is created, and the matrix the scores read is the same object.
        D = prob.distance_matrix()
    else:
        # GPU backends take their own series buffer; materialise it transiently.
        D = compute_distance_matrix(series_data.p_vec, band=band, device=eff)
        prob.set_distance_matrix(D)
    labels, medoid_indices, cost = _run_local_method(prob, method, k, max_iter)
    return Result(labels, device=backend,
                  elapsed_s=time.perf_counter() - t0, k=k, n_series=n_series,
                  medoid_indices=medoid_indices, distance_matrix=D,
                  cost=cost, name=data.name, series_names=names, problem=prob)


def plot(result, **kwargs):
    """Top-level alias for ``result.plot(...)``."""
    return result.plot(**kwargs)
