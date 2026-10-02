"""
@file _clustering.py
@brief DTWClustering: the sklearn-compatible DTW k-medoids estimator.
@details
    fit() hands C++ one Problem, configured by a Config exactly as
    dtwcpp.cluster() is, and Problem.cluster() runs the method, its n_init
    seeded restarts on one distance matrix. metric="precomputed" takes the
    sklearn convention: fit reads an N x N distance matrix, predict and
    transform an M x N one.
@author Volkan Kumtepeli
"""
import numbers

import numpy as np
from dtwcpp import distance
from dtwcpp._dtwcpp_core import DEFAULT_RANDOM_SEED, data_from_arrow_c_array

try:  # scikit-learn is an optional dependency of the base wheel.
    from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
    from sklearn.exceptions import NotFittedError
except ImportError:
    class BaseEstimator:
        """Minimal sklearn-compatible base when sklearn is not installed."""
        def get_params(self, deep=True):
            return {key: value for key, value in self.__dict__.items()
                    if not key.endswith("_") and not key.startswith("_")}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class ClusterMixin:
        pass

    class TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)

    class NotFittedError(RuntimeError):
        pass


# The methods that cluster the distance matrix they are given; onebatch, clara
# and tadpole compute the distances they need from the series (C++ chooses
# auto's method by N, so auto is not among them).
_READS_THE_MATRIX = ("pam", "kmedoids", "mip", "lrcore", "hierarchical")


def _series_list(X, *, allow_nan):
    """Raw series as a list of float64 rows; ragged rows are kept. An Arrow C
    Data source (polars, DuckDB, pyarrow, pandas) is read by nanoarrow."""
    if hasattr(X, "__arrow_c_array__") or hasattr(X, "__arrow_c_stream__"):
        return [np.asarray(s, dtype=np.float64) for s in data_from_arrow_c_array(X).p_vec]
    if type(X).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse input is not supported; provide a dense array.")
    if isinstance(X, np.ndarray):
        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")
        if X.ndim != 2:
            raise ValueError(
                "Expected a 2D raw time-series array. Reshape your data to "
                "(n_samples, n_timesteps).")
        if X.shape[0] == 0:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={X.shape}) while a minimum of 1 is required.")
        if X.shape[1] == 0:
            raise ValueError(f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required.")
        values = [np.asarray(row, dtype=np.float64) for row in X]
    elif isinstance(X, (list, tuple)):
        try:
            rows = list(X)
            if any(np.iscomplexobj(row) for row in rows):
                raise ValueError("Complex data not supported")
            values = [np.asarray(row, dtype=np.float64) for row in rows]
        except TypeError as exc:
            raise ValueError("raw input must be an iterable of one-dimensional series") from exc
    else:
        try:
            return _series_list(np.asarray(X), allow_nan=allow_nan)
        except (TypeError, ValueError) as exc:
            raise ValueError("raw input must be an array-like collection of series") from exc
    if not values:
        raise ValueError("X must contain at least one series")
    for row in values:
        if row.ndim != 1 or row.size == 0:
            raise ValueError("each time series must be a non-empty one-dimensional array")
        if not np.isfinite(row).all() and (not allow_nan or np.isinf(row).any()):
            raise ValueError("Input contains NaN or inf")
    return values


def _precomputed_matrix(X, *, square=False, n_train=None):
    if type(X).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse input is not supported; provide a dense array.")
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("precomputed distances must be a 2D array")
    if square and matrix.shape[0] != matrix.shape[1]:
        raise ValueError("fit with metric='precomputed' requires a square distance matrix")
    if n_train is not None and matrix.shape[1] != n_train:
        raise ValueError(
            f"precomputed query matrix must have {n_train} columns; got {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError("Input contains NaN or inf")
    if np.any(matrix < 0.0):
        raise ValueError("precomputed distances must be non-negative")
    if square:
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("precomputed training distance matrix must be symmetric")
        if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("precomputed training distance matrix must have a zero diagonal")
    return np.ascontiguousarray(matrix)


class DTWClustering(ClusterMixin, TransformerMixin, BaseEstimator):
    """K-medoids clustering with DTW distance, sklearn-compatible.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    method : str, default="pam"
        Any ``dtwcpp.cluster`` method: ``"pam"`` (FastPAM, Schubert &
        Rousseeuw 2021), ``"onebatch"``, ``"clara"``, ``"kmedoids"``,
        ``"mip"``, ``"lrcore"``, ``"hierarchical"``, ``"tadpole"`` or
        ``"auto"``. With ``metric="precomputed"`` the methods that compute their
        own distances (onebatch, clara, tadpole) are refused.
    variant : str, default="standard"
        DTW variant: ``"standard"``, ``"ddtw"``, ``"wdtw"``, ``"adtw"``,
        ``"msm"`` or ``"twe"``.
    band : int, default=-1
        Sakoe-Chiba band width. ``-1`` for full (unconstrained) DTW.
    max_iter : int, default=100
        Iteration limit of the method, at least 1.
    n_init : int, default=1
        Seeded restarts (PAM, kmedoids): restart ``i`` uses the seed
        ``random_state + i`` and the lowest cost is kept, on one distance matrix.
    wdtw_g, adtw_penalty, msm_c, twe_nu, twe_lambda : float
        The variant parameters (each read only by its variant).
    mv_mode : str, default="dependent"
        Multivariate combination mode when ``ndim > 1``: ``"dependent"``
        (DTW_D) or ``"independent"`` (DTW_I).
    missing_strategy : str, default="error"
        NaN handling: ``"error"``, ``"zero_cost"``, ``"arow"`` or
        ``"interpolate"``.
    metric : str, default="l1"
        Pointwise cost, ``"l1"`` or ``"squared_euclidean"`` (Standard DTW and
        DDTW), or ``"precomputed"``: X is a distance matrix.
    batch_size : int, default=-1
        OneBatchPAM's batch size (-1: automatic).
    random_state : int or None, default=None
        The seed; ``None`` is :data:`dtwcpp.DEFAULT_RANDOM_SEED` (42), the
        default of every language.
    device : str or None, default=None
        ``"cpu"``, ``"gpu"``, ``"gpu:N"``, ``"cuda"``/``"cuda:N"`` or
        ``"hpc"``; ``None`` is :func:`dtwcpp.device`. ``"hpc"`` runs the fit on
        a SLURM cluster and sets ``labels_`` only.

    Attributes
    ----------
    labels_, medoid_indices_ : int64 arrays
    cluster_centers_ : list of ndarray, the medoid series (None if precomputed)
    inertia_ : float, the total distance of the series to their medoids
    n_iter_ : int
    """

    def __init__(self, n_clusters=3, method="pam", variant="standard", band=-1,
                 max_iter=100, n_init=1, wdtw_g=0.05, adtw_penalty=1.0,
                 msm_c=1.0, twe_nu=0.001, twe_lambda=1.0, mv_mode="dependent",
                 missing_strategy="error", metric="l1", batch_size=-1,
                 random_state=None, device=None):
        self.n_clusters = n_clusters
        self.method = method
        self.variant = variant
        self.band = band
        self.max_iter = max_iter
        self.n_init = n_init
        self.wdtw_g = wdtw_g
        self.adtw_penalty = adtw_penalty
        self.msm_c = msm_c
        self.twe_nu = twe_nu
        self.twe_lambda = twe_lambda
        self.mv_mode = mv_mode
        self.missing_strategy = missing_strategy
        self.metric = metric
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def _precomputed(self):
        return self.metric == "precomputed"

    def _distance(self):
        """The distance settings, by the names distance.dtw and dtwc_cl take."""
        return {
            "variant": self.variant, "band": self.band, "metric": self.metric,
            "missing_strategy": self.missing_strategy, "wdtw_g": self.wdtw_g,
            "adtw_penalty": self.adtw_penalty, "msm_c": self.msm_c,
            "twe_nu": self.twe_nu, "twe_lambda": self.twe_lambda,
        }

    def _config(self, device):
        """The C++ Config of this estimator; C++ reads and checks every value.
        ``device`` None leaves the CPU (an hpc fit computes remotely)."""
        from dtwcpp import _dtwcpp_core
        config = _dtwcpp_core.Config()
        settings = {
            "n_clusters": self.n_clusters, "method": self.method, "max_iter": self.max_iter,
            "n_init": self.n_init, "batch_size": self.batch_size,
            "seed": DEFAULT_RANDOM_SEED if self.random_state is None else self.random_state,
            "mv_mode": self.mv_mode, **self._distance(),
        }
        if device is not None:
            settings["device"] = device
        if self._precomputed():
            del settings["metric"]  # the distances are given, not computed
        for key, value in settings.items():
            setattr(config, key, value)
        return config

    def fit(self, X, y=None):
        """Fit DTW k-medoids; ``y`` is accepted and ignored."""
        import dtwcpp
        from dtwcpp import _dtwcpp_core, _hpc_remote_device, _resolve_device
        device = self.device if self.device is not None else dtwcpp.device()
        backend, _ = _resolve_device(device)
        # C++ takes and checks every setting before X is read, as dtwc::run does
        # before it reads a file.
        config = self._config(None if backend == "hpc" else device)
        prob = _dtwcpp_core.Problem("dtw_clustering")
        _dtwcpp_core.apply(config, prob)
        if self._precomputed() and config.method not in _READS_THE_MATRIX:
            raise ValueError(
                f"method='{self.method}' requires raw series: it computes the distances "
                "it needs, so with metric='precomputed' choose one that reads the given "
                f"matrix ({', '.join(_READS_THE_MATRIX)}).")
        if self._precomputed():
            matrix = _precomputed_matrix(X, square=True)
            # The series are never read: every pair is in the matrix.
            series = [np.zeros(1) for _ in range(matrix.shape[0])]
        else:
            series = _series_list(X, allow_nan=self.missing_strategy != "error")

        if backend == "hpc":
            from dtwcpp import _hpc
            if self._precomputed():
                raise dtwcpp.InvalidInput(
                    "DTWClustering(device='hpc') clusters raw series; a precomputed "
                    "matrix is clustered locally (device='cpu').")
            self.labels_ = _hpc.cluster_on_hpc(
                [row.tolist() for row in series], self.n_clusters, method=config.method,
                band=self.band, device=_hpc_remote_device(device),
                name=f"dtwc_k{self.n_clusters}", n_init=self.n_init,
                seed=DEFAULT_RANDOM_SEED if self.random_state is None else self.random_state,
                max_iter=self.max_iter, variant=self.variant, wdtw_g=self.wdtw_g,
                adtw_penalty=self.adtw_penalty, msm_c=self.msm_c,
                twe_nu=self.twe_nu, twe_lambda=self.twe_lambda,
                mv_mode=self.mv_mode, missing_strategy=self.missing_strategy,
                metric=self.metric,
            )
            self.medoid_indices_ = None
            self.inertia_ = None
            self.n_iter_ = None
            return self

        prob.set_data([row.tolist() for row in series], [str(i) for i in range(len(series))])
        if self._precomputed():
            prob.set_distance_matrix(matrix)
        result = prob.cluster()

        self.labels_ = result.labels
        self.medoid_indices_ = result.medoid_indices
        self.inertia_ = float(result.total_cost)
        self.n_iter_ = int(result.iterations)
        self.n_samples_fit_ = len(series)
        if self._precomputed():
            self.cluster_centers_ = None
            self.n_features_in_ = self.n_samples_fit_
        else:
            self.cluster_centers_ = [series[i].copy() for i in self.medoid_indices_]
            lengths = {row.size for row in series}
            self.n_features_in_ = lengths.pop() if len(lengths) == 1 else None
            self._fit_distance_ = self._distance()
        return self

    def transform(self, X):
        """Distances from each sample of X to each medoid (M x k)."""
        fitted = "medoid_indices_" if self._precomputed() else "cluster_centers_"
        if getattr(self, fitted, None) is None:
            raise NotFittedError(
                "DTWClustering is not fitted locally; call fit before using this method")
        if self._precomputed():
            matrix = _precomputed_matrix(X, n_train=self.n_samples_fit_)
            return matrix[:, self.medoid_indices_]
        # The distance the medoids were fitted under; medoids set by hand take
        # the current settings.
        settings = getattr(self, "_fit_distance_", None) or self._distance()
        queries = _series_list(X, allow_nan=settings["missing_strategy"] != "error")
        n_features = getattr(self, "n_features_in_", None)
        if n_features is not None and any(row.size != n_features for row in queries):
            raise ValueError(
                f"X has {queries[0].size} features, but DTWClustering is expecting "
                f"{n_features} features as input.")
        return np.array([[distance.dtw(query, center, **settings)
                          for center in self.cluster_centers_] for query in queries],
                        dtype=np.float64).reshape(len(queries), len(self.cluster_centers_))

    def predict(self, X):
        """The nearest medoid of each sample of X."""
        return np.argmin(self.transform(X), axis=1).astype(np.int64, copy=False)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def score(self, X, y=None):
        """Negative total distance of X to its nearest medoids (larger is better).

        The fitted medoids score X; nothing is refitted.
        """
        return -float(np.min(self.transform(X), axis=1).sum())

    def __sklearn_tags__(self):
        """Native sklearn >= 1.6 tags, with a fallback for older versions."""
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:  # sklearn < 1.6 or the dependency-free stub
            return {"requires_y": False, "pairwise": self._precomputed(),
                    "allow_nan": False, "non_deterministic": False}
        tags.input_tags.two_d_array = True
        tags.input_tags.allow_nan = False
        tags.input_tags.pairwise = self._precomputed()
        tags.requires_fit = True
        tags.non_deterministic = False
        return tags

    def _more_tags(self):  # sklearn < 1.6
        return {"pairwise": self._precomputed(), "requires_y": False}
