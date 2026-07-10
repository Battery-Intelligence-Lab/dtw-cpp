"""scikit-learn estimator for DTWC++ k-medoids clustering.

``DTWCKMedoids`` supports both raw time-series input and sklearn's
``metric='precomputed'`` convention. In precomputed mode, ``fit`` consumes an
N-by-N training matrix while ``predict``/``transform`` consume an M-by-N matrix
of distances from new samples to the fitted training samples.
"""

from __future__ import annotations

import numbers
import numpy as np

try:  # scikit-learn is an optional dependency of the base wheel.
    from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
    from sklearn.exceptions import NotFittedError
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised in minimal-wheel smoke tests
    _HAS_SKLEARN = False

    class BaseEstimator:
        def get_params(self, deep=True):
            return {
                key: value for key, value in self.__dict__.items()
                if not key.endswith("_") and not key.startswith("_")
            }

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


def _series_list(X):
    """Normalize a rectangular or ragged raw-series collection."""
    if type(X).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse input is not supported; provide a dense array.")
    if isinstance(X, np.ndarray):
        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")
        if X.ndim != 2:
            raise ValueError(
                "Expected a 2D raw time-series array. Reshape your data to "
                "(n_samples, n_timesteps)."
            )
        if X.shape[0] == 0:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={X.shape}) while a minimum of 1 is required."
            )
        if X.shape[1] == 0:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
            )
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
            return _series_list(np.asarray(X))
        except (TypeError, ValueError) as exc:
            raise ValueError("raw input must be an array-like collection of series") from exc
    if not values:
        raise ValueError("X must contain at least one series")
    for row in values:
        if row.ndim != 1 or row.size == 0:
            raise ValueError("each time series must be a non-empty one-dimensional array")
        if not np.isfinite(row).all():
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
            f"precomputed query matrix must have {n_train} columns; got {matrix.shape[1]}"
        )
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


class DTWCKMedoids(ClusterMixin, TransformerMixin, BaseEstimator):
    """DTW k-medoids estimator with raw and precomputed distance modes.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    metric : {"dtw", "precomputed"}, default="dtw"
        Raw series use DTW. Precomputed training input must be square; query
        input to :meth:`predict` or :meth:`transform` must have one column per
        fitted training sample.
    method : {"pam", "onebatch", "clara"}, default="pam"
        K-medoids optimizer. Precomputed mode supports ``pam`` and ``clara``;
        OneBatchPAM requires raw series because it owns its distance schedule.
    band : int, default=-1
        Sakoe-Chiba band for raw DTW.
    max_iter : int, default=100
        Optimizer iteration limit.
    batch_size : int, default=-1
        Fixed batch size for ``method='onebatch'``.
    random_state : int or None, default=None
        Seed for sampling-based methods. ``None`` uses the cross-language
        Tier-1 default, :data:`dtwcpp.DEFAULT_RANDOM_SEED` (42).
    """

    def __init__(self, n_clusters=3, *, metric="dtw", method="pam", band=-1,
                 max_iter=100, batch_size=-1, random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.band = band
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    def _validate_parameters(self, n_samples):
        if not isinstance(self.n_clusters, numbers.Integral):
            raise TypeError("n_clusters must be an integer")
        if self.n_clusters <= 0 or self.n_clusters > n_samples:
            raise ValueError("n_clusters must be in [1, n_samples]")
        if self.metric not in ("dtw", "precomputed"):
            raise ValueError("metric must be 'dtw' or 'precomputed'")
        if self.method not in ("pam", "onebatch", "clara"):
            raise ValueError("method must be 'pam', 'onebatch', or 'clara'")
        if self.metric == "precomputed" and self.method == "onebatch":
            raise ValueError(
                "method='onebatch' requires raw series; its fixed-batch distance "
                "schedule cannot be reconstructed from an arbitrary query matrix"
            )
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if self.random_state is not None and not isinstance(self.random_state, numbers.Integral):
            raise TypeError("random_state must be an integer or None")

    @staticmethod
    def _problem(series, distance_matrix=None):
        from dtwcpp import Problem
        problem = Problem("sklearn_kmedoids")
        problem.set_data([row.tolist() for row in series],
                         [str(i) for i in range(len(series))])
        if distance_matrix is not None:
            problem.set_distance_matrix(distance_matrix)
        return problem

    def fit(self, X, y=None):
        """Fit the estimator; ``y`` is accepted and ignored."""
        del y
        from dtwcpp import (
            DEFAULT_RANDOM_SEED,
            fast_pam_seeded,
            fast_clara,
            one_batch_pam,
        )

        if self.metric == "precomputed":
            matrix = _precomputed_matrix(X, square=True)
            n_samples = matrix.shape[0]
            self._validate_parameters(n_samples)
            # Series values are never consulted because every required pair is
            # already populated in the injected matrix.
            series = [np.asarray([0.0], dtype=np.float64) for _ in range(n_samples)]
            problem = self._problem(series, matrix)
            self._fit_distance_matrix_ = matrix
            self._fit_series_ = None
            self.n_features_in_ = n_samples
        else:
            series = _series_list(X)
            n_samples = len(series)
            self._validate_parameters(n_samples)
            problem = self._problem(series)
            problem.band = int(self.band)
            self._fit_series_ = [row.copy() for row in series]
            self._fit_distance_matrix_ = None
            self.n_features_in_ = int(series[0].size) if all(
                row.size == series[0].size for row in series) else None

        seed = (
            DEFAULT_RANDOM_SEED
            if self.random_state is None
            else int(self.random_state)
        )
        if self.method == "onebatch":
            result = one_batch_pam(
                problem, int(self.n_clusters), batch_size=int(self.batch_size),
                max_iter=int(self.max_iter), seed=seed,
            )
        elif self.method == "clara":
            result = fast_clara(
                problem, int(self.n_clusters), max_iter=int(self.max_iter), seed=seed,
            )
        else:
            result = fast_pam_seeded(
                problem, int(self.n_clusters), seed, int(self.max_iter)
            )

        self.labels_ = np.asarray(result.labels, dtype=np.intp)
        self.medoid_indices_ = np.asarray(result.medoid_indices, dtype=np.intp)
        self.inertia_ = float(result.total_cost)
        self.n_iter_ = int(result.iterations)
        self.n_samples_fit_ = n_samples
        self.cluster_centers_ = None if self._fit_series_ is None else [
            self._fit_series_[index].copy() for index in self.medoid_indices_
        ]
        return self

    def _check_fitted(self):
        if not hasattr(self, "medoid_indices_"):
            raise NotFittedError("DTWCKMedoids is not fitted; call fit before using this method")

    def transform(self, X):
        """Return distances from each query sample to each fitted medoid."""
        self._check_fitted()
        if self.metric == "precomputed":
            matrix = _precomputed_matrix(X, n_train=self.n_samples_fit_)
            return matrix[:, self.medoid_indices_]

        from dtwcpp._dtwcpp_core import dtw_distance
        queries = _series_list(X)
        if self.n_features_in_ is not None and any(
            row.size != self.n_features_in_ for row in queries
        ):
            observed = queries[0].size
            raise ValueError(
                f"X has {observed} features, but DTWCKMedoids is expecting "
                f"{self.n_features_in_} features as input."
            )
        transformed = np.empty((len(queries), self.n_clusters), dtype=np.float64)
        for i, query in enumerate(queries):
            for cluster, center in enumerate(self.cluster_centers_):
                transformed[i, cluster] = dtw_distance(query, center, int(self.band))
        return transformed

    def predict(self, X):
        """Assign raw series or a precomputed M-by-N query matrix."""
        return np.argmin(self.transform(X), axis=1).astype(np.intp, copy=False)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def score(self, X, y=None):
        """Return negative inertia on X (larger is better)."""
        del y
        return -float(np.min(self.transform(X), axis=1).sum())

    def __sklearn_tags__(self):
        """Native sklearn >=1.6 tags, with a graceful older-version fallback."""
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:  # sklearn <1.6 or the dependency-free stub
            return {
                "requires_y": False,
                "pairwise": self.metric == "precomputed",
                "allow_nan": False,
                "non_deterministic": False,
            }
        tags.input_tags.two_d_array = True
        tags.input_tags.allow_nan = False
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.requires_fit = True
        tags.non_deterministic = False
        return tags

    def _more_tags(self):  # sklearn <1.6
        return {"pairwise": self.metric == "precomputed", "requires_y": False}


__all__ = ["DTWCKMedoids"]
