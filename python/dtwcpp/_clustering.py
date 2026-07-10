"""
@file _clustering.py
@brief sklearn-compatible DTW clustering.
@author Volkan Kumtepeli
"""
import numpy as np
from dtwcpp._dtwcpp_core import (
    DEFAULT_RANDOM_SEED, Problem, fast_pam_seeded, silhouette,
    DTWVariant, DTWVariantParams,
    MVMode, MissingStrategy,
    dtw_distance, ddtw_distance, wdtw_distance, adtw_distance,
    data_from_arrow_c_array,
)

try:
    from sklearn.base import BaseEstimator, ClusterMixin
    _HAS_SKLEARN = True
except ImportError:
    # sklearn is optional -- provide stub base classes
    class BaseEstimator:
        """Minimal sklearn-compatible base when sklearn is not installed."""
        def get_params(self, deep=True):
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class ClusterMixin:
        """Minimal sklearn ClusterMixin stub."""
        pass

    _HAS_SKLEARN = False


class DTWClustering(BaseEstimator, ClusterMixin):
    """K-medoids clustering with DTW distance.

    Implements FastPAM (Schubert & Rousseeuw 2021) with configurable DTW
    variants, exposed through an sklearn-compatible API.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    variant : str, default="standard"
        DTW variant. One of ``"standard"``, ``"ddtw"``, ``"wdtw"``, ``"adtw"``.
    band : int, default=-1
        Sakoe-Chiba band width. ``-1`` for full (unconstrained) DTW.
    max_iter : int, default=100
        Maximum number of FastPAM iterations.
    n_init : int, default=1
        Number of deterministic random restarts (best result kept). Restart
        ``i`` uses the invocation-local seed ``DEFAULT_RANDOM_SEED + i``.
    wdtw_g : float, default=0.05
        Logistic weight steepness for WDTW (ignored unless ``variant="wdtw"``).
    adtw_penalty : float, default=1.0
        Non-diagonal step penalty for ADTW (ignored unless ``variant="adtw"``).
    mv_mode : str, default="dependent"
        Multivariate combination mode when ``ndim > 1`` (Shokoohi-Yekta et al.,
        DMKD 2017; ignored for univariate series). ``"dependent"`` (DTW_D: one
        shared warping path) or ``"independent"`` (DTW_I: per-channel DTW summed).
        ``"independent"`` requires ``variant="standard"`` and
        ``missing_strategy="error"`` in this release.
    missing_strategy : str, default="error"
        How to handle NaN values in time series. One of ``"error"`` (throw),
        ``"zero_cost"`` (NaN pairs contribute zero cost), ``"arow"``
        (diagonal-only alignment), or ``"interpolate"`` (linear interpolation).
    metric : str, default="l1"
        Pointwise cost metric for the ``"standard"`` variant: ``"l1"`` (default)
        or ``"squared_euclidean"``. Matches the ``metric`` argument of the
        distance free functions (api-contract-2.0.md §1.5/§2.6).
    device : str or None, default=None
        Computation device. ``"cpu"``, ``"gpu"`` (CUDA or Metal, requiring a
        live GPU), ``"cuda"``/``"cuda:N"``, or ``"hpc"`` (offload the
        whole clustering job to a SLURM cluster). ``None`` uses the global
        default set via :func:`dtwcpp.device` (itself ``"cpu"`` unless changed).
        With ``"hpc"``, only ``labels_`` is populated (the cluster computes
        remotely); ``predict`` is unavailable until a local fit is run.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels assigned to each input series.
    medoid_indices_ : ndarray of shape (n_clusters,)
        Indices of the medoid series in the training data.
    cluster_centers_ : list of ndarray
        The medoid time series (one per cluster).
    inertia_ : float
        Total within-cluster cost (sum of DTW distances to medoids).
    n_iter_ : int
        Number of iterations run by FastPAM.
    """

    def __init__(self, n_clusters=3, variant="standard", band=-1,
                 max_iter=100, n_init=1, wdtw_g=0.05, adtw_penalty=1.0,
                 msm_c=1.0, twe_nu=0.001, twe_lambda=1.0, mv_mode="dependent",
                 missing_strategy="error", metric="l1", device=None):
        self.n_clusters = n_clusters
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
        self.device = device

    def _variant_enum(self):
        """Map string variant name to C++ DTWVariant enum."""
        mapping = {
            "standard": DTWVariant.Standard,
            "ddtw": DTWVariant.DDTW,
            "wdtw": DTWVariant.WDTW,
            "adtw": DTWVariant.ADTW,
            "msm": DTWVariant.MSM,
            "twe": DTWVariant.TWE,
        }
        v = mapping.get(self.variant)
        if v is None:
            raise ValueError(
                f"Unknown DTW variant '{self.variant}'. "
                f"Expected one of: {list(mapping.keys())}"
            )
        return v

    def _missing_strategy_enum(self):
        """Map string missing_strategy name to C++ MissingStrategy enum."""
        mapping = {
            "error": MissingStrategy.Error,
            "zero_cost": MissingStrategy.ZeroCost,
            "arow": MissingStrategy.AROW,
            "interpolate": MissingStrategy.Interpolate,
        }
        s = mapping.get(self.missing_strategy)
        if s is None:
            raise ValueError(
                f"Unknown missing_strategy '{self.missing_strategy}'. "
                f"Expected one of: {list(mapping.keys())}"
            )
        return s

    def _dtw_fn(self, x, y):
        """Compute DTW distance between two series using current variant."""
        v = self.variant
        # All raw distance bindings take zero-copy float64 ndarrays (§2.6).
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if v == "ddtw":
            return ddtw_distance(xa, ya, self.band)
        if v == "wdtw":
            return wdtw_distance(xa, ya, self.band, self.wdtw_g)
        if v == "adtw":
            return adtw_distance(xa, ya, self.band, self.adtw_penalty)
        return dtw_distance(xa, ya, self.band, self.metric)

    @staticmethod
    def _prepare_data(X):
        """Convert X to list-of-lists for the C++ Problem class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps), list of 1-D arrays,
            or any Arrow C Data interface source implementing __arrow_c_array__
            (polars/DuckDB/pyarrow/pandas -- read zero-copy, without pyarrow).
        """
        if hasattr(X, "__arrow_c_array__") or hasattr(X, "__arrow_c_stream__"):
            # nanoarrow reads the Arrow buffers directly (no pyarrow) into a C++
            # Data; hand the series back as numpy arrays for the fit pipeline.
            data = data_from_arrow_c_array(X)
            return [np.asarray(s, dtype=np.float64) for s in data.p_vec]
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return [list(row) for row in X]
        elif isinstance(X, list):
            return [list(s) for s in X]
        else:
            raise ValueError(
                "X must be a 2D numpy array, list of 1D arrays, or an Arrow C "
                "Data interface source (__arrow_c_array__)")

    def _build_problem(self, series):
        """Construct a C++ Problem object with current settings."""
        names = [str(i) for i in range(len(series))]
        prob = Problem("dtw_clustering")
        prob.set_data(series, names)
        prob.band = self.band
        prob.missing_strategy = self._missing_strategy_enum()

        vp = DTWVariantParams()
        vp.variant = self._variant_enum()
        vp.wdtw_g = self.wdtw_g
        vp.adtw_penalty = self.adtw_penalty
        vp.msm_c = self.msm_c
        vp.twe_nu = self.twe_nu
        vp.twe_lambda = self.twe_lambda
        vp.mv_mode = (MVMode.Independent if str(self.mv_mode).lower() == "independent"
                      else MVMode.Dependent)
        prob.variant_params = vp
        return prob

    def fit(self, X, y=None):
        """Fit DTW k-medoids clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            Training time series.
        y : ignored

        Returns
        -------
        self
        """
        series = self._prepare_data(X)

        if isinstance(self.n_init, (bool, np.bool_)) or not isinstance(
            self.n_init, (int, np.integer)
        ):
            raise TypeError("n_init must be an integer")
        restart_count = int(self.n_init)
        if restart_count < 1:
            raise ValueError("n_init must be at least 1")
        max_seed = (1 << 64) - 1
        if restart_count - 1 > max_seed - DEFAULT_RANDOM_SEED:
            raise ValueError("n_init is too large for distinct uint64 restart seeds")

        from dtwcpp import compute_distance_matrix, _resolve_device, get_device
        eff_device = self.device if self.device is not None else get_device()
        backend, _ = _resolve_device(eff_device)

        # 'hpc' offloads the entire job to a SLURM cluster and returns labels.
        if backend == "hpc":
            from dtwcpp import _hpc
            self.labels_ = _hpc.cluster_on_hpc(
                series, self.n_clusters, method="pam", band=self.band,
                name=f"dtwc_k{self.n_clusters}", n_init=restart_count,
                seed=DEFAULT_RANDOM_SEED,
            )
            self.medoid_indices_ = None
            self.inertia_ = None
            self.n_iter_ = None
            return self

        # Pre-compute GPU distance matrix once (shared across n_init restarts)
        dm_precomputed = None
        if backend in ("cuda", "metal"):
            if self.variant != "standard":
                raise ValueError(
                    f"device='{backend}' only supports variant='standard', "
                    f"got variant='{self.variant}'"
                )
            dm_precomputed = compute_distance_matrix(
                series, band=self.band, device=eff_device
            )

        best_result = None
        best_cost = float("inf")

        for restart in range(restart_count):
            prob = self._build_problem(series)
            if dm_precomputed is not None:
                prob.set_distance_matrix(dm_precomputed)

            result = fast_pam_seeded(
                prob,
                self.n_clusters,
                DEFAULT_RANDOM_SEED + restart,
                self.max_iter,
            )
            if result.total_cost < best_cost:
                best_cost = result.total_cost
                best_result = result

        self.labels_ = np.array(best_result.labels)
        self.medoid_indices_ = np.array(best_result.medoid_indices)
        self.inertia_ = best_result.total_cost
        self.n_iter_ = best_result.iterations

        # Store medoid series for predict()
        self.cluster_centers_ = [np.array(series[i])
                                 for i in best_result.medoid_indices]
        return self

    def predict(self, X):
        """Assign each series in X to the nearest medoid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        series = self._prepare_data(X)
        labels = np.empty(len(series), dtype=int)
        for i, s in enumerate(series):
            dists = [self._dtw_fn(s, c) for c in self.cluster_centers_]
            labels[i] = int(np.argmin(dists))
        return labels

    def fit_predict(self, X, y=None):
        """Fit and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
        y : ignored

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.fit(X, y).labels_

    def score(self, X, y=None):
        """Return the negative inertia (for sklearn compatibility).

        Higher is better, so we negate the total within-cluster cost.
        """
        self.fit(X, y)
        return -self.inertia_
