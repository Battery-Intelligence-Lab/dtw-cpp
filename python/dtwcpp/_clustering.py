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
        DTW variant. One of ``"standard"``, ``"ddtw"``, ``"wdtw"``,
        ``"adtw"``, ``"msm"``, or ``"twe"``.
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

    @staticmethod
    def _normalize_choice(name, value, choices):
        """Return a case-normalized public choice or reject it loudly."""
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        normalized = value.lower()
        if normalized not in choices:
            raise ValueError(
                f"Unknown {name} '{value}'. Expected one of: {sorted(choices)}"
            )
        return normalized

    def _validate_semantics(self, backend=None):
        """Normalize one executable distance contract before any computation."""
        variant = self._normalize_choice(
            "variant", self.variant,
            {"standard", "ddtw", "wdtw", "adtw", "msm", "twe"},
        )
        missing_strategy = self._normalize_choice(
            "missing_strategy", self.missing_strategy,
            {"error", "zero_cost", "arow", "interpolate"},
        )
        metric = self._normalize_choice(
            "metric", self.metric, {"l1", "squared_euclidean"},
        )
        mv_mode = self._normalize_choice(
            "mv_mode", self.mv_mode, {"dependent", "independent"},
        )

        if variant != "standard" and metric != "l1":
            raise ValueError(
                f"metric='{metric}' is implemented only for variant='standard'; "
                f"variant='{variant}' has its intrinsic L1 cost"
            )
        if variant != "standard" and missing_strategy != "error":
            raise ValueError(
                f"missing_strategy='{missing_strategy}' cannot be combined with "
                f"variant='{variant}'; missing-data dispatch would replace the "
                "requested variant"
            )
        if mv_mode == "independent":
            if variant != "standard":
                raise ValueError(
                    "mv_mode='independent' requires variant='standard'"
                )
            if missing_strategy != "error":
                raise ValueError(
                    "mv_mode='independent' requires missing_strategy='error'"
                )
            if metric != "l1":
                raise ValueError(
                    "metric='squared_euclidean' is not implemented for "
                    "mv_mode='independent'"
                )
        if missing_strategy != "error" and metric != "l1":
            raise ValueError(
                "metric='squared_euclidean' is not implemented with "
                f"missing_strategy='{missing_strategy}'"
            )

        if backend in ("cuda", "metal"):
            if variant != "standard":
                raise ValueError(
                    f"device='{backend}' only supports variant='standard', "
                    f"got variant='{variant}'"
                )
            if missing_strategy != "error":
                raise ValueError(
                    f"device='{backend}' does not support "
                    f"missing_strategy='{missing_strategy}'"
                )
            if mv_mode != "dependent":
                raise ValueError(
                    f"device='{backend}' does not support "
                    "mv_mode='independent'"
                )

        return {
            "variant": variant,
            "missing_strategy": missing_strategy,
            "metric": metric,
            "mv_mode": mv_mode,
        }

    def _variant_enum(self, variant=None):
        """Map string variant name to C++ DTWVariant enum."""
        mapping = {
            "standard": DTWVariant.Standard,
            "ddtw": DTWVariant.DDTW,
            "wdtw": DTWVariant.WDTW,
            "adtw": DTWVariant.ADTW,
            "msm": DTWVariant.MSM,
            "twe": DTWVariant.TWE,
        }
        key = variant or self._normalize_choice("variant", self.variant, mapping)
        return mapping[key]

    def _missing_strategy_enum(self, missing_strategy=None):
        """Map string missing_strategy name to C++ MissingStrategy enum."""
        mapping = {
            "error": MissingStrategy.Error,
            "zero_cost": MissingStrategy.ZeroCost,
            "arow": MissingStrategy.AROW,
            "interpolate": MissingStrategy.Interpolate,
        }
        key = missing_strategy or self._normalize_choice(
            "missing_strategy", self.missing_strategy, mapping,
        )
        return mapping[key]

    def _dtw_fn(self, x, y, semantics=None):
        """Compute DTW distance between two series using current variant."""
        semantics = semantics or self._validate_semantics()
        variant = semantics["variant"]
        # All raw distance bindings take zero-copy float64 ndarrays (§2.6).
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)

        # MSM/TWE and missing-data preprocessing have no complete free-function
        # surface. Route those cases through the same production Problem
        # dispatcher as fit(). Error+NaN also takes this path so its documented
        # pre-scan is not bypassed by predict(). Finite Standard-L1 stays on the
        # existing allocation-free fast path.
        use_problem = (
            variant in ("msm", "twe")
            or semantics["missing_strategy"] != "error"
            or semantics["mv_mode"] == "independent"
            or np.isnan(xa).any()
            or np.isnan(ya).any()
        )
        if use_problem:
            problem = self._build_problem(
                [xa.tolist(), ya.tolist()], semantics=semantics,
            )
            problem.fill_distance_matrix()
            return problem.dist_by_ind(0, 1)

        if variant == "ddtw":
            return ddtw_distance(xa, ya, self.band)
        if variant == "wdtw":
            return wdtw_distance(xa, ya, self.band, self.wdtw_g)
        if variant == "adtw":
            return adtw_distance(xa, ya, self.band, self.adtw_penalty)
        return dtw_distance(xa, ya, self.band, semantics["metric"])

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

    def _build_problem(self, series, semantics=None):
        """Construct a C++ Problem object with current settings."""
        semantics = semantics or self._validate_semantics()
        names = [str(i) for i in range(len(series))]
        prob = Problem("dtw_clustering")
        prob.set_data(series, names)
        prob.set_band(self.band)
        prob.missing_strategy = self._missing_strategy_enum(
            semantics["missing_strategy"]
        )

        vp = DTWVariantParams()
        vp.variant = self._variant_enum(semantics["variant"])
        vp.wdtw_g = self.wdtw_g
        vp.adtw_penalty = self.adtw_penalty
        vp.msm_c = self.msm_c
        vp.twe_nu = self.twe_nu
        vp.twe_lambda = self.twe_lambda
        vp.mv_mode = (MVMode.Independent if semantics["mv_mode"] == "independent"
                      else MVMode.Dependent)
        # Rebind after the raw missing-strategy field reaches Problem so fit and
        # pairwise predict share the exact same production dispatcher.
        prob.set_variant_params(vp)
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
        semantics = self._validate_semantics()
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
        semantics = self._validate_semantics(backend)

        # 'hpc' offloads the entire job to a SLURM cluster and returns labels.
        if backend == "hpc":
            from dtwcpp import _hpc
            self.labels_ = _hpc.cluster_on_hpc(
                series, self.n_clusters, method="pam", band=self.band,
                name=f"dtwc_k{self.n_clusters}", n_init=restart_count,
                seed=DEFAULT_RANDOM_SEED, max_iter=self.max_iter,
                variant=semantics["variant"], wdtw_g=self.wdtw_g,
                adtw_penalty=self.adtw_penalty, msm_c=self.msm_c,
                twe_nu=self.twe_nu, twe_lambda=self.twe_lambda,
                mv_mode=semantics["mv_mode"],
                missing_strategy=semantics["missing_strategy"],
                metric=semantics["metric"],
            )
            self.medoid_indices_ = None
            self.inertia_ = None
            self.n_iter_ = None
            return self

        # GPU backends require a precomputed matrix. Standard squared DTW also
        # requires one on CPU because Problem's lazy matrix is intrinsically L1.
        # Preserve the default CPU Standard-L1 lazy path and memory profile.
        dm_precomputed = None
        if backend in ("cuda", "metal") or semantics["metric"] != "l1":
            dm_precomputed = compute_distance_matrix(
                series, band=self.band, metric=semantics["metric"],
                device=eff_device,
            )

        best_result = None
        best_cost = float("inf")
        nonfinite_restarts = []

        for restart in range(restart_count):
            prob = self._build_problem(series, semantics=semantics)
            if dm_precomputed is not None:
                prob.set_distance_matrix(dm_precomputed)

            result = fast_pam_seeded(
                prob,
                self.n_clusters,
                DEFAULT_RANDOM_SEED + restart,
                self.max_iter,
            )
            result_cost = float(result.total_cost)
            if not np.isfinite(result_cost):
                nonfinite_restarts.append(
                    (restart, DEFAULT_RANDOM_SEED + restart, result_cost)
                )
                continue
            if best_result is None or result_cost < best_cost:
                best_cost = result_cost
                best_result = result

        if best_result is None:
            details = ", ".join(
                f"restart {restart} (seed {seed}): {cost}"
                for restart, seed, cost in nonfinite_restarts
            )
            raise FloatingPointError(
                f"All {restart_count} DTWClustering restarts returned non-finite "
                f"objectives ({details}). Check the input for NaN/Inf values and "
                "the selected distance parameters."
            )

        self.labels_ = np.array(best_result.labels)
        self.medoid_indices_ = np.array(best_result.medoid_indices)
        self.inertia_ = best_result.total_cost
        self.n_iter_ = best_result.iterations
        self._fit_semantics_ = semantics
        self._fit_backend_ = backend

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

        semantics = getattr(self, "_fit_semantics_", None)
        if semantics is None:
            semantics = self._validate_semantics()
        else:
            # Reject post-fit mutations that would be incompatible with the
            # backend used to construct the fitted medoids.
            self._validate_semantics(getattr(self, "_fit_backend_", None))

        series = self._prepare_data(X)
        labels = np.empty(len(series), dtype=int)
        for i, s in enumerate(series):
            dists = [
                self._dtw_fn(s, c, semantics=semantics)
                for c in self.cluster_centers_
            ]
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
