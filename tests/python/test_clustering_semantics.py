"""Adversarial semantics tests for :class:`dtwcpp.DTWClustering`."""

from types import SimpleNamespace

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _clustering


SQUARED_FIXTURE = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 4.0],
        [0.0, 2.0, 2.0],
    ]
)


def _configured_problem(estimator, series):
    """Build an independent production-Problem oracle for estimator semantics."""
    problem = dtwcpp.Problem("estimator_pair_oracle")
    problem.set_data(
        [np.asarray(sample, dtype=float).tolist() for sample in series],
        [str(i) for i in range(len(series))],
    )
    problem.set_band(estimator.band)
    problem.missing_strategy = {
        "error": dtwcpp.MissingStrategy.Error,
        "zero_cost": dtwcpp.MissingStrategy.ZeroCost,
        "arow": dtwcpp.MissingStrategy.AROW,
        "interpolate": dtwcpp.MissingStrategy.Interpolate,
    }[estimator.missing_strategy]
    params = dtwcpp.DTWVariantParams()
    params.variant = {
        "standard": dtwcpp.DTWVariant.Standard,
        "ddtw": dtwcpp.DTWVariant.DDTW,
        "wdtw": dtwcpp.DTWVariant.WDTW,
        "adtw": dtwcpp.DTWVariant.ADTW,
        "msm": dtwcpp.DTWVariant.MSM,
        "twe": dtwcpp.DTWVariant.TWE,
    }[estimator.variant]
    params.wdtw_g = estimator.wdtw_g
    params.adtw_penalty = estimator.adtw_penalty
    params.msm_c = estimator.msm_c
    params.twe_nu = estimator.twe_nu
    params.twe_lambda = estimator.twe_lambda
    params.mv_mode = (
        _clustering.MVMode.Independent
        if estimator.mv_mode == "independent"
        else _clustering.MVMode.Dependent
    )
    problem.set_variant_params(params)
    return problem


def _configured_problem_distance(estimator, x, y):
    """Return one distance from an independent production-Problem oracle."""
    problem = _configured_problem(estimator, [x, y])
    problem.fill_distance_matrix()
    return problem.dist_by_ind(0, 1)


def test_cpu_squared_metric_controls_training_objective():
    squared = dtwcpp.compute_distance_matrix(
        SQUARED_FIXTURE.tolist(), metric="squared_euclidean", use_pruning=False
    )
    oracle_problem = dtwcpp.Problem("squared_oracle")
    oracle_problem.set_data(
        SQUARED_FIXTURE.tolist(), [str(i) for i in range(len(SQUARED_FIXTURE))]
    )
    oracle_problem.set_distance_matrix(squared)
    oracle = dtwcpp.fast_pam_seeded(
        oracle_problem, 2, dtwcpp.DEFAULT_RANDOM_SEED, 100
    )
    assert oracle.medoid_indices == [4, 1]
    assert oracle.total_cost == 5.0

    estimator = dtwcpp.DTWClustering(
        n_clusters=2, metric="squared_euclidean", device="cpu"
    ).fit(SQUARED_FIXTURE)

    np.testing.assert_array_equal(estimator.medoid_indices_, oracle.medoid_indices)
    np.testing.assert_array_equal(estimator.labels_, oracle.labels)
    assert estimator.inertia_ == oracle.total_cost
    np.testing.assert_array_equal(estimator.predict(SQUARED_FIXTURE), oracle.labels)


def test_default_cpu_l1_keeps_lazy_matrix_path(monkeypatch):
    def unexpected_precompute(*args, **kwargs):
        raise AssertionError("default CPU L1 fit must keep the lazy Problem path")

    monkeypatch.setattr(dtwcpp, "compute_distance_matrix", unexpected_precompute)
    estimator = dtwcpp.DTWClustering(n_clusters=2, device="cpu").fit(
        SQUARED_FIXTURE
    )
    np.testing.assert_array_equal(estimator.medoid_indices_, [4, 2])
    assert estimator.inertia_ == 4.0


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_gpu_precompute_receives_requested_metric(monkeypatch, backend):
    real_compute = dtwcpp.compute_distance_matrix
    cpu_squared = real_compute(
        SQUARED_FIXTURE.tolist(), metric="squared_euclidean",
        device="cpu", use_pruning=False,
    )
    captured = {}

    monkeypatch.setattr(
        dtwcpp, "_resolve_device", lambda device: (backend, 0)
    )

    def compute_spy(series, *, band, metric, device):
        captured.update(band=band, metric=metric, device=device)
        return cpu_squared

    monkeypatch.setattr(dtwcpp, "compute_distance_matrix", compute_spy)
    estimator = dtwcpp.DTWClustering(
        n_clusters=2, metric="squared_euclidean", device=backend
    ).fit(SQUARED_FIXTURE)

    assert captured == {
        "band": -1,
        "metric": "squared_euclidean",
        "device": backend,
    }
    np.testing.assert_array_equal(estimator.medoid_indices_, [4, 1])
    assert estimator.inertia_ == 5.0


@pytest.mark.parametrize(
    ("variant", "params", "x", "center_1", "center_2"),
    [
        (
            "msm",
            {"msm_c": 0.7},
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 3.0],
            [1.0, 1.0, 3.0],
        ),
        (
            "twe",
            {"twe_nu": 0.1, "twe_lambda": 0.8},
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
        ),
    ],
)
def test_predict_uses_msm_and_twe_problem_dispatch(
    variant, params, x, center_1, center_2
):
    estimator = dtwcpp.DTWClustering(n_clusters=2, variant=variant, **params)
    estimator.cluster_centers_ = [
        np.asarray(center_1, dtype=float),
        np.asarray(center_2, dtype=float),
    ]
    oracle_distances = [
        _configured_problem_distance(estimator, x, center)
        for center in estimator.cluster_centers_
    ]
    standard_distances = [
        dtwcpp.distance.standard(x, center)
        for center in estimator.cluster_centers_
    ]
    assert int(np.argmin(oracle_distances)) == 0
    assert int(np.argmin(standard_distances)) == 1

    assert estimator.predict([x]).tolist() == [0]


@pytest.mark.parametrize("missing_strategy", ["zero_cost", "arow", "interpolate"])
def test_predict_uses_configured_missing_strategy(missing_strategy):
    x = [0.0, 1.0, 2.0]
    centers = [[0.0, np.nan, 100.0], [1.0, 1.0, 1.0]]
    estimator = dtwcpp.DTWClustering(
        n_clusters=2, missing_strategy=missing_strategy
    )
    estimator.cluster_centers_ = [np.asarray(c, dtype=float) for c in centers]
    oracle_distances = [
        _configured_problem_distance(estimator, x, center)
        for center in estimator.cluster_centers_
    ]
    assert all(np.isfinite(oracle_distances))
    assert int(np.argmin(oracle_distances)) == 1

    assert estimator.predict([x]).tolist() == [1]


@pytest.mark.parametrize(
    ("kwargs", "series"),
    [
        (
            {},
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
             [10, 10, 10, 10], [10, 10, 10, 11], [10, 10, 11, 11]],
        ),
        (
            {"variant": "ddtw"},
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
             [10, 10, 10, 10], [10, 10, 10, 11], [10, 10, 11, 11]],
        ),
        (
            {"variant": "wdtw", "wdtw_g": 0.2},
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
             [10, 10, 10, 10], [10, 10, 10, 11], [10, 10, 11, 11]],
        ),
        (
            {"variant": "adtw", "adtw_penalty": 0.7},
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
             [10, 10, 10, 10], [10, 10, 10, 11], [10, 10, 11, 11]],
        ),
        (
            {"mv_mode": "independent"},
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
             [10, 10, 10, 10], [10, 10, 10, 11], [10, 10, 11, 11]],
        ),
        (
            {"variant": "msm", "msm_c": 0.7},
            [[0, 0, 0], [0, 3, 3], [1, 1, 3]],
        ),
        (
            {"variant": "twe", "twe_nu": 0.1, "twe_lambda": 0.8},
            [[0, 0, 0], [0, 0, 3], [0, 2, 0]],
        ),
        *[
            (
                {"missing_strategy": strategy},
                [
                    [0, np.nan, 0],
                    [0, 0, 0],
                    [0, 0.1, 0],
                    [10, np.nan, 10],
                    [10, 10, 10],
                    [10, 10.1, 10],
                ],
            )
            for strategy in ("zero_cost", "arow", "interpolate")
        ],
    ],
)
def test_training_predict_matches_configured_problem_nearest(kwargs, series):
    estimator = dtwcpp.DTWClustering(n_clusters=2, **kwargs).fit(series)
    oracle = _configured_problem(estimator, series)
    oracle.fill_distance_matrix()
    expected = []
    for sample_index in range(len(series)):
        distances = [
            oracle.dist_by_ind(sample_index, int(medoid_index))
            for medoid_index in estimator.medoid_indices_
        ]
        expected.append(int(np.argmin(distances)))

    np.testing.assert_array_equal(estimator.labels_, expected)
    np.testing.assert_array_equal(estimator.predict(series), expected)


@pytest.mark.parametrize(
    ("kwargs", "backend", "message"),
    [
        ({"metric": "unknown"}, None, "metric"),
        ({"mv_mode": "sideways"}, None, "mv_mode"),
        ({"variant": "unknown"}, None, "variant"),
        ({"missing_strategy": "unknown"}, None, "missing_strategy"),
        ({"variant": "msm", "metric": "squared_euclidean"}, None, "metric"),
        (
            {"variant": "msm", "missing_strategy": "zero_cost"},
            None,
            "missing_strategy",
        ),
        ({"variant": "msm", "mv_mode": "independent"}, None, "mv_mode"),
        (
            {"missing_strategy": "zero_cost", "mv_mode": "independent"},
            None,
            "mv_mode",
        ),
        (
            {"metric": "squared_euclidean", "mv_mode": "independent"},
            None,
            "metric",
        ),
        (
            {"metric": "squared_euclidean", "missing_strategy": "arow"},
            None,
            "metric",
        ),
        ({"device": "cuda", "missing_strategy": "zero_cost"}, "cuda", "missing"),
        ({"device": "cuda", "mv_mode": "independent"}, "cuda", "mv_mode"),
        ({"device": "metal", "missing_strategy": "arow"}, "metal", "missing"),
        ({"device": "metal", "mv_mode": "independent"}, "metal", "mv_mode"),
    ],
)
def test_invalid_semantics_fail_before_distance_compute(
    monkeypatch, kwargs, backend, message
):
    if backend is not None:
        monkeypatch.setattr(
            dtwcpp, "_resolve_device", lambda device: (backend, 0)
        )

    def unexpected_compute(*args, **kwargs):
        raise AssertionError("distance computation began before validation")

    monkeypatch.setattr(dtwcpp, "compute_distance_matrix", unexpected_compute)
    monkeypatch.setattr(_clustering, "fast_pam_seeded", unexpected_compute)

    with pytest.raises(ValueError, match=message):
        dtwcpp.DTWClustering(n_clusters=2, **kwargs).fit(SQUARED_FIXTURE)


def test_all_nonfinite_restart_results_raise_numeric_error(monkeypatch):
    costs = iter([np.inf, np.nan, -np.inf])

    def fake_fast_pam(*args, **kwargs):
        return SimpleNamespace(
            total_cost=next(costs), labels=[0] * len(SQUARED_FIXTURE),
            medoid_indices=[0, 1], iterations=1,
        )

    monkeypatch.setattr(_clustering, "fast_pam_seeded", fake_fast_pam)
    with pytest.raises(FloatingPointError, match="3.*non-finite"):
        dtwcpp.DTWClustering(n_clusters=2, n_init=3).fit(SQUARED_FIXTURE)


def test_first_finite_restart_wins_strict_tie_after_nonfinite(monkeypatch):
    results = iter(
        [
            SimpleNamespace(
                total_cost=np.nan, labels=[0] * len(SQUARED_FIXTURE),
                medoid_indices=[0, 1], iterations=1,
            ),
            SimpleNamespace(
                total_cost=5.0, labels=[1] * len(SQUARED_FIXTURE),
                medoid_indices=[1, 2], iterations=2,
            ),
            SimpleNamespace(
                total_cost=5.0, labels=[2] * len(SQUARED_FIXTURE),
                medoid_indices=[2, 3], iterations=3,
            ),
        ]
    )
    monkeypatch.setattr(
        _clustering, "fast_pam_seeded", lambda *args, **kwargs: next(results)
    )

    estimator = dtwcpp.DTWClustering(n_clusters=2, n_init=3).fit(
        SQUARED_FIXTURE
    )
    np.testing.assert_array_equal(estimator.medoid_indices_, [1, 2])
    np.testing.assert_array_equal(estimator.labels_, [1] * len(SQUARED_FIXTURE))
    assert estimator.inertia_ == 5.0
    assert estimator.n_iter_ == 2
