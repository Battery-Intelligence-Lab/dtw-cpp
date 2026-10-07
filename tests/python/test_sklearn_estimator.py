"""DTWClustering: the scikit-learn contract (raw series and metric="precomputed"),
and the distance it fits and predicts with, against a Problem configured alike."""

import numpy as np
import pytest

import dtwcpp
from dtwcpp import DTWClustering
from dtwcpp._dtwcpp_core import MVMode


def separated_series():
    low = [np.array([0.0, 1.0, 2.0, 1.0]) + i * 0.01 for i in range(5)]
    high = [np.array([50.0, 51.0, 52.0, 51.0]) + i * 0.01 for i in range(5)]
    return np.asarray(low + high)


def test_score_reads_the_fitted_medoids_and_never_refits():
    X = separated_series()
    estimator = DTWClustering(n_clusters=2).fit(X)
    medoids = estimator.medoid_indices_.copy()
    other = X[:3] + 0.5
    assert estimator.score(other) == pytest.approx(
        -np.min(estimator.transform(other), axis=1).sum())
    np.testing.assert_array_equal(estimator.medoid_indices_, medoids)


def test_default_seed_matches_tier1_seed_contract():
    X = (np.array([0.0, 0.01, -0.02, 0.03])[None, :]
         + np.arange(8.0)[:, None])
    tier1 = dtwcpp.cluster(X, k=3, method="pam")
    estimator = DTWClustering(n_clusters=3).fit(X)

    assert dtwcpp.DEFAULT_RANDOM_SEED == 42
    np.testing.assert_array_equal(estimator.medoid_indices_, tier1.medoids)
    np.testing.assert_array_equal(estimator.labels_, tier1.labels)
    assert estimator.inertia_ == tier1.cost == 24.0


def test_precomputed_fit_and_rectangular_query_both_work():
    X = separated_series()
    D = dtwcpp.compute_distance_matrix(X.tolist())
    estimator = DTWClustering(n_clusters=2, metric="precomputed").fit(D)

    transformed = estimator.transform(D)
    assert transformed.shape == (10, 2)
    np.testing.assert_array_equal(estimator.predict(D), estimator.labels_)

    query = D[:3, :]  # M-by-N: distances from 3 queries to all fit samples
    assert estimator.transform(query).shape == (3, 2)
    np.testing.assert_array_equal(
        estimator.transform(query), query[:, estimator.medoid_indices_]
    )


def test_sklearn_common_estimator_contract():
    pytest.importorskip("sklearn")
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(DTWClustering(n_clusters=2))


def test_precomputed_native_pairwise_tag():
    pytest.importorskip("sklearn", minversion="1.6")
    from sklearn.utils import get_tags

    precomputed = DTWClustering(n_clusters=2, metric="precomputed")
    raw = DTWClustering(n_clusters=2)

    assert precomputed.__sklearn_tags__().input_tags.pairwise is True
    assert get_tags(precomputed).input_tags.pairwise is True
    assert raw.__sklearn_tags__().input_tags.pairwise is False
    assert get_tags(raw).input_tags.pairwise is False


def test_grid_search_slices_precomputed_matrix_on_both_axes():
    pytest.importorskip("sklearn", minversion="1.6")
    from sklearn.model_selection import GridSearchCV, KFold

    positions = np.array([0.0, 0.2, 0.4, 10.0, 10.2, 10.4])
    distances = np.abs(positions[:, None] - positions[None, :])
    search = GridSearchCV(
        DTWClustering(metric="precomputed", random_state=7),
        {"n_clusters": [2]},
        cv=KFold(n_splits=3, shuffle=True, random_state=11),
        error_score="raise",
    )

    search.fit(distances)

    assert search.best_estimator_.n_samples_fit_ == distances.shape[0]
    assert np.isfinite(search.best_score_)


@pytest.mark.parametrize(
    "matrix,message",
    [
        (np.ones((2, 3)), "square"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "zero diagonal"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), "non-negative"),
    ],
)
def test_precomputed_validation(matrix, message):
    with pytest.raises(ValueError, match=message):
        DTWClustering(n_clusters=1, metric="precomputed").fit(matrix)


def test_precomputed_onebatch_is_rejected_loudly():
    with pytest.raises(ValueError, match="requires raw series"):
        DTWClustering(
            n_clusters=1, metric="precomputed", method="onebatch"
        ).fit(np.zeros((2, 2)))


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
        MVMode.Independent if estimator.mv_mode == "independent" else MVMode.Dependent
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
        SQUARED_FIXTURE.tolist(), metric="squared_euclidean"
    )
    oracle_problem = dtwcpp.Problem("squared_oracle")
    oracle_problem.set_data(
        SQUARED_FIXTURE.tolist(), [str(i) for i in range(len(SQUARED_FIXTURE))]
    )
    oracle_problem.set_distance_matrix(squared)
    oracle = dtwcpp.fast_pam(
        oracle_problem, 2, max_iter=100, seed=dtwcpp.DEFAULT_RANDOM_SEED
    )
    assert oracle.medoid_indices.tolist() == [4, 1]
    assert oracle.total_cost == 5.0

    estimator = dtwcpp.DTWClustering(
        n_clusters=2, metric="squared_euclidean", device="cpu"
    ).fit(SQUARED_FIXTURE)

    np.testing.assert_array_equal(estimator.medoid_indices_, oracle.medoid_indices)
    np.testing.assert_array_equal(estimator.labels_, oracle.labels)
    assert estimator.inertia_ == oracle.total_cost
    np.testing.assert_array_equal(estimator.predict(SQUARED_FIXTURE), oracle.labels)


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
def test_predict_uses_msm_and_twe(variant, params, x, center_1, center_2):
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
        dtwcpp.distance.dtw(x, center)
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
    ("kwargs", "message"),
    [
        ({"metric": "unknown"}, "metric"),
        ({"mv_mode": "sideways"}, "mv_mode"),
        ({"variant": "unknown"}, "variant"),
        ({"missing_strategy": "unknown"}, "missing_strategy"),
        ({"variant": "msm", "metric": "squared_euclidean"}, "metric"),
        ({"variant": "msm", "missing_strategy": "zero_cost"}, "MissingStrategy"),
    ],
)
def test_invalid_semantics_are_refused_by_cpp(kwargs, message):
    """C++ reads the settings (the name tables, core::validate) before the
    series reach the Problem."""
    with pytest.raises(ValueError, match=message):
        dtwcpp.DTWClustering(n_clusters=2, **kwargs).fit(SQUARED_FIXTURE)


@pytest.mark.parametrize(
    ("settings", "series"),
    [
        (
            {"variant": "ddtw", "metric": "squared_euclidean"},
            [[0, 1, 3, 6], [0, 1, 2, 4], [5, 5, 6, 6], [5, 6, 6, 7]],
        ),
        (
            {"missing_strategy": "zero_cost", "metric": "squared_euclidean"},
            [[0, np.nan, 1, 0], [0, 1, 1, 0], [9, 9, np.nan, 8], [9, 8, 8, 8]],
        ),
    ],
)
def test_fit_computes_every_metric_cpp_computes(settings, series):
    """DDTW and the missing-data strategies take a metric in C++, so the fitted
    cost is the sum of each series' distance.dtw to its medoid."""
    estimator = dtwcpp.DTWClustering(n_clusters=2, **settings).fit(series)
    medoids = [series[m] for m in estimator.medoid_indices_]
    cost = sum(
        dtwcpp.distance.dtw(s, medoids[label], **settings)
        for s, label in zip(series, estimator.labels_)
    )
    assert estimator.inertia_ == pytest.approx(cost, rel=1e-12)
