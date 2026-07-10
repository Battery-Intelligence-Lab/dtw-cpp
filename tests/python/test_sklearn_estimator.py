"""Contract tests for the dedicated sklearn DTWCKMedoids estimator."""

import numpy as np
import pytest

import dtwcpp
from dtwcpp.sklearn import DTWCKMedoids


def separated_series():
    low = [np.array([0.0, 1.0, 2.0, 1.0]) + i * 0.01 for i in range(5)]
    high = [np.array([50.0, 51.0, 52.0, 51.0]) + i * 0.01 for i in range(5)]
    return np.asarray(low + high)


def test_raw_fit_predict_transform_and_score():
    X = separated_series()
    estimator = DTWCKMedoids(n_clusters=2, random_state=3)
    labels = estimator.fit_predict(X)

    assert labels.shape == (10,)
    assert estimator.medoid_indices_.shape == (2,)
    assert estimator.transform(X).shape == (10, 2)
    np.testing.assert_array_equal(estimator.predict(X), labels)
    assert estimator.score(X) == pytest.approx(-estimator.inertia_)
    assert len(estimator.cluster_centers_) == 2


def test_default_seed_matches_tier1_seed_contract():
    X = (np.array([0.0, 0.01, -0.02, 0.03])[None, :]
         + np.arange(8.0)[:, None])
    tier1 = dtwcpp.cluster(X, k=3, method="pam")
    estimator = DTWCKMedoids(n_clusters=3).fit(X)

    assert dtwcpp.DEFAULT_RANDOM_SEED == 42
    np.testing.assert_array_equal(estimator.medoid_indices_, tier1.medoids)
    np.testing.assert_array_equal(estimator.labels_, tier1.labels)
    assert estimator.inertia_ == tier1.cost == 24.0


def test_precomputed_fit_and_rectangular_query_both_work():
    X = separated_series()
    D = dtwcpp.compute_distance_matrix(X.tolist(), use_pruning=False)
    estimator = DTWCKMedoids(n_clusters=2, metric="precomputed").fit(D)

    transformed = estimator.transform(D)
    assert transformed.shape == (10, 2)
    np.testing.assert_array_equal(estimator.predict(D), estimator.labels_)

    query = D[:3, :]  # M-by-N: distances from 3 queries to all fit samples
    assert estimator.transform(query).shape == (3, 2)
    np.testing.assert_array_equal(
        estimator.transform(query), query[:, estimator.medoid_indices_]
    )


def test_onebatch_raw_mode_and_sklearn_clone_contract():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone

    estimator = DTWCKMedoids(
        n_clusters=2, method="onebatch", batch_size=6, random_state=11
    )
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()
    labels = cloned.fit_predict(separated_series())
    assert set(labels) == {0, 1}
    tags = cloned.__sklearn_tags__()
    assert tags is not None


def test_sklearn_common_estimator_contract():
    pytest.importorskip("sklearn")
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(DTWCKMedoids(n_clusters=2))


def test_precomputed_native_pairwise_tag():
    pytest.importorskip("sklearn", minversion="1.6")
    from sklearn.utils import get_tags

    precomputed = DTWCKMedoids(n_clusters=2, metric="precomputed")
    raw = DTWCKMedoids(n_clusters=2, metric="dtw")

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
        DTWCKMedoids(metric="precomputed", random_state=7),
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
        DTWCKMedoids(n_clusters=1, metric="precomputed").fit(matrix)


def test_precomputed_onebatch_is_rejected_loudly():
    with pytest.raises(ValueError, match="requires raw series"):
        DTWCKMedoids(
            n_clusters=1, metric="precomputed", method="onebatch"
        ).fit(np.zeros((2, 2)))
