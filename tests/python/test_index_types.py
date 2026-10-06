"""
@file test_index_types.py
@brief Labels and medoids are int64 arrays, and counts take 64-bit values, in Python.

C++ holds labels, medoids and counts as index_t (std::int64_t). The bindings hand the
vectors out as np.int64 arrays (a copy each time) and read the counts as 64-bit.
@author Volkan Kumtepeli
"""

import importlib.util

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _hpc

# Eight series of length four that three PAM medoids separate unambiguously.
_X = np.array([0.0, 0.01, -0.02, 0.03])[None, :] + np.arange(8.0)[:, None]
_BIG = 1 << 40  # past int32, so a parameter still typed int cannot take it


def _problem():
    prob = dtwcpp.Problem("index_types")
    prob.set_data(_X.tolist(), [str(i) for i in range(len(_X))])
    return prob


def _assert_int64(value, where):
    assert isinstance(value, np.ndarray), where
    assert value.dtype == np.int64, f"{where}: {value.dtype}"
    assert value.ndim == 1, where


_CLUSTERING_ROUTES = {
    "fast_pam": lambda p: dtwcpp.fast_pam(p, 3),
    "fast_clara": lambda p: dtwcpp.fast_clara(p, 3, sample_size=6, n_samples=2),
    "one_batch_pam": lambda p: dtwcpp.one_batch_pam(p, 3),
    "cut_dendrogram": lambda p: dtwcpp.cut_dendrogram(dtwcpp.build_dendrogram(p), p, 3),
}


@pytest.mark.parametrize("route", _CLUSTERING_ROUTES)
def test_clustering_result_and_problem_arrays_are_int64(route):
    prob = _problem()
    result = _CLUSTERING_ROUTES[route](prob)

    _assert_int64(result.labels, f"{route} labels")
    _assert_int64(result.medoid_indices, f"{route} medoid_indices")
    assert result.labels.shape == (len(_X),)
    assert result.medoid_indices.shape == (3,)
    # The Problem the algorithm wrote back to, under its four spellings.
    for name, value in (("labels()", prob.labels()), ("medoids()", prob.medoids()),
                        ("clusters_ind", prob.clusters_ind),
                        ("centroids_ind", prob.centroids_ind)):
        _assert_int64(value, f"{route} Problem.{name}")
    np.testing.assert_array_equal(prob.labels(), result.labels)
    np.testing.assert_array_equal(prob.clusters_ind, result.labels)
    np.testing.assert_array_equal(prob.medoids(), result.medoid_indices)
    np.testing.assert_array_equal(prob.centroids_ind, result.medoid_indices)


def test_the_arrays_are_copies_of_the_problem_state():
    prob = _problem()
    dtwcpp.fast_pam(prob, 3)
    before = prob.labels().copy()
    prob.labels()[:] = 99
    prob.clusters_ind[:] = 99
    np.testing.assert_array_equal(prob.labels(), before)


def test_problem_without_a_clustering_has_empty_int64_arrays():
    prob = _problem()
    for value in (prob.labels(), prob.medoids(), prob.clusters_ind, prob.centroids_ind):
        _assert_int64(value, "empty")
        assert value.size == 0


@pytest.mark.parametrize("make", [list, np.asarray,
                                  lambda v: np.asarray(v, dtype=np.int32)])
def test_clustering_result_takes_lists_and_arrays_and_returns_int64(make):
    result = dtwcpp.ClusteringResult()
    result.labels = make([0, 0, 1])
    result.medoid_indices = make([0, 2])

    _assert_int64(result.labels, "labels")
    _assert_int64(result.medoid_indices, "medoid_indices")
    assert result.labels.tolist() == [0, 0, 1]
    assert result.medoid_indices.tolist() == [0, 2]


_METHODS = ["pam", "onebatch", "clara", "kmedoids", "hierarchical", "tadpole", "lrcore",
            pytest.param("mip", marks=pytest.mark.skipif(
                not dtwcpp.HIGHS_AVAILABLE and importlib.util.find_spec("highspy") is None,
                reason="needs linked HiGHS or highspy (the mip extra)"))]


@pytest.mark.parametrize("method", _METHODS)
def test_cluster_result_arrays_are_int64_for_every_method(method):
    result = dtwcpp.cluster(_X, k=3, method=method)

    _assert_int64(result.labels, f"{method} labels")
    _assert_int64(result.medoids, f"{method} medoids")
    assert result.labels.shape == (len(_X),)
    assert result.medoids.shape == (3,)


def test_hpc_result_labels_are_int64(tmp_path):
    path = tmp_path / "job_labels.csv"
    path.write_text("name,cluster\n1,0\n2,1\n3,1\n", encoding="utf-8")
    labels = _hpc.parse_labels_csv(path)
    result = dtwcpp.Result(labels, device="hpc", elapsed_s=0.0, k=2, n_series=3)

    _assert_int64(labels, "parse_labels_csv")
    _assert_int64(result.labels, "hpc Result.labels")
    assert result.labels.tolist() == [0, 1, 1]


def test_estimator_arrays_are_int64():
    fitted = dtwcpp.DTWClustering(n_clusters=3).fit(_X)
    _assert_int64(fitted.labels_, "DTWClustering.labels_")
    _assert_int64(fitted.medoid_indices_, "DTWClustering.medoid_indices_")
    _assert_int64(fitted.predict(_X), "DTWClustering.predict")


def test_barycenter_clustering_labels_are_int64():
    options = dtwcpp.BarycenterClusteringOptions()
    options.n_clusters = 3
    options.target_length = _X.shape[1]
    result = dtwcpp.barycenter_kmeans(_problem(), options)

    _assert_int64(result.labels, "barycenter_kmeans labels")
    assert result.labels.shape == (len(_X),)


_PAST_INT32_COUNTS = {
    "fast_pam": lambda p: dtwcpp.fast_pam(p, _BIG),
    "fast_clara": lambda p: dtwcpp.fast_clara(p, _BIG),
    "one_batch_pam": lambda p: dtwcpp.one_batch_pam(p, _BIG),
    "cut_dendrogram": lambda p: dtwcpp.cut_dendrogram(dtwcpp.build_dendrogram(p), p, _BIG),
    "dist_by_ind i": lambda p: p.dist_by_ind(_BIG, 0),
    "dist_by_ind j": lambda p: p.dist_by_ind(0, _BIG),
}


@pytest.mark.parametrize("site", _PAST_INT32_COUNTS)
def test_counts_past_int32_reach_cpp_and_are_judged_there(site):
    """A count that does not fit an int is refused by C++ as InvalidInput (k or the
    index against N), not by the binding as a TypeError for an argument too wide."""
    with pytest.raises(dtwcpp.InvalidInput):
        _PAST_INT32_COUNTS[site](_problem())


def test_counts_that_only_size_things_take_64_bits():
    prob = _problem()
    prob.set_n_clusters(_BIG)
    assert prob.n_clusters() == _BIG
    with pytest.raises(dtwcpp.InvalidInput, match="k must not exceed"):
        dtwcpp.cluster(_X, k=_BIG)


def test_fast_clara_seed_is_a_full_uint64():
    """The C++ seed is a uint64; the binding read it as an unsigned int."""
    first = dtwcpp.fast_clara(_problem(), 3, sample_size=6, n_samples=2, seed=_BIG)
    again = dtwcpp.fast_clara(_problem(), 3, sample_size=6, n_samples=2, seed=_BIG)

    np.testing.assert_array_equal(first.labels, again.labels)
    with pytest.raises(TypeError):
        dtwcpp.fast_clara(_problem(), 3, seed=1 << 64)
