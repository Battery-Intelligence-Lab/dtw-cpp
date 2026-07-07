"""
@file test_api.py
@brief Tests for the unified device()/load()/cluster()/result.plot() interface.
@author Volkan Kumtepeli
"""
import numpy as np
import pytest

import dtwcpp


@pytest.fixture(autouse=True)
def _reset_device():
    dtwcpp.device("cpu")
    yield
    dtwcpp.device("cpu")


def _two_groups(seed=7):
    rng = np.random.default_rng(seed)
    return np.array([rng.standard_normal(12) * 0.1 + (0.0 if i < 6 else 9.0)
                     for i in range(12)])


# ---------------------------------------------------------------------------
# load() — lazy handle
# ---------------------------------------------------------------------------
class TestLoad:
    def test_load_array_returns_dataset(self):
        ds = dtwcpp.load(_two_groups())
        assert isinstance(ds, dtwcpp.Dataset)
        assert not ds.is_path

    def test_load_path_is_lazy_and_does_not_read(self):
        """A nonexistent path is fine until something materializes it."""
        ds = dtwcpp.load("definitely_missing_file.tsv")   # must NOT raise
        assert ds.is_path
        assert ds.name == "definitely_missing_file"

    def test_as_series_materializes_array(self):
        ds = dtwcpp.load([[1.0, 2.0], [3.0, 4.0]])
        assert ds.as_series() == [[1.0, 2.0], [3.0, 4.0]]


# ---------------------------------------------------------------------------
# cluster() — local cpu path
# ---------------------------------------------------------------------------
class TestClusterLocal:
    def test_recovers_two_groups(self):
        res = dtwcpp.cluster(_two_groups(), k=2)
        assert res.n_series == 12
        assert len(set(res.labels[:6])) == 1
        assert len(set(res.labels[6:])) == 1
        assert res.labels[0] != res.labels[11]

    def test_result_fields_populated(self):
        res = dtwcpp.cluster(_two_groups(), k=2)
        assert res.device == "cpu"
        assert res.cost is not None
        assert res.distance_matrix is not None
        assert res.medoid_indices is not None
        assert res.elapsed_s >= 0.0

    def test_summary_contains_device_and_timing(self):
        res = dtwcpp.cluster(_two_groups(), k=2)
        s = res.summary()
        assert "device=cpu" in s and "ms" in s

    def test_uses_global_device(self):
        dtwcpp.device("cpu")
        res = dtwcpp.cluster(_two_groups(), k=2)
        assert res.device == "cpu"

    def test_accepts_raw_array_without_explicit_load(self):
        res = dtwcpp.cluster(_two_groups(), k=2)   # not wrapped in load()
        assert res.n_series == 12


# ---------------------------------------------------------------------------
# result.plot()
# ---------------------------------------------------------------------------
class TestPlot:
    def test_plot_writes_png(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        res = dtwcpp.cluster(_two_groups(), k=2)
        out = tmp_path / "c.png"
        assert res.plot(png=str(out), show=False) == str(out)
        assert out.exists()

    def test_plot_without_matrix_returns_none(self, capsys):
        """An hpc-style result (labels only) can't plot; it reports sizes."""
        res = dtwcpp.ClusterResult([0, 0, 1, 1], device="hpc", elapsed_s=1.0,
                                   k=2, n_series=4)
        assert res.plot(show=False) is None
        assert "no local distance matrix" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cluster() — hpc path must NOT read data locally
# ---------------------------------------------------------------------------
class TestClusterHpc:
    def test_path_source_not_read_locally(self, monkeypatch):
        from dtwcpp import _hpc
        captured = {}

        def fake(source, k, **kwargs):
            captured["source"] = source
            captured["k"] = k
            return np.array([0, 0, 1, 1])

        monkeypatch.setattr(_hpc, "cluster_on_hpc", fake)
        ds = dtwcpp.load("missing_on_laptop.tsv")            # never read locally
        res = dtwcpp.cluster(ds, k=2, device="hpc")
        assert captured["source"] == "missing_on_laptop.tsv"  # path passed through
        assert res.device == "hpc"
        assert res.distance_matrix is None

    def test_array_source_passed_as_series(self, monkeypatch):
        from dtwcpp import _hpc
        captured = {}

        def fake(source, k, **kwargs):
            captured["source"] = source
            return np.zeros(len(source), dtype=int)

        monkeypatch.setattr(_hpc, "cluster_on_hpc", fake)
        dtwcpp.device("hpc")
        dtwcpp.cluster([[1.0, 2.0], [3.0, 4.0]], k=2)
        assert captured["source"] == [[1.0, 2.0], [3.0, 4.0]]   # materialized series


# ---------------------------------------------------------------------------
# cluster(method=...) dispatch — Task 0.14
#
# BUG BEING PINNED: the local (cpu/gpu) path of cluster() accepted a ``method``
# argument but never consulted it — it ran FastPAM unconditionally
# (old _api.py: ``res = fast_pam(prob, k, max_iter)``). So ``method="mip"``,
# ``method="clara"``, and even a nonsense ``method="xyz"`` all silently produced
# a FastPAM result. The fix validates the name (unknown -> ValueError) and
# dispatches each documented method to its own algorithm.
# ---------------------------------------------------------------------------
class TestClusterMethodDispatch:
    def test_unknown_method_raises(self):
        """Unknown method must raise, not silently run FastPAM.

        Pre-fix: method is ignored, FastPAM runs, a ClusterResult is returned
        with NO exception -> this test fails. Post-fix: ValueError."""
        with pytest.raises(ValueError, match="unknown method"):
            dtwcpp.cluster(_two_groups(), k=2, method="not_a_real_method")

    def test_unknown_method_rejected_before_hpc_offload(self, monkeypatch):
        """Validation is central: a bad method must never reach the cluster.

        Pre-fix: the hpc path forwarded any raw string to cluster_on_hpc, so a
        nonsense method was submitted to SLURM. Post-fix: ValueError first."""
        from dtwcpp import _hpc

        def boom(*a, **k):
            raise AssertionError("cluster_on_hpc must not be called for a bad method")

        monkeypatch.setattr(_hpc, "cluster_on_hpc", boom)
        with pytest.raises(ValueError, match="unknown method"):
            dtwcpp.cluster("data.tsv", k=2, device="hpc", method="bogus")

    def test_local_dispatch_routes_to_clara_not_fastpam(self, monkeypatch):
        """method='clara' must call fast_clara, NOT fast_pam.

        Pre-fix: the clara branch did not exist and fast_pam ran instead, so
        the fast_clara spy is never called (called['clara'] stays 0) -> fails.
        Post-fix: fast_clara is invoked exactly once and fast_pam is not."""
        import dtwcpp
        called = {"pam": 0, "clara": 0}
        real_clara = dtwcpp.fast_clara

        def spy_clara(*a, **kw):
            called["clara"] += 1
            return real_clara(*a, **kw)

        def poisoned_pam(*a, **kw):
            called["pam"] += 1
            raise AssertionError("method='clara' fell through to fast_pam")

        monkeypatch.setattr(dtwcpp, "fast_clara", spy_clara)
        monkeypatch.setattr(dtwcpp, "fast_pam", poisoned_pam)
        res = dtwcpp.cluster(_two_groups(), k=2, method="clara")
        assert called["clara"] == 1
        assert called["pam"] == 0
        assert res.n_series == 12

    def test_local_default_still_routes_to_fastpam(self, monkeypatch):
        """method='pam' (the default) must still call fast_pam — no regression."""
        import dtwcpp
        called = {"pam": 0}
        real_pam = dtwcpp.fast_pam

        def spy_pam(*a, **kw):
            called["pam"] += 1
            return real_pam(*a, **kw)

        monkeypatch.setattr(dtwcpp, "fast_pam", spy_pam)
        dtwcpp.cluster(_two_groups(), k=2)          # default method="pam"
        assert called["pam"] == 1

    def test_local_clara_runs_end_to_end(self):
        """The clara branch must actually work end-to-end (no solver needed).

        Recovers the two well-separated groups, proving real dispatch — not
        just that a non-ValueError was returned."""
        res = dtwcpp.cluster(_two_groups(), k=2, method="clara")
        assert res.n_series == 12
        assert res.distance_matrix is not None
        assert len(set(res.labels[:6])) == 1
        assert len(set(res.labels[6:])) == 1
        assert res.labels[0] != res.labels[11]

    @pytest.mark.parametrize(
        "method", ["auto", "pam", "clara", "kmedoids", "mip", "hierarchical"])
    def test_documented_methods_accepted_and_forwarded_to_hpc(self, monkeypatch, method):
        """Every documented method is accepted (no ValueError) and forwarded.

        Uses the hpc path (cluster_on_hpc stubbed) so mip/kmedoids do not need a
        solver and no local files are written — this asserts the name survives
        validation and is passed through to dtwc_cl --method verbatim."""
        from dtwcpp import _hpc
        captured = {}

        def fake(source, k, **kwargs):
            captured["method"] = kwargs.get("method")
            return np.zeros(4, dtype=int)

        monkeypatch.setattr(_hpc, "cluster_on_hpc", fake)
        dtwcpp.cluster([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                       k=2, device="hpc", method=method)
        assert captured["method"] == method

    def test_hclust_alias_normalizes_to_hierarchical(self, monkeypatch):
        """'hclust' is the CLI alias for 'hierarchical' and must normalize.

        Pre-fix: the raw 'hclust' string was forwarded unchanged. Post-fix it
        is normalized to 'hierarchical' before being forwarded."""
        from dtwcpp import _hpc
        captured = {}

        def fake(source, k, **kwargs):
            captured["method"] = kwargs.get("method")
            return np.zeros(4, dtype=int)

        monkeypatch.setattr(_hpc, "cluster_on_hpc", fake)
        dtwcpp.cluster([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                       k=2, device="hpc", method="hclust")
        assert captured["method"] == "hierarchical"


# ---------------------------------------------------------------------------
# LOCAL dispatch binding names — Task 0.14 gap closed by R4(c)
#
# The prior wave only spied the clara/pam LOCAL routes (above). These tests pin
# the remaining LOCAL routes — 'mip', 'kmedoids', 'hierarchical' — by exercising
# the LIVE dispatch function dtwcpp._api._run_local_method directly and asserting
# EXACTLY the binding names it calls. This is the class of bug the test exists to
# catch: if _api.py names a binding that does not exist (e.g. a typo'd
# build_dendrogram / cut_dendrogram / Method.MIP / Problem.cluster), these fail.
#
# All binding names below were verified present in python/src/_dtwcpp_core.cpp:
#   build_dendrogram (m.def, prob + default opts), cut_dendrogram (dend, prob, k),
#   Problem.set_number_of_clusters / .method / .cluster / .clusters_ind /
#   .centroids_ind / .find_total_cost, Method.MIP / Method.Kmedoids.
# ---------------------------------------------------------------------------
class TestLocalDispatchBindingNames:
    def test_local_mip_sets_method_mip_and_calls_cluster(self):
        """method='mip' -> Problem.method = Method.MIP, Problem.cluster(), then
        read back clusters_ind / centroids_ind / find_total_cost().

        No solver needed: a fake Problem stands in for the C++ binding, so this
        isolates the dispatch (the binding NAMES) from the MIP solve."""
        from dtwcpp import _api

        class FakeProblem:
            def __init__(self):
                self.method = None
                self.nc = None
                self.cluster_calls = 0
                self.clusters_ind = [0, 1, 0, 1]
                self.centroids_ind = [0, 1]

            def set_number_of_clusters(self, k):
                self.nc = k

            def cluster(self):
                self.cluster_calls += 1

            def find_total_cost(self):
                return 7.5

        fake = FakeProblem()
        labels, medoids, cost = _api._run_local_method(
            fake, "mip", k=2, max_iter=100, n=4)
        assert fake.method == dtwcpp.Method.MIP        # NOT Kmedoids
        assert fake.nc == 2
        assert fake.cluster_calls == 1
        assert labels == [0, 1, 0, 1]
        assert medoids == [0, 1]
        assert cost == 7.5

    def test_local_kmedoids_sets_method_kmedoids_and_calls_cluster(self):
        """method='kmedoids' -> Problem.method = Method.Kmedoids, Problem.cluster()."""
        from dtwcpp import _api

        class FakeProblem:
            def __init__(self):
                self.method = None
                self.nc = None
                self.cluster_calls = 0
                self.clusters_ind = [0, 0, 1]
                self.centroids_ind = [0, 2]

            def set_number_of_clusters(self, k):
                self.nc = k

            def cluster(self):
                self.cluster_calls += 1

            def find_total_cost(self):
                return 1.0

        fake = FakeProblem()
        labels, medoids, cost = _api._run_local_method(
            fake, "kmedoids", k=2, max_iter=100, n=3)
        assert fake.method == dtwcpp.Method.Kmedoids   # NOT MIP
        assert fake.nc == 2
        assert fake.cluster_calls == 1
        assert (labels, medoids, cost) == ([0, 0, 1], [0, 2], 1.0)

    def test_local_hierarchical_calls_build_then_cut(self, monkeypatch):
        """method='hierarchical' -> build_dendrogram(prob) then
        cut_dendrogram(dend, prob, k); read labels/medoid_indices/total_cost off
        the cut result. Must NOT fall through to fast_pam."""
        from dtwcpp import _api

        calls = {"build": 0, "cut": 0}
        sentinel_prob = object()
        dend_token = object()

        class FakeCut:
            labels = [0, 0, 1, 1]
            medoid_indices = [0, 2]
            total_cost = 3.25

        def spy_build(prob, *a, **kw):
            calls["build"] += 1
            assert prob is sentinel_prob
            return dend_token

        def spy_cut(dend, prob, k, *a, **kw):
            calls["cut"] += 1
            assert dend is dend_token        # build's output threaded into cut
            assert prob is sentinel_prob
            assert k == 3
            return FakeCut()

        def poison_pam(*a, **kw):
            raise AssertionError("hierarchical must not fall through to fast_pam")

        monkeypatch.setattr(dtwcpp, "build_dendrogram", spy_build)
        monkeypatch.setattr(dtwcpp, "cut_dendrogram", spy_cut)
        monkeypatch.setattr(dtwcpp, "fast_pam", poison_pam)

        labels, medoids, cost = _api._run_local_method(
            sentinel_prob, "hierarchical", k=3, max_iter=100, n=4)
        assert calls == {"build": 1, "cut": 1}
        assert labels == [0, 0, 1, 1]
        assert medoids == [0, 2]
        assert cost == 3.25

    def test_local_kmedoids_end_to_end_does_not_fall_through(self, monkeypatch):
        """Through the LIVE full local cluster() path, method='kmedoids' runs
        Lloyd via Problem.cluster() and must NOT call fast_pam/fast_clara/
        build_dendrogram (the pre-0.14 bug ran FastPAM for every method).

        Solver-free (Lloyd needs no MIP solver)."""
        def poison(name):
            def _p(*a, **kw):
                raise AssertionError(f"kmedoids must not call {name}")
            return _p

        monkeypatch.setattr(dtwcpp, "fast_pam", poison("fast_pam"))
        monkeypatch.setattr(dtwcpp, "fast_clara", poison("fast_clara"))
        monkeypatch.setattr(dtwcpp, "build_dendrogram", poison("build_dendrogram"))

        res = dtwcpp.cluster(_two_groups(), k=2, method="kmedoids")
        assert res.n_series == 12
        assert res.cost is not None            # find_total_cost() was read back
        assert res.distance_matrix is not None
        # Valid k=2 labeling produced by the real Lloyd path (no fallthrough).
        assert len(res.labels) == 12
        assert set(res.labels).issubset({0, 1})
