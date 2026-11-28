"""
@file test_problem.py
@brief Tests for the Problem class Python bindings.
@author Volkan Kumtepeli
@date 28 Mar 2026
"""

import numpy as np
import pytest

import dtwcpp


class TestProblemConstruction:
    """Tests for Problem construction and basic properties."""

    def test_named_construction(self):
        """Problem('name') stores the name."""
        p = dtwcpp.Problem("mytest")
        assert p.name == "mytest"

    def test_default_band(self):
        """Default band is -1 (full DTW)."""
        p = dtwcpp.Problem("test")
        assert p.band == -1


class TestProblemData:
    """Tests for set_data and data access."""

    def test_a_data_carries_its_own_names_and_ndim(self):
        """set_data(Data, names=...) and set_data(Data, ndim=3) must refuse
        extra names/ndim rather than dropping them."""
        data = dtwcpp.Data([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], ["a", "b"])
        p = dtwcpp.Problem("test")
        with pytest.raises(TypeError, match="names"):
            p.set_data(data, ["x", "y"])
        with pytest.raises(TypeError, match="ndim"):
            p.set_data(data, ndim=3)
        p.set_data(data)
        assert [p.series_name(i) for i in range(p.size)] == ["a", "b"]

    def test_series_reads_a_float32_problem(self):
        """Problem.series widens a float32 series, as Data.series does."""
        p = dtwcpp.Problem("f32")
        p.set_data(dtwcpp.Data.from_float32([[0.5, 1.25], [2.0, -3.0]], ["a", "b"]))
        assert p.series(1) == [2.0, -3.0]


class TestDistanceMatrix:
    """Tests for fill_distance_matrix and dist_by_ind."""

    def _make_problem(self, data):
        """Helper to create a filled problem."""
        names = [f"s{i}" for i in range(len(data))]
        p = dtwcpp.Problem("test")
        p.set_data(data, names)
        p.fill_distance_matrix()
        return p

    def test_known_distance(self):
        """dist_by_ind matches standalone dtwcpp.distance.dtw."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        p = self._make_problem(data)
        assert p.is_distance_matrix_filled()
        expected = dtwcpp.distance.dtw([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert p.dist_by_ind(0, 1) == pytest.approx(expected)
        assert p.dist_by_ind(1, 0) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "i, j, name, bad",
        [(3, 0, "i", 3), (0, 3, "j", 3), (100, 0, "i", 100), (0, 100, "j", 100),
         (-1, 0, "i", -1), (0, -1, "j", -1)],
    )
    def test_index_outside_the_problem_raises(self, i, j, name, bad):
        """Problem::dist_by_ind is the unchecked hot path, so the binding owns the
        range check: an index outside [0, N) raises InvalidInput naming it and N
        instead of reading past the matrix."""
        p = self._make_problem([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]])
        with pytest.raises(dtwcpp.InvalidInput, match=rf"{name} = {bad}\b.*N = 3"):
            p.dist_by_ind(i, j)

    def test_index_on_an_empty_problem_raises(self):
        """Every index is outside an empty Problem."""
        with pytest.raises(dtwcpp.InvalidInput, match=r"i = 0\b.*N = 0"):
            dtwcpp.Problem("empty").dist_by_ind(0, 0)

    def test_last_index_is_valid(self):
        """N - 1 is a valid index and the check leaves valid calls unchanged."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
        p = self._make_problem(data)
        assert p.dist_by_ind(2, 0) == pytest.approx(
            dtwcpp.distance.dtw(data[2], data[0]))
        assert p.dist_by_ind(2, 2) == 0.0


class TestIndexBoundary:
    """`series`, `series_name` and `centroid_of` read unchecked C++ (`Data::series`,
    `Data::name`, `centroids_ind[clusters_ind[i]]`), so the binding owns the range
    check: an index outside [0, N) raises InvalidInput naming it and N, and
    `centroid_of` refuses a Problem that holds no clustering."""

    _DATA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
    _NAMES = ["a", "b", "c"]

    def _problem(self):
        p = dtwcpp.Problem("idx")
        p.set_data(self._DATA, self._NAMES)
        return p

    def _clustered(self):
        p = self._problem()
        p.set_n_clusters(2)
        p.fill_distance_matrix()
        p.cluster()
        return p

    @pytest.mark.parametrize("bad", [3, 100, -1])
    def test_series_index_outside_the_problem_raises(self, bad):
        with pytest.raises(dtwcpp.InvalidInput, match=rf"series: i = {bad}\b.*N = 3"):
            self._problem().series(bad)

    @pytest.mark.parametrize("bad", [3, 100, -1])
    def test_series_name_index_outside_the_problem_raises(self, bad):
        with pytest.raises(dtwcpp.InvalidInput, match=rf"series_name: i = {bad}\b.*N = 3"):
            self._problem().series_name(bad)

    @pytest.mark.parametrize("bad", [3, 100, -1])
    def test_centroid_of_index_outside_the_problem_raises(self, bad):
        p = self._clustered()
        with pytest.raises(dtwcpp.InvalidInput, match=rf"^centroid_of: i = {bad}\b.*N = 3"):
            p.centroid_of(bad)

    @pytest.mark.parametrize("accessor", ["series", "series_name", "centroid_of"])
    def test_every_index_is_outside_an_empty_problem(self, accessor):
        with pytest.raises(dtwcpp.InvalidInput, match=rf"^{accessor}: i = 0\b.*N = 0"):
            getattr(dtwcpp.Problem("empty"), accessor)(0)

    def test_last_index_is_valid(self):
        p = self._problem()
        assert p.series(2) == self._DATA[2]
        assert p.series_name(2) == "c"
        p.set_n_clusters(2)
        p.fill_distance_matrix()
        p.cluster()
        assert p.centroid_of(2) == p.medoids()[p.labels()[2]]

    @pytest.mark.parametrize("i, message", [(0, r"holds no clustering"), (-1, r"i = -1 is outside")])
    def test_centroid_of_before_any_clustering_raises(self, i, message):
        with pytest.raises(dtwcpp.InvalidInput, match=rf"^centroid_of: .*{message}"):
            self._problem().centroid_of(i)

    def test_centroid_of_after_the_cluster_count_shrinks_raises(self):
        """The clustering held 2 medoids; k = 1 leaves a series in cluster 1 with
        no medoid to name, so the Problem no longer holds a clustering."""
        p = self._clustered()
        i = p.labels().tolist().index(1)
        p.set_n_clusters(1)
        with pytest.raises(dtwcpp.InvalidInput, match=r"^centroid_of: .*holds no clustering.*2 medoids for k = 1"):
            p.centroid_of(i)


class TestClusterFirst:
    """Every call that reads the whole clustering refuses an unclustered or stale
    Problem with InvalidInput naming the call."""

    _DATA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
    _CALLS = {
        "find_total_cost": lambda p: p.find_total_cost(),
        "write_result_files": lambda p: dtwcpp.write_result_files(p, "files"),  # relative: the test runs in tmp_path
        "calculate_medoids": lambda p: p.calculate_medoids(),
        "silhouette": dtwcpp.silhouette,
        "inertia": dtwcpp.inertia,
    }

    @pytest.fixture(autouse=True)
    def _in_tmp_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def _problem(self):
        p = dtwcpp.Problem("first")
        p.set_data(self._DATA, ["a", "b", "c"])
        return p

    @pytest.mark.parametrize("sized", [False, True], ids=["never_sized", "sized"])
    @pytest.mark.parametrize("who", list(_CALLS))
    def test_reading_the_clustering_before_clustering_raises(self, tmp_path, sized, who):
        p = self._problem()
        if sized:
            p.set_n_clusters(2)
        with pytest.raises(dtwcpp.InvalidInput, match=rf"^{who}: .*holds no clustering.*cluster it first"):
            self._CALLS[who](p)
        assert list(tmp_path.iterdir()) == []  # refused before any file is opened

    def test_the_refusal_names_the_counts_it_found(self):
        p = self._problem()
        p.set_n_clusters(2)
        with pytest.raises(dtwcpp.InvalidInput, match=r"0 labels for N = 3, 0 medoids for k = 2"):
            p.find_total_cost()

    def test_a_clustering_goes_stale_when_the_cluster_count_changes(self):
        p = self._problem()
        p.set_n_clusters(2)
        p.fill_distance_matrix()
        p.cluster()
        p.set_n_clusters(3)
        with pytest.raises(dtwcpp.InvalidInput, match=r"^find_total_cost: .*2 medoids for k = 3"):
            p.find_total_cost()

    @pytest.mark.parametrize("who", ["find_total_cost", "silhouette", "inertia"])
    def test_replacing_the_series_with_as_many_drops_the_clustering(self, who):
        """The old labels describe the old series, so a same-N set_data must not leave them counting."""
        p = self._problem()
        p.set_n_clusters(2)
        p.cluster()
        p.find_total_cost()
        p.set_data([[7.0, 7.5, 8.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]], ["a", "b", "c"])
        with pytest.raises(dtwcpp.InvalidInput, match=rf"^{who}: .*holds no clustering.*cluster it first"):
            self._CALLS[who](p)

    def test_after_clustering_the_readers_work(self, tmp_path):
        p = self._problem()
        p.set_n_clusters(2)
        p.cluster()
        assert p.find_total_cost() >= 0.0
        dtwcpp.write_result_files(p, tmp_path)
        assert sorted(f.name for f in tmp_path.iterdir()) == [
            "first_distance_matrix.csv", "first_labels.csv", "first_medoids.csv", "first_silhouettes.csv"]



class TestSetterRanges:
    """A cluster count below 1 and a band below -1 have no meaning, so the setters
    refuse them (k = -1 was an untyped "vector too long" from a resize, and a band
    of -5 ran as full DTW). k above N is not the setter's to judge: the data may
    change after it, so `cluster()` refuses it."""

    _DATA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]

    def _problem(self):
        p = dtwcpp.Problem("ranges")
        p.set_data(self._DATA, ["a", "b", "c"])
        return p

    @pytest.mark.parametrize("bad", [0, -1, -(2**31)])
    def test_set_n_clusters_below_one_raises_and_keeps_the_count(self, bad):
        p = self._problem()
        p.set_n_clusters(2)
        with pytest.raises(dtwcpp.InvalidInput, match=rf"set_n_clusters: n_clusters must be at least 1; got {bad}\b"):
            p.set_n_clusters(bad)
        assert p.n_clusters() == 2

    def test_k_above_n_is_refused_when_clustering(self):
        p = self._problem()
        p.set_n_clusters(4)  # N = 3
        assert p.n_clusters() == 4
        with pytest.raises(dtwcpp.InvalidInput):
            p.cluster()

    @pytest.mark.parametrize("bad", [-2, -5, -(2**31)])
    def test_set_band_below_minus_one_raises_and_keeps_the_band(self, bad):
        p = self._problem()
        p.set_band(2)
        with pytest.raises(dtwcpp.InvalidInput, match=rf"set_band: band must be -1 .* got {bad}\b"):
            p.set_band(bad)
        with pytest.raises(dtwcpp.InvalidInput, match=rf"set_band: .* got {bad}\b"):
            p.band = bad
        assert p.band == 2

    @pytest.mark.parametrize("good", [-1, 0, 5])
    def test_full_dtw_and_every_non_negative_band_stay_valid(self, good):
        p = self._problem()
        p.set_band(good)
        assert p.band == good


class TestClusteringIsWrittenThroughSetResult:
    """`clusters_ind` and `centroids_ind` are read-only (v1.0.0's Python never bound
    them), so a clustering reaches a Problem, and the scores that read it, only
    through `set_result`, which validates it as the C++ `Problem::set_result` does."""

    _DATA = [[0.0, 0.1], [0.5, 0.4], [1.0, 1.1], [8.0, 8.2]]

    def _problem(self):
        p = dtwcpp.Problem("res")
        p.set_data(self._DATA, [str(i) for i in range(len(self._DATA))])
        p.set_distance_matrix(dtwcpp.compute_distance_matrix(self._DATA))
        return p

    @staticmethod
    def _result(labels, medoids):
        r = dtwcpp.ClusteringResult()
        r.labels = labels
        r.medoid_indices = medoids
        return r

    @pytest.mark.parametrize("name", ["clusters_ind", "centroids_ind"])
    def test_the_fields_are_read_only(self, name):
        p = self._problem()
        assert getattr(p, name).tolist() == []
        with pytest.raises(AttributeError):
            setattr(p, name, [0])

    def test_set_result_publishes_the_clustering(self):
        p = self._problem()
        p.set_result(self._result([0, 0, 0, 1], [1, 3]))
        assert (p.n_clusters(), p.labels().tolist(), p.medoids().tolist()) == (2, [0, 0, 0, 1], [1, 3])
        assert (p.clusters_ind.tolist(), p.centroids_ind.tolist()) == ([0, 0, 0, 1], [1, 3])
        assert [p.centroid_of(i) for i in range(4)] == [1, 1, 1, 3]

    @pytest.mark.parametrize(
        "labels, medoids, why",
        [([0, 0, 1], [0, 3], "labels of the wrong length"),
         ([0, 0, 0, 1], [0, 4], "a medoid above N"),
         ([0, 0, 0, 1], [-1, 3], "a negative medoid"),
         ([0, 0, 0, 1], [3, 3], "a repeated medoid"),
         ([0, 0, 0, 2], [0, 3], "a label with no medoid"),
         ([0, 0, 0, -1], [0, 3], "a negative label"),
         ([0, 0, 0, 0], [], "no medoid")],
    )
    def test_an_invalid_result_raises_and_leaves_the_problem_unchanged(self, labels, medoids, why):
        p = self._problem()
        p.set_result(self._result([0, 0, 1, 1], [0, 2]))
        with pytest.raises(dtwcpp.InvalidInput, match="set_result"):
            p.set_result(self._result(labels, medoids))
        assert (p.n_clusters(), p.labels().tolist(), p.medoids().tolist()) == (2, [0, 0, 1, 1], [0, 2]), why


class TestDenseSemanticMutation:
    """A populated dense/precomputed matrix is bound to one exact configuration."""

    @staticmethod
    def _problem(data):
        p = dtwcpp.Problem("semantic_mutation")
        p.set_data(data, [f"s{i}" for i in range(len(data))])
        return p

    def test_band_property_invalidates_cached_distance(self):
        p = self._problem([[0.0, 0.0, 10.0], [0.0, 10.0, 10.0]])
        assert p.dist_by_ind(0, 1) == 0.0

        p.band = 0

        assert not p.is_distance_matrix_filled()
        assert p.dist_by_ind(0, 1) == 10.0

    def test_variant_setters_rebind_and_the_getter_is_a_copy(self):
        p = self._problem([[0.0], [2.0]])
        precomputed = np.array([[0.0, 123.0], [123.0, 0.0]])
        p.set_distance_matrix(precomputed)

        p.variant_params = dtwcpp.DTWVariantParams()
        assert p.is_distance_matrix_filled()
        assert p.dist_by_ind(0, 1) == 123.0

        params = dtwcpp.DTWVariantParams()
        params.variant = dtwcpp.DTWVariant.WDTW
        params.wdtw_g = 0.5
        p.variant_params = params
        assert p.dist_by_ind(0, 1) == 1.0

        p.set_variant(dtwcpp.DTWVariant.Standard)
        assert p.dist_by_ind(0, 1) == 2.0
        # variant_params is read-only: a nested write raises instead of editing
        # a copy the Problem never sees.
        with pytest.raises(AttributeError):
            p.variant_params.variant = dtwcpp.DTWVariant.WDTW
        assert p.variant_params.variant == dtwcpp.DTWVariant.Standard
        assert p.dist_by_ind(0, 1) == 2.0

    def test_missing_strategy_property_invalidates_cached_distance(self):
        p = self._problem([[0.0, np.nan, 2.0], [0.0, 2.0, 2.0]])
        p.missing_strategy = dtwcpp.MissingStrategy.ZeroCost
        assert p.dist_by_ind(0, 1) == 0.0

        p.missing_strategy = dtwcpp.MissingStrategy.Interpolate

        assert not p.is_distance_matrix_filled()
        assert p.dist_by_ind(0, 1) == 1.0

    def test_device_setters_drop_precomputed_only_on_a_change(self):
        p = self._problem([[0.0], [2.0]])
        precomputed = np.array([[0.0, 123.0], [123.0, 0.0]])
        p.set_distance_matrix(precomputed)

        p.set_device("cpu")  # already the CPU
        assert p.is_distance_matrix_filled()
        p.set_gpu_precision(dtwcpp.GpuPrecision.FP64)
        assert not p.is_distance_matrix_filled()
        assert p.dist_by_ind(0, 1) == 2.0


class TestBandProperty:
    """Tests for band feasibility on a precomputed vs partially computed matrix."""

    def test_a_complete_matrix_needs_no_feasible_band(self):
        """Every pair known: nothing is computed, so no band can be infeasible.

        Lengths 4, 10, 5, 10 under band 2 leave pairs no warping path fits.
        The oracle is the matrix's own numbers: k=2 costs 1 + 2 = 3.
        """
        p = dtwcpp.Problem("known")
        p.set_data([[0.0] * 4, [1.0] * 10, [2.0] * 5, [3.0] * 10],
                   ["a", "b", "c", "d"])
        p.band = 2
        known = np.array([[0.0, 1.0, 9.0, 9.0], [1.0, 0.0, 9.0, 9.0],
                          [9.0, 9.0, 0.0, 2.0], [9.0, 9.0, 2.0, 0.0]])
        p.set_distance_matrix(known)
        assert p.dist_by_ind(0, 1) == 1.0
        assert dtwcpp.fast_pam(p, 2).total_cost == 3.0

        # A pair left to compute brings the band check back, cached pair or not.
        holes = known.copy()
        holes[0, 1] = holes[1, 0] = np.nan
        p.set_distance_matrix(holes)
        with pytest.raises(ValueError, match="band = 2"):
            p.dist_by_ind(2, 3)

