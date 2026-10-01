"""
@file test_problem.py
@brief Tests for the Problem class Python bindings.
@author Volkan Kumtepeli
"""

import math
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import dtwcpp


class TestProblemConstruction:
    """Tests for Problem construction and basic properties."""

    def test_default_construction(self):
        """Problem() default construction works."""
        p = dtwcpp.Problem()
        assert p is not None

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

    def test_set_data_list_of_lists(self):
        """set_data works with list-of-lists."""
        p = dtwcpp.Problem("test")
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        p.set_data(data, ["a", "b", "c"])
        assert p.size == 3

    def test_set_data_numpy(self, synthetic_data):
        """set_data works with numpy array rows."""
        p = dtwcpp.Problem("test")
        names = [f"s{i}" for i in range(len(synthetic_data))]
        rows = [row.tolist() for row in synthetic_data]
        p.set_data(rows, names)
        assert p.size == len(synthetic_data)


class TestDistanceMatrix:
    """Tests for fill_distance_matrix and dist_by_ind."""

    def _make_problem(self, data):
        """Helper to create a filled problem."""
        names = [f"s{i}" for i in range(len(data))]
        p = dtwcpp.Problem("test")
        p.set_data(data, names)
        p.fill_distance_matrix()
        return p

    def test_fill_and_query(self):
        """fill_distance_matrix populates all pairs."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
        p = self._make_problem(data)
        assert p.is_distance_matrix_filled()
        # All pairs should be finite
        for i in range(3):
            for j in range(3):
                d = p.dist_by_ind(i, j)
                assert math.isfinite(d), f"dist({i},{j}) is not finite: {d}"

    def test_self_distance_zero(self):
        """dist_by_ind(i, i) == 0 for all i."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        p = self._make_problem(data)
        for i in range(3):
            assert p.dist_by_ind(i, i) == pytest.approx(0.0)

    def test_symmetry(self):
        """dist_by_ind(i, j) == dist_by_ind(j, i)."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
        p = self._make_problem(data)
        for i in range(3):
            for j in range(i + 1, 3):
                assert p.dist_by_ind(i, j) == pytest.approx(p.dist_by_ind(j, i))

    def test_known_distance(self):
        """dist_by_ind matches standalone dtwcpp.distance.dtw."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        p = self._make_problem(data)
        expected = dtwcpp.distance.dtw([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert p.dist_by_ind(0, 1) == pytest.approx(expected)

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


def _outcome_in_child(setup, call):
    """Run `setup`, then evaluate `call`, in a fresh interpreter, so a crash fails
    one test instead of ending pytest. Returns "Type: message" for the exception
    raised, "returned <value>", or "crashed with exit code <n>"."""
    code = ("import dtwcpp\n" + textwrap.dedent(setup)
            + f"\ntry:\n    print('returned', {call})\n"
              "except Exception as e:\n    print(type(e).__name__ + ':', e)\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else f"crashed with exit code {run.returncode}"


class TestIndexBoundary:
    """`series`, `series_name` and `centroid_of` read unchecked C++ (`Data::series`,
    `Data::name`, `centroids_ind[clusters_ind[i]]`), so the binding owns the range
    check: an index outside [0, N) raises InvalidInput naming it and N, and
    `centroid_of` refuses a Problem that holds no clustering."""

    _DATA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
    _NAMES = ["a", "b", "c"]
    _UNCLUSTERED = f"""
        p = dtwcpp.Problem("idx")
        p.set_data({_DATA}, {_NAMES})
    """
    _CLUSTERED = _UNCLUSTERED + """
        p.set_n_clusters(2)
        p.fill_distance_matrix()
        p.cluster()
    """

    def _problem(self):
        p = dtwcpp.Problem("idx")
        p.set_data(self._DATA, self._NAMES)
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
        outcome = _outcome_in_child(self._CLUSTERED, f"p.centroid_of({bad})")
        assert re.match(rf"InvalidInput: centroid_of: i = {bad}\b.*N = 3", outcome), outcome

    @pytest.mark.parametrize("accessor", ["series", "series_name", "centroid_of"])
    def test_every_index_is_outside_an_empty_problem(self, accessor):
        outcome = _outcome_in_child('p = dtwcpp.Problem("empty")', f"p.{accessor}(0)")
        assert re.match(rf"InvalidInput: {accessor}: i = 0\b.*N = 0", outcome), outcome

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
        outcome = _outcome_in_child(self._UNCLUSTERED, f"p.centroid_of({i})")
        assert re.match(rf"InvalidInput: centroid_of: .*{message}", outcome), outcome

    def test_centroid_of_after_the_cluster_count_shrinks_raises(self):
        """The clustering held 2 medoids; k = 1 leaves a series in cluster 1 with
        no medoid to name, so the Problem no longer holds a clustering."""
        setup = self._CLUSTERED + "\n        i = p.labels().tolist().index(1)\n        p.set_n_clusters(1)"
        outcome = _outcome_in_child(setup, "p.centroid_of(i)")
        assert re.match(r"InvalidInput: centroid_of: .*holds no clustering.*2 medoids for k = 1", outcome), outcome


class TestClusterFirst:
    """A Problem that was only sized holds no clustering. `find_total_cost()` and
    `write_clusters()` read the label vector, and crashed the interpreter (an
    access violation) on a Problem that had never been clustered; every call that
    reads the whole clustering now raises InvalidInput naming the call."""

    _DATA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
    _SETUP = f"""
        p = dtwcpp.Problem("first")
        p.set_data({_DATA}, ["a", "b", "c"])
        p.output_folder = r"{{out}}"
    """

    def _outcome(self, tmp_path, extra_setup, call):
        setup = textwrap.dedent(self._SETUP).replace("{out}", str(tmp_path)) + "\n" + extra_setup
        return _outcome_in_child(setup, call)

    @pytest.mark.parametrize("sized", ["", "p.set_n_clusters(2)"], ids=["never_sized", "sized"])
    @pytest.mark.parametrize(
        "call, who",
        [("p.find_total_cost()", "find_total_cost"),
         ("p.write_clusters()", "write_clusters"),
         ("p.write_medoid_members(0)", "write_medoid_members"),
         ("p.calculate_medoids()", "calculate_medoids"),
         ("dtwcpp.silhouette(p)", "silhouette"),
         ("dtwcpp.inertia(p)", "inertia")],
    )
    def test_reading_the_clustering_before_clustering_raises(self, tmp_path, sized, call, who):
        outcome = self._outcome(tmp_path, sized, call)
        assert re.match(rf"InvalidInput: {who}: .*holds no clustering.*cluster it first", outcome), outcome
        assert list(tmp_path.iterdir()) == []  # the refusal comes before any file is opened

    def test_the_refusal_names_the_counts_it_found(self, tmp_path):
        outcome = self._outcome(tmp_path, "p.set_n_clusters(2)", "p.find_total_cost()")
        assert "0 labels for N = 3, 0 medoids for k = 2" in outcome, outcome

    def test_a_clustering_goes_stale_when_the_cluster_count_changes(self, tmp_path):
        setup = "p.set_n_clusters(2)\np.fill_distance_matrix()\np.cluster()\np.set_n_clusters(3)"
        outcome = self._outcome(tmp_path, setup, "p.find_total_cost()")
        assert re.match(r"InvalidInput: find_total_cost: .*2 medoids for k = 3", outcome), outcome

    @pytest.mark.parametrize(
        "call, who",
        [("p.find_total_cost()", "find_total_cost"),
         ("dtwcpp.silhouette(p)", "silhouette"),
         ("dtwcpp.inertia(p)", "inertia")],
    )
    def test_replacing_the_series_with_as_many_drops_the_clustering(self, tmp_path, call, who):
        """The old labels describe the old series, so a same-N set_data must not leave them counting."""
        setup = "\n".join([
            "p.set_n_clusters(2)",
            "p.cluster()",
            "p.find_total_cost()",
            "p.set_data([[7.0, 7.5, 8.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]], ['a', 'b', 'c'])",
        ])
        outcome = self._outcome(tmp_path, setup, call)
        assert re.match(rf"InvalidInput: {who}: .*holds no clustering.*cluster it first", outcome), outcome

    def test_after_clustering_the_readers_work(self, tmp_path):
        p = dtwcpp.Problem("first")
        p.set_data(self._DATA, ["a", "b", "c"])
        p.output_folder = str(tmp_path)
        p.set_n_clusters(2)
        p.cluster()
        assert p.find_total_cost() >= 0.0
        p.write_clusters()
        assert [f.name for f in tmp_path.iterdir()] == ["first_Nc_2.csv"]


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
        # variant_params returns a copy: editing it leaves the Problem as it was.
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
    """Tests for the band property."""

    def test_set_band(self):
        """Band can be set and read back."""
        p = dtwcpp.Problem("test")
        p.band = 5
        assert p.band == 5

    def test_band_affects_distance(self):
        """Setting a tight band can change computed distances."""
        data = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
        names = ["a", "b"]

        # Full DTW
        p1 = dtwcpp.Problem("full")
        p1.set_data(data, names)
        p1.fill_distance_matrix()
        d_full = p1.dist_by_ind(0, 1)

        # Banded DTW
        p2 = dtwcpp.Problem("banded")
        p2.band = 1
        p2.set_data(data, names)
        p2.fill_distance_matrix()
        d_banded = p2.dist_by_ind(0, 1)

        # Banded should be >= full (or equal for trivial cases)
        assert d_banded >= d_full - 1e-10

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


class TestVariant:
    """Tests for DTW variant selection on Problem."""

    def test_set_variant_enum(self):
        """set_variant accepts a DTWVariant enum and updates variant_params."""
        p = dtwcpp.Problem("test")
        p.set_variant(dtwcpp.DTWVariant.WDTW)
        assert p.variant_params.variant == dtwcpp.DTWVariant.WDTW

    def test_variant_params_fields(self):
        """variant_params fields can be set via the property."""
        p = dtwcpp.Problem("test")
        p.set_variant(dtwcpp.DTWVariant.WDTW)
        vp = p.variant_params
        vp.wdtw_g = 0.1
        p.variant_params = vp
        assert p.variant_params.wdtw_g == pytest.approx(0.1)

    def test_variant_changes_distances(self):
        """Using WDTW variant produces different distances than standard."""
        data = [[1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0]]
        names = ["a", "b"]

        # Standard DTW
        p1 = dtwcpp.Problem("standard")
        p1.set_data(data, names)
        p1.fill_distance_matrix()
        d_std = p1.dist_by_ind(0, 1)

        # WDTW
        p2 = dtwcpp.Problem("wdtw")
        p2.set_variant(dtwcpp.DTWVariant.WDTW)
        p2.set_data(data, names)
        p2.fill_distance_matrix()
        d_wdtw = p2.dist_by_ind(0, 1)

        # WDTW should differ from standard
        assert d_wdtw != pytest.approx(d_std, abs=1e-6)

