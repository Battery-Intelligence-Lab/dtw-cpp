"""
@file test_api.py
@brief Tests for the unified device()/load()/cluster()/result.plot() interface.
@author Volkan Kumtepeli
"""
import subprocess
from types import SimpleNamespace

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


def _seed_sensitive_series():
    """Ambiguous nonconstant waveforms whose PAM local optimum depends on seed."""
    base = np.array([0.0, 0.01, -0.02, 0.03])
    return base[None, :] + np.arange(8.0)[:, None]


def _assert_portable_lloyd_result(result):
    """Pin the portable-v1 seed-42 Lloyd result on the shared seed fixture."""
    np.testing.assert_array_equal(result.medoids, [5, 2, 0])
    np.testing.assert_array_equal(result.labels, [2, 1, 1, 1, 0, 0, 0, 0])
    assert result.cost == 24.0


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

    def test_skip_rows_drops_leading_file_lines(self, tmp_path):
        """§1.2 parity with C++ load(..., skip_rows) and dtwc_cl --skip-rows."""
        csv = tmp_path / "hdr.csv"
        csv.write_text(
            "id,t0,t1\nunit,s,s\n1,0,0\n2,10,11\n", encoding="utf-8")
        ds = dtwcpp.load(csv, skip_cols=1, skip_rows=2, delimiter=",")
        assert ds.skip_rows == 2
        assert ds.as_series() == [[0.0, 0.0], [10.0, 11.0]]

    def test_skip_rows_drops_leading_series_in_memory(self):
        ds = dtwcpp.load([[7.0, 7.0], [7.0, 7.0], [0.0, 1.0]], skip_rows=2)
        assert ds.as_series() == [[0.0, 1.0]]

    @pytest.mark.parametrize("bad,error", [(-1, ValueError), (1.0, TypeError)])
    def test_invalid_skip_rows_is_rejected_by_load(self, bad, error):
        """As C++ dtwc::load: refused where the handle is made, before any read."""
        with pytest.raises(error, match="skip_rows"):
            dtwcpp.load([[0.0], [1.0]], skip_rows=bad)

    @pytest.mark.parametrize("content", [None, "1,2,x\n4,5,6\n"])
    def test_reader_errors_name_the_load_and_keep_their_type(self, tmp_path,
                                                             content):
        """§1.2 / §5: a file load() cannot read raises IOError prefixed
        ``load: failed to read '<path>':``, as C++ dtwc::load does."""
        path = tmp_path / ("missing.csv" if content is None else "bad.csv")
        if content is not None:
            path.write_text(content, encoding="utf-8")
        with pytest.raises(dtwcpp.IOError) as caught:
            dtwcpp.cluster(dtwcpp.load(path), k=1, device="cpu")
        assert type(caught.value) is dtwcpp.IOError
        assert str(caught.value).startswith(f"load: failed to read '{path}': ")

    def test_parquet_path_reads_like_the_same_csv(self, tmp_path):
        """load('x.parquet') reads through the installed pyarrow (the wheel
        links no Arrow C++; it used to parse the file as CSV): the same series
        and names as the same data in CSV."""
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        rows = [[0.0, 0.5], [2.5, 1.0, 0.25], [9.0, 9.5]]
        csv = tmp_path / "x.csv"
        csv.write_text("".join(",".join(map(repr, row)) + "\n" for row in rows),
                       encoding="utf-8")
        from_csv = dtwcpp.load(csv)
        parquet = tmp_path / "x.parquet"
        pq.write_table(pa.table({
            "series": pa.array(rows, type=pa.list_(pa.float64())),
            "name": from_csv.series_names()}), parquet)
        from_parquet = dtwcpp.load(parquet)
        assert from_parquet.as_series() == from_csv.as_series() == rows
        assert from_parquet.series_names() == from_csv.series_names() == ["1", "2", "3"]

    def test_arrow_ipc_path_reads_its_data_and_name_columns(self, tmp_path):
        """load('x.arrow') reads through pyarrow as dtwc_cl reads Arrow IPC: the
        series are the 'data' column, named by 'name', 'ndim' features a step."""
        pa = pytest.importorskip("pyarrow")
        rows = [[0.0, 0.5, 1.0, 1.5], [2.5, 1.0]]
        table = pa.table({"data": pa.array(rows, type=pa.list_(pa.float64())),
                          "name": ["a", "b"]}).replace_schema_metadata({"ndim": "2"})
        path = tmp_path / "x.arrow"
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        data = dtwcpp.load(path).as_data()
        assert data.p_vec == rows
        assert data.p_names == ["a", "b"]
        assert data.ndim == 2

    def test_path_source_parses_a_non_numeric_id_column(self, tmp_path):
        """§1.2: skip_cols drops FIELDS before numeric parsing, as C++ does."""
        csv = tmp_path / "named.csv"
        csv.write_text("alpha,0,0\nbeta,10,11\n", encoding="utf-8")
        ds = dtwcpp.load(csv, skip_cols=1)
        assert ds.as_series() == [[0.0, 0.0], [10.0, 11.0]]

    def test_path_source_supports_ragged_rows(self, tmp_path):
        """C++ DataLoader stores variable-length series; Python must too."""
        csv = tmp_path / "ragged.csv"
        csv.write_text("0,1,2\n3,4\n", encoding="utf-8")
        assert dtwcpp.load(csv).as_series() == [[0.0, 1.0, 2.0], [3.0, 4.0]]

    def test_in_memory_source_honours_skip_cols(self):
        """C++ erases the leading columns of in-memory rows (api.cpp)."""
        ds = dtwcpp.load([[9.0, 0.0, 1.0], [9.0, 2.0, 3.0]], skip_cols=1)
        assert ds.as_series() == [[0.0, 1.0], [2.0, 3.0]]

    def test_in_memory_skip_cols_beyond_series_length_is_rejected(self):
        ds = dtwcpp.load([[0.0, 1.0]], skip_cols=3)
        with pytest.raises(dtwcpp.InvalidInput,
                           match="skip_cols exceeds an in-memory series length"):
            ds.as_series()


# ---------------------------------------------------------------------------
# cluster() keywords: read by C++ (a Config) before any data is read
# ---------------------------------------------------------------------------
class TestClusterKeywords:
    """The keywords become a C++ Config: C++ reads and checks each name before
    the series are read or a job is submitted."""

    def test_valid_minimum_numpy_integers_run_locally(self):
        source = dtwcpp.Dataset(
            [[0.0], [1.0]], skip_cols=np.int32(0),
        )
        result = dtwcpp.cluster(
            source, k=np.int32(1), max_iter=np.int64(1), device="cpu",
        )

        assert result.n_series == 2
        assert result.k == 1
        assert type(result.k) is int

    @pytest.mark.parametrize("bad", [True, np.bool_(True), "1", 1.5, np.float32(1.9)])
    def test_a_value_of_another_kind_is_refused(self, bad):
        """The binding's casters would read True or "1" as 1 and truncate a NumPy
        float: an integer key takes an integer."""
        with pytest.raises(TypeError, match=r"^k must be an integer"):
            dtwcpp.cluster([[0.0], [1.0]], k=bad)
        with pytest.raises(TypeError, match=r"^max_iter must be an integer"):
            dtwcpp.cluster([[0.0], [1.0]], k=1, max_iter=bad)

    def test_an_unknown_key_lists_each_key_cluster_takes_once(self):
        with pytest.raises(dtwcpp.InvalidInput, match="unknown key 'bogus'") as caught:
            dtwcpp.cluster([[0.0], [1.0]], k=1, bogus=1)
        keys = str(caught.value).split("Valid keys: ")[1].rstrip(".").split(", ")
        assert len(keys) == len(set(keys))
        assert "device" in keys and "n_clusters" not in keys  # k is the cluster count

    def test_unknown_method_still_fails_before_load_or_device(self, monkeypatch):
        from dtwcpp import _api

        monkeypatch.setattr(
            _api, "load",
            lambda *args, **kwargs: pytest.fail("unknown method triggered load"),
        )
        monkeypatch.setattr(
            dtwcpp, "_resolve_device",
            lambda *args, **kwargs: pytest.fail("unknown method resolved device"),
        )
        with pytest.raises(ValueError, match="unknown method"):
            dtwcpp.cluster("must_not_be_loaded.tsv", k=1, method="bogus")

    def test_invalid_device_still_fails_before_load(self, monkeypatch):
        from dtwcpp import _api

        monkeypatch.setattr(
            _api, "load",
            lambda *args, **kwargs: pytest.fail("invalid device triggered load"),
        )
        with pytest.raises(dtwcpp.DeviceError, match="unknown device"):
            dtwcpp.cluster(
                "must_not_be_loaded.tsv", k=1, device="definitely-not-a-device",
            )


# ---------------------------------------------------------------------------
# cluster() — local cpu path
# ---------------------------------------------------------------------------
class TestClusterLocal:
    def test_k_above_series_count_is_rejected(self):
        """Parity with C++ cluster(): k must not exceed the number of series."""
        with pytest.raises(dtwcpp.InvalidInput,
                           match="k must not exceed the number of series"):
            dtwcpp.cluster([[0.0], [1.0]], k=3)

    def test_empty_dataset_is_rejected(self):
        with pytest.raises(dtwcpp.InvalidInput, match="dataset is empty"):
            dtwcpp.cluster([], k=1)

    def test_recovers_two_groups(self):
        res = dtwcpp.cluster(_two_groups(), k=2)
        assert res.n_series == 12
        assert len(set(res.labels[:6])) == 1
        assert len(set(res.labels[6:])) == 1
        assert res.labels[0] != res.labels[11]

    def test_default_pam_seed_is_local_and_matches_cpp_tier1(self):
        assert dtwcpp.DEFAULT_RANDOM_SEED == 42

        series = _seed_sensitive_series()
        names = [str(i) for i in range(len(series))]

        def seeded(seed, max_iter=100):
            problem = dtwcpp.Problem("seed_oracle")
            problem.set_data(series.tolist(), names)
            return dtwcpp.fast_pam_seeded(problem, 3, seed, max_iter)

        init_29 = seeded(29, max_iter=0)
        init_42 = seeded(42, max_iter=0)
        assert list(init_29.medoid_indices) == [4, 2, 7]
        assert list(init_42.medoid_indices) == [6, 2, 5]

        final_29 = seeded(29)
        final_42 = seeded(42)
        assert list(final_29.medoid_indices) == [4, 1, 7]
        assert final_29.total_cost == 20.0
        assert list(final_42.medoid_indices) == [6, 2, 5]
        assert final_42.total_cost == 24.0

        first = dtwcpp.cluster(series, k=3, method="pam")

        # Consume the mutable legacy engine through the unseeded Tier-2 API.
        legacy_problem = dtwcpp.Problem("legacy_rng_consumer")
        legacy_problem.set_data(series.tolist(), names)
        dtwcpp.fast_pam(legacy_problem, 3)

        second = dtwcpp.cluster(series, k=3, method="pam")
        for result in (first, second):
            np.testing.assert_array_equal(result.medoids, final_42.medoid_indices)
            np.testing.assert_array_equal(result.labels, final_42.labels)
            assert result.cost == final_42.total_cost

    def test_default_lloyd_seed_is_reproducible_across_calls(self):
        series = _seed_sensitive_series()

        first = dtwcpp.cluster(series, k=3, method="kmedoids")
        second = dtwcpp.cluster(series, k=3, method="kmedoids")
        _assert_portable_lloyd_result(first)
        _assert_portable_lloyd_result(second)
        assert dtwcpp.Problem().random_seed == dtwcpp.DEFAULT_RANDOM_SEED

    def test_default_lloyd_seed_isolated_from_legacy_tier2_rng(self):
        series = _seed_sensitive_series()
        names = [str(i) for i in range(len(series))]

        before = dtwcpp.cluster(series, k=3, method="kmedoids")

        # The unseeded Tier-2 FastPAM entry point deliberately retains its
        # mutable-global RNG contract. Consuming it must not perturb Tier-1
        # Lloyd's invocation-local default.
        legacy_problem = dtwcpp.Problem("legacy_rng_consumer")
        legacy_problem.set_data(series.tolist(), names)
        dtwcpp.fast_pam(legacy_problem, 3)

        after = dtwcpp.cluster(series, k=3, method="kmedoids")
        _assert_portable_lloyd_result(before)
        _assert_portable_lloyd_result(after)

    def test_lloyd_honors_nondefault_iteration_cap_and_keeps_default(self):
        # Seed 42 starts at medoids [4,2]. One Lloyd update publishes [4,1];
        # convergence needs a second update to the unique median at index 5.
        series = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 4.0])[:, None]

        capped = dtwcpp.cluster(
            series, k=2, method="kmedoids", max_iter=1,
        )
        default = dtwcpp.cluster(series, k=2, method="kmedoids")

        # Exact L1 oracles distinguish forwarding max_iter=1 from silently
        # retaining Problem's default 100.
        np.testing.assert_array_equal(capped.medoids, [4, 1])
        np.testing.assert_array_equal(capped.labels, [1, 1, 1, 0, 0, 0])
        assert capped.cost == 5.0
        np.testing.assert_array_equal(default.medoids, [5, 1])
        np.testing.assert_array_equal(default.labels, capped.labels)
        assert default.cost == 4.0

    def test_result_fields_populated(self):
        res = dtwcpp.cluster(_two_groups(), k=2)
        assert res.device == "cpu"
        assert res.cost is not None
        assert res.distance_matrix is not None
        assert res.medoids is not None      # canonical 2.0 name (§1.4)
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


class TestMatrixFreeBand:
    @pytest.mark.parametrize("method", ["onebatch", "clara", "tadpole"])
    def test_band_reaches_matrix_free_problem_distance(self, method):
        """Matrix-free Tier-1 methods must use the requested Sakoe-Chiba band.

        This runs each real algorithm end-to-end rather than spying on a
        setter.  The irregular pair contains an eight-step warp, so its
        registered L1 DTW distances differ materially between full and
        width-five paths and therefore so do the one-cluster costs.
        """
        x = [0.2, -0.1, 1.4, 3.2, 7.1, 12.3, 9.2, 4.4,
             1.1, -0.3, 0.5, -0.8, 0.2, 0.7, -0.4, 0.9,
             -0.2, 0.3, -0.7, 0.4, -0.1, 0.6, -0.5, 0.8]
        y = [-0.4, -0.2, 0.1, -0.3, 0.4, -0.1, 0.2, 0.0,
             0.35, 0.05, 1.55, 3.35, 7.25, 12.45, 9.35, 4.55,
             1.25, -0.15, 0.65, -0.65, 0.35, 0.85, -0.25, 1.05]

        full = dtwcpp.cluster([x, y], k=1, method=method, band=-1)
        banded = dtwcpp.cluster([x, y], k=1, method=method, band=5)

        # Matrix-free: the run itself materialises nothing (the cache is
        # empty); reading the property is an explicit N^2 request (F5).
        assert full._distance_matrix is None
        assert banded._distance_matrix is None
        assert full.cost == pytest.approx(8.6, abs=1e-12)
        assert banded.cost == pytest.approx(63.85, abs=1e-12)


class TestMatrixFreeScoring:
    """C++ Result::score/save fill the retained Problem lazily (api.cpp)."""

    _SERIES = [[0.0], [0.5], [1.0], [8.0], [8.5], [9.0]]

    @pytest.mark.parametrize("method", ["onebatch", "clara", "tadpole"])
    @pytest.mark.parametrize(
        "score", ["silhouette", "davies_bouldin", "dunn", "inertia"])
    def test_score_is_available_after_a_matrix_free_run(self, method, score):
        """The lazily filled matrix must score identically to an eager one."""
        res = dtwcpp.cluster(self._SERIES, k=2, method=method)
        assert res._distance_matrix is None
        oracle = dtwcpp.Problem("eager")
        oracle.set_data(self._SERIES, [str(i) for i in range(len(self._SERIES))])
        oracle.set_distance_matrix(dtwcpp.compute_distance_matrix(self._SERIES))
        clustering = dtwcpp.ClusteringResult()
        clustering.labels = res.labels
        clustering.medoid_indices = res.medoids
        oracle.set_result(clustering)
        expected = getattr(dtwcpp, score)(oracle)
        if score == "silhouette":
            expected = np.mean(expected)
        assert res.score(score) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("method", ["onebatch", "clara", "tadpole"])
    def test_distance_matrix_fills_on_demand_after_a_matrix_free_run(self, method):
        """F5: C++ Result::distance_matrix() fills the retained Problem.

        Reading the property is an explicit N^2 request; before this fix it
        stayed None forever and plot() refused a perfectly local cpu run.
        """
        res = dtwcpp.cluster(self._SERIES, k=2, method=method)
        assert res._distance_matrix is None          # nothing materialised yet
        filled = res.distance_matrix
        assert filled is not None
        assert res._distance_matrix is filled        # cached, filled once
        np.testing.assert_allclose(
            filled, dtwcpp.compute_distance_matrix(self._SERIES), atol=1e-12)

    def test_plot_works_after_a_matrix_free_cpu_run(self, tmp_path):
        """A cpu run always has a local matrix, so plot() must not refuse."""
        import matplotlib
        matplotlib.use("Agg")
        res = dtwcpp.cluster(self._SERIES, k=2, method="clara")
        out = tmp_path / "clara.png"
        assert res.plot(png=str(out), show=False) == str(out)
        assert out.exists()

    def test_unknown_score_still_rejected_after_matrix_free_run(self):
        res = dtwcpp.cluster(self._SERIES, k=2, method="clara")
        with pytest.raises(dtwcpp.InvalidInput, match="unknown score"):
            res.score("nope")

    def test_save_after_a_matrix_free_run_writes_all_four_files(self, tmp_path):
        res = dtwcpp.cluster(self._SERIES, k=2, method="clara")
        res.save(tmp_path)
        for suffix in ("_labels.csv", "_medoids.csv", "_distance_matrix.csv",
                       "_silhouettes.csv"):
            assert (tmp_path / f"{res.name}{suffix}").exists()

    def test_hpc_result_still_reports_the_missing_matrix(self):
        res = dtwcpp.Result([0, 0, 1, 1], device="hpc", elapsed_s=1.0,
                            k=2, n_series=4)
        with pytest.raises(dtwcpp.InvalidInput, match="no local distance matrix"):
            res.score("silhouette")


# ---------------------------------------------------------------------------
# §1.4 save() with an undefined silhouette — warn and skip, never propagate
# ---------------------------------------------------------------------------
class TestSaveUndefinedSilhouette:
    """C++ ``Result::save`` catches ``UndefinedScore``, warns, skips the file.

    ``score("silhouette")`` keeps raising: asking for the number is a different
    contract (api.cpp:281-299, api-contract-2.0.md §1.4).
    """

    _SERIES = [[0.0, 0.1], [0.5, 0.4], [1.0, 1.1], [8.0, 8.2]]

    def test_undefined_score_is_a_bound_leaf_under_invalid_input(self):
        assert issubclass(dtwcpp.UndefinedScore, dtwcpp.InvalidInput)
        assert issubclass(dtwcpp.UndefinedScore, dtwcpp.DtwcError)
        assert issubclass(dtwcpp.UndefinedScore, ValueError)

    def test_silhouette_of_one_cluster_raises_undefined_score(self):
        prob = dtwcpp.Problem("one")
        prob.set_data(self._SERIES, [str(i) for i in range(len(self._SERIES))])
        prob.set_distance_matrix(dtwcpp.compute_distance_matrix(self._SERIES))
        result = dtwcpp.ClusteringResult()
        result.labels = [0] * len(self._SERIES)
        result.medoid_indices = [0]
        prob.set_result(result)
        with pytest.raises(dtwcpp.UndefinedScore, match="at least 2 non-empty"):
            dtwcpp.silhouette(prob)

    def test_save_with_one_cluster_skips_the_silhouettes_file_silently(
            self, tmp_path, capsys):
        """C++ Result::save and the CLI skip the file for one cluster, silently.

        A Python ``warnings.warn`` here would turn a *successful* save into an
        exception under ``-W error``; the CLI does not fail, so neither may we.
        """
        import warnings

        res = dtwcpp.cluster(self._SERIES, k=1, method="pam")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res.save(tmp_path)
        assert capsys.readouterr().err == ""
        for suffix in ("_labels.csv", "_medoids.csv", "_distance_matrix.csv"):
            assert (tmp_path / f"{res.name}{suffix}").is_file()
        assert not (tmp_path / f"{res.name}_silhouettes.csv").exists()

    def test_score_silhouette_still_raises_undefined_score(self):
        res = dtwcpp.cluster(self._SERIES, k=1, method="pam")
        with pytest.raises(dtwcpp.UndefinedScore):
            res.score("silhouette")


# ---------------------------------------------------------------------------
# §1.2 ragged in-memory sources — C++ load(series_type) takes variable lengths
# ---------------------------------------------------------------------------
class TestRaggedInMemorySource:
    _RAGGED = [[0.0, 0.1, 0.2, 0.3], [0.05, 0.15], [9.0, 9.1, 9.2],
               [9.2, 9.05, 9.1, 9.3, 9.15]]

    def test_as_series_preserves_variable_lengths(self):
        assert dtwcpp.load(self._RAGGED).as_series() == self._RAGGED

    def test_cluster_runs_on_a_ragged_list(self):
        res = dtwcpp.cluster(self._RAGGED, k=2, method="pam")
        assert len(res.labels) == len(self._RAGGED)

    def test_labels_match_the_cpp_path_on_the_same_ragged_data(self):
        res = dtwcpp.cluster(self._RAGGED, k=2, method="pam")
        prob = dtwcpp.Problem("dataset")
        prob.set_band(-1)
        prob.set_data(self._RAGGED,
                      [str(i) for i in range(len(self._RAGGED))])
        prob.set_distance_matrix(dtwcpp.compute_distance_matrix(self._RAGGED))
        ref = dtwcpp.fast_pam_seeded(prob, 2, dtwcpp.DEFAULT_RANDOM_SEED, 100)
        np.testing.assert_array_equal(res.labels, ref.labels)
        np.testing.assert_array_equal(res.medoids, ref.medoid_indices)

    def test_skip_rows_and_skip_cols_apply_to_ragged_rows(self):
        ds = dtwcpp.load([[7.0, 7.0], [1.0, 0.0, 1.0], [2.0, 5.0]],
                         skip_rows=1, skip_cols=1)
        assert ds.as_series() == [[0.0, 1.0], [5.0]]

    def test_skip_cols_beyond_a_ragged_series_is_rejected(self):
        ds = dtwcpp.load([[0.0, 1.0, 2.0], [3.0]], skip_cols=2)
        with pytest.raises(dtwcpp.InvalidInput,
                           match="skip_cols exceeds an in-memory series length"):
            ds.as_series()


# ---------------------------------------------------------------------------
# Already-read data: as numpy, pandas or Python hold it
# ---------------------------------------------------------------------------
_TWO_GROUPS = [[0.0, 0.1, 0.2, 0.3], [0.05, 0.15, 0.1, 0.2], [9.0, 9.1, 9.2, 9.0],
               [9.2, 9.05, 9.1, 9.3]]


def _as(form):
    if form == "2-D array":
        return np.array(_TWO_GROUPS)
    if form == "list of 1-D arrays":
        return [np.array(row) for row in _TWO_GROUPS]
    if form == "list of lists":
        return _TWO_GROUPS
    if form == "Arrow array":
        pa = pytest.importorskip("pyarrow")
        return pa.array(_TWO_GROUPS, type=pa.list_(pa.float64()))
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(_TWO_GROUPS, index=["a", "b", "c", "d"])


@pytest.mark.parametrize("form", ["2-D array", "list of 1-D arrays", "list of lists",
                                  "Arrow array", "pandas DataFrame"])
def test_already_read_data_goes_in_as_it_is(form):
    """cluster(), DTWClustering.fit, Problem.set_data and load take each form,
    one series per row; a DataFrame's rows are named by its index, an Arrow
    array's as the Arrow converter names them."""
    data = _as(form)
    expected = dtwcpp.cluster(_TWO_GROUPS, k=2)
    np.testing.assert_array_equal(dtwcpp.cluster(data, k=2).labels, expected.labels)
    np.testing.assert_array_equal(
        dtwcpp.DTWClustering(n_clusters=2).fit(data).labels_, expected.labels)
    prob = dtwcpp.Problem("forms")
    prob.set_data(data)
    assert [prob.series_name(i) for i in range(prob.size)] == {
        "pandas DataFrame": ["a", "b", "c", "d"],
        "Arrow array": ["series_0", "series_1", "series_2", "series_3"],
    }.get(form, ["0", "1", "2", "3"])
    assert dtwcpp.load(data).series_names() == [prob.series_name(i) for i in range(prob.size)]


_ENTRIES = {
    "cluster": lambda x: dtwcpp.cluster(x, k=1),
    "load": lambda x: dtwcpp.load(x).as_data(),
    "Problem.set_data": lambda x: dtwcpp.Problem("p").set_data(x),
    "DTWClustering.fit": lambda x: dtwcpp.DTWClustering(n_clusters=1).fit(x),
    "DTWClustering.predict": lambda x: dtwcpp.DTWClustering(n_clusters=1).fit(_TWO_GROUPS).predict(x),
    "compute_distance_matrix": lambda x: dtwcpp.compute_distance_matrix(x),
}


@pytest.mark.parametrize("entry", list(_ENTRIES))
@pytest.mark.parametrize(("data", "message"), [
    (np.array(_TWO_GROUPS) + 1j, "Complex data not supported"),
    (np.array(_TWO_GROUPS[0]), "2-D array"),
    (np.array(_TWO_GROUPS)[:, :, None], "2-D array"),
], ids=["complex", "1-D", "3-D"])
def test_what_is_not_series_is_refused_everywhere(entry, data, message):
    """One conversion behind every entry: complex values are refused, never cast
    to their real part, and a 1-D or 3-D array is refused naming the forms taken."""
    with pytest.raises(TypeError, match=message):
        _ENTRIES[entry](data)


# ---------------------------------------------------------------------------
# §1.4 series names — Tier-1 output carries the loader's names, as C++ does
# ---------------------------------------------------------------------------
class TestSeriesNames:
    """``Problem::series_name(i)`` comes from the loader, not from ``range(N)``."""

    def test_batch_file_names_are_the_loader_row_numbers(self, tmp_path):
        csv = tmp_path / "batch.csv"
        csv.write_text("0,1\n2,3\n4,5\n", encoding="utf-8")
        assert dtwcpp.load(csv).series_names() == ["1", "2", "3"]

    def test_folder_names_are_file_stems(self, tmp_path):
        # A folder holds one series per file, one value per line: the reader
        # rejects a multi-field line there (FX-6).
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "alpha.csv").write_text("0\n1\n2\n", encoding="utf-8")
        (folder / "beta.csv").write_text("9\n8\n7\n", encoding="utf-8")
        assert dtwcpp.load(folder).series_names() == ["alpha", "beta"]

    def test_in_memory_names_are_the_zero_based_ordinals(self):
        assert dtwcpp.load([[0.0], [1.0]]).series_names() == ["0", "1"]

    def test_a_folder_given_with_a_trailing_separator_names_the_run(self, tmp_path):
        """The run is named as dtwc_cl names it (C++ detail::default_name):
        "data/" is "data", so save() writes data_labels.csv, not _labels.csv."""
        folder = tmp_path / "data"
        folder.mkdir()
        (folder / "a.csv").write_text("0\n1\n", encoding="utf-8")
        (folder / "b.csv").write_text("9\n8\n", encoding="utf-8")
        source = str(folder) + "/"
        assert dtwcpp.load(source).name == "data"
        dtwcpp.cluster(source, k=1).save(tmp_path / "out")
        assert (tmp_path / "out" / "data_labels.csv").is_file()

    def test_saved_labels_carry_the_file_names(self, tmp_path):
        csv = tmp_path / "named.csv"
        csv.write_text("0,0.1\n0.2,0.1\n9,9.1\n9.2,9.0\n", encoding="utf-8")
        res = dtwcpp.cluster(dtwcpp.load(csv), k=2, method="pam")
        res.save(tmp_path)
        lines = (tmp_path / "named_labels.csv").read_text().splitlines()
        assert [line.split(",")[0] for line in lines[1:]] == ["1", "2", "3", "4"]

    def test_save_is_byte_identical_to_the_cli(self, tmp_path, dtwc_cl):
        """A CLI run and a Python run on one file must write the same bytes."""
        csv = tmp_path / "parity.csv"
        np.savetxt(csv, _two_groups(), delimiter=",")
        cli_out = tmp_path / "cli"
        py_out = tmp_path / "py"
        run = subprocess.run(
            [dtwc_cl, "-i", str(csv), "-o", str(cli_out), "--name", "parity",
             "-k", "2", "-m", "pam"],
            capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        dtwcpp.cluster(dtwcpp.load(csv), k=2, method="pam").save(py_out)
        for suffix in ("_labels.csv", "_medoids.csv", "_silhouettes.csv",
                       "_distance_matrix.csv"):
            assert (py_out / f"parity{suffix}").read_bytes() == \
                (cli_out / f"parity{suffix}").read_bytes(), suffix


class TestNonAsciiSeriesNames:
    """F1: a folder holding a non-ASCII file name must round-trip as UTF-8."""

    @staticmethod
    def _folder(tmp_path):
        # One series per file, one value per line (a multi-field line in a
        # one-series file is rejected by the reader, FX-6).
        folder = tmp_path / "uni"
        folder.mkdir()
        for stem, values in (("caf\u00e9", ("0", "0.1")), ("beta", ("0.2", "0.1")),
                             ("gamma", ("9", "9.1")), ("delta", ("9.2", "9.0"))):
            (folder / f"{stem}.csv").write_text(
                "".join(v + "\n" for v in values), encoding="utf-8")
        return folder

    def test_load_decodes_a_non_ascii_file_stem(self, tmp_path):
        names = dtwcpp.load(self._folder(tmp_path)).series_names()
        assert "caf\u00e9" in names

    def test_save_is_byte_identical_to_the_cli_for_a_non_ascii_folder(
            self, tmp_path, dtwc_cl):
        """The four CSVs must be cmp-identical to dtwc_cl on a non-ASCII name.

        C++ emits the loader name as UTF-8 bytes through a text-mode ofstream;
        Result.save must therefore write UTF-8 with the platform line ending,
        not the locale encoding.
        """
        folder = self._folder(tmp_path)
        cli_out = tmp_path / "cli"
        py_out = tmp_path / "py"
        run = subprocess.run(
            [dtwc_cl, "-i", str(folder), "-o", str(cli_out), "--name", "uni",
             "-k", "2", "-m", "pam"],
            capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        dtwcpp.cluster(dtwcpp.load(folder), k=2, method="pam").save(py_out)
        for suffix in ("_labels.csv", "_medoids.csv", "_silhouettes.csv",
                       "_distance_matrix.csv"):
            assert (py_out / f"uni{suffix}").read_bytes() == \
                (cli_out / f"uni{suffix}").read_bytes(), suffix


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
        res = dtwcpp.Result([0, 0, 1, 1], device="hpc", elapsed_s=1.0,
                            k=2, n_series=4)
        assert res.plot(show=False) is None
        assert "no local distance matrix" in capsys.readouterr().out


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

        Pre-fix: method is ignored, FastPAM runs, a Result is returned
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

    def test_local_clara_runs_end_to_end(self):
        """The clara branch must actually work end-to-end (no solver needed).

        Recovers the two well-separated groups, proving real dispatch — not
        just that a non-ValueError was returned."""
        res = dtwcpp.cluster(_two_groups(), k=2, method="clara")
        assert res.n_series == 12
        # CLARA's scaling contract is O(Ns), not O(N²): Tier 1 must not
        # materialise a full matrix merely to populate an auxiliary result field.
        assert res._distance_matrix is None
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

        def fake(data, config, keys, **kwargs):
            captured["method"] = config.method
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

        def fake(data, config, keys, **kwargs):
            captured["method"] = config.method
            return np.zeros(4, dtype=int)

        monkeypatch.setattr(_hpc, "cluster_on_hpc", fake)
        dtwcpp.cluster([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                       k=2, device="hpc", method="hclust")
        assert captured["method"] == "hierarchical"


# ---------------------------------------------------------------------------
# Result write-back moved to the C++ core (api-contract-2.0.md §2.5, Task 1.6/2.1)
#
# 1.x wired labels/medoids/k back into Problem inside the *binding* lambdas
# (_dtwcpp_core.cpp fast_pam/fast_clara).
# Task 2.1 DELETES that wrapper-side wiring — the C++ algorithm free functions now
# do it (fast_pam.cpp, fast_clara.cpp, hierarchical.cpp). These tests pin the behaviour END-TO-END: after a
# REAL algorithm call on a REAL Problem, with NO Python-side assignment to
# clusters_ind/centroids_ind, the results are already visible on the Problem
# (labels()/medoids()) and the scores read them. If the write-back regressed,
# silhouette(prob) would see empty/stale state and these fail. The mechanism
# moved to C++; the end-to-end behaviour is preserved and asserted here.
# ---------------------------------------------------------------------------
class TestResultWriteBackInCpp:
    @staticmethod
    def _filled_problem(seed=1, n=12):
        rng = np.random.default_rng(seed)
        X = [list(rng.standard_normal(10) * 0.1 + (0.0 if i < n // 2 else 9.0))
             for i in range(n)]
        p = dtwcpp.Problem("wb")
        p.set_data(X, [str(i) for i in range(n)])
        p.fill_distance_matrix()
        return p

    def test_fast_pam_writes_back_without_wrapper(self):
        """drives dtwcpp.fast_pam(prob, k) — C++ core writes labels/medoids/k back."""
        p = self._filled_problem()
        res = dtwcpp.fast_pam(p, 2)            # NO Python wiring after this call
        assert list(p.labels()) == list(res.labels)
        assert sorted(p.medoids()) == sorted(res.medoid_indices)
        assert p.n_clusters() == 2
        # Scores read Problem state — only works if the write-back happened.
        assert len(dtwcpp.silhouette(p)) == 12

    def test_fast_clara_writes_back_without_wrapper(self):
        """drives dtwcpp.fast_clara(prob, k)."""
        p = self._filled_problem()
        res = dtwcpp.fast_clara(p, 2)
        assert list(p.labels()) == list(res.labels)
        assert p.n_clusters() == 2
        assert len(dtwcpp.silhouette(p)) == 12

    def test_cut_dendrogram_writes_back_without_wrapper(self):
        """drives build_dendrogram + cut_dendrogram — 2.0 also writes back (§2.5)."""
        p = self._filled_problem()
        dend = dtwcpp.build_dendrogram(p)
        res = dtwcpp.cut_dendrogram(dend, p, 2)
        assert list(p.labels()) == list(res.labels)
        assert p.n_clusters() == 2

    def test_cluster_tier1_end_to_end_results_visible(self):
        """drives dtwcpp.cluster() Tier-1 — labels/medoids/score visible with NO
        wrapper wiring (the preserved end-to-end contract; must not weaken)."""
        rng = np.random.default_rng(7)
        X = np.array([rng.standard_normal(12) * 0.1 + (0.0 if i < 6 else 9.0)
                      for i in range(12)])
        res = dtwcpp.cluster(X, k=2)
        assert res.medoids is not None
        assert len(set(res.labels[:6])) == 1 and len(set(res.labels[6:])) == 1
        assert res.score("silhouette") > 0.5
