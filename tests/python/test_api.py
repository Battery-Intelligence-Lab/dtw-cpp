"""
@file test_api.py
@brief Tests for the unified device()/load()/cluster()/result.plot() interface.
@author Volkan Kumtepeli
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import dtwcpp

_READER = Path(__file__).resolve().parents[1] / "data" / "reader"


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
    def test_load_path_is_lazy_and_does_not_read(self):
        """A nonexistent path is fine until something materializes it."""
        ds = dtwcpp.load("definitely_missing_file.tsv")   # must NOT raise
        assert ds.is_path
        assert ds.name == "definitely_missing_file"

    def test_skip_rows_drops_leading_file_lines(self, tmp_path):
        """Parity with C++ load(..., skip_rows) and dtwc_cl --skip-rows."""
        csv = tmp_path / "hdr.csv"
        csv.write_text(
            "id,t0,t1\nunit,s,s\n1,0,0\n2,10,11\n", encoding="utf-8")
        ds = dtwcpp.load(csv, skip_cols=1, skip_rows=2, delimiter=",")
        assert ds.skip_rows == 2
        assert ds.as_series() == [[0.0, 0.0], [10.0, 11.0]]

    @pytest.mark.parametrize("bad,error", [(-1, ValueError), (1.0, TypeError)])
    def test_invalid_skip_rows_is_rejected_by_load(self, bad, error):
        """As C++ dtwc::load: refused where the handle is made, before any read."""
        with pytest.raises(error, match="skip_rows"):
            dtwcpp.load([[0.0], [1.0]], skip_rows=bad)

    def test_a_dataset_refuses_the_options_it_would_ignore(self):
        """load(Dataset, ...) returned the Dataset and dropped the options. A
        Dataset keeps the options it was made with, so one given is refused by
        name, as MATLAB's dtwc.load refuses it."""
        ds = dtwcpp.load([[0.0], [1.0], [2.0]])
        assert dtwcpp.load(ds) is ds
        with pytest.raises(dtwcpp.InvalidInput, match=r"so skip_rows, name would be ignored"):
            dtwcpp.load(ds, skip_rows=1, name="x")

    def test_arrow_ipc_path_reads_its_data_and_name_columns(self, tmp_path):
        """load('x.arrow') reads through the C++ reader, as dtwc_cl reads Arrow IPC:
        the series are the 'data' column, named by 'name', 'ndim' features a step,
        and the Data is a view of the file, its p_vec and p_names read through it."""
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

    @pytest.mark.parametrize("suffix", [".csv", ".parquet"])
    def test_float32_load_clusters_as_the_from_float32_route(self, tmp_path, suffix):
        """dtype="float32" stores float32 Data, each value the float64 read rounded
        (text: the C++ reader; Parquet: pyarrow's cast), and clusters as today's
        float32 route, Data.from_float32 of the float64 read, does."""
        rng = np.random.default_rng(5)
        rows = [rng.standard_normal(30) * 100.0 + 1000.0 * (i % 3) for i in range(12)]
        path = tmp_path / f"series{suffix}"
        if suffix == ".csv":
            path.write_text("".join(",".join(repr(v) for v in row.tolist()) + "\n" for row in rows),
                            encoding="utf-8")
        else:
            pa = pytest.importorskip("pyarrow")
            pq = pytest.importorskip("pyarrow.parquet")
            pq.write_table(pa.table({"series": pa.array([row.tolist() for row in rows],
                                                        type=pa.list_(pa.float64()))}), path)
        dataset = dtwcpp.load(path, dtype="f32")
        assert dataset.dtype == "float32"
        assert dataset.as_data().is_f32()
        read = dtwcpp.load(path).as_series()
        rounded = [np.asarray(row, np.float32).astype(np.float64).tolist() for row in read]
        assert dataset.as_series() == rounded

        result = dtwcpp.cluster(path, k=3, method="pam", dtype="float32")
        oracle = dtwcpp.Problem("oracle")
        oracle.set_data(dtwcpp.Data.from_float32(read, [str(i) for i in range(len(read))]))
        expected = dtwcpp.fast_pam(oracle, 3)
        np.testing.assert_array_equal(result.labels, expected.labels)
        np.testing.assert_array_equal(result.medoids, expected.medoid_indices)
        assert result.cost == expected.total_cost
        assert result.cost != dtwcpp.cluster(path, k=3, method="pam").cost  # float32 ran


# ---------------------------------------------------------------------------
# cluster() keywords: read by C++ (a Config) before any data is read
# ---------------------------------------------------------------------------
class TestClusterKeywords:
    """The keywords become a C++ Config: C++ reads and checks each name before
    the series are read or a job is submitted."""

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

    def test_ram_limit_reads_bytes_or_a_size_as_cpp_does(self):
        """ram_limit, the one memory key: a byte count, or a size C++ reads as
        dtwc_cl reads --ram-limit; 0 is automatic."""
        config = dtwcpp._dtwcpp_core.Config()
        config.ram_limit = "1.5G"
        assert config.ram_limit == 1610612736
        prob = dtwcpp.Problem("ram")
        assert prob.ram_limit == 0
        prob.ram_limit = np.int64(4096)
        assert prob.ram_limit == 4096
        with pytest.raises(dtwcpp.InvalidInput, match="Invalid ram_limit '-1G'"):
            prob.ram_limit = "-1G"
        with pytest.raises(TypeError, match=r"^ram_limit must be a byte count"):
            dtwcpp.cluster([[0.0], [1.0]], k=1, ram_limit=1.5)
        assert len(dtwcpp.cluster([[0.0], [1.0], [5.0]], k=2, ram_limit="8G").labels) == 3

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

    def test_a_bad_value_is_refused_before_parquet_is_read(self, tmp_path):
        """C++ checks every value before any series is read, as dtwc::run does
        for a path it reads: pyarrow reads Parquet before the run, and not for
        a band the run would refuse (the missing file would be an IOError)."""
        with pytest.raises(dtwcpp.InvalidInput, match="band must be -1"):
            dtwcpp.cluster(str(tmp_path / "missing.parquet"), k=2, band=-5)

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
    def test_empty_dataset_is_rejected(self):
        with pytest.raises(dtwcpp.InvalidInput, match="dataset is empty"):
            dtwcpp.cluster([], k=1)

    def test_default_pam_seed_is_local_and_matches_cpp_tier1(self):
        assert dtwcpp.DEFAULT_RANDOM_SEED == 42

        series = _seed_sensitive_series()
        names = [str(i) for i in range(len(series))]

        def seeded(seed, max_iter=100):
            problem = dtwcpp.Problem("seed_oracle")
            problem.set_data(series.tolist(), names)
            return dtwcpp.fast_pam(problem, 3, max_iter=max_iter, seed=seed)

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

        # fast_pam's own default seed is the same 42.
        default_problem = dtwcpp.Problem("default_seed")
        default_problem.set_data(series.tolist(), names)
        assert list(dtwcpp.fast_pam(default_problem, 3).medoid_indices) == [6, 2, 5]

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
# save() with an undefined silhouette — warn and skip, never propagate
# ---------------------------------------------------------------------------
class TestSaveUndefinedSilhouette:
    """C++ ``Result::save`` catches ``UndefinedScore``, warns, skips the file.

    ``score("silhouette")`` keeps raising: asking for the number is a different
    contract (dtwc/api.cpp, Result::save).
    """

    _SERIES = [[0.0, 0.1], [0.5, 0.4], [1.0, 1.1], [8.0, 8.2]]

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
        with pytest.raises(dtwcpp.UndefinedScore, match="at least 2 non-empty"):
            res.score("silhouette")


# ---------------------------------------------------------------------------
# Ragged in-memory sources — C++ load(series_type) takes variable lengths
# ---------------------------------------------------------------------------
class TestRaggedInMemorySource:
    _RAGGED = [[0.0, 0.1, 0.2, 0.3], [0.05, 0.15], [9.0, 9.1, 9.2],
               [9.2, 9.05, 9.1, 9.3, 9.15]]

    def test_as_series_preserves_variable_lengths(self):
        assert dtwcpp.load(self._RAGGED).as_series() == self._RAGGED

    def test_labels_match_the_cpp_path_on_the_same_ragged_data(self):
        res = dtwcpp.cluster(self._RAGGED, k=2, method="pam")
        prob = dtwcpp.Problem("dataset")
        prob.set_band(-1)
        prob.set_data(self._RAGGED,
                      [str(i) for i in range(len(self._RAGGED))])
        prob.set_distance_matrix(dtwcpp.compute_distance_matrix(self._RAGGED))
        ref = dtwcpp.fast_pam(prob, 2, max_iter=100, seed=dtwcpp.DEFAULT_RANDOM_SEED)
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
# Series names — Tier-1 output carries the loader's names, as C++ does
# ---------------------------------------------------------------------------
class TestSeriesNames:
    """``Problem::series_name(i)`` comes from the loader, not from ``range(N)``."""

    def test_series_are_named_as_cpp_names_them(self, tmp_path):
        """One name per series in every language (Volkan 10-02). In memory,
        skip_rows drops leading series before they are read, and the rest are
        named 0, 1, ... as C++ dtwc::load names them (Dataset::materialize_local):
        an empty series dropped is not refused, and no name counts it. Parquet is
        read through pyarrow as the C++ reader (ParquetChunkReader) and MATLAB
        read it (test_io.py's fixtures hold its layouts): a folder's files in the
        text reader's order, the rows of a file without a string column named
        series_<i> numbered on across the files, a file's one float column one
        series named by its file (the folder of test_io_readers' Parquet case)."""
        rows = [[0.0, 0.5], [2.5, 1.0, 0.25], [9.0, 9.5]]
        assert dtwcpp.load([[]] + rows, skip_rows=1).series_names() == ["0", "1", "2"]
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        folder = tmp_path / "parquet"
        folder.mkdir()
        pq.write_table(pa.table({"series": pa.array(rows, type=pa.list_(pa.float64()))}),
                       folder / "a.parquet")
        pq.write_table(pa.table({"id": ["d", "e"], "v": pa.array([5.0, 6.0], pa.float32())}),
                       folder / "b.parquet")
        pq.write_table(pa.table({"series": pa.array([[7.0]], type=pa.list_(pa.float32()))}),
                       folder / "c.parquet")
        data = dtwcpp.load(folder).as_data()
        assert data.p_vec == rows + [[5.0, 6.0], [7.0]]
        assert data.p_names == ["series_0", "series_1", "series_2", "b", "series_4"]

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
    def test_plot_without_matrix_returns_none(self, capsys):
        """An hpc-style result (labels only) can't plot; it reports sizes."""
        res = dtwcpp.Result([0, 0, 1, 1], device="hpc", elapsed_s=1.0,
                            k=2, n_series=4)
        assert res.plot(show=False) is None
        assert "no local distance matrix" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tier-1 cluster() publishes its result with no Python wiring; the write-back
# of the algorithm free functions to the Problem is test_index_types.py's
# ---------------------------------------------------------------------------
class TestResultWriteBackInCpp:
    def test_cluster_tier1_end_to_end_results_visible(self):
        """drives dtwcpp.cluster() Tier-1 — labels/medoids/score visible with NO
        wrapper wiring (the preserved end-to-end contract; must not weaken)."""
        rng = np.random.default_rng(7)
        X = np.array([rng.standard_normal(12) * 0.1 + (0.0 if i < 6 else 9.0)
                      for i in range(12)])
        res = dtwcpp.cluster(X, k=2)
        assert res.device == "cpu"
        assert res.medoids is not None
        assert len(set(res.labels[:6])) == 1 and len(set(res.labels[6:])) == 1
        assert res.score("silhouette") > 0.5


# ---------------------------------------------------------------------------
# cluster() is dtwc::run, the run dtwc_cl makes: its report, the matrix at its
# stored width, an Arrow IPC file streamed under ram_limit
# ---------------------------------------------------------------------------
class TestClusterIsTheRun:
    def test_report_is_the_run_report_and_no_report_makes_none(self, tmp_path):
        """Result.report is the run report's text, which save writes last as
        <name>_report.txt; no_report=True, as dtwc_cl --no-report, makes none:
        report is "" and save writes the result files without it."""
        reported = dtwcpp.cluster(_two_groups(), k=2, method="pam", name="r")
        silent = dtwcpp.cluster(_two_groups(), k=2, method="pam", name="r", no_report=True)

        assert reported.report.startswith("DTWC++ run report")
        assert silent.report == ""
        reported.save(tmp_path / "on")
        silent.save(tmp_path / "off")
        assert (tmp_path / "on" / "r_report.txt").read_text(encoding="utf-8") == reported.report
        assert not (tmp_path / "off" / "r_report.txt").exists()
        assert (tmp_path / "off" / "r_labels.csv").read_bytes() == (tmp_path / "on" / "r_labels.csv").read_bytes()

    @pytest.mark.parametrize("method", ["pam", "clara"])
    def test_a_float32_matrix_comes_back_float32(self, method):
        """matrix_dtype="float32" stores each distance rounded to a float:
        distance_matrix is that array, float32, the float64 run's rounded,
        whether the method filled it (pam) or the read fills it (clara)."""
        wide = dtwcpp.cluster(_two_groups(), k=2, method=method).distance_matrix
        narrow = dtwcpp.cluster(_two_groups(), k=2, method=method, matrix_dtype="float32").distance_matrix

        assert wide.dtype == np.float64 and narrow.dtype == np.float32 and narrow.shape == (12, 12)
        assert np.array_equal(narrow, wide.astype(np.float32))

    def test_an_arrow_ipc_file_streams_under_ram_limit(self, tmp_path):
        """clara_256_lz4.arrow, four OpenMP threads as test_cli_ipc_stream runs
        it (the stream's price counts its decoding threads): at ram_limit
        300,000 the run streams it through FastCLARA, as -v says, and gives the
        in-RAM run's labels, medoids and cost. Its Problem holds no series, yet
        n_series is the file's 256; score and distance_matrix raise
        InvalidInput naming the stream; save writes the in-RAM run's labels
        and medoids files, byte for byte (the reader's names), then the
        report, and nothing else."""
        code = r"""
import json, os, sys
import dtwcpp
out = {}
for limit in (0, 300000):
    res = dtwcpp.cluster(sys.argv[1], 3, method="clara", sample_size=8, n_samples=2, seed=42, device="cpu",
                         name="t", ram_limit=limit, verbose=True)
    run = {"labels": res.labels.tolist(), "medoids": res.medoids.tolist(), "cost": res.cost,
           "n_series": res.n_series}
    for name, call in (("score", lambda: res.score("silhouette")), ("matrix", lambda: res.distance_matrix)):
        try:
            call()
            run[name] = "returned"
        except dtwcpp.InvalidInput as error:
            run[name] = str(error)
    res.save(os.path.join(sys.argv[2], str(limit)))
    out[limit] = run
print("JSON " + json.dumps(out))
"""
        run = subprocess.run([sys.executable, "-c", code, str(_READER / "clara_256_lz4.arrow"), str(tmp_path)],
                             capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "4"}, check=False)
        assert run.returncode == 0, run.stderr
        ram, streamed = json.loads(next(line[5:] for line in run.stdout.splitlines()
                                        if line.startswith("JSON "))).values()

        assert run.stdout.count("Arrow IPC walk selected streaming: 256 series") == 1
        for key in ("labels", "medoids", "cost"):
            assert streamed[key] == ram[key], key
        assert ram["n_series"] == streamed["n_series"] == 256
        assert ram["score"] == ram["matrix"] == "returned"
        for name, call in (("score", "score"), ("matrix", "distance_matrix")):
            assert streamed[name].startswith(f"Result::{call}: the run streamed its 256 series")
        in_ram, saved = tmp_path / "0", tmp_path / "300000"
        assert sorted(os.listdir(saved)) == ["t_labels.csv", "t_medoids.csv", "t_report.txt"]
        for table in ("labels", "medoids"):
            assert (saved / f"t_{table}.csv").read_bytes() == (in_ram / f"t_{table}.csv").read_bytes()
        assert (saved / "t_report.txt").stat().st_size > 0
