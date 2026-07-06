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
