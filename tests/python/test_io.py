"""
@file test_io.py
@brief Tests for dtwcpp.io — CSV, HDF5, and Parquet I/O utilities.
@author Volkan Kumtepeli
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from dtwcpp.io import (
    load_dataset_csv,
    load_dataset_hdf5,
    load_dataset_parquet,
    save_dataset_csv,
    save_dataset_hdf5,
    save_dataset_parquet,
)

_has_h5py = importlib.util.find_spec("h5py") is not None
_has_pyarrow = importlib.util.find_spec("pyarrow") is not None

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> np.ndarray:
    """3 series of length 5."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 5))


@pytest.fixture
def sample_names() -> list[str]:
    return ["series_a", "series_b", "series_c"]


@pytest.fixture
def sample_distmat() -> np.ndarray:
    """Symmetric 3x3 distance matrix."""
    d = np.array([
        [0.0, 1.5, 2.3],
        [1.5, 0.0, 0.8],
        [2.3, 0.8, 0.0],
    ])
    return d


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCSV:
    def test_roundtrip_with_header(self, tmp_path, sample_data, sample_names):
        p = tmp_path / "data.csv"
        save_dataset_csv(sample_data, p, names=sample_names)
        loaded, names = load_dataset_csv(p)
        np.testing.assert_allclose(loaded, sample_data, atol=1e-10)
        assert names == sample_names

    def test_roundtrip_no_header(self, tmp_path, sample_data):
        p = tmp_path / "data.csv"
        save_dataset_csv(sample_data, p)
        loaded, names = load_dataset_csv(p)
        np.testing.assert_allclose(loaded, sample_data, atol=1e-10)
        assert names == []  # no header

    def test_single_series(self, tmp_path):
        data = np.array([[1.0, 2.0, 3.0]])
        p = tmp_path / "single.csv"
        save_dataset_csv(data, p)
        loaded, _ = load_dataset_csv(p)
        np.testing.assert_allclose(loaded, data, atol=1e-10)

    def test_pathlib_and_str(self, tmp_path, sample_data):
        """Both str and Path arguments should work."""
        p = tmp_path / "test.csv"
        save_dataset_csv(sample_data, str(p))
        loaded, _ = load_dataset_csv(str(p))
        np.testing.assert_allclose(loaded, sample_data, atol=1e-10)


# ---------------------------------------------------------------------------
# HDF5 (skip if h5py is not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_h5py, reason="h5py not installed")
class TestHDF5:
    def test_roundtrip_basic(self, tmp_path, sample_data, sample_names):
        p = tmp_path / "data.h5"
        save_dataset_hdf5(sample_data, p, names=sample_names)
        result = load_dataset_hdf5(p)
        np.testing.assert_allclose(result["series"], sample_data)
        assert result["names"] == sample_names
        assert result["distmat"] is None
        assert result["metadata"] == {}

    def test_roundtrip_with_distmat(self, tmp_path, sample_data, sample_distmat):
        p = tmp_path / "data.h5"
        save_dataset_hdf5(sample_data, p, distance_matrix=sample_distmat)
        result = load_dataset_hdf5(p)
        np.testing.assert_allclose(result["distmat"], sample_distmat)

    def test_roundtrip_with_metadata(self, tmp_path, sample_data):
        p = tmp_path / "data.h5"
        meta = {"band": 5, "variant": "dtw", "normalized": True}
        save_dataset_hdf5(sample_data, p, metadata=meta)
        result = load_dataset_hdf5(p)
        assert result["metadata"]["band"] == 5
        assert result["metadata"]["variant"] == "dtw"
        assert result["metadata"]["normalized"] == True  # noqa: E712 — h5py returns np.bool_

    def test_no_names(self, tmp_path, sample_data):
        p = tmp_path / "data.h5"
        save_dataset_hdf5(sample_data, p)
        result = load_dataset_hdf5(p)
        assert result["names"] is None

    def test_full_roundtrip(self, tmp_path, sample_data, sample_names, sample_distmat):
        """All fields populated."""
        p = tmp_path / "full.h5"
        meta = {"k": 3, "method": "fastpam"}
        save_dataset_hdf5(
            sample_data, p,
            names=sample_names,
            distance_matrix=sample_distmat,
            metadata=meta,
        )
        result = load_dataset_hdf5(p)
        np.testing.assert_allclose(result["series"], sample_data)
        assert result["names"] == sample_names
        np.testing.assert_allclose(result["distmat"], sample_distmat)
        assert result["metadata"]["k"] == 3
        assert result["metadata"]["method"] == "fastpam"

    def test_compression_reduces_size(self, tmp_path):
        """Verify gzip compression actually compresses repetitive data."""
        data = np.ones((100, 1000))
        p = tmp_path / "big.h5"
        save_dataset_hdf5(data, p)
        file_size = p.stat().st_size
        raw_size = data.nbytes  # 800_000
        assert file_size < raw_size / 2  # gzip on all-ones should compress well


# ---------------------------------------------------------------------------
# Parquet (skip if pyarrow is not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not installed")
class TestParquet:
    def test_roundtrip_with_names(self, tmp_path, sample_data):
        col_names = [f"t{i}" for i in range(sample_data.shape[1])]
        p = tmp_path / "data.parquet"
        save_dataset_parquet(sample_data, p, names=col_names)
        loaded, names = load_dataset_parquet(p)
        np.testing.assert_allclose(loaded, sample_data, atol=1e-10)
        assert names == col_names

    def test_roundtrip_auto_names(self, tmp_path, sample_data):
        p = tmp_path / "data.parquet"
        save_dataset_parquet(sample_data, p)
        loaded, names = load_dataset_parquet(p)
        np.testing.assert_allclose(loaded, sample_data, atol=1e-10)
        assert names == [f"t{i}" for i in range(sample_data.shape[1])]

    def test_single_series(self, tmp_path):
        data = np.array([[1.0, 2.0, 3.0]])
        p = tmp_path / "single.parquet"
        save_dataset_parquet(data, p)
        loaded, _ = load_dataset_parquet(p)
        np.testing.assert_allclose(loaded, data, atol=1e-10)

    def test_wide_data(self, tmp_path):
        """Many columns (time steps)."""
        data = np.random.default_rng(99).standard_normal((5, 500))
        p = tmp_path / "wide.parquet"
        save_dataset_parquet(data, p)
        loaded, names = load_dataset_parquet(p)
        np.testing.assert_allclose(loaded, data, atol=1e-10)
        assert len(names) == 500


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not installed")
class TestParquetListColumn:
    """Polars writes time-series as a single ``large_list<float>`` column with
    metadata columns alongside. ``load_dataset_parquet`` must detect and load
    that layout instead of crashing on ``np.column_stack`` of ragged rows."""

    @staticmethod
    def _write_list_column_parquet(path, series, ride_numbers=None,
                                    list_type="large_list"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if list_type == "large_list":
            arr = pa.array(series, type=pa.large_list(pa.float64()))
        else:
            arr = pa.array(series, type=pa.list_(pa.float64()))
        cols = {"sequence": arr}
        if ride_numbers is not None:
            cols["ride_number"] = pa.array(ride_numbers, type=pa.uint32())
        pq.write_table(pa.table(cols), str(path), compression="snappy")

    def test_large_list_autodetect(self, tmp_path):
        p = tmp_path / "rides.parquet"
        rides = [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
        self._write_list_column_parquet(p, rides)
        data, names = load_dataset_parquet(p)
        assert isinstance(data, list)
        assert len(data) == 3
        np.testing.assert_array_equal(data[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(data[2], [6.0, 7.0, 8.0, 9.0])
        assert names == ["series_0", "series_1", "series_2"]

    def test_explicit_column_and_name_column(self, tmp_path):
        p = tmp_path / "rides.parquet"
        rides = [[10.0, 20.0], [30.0, 40.0, 50.0]]
        self._write_list_column_parquet(p, rides, ride_numbers=[7, 42])
        data, names = load_dataset_parquet(
            p, column="sequence", name_column="ride_number")
        assert names == ["7", "42"]
        np.testing.assert_array_equal(data[1], [30.0, 40.0, 50.0])

    def test_plain_list_type_supported(self, tmp_path):
        p = tmp_path / "rides.parquet"
        rides = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        self._write_list_column_parquet(p, rides, list_type="list")
        data, names = load_dataset_parquet(p)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "rides.parquet"
        self._write_list_column_parquet(p, [[1.0, 2.0]])
        with pytest.raises(ValueError, match="not found"):
            load_dataset_parquet(p, column="nope")

    def test_wrong_column_type_raises(self, tmp_path):
        p = tmp_path / "rides.parquet"
        self._write_list_column_parquet(p, [[1.0]], ride_numbers=[1])
        with pytest.raises(ValueError, match="list / large_list"):
            load_dataset_parquet(p, column="ride_number")

    def test_variable_length_preserved(self, tmp_path):
        p = tmp_path / "rides.parquet"
        rides = [list(range(n)) for n in (3, 7, 5, 11)]
        self._write_list_column_parquet(p, rides)
        data, _ = load_dataset_parquet(p)
        assert [len(s) for s in data] == [3, 7, 5, 11]
