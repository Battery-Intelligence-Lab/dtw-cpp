"""
@file test_arrow_slices.py
@brief An Arrow struct array is read from its own offset, length and validity: a slice gives its rows and no others,
       and a null row is refused as a null list is.
@author Volkan Kumtepeli
@date 21 Nov 2026
"""

import pytest

import dtwcpp

pa = pytest.importorskip("pyarrow")


def _struct(mask=None):
    names = pa.array(["a", "b", "c", "d", "e"], pa.string())
    data = pa.array([[0.0], [1.0, 1.0], [2.0, 2.0, 2.0], [3.0], [4.0, 4.0]], pa.list_(pa.float64()))
    return pa.StructArray.from_arrays([names, data], names=["name", "data"], mask=mask)


def test_a_sliced_struct_array_reads_its_own_rows():
    # pyarrow slices a struct by its offset and length, its children kept whole.
    dataset = dtwcpp.load(_struct().slice(2, 3))
    assert list(dataset.series_names()) == ["c", "d", "e"]
    assert [list(series) for series in dataset.as_series()] == [[2.0, 2.0, 2.0], [3.0], [4.0, 4.0]]


def test_a_null_struct_row_is_refused():
    # A row masked null keeps its children's values, which are no series.
    masked = _struct(mask=pa.array([False, True, False, False, False]))
    with pytest.raises(dtwcpp.InvalidInput, match="null series at row 1"):
        dtwcpp.load(masked).as_data()
