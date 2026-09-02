"""
@file io.py
@brief I/O utilities for saving/loading time series data and distance matrices.
@details
Supported formats:
- CSV: always available (numpy only)
- HDF5: requires h5py (optional)
- Parquet: requires pyarrow (optional)

HDF5 layout::

    /series     -- (N, L) float64 dataset, gzip-compressed
    /names      -- (N,) variable-length string dataset
    /distmat    -- (N, N) float64 dataset, gzip-compressed (optional)
    /metadata   -- HDF5 root attributes (band, variant, etc.)

@author Volkan Kumtepeli
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_dataset_csv(
    data: np.ndarray,
    path: str | Path,
    names: list[str] | None = None,
) -> None:
    """Save a time-series dataset to CSV.

    Each row is one series, each column is one time step.

    Parameters
    ----------
    data : np.ndarray
        (N, L) array of N time series, each of length L.
    path : str or Path
        Destination file path.
    names : list[str], optional
        Series names written as the header row.
    """
    path = Path(path)
    header = ",".join(names) if names else ""
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def load_dataset_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a time-series dataset from CSV.

    Expects an optional header row followed by numeric rows. Only the header
    detection is done here; the numeric rows are parsed by the C++
    ``DataLoader`` — the one reader the CLI, C++ and :func:`dtwcpp.load` share.

    Returns
    -------
    data : np.ndarray
        (N, L) float64 array.
    names : list[str]
        Column names from the header (empty list if no header).
    """
    path = Path(path)
    with open(path, newline="") as f:
        first_row = next(csv.reader(f), [])

    try:
        [float(value) for value in first_row]
    except ValueError:
        names_out, skip_rows = first_row, 1
    else:
        names_out, skip_rows = [], 0

    from dtwcpp import _dtwcpp_core
    rows = _dtwcpp_core._read_data(str(path), 0, skip_rows, ",").p_vec
    return np.array([row for row in rows if row], dtype=np.float64), names_out


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

def save_dataset_hdf5(
    data: np.ndarray,
    path: str | Path,
    names: list[str] | None = None,
    distance_matrix: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a time-series dataset (and optional distance matrix) to HDF5.

    Parameters
    ----------
    data : np.ndarray
        (N, L) array of N series.
    path : str or Path
        Destination ``.h5`` file.
    names : list[str], optional
        Series names stored in ``/names``.
    distance_matrix : np.ndarray, optional
        (N, N) pairwise distance matrix stored in ``/distmat``.
    metadata : dict, optional
        Scalar key/value pairs stored as HDF5 root attributes.

    Raises
    ------
    ImportError
        If *h5py* is not installed.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 I/O.  Install it with:  uv add h5py"
        ) from None

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("series", data=np.asarray(data, dtype=np.float64),
                         compression="gzip", compression_opts=4)
        if names is not None:
            dt = h5py.string_dtype()
            f.create_dataset("names", data=names, dtype=dt)
        if distance_matrix is not None:
            f.create_dataset(
                "distmat",
                data=np.asarray(distance_matrix, dtype=np.float64),
                compression="gzip",
                compression_opts=4,
            )
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


def load_dataset_hdf5(path: str | Path) -> dict[str, Any]:
    """Load a time-series dataset from HDF5.

    Returns
    -------
    dict
        Keys: ``series`` (ndarray), ``names`` (list[str] or None),
        ``distmat`` (ndarray or None), ``metadata`` (dict).

    Raises
    ------
    ImportError
        If *h5py* is not installed.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 I/O.  Install it with:  uv add h5py"
        ) from None

    path = Path(path)
    result: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        result["series"] = f["series"][:]
        if "names" in f:
            raw = f["names"][:]
            result["names"] = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in raw
            ]
        else:
            result["names"] = None
        if "distmat" in f:
            result["distmat"] = f["distmat"][:]
        else:
            result["distmat"] = None
        result["metadata"] = dict(f.attrs)
    return result


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def save_dataset_parquet(
    data: np.ndarray,
    path: str | Path,
    names: list[str] | None = None,
) -> None:
    """Save a time-series dataset to Parquet (Snappy-compressed).

    Each column corresponds to one time step.

    Parameters
    ----------
    data : np.ndarray
        (N, L) array.
    path : str or Path
        Destination ``.parquet`` file.
    names : list[str], optional
        Column names (default: ``t0, t1, ...``).

    Raises
    ------
    ImportError
        If *pyarrow* is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet I/O.  Install it with:  uv add pyarrow"
        ) from None

    path = Path(path)
    columns = names if names and len(names) == data.shape[1] else [
        f"t{i}" for i in range(data.shape[1])
    ]
    table = pa.table({col: data[:, i] for i, col in enumerate(columns)})
    pq.write_table(table, str(path), compression="snappy")


def load_dataset_parquet(
    path: str | Path,
    column: str | None = None,
    name_column: str | None = None,
) -> tuple[np.ndarray | list[np.ndarray], list[str]]:
    """Load a time-series dataset from Parquet.

    Two layouts are supported:

    1. **Columnar (rectangular)** — each column is one time step, each row one
       series. Returns ``(ndarray of shape (N, L), list[str] column names)``.
       This is the format written by :func:`save_dataset_parquet`.
    2. **List-column (ragged)** — one column of type ``list<float>`` or
       ``large_list<float>`` holds one variable-length series per row. This is
       what Polars writes when you store a Series-of-arrays. Returns
       ``(list of ndarrays, list[str] series names)``.

    Parameters
    ----------
    path :
        Path to the ``.parquet`` file.
    column :
        Name of the list column to extract for layout 2. If ``None``,
        auto-detected as the first list / large-list column in the schema.
        Ignored for layout 1.
    name_column :
        Name of a column to use for series names in layout 2 (e.g. ``"id"`` or
        ``"ride_number"``). If ``None``, names are generated as
        ``series_0, series_1, ...``. Ignored for layout 1.

    Returns
    -------
    data : ndarray or list of ndarray
        Layout 1: ``(N, L)`` float64 array. Layout 2: list of N float64 arrays
        with potentially different lengths.
    names : list[str]
        Column names (layout 1) or series names (layout 2).

    Raises
    ------
    ImportError
        If *pyarrow* is not installed.
    ValueError
        If ``column`` is given but does not exist or is not a list type.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet I/O.  Install it with:  uv add pyarrow"
        ) from None

    path = Path(path)
    table = pq.read_table(str(path))
    schema = table.schema

    list_col_idx: int | None = None
    if column is not None:
        idx = schema.get_field_index(column)
        if idx < 0:
            raise ValueError(
                f"Column '{column}' not found in Parquet schema: {schema.names}"
            )
        ftype = schema.field(idx).type
        if not (pa.types.is_list(ftype) or pa.types.is_large_list(ftype)):
            raise ValueError(
                f"Column '{column}' has type {ftype}, expected list / large_list. "
                f"Drop the `column` argument to use the rectangular layout."
            )
        list_col_idx = idx
    else:
        for i in range(len(schema)):
            ftype = schema.field(i).type
            if pa.types.is_list(ftype) or pa.types.is_large_list(ftype):
                list_col_idx = i
                break

    if list_col_idx is not None:
        col = table.column(list_col_idx).to_pylist()
        series = [np.asarray(s, dtype=np.float64) for s in col]
        if name_column is not None:
            if schema.get_field_index(name_column) < 0:
                raise ValueError(
                    f"name_column '{name_column}' not found in Parquet schema: "
                    f"{schema.names}"
                )
            names_out = [str(x) for x in table.column(name_column).to_pylist()]
        else:
            names_out = [f"series_{i}" for i in range(len(series))]
        return series, names_out

    names_out = table.column_names
    data = np.column_stack([table.column(c).to_numpy() for c in names_out])
    return data, names_out
