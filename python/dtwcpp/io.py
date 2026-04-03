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

    Expects an optional header row followed by numeric rows.

    Returns
    -------
    data : np.ndarray
        (N, L) float64 array.
    names : list[str]
        Column names from the header (empty strings if no header).
    """
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader)

        # Detect whether the first row is a header or data.
        try:
            first_values = [float(v) for v in first_row]
            is_header = False
        except ValueError:
            is_header = True

        rows: list[list[float]] = []
        if not is_header:
            rows.append(first_values)
            names_out: list[str] = []
        else:
            names_out = first_row

        for row in reader:
            if row:  # skip blank lines
                rows.append([float(v) for v in row])

    data = np.array(rows, dtype=np.float64)
    return data, names_out


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


def load_dataset_parquet(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a time-series dataset from Parquet.

    Returns
    -------
    data : np.ndarray
        (N, L) float64 array.
    names : list[str]
        Column names.

    Raises
    ------
    ImportError
        If *pyarrow* is not installed.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet I/O.  Install it with:  uv add pyarrow"
        ) from None

    path = Path(path)
    table = pq.read_table(str(path))
    names_out = table.column_names
    data = np.column_stack([table.column(c).to_numpy() for c in names_out])
    return data, names_out
