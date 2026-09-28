"""
@file convert.py
@brief CLI tool to convert time series data between formats.

Converts Parquet/CSV/HDF5 to Arrow IPC (.arrow).
Arrow IPC files can be memory-mapped for zero-copy access by the C++ CLI.

Usage::

    dtwc-convert input.parquet -o output.arrow
    dtwc-convert input.csv -o output.arrow
    dtwc-convert input.h5 -o output.arrow --name-column name

@author Volkan Kumtepeli
@author Claude 4.6
@date 08 Apr 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_parquet(path: Path, columns: list[str] | None, name_col: str | None):
    """Load from Parquet. Returns (list[np.ndarray], list[str]).

    Handles two Parquet layouts:
      - Columnar: each column is a timestep, each row is a series
      - List column: a single column of type List<float64>, each cell is a series
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet input. Install: uv add pyarrow"
        ) from None

    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    all_names: list[str] = []
    all_series: list[np.ndarray] = []

    # Determine feature columns once from schema
    feat_cols = columns
    if feat_cols is None:
        feat_cols = [f.name for f in schema if f.name != name_col]

    # Detect if any feature column is a list type (variable-length series layout)
    is_list_layout = False
    if feat_cols:
        first_type = schema.field(feat_cols[0]).type
        is_list_layout = pa.types.is_list(first_type) or pa.types.is_large_list(first_type)

    for batch in pf.iter_batches(batch_size=10_000):
        tbl = batch.to_pydict()
        n_rows = len(next(iter(tbl.values())))

        # Extract names
        if name_col and name_col in tbl:
            all_names.extend(str(v) for v in tbl[name_col])
        else:
            base = len(all_series)
            all_names.extend(f"series_{base + i}" for i in range(n_rows))

        if is_list_layout:
            # List column: each cell is already a full series
            col_name = feat_cols[0]
            for i in range(n_rows):
                val = tbl[col_name][i]
                if val is None:
                    raise ValueError(
                        f"NULL found in list column '{col_name}' at row {len(all_series)}. "
                        "Fill or drop NULLs before conversion.")
                series = np.array(val, dtype=np.float64)
                if np.any(np.isnan(series)):
                    raise ValueError(
                        f"NaN found in series at row {len(all_series)}. "
                        "Fill or drop NaN values before conversion.")
                assert series.ndim == 1, f"Expected 1D series, got shape {series.shape}"
                all_series.append(series)
        else:
            # Columnar layout: each column is a timestep, vectorize with numpy
            cols = [np.array(tbl[c], dtype=np.float64) for c in feat_cols]
            matrix = np.column_stack(cols)  # (n_rows, n_cols)
            # Check for NaN/NULL (None becomes NaN in numpy)
            if np.any(np.isnan(matrix)):
                for i in range(n_rows):
                    if np.any(np.isnan(matrix[i])):
                        raise ValueError(
                            f"NaN/NULL found in series at row {len(all_series) + i}. "
                            "Fill or drop NaN values before conversion.")
            for i in range(n_rows):
                all_series.append(matrix[i])

    return all_series, all_names


def _load_csv(path: Path):
    """Load from CSV. Returns (list[np.ndarray], list[str])."""
    from .io import load_dataset_csv

    data, names = load_dataset_csv(path)
    series_list = [data[i] for i in range(data.shape[0])]
    if not names:
        names = [f"series_{i}" for i in range(len(series_list))]
    return series_list, names


def _load_hdf5(path: Path):
    """Load from HDF5. Returns (list[np.ndarray], list[str])."""
    from .io import load_dataset_hdf5

    result = load_dataset_hdf5(path)
    data = result["series"]  # (N, L) array
    names = result["names"] or [f"series_{i}" for i in range(data.shape[0])]
    series_list = [data[i] for i in range(data.shape[0])]
    return series_list, names


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_arrow_ipc(
    series_list: list[np.ndarray],
    names: list[str],
    output: Path,
    ndim: int = 1,
) -> None:
    """Write Arrow IPC file (Feather v2) with zero-copy-friendly layout.

    Schema:
      - "name": utf8 (series name)
      - "data": large_list<float64> (int64 offsets, safe for >2B elements)

    Metadata:
      - "ndim": str(ndim)

    Written uncompressed for zero-copy mmap from C++.
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow IPC output. Install: uv add pyarrow"
        ) from None

    name_array = pa.array(names, type=pa.utf8())

    # LargeList<Float64> with int64 offsets — safe for >2B total elements
    flat_values = np.concatenate(series_list) if series_list else np.array([], dtype=np.float64)
    offsets = np.zeros(len(series_list) + 1, dtype=np.int64)
    for i, s in enumerate(series_list):
        offsets[i + 1] = offsets[i] + len(s)

    # Zero-copy: wrap numpy buffer directly instead of copying
    buf = pa.py_buffer(flat_values)
    values_array = pa.Array.from_buffers(pa.float64(), len(flat_values), [None, buf])
    offsets_array = pa.array(offsets, type=pa.int64())
    data_array = pa.LargeListArray.from_arrays(offsets_array, values_array)

    schema = pa.schema(
        [
            pa.field("name", pa.utf8()),
            pa.field("data", pa.large_list(pa.float64())),
        ],
        metadata={"ndim": str(ndim)},
    )

    # Write as single RecordBatch (ensures single chunk for C++ reader)
    batch = pa.record_batch({"name": name_array, "data": data_array}, schema=schema)
    with pa.OSFile(str(output), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_batch(batch)
        writer.close()

    print(f"Wrote {len(series_list)} series to {output} ({output.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dtwc-convert",
        description="Convert time series data between formats for DTWC++.",
    )
    parser.add_argument("input", type=Path, help="Input file (parquet, csv, h5)")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output file (.arrow, .ipc, or .feather)")
    parser.add_argument("--ndim", type=int, default=1,
                        help="Number of feature dimensions per timestep (default: 1)")
    parser.add_argument("--columns", nargs="+", default=None,
                        help="Feature columns to use (Parquet only)")
    parser.add_argument("--name-column", type=str, default=None,
                        help="Column containing series names (Parquet only)")

    args = parser.parse_args(argv)

    inp: Path = args.input
    out: Path = args.output

    if not inp.exists():
        print(f"Error: input file not found: {inp}", file=sys.stderr)
        return 1

    # Load based on input extension
    suffix = inp.suffix.lower()
    if suffix in (".parquet", ".pq"):
        series_list, names = _load_parquet(inp, args.columns, args.name_column)
    elif suffix in (".csv", ".tsv"):
        series_list, names = _load_csv(inp)
    elif suffix in (".h5", ".hdf5"):
        series_list, names = _load_hdf5(inp)
    else:
        print(f"Error: unsupported input format: {suffix}", file=sys.stderr)
        return 1

    print(f"Loaded {len(series_list)} series from {inp}")

    # Write based on output extension
    out_suffix = out.suffix.lower()
    if out_suffix in (".arrow", ".ipc", ".feather"):
        _write_arrow_ipc(series_list, names, out, ndim=args.ndim)
    else:
        print(f"Error: unsupported output format: {out_suffix}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
