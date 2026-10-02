"""
@file io.py
@brief Reading series, writing results, and saving/loading datasets.
@details
dtwcpp reads and writes its files in Python; the compiled core holds no file
reader or writer. The series readers follow dtwc_cl's (dtwc/fileOperations.hpp,
dtwc/io/read_data.cpp), so a file reads to the same series in every language or
is refused with the same error type, and the result writers write the CLI's
bytes.

Supported formats:
- CSV/TSV text and a folder of them: always available (standard library)
- Parquet and Arrow IPC: require pyarrow (the ``parquet`` extra)
- HDF5: requires h5py (optional)

HDF5 layout::

    /series     -- (N, L) float64 dataset, gzip-compressed
    /names      -- (N,) variable-length string dataset
    /distmat    -- (N, N) float64 dataset, gzip-compressed (optional)
    /metadata   -- HDF5 root attributes (band, variant, etc.)

@author Volkan Kumtepeli
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Reading series (dtwcpp.load): dtwc_cl's readers, rule for rule
# ---------------------------------------------------------------------------

_SPACE = b" \t\n\v\f\r"  # ASCII whitespace, whatever the locale
# std::from_chars(general) after the reader takes one leading '+': digits with an
# optional point, or a point and digits, then an optional exponent. No hex, inf,
# underscores or non-ASCII digits.
_NUMBER = re.compile(rb"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_NONZERO_MANTISSA = re.compile(rb"[^eE]*[1-9]")
_PARQUET = (".parquet", ".pq")
_ARROW_IPC = (".arrow", ".ipc", ".feather")


def _extension(path):
    return os.path.splitext(os.fspath(path))[1].lower()


def _number(token, path, row, column):
    """One numeric field, as parse_numeric_field reads it: ASCII blanks trimmed,
    ``nan`` (any case) is a missing value, and a value beyond double or a nonzero
    one below its smallest is out of range."""
    from dtwcpp import IOError as DtwcIOError
    text = token.strip(_SPACE)
    if not text:
        reason = "empty numeric field"
    elif text.lower() == b"nan":
        return math.nan
    elif not _NUMBER.fullmatch(text):
        reason = "invalid numeric field"
    else:
        value = float(text)
        if not math.isinf(value) and (value != 0.0 or not _NONZERO_MANTISSA.match(text)):
            return value
        reason = "numeric field is out of range"
    shown = text[:64].decode("utf-8", "replace") + ("..." if len(text) > 64 else "")
    raise DtwcIOError(f"Error in delimited text file: '{path}' row {row}, column {column}: "
                      f"{reason} '{shown}'.")


def _data_lines(path, skip_rows):
    """(row, line) of each data line, row 1-based in the file: ``skip_rows``
    lines skipped, a UTF-8 byte-order mark dropped, blank lines after the last
    data line ignored and a blank line before one refused."""
    from dtwcpp import IOError as DtwcIOError
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as error:
        raise DtwcIOError(f"Error in delimited text file: '{path}' could not be opened: "
                          f"{error.strerror}.") from error
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    blank = 0
    for row, line in enumerate(data.split(b"\n"), start=1):
        if row <= skip_rows:
            continue
        if not line.strip(_SPACE):
            blank = blank or row
            continue
        if blank:
            raise DtwcIOError(
                f"Error in delimited text file: '{path}' row {blank} is empty; an empty "
                "line is neither a series nor a value (write a missing value as nan).")
        yield row, line


def _fields(line, delimiter):
    # A space delimiter splits at runs of blanks; any other at each occurrence.
    return line.split() if delimiter == b" " else line.split(delimiter)


def _read_series_file(path, skip_rows, skip_cols, delimiter):
    """One series per row, named by its 1-based row count (load_batch_file)."""
    from dtwcpp import InvalidInput
    series = []
    for row, line in _data_lines(path, skip_rows):
        fields = _fields(line, delimiter)
        if skip_cols > len(fields):
            raise InvalidInput(f"Error in delimited text file: '{path}' row {row} has only "
                               f"{len(fields)} fields, fewer than start_col={skip_cols}.")
        series.append([_number(fields[i], path, row, i + 1)
                       for i in range(skip_cols, len(fields))])
    return series, [str(i + 1) for i in range(len(series))]


def _folder_files(folder):
    """A folder's regular files without dot-files, sorted by name."""
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if not name.startswith(".") and os.path.isfile(os.path.join(folder, name))]


def _read_series_folder(folder, skip_rows, skip_cols, delimiter):
    """One series per file, one value per line in column ``skip_cols``, named by
    the file's stem (load_folder, readFile). A first line whose value is empty is
    pandas' ``,0`` header, unless rows are skipped."""
    from dtwcpp import InvalidInput, IOError as DtwcIOError
    series, names = [], []
    for path in _folder_files(folder):
        values = []
        header = skip_rows == 0
        for row, line in _data_lines(path, skip_rows):
            fields = _fields(line, delimiter)
            if skip_cols >= len(fields):
                raise InvalidInput(f"Error in delimited text file: '{path}' row {row} has only "
                                   f"{len(fields)} fields, fewer than required column {skip_cols + 1}.")
            first, header = header, False
            if first and not fields[skip_cols].strip(_SPACE):
                continue
            if len(fields) > skip_cols + 1:
                raise DtwcIOError(
                    f"Error in delimited text file: '{path}' row {row} has {len(fields)} fields; "
                    "a file in a one-series-per-file folder holds one value per line, here in "
                    f"column {skip_cols + 1} (skip leading columns with --skip-cols / start_col).")
            values.append(_number(fields[skip_cols], path, row, skip_cols + 1))
        series.append(values)
        names.append(os.path.splitext(os.path.basename(path))[0])
    return series, names


def _parquet_files(path):
    """A Parquet input's files: the file itself, or a folder's .parquet/.pq files."""
    if os.path.isdir(path):
        return [f for f in _folder_files(path) if _extension(f) in _PARQUET]
    return [path] if _extension(path) in _PARQUET else []


def _read_arrow(path, parquet):
    """Parquet (each row of the first list column a series, named by the first
    string column) or an Arrow IPC file (the ``data`` column, named by ``name``,
    ``ndim`` from the schema metadata) through the installed pyarrow, into the
    Arrow C stream the compiled-in nanoarrow reads."""
    from dtwcpp import IOError as DtwcIOError, _dtwcpp_core
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Reading Parquet or Arrow IPC needs pyarrow: install dtwcpp[parquet].") from None
    ndim = 1
    try:
        if parquet:
            table = pa.concat_tables([pq.read_table(os.fspath(f)) for f in parquet])
        else:
            with pa.OSFile(os.fspath(path), "rb") as f:
                table = pa.ipc.open_file(f).read_all()
            schema = table.schema
            if "data" not in schema.names:
                raise DtwcIOError("no 'data' column; write the series as a List or LargeList "
                                  "of Float32/Float64 named 'data'.")
            data_type = schema.field("data").type
            if not ((pa.types.is_list(data_type) or pa.types.is_large_list(data_type))
                    and (pa.types.is_float32(data_type.value_type)
                         or pa.types.is_float64(data_type.value_type))):
                raise DtwcIOError("the 'data' column must be a List or LargeList of "
                                  f"Float32/Float64, got {data_type}.")
            columns = ["data"]
            if "name" in schema.names:
                name_type = schema.field("name").type
                if not (pa.types.is_string(name_type) or pa.types.is_large_string(name_type)):
                    raise DtwcIOError(f"the 'name' column must be Utf8 or LargeUtf8, got {name_type}. "
                                      "Write the names as strings, or drop the column.")
                columns.append("name")
            text = (schema.metadata or {}).get(b"ndim")
            if text is not None:
                if not re.fullmatch(rb"[0-9]+", text) or int(text) == 0:
                    raise DtwcIOError(f"schema metadata 'ndim' must be a positive integer, got "
                                      f"'{text.decode('utf-8', 'replace')}'. Set it to the number of "
                                      "features per timestep (1 for univariate data).")
                ndim = int(text)
            table = table.select(columns)
    except (OSError, pa.ArrowException) as error:
        raise DtwcIOError(f"load: failed to read '{os.fspath(path)}': {error}") from error
    data = _dtwcpp_core.data_from_arrow_c_array(table)
    if ndim != 1:
        data.ndim = ndim
        data.validate_ndim()
    return data


def _read_data(source, skip_cols=0, skip_rows=0, delimiter=None):
    """Every series ``source`` names, read as dtwc::read_data reads it.

    A .parquet/.pq file, or a folder holding one, is Parquet and an
    .arrow/.ipc/.feather file Arrow IPC, both read through pyarrow; anything
    else is CSV/TSV text, one file (a series per row) or a folder (a series per
    file). ``skip_cols`` drops leading fields and ``skip_rows`` leading lines of
    text; ``delimiter`` None infers it from the extension (tab for .tsv/.txt,
    else comma). A read failure is :class:`dtwcpp.IOError` naming the file, an
    option the input cannot honour :class:`dtwcpp.InvalidInput`.
    """
    from dtwcpp import Data, InvalidInput, IOError as DtwcIOError
    path = os.fspath(source)
    parquet = _parquet_files(path)
    if parquet or (_extension(path) in _ARROW_IPC and not os.path.isdir(path)):
        if skip_cols or skip_rows or delimiter:
            raise InvalidInput(
                "load: skip_cols, skip_rows and delimiter parse CSV/TSV text and "
                "cannot be honoured for a Parquet or Arrow IPC input; drop them.")
        return _read_arrow(path, parquet)
    if delimiter:
        delim = str(delimiter).encode("utf-8")
        if len(delim) != 1:
            raise InvalidInput("load: delimiter must be a single character.")
    else:
        delim = b"\t" if _extension(path) in (".tsv", ".txt") else b","
    read = _read_series_folder if os.path.isdir(path) else _read_series_file
    try:
        series, names = read(path, skip_rows, skip_cols, delim)
    except DtwcIOError as error:
        raise DtwcIOError(f"load: failed to read '{path}': {error}") from error
    return Data(series, names)


# ---------------------------------------------------------------------------
# Writing results: the CLI's bytes (dtwc/cli/run.cpp, dtwc/Problem_IO.cpp)
# ---------------------------------------------------------------------------

def _write_bytes(path, data, mode):
    from dtwcpp import IOError as DtwcIOError
    try:
        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
        if mode == "wb":
            with open(path, "wb") as f:
                f.write(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
    except OSError as error:
        raise DtwcIOError(f"Cannot write '{os.fspath(path)}': {error.strerror}.") from error


def _write_text(path, text):
    """A text file as C++ writes one: UTF-8, the platform's line ending, its
    folder created."""
    _write_bytes(path, text, "w")


def _write_matrix_csv(matrix, path):
    """The distance-matrix CSV as io::write_csv writes it: binary, one LF per
    row, 17 significant digits, an empty field for a pair not computed (NaN).
    An infinite distance is refused before the file is opened."""
    from dtwcpp import InvalidInput
    matrix = np.asarray(matrix, dtype=float)
    infinite = np.argwhere(np.isinf(matrix))
    if len(infinite):
        i, j = (int(x) for x in infinite[0])
        raise InvalidInput(f"distance-matrix CSV: computed non-finite value at row {i}, column {j}.")
    text = "".join(",".join("" if value != value else f"{value:.17g}" for value in row) + "\n"
                   for row in matrix)
    _write_bytes(path, text.encode("ascii"), "wb")


# v1.0.0's Problem writers, attached to dtwcpp.Problem: the files and bytes of
# C++ Problem::write_* (Problem_IO.cpp), in Problem.output_folder.

def _names(problem):
    return [problem.series_name(i) for i in range(problem.size)]


def write_clusters(problem):
    """``<name>_Nc_<k>.csv``: the medoids, each series' medoid, the cost."""
    problem.require_clustered("write_clusters")
    names, labels, medoids = _names(problem), problem.labels(), problem.medoids()
    lines = ["Cluster centroids:", ",".join(names[m] for m in medoids), "", "Data,its cluster"]
    lines += [f"{names[i]},{names[medoids[label]]}" for i, label in enumerate(labels)]
    lines.append(f"Procedure is completed with cost: {problem.find_total_cost():g}")
    _write_text(os.path.join(problem.output_folder,
                             f"{problem.name}_Nc_{problem.n_clusters()}.csv"),
                "\n".join(lines) + "\n")


def write_silhouettes(problem):
    """``<name>_silhouettes_Nc_<k>.csv``: each series' silhouette. An undefined
    silhouette (one realised cluster) is a warning on stderr and no file."""
    import dtwcpp
    try:
        silhouettes = dtwcpp.silhouette(problem)
    except dtwcpp.UndefinedScore as e:
        print(f"Warning: silhouettes skipped: {e}", file=sys.stderr)
        return
    _write_text(os.path.join(problem.output_folder,
                             f"{problem.name}_silhouettes_Nc_{problem.n_clusters()}.csv"),
                "Silhouettes:\n" + "".join(f"{name},{s:g}\n"
                                           for name, s in zip(_names(problem), silhouettes)))


def write_medoid_members(problem, iter, rep=0):
    """``medoidMembers_Nc_<k>_rep_<rep>_iter_<iter>.csv``: each cluster's series."""
    problem.require_clustered("write_medoid_members")
    names, labels = _names(problem), problem.labels()
    _write_text(os.path.join(problem.output_folder,
                             f"medoidMembers_Nc_{problem.n_clusters()}_rep_{rep}_iter_{iter}.csv"),
                "".join("".join(f"{names[i]}," for i in np.flatnonzero(labels == c)) + "\n"
                        for c in range(problem.n_clusters())))


def write_distance_matrix(problem):
    """``<name>_distanceMatrix.csv``: the matrix as it is, not filled first."""
    _write_matrix_csv(problem.distance_matrix(fill=False),
                      os.path.join(problem.output_folder, f"{problem.name}_distanceMatrix.csv"))


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

    Expects an optional header row followed by numeric rows, read with the
    rules of :func:`dtwcpp.load` (dtwc_cl's reader).

    Returns
    -------
    data : np.ndarray
        (N, L) float64 array.
    names : list[str]
        Column names from the header (empty list if no header).
    """
    path = Path(path)
    # utf-8-sig drops a byte-order mark (Excel's "CSV UTF-8"), which otherwise
    # made the first data row look like a header and dropped the first series.
    with open(path, newline="", encoding="utf-8-sig") as f:
        first_row = next(csv.reader(f), [])

    try:
        [float(value) for value in first_row]
    except ValueError:
        names_out, skip_rows = first_row, 1
    else:
        names_out, skip_rows = [], 0

    rows, _ = _read_series_file(path, skip_rows, 0, b",")
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
