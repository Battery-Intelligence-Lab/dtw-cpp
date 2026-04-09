#!/usr/bin/env python3
"""Convert UCR TSV files to Parquet format for DTWC++ benchmarking.

Reads *_TRAIN.tsv and *_TEST.tsv from a UCR dataset directory, strips the
class-label column (first column), and writes each as a Parquet file with one
row per time series and one column per timestep.

Dependencies: pyarrow (optional extra: `uv add pyarrow` or install with
`pip install dtwcpp[parquet]`).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _check_pyarrow():
    """Check that pyarrow is available; exit with helpful message if not."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        print(
            "ERROR: pyarrow is required but not installed.\n"
            "  Install with:  uv add pyarrow\n"
            "  Or:            pip install pyarrow",
            file=sys.stderr,
        )
        sys.exit(1)


def read_ucr_tsv(tsv_path: Path) -> tuple[list[list[float]], int]:
    """Read a UCR TSV file, skip the class-label column.

    Returns (rows, n_timesteps) where each row is a list of floats.
    """
    rows: list[list[float]] = []
    n_timesteps = 0
    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            # Skip first column (class label), convert rest to float
            values = [float(v) for v in row[1:] if v.strip()]
            if values:
                rows.append(values)
                n_timesteps = max(n_timesteps, len(values))
    return rows, n_timesteps


def write_parquet(rows: list[list[float]], n_timesteps: int, out_path: Path) -> None:
    """Write time-series rows to a Parquet file.

    Schema: one column per timestep named "t0", "t1", ..., "t{N-1}".
    Each row is one time series.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build column arrays (pad shorter series with NaN)
    columns: dict[str, list[float]] = {}
    for j in range(n_timesteps):
        col_name = f"t{j}"
        col_data = []
        for row in rows:
            if j < len(row):
                col_data.append(row[j])
            else:
                col_data.append(float("nan"))
        columns[col_name] = col_data

    table = pa.table(columns)
    pq.write_table(table, out_path)


def convert_file(tsv_path: Path) -> Path:
    """Convert a single TSV file to Parquet. Returns output path."""
    rows, n_timesteps = read_ucr_tsv(tsv_path)
    if not rows:
        print(f"WARNING: no data rows in {tsv_path}", file=sys.stderr)
        return tsv_path.with_suffix(".parquet")

    out_path = tsv_path.with_suffix(".parquet")
    write_parquet(rows, n_timesteps, out_path)
    print(f"  {tsv_path.name} -> {out_path.name}  ({len(rows)} series, {n_timesteps} timesteps)")
    return out_path


def convert_dataset(dataset_dir: Path) -> None:
    """Convert all *_TRAIN.tsv and *_TEST.tsv files in a UCR dataset directory."""
    tsv_files = sorted(dataset_dir.glob("*_TRAIN.tsv")) + sorted(dataset_dir.glob("*_TEST.tsv"))

    if not tsv_files:
        # Also try without the TRAIN/TEST suffix (some UCR extractions differ)
        tsv_files = sorted(dataset_dir.glob("*.tsv"))

    if not tsv_files:
        print(f"ERROR: no TSV files found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(tsv_files)} file(s) in {dataset_dir}:")
    for tsv_path in tsv_files:
        convert_file(tsv_path)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UCR TSV files to Parquet format for DTWC++."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="UCR dataset directory (e.g. data/benchmark/UCRArchive_2018/Coffee) "
        "or a single TSV file.",
    )
    args = parser.parse_args()

    _check_pyarrow()

    path: Path = args.path
    if path.is_file() and path.suffix == ".tsv":
        print(f"Converting single file:")
        convert_file(path)
    elif path.is_dir():
        convert_dataset(path)
    else:
        print(f"ERROR: {path} is not an existing directory or TSV file.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
