"""Write the Parquet reader fixtures beside this script: one file per layout of the column rule.

The C++ reader (tests/unit/test_io_readers.cpp), Python (tests/python/test_io.py) and MATLAB
(tests/matlab/test_tier1_route_parity.m) read these files and expect the same series and names:

- parquet_rows.parquet: a string id, then three Float32/Float64 columns: each row is a series, named by
  the id (a, b, c);
- parquet_list.parquet: a list<double> column, then a string column with a null: a series per row,
  named x, series_1 (the null), z;
- parquet_one_column.parquet: a string column and one Float64 column: one series, named by the file.

Every value is exact in Float32. Run with pyarrow installed: python make_parquet_fixtures.py
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent

pq.write_table(pa.table({
    "id": ["a", "b", "c"],
    "t0": pa.array([0.0, 1.0, 10.0], pa.float64()),
    "t1": pa.array([0.5, 1.5, 10.5], pa.float32()),
    "t2": pa.array([0.25, 1.25, 10.25], pa.float64()),
}), HERE / "parquet_rows.parquet")
pq.write_table(pa.table({
    "series": pa.array([[0.0, 0.5], [2.5, 1.0, 0.25], [9.0, 9.5]], pa.list_(pa.float64())),
    "name": pa.array(["x", None, "z"], pa.string()),
}), HERE / "parquet_list.parquet")
pq.write_table(pa.table({
    "unit": ["s", "s", "s", "s"],
    "v": pa.array([3.0, 1.0, 4.0, 1.5], pa.float64()),
}), HERE / "parquet_one_column.parquet")
