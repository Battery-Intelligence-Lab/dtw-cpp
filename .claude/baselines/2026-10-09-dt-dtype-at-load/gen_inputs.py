"""Inputs for unit DT's float32 comparisons: a batch CSV, Parquet (list and rows layouts) and
Arrow IPC files of the same random series, written with 17 significant digits so that the
float32 narrowing matters; and, with --big, a Parquet file of >= 200 MB of float64 values
(pyarrow's default writer: snappy, one row group below 1M rows).

usage: gen_inputs.py <out_dir> [--big]

@author Volkan Kumtepeli
@author Claude Opus 5.5
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261009)

if "--big" in sys.argv:
    n, length = 30_000, 1_000  # 30 M float64 values: 240 MB
    values = rng.standard_normal(n * length)
    offsets = np.arange(0, n * length + 1, length, dtype=np.int32)
    series = pa.ListArray.from_arrays(pa.array(offsets), pa.array(values, pa.float64()))
    table = pa.table({"series": series})
    pq.write_table(table, os.path.join(out, "big_list_f64.parquet"))
    meta = pq.ParquetFile(os.path.join(out, "big_list_f64.parquet")).metadata
    print("big:", meta.num_rows, "rows,", meta.num_row_groups, "row groups,",
          os.path.getsize(os.path.join(out, "big_list_f64.parquet")), "bytes,",
          values.nbytes, "bytes of float64 values")
    sys.exit(0)

# 60 series of 150..300 values, in three shifted groups; values with 17 significant digits.
groups = 3
series = []
for i in range(60):
    length = int(rng.integers(150, 301))
    t = np.linspace(0, 4 * np.pi, length)
    series.append(np.sin(t + (i % groups)) * (1 + (i % groups)) + 0.3 * rng.standard_normal(length))

with open(os.path.join(out, "batch.csv"), "w", newline="\n") as f:
    for s in series:
        f.write(",".join(repr(float(v)) for v in s) + "\n")

names = pa.array([f"s{i:02d}" for i in range(len(series))], pa.string())
lists = pa.array([s.tolist() for s in series], pa.list_(pa.float64()))
pq.write_table(pa.table({"name": names, "series": lists}), os.path.join(out, "list.parquet"))

# The rows layout: each row a series of its values in 40 float64 columns.
equal = np.stack([s[:40] for s in series])
pq.write_table(pa.table({"name": names, **{f"v{j:02d}": equal[:, j] for j in range(40)}}),
               os.path.join(out, "rows.parquet"))

ipc_table = pa.table({"name": names, "data": pa.array([s.tolist() for s in series], pa.large_list(pa.float64()))})
with ipc.new_file(os.path.join(out, "series.arrow"), ipc_table.schema) as writer:
    writer.write_table(ipc_table)
print("small inputs written to", out)
