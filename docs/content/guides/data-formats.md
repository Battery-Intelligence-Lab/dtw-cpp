---
title: "Data formats and conversion"
weight: 20
---

# Data formats and conversion

Rows represent time series unless a format stores an explicit list column.

| Format | Read path | Notes |
|---|---|---|
| CSV / TSV / text | Core, CLI, all bindings | One series per row; use `skip_cols` for identifiers. TSV is auto-detected from `.tsv`. |
| Arrow IPC / Feather | CLI with `-DDTWC_ENABLE_ARROW=ON`; Python via `pyarrow` | List/large-list Float32 or Float64 columns; memory-mapped native reader. |
| Parquet | CLI with Arrow/Parquet enabled; Python via `pyarrow`; MATLAB via `parquetread` | Each List/LargeList Float32/64 row = one series; several scalar Float32/64 columns = one series per row, as CSV; the only scalar Float32/64 column = one whole series. Rows are named by the first string column. `--column` reads one column ([Supported data](../../getting-started/supported-data/)). |
| HDF5 | Python conversion/I/O utilities | Optional `h5py`; convert before using the native CLI. |

Convert from CSV, Parquet, or HDF5 to Arrow IPC:

```sh
dtwc-convert input.csv -o output.arrow
dtwc-convert input.parquet -o output.arrow --name-column name
dtwc-convert input.h5 -o output.arrow --name-column name
```

Arrow/Parquet support is optional in native builds:

```sh
cmake -S . -B build -DDTWC_ENABLE_ARROW=ON
```

Series are always held in RAM. The distance matrix can live in a memory-mapped
`.dtwm` file instead (`use_mmap_distance_matrix`, `--mmap-threshold`); the Python
wheels and the release CLI archives include that support (llfio).

For data larger than an all-pairs matrix, choose a matrix-free method such as
OneBatchPAM, CLARA, or TADPole rather than changing only the input container.

The native CLI applies a nonzero `--ram-limit` to Parquet before payload
materialisation. If the conservative selected-column estimate exceeds the cap,
only non-full FastCLARA over a single file of one series per row (a list column,
or several Float32/Float64 columns) can stream row groups; a one-column file,
directories, and other methods fail before loading the payload.
Row groups are indivisible, and the cap covers series decoding/materialisation
rather than all process memory. The streaming route emits labels and medoids
without constructing dense distance or silhouette CSVs; storage requested with `--dtype f32` remains Float32 throughout the
sample, medoid, and assignment payloads.
