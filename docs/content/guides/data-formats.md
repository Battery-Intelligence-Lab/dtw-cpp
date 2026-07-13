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
| Parquet | CLI with Arrow/Parquet enabled; Python via `pyarrow` | Scalar Float32/64 = one whole series; each List/LargeList Float32/64 row = one series. `--column` is optional auto-detection. |
| `.dtws` | Converter and native cache paths | Compact internal series store intended for repeated local work. |
| HDF5 | Python conversion/I/O utilities | Optional `h5py`; convert before using the native CLI. |

Convert from CSV, Parquet, or HDF5 to Arrow IPC or `.dtws`:

```sh
dtwc-convert input.csv -o output.arrow
dtwc-convert input.parquet -o output.dtws --name-column name
dtwc-convert input.h5 -o output.arrow --name-column name
```

Arrow/Parquet support is optional in native builds:

```sh
cmake -S . -B build -DDTWC_ENABLE_ARROW=ON
```

Published Python wheels intentionally omit LLFIO and native mmap support to stay
small and portable. Requesting an unavailable explicit mmap policy fails loudly;
an automatic spill that cannot use mmap warns before retaining data in RAM.

For data larger than an all-pairs matrix, choose a matrix-free method such as
OneBatchPAM, CLARA, or TADPole rather than changing only the input container.

The native CLI applies a nonzero `--ram-limit` to Parquet before payload
materialisation. If the conservative selected-column estimate exceeds the cap,
only non-full FastCLARA over a single list-per-row file can stream row groups;
scalar columns, directories, and other methods fail before loading the payload.
Row groups are indivisible, and the cap covers series decoding/materialisation
rather than all process memory. The streaming route emits labels, medoids, and
its binary result checkpoint without constructing dense distance or silhouette
CSVs; storage requested with `--dtype f32` remains Float32 throughout the
sample, medoid, and assignment payloads.
