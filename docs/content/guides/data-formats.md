---
title: "Data formats and conversion"
weight: 20
---

# Data formats and conversion

Rows represent time series unless a format stores an explicit list column.

| Format | Read path | Notes |
|---|---|---|
| CSV / TSV / text | Core, CLI, all bindings | One series per row; use `skip_cols` for identifiers. TSV is auto-detected from `.tsv`. |
| Arrow IPC / Feather | Core, CLI, all bindings | List/large-list Float32 or Float64 columns, read in place (memory-mapped) by the native reader; compressed (LZ4, ZSTD) and big-endian files decoded. |
| Parquet | CLI with Arrow/Parquet enabled; Python via `pyarrow`; MATLAB via `parquetread` | Each List/LargeList Float32/64 row = one series; several scalar Float32/64 columns = one series per row, as CSV; the only scalar Float32/64 column = one whole series. Rows are named by the first string column. `--column` reads one column ([Supported data](../../getting-started/supported-data/)). |
| HDF5 | Python conversion/I/O utilities | Optional `h5py`; convert before using the native CLI. |

Convert from CSV, Parquet, or HDF5 to Arrow IPC:

```sh
dtwc-convert input.csv -o output.arrow
dtwc-convert input.parquet -o output.arrow --name-column name
dtwc-convert input.h5 -o output.arrow --name-column name
```

`dtwc-convert` writes an uncompressed file, which the reader maps and never decodes. A compressed Arrow IPC
file (LZ4 or ZSTD) is decoded a whole record batch to a thread, so one written as a single record batch
decodes on one core, and the writer's batch count sets how many cores decode it. A streamed run reads each
record batch whole as well, so one batch must fit its data cap. Write a compressed file in several record
batches, at least as many as the threads that should decode it. With pyarrow, from a table with the columns
[Supported data](../../getting-started/supported-data/) describes:

```python
import pyarrow as pa
import pyarrow.ipc as ipc

options = ipc.IpcWriteOptions(compression="lz4")
with pa.OSFile("data.arrow", "wb") as sink:
    with ipc.new_file(sink, table.schema, options=options) as writer:
        for batch in table.to_batches(max_chunksize=rows_per_batch):
            writer.write_batch(batch)
```

Arrow/Parquet support is optional in native builds:

```sh
cmake -S . -B build -DDTWC_ENABLE_ARROW=ON
```

Where the series live depends on the input:

- Text files, folders and series passed in memory are read whole into RAM.
- An Arrow IPC file (every build and language) is viewed where it lies when the reader can: uncompressed,
  aligned, in the precision asked for, with llfio. Its series are then views of the memory-mapped file, and
  their values stay in the page cache. Otherwise it is decoded into RAM (compressed or big-endian batches, a
  copy in the other `--dtype`, a build without llfio). Where such a file does not fit the data cap
  (`--ram-limit`, Python `ram_limit`, MATLAB `RamLimit`), non-full FastCLARA (`--method clara`, or `auto` where
  PAM's matrix does not fit) streams it, one record batch at a time; a mapped file is not streamed.
- A Parquet file is read whole into RAM, or, in a native build with Arrow (`dtwc_cl`, C++ `dtwc::run`),
  streamed through non-full FastCLARA a batch of row groups at a time where it does not fit the data cap
  (below). Python and MATLAB read it before the run, with `pyarrow` and `parquetread`, and pass the series in
  memory: whatever `ram_limit` says, it is read whole, and memory the system refuses fails the run there
  (Python `MemoryError`; MATLAB `dtwc:internal` when the MEX's copy of the series is refused:
  [Tier 1](../../api/tier-1/)). Convert a file too large for that to Arrow IPC
  (`dtwc-convert big.parquet -o big.arrow`) and a run of it maps the file, or, if it is compressed or read as
  another `--dtype`, streams it. `dtwc-convert` itself holds the series while it converts, so a file the
  machine cannot hold is converted on a larger one, or written batch by batch with pyarrow
  (`ParquetFile.iter_batches`, `pyarrow.ipc.new_file`).

The distance matrix can live in a memory-mapped
`.dtwm` file instead (`use_mmap_distance_matrix`; `dtwc_cl` maps it when it does
not fit beside the series in its `--ram-limit` plan); the Python wheels and the release CLI archives include
that support (llfio). `matrix_dtype` float32 (`--matrix-dtype`) stores the matrix
at half the bytes, each distance rounded to a float.

For data larger than an all-pairs matrix, choose a matrix-free method such as
OneBatchPAM, CLARA, or TADPole rather than changing only the input container.

The native CLI plans `--ram-limit` (0: 90 % of the memory available) for Parquet before
payload materialisation. If the conservative selected-column estimate exceeds the data cap,
only non-full FastCLARA over a single file of one series per row (a list column,
or several Float32/Float64 columns) can stream row groups; under an explicit
`--ram-limit`, a one-column file, directories, and other methods fail before loading
the payload. The automatic cap never fails a run: elsewhere the series load as they
would without a limit.
A batch holds whole row groups, decoded about 1 MiB of values at a time, and the
data cap covers what the stream holds rather than all process memory. The streaming route emits labels and medoids
without constructing dense distance or silhouette CSVs; storage requested with `--dtype f32` remains Float32 throughout the
sample, medoid, and assignment payloads.
