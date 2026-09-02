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

`Auto` spills when the estimated series footprint exceeds `ram_limit`, or —
when no limit is set — half of the memory the operating system reports as
immediately available. That figure is *not* the same quantity on every platform:
Linux uses `_SC_AVPHYS_PAGES` (MemFree, excluding reclaimable page cache),
Windows uses `MEMORYSTATUSEX::ullAvailPhys` (free plus standby), and macOS uses
`host_statistics64(HOST_VM_INFO64)` (free plus inactive pages). Linux therefore
reports the smallest budget, so the same dataset and configuration can spill to
mmap on Linux while staying on the heap on Windows or macOS. Set `ram_limit`
explicitly when you need one decision on all three. On a platform where the
query is unavailable or fails, the footprint is treated as always within budget
and data stays on the heap unless an explicit `ram_limit` or `Mmap` policy says
otherwise.

For the library APIs, `Problem.set_storage_policy` governs the next owning
`set_data` call; changing the policy does not move data already installed, and
`set_view_data` remains an explicit non-owning bypass. `Heap` retains owning
vectors. `Mmap` supports Float64 series and retains a mapped `.dtws` backing
store for the lifetime of the `Problem`; explicit Float32 Mmap is rejected.
Mapped series are not accepted by the CUDA or Metal upload paths: with
mmap-backed series installed, `fill_distance_matrix` refuses a CUDA or Metal
strategy with

```text
Problem::fill_distance_matrix: CUDA does not support mmap-backed series data;
no backend call or CPU fallback was attempted. Select StoragePolicy::Heap
before set_data.
```

and never falls back silently. For a GPU run on a dataset above the `Auto`
threshold, select `StoragePolicy::Heap` (or raise `ram_limit`) before installing
the data. The Tier-1 `cluster(...)` entry point does this for you when the
selected device is a GPU; explicit Tier-2 `Problem` use does not. The native CLI
is unaffected either way — `dtwc_cl` pins `StoragePolicy::Heap` and loads through
`DataLoader::load()`, which never routes series storage. These series policies
are separate from distance-matrix mmap and the CLI controls below.

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
