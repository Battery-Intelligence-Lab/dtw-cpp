---
title: Checkpointing
weight: 8
---

# Checkpointing

Checkpointing saves the distance matrix so that a *later* run skips the pairs a
*previous* run already computed. For large datasets the full pairwise DTW matrix
can take hours. Nothing resumes a clustering method in mid-iteration.

## One file

A checkpoint is one file, `<directory>/<name>.dtwm`, named after the `Problem`
(`distances.dtwm` for an unnamed one). It is the same file a memory-mapped
distance matrix lives in, so a checkpoint can be mapped and a mapped matrix loaded
as a checkpoint:

| Bytes | Content |
|-------|---------|
| 0-3 | magic `DTWM` |
| 4-7 | version (uint32) = 5 |
| 8-15 | N, the number of series (uint64) |
| 16-47 | SHA-256 fingerprint of everything that can change a distance |
| 48-51 | the bytes of a value (uint32): 8 for float64, 4 for float32 (`matrix_dtype`) |
| 52-55 | zero |
| 56-63 | R, the rows written (uint64) |
| 64- | the packed lower triangle, N(N+1)/2 values; NaN = not computed |

Rows from R on were never written and hold no distance, whatever the file has
there: a new mapped file is not filled with NaN (R = 0), and a fill records rows
in R only once they are on the disk, so a run stopped mid-fill resumes from the
rows it records. Version 4, a 48-byte header (bytes 0-47 above) and doubles with
every row written, from 2.0 builds before `matrix_dtype`, is read and mapped as
float64; a mapped one stays version 4, and every other save writes version 5.

The fingerprint covers the exact series bits, their order, lengths, dtype and
`ndim`; the band and every variant and multivariate parameter; the missing-data
strategy; the pointwise metric; the precision the distances are computed in
(the series' on the CPU, the GPU precision on a GPU). The value type the matrix
stores them in is the header's bytes 48-51, not the fingerprint: a file of the
other `matrix_dtype` is refused, naming the one that reads it, and never
converted. The device is not: FP64
distances from the CPU and from a GPU differ by rounding only, so either one's
file serves the other, and GPU 0's serves GPU 1; an FP32 file is refused by an
FP64 run. On Metal, which has no FP64, precision `auto` is FP32. On CUDA it
resolves on the GPU a fill meets, so a CUDA run with `auto` cannot write or read
a file: select `fp32` or `fp64`. Series names are not part of it. Integers and doubles are in the host byte order, which is
little-endian on every supported platform.

## Loading: the outcomes

| The file | `load_checkpoint` | `use_mmap_distance_matrix` |
|----------|-------------------|----------------------------|
| absent | returns `false`; nothing changes | creates it, no row written |
| for other series (another N), other settings (another fingerprint) or the other `matrix_dtype` | `InvalidInput` | `InvalidInput` |
| short, not a `.dtwm` file, another version (1-3 are earlier cache layouts), or a length that does not fit its N | `IOError` | `IOError` |
| a match | the distances are loaded, bit for bit | the file is mapped with its distances |

A rejected file is left as it is, and so is the `Problem`. Delete or rename such a
file to compute the distances again. A checkpoint is never silently recomputed
over, and a file for other data is never silently accepted.

## C++ API

```cpp
#include "checkpoint.hpp"

dtwc::Problem prob("my_problem", loader);
if (dtwc::load_checkpoint(prob, "./checkpoints"))   // ./checkpoints/my_problem.dtwm
    std::cout << "Resumed from checkpoint\n";        // computed pairs are kept
prob.fill_distance_matrix();                         // computes only the rest
dtwc::save_checkpoint(prob, "./checkpoints");
```

`save_checkpoint` creates the directory if it is missing, writes
`<name>.dtwm.tmp`, flushes it to the device and renames it over the previous
checkpoint, so a crash or power cut during a save leaves the previous file whole.
It raises `IOError` when the directory or the file cannot be written and
`InvalidInput` for a `Problem` without series. `load_checkpoint` returns `false`
only when there is no file; its C++ result is `[[nodiscard]]`. A loaded matrix is
held in RAM; a `Problem` whose matrix was mapped lets go of the mapping.

The fingerprint includes the `Problem`'s metric (`prob.set_metric(...)`, L1 by
default). `save_checkpoint(prob, path, metric)` and
`load_checkpoint(prob, path, metric)` tag or expect an explicit metric instead,
for a matrix that something other than the `Problem` computed.

### CheckpointOptions

`CheckpointOptions` drives automatic saving through `Problem::checkpoint`:

```cpp
prob.checkpoint.directory = "./checkpoints"; // Directory of <name>.dtwm
prob.checkpoint.save_interval = 100;        // Completed matrix rows between saves
prob.checkpoint.enabled = true;             // Enable automatic saving
prob.fill_distance_matrix();                // Saves every 100 rows and at the end
```

`fill_distance_matrix()` then saves after every `save_interval` rows of the
packed matrix, rounded up to the lane width (16 rows of doubles where the lanes are
128 bytes), the last save at the end. `save_interval >= 1` and a
non-empty directory are required; either violation raises `InvalidInput` before
any distance is computed. A save that fails propagates out of the fill; the
distances computed so far stay in the `Problem`.

The first save of a fill writes the whole matrix, N(N+1)/2 values, unless the
file already holds this matrix; each later save writes only the rows filled since,
in place, and flushes them to the device. A fill writes at most twice the matrix's
bytes, however small `save_interval` is, and whenever a save stops, every entry of
the file is NaN or its distance.

## Memory-mapped matrix

`prob.use_mmap_distance_matrix(path)` keeps the matrix in a `.dtwm` file instead
of RAM (the wheels and release archives have it; a source build needs
`DTWC_ENABLE_LLFIO=ON`, the default, and raises `IOError` without it). The fill
writes into the file through the page cache, so the file is a checkpoint at every
moment: a process that dies keeps the rows the file records as written (every
slab but the last one or two, which a resume computes again), and reopening the
file with the same data and settings resumes. A `Problem` mapped to
`<directory>/<name>.dtwm` saves by flushing the mapping in place, so
`CheckpointOptions` with that directory flushes the rows of every save. A mapped
matrix is filled in slabs of an eighth of what the series leave of the
`Problem`'s `ram_limit` (0, the default: the memory available), at most 1 GiB:
the next slab is read ahead while one fills, and a finished one is written back.

A new file gets its header, with no row written, and no NaN: its zeros are never
read as distances. Each slab makes its rows NaN as it starts, and the header's
rows written advance only once the rows before are on the device, so a power cut
leaves a file whose recorded rows are all distances or NaN. A direct
`dist_by_ind` of a row past them, before the fill, reads 0.

Set data, band, variant, backend, metric and precision before mapping. The
semantic setters detach the mapping and leave the file as it is. Raw in-place
series edits after the first use are unsupported because warm lookup is kept
O(1): call `refresh_distance_matrix()` before editing, or use `set_data()`. CUDA
matrices need an explicit FP32 or FP64 precision rather than `auto`.

## Python

```python
import dtwcpp

prob = dtwcpp.Problem("my_clustering")
prob.set_data(series, names)

if not dtwcpp.load_checkpoint(prob, "./checkpoints"):  # ./checkpoints/my_clustering.dtwm
    print("No checkpoint yet")
prob.fill_distance_matrix()
dtwcpp.save_checkpoint(prob, "./checkpoints")
```

The optional `metric` argument mirrors the CLI's `--metric` and is part of the
fingerprint, so a matrix written under one metric is refused by a run using
another:

```python
dtwcpp.save_checkpoint(prob, "./ckpt_sq", dtwcpp.MetricType.SquaredL2)

dtwcpp.load_checkpoint(prob, "./ckpt_sq")                              # InvalidInput
dtwcpp.load_checkpoint(prob, "./ckpt_sq", dtwcpp.MetricType.SquaredL2)  # True
```

Both functions release the GIL. `load_checkpoint` replaces the `Problem`'s
matrix: do not run it concurrently with another method on the same `Problem`.
`dtwcpp.CheckpointOptions` and `Problem.checkpoint` mirror the C++ options.

## CLI

```bash
dtwc_cl --input data.csv -k 5 --method pam --checkpoint ./checkpoints
```

With `--checkpoint <dir>` the CLI

1. creates the directory before any data is read, and stops with exit status 1
   if the path is not a directory;
2. loads `<dir>/<name>.dtwm` if it exists (`--name`, default the input's file or folder name); a file for
   other data or settings, or a damaged one, stops the run with exit status 1 and
   the reason, before anything is computed;
3. saves the matrix there after the result files are written, and every
   `--checkpoint-interval` rows during the fill when that is non-zero. A save that
   fails (a full disk) stops the run with exit status 1; the results already
   written stay.

When the matrix is mapped (its bytes over half of what the series leave of
`--ram-limit`), the mapped matrix is `<dir>/<name>.dtwm` itself: it is reopened if
present and written in place.
Without `--checkpoint`, a mapped matrix is `<name>.dtwm` in the output directory.

## Example workflow

```bash
dtwc_cl --input large_dataset.csv -k 10 --method pam --checkpoint ./ckpt --verbose
```

```
Data loaded: 2000 series [0.5s]
No checkpoint in ./ckpt, starting fresh.
Running FastPAM (k=10) ...
```

After an interruption, run the same command again:

```
Data loaded: 2000 series [0.5s]
Resumed from checkpoint: ./ckpt
Running FastPAM (k=10) ...
```

The fill skips every pair the checkpoint holds. A run with `--checkpoint-interval`
saves during the fill, so an interrupted fill resumes from its last block.
