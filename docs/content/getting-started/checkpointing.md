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
| 4-7 | version (uint32) = 4 |
| 8-15 | N, the number of series (uint64) |
| 16-47 | SHA-256 fingerprint of everything that can change a distance |
| 48- | the packed lower triangle, N(N+1)/2 doubles; NaN = not computed |

The fingerprint covers the exact series bits, their order, lengths, dtype and
`ndim`; the band and every variant and multivariate parameter; the missing-data
strategy; the pointwise metric; the backend and its precision. Series names are
not part of it. Integers and doubles are in the host byte order, which is
little-endian on every supported platform.

## Loading: the outcomes

| The file | `load_checkpoint` | `use_mmap_distance_matrix` |
|----------|-------------------|----------------------------|
| absent | returns `false`; nothing changes | creates it, every entry NaN |
| for other series (another N) or other settings (another fingerprint) | `InvalidInput` | `InvalidInput` |
| short, not a `.dtwm` file, another version (1-3 are earlier cache layouts), or a length that does not fit its N | `IOError` | `IOError` |
| a match | the distances are loaded, bit for bit | the file is mapped with its distances |

A rejected file is left as it is, and so is the `Problem`. Delete or rename such a
file to compute the distances again. A checkpoint is never silently recomputed
over, and a file for other data is never silently accepted.

## C++ API

```cpp
#include "dtwc/checkpoint.hpp"

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

`fill_distance_matrix()` then fills rows in consecutive blocks of `save_interval`
rows and saves after each block, the last included. `save_interval >= 1` and a
non-empty directory are required; either violation raises `InvalidInput` before
any distance is computed. A save that fails propagates out of the fill; the
distances computed so far stay in the `Problem`.

A save of a matrix in RAM writes all N(N+1)/2 doubles, 4N² bytes, so choose
`save_interval` so that a block (about `save_interval * N` DTW computations) costs
much more than a save.

## Memory-mapped matrix

`prob.use_mmap_distance_matrix(path)` keeps the matrix in a `.dtwm` file instead
of RAM (the wheels and release archives have it; a source build needs
`DTWC_ENABLE_LLFIO=ON`, the default, and raises `IOError` without it). The fill
writes into the file through the page cache, so the file is a checkpoint at every
moment: a process that dies keeps every distance written, and reopening the file
with the same data and settings resumes. A `Problem` mapped to
`<directory>/<name>.dtwm` saves by flushing the mapping in place, so
`CheckpointOptions` with that directory flushes after every block.

A new file is filled with NaN and flushed to the device before its header is
written, so a power cut during creation leaves a file that does not open
(`IOError`), never one whose zeros read as distances.

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
2. loads `<dir>/<name>.dtwm` if it exists (`--name`, default `dtwc`); a file for
   other data or settings, or a damaged one, stops the run with exit status 1 and
   the reason, before anything is computed;
3. saves the matrix there after the result files are written, and every
   `--checkpoint-interval` rows during the fill when that is non-zero. A save that
   fails (a full disk) stops the run with exit status 1; the results already
   written stay.

When `--mmap-threshold` selects mapped storage, the mapped matrix is
`<dir>/<name>.dtwm` itself: it is reopened if present and written in place.
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
