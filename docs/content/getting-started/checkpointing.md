---
title: Checkpointing
weight: 8
---

# Checkpointing

Checkpointing allows you to save and resume distance-matrix computation. For
large datasets, computing the full pairwise DTW matrix can take hours. If that
work is interrupted, a directory checkpoint or validated mmap cache can avoid
recomputing finished pairs. The separate binary mechanism replays a completed
clustering result; it does not resume a method in mid-iteration.

DTWC++ has three persistence mechanisms: the directory checkpoint documented
below, a binary clustering-result checkpoint used by `--resume`, and the packed
memory-mapped distance cache selected by `--mmap-threshold`. The mmap cache
resumes automatically when its identity matches; it is not the same format as
the dense CSV directory checkpoint.

## Directory checkpoint format

Directory checkpoint v2 publishes a small root `CURRENT` file whose lowercase
64-hex generation ID selects immutable
`generations/<id>/{distances.csv,metadata.txt}` files. A successful save writes
and validates a new generation before atomically replacing `CURRENT`; it does
not overwrite the active generation in place.

- **`distances.csv`** -- the exact N-by-N matrix; an empty field is the only
  uncomputed representation.
- **`metadata.txt`** -- the exact seven-key manifest described below.

### Metadata fields

| Key | Description |
|-----|-------------|
| `n` | Number of time series (matrix dimension) |
| `pairs_computed` | Number of distance pairs already computed |
| `timestamp` | ISO 8601 UTC timestamp of when the checkpoint was saved |
| `format` / `version` | `dtwc-dense-checkpoint` / `2` |
| `identity_sha256` | Exact data and distance-configuration identity |
| `payload_sha256` | Digest of the canonical CSV bytes |

Example `metadata.txt`:

```
format=dtwc-dense-checkpoint
version=2
n=500
pairs_computed=125000
timestamp=2026-03-29T14:30:00Z
identity_sha256=<64 lowercase hex characters>
payload_sha256=<64 lowercase hex characters>
```

## C++ API

### save_checkpoint

Save the current distance matrix state to a checkpoint directory. Creates the directory if it does not exist.

```cpp
#include "dtwc/checkpoint.hpp"

dtwc::Problem prob("my_problem", loader);
// ... compute some or all of the distance matrix ...

dtwc::save_checkpoint(prob, "./checkpoints/run1");
```

### load_checkpoint

Load a checkpoint and restore the distance matrix into the Problem. Returns `true` on success, `false` if no valid checkpoint was found or the dimensions do not match.

```cpp
dtwc::Problem prob("my_problem", loader);

if (dtwc::load_checkpoint(prob, "./checkpoints/run1")) {
    std::cout << "Resumed from checkpoint\n";
    // Continue computation -- already-computed pairs are preserved
} else {
    std::cout << "No checkpoint found, starting fresh\n";
}
```

Load validates `CURRENT`, the exact manifest, full data/configuration identity,
payload digest, CSV shape, finite full-token values, bit-identical symmetry, and
computed-pair count before publishing any matrix state. Missing, incompatible,
legacy, or malformed state returns `false` without changing `Problem`.

### CheckpointOptions

`CheckpointOptions` is currently a passive configuration carrier:

```cpp
dtwc::CheckpointOptions opts;
opts.directory = "./checkpoints";  // Directory to save checkpoint files
opts.save_interval = 100;          // Save every N pairs computed (reserved)
opts.enabled = false;              // Whether checkpointing is enabled
```

No algorithm or save/load function currently consumes these fields. Call
`save_checkpoint` and `load_checkpoint` explicitly.

## Python API

The same functions are exposed through the Python bindings:

```python
import dtwcpp

prob = dtwcpp.Problem("my_clustering")
prob.set_data(series, names)

# Save checkpoint
dtwcpp.save_checkpoint(prob, "./checkpoints/run1")

# Load checkpoint (returns True/False)
if dtwcpp.load_checkpoint(prob, "./checkpoints/run1"):
    print("Resumed from checkpoint")
```

The `CheckpointOptions` class is also available:

```python
opts = dtwcpp.CheckpointOptions()
opts.directory = "./checkpoints"
opts.enabled = True
```

## CLI usage

The command-line tool supports checkpointing via the `--checkpoint` flag:

```bash
dtwc_cl --input data.csv -k 5 --method pam --checkpoint ./checkpoints
```

When `--checkpoint` is specified, the CLI will:

1. **On startup**: attempt to load a checkpoint from the given directory. If a valid checkpoint is found and the dimensions match, the saved distance matrix is restored.
2. **On completion**: save the current state to the checkpoint directory, so it can be resumed if run again.

### Memory-mapped cache safety and migration

The current mmap cache is format version 3. Its 64-byte header contains a
SHA-256 fingerprint of all inputs that can change a distance: exact series bits,
order, lengths, dtype and `ndim`; band and every variant/multivariate parameter;
missing-data strategy; pointwise metric; backend; and backend precision. Names
are intentionally excluded. Header CRC, reserved bytes, and exact file length
are validated before cached distances can be read.

A same-sized cache from different data or configuration therefore fails loudly
instead of returning stale distances. A footer holds two digest words per
logical row; reopening under a nonblocking exclusive file lease recomputes
those digests before exposing the mapping. This detects accidental packed-value
or computed-sentinel corruption, including mutations made through the legacy
raw pointer. It is not keyed cryptographic authentication, and `sync()` remains
the durability boundary.

Version-1 caches contained only the matrix dimension. Version 2 authenticated
the data/configuration identity but not the mutable packed payload. Both are
rejected with recompute guidance. Delete or rename a legacy/mismatched cache and
rerun; the source data is not modified.

Set data, band, variant, backend, metric, and precision before binding a cache.
Use the semantic setters after binding; they detach the old mapping. Raw in-place
series edits after the first cache use are unsupported because warm lookup is
kept O(1): call `refresh_distance_matrix()` before editing, or use `set_data()`.
CUDA-backed mmap caches require explicit FP32 or FP64 rather than `auto`.

When `--mmap-threshold` selects mmap storage, `--checkpoint` and
`--dist-matrix` are rejected because both require a legacy dense CSV matrix.
Omit those options to use the fingerprinted cache's automatic resume, or raise
the threshold only if the dense matrix and CSV checkpoint fit in RAM.

## Completed binary-result replay

Every successful fresh CLI clustering run writes
`<output>/<name>_checkpoint.bin`. `--resume` selects that exact path, validates
the result against the current N and k, restores labels, medoids, total cost,
iterations, and convergence, and skips clustering:

```bash
dtwc_cl --input data.csv -k 5 --output ./results --name run1
dtwc_cl --input data.csv -k 5 --output ./results --name run1 --resume
```

The source binary remains unchanged. `converged=false` is replayed as a
completed iteration-capped result; `--max-iter` is not an extra continuation
budget. Missing or incompatible requested state fails instead of silently
starting a fresh clustering run.

Binary format v1 does not store data/configuration identity, input order, or
the method that produced the result. Use the same input order and clustering
configuration, and treat the unconditional `checkpoint replay` summary as the
result source rather than method provenance. Result replay is independent of
the directory checkpoint and mmap cache. An explicitly imported or already
available distance matrix may still be used for scoring, but replay does not
create or compute unused matrix state.

## Example workflow

### Start a long computation

```bash
dtwc_cl --input large_dataset.csv -k 10 --method pam \
        --checkpoint ./ckpt --verbose
```

Output:

```
Data loaded: 2000 series [0.5s]
No checkpoint found at ./ckpt, starting fresh.
Running FastPAM (k=10) ...
```

### Interrupt and resume

If the process is interrupted (e.g., Ctrl+C, system crash), restart with the same command:

```bash
dtwc_cl --input large_dataset.csv -k 10 --method pam \
        --checkpoint ./ckpt --verbose
```

Output:

```
Data loaded: 2000 series [0.5s]
Checkpoint partially loaded from ./ckpt (1250000/4000000 entries computed)
Resumed from checkpoint: ./ckpt
Running FastPAM (k=10) ...
```

The computation continues from where it left off, skipping the already-computed distance pairs.

### Python equivalent

```python
import dtwcpp

series = [...]  # 2000 time series
names = [str(i) for i in range(len(series))]

prob = dtwcpp.Problem("large_run")
prob.set_data(series, names)
prob.band = 10

# Try to resume
if not dtwcpp.load_checkpoint(prob, "./ckpt"):
    print("Starting fresh")

# Run clustering (computes remaining distances as needed)
result = dtwcpp.fast_pam(prob, n_clusters=10, max_iter=100)

# Save for future runs
dtwcpp.save_checkpoint(prob, "./ckpt")

print(f"Total cost: {result.total_cost}")
print(f"Labels: {result.labels}")
```
