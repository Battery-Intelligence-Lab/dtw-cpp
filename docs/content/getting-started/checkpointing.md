---
title: Checkpointing
weight: 8
---

# Checkpointing

Checkpointing allows you to save and resume distance matrix computations. For large datasets, computing the full pairwise DTW distance matrix can take hours. If the process is interrupted, checkpointing lets you resume from where you left off instead of restarting from scratch.

## How it works

A checkpoint is a directory containing two files:

- **`distances.csv`** -- the NxN distance matrix, with `NaN` for pairs that have not yet been computed
- **`metadata.txt`** -- key=value pairs describing the computation state

### Metadata fields

| Key | Description |
|-----|-------------|
| `n` | Number of time series (matrix dimension) |
| `band` | Sakoe-Chiba band width used |
| `variant` | DTW variant (Standard, DDTW, WDTW, ADTW, SoftDTW) |
| `pairs_computed` | Number of distance pairs already computed |
| `timestamp` | ISO 8601 UTC timestamp of when the checkpoint was saved |

Example `metadata.txt`:

```
n=500
band=10
variant=Standard
pairs_computed=125000
timestamp=2026-03-29T14:30:00
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

Validation rules:
- The matrix dimension in the checkpoint (`n`) must match `prob.size()`
- If all pairs are computed, the distance matrix is marked as fully filled
- If only some pairs are computed, the matrix is marked as partial so remaining pairs can be computed

### CheckpointOptions

The `CheckpointOptions` struct controls automatic checkpoint behavior:

```cpp
dtwc::CheckpointOptions opts;
opts.directory = "./checkpoints";  // Directory to save checkpoint files
opts.save_interval = 100;          // Save every N pairs computed (reserved)
opts.enabled = false;              // Whether checkpointing is enabled
```

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
