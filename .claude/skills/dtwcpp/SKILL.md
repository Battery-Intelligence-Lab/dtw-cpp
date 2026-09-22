---
name: dtwcpp
description: Use the DTWC++ library to cluster time series with dynamic time warping, compute DTW distances, convert between data formats, evaluate clustering quality, plot results, or fix a dtwcpp installation. Use whenever someone wants to cluster, compare or group time-series data (sensors, battery cycles, ECG, prices, trajectories) rather than to develop this repository.
---

# Using DTWC++

This skill is for **using** the library, not developing it. Repository work is in
`.claude/CLAUDE.md`.

Each task below has a full procedure in `.claude/commands/`. Read the one you need and follow it,
treating the user's request as its `$ARGUMENTS`. Do not summarise the procedure from memory — the
step order matters, particularly data loading and method selection.

| The user wants | Follow |
| --- | --- |
| to cluster a dataset end to end | `.claude/commands/cluster.md` |
| a DTW distance or distance matrix between series | `.claude/commands/distance.md` |
| to convert data between CSV/TSV/Parquet/Arrow/numpy | `.claude/commands/convert.md` |
| to judge clustering quality (silhouette, ARI, NMI, choosing k) | `.claude/commands/evaluate.md` |
| a plot of clusters, medoids or the distance matrix | `.claude/commands/visualize.md` |
| to fix an install, a missing backend, or an error | `.claude/commands/troubleshoot.md` |
| an overview of what the library can do | `.claude/commands/help.md` |

## Before anything else

Check what this installation can actually do, and say so:

```python
import dtwcpp
dtwcpp.check_system()          # OpenMP threads, CUDA / Metal device, MPI, HiGHS
```

The flags are compile-time; `dtwcpp.test.parallelisation()` and `dtwcpp.test.gpu()` prove
engagement — threads that really ran, and a GPU result validated against the CPU oracle. Use them
before promising a device will be used.

## Things worth knowing before advising

- **Conventions are ours, not universal.** The local cost is L1, the band is in integer cells, and
  there is no final square root. dtaidistance, tslearn and aeon each differ on at least one of
  these, so numbers do not transfer without conversion. Say this whenever the user compares.
- **Pick the method by size, not by habit.** Exact PAM on a full matrix is O(N²) memory: about
  8 GB at N = 45,000. Past that, CLARA or a matrix-free route. Let `method="auto"` decide unless
  the user has a reason.
- **A band is an approximation.** It is usually a good one, but it changes the answer; say so
  rather than presenting a banded result as exact.
- **Errors are typed and name the fix.** Nothing silently falls back — if a device, format or
  solver was requested and is unavailable, that is an error, and the message says what to install.
  Read it to the user instead of guessing.
- **Variable-length series are supported**, and `max()` band infeasibility is a real failure mode:
  if a band is narrower than the length difference between two series, the pair has no valid path.

## Rules

- Never invent a result. Run the code, or say what you would run.
- Never modify the user's data in place; write outputs to a new file and name the path.
- If the dataset is large enough that a run will take minutes, say so before starting it.
