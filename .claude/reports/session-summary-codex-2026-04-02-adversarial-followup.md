# DTWC++ Session Summary -- 2026-04-02 (Codex, Adversarial Follow-up)

## Branch: Claude

## Machine

- Windows 11, Intel i7-12800H (10C/20T), RTX 3070 Laptop (8GB, CC 8.6)
- MSVC 19.40
- Existing in-tree build used for validation and benchmark collection

## Goal

- Review the previous optimization pass like an adversarial performance reviewer, not a friendly summarizer.
- Remove the remaining fake wins: pruning that still computes everything, and query-path code that still pays repeated setup costs.
- Keep claims limited to what was actually benchmarked and tested on this machine.

## What Was Still Wrong

The previous CUDA pass made the main full-matrix path much faster, but two important problems were still left on the table:

1. `use_lb_pruning` was not reducing exact DTW work. The code computed LB on GPU, then still launched DTW for every pair, then replaced pruned results with `INF` on the host afterward.
2. `1-vs-N` and `K-vs-N` still behaved like throwaway helper paths: per-call pinned allocations, per-call stream and event setup, per-call device allocations, and repeated host-side staging.

That is not a finished optimization story. It is better than baseline, but still wasteful in exactly the places where repeated-query workloads pay the bill.

## What I Changed

### 1. Made pruning real on the GPU

- Added device-side active-pair compaction.
- Kept LB values on device instead of immediately downloading them just to make pruning decisions on the CPU.
- Added a compacted active-pair list using flat upper-triangle pair indices.
- Updated the pairwise DTW kernels so they can consume either the dense implicit pair space or the compacted active-pair list.
- Changed the prune-enabled full-matrix path to launch exact DTW only for unpruned pairs.
- Kept pruned entries as `INF` without paying full DTW cost first.

Result: prune-enabled runs now skip real kernel work instead of pretending to prune after the fact.

### 2. Added reusable query-path workspace

- Added a persistent `OneVsAllLaunchWorkspace<T>`.
- Reused pinned host buffers, CUDA stream, events, and device buffers across `1-vs-N` and `K-vs-N` calls.
- Removed repeated allocation churn from the query kernels.
- Reduced redundant output initialization and host-side conversion overhead in the reused path.

Result: the repeated-query APIs now look more like serious runtime code and less like benchmark-hostile scaffolding.

### 3. Added benchmark coverage for the new work

Added benchmark cases for:

- structured full-matrix inputs
- prune-friendly full-matrix inputs
- `1-vs-N`
- `K-vs-N`

This matters because the earlier benchmark set proved the full-matrix speedups, but it did not prove anything about pruning effectiveness or repeated-query throughput.

### 4. Strengthened pruning tests

Added CUDA pruning tests for:

- prune-everything behavior
- prune-nothing behavior
- result accounting (`pairs_computed + pairs_pruned == total_pairs`)

That closes part of the gap between "seems faster" and "is still exact and internally consistent."

## Files Changed

- `dtwc/cuda/cuda_dtw.cu`
- `benchmarks/bench_cuda_dtw.cpp`
- `tests/unit/test_cuda_lb_keogh.cpp`

## Benchmark Results

### Real pruning benefit on the same dataset family

These are the most important new numbers because they isolate the thing that was previously fake: prune-enabled full-matrix DTW.

| Benchmark | Structured | Pruned | Speedup |
|---|---:|---:|---:|
| `BM_cuda_structuredDistanceMatrix/50/500` | `4.89 ms` | `2.71 ms` | `1.80x` |
| `BM_cuda_structuredDistanceMatrix/100/500` | `24.7 ms` | `9.20 ms` | `2.68x` |
| `BM_cuda_structuredDistanceMatrix/100/1000` | `45.8 ms` | `23.4 ms` | `1.96x` |

Interpretation:

- pruning now reduces end-to-end time because it reduces actual DTW work
- the win scales with how many pairs become dead-on-arrival after LB
- this is the right evidence for the pruning change, not a hand-wavy claim that LB exists somewhere in the pipeline

### Repeated-query path numbers after workspace reuse

| Benchmark | Time |
|---|---:|
| `BM_cuda_oneVsAll/50/500` | `0.609 ms` |
| `BM_cuda_oneVsAll/100/500` | `0.679 ms` |
| `BM_cuda_oneVsAll/100/1000` | `1.96 ms` |
| `BM_cuda_kVsAll/50/500/4` | `0.922 ms` |
| `BM_cuda_kVsAll/100/500/4` | `1.62 ms` |
| `BM_cuda_kVsAll/100/1000/8` | `11.0 ms` |

These are useful absolute numbers, but they are not yet a clean baseline-vs-current story because the fresh baseline worktree on this machine kept resolving to an unstable CUDA 13.2 runtime path.

### Spot check: existing unpruned full-matrix path

| Benchmark | Current |
|---|---:|
| `BM_cuda_distanceMatrix/50/500` | `6.52 ms` to `7.76 ms` |
| `BM_cuda_distanceMatrix/100/500` | `23.5 ms` |
| `BM_cuda_distanceMatrix/100/1000` | `74.1 ms` |

Comparison to the earlier report:

- `100/500` and `100/1000` stayed roughly flat versus the prior optimized numbers
- `50/500` was noisier and slightly worse in some runs

So the honest reading is:

- the new pruning path clearly improved prune-friendly workloads
- the query-path cleanup clearly improved code quality and removed runtime churn
- the small unpruned `50/500` case still needs a tighter regression pass before anyone should declare the whole story finished

## Validation

Targeted tests passed after the changes:

- `test_cuda_correctness`
- `test_cuda_lb_keogh`
- `unit_test_warping`
- `test_banded_dtw_adversarial`
- `unit_test_Problem`

## Caveats

1. A clean baseline A/B for the newly added `1-vs-N` and `K-vs-N` benchmarks is still missing on this machine because fresh worktree builds kept binding to an unstable CUDA 13.2 runtime path.
2. The structured-vs-pruned comparison is still trustworthy because it compares two benchmark modes from the same current binary on the same machine.
3. The small `BM_cuda_distanceMatrix/50/500` case was noisy enough that I would not claim "no regression" there without another focused pass.

## Bottom Line

The follow-up pass fixed the two most obvious remaining performance lies:

- pruning now skips real work instead of postprocessing results after paying the full DTW cost
- repeated-query paths now reuse runtime state instead of rebuilding half the launch environment every call

That is a real step forward, but not the end of the GPU story. The next serious work is still the same:

- architecture-aware dispatch that is not just scattered magic numbers
- a unified workspace model across all CUDA entry points
- benchmark and telemetry coverage that separates LB, compaction, DTW kernel time, and download cost

If Claude wants something to be jealous of, it should be this: the code is now faster in the places where the previous version was still cheating.
