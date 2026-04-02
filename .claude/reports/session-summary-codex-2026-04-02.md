# DTWC++ Session Summary -- 2026-04-02 (Codex)

## Branch: Claude

## Machine

- Windows 11, Intel i7-12800H (10C/20T), RTX 3070 Laptop (8GB, CC 8.6)
- MSVC 19.40, CUDA 12.2 runtime path used for stable GPU benchmarking on this machine
- MS-MPI SDK installed, OpenMP via MSVC `/openmp:experimental`

## Goal

- Improve the existing benchmarked CPU/GPU paths, not a synthetic side path.
- Compare against the original `HEAD` (`e6ac63a`) in a temporary worktree on the same machine.
- Keep only changes that survive targeted correctness tests.

## What Was Done

### CPU Path Fixes

1. Made `Problem::fillDistanceMatrix()` quiet by default and added `Problem::verbose`, so benchmark runs stop paying for unconditional console I/O.
2. Reworked `fillDistanceMatrix_BruteForce()` to fill the diagonal once and compute only the upper triangle directly, instead of routing every pair through `distByInd()` and computed-flag checks.
3. Tightened `DistanceMatrixStrategy::Auto`: pruning is no longer forced for small/full-DTW cases where the lower-bound setup cost dominates and pruning ratio is effectively zero.
4. Fixed `dtwBanded()` early-abandon handling. The threshold parameter existed, but the implementation was not actually using it to terminate rows early.
5. Precomputed band bounds once per call inside `dtwBanded()` instead of recomputing them for every row transition.

### GPU Path Fixes

1. Added a reusable CUDA workspace for `compute_distance_matrix_cuda()`:
   stream, events, pinned host buffers, device buffers, and persistent counter are now reused across calls.
2. Removed repeated per-call allocation churn from the benchmark path and flattened series directly into reusable buffers.
3. Reduced host-side result conversion overhead.
4. Raised the wavefront preload threshold from `L <= 256` to `L <= 512`, which directly targets the benchmarked `L = 500` regime.
5. Standardized the wavefront long-series launch to `256` threads for the benchmarked shapes.
6. Aligned the 1-vs-N/K-vs-N wavefront shared-memory sizing with the new preload threshold.

## Files Changed

- `dtwc/Problem.cpp`
- `dtwc/Problem.hpp`
- `dtwc/warping.hpp`
- `dtwc/cuda/cuda_dtw.cu`
- `dtwc/dtwc_cl.cpp`

## Benchmark Results

All numbers below come from the existing benchmark binaries:

- `build/bin/bench_dtw_baseline.exe`
- `build/bin/bench_cuda_dtw.exe`

Baseline = original `HEAD` (`e6ac63a`) rebuilt separately on the same machine.

### GPU Distance Matrix

| Benchmark | Baseline | Codex | Speedup |
|---|---:|---:|---:|
| `BM_cuda_distanceMatrix/20/100` | `0.626 ms` | `0.189 ms` | `3.31x` |
| `BM_cuda_distanceMatrix/50/100` | `0.755 ms` | `0.240 ms` | `3.15x` |
| `BM_cuda_distanceMatrix/100/100` | `1.09 ms` | `0.452 ms` | `2.41x` |
| `BM_cuda_distanceMatrix/20/500` | `15.1 ms` | `1.19 ms` | `12.7x` |
| `BM_cuda_distanceMatrix/50/500` | `90.0 ms` | `5.99 ms` | `15.0x` |
| `BM_cuda_distanceMatrix/100/500` | `359 ms` | `23.2 ms` | `15.5x` |
| `BM_cuda_distanceMatrix/50/1000` | `465 ms` | `19.2 ms` | `24.2x` |
| `BM_cuda_distanceMatrix/100/1000` | `1874 ms` | `74.6 ms` | `25.1x` |
| `BM_cuda_distanceMatrix/200/500` | `1435 ms` | `91.9 ms` | `15.6x` |

### CPU Distance Matrix

| Benchmark | Baseline | Codex | Speedup |
|---|---:|---:|---:|
| `BM_fillDistanceMatrix/20/100/-1` | `1.43 ms` | `0.990 ms` | `1.44x` |
| `BM_fillDistanceMatrix/50/100/-1` | `4.69 ms` | `3.57 ms` | `1.31x` |
| `BM_fillDistanceMatrix/20/500/-1` | `27.7 ms` | `26.1 ms` | `1.06x` |
| `BM_fillDistanceMatrix/50/500/-1` | `102 ms` | `81.6 ms` | `1.25x` |
| `BM_fillDistanceMatrix/50/500/10` | `6.88 ms` | `4.48 ms` | `1.54x` |
| `BM_fillDistanceMatrix/50/500/50` | `22.7 ms` | `17.0 ms` | `1.34x` |
| `BM_fillDistanceMatrix/50/1000/50` | `47.2 ms` | `34.1 ms` | `1.38x` |

### Additional CPU Check From `bench_cuda_dtw`

| Benchmark | Baseline | Codex | Speedup |
|---|---:|---:|---:|
| `BM_cpu_distanceMatrix/50/500` | `89.4 ms` | `82.8 ms` | `1.08x` |

## Validation

Targeted tests passed after the changes:

- `test_banded_dtw_adversarial`
- `unit_test_pruned_distance_matrix`
- `test_cuda_correctness`
- `test_cuda_lb_keogh`
- `unit_test_Problem`
- `unit_test_warping`

## Notes

- The original benchmarked CPU path was spending time on logging and on an `Auto -> Pruned` choice that produced essentially zero pruning on these synthetic inputs.
- The original benchmarked GPU path was dominated by host-side allocation and transfer overhead relative to the actual kernel work for the tested sizes.
- Net effect: the existing benchmarks got faster without inventing friendlier benchmarks.
- If Claude wants to catch up, he can start by making `early_abandon` actually abandon.
