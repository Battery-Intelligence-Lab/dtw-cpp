# DTWC++ Session Plan -- 2026-04-02 (Codex + Claude Combined)

## Branch

- `Claude`

## Machine

- Windows 11, Intel i7-12800H (10C/20T), RTX 3070 Laptop (8 GB, CC 8.6)
- MSVC 19.40, CUDA 12.2, driver 591.74 (max CUDA 13.1)
- MS-MPI SDK installed, OpenMP via MSVC `/openmp:experimental`
- NOTE: CUDA_PATH may point to v13.2 (VS integration); Directory.Build.props pins to CUDA_PATH

## Measured Performance (2026-04-02, all optimizations applied)

### GPU vs CPU Pairwise Distance Matrix (FP32 Auto)

| N | L | GPU (ms) | CPU (ms) | Speedup |
|---|---|---|---|---|
| 20 | 100 | 0.15 | 1.12 | 7.5x |
| 50 | 100 | 0.19 | 4.12 | 22x |
| 100 | 100 | 0.47 | 15.2 | 32x |
| 100 | 500 | 25.9 | 391 | 15x |
| 200 | 500 | 96.9 | 1521 | 16x |
| 100 | 1000 | 77.4 | 1535 | 20x |

### GPU Kernel Throughput (N=50, FP32)

| L | Gcells/sec | Kernel |
|---|---|---|
| 100 | 64 | Register-tiled (TILE_W=4) |
| 250 | **144** | Register-tiled (TILE_W=8) — PEAK |
| 500 | 50 | Wavefront (preload, persistent) |
| 1000 | 63 | Wavefront (persistent) |
| 4000 | 84 | Wavefront (double-buffer) |

### Tests: 41/41 CTest, 312/312 MPI

## Build Commands

```bash
cmake -S . -B build \
  -DDTWC_BUILD_TESTING=ON -DDTWC_BUILD_BENCHMARK=ON \
  -DDTWC_ENABLE_CUDA=ON -DDTWC_ENABLE_MPI=ON
cmake --build build --config Release -j
ctest --test-dir build --build-config Release -j8
./build/bin/bench_cuda_dtw                       # GPU vs CPU benchmark
mpiexec -n 4 ./build/bin/unit_test_mpi           # MPI tests
```

## Objective

- Ship the next CUDA optimization phase on the existing public APIs and benchmarked code paths.
- Keep results exact by default.
- Stay architecture-aware across P100, V100, A30, A100, RTX-class consumer GPUs, L40S, and H100.
- Optimize by removing useless DTW work and off-chip traffic, not by chasing theoretical DRAM bandwidth numbers that do not move end-to-end runtime.

## Verified Current Bottleneck

The next serious win is already visible in `dtwc/cuda/cuda_dtw.cu`.

Current `compute_distance_matrix_cuda()` behavior for `use_lb_pruning && skip_threshold > 0` is:

1. Compute LB_Keogh values on GPU.
2. Launch full DTW for all `N * (N - 1) / 2` pairs anyway.
3. Decode pruned pairs on the host and overwrite their entries with `INF`.

That means pruning currently changes the returned matrix and the counters, but not the expensive DTW kernel work. It is a correctness-preserving post-process, not a real compute reduction. If a pruning-heavy benchmark does not reduce active DTW work on device, it is not a win yet.

The existing cached launch workspace is a good start, but it is still too pairwise-centric. The next phase should treat workspace reuse, pruning, compaction, and dispatch as first-class infrastructure shared by full-matrix, `1-vs-N`, and `K-vs-N`.

## Implementation Plan

### Phase 1: Device-side pruning and compaction

Primary target: stop launching DTW for pairs that LB_Keogh already proved can be skipped.

Required changes:

- Keep full-matrix work items as flat upper-triangle pair indices `k`.
- Use CUDA-bundled CUB only. Preferred path: `cub::DeviceSelect::Flagged`. Acceptable fallback: `DeviceScan + scatter`.
- Extend the cached CUDA workspace so it owns and reuses:
  - flattened series/query buffers
  - result buffers
  - LB buffers
  - prune flags
  - compacted active-pair buffers
  - active-count buffer
  - CUB temporary storage
  - pinned host buffers for final download
- Add a device kernel that converts LB values into prune flags against `skip_threshold`.
- Compact active pair indices on device.
- Launch DTW only on the compacted active-pair list.
- Write `INF` for pruned pairs on device before the final copy back to host.
- Preserve symmetric writes and diagonal zeroing on device.
- Update accounting so `pairs_computed` means actual DTW pairs launched, not total candidate pairs.

Implementation detail that should stay explicit:

- For full matrix, compacted work ids remain upper-triangle `k` values.
- For `1-vs-N` and `K-vs-N`, reuse the same workspace abstraction and CUB plumbing, but use flat row-major work ids because those APIs are not upper-triangular.

Recommended internal result fields:

- `upload_time_sec`
- `prune_time_sec`
- `compaction_time_sec`
- `download_time_sec`
- `active_pairs`

Do not change the public function names or add a public kernel-selection API in this phase.

### Phase 2: Internal architecture-aware dispatch

The current hard-coded thresholds are no longer enough. Move dispatch policy into an internal profile selected from `query_gpu_config()`.

Create an internal `DispatchProfile` with at least:

- `preload_threshold`
- `wavefront_block_size`
- `persistent_min_pairs`
- `preferred_shared_mem_carveout`
- `max_dynamic_smem_optin`
- `regtile_tile_width`
- `allow_fp32_auto`

Bucket GPUs by compute capability:

- `6.x` -> Pascal profile
- `7.0` -> Volta/V100 profile
- `7.5`, `8.6`, `8.9` -> Turing / consumer Ampere / Ada / L40S profile
- `8.0`, `8.7` -> A100 / A30 style HPC Ampere profile
- `9.0+` -> Hopper profile

Required behavior:

- Use the same dispatch selection for full matrix, `1-vs-N`, and `K-vs-N`.
- Remove duplicated launch heuristics so these code paths do not drift.
- Use `cudaFuncSetAttribute()` where needed for max dynamic shared memory and carveout preference instead of relying on implicit defaults.
- Keep `CUDADistMatOptions` stable in `dtwc/cuda/cuda_dtw.cuh`.

Practical guidance:

- Consumer GPUs should continue to prefer FP32 in auto mode when FP64 throughput is poor.
- HPC profiles should be allowed to spend more shared memory per block if it buys occupancy-stable wavefront execution.
- Hopper-specific policy belongs behind the profile, not scattered through kernel call sites.

### Phase 3: Benchmark and reporting expansion

Keep existing benchmark entries in `benchmarks/bench_cuda_dtw.cpp` and `benchmarks/bench_dtw_baseline.cpp` unchanged. They are regression gates, not optional legacy clutter.

Add new CUDA benchmark coverage for:

- standalone LB_Keogh throughput
- pruned full-matrix DTW with thresholding
- `1-vs-N`
- `K-vs-N`

For pruning benchmarks, use two deterministic data regimes:

- low-prune random data, to prove no meaningful regression on the unpruned path
- high-prune structured data, to prove the compacted path actually avoids DTW work

Each benchmark report should capture:

- end-to-end wall time
- LB time
- prune/flag time
- compaction time
- DTW kernel time
- download/conversion time
- total candidate pairs
- active pairs
- pruned pairs

Save raw JSON outputs under `benchmarks/results/`.

The follow-up session report for this phase should show sequential before/after runs on the same machine and toolkit combination. No mixed-toolkit comparison, no parallel benchmark runs, no cherry-picked shapes.

## Test Plan

Rerun existing correctness suites:

- `test_cuda_correctness`
- `test_cuda_lb_keogh`
- `unit_test_warping`
- `test_banded_dtw_adversarial`
- `unit_test_Problem`

Add pruning-equivalence tests for:

- threshold disabled
- threshold prunes nothing
- threshold prunes everything
- mixed prune sets
- variable-length series
- banded mode

Add result-accounting assertions for:

- `pairs_computed + pairs_pruned == total_pairs`
- symmetry preserved
- diagonal remains zero
- unpruned entries exactly match the non-pruned reference path

Benchmark acceptance criteria:

- No regression greater than 5% on existing unpruned benchmark shapes.
- Mixed-prune workloads must show both fewer computed DTW pairs and lower end-to-end runtime than the current host-postprocess design.
- Any claimed improvement must come from sequential runs on the same machine with the same effective toolkit/runtime path.

## Non-Goals For This Phase

- No approximate DTW mode.
- No low-precision output contract changes.
- No multi-GPU scheduling work.
- No public kernel-selection flags.

## Later Experimental Work

Document these, but do not block the current phase on them:

- Hopper DPX experiments if the recurrence can be mapped safely enough to justify the complexity.
- Hopper TMA or distributed shared-memory experiments if future kernels start to bottleneck on staging rather than dependency chains.
- More aggressive reduced-precision kernels behind opt-in flags for data sets that can tolerate them.

Those are future research tasks. The current branch still has enough exact, portable performance left on the table that it should not hide behind architecture-specific science projects yet.

## Definition Of Done

- Device-side pruning actually reduces launched DTW work.
- Dispatch policy is centralized and architecture-aware.
- Existing public CUDA APIs remain stable.
- Existing benchmark shapes do not regress materially.
- New pruning benchmarks show real end-to-end gains, not just better counters.
- The report for the implementation phase includes raw benchmark evidence and the exact test set that passed.

## References To Use While Implementing

- `dtwc/cuda/cuda_dtw.cu`
- `dtwc/cuda/cuda_dtw.cuh`
- `dtwc/cuda/gpu_config.cuh`
- `benchmarks/bench_cuda_dtw.cpp`
- `tests/unit/test_cuda_correctness.cpp`
- `tests/unit/test_cuda_lb_keogh.cpp`

## Closing Note

The current code already proved that host overhead and dead CPU work were worth removing. The next phase is simpler to state and harder to fake: if LB pruning says a pair is dead, the GPU should never compute it. Anything less is just moving numbers around after the fact.
