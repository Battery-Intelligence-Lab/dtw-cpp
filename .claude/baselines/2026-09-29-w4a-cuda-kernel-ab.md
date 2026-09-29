# 2026-09-29 — W4a: CUDA kernel A/B through the forced paths (RTX 4000 Ada)

**Question:** which CUDA kernel variants are within 5 % of each other, so that W4d can delete `KernelOverride`, the
fallback flags and every variant that buys nothing? (chair §4 W4; digest backends-04, backends-15.)

**Answer:** none. On every case, FP32 and FP64, each path Auto picks beats every path that could replace it in its
regime by more than 5 %: warp (the nearest alternative takes ≥ 62 % more time), regtile<4> (≥ 58 %), regtile<8>
(≥ 69 %), the double-buffer wavefront (≥ 15 %). backends-15's two deletions are FALSIFIED. W4d deletes
`KernelOverride` and the fallback flags, which force nothing Auto needs, and no kernel. Every launch of every
path equalled the host oracle bitwise. Three surprises follow the tables:
- The preload wavefront (Auto at 257–512) takes 15–17 % less time (FP32) when compiled without the double-buffer
  branch.
- A 64 KB shared-memory carveout cuts the L > 2048 path's time by 18 % (measured at L = 2049 only).
- The public entry throws at FP32 max_L 4095 and 4096, because static shared memory is left out of the 48 KiB
  opt-in test.

The baseline's fall from 293 to 90 Gcell/s (N = 50) is the regtile<8> → wavefront switch at L = 256/257. At
N = 200 in FP32 that switch drops the fill from 616 to 70 Gcell/s.

## Band — registered 2026-09-29 20:46 BST, before the first timed run

Measure: kernel-only time from two CUDA events recorded on the stream immediately around the one launch of a
path (no H2D, D2H, memset or host work inside), with the launch geometry of `launch_dtw_kernel`. Median over
R ≥ 5 repetitions; within a repetition every path runs once, in an order rotated each repetition; one untimed
warm-up launch per path first. r = median(path) / median(path Auto picks), same N, L and precision.

1. Correctness gates timing. Every launch, warm-up and timed, must reproduce the host oracle — the library's
   `dtwFull_L<T>` (L1, full band) in the kernel's own precision T — bitwise, on all N(N−1) off-diagonal entries.
   A path with one mismatch is reported wrong and its times are not used.
2. A path P that Auto picks in a regime (warp: L ≤ 32; double-buffer wavefront: 1024 < L ≤ 2048; regtile<4>:
   32 < L ≤ 128; regtile<8>: 128 < L ≤ 256) stays only if every other path that accepts the regime is > 5 %
   slower than P (r > 1.05) on at least one measured case of that regime. If some path Q has r ≤ 1.05 on every
   measured case of the regime, FP32 and FP64 both, P buys nothing there: W4d deletes P and Auto takes Q.
3. A path > 5 % faster than Auto's pick on a case (r < 0.95) means Auto should pick it there.
4. Noise: a difference outside ±5 % counts only if the two paths' repetition ranges (min–max) do not overlap.
   If they overlap, the case is re-run once; if they still overlap, it is recorded as within the band.
5. A single launch must stay under ~1.5 s (Windows WDDM timeout detection, 2 s by default). A case whose slowest
   path would exceed that runs at a smaller N, stated in its table; persistent mode must still be on.

## Machine and build

| Fact | Value |
| --- | --- |
| GPU | NVIDIA RTX 4000 Ada Generation: 20 GB, 48 SMs, compute 8.9, WDDM, 130 W cap. `query_gpu_config` classes its FP64 rate as Slow, so Auto = FP32 |
| Driver | 596.72 (CUDA 13.2 driver) |
| nvcc | 13.0.48 with `-allow-unsupported-compiler`, `--generate-code=arch=compute_89,code=[compute_89,sm_89]` |
| Host compiler | MSVC cl 19.50.35723.0 (Visual Studio 18, toolset 14.50) |
| CPU, OS | Intel Core Ultra 9 285; Windows 11 Enterprise 10.0.26200 |
| Build | `build-cuda` of the W4a worktree (recipe of `plans/2026-09-27-audit/phaseA_measurements.md`): Ninja, Release, CUDA ON, tests and benchmarks ON; Arrow, llfio, HiGHS and Gurobi OFF |
| Source | pb/W4a = design-2.0 @ `959dc5b`; no library change |
| Load | other agents built on the CPU during the run. The GPU also serves desktop apps (WDDM) over RDP. Under load the SM clock was 2010–2325 MHz, 137 of 193 one-second nvidia-smi samples at 2295–2310 MHz; 38–75 °C; the 130 W cap was reached |

## Method

Sources and raw output are in `2026-09-29-w4a-cuda-kernel-ab/`.

- **Driver.** `ab.cu` includes `dtwc/cuda/cuda_dtw.cu` verbatim and is compiled with build-cuda's nvcc flags. All
  eight library kernel instantiations have SASS identical to `dtwc++`'s `cuda_dtw.cu.obj` (`cuobjdump -sass`).
  Each path launches with `launch_dtw_kernel`'s geometry: grid, 256 threads, shared memory,
  `wavefront_buffer_count`, and the occupancy-derived persistent grid and switch. Every wavefront case at the
  N used ran persistent.
- **Clone.** `wavefront_3buf` is the library's `dtw_wavefront_kernel` with `DOUBLE_BUF_MAX = 0`, made by
  `make_clone.sh` (exactly two lines differ). It is the kernel as it would be with the double-buffer mode
  deleted, and its launch sizes 3 buffers (5 in preload mode). No runtime knob selects 3 buffers at
  1024 < L ≤ 2048; backends-15 foresaw that patch.
- **Data.** `benchmark_series_set(N, L, 200)`, bench_cuda_dtw's generator: values in [−1, 1), equal lengths. L1
  metric, full band.
- **Timing.** One untimed, checked warm-up launch per path, then 11 repetitions (7 for FP64 at L ≥ 1024), with the
  paths interleaved and their order rotated each repetition. Gcell/s = N(N−1)/2 · L² / median.
- **Oracle.** `dtwFull_L<T>` over all pairs, in the kernel's precision. Before each launch the result buffer is
  filled with NaN; the full matrix is then compared bitwise on all N(N−1) off-diagonal entries.
- **Public entry.** Each case also calls `compute_distance_matrix_cuda` with `KernelOverride` Auto, RegTile and
  Wavefront, and compares the result bitwise with the oracle.

```sh
bash make_clone.sh <repo> <out> && build.bat <repo> <out>      # in 2026-09-29-w4a-cuda-kernel-ab/
bash run_all.sh <out>                                          # results.txt = cat results/f32_* results/f64_*
uv run --no-project python summarize.py results.txt            # the tables below
ab.exe f32 1000 16 11; ab.exe f32 200 128 11                   # band rule 4 re-runs -> rerun.txt
W4A_CARVEOUT=<16|32|64|100|-1> ab.exe f32 200 <L> 7            # mechanism runs -> mech.txt
smem_edge.exe                                                  # links build-cuda's dtwc++.lib -> smem_edge.txt
```

The committed `ab.cu` differs from the binary that produced `results.txt` and `rerun.txt` in two ways: its
`#include` paths go through `-I` instead of absolute worktree paths, and it adds the `W4A_CARVEOUT` hook, which
does nothing when the variable is unset. Rebuilt from these files, all 10 kernels have the registered binary's
SASS.

## Correctness [confirmed]

- **ctest.** `ctest -R test_cuda -j1` passed all three CUDA tests, each of which ran on the GPU:
  - `test_cuda_correctness`: 48 of 49 cases, 7294 assertions. The skip is "mmap support not compiled in".
  - `test_cuda_kernel_override`: 3 cases, 429 assertions.
  - `test_cuda_launch_guards`: 4 of 5 cases, 24 assertions. The skip is "A CUDA device is present".
- **Driver.** Every path, every launch, every case and both precisions: 0 mismatches against the oracle, bitwise
  (70 path rows). The 78 public-entry calls also had 0 mismatches. No path was wrong, so every path was timed.

## Results [confirmed]

Kernel-only medians. Spread = (max − min) / median. r = median / Auto's median (Auto's pick in bold). FP64 runs at
the N given in the "FP64 N" column.

Auto's pick at N = 200:

| L | Auto path (wavefront mode) | FP32 regs, blocks/SM | FP32 Gcell/s | FP64 Gcell/s |
|---|---|---|---|---|
| 128 | regtile_w4 | 48, 5 | 483 | 35 |
| 256 | regtile_w8 | 58, 4 | 616 | 40 |
| 257 | wavefront, preload, 5 buffers | 68, 3 | 70 | 22 |
| 384 | wavefront, preload, 5 buffers | 68, 3 | 81 | 24 |
| 500 | wavefront, preload, 5 buffers | 68, 3 | 95 | 24 |
| 1024 | wavefront, 3 buffers | 68, 3 | 126 | 25 |
| 2048 | wavefront, 2 buffers (double buffer) | 68, 3 | 150 | 29 |
| 2049 | wavefront, 3 buffers | 68, 3 | 104 | 25 |

Warp vs regtile, N = 1000:

| L | path | FP32 median ms | spread | Gcell/s | r | FP64 N | FP64 median ms | spread | Gcell/s | r | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | **warp** | 1.387 | 117 % | 92 | 1.000 | 1000 | 17.154 | 10 % | 8 | 1.000 | y |
| 16 | regtile_w4 | 2.265 | 130 % | 56 | 1.633 | 1000 | 29.204 | 7 % | 4 | 1.702 | y |
| 16 | regtile_w8 | 3.429 | 2 % | 37 | 2.472 | 1000 | 50.104 | 4 % | 3 | 2.921 | y |
| 16 | wavefront | 29.484 | 9 % | 4 | 21.250 | 1000 | 44.389 | 5 % | 3 | 2.588 | y |
| 16 | wavefront_3buf | 21.850 | 11 % | 6 | 15.748 | 1000 | 32.549 | 7 % | 4 | 1.897 | y |
| 32 | **warp** | 2.462 | 3 % | 208 | 1.000 | 1000 | 33.855 | 3 % | 15 | 1.000 | y |
| 32 | regtile_w4 | 4.210 | 40 % | 122 | 1.710 | 1000 | 59.023 | 4 % | 9 | 1.743 | y |
| 32 | regtile_w8 | 6.603 | 5 % | 78 | 2.682 | 1000 | 102.684 | 2 % | 5 | 3.033 | y |
| 32 | wavefront | 51.300 | 8 % | 10 | 20.839 | 1000 | 79.165 | 3 % | 6 | 2.338 | y |
| 32 | wavefront_3buf | 36.622 | 4 % | 14 | 14.877 | 1000 | 54.568 | 2 % | 9 | 1.612 | y |

The large spreads are single-launch spikes, for example one warp launch of 3.005 ms among ten of 1.38–1.41 ms.
They left the min–max ranges of FP32 L = 16 (warp vs regtile_w4) and FP32 L = 128 (regtile_w4 vs regtile_w8)
overlapping, so band rule 4 re-ran both cases once (`rerun.txt`). The re-runs gave warp 1.402 ms [1.387–1.434]
vs regtile_w4 2.276 [2.267–2.371] (r = 1.624), and regtile_w4 0.675 [0.671–0.682] vs regtile_w8
1.073 [1.064–1.077] (r = 1.590). Neither pair of ranges overlaps, so both differences count.

Boundary set, N = 200:

| L | path | FP32 median ms | spread | Gcell/s | r | FP64 N | FP64 median ms | spread | Gcell/s | r | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | **regtile_w4** | 0.675 | 301 % | 483 | 1.000 | 200 | 9.430 | 6 % | 35 | 1.000 | y |
| 128 | regtile_w8 | 1.066 | 3 % | 306 | 1.580 | 200 | 16.454 | 7 % | 20 | 1.745 | y |
| 128 | wavefront | 8.040 | 15 % | 41 | 11.915 | 200 | 18.609 | 5 % | 18 | 1.973 | y |
| 128 | wavefront_3buf | 6.083 | 24 % | 54 | 9.014 | 200 | 15.906 | 5 % | 20 | 1.687 | y |
| 256 | **regtile_w8** | 2.119 | 12 % | 616 | 1.000 | 200 | 32.858 | 1 % | 40 | 1.000 | y |
| 256 | wavefront | 18.755 | 5 % | 70 | 8.852 | 200 | 59.331 | 5 % | 22 | 1.806 | y |
| 256 | wavefront_3buf | 15.693 | 6 % | 83 | 7.407 | 200 | 55.723 | 2 % | 23 | 1.696 | y |
| 257 | **wavefront** | 18.881 | 10 % | 70 | 1.000 | 200 | 59.476 | 5 % | 22 | 1.000 | y |
| 257 | wavefront_3buf | 15.730 | 13 % | 84 | 0.833 | 200 | 56.121 | 2 % | 23 | 0.944 | y |
| 384 | **wavefront** | 36.342 | 11 % | 81 | 1.000 | 200 | 124.395 | 1 % | 24 | 1.000 | y |
| 384 | wavefront_3buf | 30.663 | 9 % | 96 | 0.844 | 200 | 121.167 | 1 % | 24 | 0.974 | y |
| 500 | **wavefront** | 52.352 | 8 % | 95 | 1.000 | 200 | 204.468 | 3 % | 24 | 1.000 | y |
| 500 | wavefront_3buf | 44.236 | 6 % | 112 | 0.845 | 200 | 202.347 | 1 % | 25 | 0.990 | y |
| 1024 | **wavefront** | 165.027 | 2 % | 126 | 1.000 | 200 | 839.701 | 1 % | 25 | 1.000 | y |
| 1024 | wavefront_3buf | 210.471 | 2 % | 99 | 1.275 | 200 | 839.339 | 1 % | 25 | 1.000 | y |
| 2048 | **wavefront** | 558.009 | 9 % | 150 | 1.000 | 100 | 706.607 | 4 % | 29 | 1.000 | y |
| 2048 | wavefront_3buf | 800.010 | 6 % | 104 | 1.434 | 100 | 848.175 | 2 % | 24 | 1.200 | y |
| 2049 | **wavefront** | 800.712 | 1 % | 104 | 1.000 | 100 | 840.337 | 2 % | 25 | 1.000 | y |
| 2049 | wavefront_3buf | 801.253 | 1 % | 104 | 1.001 | 100 | 840.905 | 1 % | 25 | 1.001 | y |

Double vs 3 buffers, N = 200 (L = 2048 is in the table above):

| L | path | FP32 median ms | spread | Gcell/s | r | FP64 N | FP64 median ms | spread | Gcell/s | r | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1100 | **wavefront** (2 buffers) | 194.683 | 3 % | 124 | 1.000 | 200 | 833.009 | 1 % | 29 | 1.000 | y |
| 1100 | wavefront_3buf | 244.721 | 5 % | 98 | 1.257 | 200 | 962.184 | 0 % | 25 | 1.155 | y |
| 1500 | **wavefront** (2 buffers) | 328.222 | 4 % | 136 | 1.000 | 140 | 757.166 | 0 % | 29 | 1.000 | y |
| 1500 | wavefront_3buf | 417.469 | 5 % | 107 | 1.272 | 140 | 902.603 | 0 % | 24 | 1.192 | y |
| 2000 | **wavefront** (2 buffers) | 530.284 | 7 % | 150 | 1.000 | 100 | 675.402 | 1 % | 29 | 1.000 | y |
| 2000 | wavefront_3buf | 696.953 | 6 % | 114 | 1.314 | 100 | 801.137 | 1 % | 25 | 1.186 | y |

Kernel resources (FP32 / FP64 registers, from cuobjdump):
- warp: 29 / 37.
- regtile<4>: 48 / 64. regtile<8>: 58 / 90. Both regtile kernels also use a 16–64 B stack.
- library wavefront: 68 / 78.
- wavefront_3buf: 44 / 40.
- Static shared memory: 16 B in both wavefront kernels.

## Verdict per variant

| variant (Auto's regime) | nearest alternative that accepts the regime | r of the alternative, FP32 | FP64 | verdict |
|---|---|---|---|---|
| `dtw_warp_kernel` (L ≤ 32) | regtile<4> | 1.62–1.71 (L = 16, 32) | 1.70–1.74 | keep |
| regtile<4> (33–128) | regtile<8> | 1.58 (L = 128; re-run 1.59) | 1.74 | keep |
| regtile<8> (129–256) | wavefront_3buf | 7.41 (L = 256) | 1.70 | keep |
| double-buffer mode (1025–2048) | 3 buffers (clone) | 1.26–1.43 (L = 1100, 1500, 2000, 2048) | 1.16–1.20 | keep |
| wavefront, preload and 3 buffers (257–1024, > 2048) | none: only the wavefront accepts | — | — | keep |
| `KernelOverride`, `max_length_hint`, fallback flags | force only paths Auto already picks where they win | — | — | delete in W4d |

Auto should change (rule 3). At every measured L, Auto's pick is the fastest library path; at L = 2049 the clone
ties it (r = 1.001). The only faster variant is a compile variant, not a runtime path. Compiled without the
double-buffer branch, the wavefront needs 44 instead of 68 registers and runs 5 instead of 3 blocks/SM. In the
preload regime that is faster:

| L | r (FP32) | r (FP64) |
|---|---|---|
| 257 | 0.833 | 0.944 |
| 384 | 0.844 | within ±5 % |
| 500 | 0.845 | within ±5 % |

The same clone is 27.5 % slower at FP32 L = 1024 (r = 1.275). The gain therefore needs the preload mode compiled
apart, for example as a template parameter, while the global-read modes keep their current build.

## Mechanism: shared-memory carveout [observed runs; inferred cause]

The preload mode (L ≤ 512) reads both series from shared memory. The other modes read them from global memory
through L1 (`__ldg`), so the shared-memory carveout the driver picks sets how much L1 is left. These runs are in
`mech.txt`: FP32, N = 200, median of 7 repetitions. `W4A_CARVEOUT` sets the preferred carveout, in % of 100 KB, on
both wavefront kernels. Cells give ms, with blocks/SM from the occupancy API in parentheses.

| L | kernel (buffers) | default | 16 | 32 | 64 | 100 |
|---|---|---|---|---|---|---|
| 1024 | library (3) | 164.8 (3) | 327.6 (1) | 197.9 (2) | 220.3 (3) | 222.1 (3) |
| 1024 | clone (3) | 210.9 (5) | 320.9 (1) | 197.7 (2) | 210.7 (4) | 214.9 (5) |
| 2048 | library (2) | 552.3 (3) | | | 546.5 (3) | 751.0 (3) |
| 2048 | clone (3) | 783.2 (3) | | | 653.9 (2) | 788.0 (3) |
| 2049 | library (3) | 798.3 (3) | | | 657.6 (2) | 798.1 (3) |
| 2049 | clone (3) | 803.7 (3) | | | 654.9 (2) | 799.1 (3) |

Setting the value to −1 (no preference) explicitly reproduces the default at L = 1024: 164.9 and 211.4 ms.

1. **Register count acts only through occupancy.** At the same carveout and the same blocks/SM, the two kernels run
   the 3-buffer code equally fast:
   - L = 2049, at every setting;
   - L = 1024 at 32 KB, and 2 % apart at 16 KB;
   - FP64 L = 1024: 839.7 vs 839.3 ms.
2. **Most of the double buffer's lead is its smaller shared-memory footprint, not the ping-pong itself.** Forced to
   the 100 KB carveout at L = 2048, with the same 3 blocks/SM, the double buffer slows from 552 to 751 ms. That is
   4.9 % ahead of the clone in the same setting (788 ms).
3. **Deleting the double buffer would still cost ≥ 18 %.** The best 3-buffer setting measured at L = 2048 (64 KB,
   2 blocks/SM, 654 ms) takes 18 % more time than the double buffer's default (552 ms), even with a tuned carveout.
4. **Above 2048 a 64 KB carveout cuts the time by 18 %.** A 64 KB carveout at 2 blocks/SM runs L = 2049 in 658 ms,
   against 798 ms at the default (3 blocks/SM). This is one L only; W4d or W13a should A/B it across L > 2048
   before adopting it.
5. **Unknown.** The library kernel's default at L = 1024 (165 ms) beats every explicit carveout (the best is 198 ms,
   at 32 KB), so the driver's default choice there cannot be reproduced with a preference value. After two attempts
   (the sweep and the explicit −1 control) this is recorded and not pursued. L1 hit rates were not measured:
   Nsight Compute returned ERR_NVGPUCTRPERM, because GPU performance counters are not enabled for this user.

## Bug found: the 48 KiB opt-in ignores static shared memory [confirmed]

`compute_distance_matrix_cuda`, linked from build-cuda's `dtwc++.lib`, throws at FP32 max_L 4095 and 4096. The
error is `DeviceError: CUDA error at cuda_dtw.cu:1053: invalid argument`; max_L 4094 and 4097 succeed
(`smem_edge.txt`).

Cause [inferred from the arithmetic, which matches all four lengths]: `cuda_dtw.cu:1014` requests more than 48 KiB
of dynamic shared memory only when the dynamic size alone exceeds 48 KiB. The wavefront also has 16 B of static
shared memory (`s_pid`), and 3 · L · 4 + 16 > 49,152 while 3 · L · 4 ≤ 49,152 exactly for L ∈ {4095, 4096}. The FP64
3-buffer layout reaches the same edge at L = 2048: the clone's first launch failed there until its launch counted
the static bytes. Today the double buffer hides that case. The result is a typed error, not a wrong answer. W4d's
`gpu_config` rewrite should count `cudaFuncAttributes::sharedSizeBytes`.

## What W4d should delete

1. `KernelOverride` and its validators, `DistMatOptionsBase::kernel_override` and `max_length_hint`,
   `kernel_selection_length`, `KernelSelection` with `fell_back_to_auto`, `kernel_override_fell_back`, and
   `select_kernel` (callers use `auto_kernel(max_L)`); also `test_cuda_kernel_override.cpp` and its registration.
2. Rewrite the forced-path cases of `test_cuda_correctness.cpp` as auto-regime inputs, FP32 and FP64:
   L = 16, 32, 128, 256, 257–512, 1024, 1025–2048, and > 2048.
3. Delete no kernel and no wavefront mode: warp, regtile<4>, regtile<8> and the double buffer each win their
   regime by ≥ 15 %.
4. Fix in passing: the 48 KiB opt-in must count static shared memory.
5. Leads, each needing its own band: compile the preload mode apart from the double-buffer branch (15–17 % less time at
   FP32 L = 257–500); use a 64 KB carveout above L = 2048 (18 % less time at L = 2049).

## Not run

- Metal (W4e, macOS CI).
- Other GPUs, including full-rate FP64 GPUs such as the H100, where Auto picks FP64.
- FP64 at N = 200 for L ≥ 1500: those cases ran at N = 140 (L = 1500) and N = 100 (L ≥ 2000) to stay under the
  WDDM timeout.
- Banded DTW, squared L2, variable-length series, N > 1000, and L in (2049, 4096].
- Small-N launches, where tail effects dominate. A single probe at N = 50, L = 1024 turned the clone's ratio to
  0.94.
- Nsight Compute counters (permission denied).
- Public-entry wall time (H2D, D2H and host conversion). The measure is kernel-only by design.
