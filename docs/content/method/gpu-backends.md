---
title: GPU Backends
weight: 6
---

# GPU backends — CUDA and Metal

DTW is embarrassingly parallel across pairs but **sequential within** each pair (the DP recurrence has a diagonal dependency). DTWC++ exploits the parallelism with two GPU backends:

- **CUDA** (NVIDIA) — consumer and HPC discrete GPUs of compute capability 8.0 or newer (Ampere, 2021: A30/A100/RTX 30
  and later). An older GPU is refused with `DeviceError` naming its compute capability, before anything is allocated.
- **Metal** (Apple Silicon) — targets M-series integrated GPUs.

Both inherit a shared option/result base, then add backend-specific fields and
defaults. Through `Problem`, an unavailable or uncompiled requested backend
raises `DeviceError` rather than changing to CPU. A Metal buffer allocation or
kernel that fails raises `DeviceError` too.

Both write the matrix the CPU fill writes, `DistanceMatrix`'s packed lower
triangle, in place, whether it is on the heap or memory-mapped; the result
structs report what ran (kernel, pair count, GPU time), not the distances. A
backend that refuses a request (no device, a GPU that cannot hold the fill's
buffers) does so before the matrix is allocated. Both accept series of any
length. CUDA computes at most 2^27 pairs per launch and copies each launch's
share of the matrix to the host as it finishes: the GPU holds the padded series
and about 1 GiB of output whatever N is (plus, for series too long for shared
memory, three anti-diagonals per resident block), and the matrix itself lives in
host memory.

## What a `Problem` can run on a GPU

`Problem::set_device(Device::GPU)` (Python `Problem(device="gpu")`, MATLAB
`dtwc.Problem(name, 'Device', 'gpu')`) selects the build's backend, CUDA else
Metal, and `Problem::set_gpu_precision(GpuPrecision)` what it computes in. Before
any pair is computed, one validator checks the request
against what the kernels implement — Standard DTW on univariate Float64 series
held in RAM, L1 or squared L2, `MissingStrategy::Error` — and anything else
raises `DeviceError` naming the setting and its value:

| Request | GPU result |
|---|---|
| a variant other than Standard, a missing-data strategy, `ndim > 1` | `DeviceError` |
| Float32, mmap-backed or view-mode series | `DeviceError` |
| squared L2 (`Problem::set_metric(MetricType::SquaredL2)`, dense or mapped) | computed with `use_squared_l2` |
| Metal: precision FP64, or a GPU index other than 0 (refused by `set_device`) | `DeviceError` |
| CUDA: precision Auto, FP32 or FP64; any device index | honoured |
| a band narrower than the longest-minus-shortest series length (every device) | `InvalidInput` naming both series and the smallest feasible band |

The same validator runs in `Problem::dtw_function()`, through which
OneBatchPAM, FastCLARA's assignment and TADPole compute the pairs they need.
Known gap: those matrix-free schedules compute on the CPU when the device is a
GPU.

> **Compile-time flags.** CUDA defaults OFF and is enabled with
> `-DDTWC_ENABLE_CUDA=ON`. `DTWC_ENABLE_METAL` defaults ON but is built only on
> Apple platforms; non-Apple configuration disables it. With neither backend,
> explicit GPU requests error, while an ordinary CPU `Problem` uses its selected
> CPU distance strategy.

## The DTW recurrence on a GPU

For two series $$x \in \mathbb{R}^n$$ and $$y \in \mathbb{R}^m$$, the cumulative cost matrix $$C \in \mathbb{R}^{n \times m}$$ satisfies

$$
c_{i,j} = d(x_i, y_j) + \min\bigl\{c_{i-1,j-1},\; c_{i-1,j},\; c_{i,j-1}\bigr\}
$$

with $$c_{0,0} = d(x_0, y_0)$$ and $$d(\cdot, \cdot)$$ the pointwise metric (L1 by default; squared L2 optionally). The final distance is $$c_{n-1,m-1}$$.

The sequential dependency is on the three predecessors of $$(i, j)$$. Two natural parallel decompositions exist:

1. **Anti-diagonal wavefront.** Cells on the same anti-diagonal ($$i + j = k$$) are mutually independent — once diagonal $$k-1$$ and $$k-2$$ are done, diagonal $$k$$ can be computed in parallel.
2. **Register tile.** Each thread holds a column *stripe* in registers and propagates the left-neighbour cost through the warp via `shfl_up` / `simd_shuffle_up`. No barrier is needed between cells in the same row — the shuffle is implicitly synchronised within a warp / SIMD-group.

DTWC++ uses both, plus a third row-major scheme for tight Sakoe-Chiba bands.

## Kernel dispatch tables

Each backend chooses its kernel itself; no option forces one. CUDA chooses by
the longest series length. Metal chooses by that length, the band, and the
device's threadgroup-memory cap.

### Metal — five kernels

| Condition | Kernel | Notes |
|---|---|---|
| `band > 0` and `band·20 < max_L` and `band ≤ 512` | `dtw_banded_row` | Row-major, one thread / pair, no barriers |
| `band == -1` and `max_L ≤ 128` | `dtw_regtile_w4` | Register-tile, `TILE_W=4`, `simd_shuffle_up` |
| `band == -1` and `128 < max_L ≤ 256` | `dtw_regtile_w8` | Register-tile, `TILE_W=8` |
| `3·max_L·sizeof(float)` exceeds the device cap | `dtw_wavefront_global` | Anti-diagonals in device memory |
| otherwise | `dtw_wavefront` | Anti-diagonals in threadgroup memory |

### CUDA — three kernels

| Condition | Kernel | Notes |
|---|---|---|
| `max_L ≤ 32` | `dtw_warp_kernel` | One warp per pair, full series in registers |
| `32 < max_L ≤ 256` | `dtw_regtile_kernel<TILE_W>` | `TILE_W=4` for `≤128`, `TILE_W=8` for `≤256` |
| `max_L > 2048` and three blocks of `3·max_L·sizeof(T)` bytes (plus the kernel's static and the runtime's reserved shared memory each) do not fit an SM's shared memory | `dtw_wavefront_kernel` (global) | Anti-diagonals in global memory, one slice per resident block |
| otherwise | `dtw_wavefront_kernel` | Anti-diagonals in shared memory |

On an RTX 4000 Ada the warp and register-tile kernels are the fastest of those
that accept their length ranges, FP32 and FP64 alike. Its SM (100 KB of shared
memory) holds three shared-memory wavefront blocks up to `max_L` = 2,757 in FP32
and none above 2,048 in FP64; beyond that the global-memory wavefront, which runs
up to six blocks per SM with its anti-diagonals in the L2 cache, takes 0.64–0.68
of the shared one's time at FP32 `max_L` 6,000–8,446 and 0.76–0.84 at FP64
`max_L` 2,049–4,223. It runs the same cells, so its distances are the shared
kernel's.

### Options

Both option structs inherit `band`, `use_squared_l2` and `verbose`, and take a
`GpuPrecision`; CUDA adds `device_id`. Metal's kernels are FP32:
`GpuPrecision::FP64` raises `DeviceError` at every Metal entry point.

## Historical measurements (Apple M2 Max, 38-core GPU)

These 2026-04-12 results are historical, advisory measurements from a different
machine; they are not a current release gate. The originating record is
`benchmarks/mac_metal_benchmarks.md`, with raw Google Benchmark output in
`benchmarks/results/mac_m2max/metal_vs_cpu.json`. The CPU baseline used 12
threads; the workload is unbanded DTW over random series.

| Workload | CPU (ms) | Metal (ms) | Speedup |
|---|---|---|---|
| 100 × 1000 | 1 648 | 139 | **11.9×** |
| 75 × 2500 | 5 914 | 342 | **17.3×** |
| 10 × 10 000 (global-mem path) | 3 058 | 108 | **28.3×** |
| 30 × 10 000 | 16 061 | 929 | **17.3×** |
| 75 × 10 000 | 92 500 | 5 800 | **15.9×** |

## When to pick which backend

Select a GPU explicitly through Tier 1 (`device="gpu"`) or on a `Problem`
with `set_device(Device::GPU)`. Apple unified memory reduces
transfer overhead, but the implementation still converts input into padded
Metal buffers and copies the result back to host storage.

A `Problem` computes on the CPU until `set_device` says otherwise. Tier-1
`device="gpu"` chooses a compiled GPU backend explicitly and raises if that
request cannot be delivered; there is no CUDA→Metal→CPU fallback chain.

## Citations

The algorithms and kernel shapes in this backend draw on:

- **Register-tile + warp-shuffle cost propagation:** Schmidt, B., & Hundt, C. (2020). *"cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-Enabled GPUs."* Euro-Par 2020, LNCS 12247, 597–612. Springer. https://doi.org/10.1007/978-3-030-57675-2_37. DTWC++'s CUDA kernels are inspired by cuDTW++; the Metal kernels adapt the shuffle idea with `simd_shuffle_up`.
- **Sakoe-Chiba band constraint:** Sakoe, H., & Chiba, S. (1978). *"Dynamic programming algorithm optimization for spoken word recognition."* IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43–49.

See [`.claude/CITATIONS.md`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/.claude/CITATIONS.md) for the full bibliography.

## Reference: source files

| Component | File | Notes |
|---|---|---|
| CUDA kernels | [dtwc/cuda/cuda_dtw.cu](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cu) | Warp, regtile, wavefront |
| CUDA API | [dtwc/cuda/cuda_dtw.cuh](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cuh) | `CUDADistMatOptions`, `CUDADistMatResult` |
| Metal kernels | [dtwc/metal/metal_dtw.mm](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.mm) | Wavefront × 2, banded-row, regtile × 2 |
| Metal API | [dtwc/metal/metal_dtw.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.hpp) | `MetalDistMatOptions`, `MetalDistMatResult` |
| CPU lower bounds | [dtwc/core/lower_bound_impl.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/core/lower_bound_impl.hpp) | `compute_envelope`, `lb_keogh_symmetric` |
| Dispatcher | [dtwc/Problem.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/Problem.cpp) | `fill_distance_matrix` routes through the strategy enum |
