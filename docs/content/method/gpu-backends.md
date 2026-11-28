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
raises `DeviceError` rather than changing to CPU (unless `fallback` is set). A Metal buffer allocation or
kernel that fails raises `DeviceError` too.

Both write the matrix the CPU fill writes, `DistanceMatrix`'s packed lower
triangle, in place, whether it is on the heap or memory-mapped; the result
structs report what ran (kernel, pair count, GPU time), not the distances. A
backend that refuses a request (no device, a GPU that cannot hold the fill's
buffers) does so before the matrix is allocated. Both accept series of any
length. CUDA computes at most $2^{27}$ pairs per launch and copies each launch's
share of the matrix to the host as it finishes: the GPU holds the padded series
and output staging whatever $N$ is (plus, for multi-pass or fallback kernels on long series, persistent warp borders or three anti-diagonals), and the matrix itself lives in host memory.

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
| Float32 series, owned or a view | `DeviceError` |
| Float64 series that a `Data` only views (an Arrow IPC file read in place or decoded, `set_view_data`) | CUDA reads them where they lie (its fill, FastCLARA's samples and assignment pad them from the view; no copy is made); Metal reads Float64 series directly via `SeriesAt` spans and converts/uploads to FP32 in parallel without an intermediate owned host copy |
| squared L2 (`Problem::set_metric(MetricType::SquaredL2)`, dense or mapped) | computed with `use_squared_l2` |
| Metal: precision FP64, or a GPU index other than 0 (refused by `set_device`) | `DeviceError` |
| CUDA: precision Auto, FP32 or FP64; any device index | honoured |
| a band narrower than the longest-minus-shortest series length (every device) | `InvalidInput` naming both series and the smallest feasible band |

`Problem::dtw_function()` remains CPU-only. When `Problem::cluster()`, `tadpole()`, or `one_batch_pam()` is called on a `Problem` configured for `Device::GPU` (or for `Device::CPU_GPU` with a CPU-only clustering method such as OneBatchPAM or TADPole), `DeviceScope` refuses the request with `DeviceError` unless `fallback = true` is set, in which case it falls back to the CPU and logs one notice on `stderr`.

FastCLARA's assignment runs on the GPU with CUDA: `cuda::compute_medoid_argmin_cuda`
deduplicates unique medoids across all subsamples, computes unique medoid distances on
the fill's kernels and route rule, and reduces `(best_label, best_dist)` per sample
directly on the GPU (preserving lowest-slot and self tie-breaking). Blocks move through
a two-stream host/device pipeline so one block pads and uploads while the previous
computes and assigns. A failed CUDA call raises `DeviceError` whose text names
its file, line and call (`CUDA error at <file>:<line> (<call>): <CUDA's message>`);
where the runtime refuses pinned host memory a buffer is pageable instead (slower
copies, the same results) and one line on `stderr` says so. Metal has no medoid
assignment kernel, so FastCLARA's assignment runs on the CPU.

With `Device::CPU_GPU` (`cpu+gpu`, CUDA only) the fill and FastCLARA's assignment share
their work between this GPU session and the CPU's threads, in one parallel region per
slab of the fill and one for the pass; the kernels and the CUDA entry points are the same,
the GPU's session filling row ranges of the matrix (`cuda::DistanceFill`). The rule
and its limits are on the [Devices](../../guides/devices/) page; under explicit `FP32`
the GPU writes `FP32` rows and the CPU writes `FP64` rows.

> **Compile-time flags.** CUDA defaults OFF and is enabled with
> `-DDTWC_ENABLE_CUDA=ON`. `DTWC_ENABLE_METAL` defaults ON on Apple platforms and
> OFF elsewhere. Asking for a backend the build cannot have (CUDA without `nvcc` or
> on macOS, Metal off Apple) stops the configure with an error. With neither backend,
> explicit GPU requests error, while an ordinary `Problem` computes on the CPU.

## The DTW recurrence on a GPU

For two series $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$, the cumulative cost matrix $C \in \mathbb{R}^{n \times m}$ satisfies

$$
c_{i,j} = d(x_i, y_j) + \min\bigl\{c_{i-1,j-1},\; c_{i-1,j},\; c_{i,j-1}\bigr\}
$$

with $c_{0,0} = d(x_0, y_0)$ and $d(\cdot, \cdot)$ the pointwise metric (L1 by default; squared L2 optionally). The final distance is $c_{n-1,m-1}$.

The sequential dependency is on the three predecessors of $(i, j)$. Two natural parallel decompositions exist:

1. **Warp register-tile (`warptile` / `regtile`).** Each thread holds a column *stripe* of `TILE_W` elements in registers and propagates the left-neighbour cost across the warp/SIMD-group via `__shfl_up_sync` / `simd_shuffle_up` without shared-memory barriers between cells in a row. On CUDA, persistent multi-pass tiles extend this to arbitrary series lengths.
2. **Anti-diagonal wavefront.** Cells on the same anti-diagonal ($i + j = k$) are mutually independent — once diagonals $k-1$ and $k-2$ complete, diagonal $k$ evaluates in parallel.

DTWC++ uses both, plus a coalesced row-major scheme (`dtw_banded_row`) on Metal for tight Sakoe–Chiba bands.

## Kernel dispatch tables

Each backend chooses its kernel itself (`dtwc/cuda/kernel_selection.hpp` on CUDA, host dispatcher in `dtwc/metal/metal_dtw.mm` on Metal); no option forces one.

### CUDA — unified `dtw_warptile_kernel` and wavefront fallback

On CUDA, `select_kernel` uses the unified multi-pass warp register-tiled kernel (`dtw_warptile_kernel`) by default for arbitrary series lengths (single-pass up to $32 \cdot \mathrm{TILE\_W}$ steps, persistent multi-pass above that using a warp border buffer in VRAM), falling back to the 3-buffer anti-diagonal wavefront only when pair parallelism is too low for warps on long series (`num_pairs < 8 * sm_count && max_L > 1024`) or when the persistent warp border buffer does not fit available VRAM (`!warptile_border_fits`):

| Condition | `kernel_used` | Notes |
|---|---|---|
| default, `max_L <= 32` | `warptile_w1` | Warp register-tile, `TILE_W=1` (single-pass) |
| default, `32 < max_L <= 128` or FP64 with `max_L > 32` | `warptile_w4` | Warp register-tile, `TILE_W=4` (1-pass `<= 128`, multi-pass `> 128`) |
| default, FP32 with `max_L > 128` | `warptile_w8` | Warp register-tile, `TILE_W=8` (1-pass `<= 256`, multi-pass `> 256`) |
| wavefront fallback and `3 * max_L * sample_bytes + block_overhead <= max_shared_per_block` | `wavefront` | Shared-memory 3-anti-diagonal wavefront |
| wavefront fallback otherwise | `wavefront_global` | Global-memory / L2 3-anti-diagonal wavefront |

### Metal — five kernels

Metal selects from five FP32 kernels by band ratio, longest length `max_L`, and `maxThreadgroupMemoryLength`. Before `regtile` routing, any non-negative `band` not selecting `banded_row` that covers the batch (`band >= max_L - 1`) normalises to `-1`. For `regtile`, the host computes dynamic `pairs_per_tg` (up to 8 warps bounded by both threadgroup thread count and threadgroup memory) and lane 0 broadcasts decoded 64-bit series indices as 32-bit `simd_broadcast_first` values. Chunks stay under ~5 billion cells per command buffer to avoid macOS watchdog timeouts, with double-buffered device scratch (`scratch[2]`) for multi-chunk `banded_row` and `wavefront_global` launches:

| Condition | Kernel (`kernel_used`) | Notes |
|---|---|---|
| `band > 0`, `band * 20 < max_L`, and `band <= 512` | `banded_row` | Row-major coalesced scratch, 1 thread per pair |
| `band == -1` and `max_L <= 128` | `regtile_w4` | SIMD-group register-tile, `TILE_W=4`, `simd_shuffle_up` |
| `band == -1` and `128 < max_L <= 256` | `regtile_w8` | SIMD-group register-tile, `TILE_W=8`, dynamic `pairs_per_tg` |
| `3 * max_L * sizeof(float) > maxThreadgroupMemoryLength` | `wavefront_global` | Anti-diagonals in device memory, double-buffered scratch |
| otherwise | `wavefront` | Anti-diagonals in threadgroup memory |

### Options

Both option structs inherit `band`, `use_squared_l2` and `verbose`, and take a `GpuPrecision`; CUDA adds `device_id`. Metal's kernels are FP32: `GpuPrecision::FP64` raises `DeviceError` at every Metal entry point.

## GPU Cross-Collection (`cdist` / `cdist_f32`)

- **CUDA (`Device::GPU`):** computes dense rectangular $(N_X \times N_Y)$ cross-collection distances via `compute_medoid_distances_cuda`, honouring `gpu_precision` (`Auto`, `FP32`, `FP64`).
- **CUDA Hybrid (`Device::CPU_GPU`):** `GpuPrecision::Auto` resolves to `FP64` so GPU and CPU rows share precision (`FP32` runs both sides in single precision). When `gpu_share >= 0`, `gpu_share == 0.0` runs CPU-only and positive shares split rows `[0, gpu_rows)` to GPU and `[gpu_rows, N_X)` to CPU; when `gpu_share == -1.0` (auto), GPU and OpenMP CPU threads share a dynamic two-ended work queue (`gpu_batch` in `[64, 512]` from head, `cpu_batch` in `[1, 16]` from tail).
- **Apple Metal:** cross-collection rectangular `cdist` is not implemented on Metal and raises `DeviceError` (or runs on CPU if `fallback = true` is set).

## Historical measurements (Apple M2 Max, 38-core GPU)

These results are historical, advisory measurements from a different
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
transfer overhead, while Metal converts Float64 input into shared FP32 staging
buffers in parallel and widens the packed triangle back to host precision.

A `Problem` computes on the CPU until `set_device` says otherwise. Tier-1
`device="gpu"` chooses a compiled GPU backend explicitly and raises if that
request cannot be delivered; there is no CUDA→Metal→CPU fallback chain unless
`fallback = true` is explicitly passed ([Devices](../../guides/devices/)).

## Citations

The algorithms and kernel shapes in this backend draw on:

- **Register-tile + warp-shuffle cost propagation:** Schmidt, B., & Hundt, C. (2020). *"cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-Enabled GPUs."* Euro-Par 2020, LNCS 12247, 597–612. Springer. https://doi.org/10.1007/978-3-030-57675-2_37. DTWC++'s CUDA kernels are inspired by cuDTW++; the Metal kernels adapt the shuffle idea with `simd_shuffle_up`.
- **Sakoe-Chiba band constraint:** Sakoe, H., & Chiba, S. (1978). *"Dynamic programming algorithm optimization for spoken word recognition."* IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43–49.

See [`docs/CITATIONS.md`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/CITATIONS.md) for the full bibliography.

## Reference: source files

| Component | File | Notes |
|---|---|---|
| CUDA kernel selection | [dtwc/cuda/kernel_selection.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/kernel_selection.hpp) | `select_kernel` (`warptile_w1`/`w4`/`w8`, `wavefront`, `wavefront_global`) |
| CUDA kernels | [dtwc/cuda/cuda_dtw.cu](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cu) | `dtw_warptile_kernel`, `dtw_wavefront_kernel`, `compute_medoid_argmin_cuda` |
| CUDA API | [dtwc/cuda/cuda_dtw.cuh](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cuh) | `CUDADistMatOptions`, `CUDADistMatResult`, `DistanceFill`, `compute_medoid_argmin_cuda` |
| Metal kernels | [dtwc/metal/metal_dtw.mm](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.mm) | `banded_row`, `regtile_w4`/`w8`, `wavefront`, `wavefront_global` |
| Metal API | [dtwc/metal/metal_dtw.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.hpp) | `MetalDistMatOptions`, `MetalDistMatResult` |
| Cross-collection distances | [dtwc/cdist.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cdist.cpp) | CPU SIMD lanes, CUDA rectangular, `cpu+gpu` dynamic queue |
| CPU lower bounds | [dtwc/core/lower_bound_impl.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/core/lower_bound_impl.hpp) | `compute_envelope`, `lb_keogh_symmetric`, `lb_keogh_at_least` |
| Dispatcher | [dtwc/Problem.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/Problem.cpp) | `fill_distance_matrix` routes on CPU, CUDA, Metal, or `cpu+gpu` |
