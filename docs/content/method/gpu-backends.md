---
title: GPU Backends
weight: 6
---

# GPU backends — CUDA and Metal

DTW is embarrassingly parallel across pairs but **sequential within** each pair (the DP recurrence has a diagonal dependency). DTWC++ exploits the parallelism with two GPU backends:

- **CUDA** (NVIDIA) — targets consumer and HPC discrete GPUs.
- **Metal** (Apple Silicon) — targets M-series integrated GPUs.

Both inherit a shared option/result base, then add backend-specific fields and
defaults. Through `Problem`, an unavailable or uncompiled requested backend
raises `DeviceError` rather than changing to CPU. Some lower-level operational
Metal failures still escape as `std::runtime_error` (F31), and several explicit
GPU options currently degrade without a universally visible signal (F30).

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

The two dispatchers use different inputs. CUDA auto-selection uses the scanned
actual maximum length. Metal uses the scanned length, band, a larger
`max_length_hint` when provided, and the runtime device threadgroup-memory cap.

### Metal — five kernels

| Condition | Kernel | Notes |
|---|---|---|
| `band > 0` and `band·20 < max_L` and `band ≤ 512` | `dtw_banded_row` | Row-major, one thread / pair, no barriers |
| `band == -1` and `max_L ≤ 128` | `dtw_regtile_w4` | Register-tile, `TILE_W=4`, `simd_shuffle_up` |
| `band == -1` and `128 < max_L ≤ 256` | `dtw_regtile_w8` | Register-tile, `TILE_W=8` |
| `3·heuristic_L·sizeof(float)` exceeds the device cap | `dtw_wavefront_global` | Anti-diagonals in device memory |
| otherwise | `dtw_wavefront` | Anti-diagonals in threadgroup memory |

### CUDA — three kernels (plus 1-vs-N / K-vs-N variants)

| Condition | Kernel | Notes |
|---|---|---|
| `max_L ≤ 32` | `dtw_warp_kernel` | One warp per pair, full series in registers |
| `32 < max_L ≤ 256` | `dtw_regtile_kernel<TILE_W>` | `TILE_W=4` for `≤128`, `TILE_W=8` for `≤256` |
| otherwise | `dtw_wavefront_kernel` | Anti-diagonals in shared memory |

### User hints

Both option structs inherit shared fields, including the common
`dtwc::KernelOverride` enum:

```cpp
dtwc::metal::MetalDistMatOptions metal_opts;
dtwc::cuda::CUDADistMatOptions cuda_opts;
metal_opts.kernel_override = dtwc::KernelOverride::Wavefront;
cuda_opts.kernel_override = dtwc::KernelOverride::RegTile;
```

- Actual series lengths are always scanned. Metal lets a positive hint larger
  than the scan influence its heuristic; CUDA currently ignores the hint.
- `kernel_override` requests a path. Unsupported requests silently use Auto:
  CUDA exposes `kernel_override_fell_back`, while Metal exposes no matching
  flag. That violates the explicit-option rule and is tracked as F30.
- CUDA adds `device_id`, `CUDAPrecision`, and a `-1.0` threshold-off default.
  Metal adds `MetalPrecision`, `lb_envelope_band`, and a `0.0` threshold
  default. Metal FP64 currently becomes FP32, another F30 path.

## Lower-bound pruning (LB_Keogh)

For large workloads (`N² · max_L²` DTW cells), the majority of pairs may be "obviously far apart." Computing a cheap lower bound first and skipping pairs whose lower bound already exceeds a threshold (e.g. the current best cluster-to-medoid distance) avoids the expensive DTW for those pairs.

### The Sakoe-Chiba envelope

For a series $$x$$ and Sakoe-Chiba band width $$r$$, define the upper/lower envelopes as

$$
U_i^x = \max_{|k - i| \le r} x_k, \qquad L_i^x = \min_{|k - i| \le r} x_k.
$$

Intuitively, $$[L_i^x, U_i^x]$$ is the set of values any $$y_j$$ could be warped to at time $$i$$ under the band constraint.

```
   ^ value
   │          ┌───────── U (upper envelope)
   │     ____ │  _______
   │    /    ╲│ /       ╲
   │   / x    V/  x    x ╲      ← series x
   │  /  _____╱╲___      ╲
   │ /  /      ╲   ╲_____╱
   │   L (lower envelope)
   └────────────────────────► time
       ◄─r─►
```

### LB_Keogh

For equal-length series under L1 cost, the current GPU kernels compute

$$
\mathrm{LB}_{\mathrm{Keogh}}(q;\, U, L) = \sum_{i=0}^{n-1}
\begin{cases}
 q_i - U_i & \text{if}\ q_i > U_i,\\
 L_i - q_i & \text{if}\ q_i < L_i,\\
 0 & \text{otherwise.}
\end{cases}
$$

The envelope window must cover the actual DTW warping window. Under those
conditions, the bound is no greater than L1 DTW; identical series give zero.
The current GPU implementation is not universally admissible:

- squared-L2 DTW still receives the L1 expression above (F27);
- Metal can use a narrow envelope while DTW is unbanded (F28);
- CUDA and Metal truncate unequal lengths to `min(Li,Lj)` without a validity
  proof (F29).

Until those findings close, GPU threshold pruning is supported only for
equal-length L1 series with a matching admissible envelope. Do not infer
`LB <= DTW` outside that regime.

The kernels use the symmetric form—the tighter of the two directions:

$$
\mathrm{LB}^{\mathrm{sym}}_{\mathrm{Keogh}}(x, y) = \max\bigl(\mathrm{LB}_{\mathrm{Keogh}}(x; U^y, L^y),\ \mathrm{LB}_{\mathrm{Keogh}}(y; U^x, L^x)\bigr)
$$

### GPU pipeline

```
    ┌───────────────────┐
    │ N time series     │
    └────────┬──────────┘
             │
             ▼
    ┌─────────────────────────┐   kernel 1: compute_envelopes
    │ U, L envelopes (N×max_L)│   one threadgroup / block per series
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐   kernel 2: compute_lb_keogh
    │ LB values (N·(N-1)/2)   │   one thread per pair
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐   kernel 3: compact_active_pairs
    │ active_pairs[]          │   atomic append; stamp +∞ into result
    │ active_count            │   matrix for pruned pairs
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐   supported survivor DTW kernel
    │ thresholded matrix      │   supported kernels run survivors;
    │                         │   pruned pairs remain +inf
    └─────────────────────────┘
```

This is a threshold-query result, not an exact all-pairs distance matrix.
Pairs with `LB > threshold` are represented by `+inf`; only survivors contain
DTW values. Metal currently executes this compaction only on its wavefront and
wavefront-global paths. A requested LB stage on regtile/banded-row, an LB-buffer
allocation failure, or several other explicit-option conflicts can silently
degrade (F30).

### Enabling it

The direct backend controls differ. This safe Metal example assumes all series
have equal length and uses the same band for DTW and its L1 envelope:

```cpp
dtwc::metal::MetalDistMatOptions opts;
opts.band = 50;
opts.use_squared_l2 = false;
opts.use_lb_keogh = true;
opts.lb_threshold = 0.5;       // +inf when LB > 0.5
opts.lb_envelope_band = 50;    // must cover opts.band
opts.kernel_override = dtwc::KernelOverride::Wavefront;
```

`Problem::lb_strategy()` is CPU-only today. Configure it through
`Problem::set_lb_strategy(LowerBoundStrategy)`. It controls the CPU pruned path
and includes Kim, Keogh, Enhanced, Webb, and cascade selections; `Problem` does
not copy it into CUDA/Metal options or automatically enable GPU LB pruning.

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

Select a GPU explicitly through Tier 1 (`device="gpu"`) or set
`Problem::distance_strategy` to CUDA/Metal. Apple unified memory reduces
transfer overhead, but the implementation still converts input into padded
Metal buffers and copies the result back to host storage.

`DistanceMatrixStrategy::Auto` is CPU-only: it resolves to CPU BruteForce or
Pruned from the CPU policy. Tier-1 `device="gpu"` chooses a compiled GPU backend
explicitly and raises if that request cannot be delivered; it does not use Auto
as a CUDA→Metal→CPU fallback chain. For an exact CPU matrix, the LB
early-abandon/recompute `Pruned` route is a known pessimisation; use the ordinary
CPU path unless a thresholded consumer such as TADPole can actually skip pairs.

### When LB_Keogh helps

For `N` equal-length series of length `L` and envelope radius `r`, the current
GPU envelope kernels scan `O(r)` values at each position:

- envelope preprocessing: `O(N·L·r)`;
- all pairwise lower bounds: `O(N²·L)`;
- full DTW: paid only for the threshold survivors.

This helps only when the caller wants threshold semantics and enough pairs can
be discarded to repay preprocessing/launch work. A narrower envelope may be
tighter, but it is valid only when it still covers the actual DTW window. No
machine-independent crossover in `N` or `L` is currently registered.

## Citations

The algorithms and kernel shapes in this backend draw on:

- **Register-tile + warp-shuffle cost propagation:** Schmidt, B., & Hundt, C. (2020). *"cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-Enabled GPUs."* Euro-Par 2020, LNCS 12247, 597–612. Springer. https://doi.org/10.1007/978-3-030-57675-2_37. DTWC++'s CUDA kernels are inspired by cuDTW++; the Metal kernels adapt the shuffle idea with `simd_shuffle_up`.
- **LB_Keogh:** Keogh, E., & Ratanamahatana, C. A. (2005). *"Exact Indexing of Dynamic Time Warping."* Knowledge and Information Systems, 7(3), 358–386.
- **Symmetric LB_Keogh:** Rakthanmanon, T. et al. (2012). *"Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping."* KDD '12.
- **Sakoe-Chiba band constraint:** Sakoe, H., & Chiba, S. (1978). *"Dynamic programming algorithm optimization for spoken word recognition."* IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43–49.
- **LB_Improved provenance:** Lemire, D. (2009). *"Faster retrieval with a two-pass dynamic-time-warping lower bound."* Pattern Recognition, 42(9), 2169–2180. This GPU path implements LB_Keogh only; other live lower-bound strategies are CPU-side.

See [`.claude/CITATIONS.md`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/.claude/CITATIONS.md) for the full bibliography.

## Reference: source files

| Component | File | Notes |
|---|---|---|
| CUDA kernels | [dtwc/cuda/cuda_dtw.cu](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cu) | Warp, regtile, wavefront + envelope/LB/compact |
| CUDA API | [dtwc/cuda/cuda_dtw.cuh](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cuh) | `CUDADistMatOptions`, `CUDADistMatResult` |
| Metal kernels | [dtwc/metal/metal_dtw.mm](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.mm) | Wavefront × 2, banded-row, regtile × 2, K-vs-N × 2, envelope/LB/compact |
| Metal API | [dtwc/metal/metal_dtw.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.hpp) | `MetalDistMatOptions`, `MetalDistMatResult` |
| CPU pruned path | [dtwc/core/pruned_distance_matrix.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/core/pruned_distance_matrix.cpp) | CPU lower-bound/EAP implementation; exact-matrix abandoned pairs are recomputed |
| CPU lower bounds | [dtwc/core/lower_bound_impl.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/core/lower_bound_impl.hpp) | `compute_envelope`, `lb_keogh_symmetric` |
| Dispatcher | [dtwc/Problem.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/Problem.cpp) | `fill_distance_matrix` routes through the strategy enum |
