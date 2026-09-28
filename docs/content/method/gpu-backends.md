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
raises `DeviceError` rather than changing to CPU. A Metal buffer allocation or
kernel that fails raises `DeviceError` too, and some explicit GPU options — a
kernel override or an LB stage the chosen kernel lacks — currently degrade
without a universally visible signal (F30).

## What a `Problem` can run on a GPU

`Problem::set_device(Device::GPU)` (Python `Problem(device="gpu")`, MATLAB
`dtwc.Problem(name, 'Device', 'gpu')`) selects the build's backend, CUDA else
Metal; `DistanceMatrixStrategy::CUDA` and `Metal` are the same request spelled
per backend. Before any pair is computed, one validator checks the request
against what the kernels implement — Standard DTW on univariate Float64 series
held in RAM, L1 or squared L2, `MissingStrategy::Error` — and anything else
raises `DeviceError` naming the setting and its value:

| Request | GPU result |
|---|---|
| a variant other than Standard, a missing-data strategy, `ndim > 1` | `DeviceError` |
| Float32, mmap-backed or view-mode series | `DeviceError` |
| squared L2 (`Problem::set_metric(MetricType::SquaredL2)`, dense or mapped) | computed with `use_squared_l2` |
| Metal: precision FP64, or a GPU index other than 0 | `DeviceError` |
| CUDA: precision Auto, FP32 or FP64; any device index | honoured |
| a band narrower than the longest-minus-shortest series length (every device) | `InvalidInput` naming both series and the smallest feasible band |

The same validator runs where the lazy `dist_by_ind` path starts. Known gap:
that path, like the matrix-free schedules, still computes on the CPU when the
device is a GPU, and OneBatchPAM and FastCLARA's assignment compute through
`Problem::dtw_function()`, which the validator does not see.

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
  default. Metal's kernels are FP32: `MetalPrecision::FP64` raises
  `DeviceError` at every Metal entry point.

## Lower-bound pruning (LB_Keogh)

The integrated CUDA/Metal distance-matrix routes can produce a threshold-query
matrix: compute a cheap lower bound first, retain pairs whose bound is at most
the requested threshold, and avoid DTW for the rest. This is not an exact
all-pairs matrix. Its safety depends on the lower bound satisfying the precise
window and metric contract below.

### The Sakoe-Chiba envelope

For a candidate series $$y=(y_0,\ldots,y_{m-1})$$ and envelope radius $$r$$,
define the clipped upper/lower envelopes for an included query row $$i$$ as

$$
U_i^y = \max_{\substack{0\le j<m\\|j-i|\le r}} y_j,
\qquad
L_i^y = \min_{\substack{0\le j<m\\|j-i|\le r}} y_j.
$$

The interval $$[L_i^y,U_i^y]$$ contains every candidate value that a fixed
window of radius at most $$r$$ can align with query row $$i$$. A wider
envelope can only lower or preserve the resulting bound.

Full DTW requires the global envelope: repeat the candidate's global minimum
and maximum at every included query row. For equal lengths, or for the first
$$\min(n,m)$$ rows used by the GPU prefix, radius $$m-1$$ covers a candidate of
length $$m$$; the CPU candidate-envelope helper's explicit global fast path
also accepts radius $$m$$. A caller that evaluates later rows of a longer query
must materialise the global extrema for those rows rather than assume
$$m-1$$ reaches them. A negative radius is a full-envelope request: the CPU
helper and both GPU envelope kernels build the global envelope for it (FX-13;
the CPU helper used to coerce it to radius zero).

```
   ^ value
   │          ┌───────── U (upper envelope)
   │     ____ │  _______
   │    /    ╲│ /       ╲
   │   / y    V/  y    y ╲      ← series y
   │  /  _____╱╲___      ╲
   │ /  /      ╲   ╲_____╱
   │   L (lower envelope)
   └────────────────────────► time
       ◄─r─►
```

### LB_Keogh

Let the projection excess be

$$
\delta(q_i;L_i,U_i)=
\begin{cases}
 q_i - U_i & \text{if}\ q_i > U_i,\\
 L_i - q_i & \text{if}\ q_i < L_i,\\
 0 & \text{otherwise.}
\end{cases}
$$

For point cost $$|a-b|^p$$, with $$p=1$$ for L1 and $$p=2$$ for unrooted
squared L2, the directional prefix bound is

$$
\mathrm{LB}^{(p)}_{\mathrm{Keogh}}(q;\,U,L)
=\sum_{i=0}^{k-1}\delta(q_i;L_i,U_i)^p.
$$

For fixed-window DTW, envelope coverage—not equality—is the admissibility
condition: `r >= w`. Every feasible path visits every included query row, and
the envelope excess at that row is no larger than one distinct path-cell cost.
Therefore the directional bound is no larger than every path and hence no
larger than DTW. The reverse direction is admissible by the same argument, so
their maximum is admissible; their sum need not be.

For feasible unequal lengths (`w >= |n-m|`), summing only the first
`min(n,m)` query rows is still admissible. Truncation merely omits nonnegative
terms. This result assumes the current fixed geometry `|i-j| <= w`; it does not
automatically transfer to a slope-scaled or data-dependent corridor.

The L1 expression has amplitude units $$U$$. The squared expression must
square each excess and has units $$U^2$$, matching DTWC++'s unrooted
squared-L2 objective.

Keogh and Ratanamahatana's proposition treats same-length series with squared
costs and a final square root. The L1, unrooted-squared, symmetric-maximum, and
unequal-prefix extensions used here are derived in DTWC++'s
[D2 derivation](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/derivations/02-envelopes-lb-keogh.md).

The GPU implementation follows the same contract since FX-13 (2026-09-23):

- **F27 (fixed):** CUDA and Metal square each excess when their DTW kernel
  uses squared local costs, so the bound has the units of the distance.
- **F28 (fixed):** Metal's default envelope is the DTW window, `band` or the
  whole series for full DTW. An explicit `lb_envelope_band` narrower than that
  window raises `InvalidInput`; a wider one is accepted (a looser bound).
- **F29:** the `min(Li,Lj)` prefix now has the fixed-window proof above.
  F29 remains an executable CUDA/Metal conformance gate, not a mathematical
  repair request.
- **F50 (fixed):** both device envelope kernels clamp the radius to the series
  length, so an `INT_MAX` band builds the global envelope instead of
  overflowing the signed `k+w+1`.

Before FX-13 the conservative exercised configuration was equal-length L1 data
with a nonnegative fixed DTW band and an envelope radius covering that band.
FX-13's property test adds squared L2, full DTW, and feasible unequal lengths
on a real Metal device; its CUDA twin awaits the RTX machine. Do not infer
`LB <= DTW` from shape alone; metric, coverage, finiteness, and device
arithmetic are all part of the contract.

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
    │ active_pairs[]          │   atomic append; stamp NaN into result
    │ active_count            │   matrix for pruned pairs
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐   supported survivor DTW kernel
    │ thresholded matrix      │   supported kernels run survivors;
    │                         │   pruned pairs stay NaN (not computed)
    └─────────────────────────┘
```

This is a threshold-query result, not an exact all-pairs distance matrix.
Only when the bound is admissible does `LB > threshold` certify that the
omitted DTW value is also above the threshold. Such pairs are NaN, the only
"not a distance" value outside a kernel; only survivors contain
selected-precision DTW results. A survivor with no warping path under the band
is the finite no-path sentinel `numeric_limits<double>::max()`,
not IEEE infinity. Metal currently executes this
compaction only on its wavefront and wavefront-global paths. A requested LB stage on
regtile/banded-row, an LB-buffer allocation failure, or several other
explicit-option conflicts can silently degrade (F30).

The proof is in exact arithmetic. A universal last-ULP guard for a floating
bound and DTW accumulated in different orders remains open under D17; no
near-threshold bit-level guarantee is implied here.

### Enabling it

The direct backend controls differ. This exact-arithmetic-admissible Metal
example assumes all series have equal length and uses the same band for DTW
and its L1 envelope:

```cpp
dtwc::metal::MetalDistMatOptions opts;
opts.band = 50;
opts.use_squared_l2 = false;
opts.use_lb_keogh = true;
opts.lb_threshold = 0.5;       // NaN when LB > 0.5
opts.lb_envelope_band = 50;    // must cover opts.band (else InvalidInput)
opts.kernel_override = dtwc::KernelOverride::Wavefront;
```

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

`DistanceMatrixStrategy::Auto` is CPU-only: it resolves to the CPU brute-force
fill. Tier-1 `device="gpu"` chooses a compiled GPU backend
explicitly and raises if that request cannot be delivered; it does not use Auto
as a CUDA→Metal→CPU fallback chain.

### When LB_Keogh helps

For `N` equal-length series of length `L` and envelope radius `r`, the current
GPU envelope kernels scan at most `min(L, 2r+1)` values at each position:

- envelope preprocessing: `Θ(N·L·min(L, 2r+1))`;
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
- **Two envelope directions in a search cascade:** Rakthanmanon, T. et al. (2012). *"Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping."* KDD '12. The admissibility of taking their maximum is proved directly in the D2 derivation above rather than attributed to that paper.
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
| CPU lower bounds | [dtwc/core/lower_bound_impl.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/core/lower_bound_impl.hpp) | `compute_envelope`, `lb_keogh_symmetric` |
| Dispatcher | [dtwc/Problem.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/Problem.cpp) | `fill_distance_matrix` routes through the strategy enum |
