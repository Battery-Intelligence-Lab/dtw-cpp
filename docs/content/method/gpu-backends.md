---
title: GPU Backends
weight: 6
---

# GPU backends — CUDA and Metal

DTW is embarrassingly parallel across pairs but **sequential within** each pair (the DP recurrence has a diagonal dependency). DTWC++ exploits the parallelism with two GPU backends:

- **CUDA** (NVIDIA) — targets consumer and HPC discrete GPUs.
- **Metal** (Apple Silicon) — targets M-series integrated GPUs.

Both expose the same C++ option surface and the same CPU fallback on error. This page explains the kernels, how to pick between them, and how lower-bound pruning (LB_Keogh) accelerates large workloads.

> **Compile-time flags.** Backends are opt-in: `-DDTWC_ENABLE_CUDA=ON` and/or `-DDTWC_ENABLE_METAL=ON`. If neither is enabled, `Problem::fillDistanceMatrix` runs on CPU (`BruteForce` or `Pruned`).

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

The dispatcher inspects `band`, `max_L`, and the user hints in `MetalDistMatOptions` / `CUDADistMatOptions`, then picks one kernel. Auto-dispatch is a function of `(max_L, band)`:

### Metal — five kernels

| Condition | Kernel | Notes |
|---|---|---|
| `band > 0` and `band·20 < max_L` and `band ≤ 512` | `dtw_banded_row` | Row-major, one thread / pair, no barriers |
| `band == -1` and `max_L ≤ 128` | `dtw_regtile_w4` | Register-tile, `TILE_W=4`, `simd_shuffle_up` |
| `band == -1` and `128 < max_L ≤ 256` | `dtw_regtile_w8` | Register-tile, `TILE_W=8` |
| `3·max_L·4 > 32 KB` (≈ `max_L > 2730`) | `dtw_wavefront_global` | Anti-diagonals in device memory |
| otherwise | `dtw_wavefront` | Anti-diagonals in threadgroup memory |

### CUDA — three kernels (plus 1-vs-N / K-vs-N variants)

| Condition | Kernel | Notes |
|---|---|---|
| `max_L ≤ 32` | `dtw_warp_kernel` | One warp per pair, full series in registers |
| `32 < max_L ≤ 256` | `dtw_regtile_kernel<TILE_W>` | `TILE_W=4` for `≤128`, `TILE_W=8` for `≤256` |
| otherwise | `dtw_wavefront_kernel` | Anti-diagonals in shared memory |

### User hints

Both option structs accept two escape hatches for power users:

```cpp
struct MetalDistMatOptions {
  // ...
  int max_length_hint = 0;              // 0 = auto-detect
  MetalKernelOverride kernel_override = MetalKernelOverride::Auto;
};
```

- `max_length_hint > 0` skips the runtime length scan (tiny win) and lets the dispatcher commit to a kernel upfront.
- `kernel_override` forces a specific kernel. If the request is impossible for the actual data (e.g. regtile with `max_L = 500`), the dispatcher falls back to `Auto` with a verbose warning — **correctness is always preserved**.

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

Given a query $$q$$ and an envelope $$(U, L)$$ computed from a reference series, the one-directional LB_Keogh lower bound is

$$
\mathrm{LB}_{\mathrm{Keogh}}(q;\, U, L) = \sum_{i=0}^{n-1}
\begin{cases}
 (q_i - U_i)^2 & \text{if}\ q_i > U_i,\\
 (L_i - q_i)^2 & \text{if}\ q_i < L_i,\\
 0 & \text{otherwise.}
\end{cases}
$$

(For L1 metrics DTWC++ drops the square.) Intuitively: sum up the amount by which the query falls outside the envelope. If the query is entirely inside, the bound is zero. The bound is **exact** (`=` DTW) when the two series are identical, and always satisfies $$\mathrm{LB}_{\mathrm{Keogh}} \le \mathrm{DTW}$$.

DTWC++ uses the **symmetric** form — the tighter of the two single-direction bounds:

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
    ┌─────────────────────────┐   DTW kernel (wavefront / regtile)
    │ result matrix (N×N)     │   runs only on survivors via pair_indices
    └─────────────────────────┘
```

### Enabling it

The user-facing controls are unified across CUDA and Metal:

```cpp
dtwc::metal::MetalDistMatOptions opts;
opts.use_lb_keogh = true;        // master gate
opts.lb_threshold = 0.5;          // prune pair if LB > 0.5
opts.lb_envelope_band = 50;       // width of Sakoe-Chiba envelope window
```

Or via `Problem::lower_bound_strategy` (coming in a later commit), which auto-picks per-backend: CPU does the `LB_Kim + LB_Keogh + early-abandon` cascade; GPUs do `LB_Keogh` only.

| Strategy | CPU | CUDA | Metal |
|---|---|---|---|
| `Auto` | Kim + Keogh + early-abandon | Keogh only | Keogh only |
| `None` | no LB | no LB | no LB |
| `Kim` | LB_Kim only | falls back to Keogh | falls back to Keogh |
| `Keogh` | LB_Keogh only | LB_Keogh | LB_Keogh |
| `KimKeogh` | Kim → Keogh cascade | falls back to Keogh | falls back to Keogh |

Kim is O(1) per pair but much looser than Keogh; it's worth running only on CPU where the launch overhead is negligible. On GPU the Keogh kernel launch amortises across all pairs, so Kim is redundant.

## Measured speedups (Apple M2 Max, 38-core GPU)

12-thread CPU baseline vs Metal, unbanded DTW over random series:

| Workload | CPU (ms) | Metal (ms) | Speedup |
|---|---|---|---|
| 100 × 1000 | 1 648 | 139 | **11.9×** |
| 75 × 2500 | 5 914 | 342 | **17.3×** |
| 10 × 10 000 (global-mem path) | 3 058 | 108 | **28.3×** |
| 30 × 10 000 | 16 061 | 929 | **17.3×** |
| 75 × 10 000 | 92 500 | 5 800 | **15.9×** |

Register-tile vs baseline wavefront (per-cell throughput at the sizes where regtile fires, `N=100`):

| Length `L` | Wall time | Cells/ms | vs baseline wavefront @ L=500 |
|---|---|---|---|
| 64 | 0.38 ms | 54 M | 1.4× |
| 128 | 0.53 ms | 152 M | **3.9×** |
| 192 | 0.88 ms | 208 M | 5.3× |
| 256 | 1.09 ms | 298 M | **7.6×** |

LB_Keogh pruning (`N=100, L=1000`, random uniform series, `lb_envelope_band = L/10`):

| Mode | Wall time | Pairs pruned | vs baseline |
|---|---|---|---|
| Baseline wavefront (no LB) | 158 ms | 0 / 4 950 | 1× |
| LB enabled, `threshold = +∞` | 159 ms | 0 / 4 950 | +0.6% overhead |
| LB enabled, `threshold = 0.0` | **1.53 ms** | 4 950 / 4 950 | **103×** |

Strict-threshold wall time is essentially the `envelope + LB + compaction` cost; DTW never runs. Realistic workloads land between the two rows depending on data shape and threshold.

## When to pick which backend

```
┌─────────────────────────────────────────────────────────┐
│ Have NVIDIA GPU?                                        │
│  └─ Yes  ──► use CUDA.                                  │
│      └─ HPC GPU (A100/H100)? Also enable FP64.          │
│                                                         │
│ Apple Silicon (M1/M2/M3/M4)?                            │
│  └─ Yes  ──► use Metal. Unified memory removes H2D/D2H. │
│                                                         │
│ Neither?                                                │
│  └─ CPU with DistanceMatrixStrategy::Pruned + a         │
│     Sakoe-Chiba band usually wins on moderate workloads.│
└─────────────────────────────────────────────────────────┘
```

Selecting `DistanceMatrixStrategy::Auto` on a build with both backends enabled picks CUDA if a device is present; otherwise Metal; otherwise CPU.

### When LB_Keogh helps

LB_Keogh pays off when:

- **Lots of pairs** are far apart — clustering with well-separated clusters, nearest-neighbour queries where most candidates are irrelevant.
- **Long series** — per-pair envelope + LB cost is O(L), DTW is O(L²). Ratio grows linearly with `L`.
- **Tight band matters** — a narrow envelope gives a tighter lower bound, so more pairs are pruned at a given threshold.

It is a small loss when:

- **Few pairs can be pruned** (highly similar data, e.g. a single cluster). You pay the O(L) envelope + LB cost for no skip.
- **Tiny workloads** (`N < 20`, `L < 100`) — kernel-launch overhead dominates.

## Citations

The algorithms and kernel shapes in this backend draw on:

- **Register-tile + warp-shuffle cost propagation:** Schmidt, B., & Hundt, C. (2020). *"cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-Enabled GPUs."* Euro-Par 2020, LNCS 12247, 597–612. Springer. https://doi.org/10.1007/978-3-030-57675-2_37. The CUDA reference implementation in DTWC++ is a direct port; the Metal kernels translate `__shfl_sync` to `simd_shuffle_up`.
- **LB_Keogh:** Keogh, E., & Ratanamahatana, C. A. (2005). *"Exact Indexing of Dynamic Time Warping."* Knowledge and Information Systems, 7(3), 358–386.
- **Symmetric LB_Keogh:** Rakthanmanon, T. et al. (2012). *"Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping."* KDD '12.
- **Sakoe-Chiba band constraint:** Sakoe, H., & Chiba, S. (1978). *"Dynamic programming algorithm optimization for spoken word recognition."* IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43–49.
- **Tighter lower bounds (future work, not yet implemented):** Lemire, D. (2009). *"Faster retrieval with a two-pass dynamic-time-warping lower bound."* Pattern Recognition, 42(9), 2169–2180.

See [`.claude/CITATIONS.md`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/.claude/CITATIONS.md) for the full bibliography.

## Reference: source files

| Component | File | Notes |
|---|---|---|
| CUDA kernels | [dtwc/cuda/cuda_dtw.cu](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cu) | Warp, regtile, wavefront + envelope/LB/compact |
| CUDA API | [dtwc/cuda/cuda_dtw.cuh](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/cuda/cuda_dtw.cuh) | `CUDADistMatOptions`, `CUDADistMatResult` |
| Metal kernels | [dtwc/metal/metal_dtw.mm](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.mm) | Wavefront × 2, banded-row, regtile × 2, K-vs-N × 2, envelope/LB/compact |
| Metal API | [dtwc/metal/metal_dtw.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/metal/metal_dtw.hpp) | `MetalDistMatOptions`, `MetalDistMatResult` |
| CPU pruned path | [dtwc/core/pruned_distance_matrix.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/core/pruned_distance_matrix.cpp) | LB_Kim + LB_Keogh + early-abandon cascade |
| CPU lower bounds | [dtwc/core/lower_bound_impl.hpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/core/lower_bound_impl.hpp) | `compute_envelope`, `lb_keogh_symmetric` |
| Dispatcher | [dtwc/Problem.cpp](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/dtwc/Problem.cpp) | `fillDistanceMatrix` routes through the strategy enum |
