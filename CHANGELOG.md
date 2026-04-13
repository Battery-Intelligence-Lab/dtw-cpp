# Changelog {#changelog}

[TOC]

This changelog contains a non-exhaustive list of new features and notable bug-fixes (not all bug-fixes will be listed).


<br/><br/>
# Unreleased

### Changed (warping header family — unified DTW kernel, Phase 1 + Phase 2)

The Standard / ADTW / WDTW / DDTW + ZeroCost-missing paths now share **one** templated DTW kernel instead of per-variant copy-paste loops. Three axes of variation — pointwise cost, cell recurrence, and window shape — each become a policy; the banded/linear/full loop bodies live in exactly one place.

- **New `dtwc::core::dtw_kernel_{full,linear,banded}<T, Cost, Cell>`** in [dtwc/core/dtw_kernel.hpp](dtwc/core/dtw_kernel.hpp). All variants dispatch through these. `Cell` policies: `StandardCell` (min of 3 + cost), `ADTWCell<T>{penalty}` (penalty on horizontal/vertical steps).
- **New `dtwc::core::Span*Cost<T>`** cost functors in [dtwc/core/dtw_cost.hpp](dtwc/core/dtw_cost.hpp): `SpanL1Cost`, `SpanSquaredL2Cost`, `SpanWeightedL1Cost` (for WDTW), `SpanNanAwareL1Cost` / `SpanNanAwareSquaredL2Cost` (for ZeroCost missing), plus their multivariate counterparts. `dispatch_metric()` / `dispatch_mv_metric()` also live here now.
- [dtwc/warping.hpp](dtwc/warping.hpp), [warping_adtw.hpp](dtwc/warping_adtw.hpp), [warping_wdtw.hpp](dtwc/warping_wdtw.hpp), [warping_missing.hpp](dtwc/warping_missing.hpp) are now thin public-API wrappers that build a Cost + Cell and call the shared kernel. Touched-file LOC: 2,103 → 1,651 (−452 LOC); the banded loop previously duplicated in 5 files now exists once.
- [warping_ddtw.hpp](dtwc/warping_ddtw.hpp) unchanged — it already delegated to `dtwBanded` after derivative preprocessing.

### Fixed (silent dispatch / fallback bugs)

- **`dtwc::core::dtw_runtime()`** now honours `opts.variant_params.variant`. Previously [dtwc/core/dtw.cpp](dtwc/core/dtw.cpp) ignored the variant field and always ran Standard DTW, silently dropping ADTW / WDTW / DDTW requests from the simple binding entry point.
- **`adtwBanded_mv` / `wdtwBanded_mv` / `dtwMissing_banded_mv`** now use a real banded MV kernel instead of falling back to the unbanded MV path. Previous TODO comments acknowledged this was a "for now" — the unified kernel promotes banded MV to a first-class path across all variants. Regression tests in [unit_test_mv_variants.cpp](tests/unit/unit_test_mv_variants.cpp) and [unit_test_mv_missing.cpp](tests/unit/unit_test_mv_missing.cpp) confirm tight bands now produce strictly different (tighter) results than unbanded.

### Changed (dispatch unification — Phase 3 part 1)

`Problem::rebind_dtw_fn` previously held a ~130-line nested switch that dispatched on `{missing_strategy, variant, ndim}` with UV/MV forks duplicated across every variant case. Collapsed to two lines that call a new templated resolver:

- **New `dtwc::core::resolve_dtw_fn<T>(const Problem&)`** in [dtwc/core/dtw_dispatch.hpp](dtwc/core/dtw_dispatch.hpp) / [dtw_dispatch.cpp](dtwc/core/dtw_dispatch.cpp). Explicit instantiations for `T = data_t` (f64) and `T = float` (f32). Resolution happens once at rebind time; the returned `std::function` inner body is the existing templated kernel call — zero per-cell dispatch overhead. Benchmarked: 0% change on `BM_dtwBanded/1000/50`, `BM_wdtwBanded_g/1000/50` vs pre-refactor.
- Adding a new DTW variant now localises to `resolve_dtw_fn` — one `make_*<T>` helper + one switch arm, instead of two touch points (public wrapper + rebind case) with UV/MV duplication.
- New public `Problem::wdtw_weights_cache()` const accessor for the resolver to read the WDTW weights cache without friend machinery.

### Fixed (silent dispatch — f32 path)

- **`Problem::dtw_function_f32()`** now honours `variant_params.variant` and `missing_strategy`. Previously the float32 distance function was unconditionally bound to Standard DTW with `MissingStrategy::Error`, regardless of Problem configuration. The primary user of this path is `fast_clara`'s chunked-Parquet assignment (memory-saving f32 loader): anyone clustering with `variant = WDTW/ADTW/DDTW/SoftDTW` or with missing-data strategies and using the chunked path silently got Standard DTW results. Regression tests in [unit_test_mv_variants.cpp](tests/unit/unit_test_mv_variants.cpp) `[f32]` tag exercise ADTW, WDTW, and ZeroCost-missing via `dtw_function_f32()`.

### Changed (AROW via unified kernel — Phase 3 part 2)

The `Problem`-level AROW missing-data path now runs on `dtw_kernel_banded` with two new policies, completing the kernel-family unification for banded AROW:

- **`AROWCell`** in [dtwc/core/dtw_kernel.hpp](dtwc/core/dtw_kernel.hpp): interprets a NaN cost as the "missing pair" sentinel and carries the diagonal predecessor without adding cost. Boundary-aware fallback (uses `up` or `left` when `diag` is out-of-bounds).
- **`SpanAROWL1Cost<T>`** / **`SpanAROWSquaredL2Cost<T>`** in [dtwc/core/dtw_cost.hpp](dtwc/core/dtw_cost.hpp): return NaN when either operand is missing — signal to `AROWCell` without widening the Cell contract with an extra bool.
- **Cell contract extended with `seed(cost, i, j)`**: seed value at DP cell (0,0). Default (`StandardCell`/`ADTWCell`) returns `cost` unchanged. `AROWCell` returns 0 when cost is NaN, preventing NaN propagation through the DP when both (0,0) operands are missing.
- **Cross-validated** bit-for-bit against the legacy `dtwAROW_banded` on {no-NaN, interior-NaN, leading-NaN, trailing-NaN, all-NaN} × bands {1,2,3,4}. See [unit_test_arow_dtw.cpp](tests/unit/unit_test_arow_dtw.cpp) `[phase3]` tag.
- **Perf**: `BM_dtwBanded/1000/50` 146 μs → 145 μs (−0.7%); `seed()` method is inlined to a direct assignment for non-AROW cells — zero overhead on Standard/ADTW/WDTW/DDTW hot paths.
- `warping_missing_arow.hpp` was retained unchanged as the standalone public API surface; it has now been folded into the unified kernel (see Phase 4 entry below).

### Changed (Soft-DTW via unified kernel — Phase 3 part 3)

Problem-level SoftDTW dispatch now runs on `dtw_kernel_full<T, SpanL1Cost<T>, SoftCell<T>>` instead of the standalone `soft_dtw()` function.

- **`SoftCell<T>{gamma}`** in [dtwc/core/dtw_kernel.hpp](dtwc/core/dtw_kernel.hpp): log-sum-exp softmin with max-subtract stabilisation. Sentinel-aware — out-of-bounds predecessors (`maxValue`) are excluded from the LSE accumulator, so first-row/column cells where only one predecessor is valid reduce automatically to `predecessor + cost` (hard accumulation), matching the legacy `soft_dtw()` boundary treatment.
- **Cross-validated** bit-for-bit against `soft_dtw()` on equal-length, different-length, identical, and swap-symmetric inputs across gamma {0.1..10.0}. See [unit_test_soft_dtw.cpp](tests/unit/unit_test_soft_dtw.cpp) `[phase3]` tag. 4 test cases, 14 assertions, all passing within 1e-10 tolerance.
- `soft_dtw.hpp` retained unchanged as the standalone public API surface (and for `soft_dtw_gradient()`, which still needs its own forward+backward matrices).

### Added (Multivariate AROW — Phase 3 part 4)

Problem-level AROW dispatch now has a first-class MV branch; previously ndim > 1 was silently flattened to scalar AROW (documented TODO from pre-refactor).

- **`SpanMVAROWL1Cost<T>`** / **`SpanMVAROWSquaredL2Cost<T>`** in [dtwc/core/dtw_cost.hpp](dtwc/core/dtw_cost.hpp): per-channel skip for cost (summing `|x[d] - y[d]|` over comparable channels). Returns NaN only when a pair has *no* comparable channels — every channel has at least one NaN operand. Triggers `AROWCell`'s diagonal carry only when the whole pair is uninformative.
- **Design note**: the direct scalar→MV lift ("any channel missing → diagonal carry") would discard usable per-channel data. The per-channel-skip semantics here preserves information and is consistent with `SpanMVNanAwareL1Cost` (ZeroCost MV). Reduces exactly to scalar AROW when `ndim = 1`.
- **Tests** in [unit_test_arow_dtw.cpp](tests/unit/unit_test_arow_dtw.cpp) `[mv][phase3]` tag: ndim=1 parity with `dtwAROW_banded` across bands {1..4}, identical MV zero-distance, per-channel-skip correctness, fully-missing-step triggers diagonal carry.

### Phase 3 complete

All four deferred items from Phase 2 are now implemented:

| Part | Deliverable | Commits |
|------|-------------|---------|
| 3.1 | `rebind_dtw_fn` templated resolver + f32 silent-variant bug fix | `4d92881`, `cbdb942` |
| 3.2 | AROW fold via `AROWCell` + `SpanAROWL1Cost` + Cell `seed()` | `d595035` |
| 3.3 | Soft-DTW fold via `SoftCell` (log-sum-exp + stabilisation) | `ee0f798` |
| 3.4 | MV AROW first-class path via `SpanMVAROWL1Cost` | _(this)_ |

Kernel family now handles Standard / ADTW / WDTW / DDTW / Soft-DTW / AROW / ZeroCost-missing / Interpolate-missing with one templated core and orthogonal Cost + Cell policies. Adding a new variant = one Cost policy + one Cell policy + one switch arm in `resolve_dtw_fn`.

### Changed (standalone AROW API folded — Phase 4 cleanup)

The standalone [dtwc/warping_missing_arow.hpp](dtwc/warping_missing_arow.hpp) public API (`dtwAROW`, `dtwAROW_L`, `dtwAROW_banded`) now delegates to the unified DTW kernel. The hand-rolled `detail::dtwAROW_*_impl` helpers (full / linear / banded × span / pointer overloads, ~260 LOC) are removed — each wrapper builds a `SpanAROW{L1,SquaredL2}Cost` functor and calls `core::dtw_kernel_{full,linear,banded}<T, Cost, AROWCell>`.

- **Behaviour**: identical within the test suite (AROW tests, adversarial tests, MV tests, Problem-level AROW tests — 70/70 ctest passing). The legacy banded impl used a simpler `std::ceil/std::floor` band-bounds calculation and the unified kernel uses the `round-100` variant from [dtw_kernel.hpp:157-163](dtwc/core/dtw_kernel.hpp#L157-L163); the legacy's own comment (now removed) called out that the difference is harmless because out-of-band cells are sentinel-valued and never selected.
- **No API break**: all span / vector / pointer+size overloads retained with identical signatures.
- **No perf regression**: `dtw_kernel_banded` is the same kernel already exercised on the Problem-level AROW hot path via `resolve_dtw_fn` — the standalone wrapper now joins it. `BM_dtwBanded/1000/50` unchanged (145 μs).
- Wrapper header shrinks from 459 LOC to 214 LOC.

### Added (GPU parity + configuration)

- **`MetalDistMatResult::lb_time_sec`**: Metal now reports LB_Keogh pre-pass time separately from total GPU time, matching the CUDA field of the same name.
- **`dtwc::metal::compute_lb_keogh_metal(series, band)`**: standalone LB_Keogh lower bounds over all `N*(N-1)/2` pairs on the default Metal device. Mirrors `dtwc::cuda::compute_lb_keogh_cuda` (envelope + symmetric pairwise LB, no DTW dispatch). Returns `MetalLBResult{lb_values, n, gpu_time_sec}`.
- **`dtwc::LowerBoundStrategy` enum** (`Auto`/`None`/`Kim`/`Keogh`/`KimKeogh`) and `Problem::lb_strategy` field. Controls which lower bound(s) feed the Pruned CPU path. `Auto` keeps the historical Kim+Keogh cascade; `None` short-circuits to BruteForce.
- **`dtwc::KernelOverride` enum** (`Auto`/`Wavefront`/`WavefrontGlobal`/`BandedRow`/`RegTile`) and `max_length_hint` field on both `CUDADistMatOptions` and `MetalDistMatOptions`. Lets advanced users force a kernel path or hint the expected max series length for the selector. Unsupported overrides silently fall back to `Auto`. Metal wires the hint into its kernel-selection heuristics; CUDA stores the fields for API uniformity.

### Changed (GPU backend refactor — non-breaking)

- **Base structs `dtwc::gpu::DistMatOptionsBase` / `DistMatResultBase`** in [core/gpu_dtw_common.hpp](dtwc/core/gpu_dtw_common.hpp). `CUDADistMatOptions`, `MetalDistMatOptions`, `CUDADistMatResult`, `MetalDistMatResult` now inherit common fields (`band`, `verbose`, `use_lb_keogh`, `max_length_hint`, `kernel_override`, `matrix`, `n`, `gpu_time_sec`, `lb_time_sec`, `pairs_computed`, `pairs_pruned`, `kernel_used`). `lb_threshold` stays per-backend: CUDA default `-1.0` (sentinel), Metal default `0.0` (applied). Designated aggregate init preserved.
- **`dispatch_gpu_backend` lambda in `Problem::fillDistanceMatrix`**: collapses the parallel CUDA/Metal case blocks (result → matrix copy, verbose log, empty-result fallback) into one generic post-processor parameterised by backend result type.

### Changed (naming unification — breaking, pre-v2.0.0)

Unified option/field names across CPU, CUDA, and Metal backends. Hard renames (no compat aliases):

| Old | New | Scope |
|---|---|---|
| `DEFAULT_BAND_LENGTH` | `DEFAULT_BAND` | [settings.hpp](dtwc/settings.hpp) + 32 call sites |
| `DTWOptions::band_width` | `DTWOptions::band` | [dtw_options.hpp](dtwc/core/dtw_options.hpp) + 4 call sites |
| `CUDADistMatOptions::use_lb_pruning` | `use_lb_keogh` | CUDA — matches Metal, CPU naming |
| `CUDADistMatOptions::skip_threshold` | `lb_threshold` | CUDA |
| `MetalDistMatOptions::enable_lb_keogh` | `use_lb_keogh` | Metal — matches CUDA |
| `DistanceMatrixStrategy::GPU` | `::CUDA` | `::GPU` was ambiguous once `::Metal` landed. Enum / CLI `--device` / Python `DistanceMatrixStrategy.CUDA` / MATLAB `"cuda"` string |
| `CUDASettings::precision_mode` (int 0/1/2) | `precision` | Kept as int for header independence from `DTWC_HAS_CUDA` |

MATLAB strategy string `'gpu'` is no longer accepted — pass `'cuda'` instead.

Motivation: three different spellings for LB_Keogh (`use_lb_pruning`, `enable_lb_keogh`, implicit-via-`Pruned`-strategy) and two for the band (`band`, `band_width`). With two GPU backends now shipped, the drift would only compound.

Cited cuDTW++ (Schmidt & Hundt 2020) and LB_Keogh (Keogh & Ratanamahatana 2005) in [metal_dtw.hpp](dtwc/metal/metal_dtw.hpp) / [metal_dtw.mm](dtwc/metal/metal_dtw.mm) headers and in the Register-tile / LB_Keogh CHANGELOG entries below — these were missing when the kernels first landed.

### Added (Python bindings)

- `compute_distance_matrix_cuda()` now accepts `use_lb_keogh` and `lb_threshold` kwargs (previously CUDA pruning was exposed in C++ only). Default `use_lb_keogh=False` preserves existing behavior. Semantics match the C++ surface: pair pruned if `LB_Keogh > lb_threshold` and `lb_threshold > 0`.

### Added (Documentation)

- New **GPU Backends** page in the docs site ([docs/content/method/gpu-backends.md](docs/content/method/gpu-backends.md)): kernel dispatch tables for both CUDA (3 kernels) and Metal (5 kernels), Sakoe-Chiba envelope definition with ASCII diagram, `LB_Keogh` equation (one-directional + symmetric), full GPU LB_Keogh pipeline flowchart, measured speedups on Apple M2 Max, when-to-use-which decision tree, and citations to Schmidt & Hundt 2020, Keogh & Ratanamahatana 2005, Rakthanmanon 2012, Sakoe & Chiba 1978, and Lemire 2009 (future work).

### Added (Metal GPU backend — Apple Silicon)

- **New `dtwc::metal` backend**: anti-diagonal wavefront DTW kernel in MSL (Metal Shading Language), compiled at runtime via `newLibraryWithSource:`. Pairwise distance matrix on the Apple GPU, one threadgroup per pair, 3 rotating threadgroup-memory buffers for anti-diagonals.
- **CMake**: `DTWC_ENABLE_METAL` option (default ON on APPLE, silently disabled elsewhere). Adds `OBJCXX` language, links `Foundation` + `Metal` frameworks, defines `DTWC_HAS_METAL`.
- **`DistanceMatrixStrategy::Metal`** dispatch case in `Problem::fillDistanceMatrix`; graceful CPU fallback when Metal is unavailable or when `max_L` exceeds the threadgroup-memory cap (32 KB on M1/M2/M3, equivalent to ~2730 FP32 elements).
- **Python binding**: `DistanceMatrixStrategy.Metal`, `metal_available()`, `metal_device_info()`, `compute_distance_matrix_metal()`, `METAL_AVAILABLE` attribute. Integrated into `system_info()`. Measured Python wrapper overhead vs native C++ `compute_distance_matrix_metal`: **−0.5% to +0.9%** (well under 10% target).
- **MATLAB binding**: `"metal"` distance strategy accepted by `set_distance_strategy`; `system_check` struct now includes `metal` + `metal_info` fields; `dtwc.check_system()` reports Metal status. Measured MATLAB wrapper overhead vs Python (both using `Problem::fillDistanceMatrix`): **≤1%** across all 6 workloads. (Initial apples-to-oranges comparison showed 8–17%, but the gap turned out to be `Problem::fillDistanceMatrix` vs direct `compute_distance_matrix_metal` — same gap appears in Python too. Language-wrapper overhead itself is sub-1%.)
- **MATLAB R2024b linker fix**: `CMakeModules/FindMatlab.cmake` (shipped with CMake) unconditionally adds `cppMexFunction.map` to the exported-symbols list on macOS, which requires `_mexCreateMexFunction` / `_mexDestroyMexFunction` / `_mexFunctionAdapter` symbols we don't provide (we use the legacy C `mexFunction` API). [bindings/matlab/CMakeLists.txt](bindings/matlab/CMakeLists.txt) now clears `Matlab_HAS_CPP_API` before `matlab_add_mex()` so only `c_exportsmexfileversion.map` is linked.
- **Measured on Apple M2 Max (38-core GPU)** vs the 12-thread CPU path:

  | Workload | CPU (ms) | Metal (ms) | Speedup |
  |---|---|---|---|
  | 50 series × 500 length | 106 | 6.9 | 15.4× |
  | 100 × 500 | 404 | 26.2 | 15.4× |
  | 100 × 1000 | 1648 | 139 | 11.9× |
  | 200 × 500 | 1626 | 103 | 15.8× |
  | 50 × 2500 | n/a | 151 | — |
  | 10 × 10000 (long series, global-mem kernel) | 3058 | 108 | **28.3×** |
  | 30 × 10000 | 16061 | 929 | **17.3×** |

  Throughput: ~30–52 × 10⁹ DTW cells/sec; ~180–320 GFLOPS (≈1.3–2.4% of 13.6 TFLOPS FP32 peak). The low FLOP fraction is expected for DTW's memory-bandwidth-bound DP recurrence; further gains require register-tiling and warp-shuffle kernels (follow-on).
- **Three kernel variants share the same API:**
  - `dtw_wavefront` (threadgroup memory): 3 × max_L floats in 32 KB threadgroup memory → max_L ≤ 2730 on M1/M2/M3.
  - `dtw_wavefront_global` (device memory): scratch lives in unified GPU memory; any max_L supported.
  - `dtw_banded_row` (tight-band row-major): one thread per pair, no intra-threadgroup barriers, coalesced device-memory scratch with register-rotated prev/cur window. Fires when `band > 0 AND band * 20 < max_L AND band ≤ 512` — the regime where anti-diagonal barrier overhead dominates the wavefront kernel. On 75 × 10000 band=100: **1.01 s vs 1.40 s for wavefront (1.4×) and 1.76 s for CPU (1.7×)**. Dispatcher picks automatically; `MetalDistMatResult::kernel_used` reports which kernel ran.
- **macOS GPU-watchdog avoidance**: long dispatches (>~2 s per command buffer) fail with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. The dispatcher now chunks pairs across multiple command buffers (budget ≈ 5 × 10⁹ cells per buffer) so arbitrary N × L workloads complete without triggering the watchdog.
- **LB_Keogh pruning path (Metal)**: opt-in lower-bound prune pipeline mirroring the CUDA envelope/LB/compaction kernels (`dtwc/cuda/cuda_dtw.cu:785-910`). Algorithm from Keogh & Ratanamahatana (2005) *"Exact Indexing of Dynamic Time Warping"* (KAIS 7(3), 358–386); symmetric `max(LB(j|env_i), LB(i|env_j))` form per Rakthanmanon et al. (2012). GPU orchestration pattern (envelope → LB → atomic compaction → DTW on survivors) follows the CUDA reference which itself derives from cuDTW++ (Schmidt & Hundt 2020). Three new MSL kernels:
  - `compute_envelopes` — sliding min/max window per series (one threadgroup per series).
  - `compute_lb_keogh` — symmetric `max(LB(j, env_i), LB(i, env_j))` per pair (one thread per pair).
  - `compact_active_pairs` — threshold filter: active pairs atomically appended to `active_pairs[]`; pruned pairs have `+∞` stamped into the result matrix.

  `dtw_wavefront` and `dtw_wavefront_global` gained an optional `pair_indices` buffer (+ `has_pair_indices` flag) so the DTW dispatch skips pruned pairs entirely. When pruning is disabled the wavefront path reads a 1-int dummy buffer (zero cost). Banded-row and regtile paths don't support pruning; requesting LB_Keogh on those paths silently disables it (with a `verbose` warning).

  New `MetalDistMatOptions` fields: `enable_lb_keogh` (default false), `lb_threshold` (default 0.0; pairs with `LB_Keogh > threshold` are pruned), `lb_envelope_band` (default −1 → use `band` or `max_L/10`). Result struct gained `pairs_pruned` counter.

  Measured on Apple M2 Max (random uniform series):

  | Workload | Baseline | Permissive LB (0% prune, pure overhead) | Strict LB (100% prune) |
  |---|---|---|---|
  | N=100, L=1000 | 158 ms | 159 ms (+0.6%) | **1.54 ms (102×)** |
  | N=200, L=1000 | 635 ms | 635 ms (~0%)  | **3.60 ms (176×)** |

  Strict numbers reflect 100% prune rate (random data + `lb_threshold=0`); real k-medoids cluster-assignment workloads land between depending on data + threshold tuning.

  Tests: `test_metal_lb_keogh.cpp` (4 cases, 136 assertions) covers: disabled path bit-identical to non-LB, permissive threshold keeps all pairs, strict threshold prunes + stamps `+∞`, and silent disable on banded_row. Exposed through Python (`compute_distance_matrix_metal(..., enable_lb_keogh=True, lb_threshold=..., lb_envelope_band=...)`). MATLAB exposure deferred (would require Metal-specific options on the backend-agnostic `Problem` class).

- **Register-tile Metal kernels** (`dtw_regtile_w4`, `dtw_regtile_w8`): short/medium unbanded workloads (max_L ≤ 256) now use a SIMD-group register-tile kernel. Algorithm from Schmidt & Hundt (2020) *"cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-Enabled GPUs"* (Euro-Par 2020, LNCS 12247, pp. 597–612); ported via the existing CUDA reference (`dtwc/cuda/cuda_dtw.cu:543-780`). Each of 32 lanes in a SIMD-group holds `TILE_W` (4 or 8) columns in registers; left-neighbor costs propagate via `simd_shuffle_up`, eliminating the per-diagonal `threadgroup_barrier` that dominated the wavefront kernel on short series. Eight pairs per threadgroup (`PAIRS_PER_TG = 8`). Dispatch: `TILE_W=4` for `max_L ≤ 128`, `TILE_W=8` for `128 < max_L ≤ 256`, band `== -1` only. Measured on Apple M2 Max (N=100, 4 950 pairs): L=64 → 0.37 ms; L=128 → 0.53 ms; L=192 → 0.87 ms; L=256 → 1.09 ms. Approximately **5–7× faster per cell than the extrapolated wavefront path** at these lengths. `MetalDistMatResult::kernel_used` reports `regtile_w4` / `regtile_w8`. Banded regtile is deferred (register pressure in the tile loop).
- **K-vs-N Metal kernels** (`dtw_kvn_wavefront`, `dtw_kvn_wavefront_global`): K queries × N targets parallel DTW on GPU, mirroring the CUDA `compute_dtw_k_vs_all` API. New host functions `compute_dtw_k_vs_all_metal` (two overloads: by-indices-into-series, and separate queries + targets) and `compute_dtw_one_vs_all_metal` (two overloads: by index, and external query). Result types `MetalKVsNResult` / `MetalOneVsNResult` match the CUDA shape. Enables GPU-accelerated k-medoids cluster-assignment loops on Apple Silicon — K*N pairs (instead of precomputing N² for small-K workloads).
- **Band-edge correctness fix** (affects all four wavefront kernels — `dtw_wavefront`, `dtw_wavefront_global`, `dtw_kvn_wavefront`, `dtw_kvn_wavefront_global`): the band-range iteration optimization relied on "out-of-band cells retain their INF initialization," which was wrong given the 3-buffer rotation — every 3 diagonals each buffer is reused as `cur`, so cells outside the current band can retain real DTW values from 3 diagonals ago. When the band range shifts by ±1 between diagonals, in-band reads at the new band edge hit those stale values and produce distances smaller than the true banded DTW. Pre-existing NxN wavefront banded paths silently returned wrong values (≤0.12% for the random-seed tests we had, up to ~7% on data where the stale values happen to lie on the DP path). Fix: after each diagonal's in-band writes, stamp INF into the two band-adjacent positions (`i_lo-1` and `i_hi+1`). Two extra writes per diagonal per pair; wavefront-banded benchmarks shift 0–9% (e.g. 75×2500 band=250: 212 → 232 ms; 75×10000 band=100 tight-band uses `dtw_banded_row` and is unaffected). Verified against CPU `dtwBanded` by new directed test "Metal wavefront NxN banded matches CPU dtwBanded".
- **Tests**: `test_metal_correctness.cpp` (18 cases, 2 044 assertions) validates GPU output against CPU reference at FP32 tolerance, covers length-2000, the 10 000-length global-memory fallback, three banded-row dispatch scenarios, an NxN wavefront-banded cross-check against `dtwBanded`, four K-vs-N cases (1-vs-N by index, 1-vs-N external query, K-vs-N by indices, K-vs-N banded), and four regtile cases (`w4` coverage, `w8` coverage, partial-tile edges L=127/255, variable-length asymmetric orientation). `test_metal_mmap.cpp` exercises the Metal strategy through `Problem::fillDistanceMatrix` into both `DenseDistanceMatrix` and the memory-mapped distance-matrix paths.
- **Benchmark**: `benchmarks/bench_metal_dtw.cpp` (Google Benchmark) compares Metal vs CPU at matching sizes; JSON lands in `benchmarks/results/mac_m2max/`.

### Added (macOS support)

- **`FindGUROBI.cmake`**: macOS search paths (`/Library/gurobi*/macos_universal2`, `/Library/gurobi*/mac64`) and `.dylib` library glob.
- **`CMakePresets.json`**: build and test presets for `clang-macos` (previously only `clang-win`). The preset now pins `/usr/bin/clang++` to avoid libc++ ABI conflicts when Homebrew LLVM is also installed.
- **`README.md`**: macOS-specific installation section (Homebrew libomp, Ninja, Gurobi location).
- **macOS CI workflow**: now installs Ninja, uses the `clang-macos` preset, enables HiGHS, tests Release config.

### Added (AI-Assisted Workflow)

- **Claude Code slash commands** in `.claude/commands/`: `/cluster`, `/distance`, `/evaluate`, `/convert`, `/visualize`, `/help`, `/troubleshoot`. Each command is a self-contained markdown file that guides Claude Code to drive the DTWC++ library on the user's behalf.
- **Docs page** `docs/content/getting-started/ai-commands.md` documenting the commands with examples and design principles.

### Fixed

- **`soft_dtw` / `soft_dtw_gradient`**: now throw `std::invalid_argument` if `gamma <= 0` instead of producing `inf`/`NaN`. `softmin_gamma` has a debug assert on the same condition.
- **`Problem.cpp`**: resolved TODO at line 808 — confirmed k-medoids objective uses raw DTW distances (not squared, unlike k-means).
- **`types/Index.hpp`**: added a debug assert for pointer underflow in `Index::operator-(difference_type)`.
- **`tests/unit/unit_test_clustering_algorithms.cpp`**: tests now write output CSVs to the system temp directory via `std::filesystem::temp_directory_path()`; previously tests polluted the project root with `test_clustering*.csv` files.
- **`CMakeLists.txt`**: Gurobi "not found" warning message now includes the macOS install path hint.

### Changed (Cleanup)

- Removed obsolete session artifacts: `.claude/reports/` (8 files), `.claude/summaries/` (3 files), `.claude/superpowers/` (1 file), `benchmarks/results/*.json` (16 machine-specific timing snapshots). These directories are now in `.gitignore`.
- No remaining PII in tracked files (paths/usernames). The `scripts/slurm/env.example` uses clear placeholder values.

### Added (I/O Formats)

- **Apache Arrow IPC reader** (`dtwc/io/arrow_ipc_reader.hpp`): zero-copy mmap loading via `ArrowIPCDataSource`. List + LargeList dispatch with int64 offsets for >2B elements.
- **Parquet reader** (`dtwc/io/parquet_reader.hpp`): scalar + list columns, directory loading, column auto-detection or `--column` override.
- **Parquet row-group streaming** (`dtwc/io/parquet_chunk_reader.hpp`): `ParquetChunkReader` class for reading individual row groups on demand — enables RAM-aware chunked CLARA.
- **`dtwc-convert`** Python CLI tool: converts Parquet/CSV/HDF5 → Arrow IPC or `.dtws` format.
- **CMake `DTWC_ENABLE_ARROW`**: optional Apache Arrow + Parquet support via `find_package` or CPM from source (static, ~20MB CLI binary).
- Auto-detection of input format from file extension: `.parquet`, `.arrow`, `.ipc`, `.feather`, `.dtws`, `.csv`.

### Added (Float32 Precision)

- **Runtime float32 precision**: `Precision` enum, `Data` float32 storage (`p_vec_f32`), `series_f32(i)` accessor, `dtw_fn_f32_t` dispatch.
- **Float32 view-mode**: `Data` supports non-owning `span<const float>` views for CLARA subsampling with float32 data.
- **CLI**: `--dtype float32|float64` (default float32, aliases: f32/fp32/float/f64/fp64/double). 2x memory saving, 0.003% max DTW error.

### Added (SLURM HPC Support)

- **Generalized SLURM scripts** (`scripts/slurm/`): portable remote helper, job templates (CPU, GPU, checkpoint, Parquet), `.env`-based configuration.
- **`slurm_remote.py`**: SSH/SFTP remote helper with SSH-key-first auth, two-hop gateway support, batch build/submit/download commands.
- **Job templates**: CPU test (Coffee k=2, Beef k=5), GPU test (fp32+fp64), checkpoint/resume verification, Parquet I/O test.
- **`verify_results.py`**: ARI-based ground truth comparison against UCR class labels.
- **`convert_ucr.py`**: UCR TSV to Parquet converter for testing Parquet I/O path.
- **`env.example`**: Configuration template with Oxford ARC example values.
- **Full UCR benchmark scripts** (`scripts/slurm/jobs/ucr_benchmark_cpu.slurm`, `ucr_benchmark_gpu.slurm`): run all 128 UCR datasets with Lloyd k-medoids, save distance matrices, per-dataset timing JSON, and aggregate summary.
- **`aggregate_results.py`**: merges per-dataset timing JSONs into a single benchmark results JSON for the docs website. Supports multiple runs (CPU, GPU, different architectures).
- **`slurm_remote.sh` benchmark commands**: `submit-benchmark-cpu`, `submit-benchmark-gpu [gpu_type]` for full UCR benchmarking on ARC.
- **SLURM documentation**: `docs/content/getting-started/slurm.md` with partition tables, GPU gres syntax, and troubleshooting.

### Changed (CLI)

- **`--precision` renamed to `--dtype`** (aliases: `--data-precision`, `--data-type`) to resolve CLI11 crash from duplicate `--precision` flag.
- **`--gpu-precision`** (alias: `--gpu-dtype`): GPU kernel precision with full alias set (f32/fp32/float32/f64/fp64/float64/double). Default: `auto`.
- **`--restart` renamed to `--resume`** (`--restart` kept as deprecated alias). Avoids confusion with "restart from scratch".
- **Verbose diagnostics**: `--verbose` now prints OpenMP threads, SLURM env vars (job ID, node, CPUs), CPU model, memory usage, and data memory estimate.
- **YAML/TOML config**: `precision` key split into `dtype` and `gpu-precision`. Added `resume` key.

### Fixed

- **CLI crash**: duplicate `--precision` flag caused CLI11 `OptionAlreadyAdded` exception at startup. Fixed by renaming to `--dtype` + `--gpu-precision`.
- **YAML config**: data precision was never loaded from YAML (only GPU precision was bound). Now both `dtype` and `gpu-precision` are loaded.

### Added (Data Access)

- **`Data::series(i)` → `span<const data_t>`**: uniform accessor for heap, mmap, and view modes.
- **CLARA zero-copy views** via `set_view_data()`: 48x subsample speedup by sharing parent memory.
- **`StoragePolicy` enum** (Auto/Heap/Mmap) in `dtwc/core/storage.hpp`.
- **Span overloads** for `compute_summary`, `compute_envelope`, `lb_keogh`, `lb_keogh_symmetric`.

### Added (RAM-Aware Chunked Processing)

- **`--ram-limit`** CLI flag: parsed with T/G/M/K suffixes (e.g., `--ram-limit 2G`).
- **Chunked CLARA**: when `--ram-limit` is set with Parquet input, CLARA streams row groups within the RAM budget. Subsamples and medoid series are loaded on demand; no full dataset in memory.
- **`ParquetChunkReader::read_rows()`**: sparse row access for loading subsamples by index.

### Changed (Build)

- **C++20 minimum:** All CMake targets upgraded from C++17 to C++20. CI matrix drops GCC 10, Clang 12, Clang 13 (no C++20 support).

### Changed (API)

- **`std::span` interfaces:** All public DTW distance functions (dtwFull, dtwFull_L, dtwBanded, ddtw*, adtw*, wdtw*, soft_dtw*, dtwMissing*, dtwAROW*) now accept `std::span<const data_t>` as primary overloads. `const std::vector<data_t>&` convenience overloads are retained for backward compatibility.
- **`dtw_fn_t` signature:** Changed from `std::function<data_t(const vector&, const vector&)>` to `std::function<data_t(std::span<const data_t>, std::span<const data_t>)>`.
- **`missing_utils.hpp`:** `has_missing()`, `missing_rate()`, `interpolate_linear()` now take `std::span<const T>` with vector convenience overloads.

### Added (SIMD)

- `lb_keogh()` dispatches to `dtwc::simd::lb_keogh_highway()` for `double` when `DTWC_ENABLE_SIMD=ON`, giving 2.7–3.3× speedup (measured AVX2, MSVC, i7).
- `DTWC_ENABLE_SIMD` now defaults to ON for standalone top-level builds (OFF for sub-projects and Python wheels). Google Highway provides runtime ISA dispatch — one binary runs optimally across SSE4/AVX2/AVX-512 nodes.

### Changed (SIMD performance)

- **Branchless scalar `lb_keogh`:** Replaced `std::max(T(0), std::max(eu, el))` with decomposed ternaries `max(0,eu) + max(0,el)` (valid for L≤U envelopes). Each ternary maps to a single `vmaxpd` instruction. Result: scalar lb_keogh is now **3.2–4.3× faster** and matches Highway performance — MSVC auto-vectorizer can now handle the loop. Added `#pragma omp simd reduction(+:sum)` to `lb_keogh_squared`, `lb_keogh_mv`, `lb_keogh_mv_squared`.
- **`dtw_multi_pair` uniform-length fast path:** When all 4 SIMD-lane pairs share the same dimensions (common case in DTW clustering), OOB masks are always all-false. New uniform path skips all `IfThenElse` and mask computation — 30% fewer ops per cell. Result: **4.6–5.6× faster** vs previous SIMD path; SIMD now **2.8× faster than sequential** (was 1.5× slower before).
- **Pre-hoisted row masks in `dtw_multi_pair` variable-length path:** `i_oob` masks (per-row OOB checks) are computed once before the j-loop into a `thread_local` buffer. Saves 4 scalar comparisons + stack write per inner-loop cell.
- **FMA in `z_normalize_highway`:** Normalize pass uses `MulAdd(val, inv_sd, bias)` (one FMA) instead of `Mul(Sub(val, mean), inv_sd)` (Sub + Mul). `bias = -mean * inv_sd` precomputed once.
- **`z_normalize_simd.cpp` header corrected:** Comment now accurately describes the two-pass König-Huygens algorithm (sum + sum-of-squares in one pass).

### Added (HPC build support)

- `DTWC_ARCH_LEVEL` CMake option (`""` / `"v3"` / `"v4"`): overrides `-march=native` with a portable x86-64 microarchitecture level. `v3` (AVX2+FMA) is safe for all modern HPC CPUs; `v4` targets AVX-512 nodes (Cascade Lake Xeon, Sapphire/Emerald Rapids, Genoa, Turin).
- CUDA builds now default to `CMAKE_CUDA_ARCHITECTURES=70;80;86;89;90` (V100 through H100) when not explicitly set. Override with `-DDTWC_CUDA_ARCH_LIST=...` or `-DCMAKE_CUDA_ARCHITECTURES=...`.
- GCC/Clang fast-math flags completed: added `-fno-rounding-math` and `-fno-signaling-nans` (safe for this codebase; complete the safe subset of `-ffast-math` excluding `-ffinite-math-only`).
- MSVC Release builds now include `/Gy` (function-level linking) for linker COMDAT elimination.

### Added (MIP Solver Improvements)

- MIP warm start: `--method mip` now runs FastPAM first and feeds the solution as a MIP start, dramatically reducing branch-and-bound solve time. Controlled by `--no-warm-start` flag.
- MIP solver settings exposed in CLI and TOML config: `--mip-gap`, `--time-limit`, `--no-warm-start`, `--numeric-focus`, `--mip-focus`, `--verbose-solver`.
- Gurobi branching priority on medoid selection variables A[i,i] — once medoids are fixed, assignment is a TU transportation problem (LP-integral).
- Optional YAML configuration file support (`--yaml-config config.yaml`) via yaml-cpp (`-DDTWC_ENABLE_YAML=ON`).
- `MIPSettings` struct on `Problem` for programmatic solver tuning.

### Added (MATLAB MEX Bindings)

- MATLAB MEX gateway (`bindings/matlab/dtwc_mex.cpp`) using C++ MEX API (R2018a+, RAII-safe).
- `+dtwc` MATLAB package: `dtw_distance`, `compute_distance_matrix`, `DTWClustering` class.
- `DTWClustering` handle class with `fit`, `fit_predict`, `predict` (mirrors Python API).
- CMake integration: `DTWC_BUILD_MATLAB=ON` automatically finds MATLAB and builds MEX.
- 1-based indexing conversion for all labels and medoid indices.
- FastPAM used for clustering (not legacy Lloyd).

### Changed (MIP Solver Improvements)

- Gurobi `NumericFocus` reduced from 3 to 1 (sufficient for 0/1/-1 constraint matrix, avoids 1.5-3x overhead).
- Gurobi now uses `MIPFocus=2` (optimality-focused) by default.
- MIP solver output suppressed by default (use `--verbose-solver` to see solver logs).

### Added (Wave 2A — Clustering Algorithms)

- Deferred dense distance-matrix allocation: `Problem::set_data()` no longer forces O(N^2) memory. Dense matrix allocated lazily on first `fillDistanceMatrix()` or `distByInd()` call.
- Shared medoid utilities (`algorithms/detail/medoid_utils.hpp`): `assign_to_nearest`, `compute_nearest_and_second`, `find_cluster_medoid`, `validate_medoids` — reusable, decoupled from Problem.
- Hierarchical agglomerative clustering (`algorithms/hierarchical.hpp`): Single, complete, and average linkage. `build_dendrogram()` + `cut_dendrogram()`. Small-N feature with hard `max_points=2000` guard. Ward's excluded (mathematically invalid for DTW).
- CLARANS experimental (`algorithms/clarans.hpp`): Bounded randomized k-medoids with `max_dtw_evals` and `max_neighbor` budget controls. Not exposed in CLI — requires benchmark evidence before promotion.

### Fixed (Wave 2A)

- FastCLARA now propagates `data.ndim`, `missing_strategy`, `distance_strategy`, and `verbose` to sub-problems. Previously, multivariate data was silently treated as univariate, and NaN data caused crashes.
- FastCLARA default sample size improved to `max(40+2k, min(N, 10k+100))` per Schubert & Rousseeuw 2021.
- `distByInd()` lazy allocation fix for checkpoint compatibility.

### Changed (Wave 2A)

- `Problem::set_data()` no longer calls `distMat.resize()`. The dense matrix is deferred to first actual use.

### Added (Wave 2B — Multivariate Variants + Lower Bounds)
- Multivariate WDTW: `wdtwFull_mv()`, `wdtwBanded_mv()` with position-dependent weights.
- Multivariate ADTW: `adtwFull_L_mv()`, `adtwBanded_mv()` with non-diagonal step penalty.
- Multivariate DDTW: via `derivative_transform_mv()` + standard multivariate DTW.
- Per-channel `compute_envelopes_mv()` and `lb_keogh_mv()`: valid lower bound on dependent multivariate DTW.
- `lb_keogh_squared()` and `lb_keogh_mv_squared()`: SquaredL2 metric LB_Keogh variants.
- Multivariate missing-data DTW: `dtwMissing_L_mv()`, `dtwMissing_banded_mv()` with per-channel NaN handling.

### Changed (Wave 2B)
- `Problem::rebind_dtw_fn()` dispatches to multivariate variants for WDTW, ADTW, DDTW, and ZeroCost missing when `data.ndim > 1`.

### Added (Wave 1B — Multivariate Foundation)
- Multivariate time series support via `Data.ndim` field (default 1, backward-compatible).
- `Data::series_length(i)` and `Data::validate_ndim()` for multivariate data management.
- `TimeSeriesView.ndim` with `at(i)` timestep access and `flat_size()`.
- Multivariate distance functors `MVL1Dist` and `MVSquaredL2Dist` in `warping.hpp`.
- `dtwFull_L_mv()` and `dtwBanded_mv()`: multivariate DTW with interleaved layout. `ndim=1` dispatches to existing scalar code (zero overhead).
- `derivative_transform_mv()`: stride-aware per-channel derivative transform for multivariate DDTW.

### Added (Wave 1C — Multivariate WDTW / ADTW / DDTW)

- `wdtwFull_mv()` and `wdtwBanded_mv()`: multivariate Weighted DTW with interleaved layout. `ndim=1` delegates to existing scalar code.
- `adtwFull_L_mv()` and `adtwBanded_mv()`: multivariate Amerced DTW with interleaved layout. `ndim=1` delegates to existing scalar code.
- `Problem::rebind_dtw_fn()` now dispatches WDTW, ADTW, and DDTW variants to their `_mv` counterparts when `data.ndim > 1`.

### Changed (Wave 1B)
- `Problem::rebind_dtw_fn()` dispatches to multivariate DTW when `data.ndim > 1`.
- `Problem::set_data()` calls `data.validate_ndim()` to catch malformed interleaved layouts early.

## Added

* **`missing_utils.hpp`**: Bitwise NaN check (`is_missing()`) safe under `-ffast-math`/`/fp:fast`, plus `has_missing()`, `missing_rate()`, `interpolate_linear()` with LOCF/NOCB edge handling.
* **`MissingStrategy` enum**: `Error` (default), `ZeroCost`, `AROW`, `Interpolate` for controlling missing-data handling in `Problem`.
* **DTW-AROW algorithm** (`warping_missing_arow.hpp`): One-to-one diagonal-only alignment for missing values (Yurtman et al., ECML-PKDD 2023). Linear-space, full-matrix, and banded variants.
* **5 new cluster quality metrics** in `scores.hpp`:
  * `dunnIndex()`: Min inter-cluster distance / max intra-cluster diameter.
  * `inertia()`: Total within-cluster sum of distances to medoids.
  * `calinskiHarabaszIndex()`: Medoid-adapted Calinski-Harabasz (uses overall medoid as global reference).
  * `adjustedRandIndex()`: Combinatorial agreement with ground-truth labels.
  * `normalizedMutualInformation()`: Information-theoretic agreement with ground-truth labels.

## Fixed

* **`warping_missing.hpp`**: Replaced `std::isnan()` with bitwise `is_missing()` check. The missing-data DTW feature was silently broken in Release builds due to `-ffast-math`/`/fp:fast` making `std::isnan()` unreliable.

## Changed

* **`Problem::fillDistanceMatrix()`** now pre-scans for NaN and throws with a helpful message under `MissingStrategy::Error`. Auto-disables LB pruning when missing data is detected under `ZeroCost`/`AROW` strategies.
* **`Problem::rebind_dtw_fn()`** dispatches based on `missing_strategy` member.

## Core

* **Parallel pruned distance matrix**: `fill_distance_matrix_pruned()` (Problem-based) is now parallelised with OpenMP. Uses lock-free atomic CAS on `nn_dist` for nearest-neighbor tracking across threads, with `schedule(dynamic, 16)` for load balancing. Precomputation of summaries and envelopes is also parallelised.
* **Distance matrix strategy selection**: new `DistanceMatrixStrategy` enum (`Auto`, `BruteForce`, `Pruned`, `GPU`) and `Problem::distance_strategy` member. `fillDistanceMatrix()` now dispatches based on the selected strategy. `Auto` (default) selects `Pruned` for Standard DTW variant, `BruteForce` for non-standard variants (DDTW, WDTW, ADTW, SoftDTW).

## CUDA

* **1-vs-N and K-vs-N GPU DTW kernels**: new `compute_dtw_one_vs_all()` and `compute_dtw_k_vs_all()` functions compute DTW distances from one or K query series against all N target series. Dedicated kernels (`dtw_one_vs_all_wavefront_kernel`, `dtw_one_vs_all_warp_kernel`, `dtw_one_vs_all_regtile_kernel`) use a 2D grid (targets x queries) with query loaded into shared memory per block. For k-medoids clustering with K=5 medoids and N=200 series, this avoids recomputing the full NxN matrix when medoids change (K*N = 1,000 pairs vs 19,900). Supports FP32/FP64 auto-precision, banded DTW, L1/squared-L2 metrics, and external query series. New result types: `CUDAOneVsNResult`, `CUDAKVsNResult`.
* **GPU-accelerated LB_Keogh pruning**: new `compute_lb_keogh_cuda()` function computes symmetric LB_Keogh lower bounds for all N*(N-1)/2 pairs on GPU. Two new CUDA kernels: `compute_envelopes_kernel` (sliding-window min/max per series) and `compute_lb_keogh_kernel` (embarrassingly parallel, one thread per pair). New `CUDADistMatOptions` fields: `use_lb_pruning` enables LB computation before DTW, `skip_threshold` prunes pairs with LB exceeding the threshold (set to INF without computing DTW). Standalone `CUDALBResult compute_lb_keogh_cuda(series, band)` API for use in clustering and nearest-neighbor search.
* **On-device pair index computation**: eliminated host-side pair index arrays (`h_pair_i`, `h_pair_j`) and their H2D transfers by computing `(i,j)` from the flat upper-triangle index directly on GPU via `decode_pair()`. For N=1000 this saves 4 MB of transfers and 2 device allocations.
* **GPU-side NxN result matrix**: kernels now write DTW distances directly into a symmetric NxN matrix on device, eliminating the per-pair distance array, its D2H transfer, and the host-side O(N^2) fill loop. A single contiguous `cudaMemcpyAsync` transfers the complete result.
* **Precomputed integer band boundaries**: banded DTW band checks (6 FP64 operations per cell) are now precomputed once per pair into shared memory (wavefront kernel) or registers (warp kernel), replacing per-cell FP64 math. On consumer GPUs with 1:64 FP64 rate, this eliminates ~192 FP32-equivalent cycles per cell in the banded path.
* **Stream-based async pipeline**: `launch_dtw_kernel` now uses a CUDA stream with `cudaMemcpyAsync` for H2D/D2H transfers and stream-ordered kernel launches. Pinned host memory (`cudaMallocHost`) is used for the two large buffers (flat series data and output distances) to enable true overlap of transfers with compute. Falls back gracefully to pageable memory if pinned allocation fails. GPU timing now uses `cudaEvent`-based measurement for accurate pipeline profiling. Added RAII wrappers (`PinnedPtr`, `CudaStream`, `CudaEvent`) to `cuda_memory.cuh`.
* **Persistent kernel mode**: the `dtw_wavefront_kernel` now supports persistent scheduling. When `num_pairs` significantly exceeds the GPU's resident block capacity (>4x), blocks loop over pairs using a global atomic work counter instead of the one-pair-per-block model. This eliminates block scheduling overhead for large-N workloads (e.g. N=1000: 499,500 pairs, but only ~80-160 blocks resident). Falls back to original behavior for small workloads. Fully transparent to the caller.
* Added optional CUDA GPU acceleration for batch DTW distance matrix computation (`DTWC_ENABLE_CUDA` CMake option, OFF by default).
* New `dtwc::cuda::compute_distance_matrix_cuda()` computes all N*(N-1)/2 DTW pairs on GPU.
* **Anti-diagonal wavefront kernel**: replaced single-threaded-per-block kernel with multi-threaded anti-diagonal wavefront parallelism. Achieved **24x kernel speedup** (910M -> 22 Gcells/sec), making GPU **5-7x faster than 10-core CPU** with OpenMP.
* **Warp-level DTW kernel for short series** (`dtw_warp_kernel`): for series with `max_L <= 32`, a new kernel packs 8 DTW pairs per block (one warp per pair) using register-based anti-diagonal propagation with `__shfl_sync()`. Eliminates shared-memory cost-matrix buffers and dramatically improves occupancy for short series workloads.
* **Register-tiled DTW kernel** (`dtw_regtile_kernel`): inspired by the cuDTW++ approach (Euro-Par 2020), a new kernel handles medium-length series (32 < max_L <= 256) using register tiling. Each thread processes a stripe of TILE_W columns entirely in registers, with inter-thread communication via `__shfl_sync`. TILE_W=4 covers up to 128 columns, TILE_W=8 covers up to 256. Eliminates shared-memory cost-matrix buffers for medium series, bridging the gap between the warp kernel (L<=32) and the shared-memory wavefront kernel (L>256).
* **`__ldg()` texture cache reads**: global memory reads for series data now use `__ldg()` intrinsic, forcing the read-only texture cache path for ~5-15% improvement on longer series.
* **Banded DTW on GPU**: the `CUDADistMatOptions::band` parameter is now honored by the kernel. Uses the same slope-adjusted Sakoe-Chiba window as the CPU `dtwBanded` implementation. When `band < 0` (default), full unconstrained DTW is computed with zero overhead.
* Python bindings expose `cuda_available()`, `cuda_device_info()`, `compute_distance_matrix_cuda()`, and `CUDA_AVAILABLE` flag.
* **GPU architecture detection** (`gpu_config.cuh`): runtime query and caching of GPU compute capability, SM count, shared memory limits, and FP64 throughput classification (Full vs Slow). Supports up to 16 devices.
* **FP32/FP64 templated kernel**: the DTW wavefront kernel is now templated on compute type (`float` or `double`). New `CUDAPrecision` enum (`Auto`, `FP32`, `FP64`) in `CUDADistMatOptions` controls precision. `Auto` selects FP32 on consumer GPUs (1:32 FP64 rate) and FP64 on HPC GPUs (1:2 FP64 rate), giving up to 32x throughput improvement on consumer hardware with ~1e-5 relative error.
* Requires CUDA Toolkit; all CUDA code is behind `#ifdef DTWC_HAS_CUDA` so the library builds without it.

## CI

* Added CUDA/MPI detection smoke test workflow (`.github/workflows/cuda-mpi-detect.yml`): Linux CUDA compile, Linux MPI build+test, macOS CUDA graceful rejection, macOS MPI build, and Windows MPI configure.

## Build system

* Fixed CUDA detection on Windows with multiple CUDA toolkit versions: auto-sets missing `CUDA_PATH_Vxx_y` env-vars and generates `Directory.Build.props` to persist `CudaToolkitCustomDir` for MSBuild at build time.
* Fixed MSVC flags (`/diagnostics:column`, `/fp:fast`, `/openmp:experimental`) leaking into nvcc by adding `$<$<COMPILE_LANGUAGE:C,CXX>:...>` generator expression guards.
* Fixed `find_package(OpenMP)` failure when CUDA language is enabled by requesting only the CXX component.
* Fixed MPI detection: `MPI_CXX_FOUND` variable didn't propagate from `dtwc_setup_dependencies()` function scope; now checks `TARGET MPI::MPI_CXX` instead.
* Improved MS-MPI SDK detection on Windows with fallback to default install path and actionable error messages.
* Added llfio dependency (optional, `DTWC_ENABLE_MMAP=ON`) for cross-platform memory-mapped I/O.

## Benchmarks

* Added GPU benchmark suite (`benchmarks/bench_cuda_dtw.cpp`): GPU vs CPU comparison, N-scaling, L-scaling with throughput counters (pairs/sec, cells/sec).
* Added MPI benchmark suite (`benchmarks/bench_mpi_dtw.cpp`): distributed distance matrix scaling across ranks with speedup/efficiency reporting.
* Added `benchmarks/README.md` with baseline performance numbers and optimization history.

## Performance / API

* Added `MmapDistanceMatrix` — memory-mapped distance matrix via llfio for large-N problems. Supports warmstart: reopen existing cache file to resume interrupted computation. Binary format with 32-byte header (magic, version, CRC32, N).
* Added `MmapDataStore` — memory-mapped contiguous cache for time series data. Supports variable-length and multivariate series. Binary format with 64-byte header (magic, version, CRC32, N, ndim) + offset table + contiguous data. Extracted shared `crc32.hpp` utility.
* Added `DataLoader::count()` — count series without loading data (directory iteration or line counting).
* Added pointer+length overloads for all core DTW functions (`dtwFull`, `dtwFull_L`, `dtwBanded`, `dtwMissing_L`, `dtwMissing_banded`) enabling zero-copy calls from bindings. The `detail::*_impl` functions now operate on raw pointers; vector overloads forward to them.
* Python `dtw_distance` and `dtw_distance_missing` now accept numpy arrays via `nb::ndarray` (zero-copy, no vector allocation).
* Eliminated vector copies in `dtwc::core::dtw_distance` pointer overload and `dtw_runtime`.

## CLI

* Rewrote `dtwc_cl` CLI with full TOML configuration file support via CLI11 `--config` flag.
* CLI now supports all clustering methods (FastPAM, FastCLARA, kMedoids Lloyd, MIP), all DTW variants (standard, DDTW, WDTW, ADTW, Soft-DTW), checkpointing, and flexible CSV output (labels, medoids, silhouette scores, distance matrix).
* Added `--method auto` (new default): auto-selects `pam` for N≤5000, `clara` for N>5000.
* Added CLARA sample size auto-scaling for N>50K: `max(40+2k, sqrt(N)*k)`.
* Added `--mmap-threshold` to control when memory-mapped distance matrix activates (default 50K).
* Added `--restart` to resume from binary checkpoint (distance matrix cache + clustering state).
* Added case-insensitive option validation for method, metric, variant, and solver flags.
* Added example TOML configuration file at `examples/config.toml`.

## Tests

* Added cross-language integration tests (`tests/integration/test_cross_language.py`): verifies C++ and Python interfaces produce identical results for DTW distances (L1/squared-euclidean, banded, missing-data), compute_distance_matrix, FastPAM/CLARA clustering, DTW variant consistency (DDTW/WDTW/ADTW/Soft-DTW), checkpoint save/load roundtrip, and end-to-end pipeline (data -> distance matrix -> clustering -> evaluation scores).

## Documentation

* Added MPI and CUDA installation guide (`docs/1_getting_started/3_mpi_cuda_setup.md`) covering all platforms (Windows, Linux, macOS), CMake detection, build flags, and troubleshooting.
* Updated installation guide: added Python installation section, GPU/MPI cross-references, macOS OpenMP note, and CUDA_PATH tip for Linux.

## Examples

* Added Python examples: `04_missing_data.py` (NaN-aware DTW), `05_fast_clara.py` (scalable clustering), `06_distance_matrix.py` (fast pairwise computation with timing), `07_checkpoint.py` (save/resume distance matrices).
* Added MATLAB quickstart example: `bindings/matlab/examples/example_quickstart.m` (DTW distance, distance matrix, clustering).
* Added C++ new-features example: `examples/example_new_features.cpp` (DTW variants, missing data, FastCLARA, checkpointing).

## Documentation
* Fixed DTW formula in `docs/2_method/2_dtw.md`: changed from squared L2 `(x_i - y_j)^2` to L1 `|x_i - y_j|` to match actual code implementation.
* Fixed pairwise comparison count: corrected from `1/2 * C(p,2)` to `C(p,2) = p(p-1)/2`.
* Fixed warping window description: band=1 allows a shift of 1 (not equivalent to Euclidean distance); band=0 forces diagonal alignment.
* Fixed spelling errors: "seies" -> "series", "wapring" -> "warping", "assertain" -> "ascertain".
* Added note clarifying that the default pointwise metric is L1 (absolute difference).
* Added z-normalization section with note that population stddev (N, not N-1) is used.
* Added `docs/2_method/5_algorithms.md`: clustering algorithm documentation with corrected FastPAM citation (JMLR 22(1), 4653-4688) and accurate complexity descriptions.
* Added `docs/2_method/6_metrics.md`: distance metrics documentation with LB_Keogh compatibility table and corrected Huber LB_Keogh reasoning.

## I/O

* Added **Python I/O utilities** (`dtwcpp.io`): `save_dataset_csv` / `load_dataset_csv` (always available), `save_dataset_hdf5` / `load_dataset_hdf5` (requires `h5py`), `save_dataset_parquet` / `load_dataset_parquet` (requires `pyarrow`).
* HDF5 files store series data, names, distance matrices, and metadata in a single compressed file.
* Added optional dependency groups in `pyproject.toml`: `hdf5`, `parquet`, `io` (both).

## New features

* Added **MATLAB MEX bindings** — new `bindings/matlab/` directory with C++ MEX API gateway and MATLAB `+dtwc` package: `dtwc.dtw_distance`, `dtwc.compute_distance_matrix`, `dtwc.DTWClustering`. Build with `cmake .. -DDTWC_BUILD_MATLAB=ON`.
* Added **checkpoint save/load** for distance matrix computation (`save_checkpoint`, `load_checkpoint`). Saves partial or complete distance matrices to disk as CSV + metadata text file, enabling resume after crashes.
* Added **binary checkpoint** (`save_binary_checkpoint`, `load_binary_checkpoint`) for clustering state (medoids, labels, cost, iteration). Used by `--restart` CLI flag.
* Added `count_computed()` and `all_computed()` methods to `DenseDistanceMatrix`.
* Added `distance_matrix()` accessors and `set_distance_matrix_filled()` to `Problem` class.
* Added Python bindings for `save_checkpoint`, `load_checkpoint`, and `CheckpointOptions`.
* Added **DTW with missing data** (`dtwMissing`, `dtwMissing_L`, `dtwMissing_banded`). NaN values contribute zero cost. Supports L1/SquaredL2, early abandon, banding. Reference: Yurtman et al. (2023), ECML-PKDD.
* Added `dtw_distance_missing` Python binding.
* Added **FastCLARA** scalable k-medoids clustering (`dtwc::algorithms::fast_clara`). Subsampling + FastPAM avoids O(N^2) memory. Reference: Kaufman & Rousseeuw (1990); Schubert & Rousseeuw (2021, JMLR).
* Added Python binding for `fast_clara()`.
* Added `MetricType` parameter (L1, SquaredL2) to all core DTW functions. Template lambda dispatch for zero inner-loop overhead.
* Refactored DTW into `detail::*_impl` helpers with distance callable template parameter.
* Added `metric` parameter to `dtw_distance` Python binding.
* Added `compute_distance_matrix` Python function with OpenMP parallelism.
* Added **LB-pruned distance matrix** (`compute_distance_matrix_pruned`). Precomputes envelopes and summaries once, then uses LB_Kim (O(1)) and LB_Keogh (O(n)) as early-abandon thresholds for each DTW computation. Reduces inner-loop work by 30-60% for correlated series. Enabled by default in the Python `compute_distance_matrix` binding via `use_pruning=True`.
* Added `distance_matrix_numpy()` method to `Problem` Python class.
* Added `-ffast-math` (GCC/Clang) and `/fp:fast` (MSVC) for Release builds.
* Added **Google Highway SIMD infrastructure** (`DTWC_ENABLE_SIMD` option, default OFF). Prototype kernels for future use.
* Added `#pragma omp simd` hints to LB_Keogh and z_normalize.
* Added LB_Keogh, z_normalize, envelope benchmarks.
* Added `DTWClustering` sklearn-compatible Python class.
* Added core type system (`dtwc::core` namespace).
* Added FastPAM1 k-medoids clustering algorithm.
* Added z-normalization, lower bounds, DenseDistanceMatrix, pruned distance matrix.
* Added Google Benchmark integration.

## Architecture

* Removed Armadillo dependency from all hot-path code.
* Replaced `arma::Mat` with `core::ScratchMatrix` and `core::DenseDistanceMatrix`.
* Unified `FastPAMResult` with `core::ClusteringResult`.

## Bug fixes

* Fixed `DenseDistanceMatrix` NaN sentinel broken under `-ffast-math` / `/fp:fast`. Replaced with boolean vector.
* Fixed `throw 1` bare integer throw — now throws `std::runtime_error`.
* Fixed `static std::mt19937` ODR violation — changed to `inline`.
* Fixed `.at()` in hot DTW loop — replaced with `operator[]`.
* Fixed `dtwBanded` 512MB/thread allocation — rolling buffer.
* Fixed `fillDistanceMatrix` integer overflow at N>46K.
* Fixed `dtwBanded` template default from `float` to `double`.
* Fixed `static` vectors data race in `calculateMedoids`.
* Fixed DBI formula.
* Fixed broken examples.
* Renamed `cluster_by_kMedoidsPAM` to `cluster_by_kMedoidsLloyd`.
* Fixed z_normalize tests.

<br/><br/>
# DTWC v1.0.0

## New features
* HiGHS solver is added for open-source alternative to Gurobi (which is now not necessary for compilation and can be enabled by necessary flags). 
* Command line interface is added. 
* Documentation is improved (Doxygen website).

## Notable Bug-fixes
* Sakoe-Chiba band implementation is now more accurate. 

## API changes
* Replaced `VecMatrix<data_t>` class with `arma::Mat<data_t>`. 

## Dependency updates:
* Required C++ standard is reduced from C++20 to C++17 as it was causing `call to consteval function 'std::chrono::hh_mm_ss::_S_fractional_width' is not a constant expression` error for clang versions older than clang-15.
* `OpenMP` for parallelisation is adopted as `Apple-clang` does not support `std::execution`. 

## Developer updates: 
* The software is now being tested via Catch2 library. 
* Dependabot is added. 
* `CURRENT_ROOT_FOLDER` and `DTWC_ROOT_FOLDER` are seperated as DTW-C++ library can be included by other libraries. 

<br/><br/>
# DTWC v0.3.0

## New features
* UCR_test_2018 data integration for benchmarking. 

## Notable Bug-fixes
* N/A

## API changes
* DataLoader class is added for data reading. 
* `settings::resultsPath` is changed with `out_folder` member variable to have more flexibility. 
* `get_name` function added to remove `settings::writeAsFileNames` repetition)
* `std::filesystem::path operator+` was unnecessary and removed. 

<br/><br/>
# DTWC v0.2.0

A user interface is created for other people's use. 

## New features / updates
- Scores file with silhouette score is added. 
- `dtwFull_L` (L = light) is added for reducing memory requirements substantially.  

## API changes
- Problem class for a better interface. 
- `mip.hpp` and `mip.cpp` files are created to contain MIP functions.

## Notable Bug-fixes
* Gurobi better path finding in macOS. 
* TBB could not be used in macOS so it is now option with alternative thread-based parallelisation. 
* Time was showing wrong on macOS with std::clock. Therefore, moved to chrono library.

## Formatting: 
- Include a clang-format file. 

## Dependency updates
  * Required C++ standard is upgraded from C++17 to C++20. 

<br/><br/>
# DTWC v0.1.0

This is the initial release of DTWC. 

## Features
- Iterative algortihms for K-means and K-medoids 
- Mixed-integer programming solution support via YALMIP/MATLAB. 
- Support for `*.csv` files generated by Pandas.  

## Dependencies
  * A compiler with C++17 support. 
  * We require at least CMake 3.16. 