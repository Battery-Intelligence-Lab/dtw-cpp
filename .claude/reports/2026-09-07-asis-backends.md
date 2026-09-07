# AS-IS map — accelerator / distributed backends (`cuda/`, `metal/`, `mpi/`)

Repo `C:\D\git\dtw-cpp`, branch `Claude`, HEAD `a31956e`. Line-by-line pass over every
file in scope. Evidence tags: `[confirmed path:line]` = read at HEAD; `[inferred]` = a
conclusion from read evidence, with the confirming test named. Metal runtime is
`[BLOCKED-ENV]` on this Windows host — every Metal claim is source-level only.

---

## 1. Module map

| File | Lines | Responsibility | Entry points | Needs from the host side |
|---|---:|---|---|---|
| `dtwc/cuda/cuda_dtw.cu` | 2475 | All CUDA device kernels + host launchers: 3 pairwise kernels (wavefront/warp/regtile), 3 one-vs-N kernels, envelope + LB_Keogh + compaction kernels, 2 thread-local workspaces | `compute_distance_matrix_cuda` :1436, `compute_lb_keogh_cuda` :1587, `compute_dtw_one_vs_all` x2 :2272/:2338, `compute_dtw_k_vs_all` :2401 | Only `const std::vector<std::vector<double>>&` + an options POD. No `Problem`/`Data` include `[confirmed cuda_dtw.cu:22-41]` |
| `dtwc/cuda/cuda_dtw.cuh` | 140 | Public CUDA ABI: `CUDAPrecision`, `CUDADistMatOptions` (extends `DistMatOptionsBase`), 4 result structs | struct/function decls | `enums/KernelOverride.hpp`, `error.hpp`, `core/gpu_dtw_common.hpp` `[confirmed :19-21]` |
| `dtwc/cuda/cuda_memory.cuh` | 133 | RAII for device mem / pinned host mem / stream / event; `CUDA_CHECK_ALLOC` | `cuda_alloc`, `pinned_alloc(_nothrow)`, `make_cuda_stream/event` | nothing from dtwc |
| `dtwc/cuda/gpu_config.cuh` | 110 | Per-device capability cache (`GPUConfig`), FP64-rate classification | `query_gpu_config(int)` :324 | nothing from dtwc |
| `dtwc/cuda/kernel_selection.hpp` | 72 | **Host-only, CUDA-free** kernel-path policy | `auto_kernel` :415, `select_kernel` :423, `kernel_path_name` :444 | `enums/KernelOverride.hpp` only |
| `dtwc/cuda/launch_prep.hpp` | 114 | **Host-only, CUDA-free** preconditions + geometry: pair-count limit, device presence, length scan, hint fold, wavefront buffer count | `require_pair_count_fits` :502, `require_cuda_device` :514, `scan_series_lengths` :525, `kernel_selection_length` :546, `wavefront_buffer_count` :562 | `error.hpp` only |
| `dtwc/metal/metal_dtw.mm` | 2237 | MSL source as a raw string (lines 60-992, 10 kernels) + ObjC++ host: lazy `MetalContext`, pairwise dispatch, LB pipeline, K-vs-N | `metal_available` :1200, `metal_device_info` :1206, `compute_distance_matrix_metal` :1230, `compute_lb_keogh_metal` :1794, `compute_dtw_k_vs_all_metal` x2 :2172/:2190, `compute_dtw_one_vs_all_metal` x2 :2200/:2219 | `const std::vector<std::vector<double>>&` + options POD. No `Problem`/`Data` `[confirmed :27-43]` |
| `dtwc/metal/metal_dtw.hpp` | 159 | Public Metal ABI: `MetalPrecision`, `MetalDistMatOptions` (+ `lb_threshold`, `lb_envelope_band`), result structs | decls | same three dtwc headers as CUDA `[confirmed :27-29]` |
| `dtwc/metal/detail/chunk_dispatch.hpp` | 26 | **Host-only** `pair_chunk_offset(size_t) -> int64_t` — one line, exists purely as a testable seam | `pair_chunk_offset` :21 | nothing |
| `dtwc/mpi/mpi_distance_matrix.cpp` | 193 | Rank block decomposition, OpenMP inner fill via the **CPU** DTW, chunked `MPI_Allreduce` | `compute_distance_matrix_mpi` :160, `ensure_mpi_initialized` :122, `mpi_rank_and_size` :131 | `../warping.hpp` (`dtwBanded`/`dtwFull_L`), `../parallelisation.hpp` (`omp_chunk_size`) `[confirmed :104,:116]` |
| `dtwc/mpi/mpi_distance_matrix.hpp` | 86 | `MPIDistMatOptions` (band + verbose **only**), `MPIDistMatResult` | decls | nothing |
| `dtwc/mpi/allreduce_chunking.hpp` | 58 | **MPI-free** INT_MAX chunking arithmetic | `max_allreduce_chunk` :313, `allreduce_chunk_count` :329 | nothing |
| `dtwc/core/gpu_dtw_common.hpp` | 60 | `DistMatOptionsBase` / `DistMatResultBase` shared by both GPUs; re-exports `normalize_public_distance` into `dtwc::gpu::detail` | structs | `core/public_distance.hpp`, `enums/KernelOverride.hpp` |

**Host-side dependency, exactly.** The only members `Problem::fill_distance_matrix` hands
to either GPU backend are:

- `data_.p_vec` — `std::vector<std::vector<data_t>>`, `[confirmed Data.hpp:33]`, passed at
  `[confirmed Problem.cpp:1061]` (CUDA) and `[confirmed Problem.cpp:1085]` (Metal).
- `band` — `int`, `[confirmed Problem.hpp:307]`, at `Problem.cpp:1051, 1080`.
- `cuda_settings.device_id` / `.precision` — `[confirmed Problem.hpp:49,54,315]`, at
  `Problem.cpp:1052-1056`.
- `verbose_` — `[confirmed Problem.hpp:191]`, at `Problem.cpp:1058, 1082, 1010`.
- `data_.size()` — only for the `pairs_computed == 0` guard `[confirmed Problem.cpp:998]`.
- Write-back sink: `distMat` (`std::variant<core::DenseDistanceMatrix,
  core::MmapDistanceMatrix>` `[confirmed Problem.hpp:125,135]`) through `visit_distmat` +
  `m.resize(result.n)` + `m.set(i, j, ...)` `[confirmed Problem.cpp:1002-1009]`.

MPI is reached by **no** `Problem` code path — its only callers are
`benchmarks/bench_mpi_dtw.cpp:84,94` and `tests/unit/unit_test_mpi.cpp` `[confirmed via
git grep]`. `DistanceMatrixStrategy` has no `MPI` member `[confirmed Problem.cpp:1024-1104:
Pruned / CUDA / Metal / BruteForce / Auto only]`.

---

## 2. Data and control flow

### 2.1 CUDA pairwise, in call order

1. `compute_distance_matrix_cuda` :1436 -> `validate_cuda_precision` :1440,
   `validate_kernel_override` :1441.
2. `detail::require_cuda_device(cuda_available(), ...)` :1447 — **before** the `N*N` resize
   :1454. `detail::require_pair_count_fits(upper_triangle_pairs(N), ...)` :1448 — 64-bit
   throughout `[confirmed launch_prep.hpp:495-498]`.
3. `cudaSetDevice(opts.device_id)` :1458.
4. `detail::scan_series_lengths(series, lengths)` :1462 — fills `vector<int> lengths`,
   returns `max_L` (`size_t`).
5. `detail::select_kernel(detail::kernel_selection_length(max_L, opts.max_length_hint),
   opts.kernel_override)` :1466-1468. `kernel_selection_length` only ever **raises** the
   heuristic input `[confirmed launch_prep.hpp:546-552]`; buffer sizing keeps the scanned
   `max_L`.
6. `resolve_fp32(opts.precision, opts.device_id)` :1479 -> `query_gpu_config(...).fp64_rate`
   :81.
7. Optional LB pre-pass, only when `use_lb_keogh && band >= 0` :1487:
   `upload_series_to_workspace` :1493 -> `launch_lb_keogh_kernel` :1494 ->
   (if `lb_threshold > 0`) `compact_active_pairs` :1497.
8. `launch_dtw_kernel<T>` :1508 / :1516 / :1523 (and the mirrored `double` block
   :1547-1566).

**Packing / padding.** `flatten_series_buffer` :1027-1049 writes a dense `[N x max_L]`
row-major array, `memcpy` for `T=double`, per-element cast for `float`, and
**zero-pads** the tail `[confirmed :1046-1047]`. Padding is never read: every kernel bounds
its loops by `lengths[]`.

**Transfer.** `cudaMemcpyAsync` H2D of series + lengths on the workspace stream
:1179-1182; kernel; `cudaMemcpyAsync` D2H of the whole `N x N` matrix :1291-1292;
`cudaEventRecord` / `cudaStreamSynchronize` :1297-1298. Host buffers are pinned above a
256 KiB threshold `[confirmed :1160-1166]` with a silent, deliberate `std::vector`
fallback when `cudaMallocHost` fails `[confirmed cuda_memory.cuh:215-225,
cuda_dtw.cu:890-910]` — a **performance** fallback, not a result fallback. DELIBERATE.

**Launch geometry.**
- Warp path :1187-1198 — `grid = ceil(num_pairs/8)`, `block = 256` (8 warps x 32),
  `shared = 8*2*32*sizeof(T)`; one warp per pair, register anti-diagonal via
  `__shfl_sync` :456-457.
- RegTileW4/W8 :1199-1224 — same grid/block, `shared = 8*2*max_L*sizeof(T)`; each lane owns
  a `TILE_W`-column stripe in registers, `__shfl_sync(..., lane-1)` for the left/diagonal
  boundary :627-629.
- Wavefront :1225-1281 — `n_bufs = wavefront_buffer_count(max_L)` (5 / 3 / 2 / 3)
  `[confirmed launch_prep.hpp:562-567]`, `shared = n_bufs*max_L*sizeof(T)`,
  `block_size = 256` **hard-coded** :1242, `require_shared_mem_fits` :1244 ->
  `cudaFuncSetAttribute` above 48 KiB :1247-1251. Persistent mode when
  `num_pairs > 4 * sm_count * blocks_per_sm` :1259-1260; blocks then pull pairs from a
  device `atomicAdd` counter :196.
- **Three-buffer route for long series.** `use_double_buf = max_L>1024 && max_L<=2048 &&
  !preload` :163-164; above 2048 the kernel reverts to 3 rotating buffers because the
  double-buffer path caches each thread's cost-diagonal in `T cd[MAX_SI=8]` and
  `blockDim.x(256) * 8 = 2048` cells is its ceiling :158-162, :276-282. The host mirrors the
  cap in `wavefront_buffer_count` and warns once :1230-1238. Preload mode (`max_L <= 512`)
  additionally stages both series in shared memory :244-253. DELIBERATE — correctness cap.

**Pair decomposition.** No host-side pair arrays exist; every kernel decodes its own index
with the SSOT `dtwc::detail::decode_pair` (`using` at :63), which is
`__host__ __device__` `[confirmed decode_pair.hpp:32-36,50]`. `si`/`sj` are `int64_t`,
and a `static_assert` pins the 64-bit matrix index :213-214.

**Write-back.** Kernels write **both** triangles directly into the device `N x N` matrix
(:341-342, :493-494, :713-714). The host converts with `convert_result_matrix` :1077-1106:
for `double` a per-row `memcpy` with the diagonal forced to `0.0`; for `float` a per-element
`normalize_public_distance` that maps `FLT_MAX -> DBL_MAX` `[confirmed
public_distance.hpp:19-24]`. The result is always a **dense `std::vector<double>` of N^2**;
there is no mmap or streaming sink anywhere in `cuda/`. `Problem` then copies it a second
time, scalar by scalar, into `distMat` `[confirmed Problem.cpp:1006-1008]`.

**f32/f64.** `CUDAPrecision::Auto` -> FP32 iff `fp64_rate == Slow` :81, classified from
compute-capability minor == 0 `[confirmed gpu_config.cuh:368-375]`. Every launcher is a
template over `T` and the whole FP32/FP64 decision is a source-duplicated `if/else`
:1490-1568.

**LB pre-pass.** `compute_envelopes_kernel` :727 (one block per series, `w = max(band,0)`,
zero-fills padding :763-766) -> `compute_lb_keogh_kernel` :777 (one thread per pair,
symmetric `max(lb1, lb2)` over the `min(Li,Lj)` prefix) -> `compact_active_pairs_kernel`
:827 (atomic append of survivors; pruned pairs stamped with the compute-type `INF` at both
`(si,sj)` and `(sj,si)` :851-852). Standalone entry `compute_lb_keogh_cuda` :1587 always
uses FP64 :1610. Note: when `use_lb_keogh` is set but `lb_threshold <= 0`, the whole LB
pipeline runs and its output is **discarded** except for the timing field :1515-1521.

**Error / edge handling (CUDA).**
- No device -> `dtwc::DeviceError` at every public entry (:1447, :1592, :2280, :2346, :2410).
- Pair count > INT_MAX -> `dtwc::InvalidInput` at every public entry, plus a last-line
  defence inside `launch_dtw_kernel` :1148.
- Over-large shared request -> `dtwc::DeviceError` :92-96 via `require_shared_mem_fits`.
- Zero-length series -> `INF` written by each kernel :228-238, :406-412, :555-561.
- **Every other CUDA runtime failure throws bare `std::runtime_error`**
  `[confirmed cuda_dtw.cu:43-51, cuda_memory.cuh:157-165]` — outside the `dtwc::Error`
  taxonomy (PLAN F31, which currently names only Metal).

### 2.2 Metal pairwise, in call order

`compute_distance_matrix_metal` :1230 -> validate :1234-1235 -> **`if (N <= 1) return`
:1241 runs BEFORE the availability check** :1246-1249 -> length scan :1252-1257 -> buffer
alloc :1272-1296 (`MTLResourceStorageModeShared`, `memset` then per-element `float` cast —
no pinned/async concept, unified memory) -> kernel choice :1317-1389 -> chunking :1391-1404
-> optional LB pipeline :1510-1661 -> chunked dispatch loop :1681-1744 -> FP32->FP64
copy-out with `normalize_public_distance` :1756-1761.

- **Kernel choice** is a different, independent policy from CUDA's: `use_banded_row` when
  `0 < band <= 512 && band*20 < heuristic_L` :1328-1331; `use_regtile` when unbanded and
  `heuristic_L <= 256` :1334-1336; `use_global` when
  `3*heuristic_L*4 > maxThreadgroupMemoryLength` :1338-1339. `KernelOverride` is honoured
  here :1344-1364 (unlike the K-vs-N path).
- **Band normalisation**: a non-negative band `>= max_L-1` is rewritten to `-1` before
  device arithmetic :1370-1373 (LESSONS "clamp logical full windows"). CUDA does **not**
  do this for the DTW band.
- **Chunking** exists only in Metal, to dodge the ~2 s macOS GPU watchdog: `cells_budget =
  5e9`, `chunk = max(1, budget/cells_per_pair)` :1399-1404; `pair_offset` is passed as
  `int64` through `detail::pair_chunk_offset` :1682. When scratch is reused
  (`use_global` / `use_banded_row`) each chunk is waited on :1736-1743; otherwise only the
  last command buffer is awaited :1745-1752 (the in-order queue makes this sufficient).
  DELIBERATE, documented at :1391-1398 and :1733-1735.
- **Write-back** is the same dense `N x N` `vector<double>`; the `O(N^2)` copy-out loop
  :1756-1761 is scalar.
- **f32/f64**: Metal has **no FP64 path at all**; `MetalPrecision::FP64` degrades to FP32
  and only says so under `verbose` :1263-1265, :1994-1996.

### 2.3 MPI

`compute_distance_matrix_mpi` :160 -> `ensure_mpi_initialized` :164 (`MPI_Init`, never
finalised anywhere in `dtwc/`) -> `total_pairs = N*(N-1)/2` :168 -> dense
`matrix.resize(N*N)` :174 -> **contiguous linear-index block decomposition** :190-195
(`start_k = rank*pairs_per_rank + min(rank, remainder)`, `local_count = pairs_per_rank +
(rank < remainder)`) -> `#pragma omp parallel for schedule(dynamic, omp_chunk_size(...))`
:216-232 calling `dtwBanded<double>` / `dtwFull_L<double>` :227-229 -> serial scatter into
both triangles :238-242 -> **`MPI_Allreduce(MPI_IN_PLACE, ..., MPI_SUM)` in <= INT_MAX
chunks** :257-267.

The gather is `O(N^2)` doubles of all-to-all traffic per rank regardless of how few pairs a
rank computed — the decomposition is by *pair*, the reduction is by *matrix*. The
load imbalance of contiguous triangular blocks is a **documented deliberate deferral**
`[confirmed mpi_distance_matrix.cpp:182-189]`.

---

## 3. Shared structure with the CPU core

Classification: **byte-identical** = same tokens; **near** = same algorithm, renamed
symbols / different types; **semantic** = same intent, materially different behaviour.

**D1 — CUDA wavefront diagonal loop, x4, near-to-byte-identical.** Independently confirmed
by the repo's own duplicate scan: `4x dtwc/cuda/cuda_dtw.cu:265 | :305 | :1720 | :1756`
`[confirmed .claude/reports/2026-09-07-dupscan.txt:4]`. Pairwise double-buffer :265-301 vs
1-vs-N double-buffer :1720-1754; pairwise 3-buffer :305-333 vs 1-vs-N 3-buffer :1756-1784.

**D2 — CUDA wavefront *launch* config, x2, near (prior review B4: still OPEN, changed).**
`:1225-1281` vs `:2203-2234`. Since 09-02 the buffer-count policy was factored into
`detail::wavefront_buffer_count`, so B4 is *partially* fixed; still duplicated verbatim are
the `max_L > 2048` warn-once block (:1230-1238 vs :2206-2214), `shared_mem = n_bufs*max_L*
sizeof(T)` (:1239 vs :2215), `require_shared_mem_fits` (:1244 vs :2221) and the
`cudaFuncSetAttribute` request (:1247-1251 vs :2224-2228). Block size deliberately diverges
(256 vs 128/256 :2217-2219).

**D3 — CUDA warp + regtile kernels, x2 each, near.** `dtw_warp_kernel` :371-496 vs
`dtw_one_vs_all_warp_kernel` :1796-1880; `dtw_regtile_kernel` :518-716 vs
`dtw_one_vs_all_regtile_kernel` :1884-2021. The only differences are the pair source
(decode vs `blockIdx.y`) and the output index.

**D4 — CUDA FP32/FP64 dispatch, x2, byte-identical modulo `float`/`double`.** :1490-1528 vs
:1529-1567 in `compute_distance_matrix_cuda`.

**D5 — CUDA workspace machinery, x2, near.** `DTWLaunchWorkspace` :913-972 vs
`OneVsAllLaunchWorkspace` :2029-2075 — the same `ensure_runtime` device-change/reset
protocol written twice.

**D6 — CUDA upload/download helpers duplicated inline.** `upload_series_to_workspace`
:1051-1074 is re-implemented inside `launch_dtw_kernel` :1163-1182, and
`download_result_matrix` :1108-1124 inside it at :1291-1292. Consequence: on the LB-pruned
path the series are uploaded **twice** (:1493 then :1179).

**D7 — Metal wavefront kernel body, x4, near.** `dtw_wavefront` :77-190,
`dtw_wavefront_global` :197-299, `dtw_kvn_wavefront` :448-540,
`dtw_kvn_wavefront_global` :543-636. Identical recurrence, identical band clip
(:139-144, :252-257, :503-508, :599-604), identical INF edge stamping (:173-176, :284-287,
:528-531, :624-627); only the address space (`threadgroup` vs `device`) and the pair source
differ.

**D8 — CUDA <-> Metal, semantic.** Both compute the same anti-diagonal recurrence but they
are *not* ports of one another at HEAD:
- CUDA orients rows = short side (`M = min(ni,nj)` :224-225); Metal's wavefront does **not**
  (`La` rows straight from `a_idx` :105-111). Metal's regtile *does* orient :781-784.
- CUDA clips the band per-cell with `fixed_band_contains(i,j,band)` :67-71; Metal clips the
  anti-diagonal *interval* `[(k-band+1)/2, (k+band)/2]` in 64-bit :139-144 and stamps INF at
  the two band-adjacent cells :173-176.
- Metal has two kernels CUDA lacks (`dtw_banded_row` :322, `dtw_wavefront_global` :197);
  CUDA has one Metal lacks (persistent / atomic-counter wavefront :191-205).
- CUDA supports FP64; Metal does not (:1263-1265).
- CUDA's option struct carries `device_id`, `precision`, `lb_threshold = -1.0`; Metal's
  carries `precision`, `lb_threshold = 0.0`, `lb_envelope_band`. The threshold-default
  divergence is **DELIBERATE**, documented as a backward-compatibility freeze
  `[confirmed gpu_dtw_common.hpp:12-15, cuda_dtw.cuh:51-55, metal_dtw.hpp:59-66]`.

**D9 — GPU kernels <-> `core/dtw_kernel.hpp`, semantic.** No GPU file includes any core DTW
header. Divergences that matter:
- Band bounds. CPU: `dtw_band_bounds(band,row,column_count)` clamped once
  `[confirmed dtw_kernel.hpp:191-200]`, plus a *terminal-cell feasibility* pre-check
  `if (n_long - n_short > band_width) return maxValue` :432 and a full-coverage
  short-circuit `band_width >= n_long-1 -> linear kernel` :435-436. CUDA has neither; it
  relies on the last anti-diagonal cell being stamped `INF` by `fixed_band_contains`
  :318-319 to produce the same answer `[inferred — proved for the last diagonal, where
  len_k == 1 and p == 0; the confirming test is the F12 fixed-band oracle
  tests/unit/gpu_fixed_band_oracle.hpp]`.
- No-path sentinel. CPU uses `std::numeric_limits<T>::max()` `[confirmed
  dtw_kernel.hpp:137,169,217]`; GPU kernels hard-code the same value as a **literal**
  (`3.402823466e+38f` / `1.7976931348623157e+308`) in 12 places
  (`cuda_dtw.cu:133-135, 383-385, 533-535, 845-847, 1643-1645, 1811-1813, 1902-1904`;
  `metal_dtw.mm:110, 227, 349, 482, 577, 667, 786, 986`).
- Identity short-circuit `x == y && nx == ny -> 0` `[confirmed warping.hpp detail
  dtwBanded_impl body, 5th line]` has no GPU counterpart; the diagonal is forced to 0 only
  in the host copy-out `[confirmed cuda_dtw.cu:1086,1099]`.
- Cost policy. GPU supports **L1 and squared-L2 only** (`use_squared_l2` :292, :323). Every
  other axis the core supports (ADTW / WDTW / DDTW / Soft-DTW / MSM / TWE,
  `MissingStrategy`, multivariate `ndim`) has **no GPU representation at all**.

**D10 — MPI <-> the OpenMP CPU fill, semantic.** MPI's inner loop is the *only* backend that
calls the real core kernels (`warping.hpp` :116, used :227-229), so its per-pair numbers are
digit-identical to `BruteForce`. But it calls `dtwBanded` / `dtwFull_L` **directly**, not the
bound `Problem::dtw_fn_` dispatcher — so `--variant`, `missing_strategy`, `lb_strategy` and
`MetricType` are all silently ignored `[confirmed mpi_distance_matrix.cpp:226-230]`
(previously logged as `.claude/reports/2026-09-02-review-io-cli.md:35`). The
parallel-for / `omp_chunk_size` / dynamic-schedule shape is duplicated from
`Problem::fillDistanceMatrix_BruteForce`, not shared.

**D11 — `decode_pair` is genuinely single-source.** CUDA `using dtwc::detail::decode_pair`
:63; MPI a thin `size_t` adapter :150-156; Metal prepends `kDecodePairMSL` to its runtime
library source `[confirmed metal_dtw.mm:1040-1042]`. No local copy survives.

---

## 4. Performance-critical structures and rationale — DO-NOT-BREAK

1. **`decode_pair` cost shape.** `row_start` is hoisted so the down-correction is *one
   comparison*, and the clamps are ordered high-then-low `[confirmed decode_pair.hpp:59-75]`.
   Rationale recorded verbatim: "A per-pair decode is device code ... hoisting `row_start`
   makes the down-correction one comparison. Also order defensive clamps so the LAST one is
   the one that must hold" `[confirmed .claude/LESSONS.md ~:1050-1057]`. Never re-derive a
   local copy: the retired int32/float copies produced out-of-bounds pairs
   `[confirmed decode_pair.hpp:12-16]`.
2. **Lock-free `query_gpu_config`.** Acquire-load fast path, mutex only for the one-time
   fill, failed query returns a shared immutable default rather than a reference into the
   slot `[confirmed gpu_config.cuh:316-343, 377]`. Rationale: called once per kernel launch
   per host thread, and `GPUConfig::device_name` (a `std::string`) makes an unsynchronised
   double write a real race `[confirmed LESSONS.md ~:1000-1005]`. Prior review G1 — FIXED.
3. **Warn-once latches are `std::atomic<bool>::exchange`, never `static bool`**
   `[confirmed cuda_dtw.cu:1232-1233, 2208-2209]`. Prior review G2 — FIXED.
4. **Thread-local workspaces with monotonic capacity growth.**
   `get_dtw_launch_workspace<T>` :974-980 and `get_one_vs_all_launch_workspace<T>`
   :2077-2083 are `thread_local`; device buffers only ever grow
   (`ensure_dtw_device_capacity` :982-1001). This is the "buffer > thread_local >> heap"
   rule in `.claude/CLAUDE.md` and the evidence that multi-threaded host dispatch is the
   intended model.
5. **Pinned host memory above 256 KiB, with a non-throwing fallback**
   `[confirmed cuda_dtw.cu:1062, 1115, 1160; cuda_memory.cuh:213-225]` — required for
   `cudaMemcpyAsync` to actually overlap.
6. **On-device pair decode instead of host pair arrays** — eliminates the
   `2 * num_pairs * sizeof(int)` host allocation and its H2D transfer (4 MB at N=1000)
   `[confirmed cuda_dtw.cu:1129-1132, 1473-1474]`.
7. **Kernels write the `N x N` matrix directly** — "eliminating the per-pair distance array,
   its D2H transfer, and the host-side fill loop" `[confirmed cuda_dtw.cu:1131-1132]`.
8. **The `max_L > 2048` three-buffer fallback is a correctness cap, not a heuristic**
   `[confirmed cuda_dtw.cu:156-164; launch_prep.hpp:554-567]`. Reducing it silently drops
   anti-diagonal cells (Task 0.1). DELIBERATE.
9. **Metal's 5e9-cell chunk budget** exists to survive the macOS GPU watchdog
   `[confirmed metal_dtw.mm:1391-1399]`; removing it turns long runs into
   `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. DELIBERATE.
10. **Metal's per-chunk `waitUntilCompleted` only when scratch is reused**
    `[confirmed metal_dtw.mm:1733-1743]` — a correctness serialisation, deliberately absent
    on the threadgroup-memory path (:1745-1752). DELIBERATE.
11. **Metal `dtw_banded_row`'s coalesced scratch layout** `scratch[row_half*W*stride +
    r*stride + gid]` `[confirmed metal_dtw.mm:317-321, 353-354]` — the comment states the
    naive per-thread-strip layout costs a 32x bandwidth penalty. Do not "simplify" the
    indexing. DELIBERATE.
12. **MPI `MPI_IN_PLACE` + INT_MAX chunking** `[confirmed mpi_distance_matrix.cpp:248-267,
    allreduce_chunking.hpp:312-335]` — avoids a duplicate `N*N` scratch buffer and the
    negative-count corruption above N ~ 46340.
13. **CUDA fat-binary arch list `60;70;75;80;86;89;90`** `[confirmed CMakeLists.txt:158]`
    covers P100 through H100; predates Phase 3 `[confirmed
    .claude/baselines/2026-07-07-cuda-first-runtime-verification.md:28]`.
14. **Perf record, stated honestly.** The "H100 14.2x UCR" figure lives in session memory,
    but the repository's own baseline says "H100 (sm_90) perf claims remain **ADVISORY**
    until a real ARC run" `[confirmed .claude/baselines/2026-07-07-cuda-first-runtime-
    verification.md:28]` and the R1 reconciliation records "No real H100 run is claimed"
    `[confirmed .claude/baselines/2026-07-23-r1-todo-reconciliation.md:189]`. The only
    hardware-confirmed device baseline in-repo is the local **RTX 4000 Ada (CC 8.9)**
    kernel-override run `[confirmed .claude/baselines/2026-07-10-cuda-kernel-overrides.md:3-6]`.
    Treat 14.2x as **not established** in this repository.

---

## 5. Layering

**What each backend pulls from dtwc.** `cuda_dtw.cuh` and `metal_dtw.hpp` each include
exactly three dtwc headers — `enums/KernelOverride.hpp`, `error.hpp`,
`core/gpu_dtw_common.hpp` `[confirmed cuda_dtw.cuh:19-21, metal_dtw.hpp:27-29]`. The
translation units add `detail/decode_pair.hpp` (both) and, for Metal only,
`metal/detail/chunk_dispatch.hpp`. MPI adds `../warping.hpp` and `../parallelisation.hpp`
`[confirmed mpi_distance_matrix.cpp:104,116,117,118]`.

**No backend includes `Problem.hpp` or `Data.hpp`.** Confirmed by the full include listing.
The coupling is entirely in the *shape of the arguments*: `const
std::vector<std::vector<double>>&` in, dense `std::vector<double>` of N^2 out.

**Could a backend be a "filler" over an abstract matrix sink + series source?**
- *Series source*: yes, with one real change. Every backend's first act is a flatten-and-pad
  into `[N x max_L]` (`flatten_series_buffer` :1027; `metal_dtw.mm:1278-1282`;
  `upload_series` :2001-2015). Any source yielding `(i) -> span<const double>` plus a count
  would do; `vector<vector<double>>` is not required. This same coupling is what currently
  forbids mmap-backed series, and the guard is explicit and loud
  `[confirmed Problem.cpp:931-940]`.
- *Matrix sink*: **no, not without work.** The GPU kernels write both triangles into a
  device-resident dense `N x N` buffer (`result_matrix[si*N+sj]` and `[sj*N+si]`), and the
  host materialises a second full `N x N` `vector<double>` before `Problem` copies it a
  third time into `distMat`. Three full N^2 materialisations per fill. A sink abstraction
  needs either a tiled/blocked launch (Metal chunks *pairs* but still writes one full
  `N x N` output buffer :1292-1296) or a per-pair callback the kernels do not have.
- MPI is closest to a filler — it owns only `matrix`, `local_ij`, `local_distances` — but it
  too resizes a dense `N*N` on **every** rank :174.

Conclusion: the *series source* seam is nearly free; the *matrix sink* seam is the real
design work, and it is the same seam the 100M-series target needs.

---

## 6. Design problems

Severity and blast radius are my assessment; each item carries one line of candidate action
only.

**P1 · HIGH · silent zero-matrix fallback in Metal — `metal_dtw.mm:1417-1433, 1465-1484`.**
A failed scratch allocation returns an all-zero `N x N` matrix, sets `pairs_computed = 0`,
and prints nothing unless `verbose`. The comment says "falling back to CPU" (:1427, :1478) —
**there is no CPU fallback**; the comment is false. `compute_kvn_impl:2077-2087` is the same
shape for K-vs-N. This is the exact A16 pattern the CUDA side eliminated. Blast radius: any
direct `dtwc::metal::*` caller (Python binding, tests); `Problem` happens to catch the N x N
case via `pairs_computed == 0` `[confirmed Problem.cpp:998-1001]` but not the K-vs-N one.
*Candidate action:* throw `DeviceError` instead of returning, and delete the "CPU fallback"
wording.

**P2 · HIGH · Metal wavefront kernels have no zero-length guard — `metal_dtw.mm:105-111 &
186`, `:222-228 & 295`, `:477-483 & 538`, `:572-578 & 634`.** With `La == 0` (one empty
series among non-empty ones, so the host's `max_L == 0` early return :1258 does not fire),
`K = Lb - 1`, every `diag_len <= 0`, and the final read is `cur[La-1] == cur[-1]` — an
out-of-bounds threadgroup / device read. CUDA guards this explicitly (:228-238, :406-412,
:555-561) and Metal's own regtile body does too (:788-794); only the four wavefront kernels
do not. `[inferred — derived from the indices; [BLOCKED-ENV] for runtime confirmation.
Confirming test: an N=2 Metal fixture with one empty series.]`
*Candidate action:* add the `La == 0 || Lb == 0 -> INF` guard the regtile body already has.

**P3 · HIGH · `int` truncation of the MPI local pair count — `mpi_distance_matrix.cpp:217,
220`.** `omp_chunk_size(static_cast<int>(local_count))` and
`for (int idx = 0; idx < static_cast<int>(local_count); ++idx)`. Above INT_MAX local pairs
the loop bound is wrong or negative and the rank contributes zeros to a `SUM` reduction — a
silent wrong answer. This is the **same defect class** the same file already fixed for
`MPI_Allreduce` (:252-256) and that CUDA fixed with `require_pair_count_fits`. Directly on
the 100M-series path. Previously logged
`[confirmed .claude/reports/2026-09-02-review-io-cli.md:35]`; still open at HEAD.
*Candidate action:* give MPI the same public-entry guard helper CUDA has, or make the
induction variable 64-bit and chunk the OpenMP range.

**P4 · HIGH · GPU envelope window overflows at `INT_MAX` — `cuda_dtw.cu:745-749`,
`metal_dtw.mm:890-894`.** `const int hi = (k + w + 1 < L) ? k + w + 1 : L;` with
`w = band`; the host launchers accept any non-negative `int` radius and never clamp
(`launch_lb_keogh_kernel` passes `band` straight through :1337; Metal's `env_band`
:1512-1515 is likewise unclamped). Signed overflow -> UB. Registered as **PLAN F50, open**
`[confirmed PLAN.md:837-847]` and matching the LESSON "Clamp logical full windows before
device integer arithmetic" `[confirmed LESSONS.md ~:88-93]`.
*Candidate action:* clamp the host envelope radius to `max_L - 1` with a checked conversion,
in both backends.

**P5 · HIGH · the GPU path silently ignores every distance axis except band and metric —
`Problem.cpp:1050-1058, 1079-1082`.** `cuda_opts` / `metal_opts` set only `band`,
`device_id`, `precision`, `use_squared_l2 = false`, `verbose`. Consequences, none of which
raise: (a) `variant_params.variant` — MSM / TWE / ADTW / WDTW / DDTW / Soft-DTW all silently
become plain L1 DTW; (b) `missing_strategy` other than `Error` — the NaN pre-scan at
:891-918 covers only `Error` and `Interpolate`, so `ZeroCost` / `AROW` reach the GPU as raw
NaN; (c) `ndim > 1` — `p_vec[i]` is an interleaved flat buffer `[confirmed Data.hpp:36,48]`
and the kernels treat it as a univariate series of length `ndim*L`; (d) `lb_strategy_` — the
GPU LB pre-pass is never enabled from `Problem` (`use_lb_keogh` is never set). No guard
exists: every `DistanceMatrixStrategy::CUDA|Metal` site — `Problem.cpp:537, 932, 1041,
1070`, `Problem.hpp:107-108`, `api.cpp:106,108`, `dtwc_cl.cpp:1523` — checks none of the
variant, the missing strategy, or `ndim` `[confirmed by git grep]`.
*Candidate action:* reject unsupported variant / missing / ndim combinations at the top of
the CUDA and Metal cases, the way `Pruned + Float32` is already rejected at :984-990.

**P6 · MEDIUM-HIGH · CUDA runtime errors escape the `dtwc::Error` taxonomy —
`cuda_dtw.cu:43-51`, `cuda_memory.cuh:157-165`.** `CUDA_CHECK` / `CUDA_CHECK_ALLOC` throw
`std::runtime_error`, so a `cudaMalloc` OOM or a launch failure is neither a `DeviceError`
nor catchable through `dtwc::Error`. PLAN F31 records this for Metal only
`[confirmed PLAN.md:614-620]`; it applies verbatim to CUDA. The two macros are also
duplicated `[confirmed dupscan.txt:15 — "2x cuda_dtw.cu:45 | cuda_memory.cuh:19"]`.
*Candidate action:* one shared macro throwing `dtwc::DeviceError`.

**P7 · MEDIUM · `-fno-finite-math-only` is added without a language guard —
`dtwc/CMakeLists.txt:198-199`.** `target_compile_options(dtwc++ PRIVATE
-fno-finite-math-only)` applies to *all* languages on the target, including the CUDA source
added at :239-242. The file's own OpenMP block 30 lines earlier uses
`$<$<COMPILE_LANGUAGE:C,CXX>:...>` for exactly this reason :167-168, and the project has a
recorded lesson "**MSVC flags leak into nvcc:** Use `$<$<COMPILE_LANGUAGE:C,CXX>:...>`
generator expressions" `[confirmed .claude/LESSONS.md:306]`. `[inferred — would affect
DTWC_ENABLE_CUDA=ON with a GCC/Clang host compiler on Linux; the confirming test is a
configure + compile of cuda_dtw.cu in such a build. Not reproducible on this MSVC host,
where the outer if(CMAKE_CXX_COMPILER_ID MATCHES Clang|GNU) is false.]`
*Candidate action:* wrap it in `$<$<COMPILE_LANGUAGE:C,CXX>:...>` like its neighbour.

**P8 · MEDIUM · `int32` survivor list caps LB pruning at N ~ 65536 — `metal_dtw.mm:976-982`
and the `pair_indices` buffer at `:86, :207`.** `active_pairs[slot] = (int)pid` truncates,
and the consuming `device const int* pair_indices` is int32, even though `num_pairs` was
widened to `long`. Self-documented as out of scope :978-981. CUDA has the identical
constraint but bounds it: `require_pair_count_fits` refuses N >= 65537 outright; Metal has
no pair-count guard at all.
*Candidate action:* widen the survivor buffer to `long`, or add Metal's missing pair-count
guard.

**P9 · MEDIUM · Metal's availability check runs after the `N <= 1` early return —
`metal_dtw.mm:1241` vs `:1246`.** A 0- or 1-series call on a device-less host returns a
"successful" empty result where CUDA throws (`require_cuda_device` is the *first* executable
check, :1447 before :1456). A small asymmetry, but exactly the class of inconsistency A16
was raised about.
*Candidate action:* move the check above the early return.

**P10 · MEDIUM · magic numbers with no named constant, several of them cross-file
invariants enforced only by comments.** `PRELOAD_THRESHOLD = 512` (:138 and :1668 — two
copies), `DOUBLE_BUF_THRESHOLD = 1024` / `DOUBLE_BUF_MAX = 2048` (:156-162 and :1670-1674 —
two copies, plus a third encoding in `launch_prep.hpp:562-567`),
`PINNED_THRESHOLD = 256*1024` (four copies: :1062, :1115, :1160, :2133),
`block_size = 256` (:1242) vs `128/256` (:2218-2219), `MAX_SI = 8` (:276, :1730),
`PAIRS_PER_BLOCK = 8` (:369) vs MSL `PAIRS_PER_TG = 8` (`metal_dtw.mm:657`) vs host
`kPairsPerTG = 8` (`metal_dtw.mm:1446`, with the comment "must match PAIRS_PER_TG in MSL"),
`cells_budget = 5e9` (:1399, :2066), `band*20 < L` (:1330), `band <= 512` (:1329),
`max(1, max_L/10)` envelope default (:1514).
*Candidate action:* one `constexpr` header shared by host and device for the thresholds that
must agree.

**P11 · MEDIUM · `blocks_per_sm` query result is unchecked — `cuda_dtw.cu:1256-1258`.**
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns an error code that is discarded, and
if `query_gpu_config` failed earlier its `kUnavailable` default gives `sm_count == 0` ->
`persistent_grid == 0` -> a zero-block launch.
*Candidate action:* `CUDA_CHECK` the occupancy call and floor the grid at 1.

**P12 · MEDIUM · ObjC buffers leak on the throwing paths — `metal_dtw.mm:2019-2021,
2029-2031, 2035`.** `newBufferWith...` returns +1 retained objects the surrounding
`@autoreleasepool` does not own; these `throw std::runtime_error` sites release nothing,
unlike the sibling early-return paths (:1420-1423, :2078-2082) which do.
*Candidate action:* an RAII holder for `id<MTLBuffer>`, or release before throwing.

**P13 · MEDIUM · double transfer on the CUDA LB-pruned path — `cuda_dtw.cu:1493` then
`:1167, :1179`.** `upload_series_to_workspace` flattens and uploads, then
`launch_dtw_kernel` flattens and uploads the *same* series into the *same* workspace buffers
again. Pure waste on the path that is meant to be the fastest.
*Candidate action:* let `launch_dtw_kernel` take an "already uploaded" flag.

**P14 · LOW-MEDIUM · the MPI options struct is a two-field stub —
`mpi_distance_matrix.hpp:28-34`.** `band` and `verbose` only; no metric, variant,
kernel-override, or precision. Combined with the direct `warping.hpp` calls (D10) this makes
MPI a semantically different product from every other backend.
*Candidate action:* route MPI through the same options POD the CPU dispatcher consumes.

**P15 · LOW-MEDIUM · MPI never finalises MPI and allocates `N*N` on every rank —
`mpi_distance_matrix.cpp:122-129, 174`.** `MPI_Init` with no matching `MPI_Finalize`
anywhere under `dtwc/` (only the test and benchmark `main()`s call it), and a full dense
matrix per rank makes the replicated-data design the memory ceiling. `local_ij` also stores
24 bytes per local pair :210 purely to replay the decode.
*Candidate action:* register an `MPI_Finalize` owner and drop `local_ij` (re-decode in the
scatter loop).

**P16 · LOW · `Problem`'s GPU write-back is a scalar `O(N^2)` double loop —
`Problem.cpp:1006-1008`.** `m.set(i, j, result.matrix[i*result.n + j])` element by element,
after the backend already produced a contiguous row-major `N x N`. Third full
materialisation of the matrix (device -> host vector -> `distMat`).
*Candidate action:* a bulk `assign_from_row_major` on the matrix types.

**P17 · LOW · SRP.** `cuda_dtw.cu` is 2475 lines holding device kernels, memory management,
occupancy policy, timing, precision policy, the LB pipeline and five public entry points;
`metal_dtw.mm` is 2237 lines of which 933 are a string literal of shader source. Neither
file has an internal seam a reviewer can test in isolation beyond the two extracted headers.
*Candidate action:* split kernels / launchers / entry points per backend.

**P18 · LOW · YAGNI.** `metal/detail/chunk_dispatch.hpp` is a 26-line header wrapping one
`static_cast` `[confirmed :21-24]`. It exists only because a prior tautological test was
rejected `[confirmed LESSONS.md:479]` — the seam is legitimate, but it is the whole file.

---

## 7. Obsolete and dead code, stale comments

- **Stale comment, "falling back to CPU"** — `metal_dtw.mm:1427`, `:1478`, `:2084`. No CPU
  code runs; the function returns zeros. `grep -n "CPU fallback\|falling back to CPU"
  dtwc/metal/metal_dtw.mm` -> 3 hits, 0 CPU calls in the file.
- **Stale doc, `KernelOverride.hpp:8-10`**: "Backends silently fall back to `Auto` when the
  requested path is unsupported". No longer true for CUDA — `select_kernel` :436-439 returns
  `fell_back_to_auto = true` and every CUDA result reports it
  `[confirmed cuda_dtw.cuh:64,104,127]`. It remains true for Metal, whose
  `MetalDistMatResult` carries no fallback bit at all `[confirmed metal_dtw.hpp:76-81]`.
- **Prior review D, "`cuda_dtw.cu:34,36,39` include `<chrono>`, `<numeric>`, `<climits>` —
  unused" — FIXED.** The include block at HEAD is
  `algorithm / atomic / cmath / cstring / iostream / stdexcept / string / type_traits`
  `[confirmed :34-41]`; all eight are used (`std::max` :1258, `std::atomic` :1232,
  `memcpy` :1040, `cerr` :1234, `logic_error` :1283, `is_same_v` :1038).
- **`mpi_distance_matrix.cpp:110` includes `<cmath>`; no `<cmath>` symbol is used** — the
  local `decode_pair` adapter :150-156 delegates to the SSOT header.
  `grep -nE "std::(sqrt|floor|fabs|pow)" dtwc/mpi/mpi_distance_matrix.cpp` -> no match.
- **`CUDADistMatOptions::max_length_hint` was dead (prior review E5) — FIXED.** Now consumed
  via `detail::kernel_selection_length` at all four CUDA entry points
  `[confirmed cuda_dtw.cu:1467, 2301, 2364, 2436]`.
- **`GPUConfig::max_shared_per_block` was computed and never read (prior review C6) —
  FIXED.** Read by `require_shared_mem_fits` `[confirmed cuda_dtw.cu:88-97]`, called at
  :1244 and :2221.
- **`compute_kvn_impl` validates `kernel_override` and never uses it** —
  `metal_dtw.mm:1959`; the pipeline is chosen purely by threadgroup memory at :2061-2063.
  Prior review E5's Metal half: still open.
- **`kernel_used` recomputed after the dispatch instead of recorded during it** —
  `metal_dtw.mm:2158-2161` re-evaluates `3*max_L*4 > maxThreadgroupMemoryLength`,
  duplicating :2059-2061; it is also skipped entirely on the scratch-failure return :2086,
  leaving `kernel_used` empty.
- **`dtw_banded_row` cannot participate in LB pruning** (no `pair_indices` parameter,
  `metal_dtw.mm:322-333`), which is what forces the silent LB disable at :1497-1504.
- **CUDA LB pre-pass with `lb_threshold <= 0` computes envelopes and bounds and discards
  them** — `cuda_dtw.cu:1487-1521` — only `result.lb_time_sec` survives. Not dead code, but
  dead work.
- No dead *functions* found: every kernel and helper in all three directories has at least
  one call site (checked by grep per symbol).

---

## 8. Lock / atomic / stream / synchronisation inventory

| Primitive | Site | Hot or cold |
|---|---|---|
| `std::mutex fill_mtx` | `gpu_config.cuh:327, 334` | **Cold** — one-time fill only; the cache hit is an acquire load :331 |
| `std::atomic<bool> ready[16]` | `gpu_config.cuh:326, 331, 335, 377` | Hot read, once per launch per thread; lock-free |
| `std::atomic<bool> logged` (x2) | `cuda_dtw.cu:1232, 2208` | Cold — gated behind `max_L > 2048` |
| `thread_local` workspaces (x2) | `cuda_dtw.cu:977, 2080` | Hot; no sharing, hence no lock |
| Device `atomicAdd(work_counter, 1)` | `cuda_dtw.cu:196` | **Hot** — persistent-mode work stealing, one per pair per block, thread 0 only, broadcast via `__shared__ int s_pid` :150 |
| Device `atomicAdd(active_count, 1)` | `cuda_dtw.cu:840` | Warm — once per surviving pair, LB path only |
| `__syncthreads()` | `cuda_dtw.cu:198, 236, 250, 283, 300, 332, 350, 1709, 1737, 1753, 1783` | **Hot** — 1-2 per anti-diagonal |
| `__syncwarp()` | `cuda_dtw.cu:427, 577, 1841, 1934` | Warm — after the shared-memory series load |
| `__shfl_sync(0xFFFFFFFF, ...)` | `cuda_dtw.cu:456, 457, 627, 629, 710, 1852, 1853, 1959, 1960, 2017` | **Hot** — the warp / regtile recurrence itself |
| One `cudaStream_t` per workspace | `cuda_memory.cuh:239-247`; `cuda_dtw.cu:936, 2045` | Created once per thread/device; all work serialised on it |
| Two `cudaEvent_t` per workspace, + 2 ad-hoc in the LB launcher | `cuda_dtw.cu:937-938, 1322-1323` | Cold; the LB launcher creates a fresh pair per call (avoidable allocation) |
| `cudaStreamSynchronize` | `cuda_dtw.cu:1122, 1298, 1356, 1375, 1415, 2245` | Six full host-device barriers per pruned fill |
| `std::once_flag` + `std::call_once` | `metal_dtw.mm:1018-1019` | Cold — one runtime shader compile per process |
| `threadgroup_barrier(mem_threadgroup)` | `metal_dtw.mm:126, 178, 494, 532` | **Hot** — 1 per anti-diagonal |
| `threadgroup_barrier(mem_device)` | `metal_dtw.mm:243, 289, 590, 628` | **Hot** and expensive — the global-scratch wavefront |
| `simdgroup_barrier` | `metal_dtw.mm:804` | Warm |
| `simd_shuffle_up` / `simd_shuffle` | `metal_dtw.mm:695, 696, 750` | **Hot** — regtile recurrence |
| `atomic_fetch_add_explicit(active_count, 1, relaxed)` | `metal_dtw.mm:977` | Warm — LB compaction |
| `[cmd waitUntilCompleted]` | `metal_dtw.mm:1576, 1606, 1635, 1737, 1746, 1883, 1916, 2124, 2133` | Per chunk on the scratch paths (deliberate); once at the end otherwise |
| `#pragma omp parallel for schedule(dynamic, chunk)` | `mpi_distance_matrix.cpp:216-218` | **Hot**; explicitly lock-free — "each thread writes only to `local_distances[idx]` and `local_ij[idx]`" :213-214 |
| `MPI_Allreduce(MPI_IN_PLACE, ..., MPI_SUM)` | `mpi_distance_matrix.cpp:260-266` | **Hot and dominant** — `O(N^2)` doubles, chunked at INT_MAX |
| `MPI_Init` / `MPI_Initialized` | `mpi_distance_matrix.cpp:124-128` | Cold; not thread-safe against a concurrent first call (no `MPI_Init_thread`) |

No host-side mutex, atomic, or mutable global exists on the per-pair path of any backend.

---

## 9. Testability

**CUDA-OFF seams that actually execute in the canonical gate.** `cuda/launch_prep.hpp` and
`cuda/kernel_selection.hpp` are outside `#ifdef DTWC_HAS_CUDA` `[confirmed — neither file
contains the macro]`, and `test_cuda_launch_guards.cpp` reaches them through
`#if __has_include(<cuda/launch_prep.hpp>)` with a hard `FAIL` if the seam disappears
`[confirmed tests/unit/test_cuda_launch_guards.cpp:25-30, 42-44]`. The same pattern holds
for `mpi/allreduce_chunking.hpp` (no `DTWC_HAS_MPI` guard, exercised by
`unit_test_mpi_allreduce_chunking.cpp`) and `metal/detail/chunk_dispatch.hpp`. This
satisfies the project rule that a guard which must fire without the optional dependency
lives outside its `#ifdef` `[confirmed launch_prep.hpp:460-465]`.

**Requires hardware.** All of `test_cuda_correctness.cpp` (1860 lines),
`test_cuda_kernel_override.cpp` (390), `test_cuda_lb_keogh.cpp` (357), the device sections
of `test_cuda_launch_guards.cpp` (:97-165), `test_metal_correctness.cpp` (684),
`test_metal_lb_keogh.cpp` (248), `test_metal_mmap.cpp` (115), and `unit_test_mpi.cpp` (219,
needs `mpiexec`). Metal is `[BLOCKED-ENV]` on this host; MPI is OFF by default
`[confirmed CMakeLists.txt:43]`.

**The skip-gate hole is still open.** Every `test_cuda_*`, `test_metal_*`, `unit_test_mpi*`
target is registered by the plain glob loop `[confirmed tests/CMakeLists.txt:1, 125-141]`
and **none** of them appears in any `set_tests_properties(... FAIL_REGULAR_EXPRESSION ...)`
block — the gated targets are all CPU/IO tests (`unit_test_checkpoint_binary`,
`test_lb_keogh_derivation`, `test_lb_enhanced_webb_derivation`, `test_lb_webb_intmax`,
`test_problem_api_2_0`, `unit_test_nearest_medoid_assignment`, `unit_test_distance_matrix_csv`,
`unit_test_problem_encapsulation`, `unit_test_problem_storage_policy`,
`unit_test_deterministic_series`, `test_supply_chain_pinning`, `test_io_readers`,
`test_distance_matrix_csv_contract`, `test_cli_resume_state`, `test_cli_rejects_yaml_config`,
`test_cli_config_formats`, `test_fast_clara_parquet_parity`,
`test_fast_clara_assignment_contract`) `[confirmed by enumerating every
set_tests_properties target in tests/CMakeLists.txt]`. So a GPU-less host reports the whole
CUDA and Metal suite **green via SKIP**. This is prior review E7 and the recurring
finding-F9 pattern, unchanged at HEAD.

**Forcing the negative branch.** `CUDA_VISIBLE_DEVICES=-1` (not the empty string) is the
only way to exercise the no-device contract on a GPU host — recorded in the test itself
`[confirmed test_cuda_launch_guards.cpp:125-131]` and in `LESSONS.md ~:991-999`.

**Untested surface (no test reaches it):**
- The CUDA persistent-mode path (the atomic work counter is only taken when
  `num_pairs > 4*sm_count*blocks_per_sm`).
- The pinned-allocation failure path (`pinned_alloc_nothrow` returning null).
- The `max_L > 2048` three-buffer wavefront route and its warn-once.
- `require_shared_mem_fits` firing (no test constructs an over-large shared request).
- Metal's scratch-allocation-failure returns (P1) — unreachable without OOM injection.
- Metal's zero-length-series wavefront path (P2).
- The MPI `local_count > INT_MAX` path (P3) — untestable at scale, but the *guard* would be
  host-testable exactly the way `require_pair_count_fits` is.
- `ensure_runtime`'s device-change reset (`workspace.device_id != new_device_id`) — needs
  two GPUs.
- MPI with `world_size > 1` in CI (`unit_test_mpi.cpp` has its own `main`; nothing in
  `tests/CMakeLists.txt` launches it under `mpiexec`).
- The GPU + non-Standard-variant / non-Error-missing / `ndim>1` combinations of P5 — no test
  asserts either a correct result or a refusal.

---

## 10. Open questions for the designer

1. **What is the matrix sink?** Three full `N^2` materialisations per GPU fill (device
   buffer -> host `vector<double>` -> `distMat`) is the single largest obstacle to the
   100M-series target. Does the redesign introduce a tiled/blocked launch (Metal chunks
   pairs but still writes one full `N x N`), or a per-block sink callback?
2. **Is `use_squared_l2` the whole GPU cost model, permanently?** If yes, `Problem` must
   *reject* every other variant on the CUDA/Metal path (P5). If no, what is the device
   representation of ADTW penalties, WDTW weights, MSM and TWE?
3. **Does MPI stay a replicated-data design?** `matrix.resize(N*N)` on every rank plus an
   `O(N^2)` all-reduce caps N at roughly 46 k on a 16 GB node regardless of rank count. An
   `Allgatherv` of the local pair values, or a genuinely distributed matrix, is a different
   product.
4. **One kernel-selection policy or two?** CUDA's lives in a CUDA-free header
   (`kernel_selection.hpp`); Metal's is 45 lines inline in the `.mm` (:1317-1389) with
   different thresholds and different override semantics. Unifying them would also close the
   `kernel_override_fell_back` reporting asymmetry.
5. **Should `lb_threshold`'s default stay per-backend?** `-1.0` (CUDA) vs `0.0` (Metal) is
   documented as a deliberate compatibility freeze `[confirmed gpu_dtw_common.hpp:12-15]`.
   Is that still worth carrying?
6. **How do device errors join the taxonomy?** CUDA and Metal both throw
   `std::runtime_error` for runtime failures; PLAN F31 is open. A single `DeviceError`
   -throwing check macro would settle it for both.
7. **Should Metal's LB restriction be an error or a capability?** Today requesting
   `use_lb_keogh` with a regtile / banded-row path silently disables pruning (:1497-1504) —
   PLAN F30. Either implement `pair_indices` in those kernels or refuse the combination.
8. **Where does the pair-count guard live for Metal and MPI?** CUDA has
   `require_pair_count_fits` at every public entry; Metal has none and MPI has none. Is the
   guard per-backend or one shared `backends/limits.hpp`?
9. **Does the redesign keep four copies of the wavefront recurrence?** A `__device__` /
   `static inline` recurrence body shared by the pairwise and 1-vs-N kernels would remove
   D1, D3 and D7 without touching the launch geometry.
10. **What is the authoritative device performance record?** The repository contains no
    confirmed H100 measurement (section 4, item 14); the only hardware baseline is the local
    RTX 4000 Ada. Any "do not regress" number the redesign is held to needs a re-measured
    baseline first.

---

## Verification of the 2026-09-02 prior review

| Item | Status at HEAD | Evidence |
|---|---|---|
| **A15** `num_pairs` truncated to `int` before the LB pre-pass | **FIXED** | `require_pair_count_fits` at `cuda_dtw.cu:1448, 1593, 2282, 2348, 2412` plus `:1148`; helper `launch_prep.hpp:502-510` outside the `#ifdef`; test `test_cuda_launch_guards.cpp:39-78` |
| **A16** all-zero matrix when no device is present | **FIXED (CUDA)**; **still present in Metal in a different form** | `require_cuda_device` at `cuda_dtw.cu:1447, 1592, 2280, 2346, 2410`; Metal throws `DeviceError` at `metal_dtw.mm:1246-1249, 1806-1809, 1973-1976`, **but** `:1417-1433`, `:1465-1484` and `:2077-2087` still return an all-zero matrix on scratch-allocation failure (P1) |
| **A17** `decode_pair` corrects the row in one direction only | **FIXED** | `decode_pair.hpp:72-82` — down-loop then up-loop, both `while`; the comment at :69-71 records why |
| **B4** CUDA wavefront config duplicated verbatim | **CHANGED — partially fixed, still open** | Buffer count factored into `launch_prep.hpp:562-567`; warn-once, shared-mem sizing, `require_shared_mem_fits` and `cudaFuncSetAttribute` still duplicated at `cuda_dtw.cu:1230-1251` vs `:2206-2228`. The "scan lengths x5" half **is** fixed — `detail::scan_series_lengths` called at `:1462, 1604, 2297, 2360, 2432` |
| **C6** `max_shared_per_block` computed and never read | **FIXED** | `require_shared_mem_fits` `cuda_dtw.cu:88-97`, used `:1244, :2221` |
| **E5** CUDA silently ignores `max_length_hint` | **FIXED (CUDA)**; Metal's `kernel_override` half **still open** | `kernel_selection_length` `launch_prep.hpp:546-552`, used `cuda_dtw.cu:1467, 2301, 2364, 2436`; `compute_kvn_impl` still ignores `kernel_override` `metal_dtw.mm:1959` |
| **E7** CUDA device tests are bare `SKIP` with no `FAIL_REGULAR_EXPRESSION` gate | **STILL OPEN** | No `test_cuda_*` / `test_metal_*` / `unit_test_mpi*` target appears in any `set_tests_properties` block in `tests/CMakeLists.txt` |
| **G1** `query_gpu_config` takes a process-global mutex on every call | **FIXED** | `gpu_config.cuh:331` acquire-load fast path; mutex `:334` for the one-time fill only |
| **G2** racy `static bool logged` warning latches | **FIXED** | `cuda_dtw.cu:1232-1233, 2208-2209` — `std::atomic<bool>::exchange` |
| **D** unused `<chrono>` / `<numeric>` / `<climits>` in `cuda_dtw.cu` | **FIXED** | include block `cuda_dtw.cu:34-41`; all eight headers used |
| Also verified as **open** and carried forward from PLAN | F27 (GPU LB uses L1 excess under squared-L2), F28 (Metal narrow LB envelope), F29, F30 (silent option degradation), F31 (device-error taxonomy), F50 (INT_MAX envelope overflow) | `PLAN.md:572-620, 837-847` |
