# DTWC++ 2.0 — PLAN

Branch `design-2.0`. The approved design review (`plans/2026-09-27-design-review.md`, Volkan 2026-09-28) is the
plan: seven phases A–G. Step detail — files owned, dependencies, gates, LOC — is in
`plans/2026-09-27-audit/advisor_exec.md` §2 (rows) and §3 (Windows gate commands) and `chair.md` §4 (waves);
finding ids resolve in `plans/2026-09-27-audit/digest.md` (grep, never read whole). Where the audit files and the
review disagree, the review wins (no `system_check()`, no index macro, the W13 items below).

Marks: ☐ open · ◐ in progress · ☑ done (commit). One implementer per step, never two on the same files, at most
four agents; each proven step is one local commit.

## Proof rule (every step)

Serial `ctest -j1`; `cpp_conformance` digit-identical to the phase base unless the step pre-registers a change;
pytest from a fresh `uv` venv when bindings, readers or defaults change; `matlab_suite` when the MEX changes; the
CUDA build (`build/cuda-verify-0928`, RTX 4000 Ada) for GPU steps, macOS CI for Metal; `git grep` proof for every
deleted name (and `git grep <name> v1.0.0` = 0). Gates: `scripts/check_docs.py --cli <dtwc_cl>`,
`scripts/check_pins.py`, gitleaks in CI. A new gate is shown to bite by a reverted mutation.
A step that changes a hot loop (kernel cell, matrix get/set, `dist_by_ind`, swap or assignment loop) shows the
inner loop's assembly before and after: no call, reload or spill added (CHARTER 2026-09-29).

**Pre-registered result changes** (everything else stays digit-identical): `pam` results (FasterPAM only); the
LR-core seed (same cost, medoids may differ on ties); Soft-DTW summation order (1e-12 relative); the v1
one-argument `init::Kmeanspp` sequence; `Method::MIP` above N = 200 uses the selected solver.

## A — baseline, gates, records

- ☑ S0 `test_config_spellings` strips the CR of Windows text-mode stdout — `f859ab5`
- ☑ W0.1 `check_pins.py` replaces the pin stack; CPMLicenses, benchmark, nanobind pinned to commits — `159c8fe`
- ☑ W0.2 `check_docs.py`: every flag the docs show is in the live `--help`, plus the harness self-check — `2db972d`
- ☑ W0.3 gitleaks in CI; Arrow-without-Parquet is a SKIP case in `test_io_readers`; literal-count markers cut
  (the D2, D3, F57 markers wait for W6d) — `0e47d37`
- ☑ W1.1 records floor: reports, specs, archive, old handoffs, uncited baselines, −31k lines — `80c4dcb`
- ☑ W1.2 PLAN, MAP, DECISIONS, LESSONS, runbook rewritten; the design document folded into MAP and DECISIONS — `dac62a7`
- ☑ CHARTER carries the 09-27 instructions verbatim — `0068386`
- ◐ HEAD baseline: conformance digits and three benchmarks (CPU fill, GPU fill, PAM swap) → `baselines/` — waits for a
  quiet machine (COMSOL runs on this box; no timing under load)
- ☑ CUDA build proven in `build/cuda-verify-0928`: vcvars64 + explicit `CMAKE_CUDA_COMPILER` + `-allow-unsupported-compiler`
  (recipe in `plans/2026-09-27-audit/phaseA_measurements.md`); 4 / 4 CUDA tests pass; `--device gpu` byte-identical to
  CPU on the fixture. MSVC findings: 9 squared-L2 cross-path `==` checks differ in the last bit
  (`test_problem_metric:109,322`, `test_run_resolution:148`) → compare with a tight relative tolerance;
  `bench_openmp_schedule` needs the OpenMP 3.0 API that `/openmp:experimental` lacks → delete it (W4); both done (Z1 `a238336`)
- ☑ Q2 not mio (DECISIONS §3, 2026-09-28): llfio stays, header-only without the superbuild if the spike proves it
  builds in wheels, else Boost.Interprocess; files created non-sparse (1.8× random reads on Windows)
- ☑ Q3 OneBatchPAM and CLARA both stay: a measured trade-off (DECISIONS §3, 2026-09-28)
- ☑ llfio header-only spike: pinned header downloads, no quickcpplib bootstrap; wheel build with llfio ON — `78af336`
  (lands with W5d; DECISIONS §3)
- ☑ `build/` reports Arrow ON but builds without it (Arrow not found with clang on Windows): Parquet tests run only in
  `build/arrow-pyarrow-23` until W14b makes an unhonoured `ON` a configure error. That tree finds Arrow through the
  shim `build/arrow-pyarrow-23/pyarrow-config` over `.venv`'s pyarrow 23.0.1; an Arrow gate counts only if
  `ctest -N` lists `test_io_readers` (W14b: an unhonoured ON stops the configure; build/ now configures ARROW=OFF)

## B — deletions (W2, W3, W5, W6)

- ☑ W2a A/B `lb_webb_symmetric` vs Keogh in TADPole at N = 200; Webb stays only if it prunes ≥ 83 %
  (X1 `ba5546f`: FALSIFIED, both prune 78.0 %; Webb deleted)
- ☑ W2b delete the Pruned fill, `LowerBoundStrategy`, the bindings' pruning options; Auto = brute force
  (X1 `872349d`, `fe06efd`)
- ☑ W2c bounds → `compute_envelopes`, `lb_keogh`, `lb_keogh_symmetric`; the LB tests merged (X1 `b1d6514`)
- ☑ W2d derivation 03 and the GPU LB docs follow (X1 `afdfc50`, `5aff090`; merged `227c956`)
- ☑ W3a config keys `benders`, `max-benders-iter`, `batch-weighting` go (goldens 49 → 46) (X2 `104158c`; merged `4de2ce9`)
- ☑ W3b delete Benders, PDLP, `DTWC_HIGHS_GPU` (X2 `6e650f9`; merged `4de2ce9`)
- ☑ W3c delete CLARANS, `PAMVariant`, FastPAM1, public `fast_pam_swap`, `medoid_utils`; FasterPAM only;
  `OneBatchWeighting` goes (NNIW stays the one weighting) (X2 `3bdb5d8`; merged `4de2ce9`)
- ☑ W3d `decode_assignment` + `Problem::set_result` replace `solution_transaction` and `warm_start` (X2 `a85682b`; merged `4de2ce9`)
- ☑ W3e `Method::MIP` uses the selected solver at every N (X2 `6e650f9`; merged `4de2ce9`)
- ☑ W3f unclustered `silhouette()` and `batch_size < k` → `InvalidInput`; score aliases go (X2 `8995ad7`; merged `4de2ce9`)
- ☑ W3g decisive test: duplicate series `{a,a,b,c}`, k = 4; fix only what it falsifies (X2 `0e583e8`, `9edfad2`; merged `4de2ce9`)
- ☑ W5a delete `--resume` / `--restart` and the binary result checkpoint (Y1 `469e545`, `3ff7469`, `97bb54f`)
- ☑ W5b delete `StoragePolicy`, `.dtws`, `MmapDataStore`, CRC32, the auto-spill; `load()` = heap (Y1 `46b3a89`)
- ☑ W5c `Env` → two free functions over a static `{Device, int}` (Y1 `7ba0b4c`, `07b0ea4`)
- ☑ W5d one `.dtwm` file (magic, version, N, SHA-256, packed doubles); the mapped cache is the checkpoint;
  identity mismatch → `InvalidInput`, malformed → `IOError`, absent → fresh; llfio header-only (`78af336`) confined to
  one `.cpp`; Python `distance_matrix()` on a mapped `Problem` fixed; `DTWC_ENABLE_LLFIO=ON` in the wheel and release
  builds (the MEX waits for F43, W6e/f) (Y2 `a539d5d`, `75a65a4`, `6a96642`, `6d0f0c3`, `1d8c82a`; merged
  `959dc5b`; Dense and Mmap are one DistanceMatrix)
- ☑ W5e Parquet saturating helpers and leaf guards → asserts (Y1 `3000ca9`, `d8dd4f7`; merged `7eb928b`)
- ☑ W6a `index_t` alias in `base/settings.hpp`; every count guard deleted; `mip/index_guard.hpp` → two inline throws
  (Y3 `17843f4`, `36c7883`; merged `b260415`; outside algorithms/ and mip/; X2 `a587964` carries their part)
- ☑ W6b enum validator tails → `-Werror=switch` (X3b `4903d8e` re-applies X3 `8c73029` on W4d; merged 6e346a7; clang -Werror=switch and MSVC C4062 both break the build at the same six switches on a dummy enumerator)
- ☑ W6c `run_openmp` captures failures in per-thread slots (no critical, no atomic); `parse_ram_limit` shrinks;
  `GpuPrecision{Auto, FP32, FP64}` (Y3 `36c7883`, `dc55d13`, `b88ae0c`, `119216d`; merged `b260415`)
- ☑ W6d `cluster_by_kMedoidsPAM` shim restored; non-v1 root forwarders, D2/D3/F57 markers, tracker ids in
  comments go (Y3 `9c0b8cb`, `94ae95f`, `8b48caa`, `119216d`; merged `b260415`; tracker ids in comments move to a
  later sweep)
- ☑ W6e never-released Python and MATLAB aliases and the bindings of deleted surface go (Python half: W6e 6f5048c, 7f6a46a; merged fe1bf9b; MATLAB half W6m 8cde5c1, fef03d7, 289cc0f, dcc6c9a, 7cb8cf8, 4ee4d19, 883e5cd; merged e053594)
- ☑ B1 the Python binding checks every index it passes into unchecked C++ (`series`, `series_name`, `centroid_of`); `clusters_ind` / `centroids_ind` read-only, `set_result` bound as the write route (B1 2f8dc96; merged 4968d44)
- ☑ B2 the clustering outputs stay empty until a clustering writes them and one `require_clustered` guards every whole-result reader (a crash before); `set_n_clusters(k < 1)` and `set_band(b < -1)` are refused (B2 0765ec7, 974a4a4, d64c1d1, 463d642, 03fc6a3; merged e0085fd)
- ☑ B3 `fast_pam` `max_iter = 0` is BUILD only in every language, a negative count refused once in C++; `set_data` / `set_view_data` clear the clustering; dead `Nc` guards go (B3 049126e, bb2db4c, c814845; merged 65edcf7)
- ☑ W6f C++ tests of deleted surface trimmed (W6f f125189, 5958063, e9ad125, 88c4a73, 78b27ef, f2c959e; merged ce34bf8; test_problem_api_2_0 one 25-row table; three probe files deleted)
- ☑ `test_run_resolution` runs MIP and LR-core on the CPU without a HiGHS guard: 2 of 7 cases fail in a build with
  `DTWC_ENABLE_HIGHS=OFF` (as `build/arrow-pyarrow-23`); guard them (Y3 merge report) (G1 `0c695c6`; merged 1bb9413; only MIP needs HiGHS — LR-core is exact without it; the GPU branch of the test is unproven without a CUDA build lacking HiGHS)
- ☑ Race-free sweep (DECISIONS §2 rule 6), after X2: failure capture in `fast_pam`, `fast_clara` and
  `one_batch_pam` through `run_openmp`; the FasterPAM and TADPole reductions → per-thread slots combined serially;
  OneBatchPAM's warning mutex → a serial warning. Proof: TSan in WSL (LLVM libomp + Archer) (R1 1e03509, 5f0b65a, 091ea8b, 2a79aa9, 45a7cd8; merged 1ea8327; TSan with LLVM libomp + Archer in WSL: 0 reports at base and head, controls bite; `baselines/2026-09-30-r1-tsan.md`)
- ☑ K1 (2026-09-29) the DP cell makes no library call: `std::min({…})` is `__std_min_d` on the MSVC STL, 7.2 ns/cell
  vs 1.06 on the Mac; nested two-argument min, `dp[i-1, j]` carried in a register; digit-identical
  (`baselines/2026-09-29-windows-kernel-msvc-stl-min.md`) (K1 `8bd6881`, `123146b`, `d114677`, `f705329`; merged
  `4441969`); the fill band is FALSIFIED — the unbanded fill runs the EAPruned kernel, which made no call; P1 measures
  lanes against it
- ☑ Y4 `bindings/matlab` and `tests/matlab` follow Y1, Y2, Y3 and X2 in one unit, then `matlab_suite`; `dtwc_mex` does not
  compile since Y1 (`7eb928b`: deleted checkpoint and storage-policy functions) (Y4 a46b0f7, ac311c4, aa7487b; merged 93eefe1; matlab_suite 119/120 + 1 allowed incomplete on R2024b and R2025b)
- ☑ P1 lanes in the CPU fill, after K1 and Y2: `dtw_kernel_lanes<T, W, Cell>` beside `_linear` / `_banded`, W one cache
  line of T; the fill steps a row by W columns of equal length, per-pair kernel otherwise; Standard DTW, L1 and
  squared L2, full and banded first; bitwise equal to the per-pair fill; band ≥ 2× on the 24-thread fill
  (P1 `94ef14b`, `e34c37f`, `940cd8a`, `3c95dc7`, `b930ff8`; merged `d61c499`; fill 14.5× unbanded, 5.1× band 50,
  15.3× ECG5000; cl builds get unpacked lanes — the Windows wheel is built by cl)
- ☑ P3 unbanded per-pair Standard DTW runs the linear kernel and EAPruned goes: after K1 the linear kernel is
  1.5–3.6× faster on 7 of 7 UCR datasets, bitwise equal (`baselines/2026-09-29-p2-eap-vs-linear.md`) (P3 6587cda, 9e64074; merged 5e15459)

## C — GPU to one fill (W4 + W13's GPU half)

- ☑ W4a kernel A/B through `KernelOverride` (warp vs regtile, 2- vs 3-buffer), ±5 % band registered first: no variant
  within 5 % of its replacement, every kernel stays (`035e70d`, `baselines/2026-09-29-w4a-cuda-kernel-ab.md`)
- ☑ W4b delete MPI (Z1 `b3041d5`)
- ☑ W4c delete the 1-vs-N / K-vs-N kernels and GPU LB_Keogh (Z1 `f2b1cfe`, `8fbb7c9`; merged `741ca3b`)
- ☑ W4d CUDA: `KernelOverride` and fallback flags go; `gpu_config.cuh` reads attributes once at bind (sm_120 FP64
  fixed; its mutex and atomics go); FP32 L = 4095–4096 "invalid argument" fixed (the 48 KiB check ignores static
  shared memory; W4a) (W4d 4479ec0, 3a1e54e, e4a1d77, d46ac35, f22155f, 6b83682, d62c417, c3e24ae, 350ae39, a14454b; merged febd25f)
- ☑ W4e Metal: one pipeline helper, one wavefront body, the dead pair-index plumbing gone, every buffer and the
  autorelease pool owned by one holder so each exit releases once (the out.resize bad_alloc and five other throw paths
  leaked; a leak probe went from 192 MB to 0); scratch failure was already `DeviceError` (W13a). Merged `6cb0c3c1`
  on the Mac (test_metal_* 2168/16 and 169/5 assertions unchanged; CLI outputs byte-identical on every kernel route;
  `baselines/2026-10-05-macos-design-2-0.md`). Its review found the regtile and threadgroup-wavefront routes checking
  only the last chunk's command buffer, so a failed earlier chunk left its pairs at 0: every chunk is checked since
  `bc9469fd` (CHANGELOG)
- ☑ W13a one `fill()` TU; the GPU writes the packed matrix; CUDA launches chunk on an int64 pair offset (W13a e9216f4, ccc07db, c67dd3f, 71a26b6, 08db34d, 1affff2, ace7f07; merged 62d5822; no fill.cpp — the fill was already one function; FP32 L 100 fill 0.756× base time, host memory at N 20,000 L 1000 6.2 → 1.6 GB)
  (the N ≤ 65,536 refusal goes); a backend refuses before the N×N matrix is allocated — today `Problem` resizes first,
  so a huge N on a host without a GPU hits bad_alloc before DeviceError (W4d review)
- ☑ GPU assignment for CLARA (rectangular medoids × series on the pairwise kernels) — Q4: in 2.0, after C (GC 870a3cde, 7a65d88c, be6a7999, ab6c1f08, af885643; records aa114aa9, 808cf182, fc5f387d; merged 63415b4f; entry at the fill's rate; 6.6x FP32 / 1.6x FP64 vs the 24-thread CPU assignment; FP64 equal to the CPU)
- ☑ CUDA tuning, each behind its own band (W4a): a separately compiled preload wavefront for L 257–1024 (−15–17 %
  FP32 measured); a 64 KB carveout above L = 2048 (−18 % at L = 2049) (preload landed, C1 15b9143: FP32/FP64 L 257–512 at 0.82–0.89 of base; the carveout passed its band but needs the CUDA 12.5 API and the floor stays CUDA 12.0 — ARC loads 12.4 — so it was reverted, b803e47)
- ☑ CUDA: the global wavefront above L 2048 where fewer than 3 blocks fit an SM (C1's probe: FP32 L 6000/8000 at 0.64/0.66 of the shared route, FP64 L 2049–4000 at 0.76–0.84) — its own band; the Shared kernel's unreachable preload branch goes with it (C1) (C2 63d5d0a; merged dd33a0e; FP32 L 6000–8446 at 0.64–0.68 of base, FP64 L 2049–4223 at 0.76–0.84; deleting the Shared kernel's preload branch FALSIFIED — it raises occupancy and slows FP32 L 513–2757 by 3–23 %, so the branch stays)
- ☑ CUDA: the FP64 Shared kernel at 4 blocks per SM (C2 lead) — its own band (C3 f8093c6; merged 370c991; FP64 Shared 79 → 62 registers; FP64 L 513/768/1024 at 0.843/0.885/0.881 of base; FP32 SASS byte-identical)
- ☑ CUDA: C2's route rule ignores shared memory's 128-byte allocation unit, so FP32 L 2751–2757 take the shared route at two blocks per SM (C3 lead) — its own band — FALSIFIED (C4 76c4542, 667bbc3; merged 52fb6ce: L 2751–2757 at 0.945–0.963 of base against ≤ 0.95; the patch is kept)
- ☑ CUDA floor: compute capability 8.0 (the A30's generation, Volkan 09-30); older devices get a typed DeviceError before any allocation (C1 c445089)
- ☑ CPU floor x86-64-v3 for release archives and wheels (Volkan 09-30); one `DTWC_ARCH_LEVEL` (native | v3 | v4) (V3) (V3 165e48d, 41a1be1; merged 169db40)
- ☑ cross-route checks (lanes vs per-pair) within a path-length bound; each compiler keeps its contraction (Volkan
  10-01); the GCC-only test failures explained (V4) (V4 0721563, 52b557a, ba26ec9, cf03bd5, 499efdf; merged d6a9d54; GCC 13.3 v3: 113 / 0 failed, conformance identical)
- ☑ ARC scripts follow the CUDA floor; a build on a GPU node is native (S1) (S1 5cc52a0, 0268df1; merged c54e375; htc-gpu 80;86;89)
- ☐ lead: 16 double lanes to hide the min-then-add latency (V3: x86-64-v3 vs SSE2 ~1.0× unbanded, 1.17× banded) —
  its own band; on the M5 alone 1.0–1.19×, with an `fminnm` min 1.41–2.00× (After G, AArch64 lanes)
- ☑ CUDA: `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` is process-wide, so two threads filling at different long
  L can shrink it under each other's launch; set it once to the opt-in maximum less the static bytes, behind a band (W4d) (350ae39)
- ☑ CUDA has no global-memory wavefront: FP32 L > 8446 and FP64 L > 4223 are refused on sm_89 (typed), so `data/dummy`
  (L 9406) cannot run with `--device gpu`; the 8K-sample target fits FP32 only (W4d) (C1 9cc754a, 60e2ec7; merged ceb7f91; FP32 L 10,000 82 Gcell/s vs the CPU fill's 39, L 20,000 73 vs 34, inferred under load)
- ☑ `launch_dtw_kernel`'s warn-once latch prints to stderr above L 2048 whatever `verbose` says; a library does not
  print unasked (W4d) (6b83682)

## D — `index_t` in public counts (W11)

- ☑ W11a `Problem`, `Data`, `ClusteringResult`, `Config`, algorithm signatures and loops; `[[deprecated]]`
  `set_clusters(std::vector<int>)` for one release (W11a 5cb6693, 28cb2eb, e84ef04, 77e4227; merged cfeac4b; CLI outputs and conformance byte-identical; PAM swap counters identical)
- ☑ W11b Python `np.int64`, MATLAB double 1-based labels (W11b d3f5aaf, 9d9233d, 07cbf78; merged 8a9db51)
- ☑ W11c the last 32-bit guards go — Python `_CLI_INT_MAX` / `_CLI_UINT_MAX` / `_CPP_INT_MAX` and the slurm_remote.sh range checks; nanobind and dtwc_cl's parser refuse what they cannot hold (W11c 8a0d5e4, 35d9de7, 9ca7ecd, 8e39b96; merged d8d831d)

## E — interface (W7 → W8 ‖ W9 + W10)

- ☑ W7a `DistanceConfig`; `set_distance / set_band / set_metric / set_variant / set_missing_strategy`
  invalidate the matrix; `bool filled_` — MATLAB cannot set `msm_c`, `twe_nu`, `twe_lambda` today (W6m); the same setters clear the clustering too (B3) (E1 2645fbc, 3c15a4e; merged 770816e)
- ☑ W7b `resolve_dtw_fn(const DistanceConfig&)`; O(1) `dist_by_ind`; the preflight machinery goes. Acceptance:
  `dist_by_ind`'s parallel read path has no critical, atomic, validation flag or lazy allocation; a method that
  needs the matrix prepares it serially at entry (E1 2645fbc, 3c15a4e; merged 770816e; PAM swap 5.6–5.9× faster, no lock or atomic left in dtwc/)
- ☑ W7c one `validate(DistanceConfig)`; `core/dtw.*`, `DTWOptions`, selector validation go (W7c 5bcd7bd, 78dd73f, 1afa573, 143f138; merged 6a44e4d; +517 / −1,927; the metric rule is the facade's)
- ☑ W7d one orientation helper replaces the copied preambles (W7d 23b1978, a031ea1, 9e15eb4, ab72231, 243a715; merged 1ae4a30; −972 / +377; 26 preambles → core::orient + core::run_dtw; no per-cell Cost reloads in any build)
- ☑ W7e WDTW weights at bind; Soft-DTW on the linear kernel; Interpolate thread_local buffers (WDTW weights at bind done in E1); the mutable
  `distance_matrix()` overload gets its own name, so a reader cannot clear `filled_` by accident (E1) (W7ef cb0e20a, c275534, a09302a, fb8f1e6; merged c37e267; Soft-DTW peak 3,431 → 14 MB at 8 × 8,000 samples; Interpolate allocates nothing per pair; the mutable accessor is writable_distance_matrix())
- ☑ W7f dead NaN functors and public helpers go (W7ef ac248dd, 441e64a, 1f7ce19; merged c37e267; TimeSeries/View, softmin_gamma, the DDTW pointer overloads and the full-matrix kernel gone; core::validate refuses an out-of-range enum)
- ☑ W7g one `distance::dtw` per language (W7g fdb005d, f2f48bf, e597f19; merged 601748f; −1,672 / +685; both DTWClusterings ask C++; Problem.set_distance in both bindings)
- ☑ W13b one finite scan at each matrix intake; read-only loops lose per-lookup checks (W13b 5605f4e, 3599ebc, a7f5af1, 8895c95, 7821cd7; merged 6605829; find_best_swap 27 → 21.5 instructions per lookup; Lloyd's assignment keeps its check: a fill of ±DBL_MAX series gives +inf)
- ☑ W8a one reader entry (`read_data`); Parquet names and IPC nulls fixed; `load('x.parquet')` in Python (W8a 3dfbffe, 8b5bc1c, 34b031b, fcf6911, 292acc9; merged 761346a; −996/+749; Python reads Parquet through the installed pyarrow, Volkan 10-01)
- ☑ W8b one writer (`write_result_files`); `Result::save` after streaming fixed (W8b d17f498, e65d6e2, ec47aef, 5bf7422, 9cdea99, 26b0409; merged 862a08f; one open/close pair, one write_result_files; a streamed Result::save writes series_<i> labels and refuses the matrix with InvalidInput, where it read an empty Data)
- ☑ W8c Python and MATLAB `compute_distance_matrix` through `Problem` (the binding's own failure-slot loop, which rethrows by thread number, goes with it — R1) (W8c 3c412c4, b79bbea, db49209, abfa711; merged 1f50070; matrices byte-identical in 23 configurations, about 3× faster; an infeasible band, a band below −1 and an empty series are InvalidInput on every device)
- ☑ L1 measure what the wheel and the MEX link (linker maps) → L2 split `dtwc_core` (no file formats, no CLI) from
  `dtwc_io` and the CLI; the bindings link the core; Python reads and writes files with numpy/pandas/pyarrow, MATLAB
  with its built-ins; v1 Python `DataLoader` / `write*` stay as thin Python (Volkan 10-01) (L1 1f951e1, record:
  HiGHS 75–78 % of each binding, CLI + readers ~2 %, two edges pull them in). L2a: each dtwc/ source folder lists its
  own files (Volkan 10-01; L2a 87b8e88, 15d87f2; merged 1110fe10; compile commands and executable link lines identical in
  every tree; dtwc++.lib's member order follows the folders); Python reads `.arrow` through pyarrow too. Python half done
  by W9b (5e710ba4 … 8ec61075, row W9b; merged 4265e3a6): the wheel's link map (W9b record) holds no `run`, `api` or `config`
  object (CLI11, fkYAML), and Python reads text through the bound C++ reader and writes through the C++ writer (Volkan
  10-02, not numpy/pandas), Parquet and Arrow IPC through pyarrow; the MATLAB half by W9e (merged bf82dc1b). L2b (12b3bce3, 2b1e384d,
  95b10a20, acf86f93, 5b1d3cd0, c8e47e67, d33ed830; merged on the Mac): `dtwc_core` / `dtwc_cli` / `dtwc_io` (STATIC;
  the last in a build with Arrow) behind an INTERFACE `dtwc++`; every folder lists its headers in a FILE_SET HEADERS
  with dtwc/ as the base (40 unlisted headers listed, each with an includer); the core reads text only, dtwc::run
  dispatches Parquet/Arrow IPC to `io::read_arrow` (one if) and streams through `algorithms::fast_clara_parquet`;
  nanoarrow stays in the core (Python's Arrow C data interface); compile commands identical after the file sets
  (254/254), the MEX and the extension byte-identical after the split, +704/+624 bytes after the reader move; with a
  pyarrow-25 Arrow shim the MEX and the wheel link no Arrow (byte-identical to the Arrow-off ones) and 16
  Parquet/Arrow CLI runs match base; `fast_clara` refuses a Parquet stream request (its review); Windows Arrow-ON and
  Linux link order unproven
- ☑ M1 Python solves the MIP with the user's highspy (optional extra; the wheel drops HiGHS); the MEX keeps HiGHS
  linked (CI MEX: HiGHS ON, Gurobi OFF); the model leaves C++ as arrays for Python (Volkan 10-01) (M1 a6de1c71,
  6aa7048c, caec354d; merged c0580948 on the Mac; one builder `dtwc::mip::build_p_median_model` (row-wise arrays,
  solver_types.hpp and the triplet sort go) serves linked HiGHS and highspy, dtwc_cl's MIP outputs byte-identical; the
  extension 5,034,320 → 1,203,728 bytes, the `mip` extra is highspy>=1.8 (the first with resetGlobalScheduler), a
  missing highspy is SolverError naming `pip install "dtwcpp[mip]"`; the MEX links HiGHS statically (4,557,072 bytes);
  Gurobi's builder untouched and not compiled on the Mac; pytest 973/12/0 with the extra, 969/16/0 without; its
  adversarial review's five findings merged bb35d337: DTWC_REQUIRE_HIGHSPY fails a skipped highspy case in CI, the CI
  MEX job asserts test_cluster_mip ran, THIRD_PARTY_LICENSES corrected; open: the wheel's lrcore runs the subgradient
  root, not Kelley (needs linked HiGHS) — measured `3cb49bc8`: same answers everywhere, the subgradient root certifies
  more noisy-DTW roots and Kelley wins only on a line metric; recommendation accept, Volkan rules. Candidate: OpenMP on
  `mip-solvers` (the LR phase is serial in every build; 3.6–5.6× at N ≥ 800 in a scratch build))
- ☑ a MEX built with Gurobi ON needs gurobi130.dll (38.7 MB) to load: delay-load it, or Gurobi OFF for MEX builds (W14b: Gurobi defaults OFF; the default MEX imports no Gurobi DLL; an explicit ON needs Gurobi's bin on PATH)
- ☑ W9a `Method` nine values; `ClusterMethod` goes; `run()` = apply, load, cluster, write; v1 CLI aliases (W9a 2cabc18, adb9031, 6053239, 151f14e; merged 357d76e3; k required: v1.0.0 with no --Nc exited 0 having clustered nothing; name = the input stem; C++ method default auto; 23 v1 spellings warn once; --Nc i..j refused)
- ☑ W9b Python on `run(Config)`; one `DTWClustering` (matrix once, `score` never refits); `variant_params` / `cuda_settings` return read-only objects, so a nested write raises instead of
  silently editing a copy (E1) — Python `DTWClustering` refuses `max_iter = 0` like `sklearn.py` and MATLAB (B3) (W9b 5e710ba4, 4b0c6710, da4fdb09, 7ab98ce0, 7a00b225, d6129e1f, 55f2d3eb, 98190ae4, 29c7507b, f3a552d4, 49e443f6, 004d9b38, 9d186b46, 8ec61075; records 7b397a95, c689a4ac, 974ca3cc; merged 4265e3a6; `cluster()`, `DTWClustering`, `load()` and `set_data` run Config → apply() → `Problem::cluster()`; Python reads text through the bound C++ reader and writes through the C++ writer (Volkan 10-02), so a Ctrl-Z is refused, not an end of file; pytest 962 / 20 skipped / 0 failed against base 1094 / 19 / 0 (199 ids removed, 68 added); the CLI outputs byte-identical to base)
- ☑ W9c `hpc` → `job.toml`; the positional transport goes; `device=hpc` takes `gpu_device=` (a100, a6000, l40s, h100, …) to pick the
  SLURM GPU and the build's CUDA arch; a build on the target node detects both itself (Volkan 09-30). S1 notes:
  `slurm_remote.sh build` submits with no `--gres` (always portable); `jobs/gpu_test.slurm` and
  `ucr_benchmark_gpu.slurm` ask for any GPU and can land on a refused V100; ARC documents `gpu:<type>:<n>` (P100,
  V100, RTX, RTX8000, A100) and constraints `gpu_sku`/`gpu_gen`/`gpu_cc`/`gpu_mem`/`nvlink`, no type for RTX A6000,
  H100 or L40S
  (W9c e9d54d8b … 3514c7e8, 17 commits; merged 834b7904 on the Mac: Python writes job.toml (the keys given, in
  dtwc_cl's config_value grammar; C++ apply() checks every value first), `submit-job <rundir> [--gpu | --gpu-device
  <type>]`, one GPU table `_slurm/gpu_devices.txt` read by Python and bash (a100 by gres type; a6000/l40s/h100 by
  `gpu_cc:` constraint [inferred]; below the 8.0 floor refused), `build --gpu-device <type>` (CUDA native, CPU
  portable), `smoke.slurm MODE=`; pytest 970/12 → 922/12 reconciled by id, 53 bash-driven cases ran; the ARC leg is
  Volkan's: six commands in `baselines/2026-10-06-w9c-hpc-job-toml-mac.md`)
- ◐ W9e MATLAB on the `run(Config)` MEX route; `cmd_cluster_legacy` and snake_case keys go here (DECISIONS 09-30); MATLAB
  regains read access to band, verbose, max_iter and n_repetitions under the Python names, and its own metric lists
  (`DTWClustering.resolve_metric`, `validate_metric.m`) give way to the C++ table (W6m) (done in W7g) (W9e 9230bd5e,
  1000f8d0, d4c43410, 8dccdc68, bebcb149, 43fe390e; merged on the Mac: `dtwc.cluster(data, k, Name, Value)` and
  `DTWClustering` set a dtwc::Config field by field, apply() checks it, Problem::cluster() runs; CamelCase keys are
  Python's words; the MEX links no run() or api (link map), text through the bound C++ reader, Parquet through
  parquetread, Arrow IPC refused; matlab_suite 140/139 → 142/141 reconciled by name, the 25 CLI runs byte-identical;
  ◐ until Windows R2024b runs matlab_suite — the brief `plans/2026-10-06-briefs/W9e.md` stays until then)
- W9e note: MATLAB keys are CamelCase throughout, the same words as Python's snake_case (Volkan 10-01: "camelcase
  and snake case can change between languages"); `dtwc.cluster`'s `band`, `max_iter` become `Band`, `MaxIter`
- W9b/W9e note (Volkan 10-01, lighter bindings): Python and MATLAB share the Config names, not the CLI's file
  pipeline — `run(Config)` reads and writes files, which stays with the CLI (L2)
- ☐ W9f Python test and example trims
- ☑ W10a `DistanceMatrixStrategy`, `CUDASettings` → `set_device` + `set_gpu_precision`; the fingerprint
  hashes the resolved backend (W10 8124528, 373c039, d99734a; merged 73d7361; the cache identity hashes the computed precision, not the device: a CPU FP64 cache serves a CUDA FP64 run, GPU 0 and GPU 1 agree, FP32 is refused by FP64; Metal's Auto is FP32; CUDA's Auto is refused for a persistent cache)
- ☑ W10b `gpu_available()` / `gpu_info()`; `gpu:1` on Metal refused; `system_info`, `check_system`,
  `*_AVAILABLE` go (`test.parallelisation()` / `test.gpu()` stay the diagnostics) (W10 3d9e6c4, a75973c; merged 73d7361)

## F — tests to their oracles (W12)

- ☑ W12a unrun and wave/phase test files go (rescued cases named) (W12a 3368bc9, 4c0a163, 1be6b73, 0dc3480, 58568f9, ed61a1a, 7b99ce1, da1739d, 11a468a, c7111cf, cce3398, 2d02451; merged 6c6f3d4; −4,217/+243; the hidden benches that records cite stay)
- ☑ W12b `tests/unit/adversarial/` dissolved per subject (part 1: W12b beaaff0, d9746f0, 8a9603a, 12d2940, 180c28c, 35eadcd; merged 65aadc75; five files, 3,007 → 865 lines; part 2: 38d7d97, 2bdef6c, 2a7592b, 6fb95f4; merged a0d58242; three files, 2,346 → 0 lines; one oracle for the missing-data rules, tests/support/missing_dtw_oracle.hpp)
- ☑ W12c repeated DTW property tests → one table-driven `core/test_dtw.cpp` against a new `tests/support/dtw_oracle.hpp` (W12c f1b2d13, f6c7a14, 6cec77c, 8d75fde, 8162a11; merged b3f6219; +820 / −2,906; the table bites on every axis)
- ☑ W12d `test_contract_parity.py` existence lists → one table (W12d 4d49609; merged eeea331; 129 rows)
- ☑ every test writes to its own temp dir (a fixed `%TEMP%/dtwc_test` collides under concurrent runs:
  `unit_test_variant_distmat`) (F1 992d51f, 50d2c46; merged b5c7048; the 3 FIXTURE_ROOT tests and 6 cmake -P CLI tests keep one directory inside the build tree, so only two runs of the same build tree collide)
- ☑ `test_hpc` and `test_api` honour `DTWC_CL_PATH`; today `find_dtwc_binary` takes the newest `dtwc_cl` under
  `build*/`, e.g. an Arrow build that cannot load its DLLs (F2 2a3ae53; merged a965ad5)
- ☑ `.github/workflows/python-tests.yml` runs pytest with no dtwc_cl and no `DTWC_CL_PATH`; `test_api`'s two
  CLI-parity cases assert a binary exists (F2 note; CI not run here) (W14b 015dc71: the job builds dtwc_cl, sets DTWC_CL_PATH and runs test_conformance.py; CI not run here)
- ☑ tests narrow `index_t` to `int` (`std::set<int>` built from `centroids_ind` / `medoid_indices`, `for (int m :
  prob.centroids_ind)`; MSVC C4244 in the CUDA tree): unit_test_clustering_algorithms.cpp, algorithms/
  unit_test_duplicate_series.cpp, unit_test_fast_clara.cpp, unit_test_fast_pam.cpp, unit_test_one_batch_pam.cpp — use
  `index_t`, with the comment sweep; also dtwc/cli/run.cpp's `std::as_const(prob).distance_matrix()` and its comment
  (redundant since `writable_distance_matrix()`, W7ef) (SW 88a6996, 79dcfc3, 1b6d3f6, d4e7f35; merged 6fd10ed1; tracker-citing comment lines 259 -> 17; C4244 in tests 2)

## G — docs and release prep (W14)

- ☐ W14a hand-written tier pages and a v1.0.0 → 2.0 migration page; `api-contract-2.0.md` deleted
- ☑ VI `dtwc_cl.exe` carries a VERSIONINFO resource: name, version, copyright (Volkan 10-02) (VI 20579e0d, 45719487, e5456cc6, b63bc153; merged 96547a5a; rc.exe and llvm-rc .res byte-identical)
- ☐ WM the Windows wheel's fill, MSVC against clang-cl, on a quiet machine with a registered band; Volkan then
  decides the wheels' compiler (Volkan 10-02: measure first)
- ☑ W14b CMake `FATAL_ERROR` for an explicit `ON` it cannot honour; CUDA CI asserts CUDA built;
  `test_conformance.py` collected (W14b 9d56aab, e7354b9, 015dc71, 836bddc, d99b15d; merged 9056fcb9; Gurobi defaults OFF; Arrow without Parquet is an IPC-only build that says so)
- ☐ W14c CHANGELOG → one `2.0.0 (unreleased)` section vs v1.0.0; MAP regenerated; audit folder deleted
- ☑ `scripts/generate_docs.py` keeps each page's line endings: on Windows it rewrites untouched pages with LF, so `git status` shows them modified (W6e) (H1 e5c7436; merged 1a079cd; the msvc preset names no generator either, f010a9a)

## After G (each behind a registered benchmark band)

- ☑ one `kmedoids_pp` (W13c d6a771eb; merged b36fad43; −490/+291; conformance, CLI 25, matlab_suite 142/141 and
  every seeded pin identical; the v1 one-argument `init::*` and unseeded `fast_pam` sequences changed as
  pre-registered; pytest 923/11/0) · ☑ HiGHS model built row-wise (W13d: done by M1's `build_p_median_model`,
  c0580948) · ☐ barycenter workspace (W13e)
- ☑ OneBatchPAM's final exact assignment (N·k DTW calls) runs in parallel (P4 9168ed9; merged 69be49c; that step 21.9× at 24 threads, the whole call 1.89× at N 2000, k 10, L 200 — identical labels, medoids, cost)
- ☑ OneBatchPAM's batch table fill (m·(N−1) DTW calls, 95 % of the call after P4) on the P1 lanes kernel (P5 1b69614; merged bac9120; table fill 4.61×, whole call 3.87× at N 2000, k 10, L 200, 24 threads; bitwise identical over 24 configurations)
- ☑ `check_docs.py` also checks the reverse direction (every live, non-hidden flag documented) — with W9's flag changes
  (1581b7a8; merged on the Mac: a table row of `getting-started/cli.md` per flag the help prints; 0 of 60 missing)
- ☑ PF-5 probe: SIMD lanes across pairs PASS, 3.7–7.9× single-thread f64, bit-identical (2026-09-29; P1 in phase B
  integrates it)
- ☑ the wheel's binding file at `-O3` (`NOMINSIZE`): LTO ran its `-Os` copy of the per-pair kernel in the whole module
  (659f889f; Python `dtw` 1.1–1.5×, ragged fill 1.5×; `baselines/2026-10-06-mac-kernel-assembly.md`)
- ◐ AArch64 lanes: the min as `fminnm`, 128-byte blocks (16 doubles, 32 floats): 1.41–2.00× single thread, fill
  1.48–1.72× at 18 threads, bitwise over the sweep, x86 untouched (23c88336; merged e26d5680: bitwise — 15.36 M
  lane outputs, 64 CLI files, conformance; x86 `.s` identical; f64 L1 loop = `v_fmin_w16`; speed waits for the
  quiet run of its kit, `baselines/2026-10-06-mac-arm-lanes.md`)
- ☐ per-pair kernels two columns per pass: kernel 1 1.44–1.98× unbanded, bitwise; kernel 2 not tried (Volkan rules)

## Blocked on another machine or on Volkan

- Metal and macOS: first pass done 2026-10-05 on `4dd4dcaf` (`baselines/2026-10-05-macos-design-2-0.md`): the
  22 Metal commits built first time; ctest 95 with the Metal tests running; conformance digit-identical but one ulp of
  silhouette; docs gates, pytest 1102/11/0, MEX + matlab_suite 139/140 (one registered filter) green. Fixed there:
  Apple clang's `memset_pattern16` idiom in the lanes kernel (`a332d671`, failed `test_codegen_no_calls`) and the
  banded kernel (`55911b2a`, two calls per column the gate cannot see); W4e merged; the Metal chunk check `bc9469fd`. The second short pass after L2b ran 2026-10-06 (`15b289be`: ctest 95 with the Metal tests,
  the 12 GPU CLI routes byte-identical to the pre-L2b binary, `baselines/2026-10-06-l2b-core-io-cli-split-mac.md`);
  pytest again once W9b lands (done 2026-10-06 at `33302535`: 970/12/0, `baselines/2026-10-06-macos-after-w9b.md`).
- ARC (Volkan, rule 11): W9c's remote leg — `test`, the `sinfo` check of `gpu_cc:8.6/8.9/9.0` and the `a100` gres
  type, `upload`, `build htc-cpu`, `build htc-gpu`, `build htc-gpu --gpu-device a100`, three `dtwcpp.cluster` runs
  (`hpc`, `hpc:gpu`, `hpc:gpu` + `gpu_device="a100"`), the unknown-key job, `submit-smoke gpu`; exact commands in
  `baselines/2026-10-06-w9c-hpc-job-toml-mac.md`. Windows: W9e's matlab_suite under R2024b (the brief stays until then).
- Release archives: `cpack` + `scripts/smoke_release_archive.py` on Linux and Windows (Windows needs a
  `dumpbin /dependents` leg).
- `cpp_conformance` under GCC and MSVC Release, `strict` and `fast`: the same 17 significant figures.
- Linux wheel: whether `libgomp` ships, and the notice says so.
- A crash recorded on Windows and never re-run: CUDA `Auto` precision through `Problem` (F42). F43 (an llfio-ON MEX under R2024b crashing in `std::mutex`) did not reproduce on 2026-09-30, and no MEX route maps a matrix yet (Y4, `baselines/2026-09-30-y4-matlab.md`).
- X2 merged `4de2ce9` on 2026-09-30; its Release `dtwc_cl.exe` ran without a Sophos event

## Records

Charter `CHARTER.md` · map `MAP.md` · rules and rulings `DECISIONS.md` · lessons `LESSONS.md` · measurements
`baselines/` · session state `summaries/` (newest two, via `session-handoff`) · literature `CITATIONS.md` ·
LR-core maths `UNIMODULAR.md`. Anything superseded is deleted; git keeps it.
