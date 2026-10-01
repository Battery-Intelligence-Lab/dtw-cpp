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
- ☐ `build/` reports Arrow ON but builds without it (Arrow not found with clang on Windows): Parquet tests run only in
  `build/arrow-pyarrow-23` until W14b makes an unhonoured `ON` a configure error. That tree finds Arrow through the
  shim `build/arrow-pyarrow-23/pyarrow-config` over `.venv`'s pyarrow 23.0.1; an Arrow gate counts only if
  `ctest -N` lists `test_io_readers`

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
- ☐ W6f C++ tests of deleted surface trimmed
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
- ☐ W4e Metal: one pipeline, one wavefront template, scratch failure → `DeviceError` (macOS CI) — a bad_alloc in Metal's out.resize leaks its released buffers (W13a review)
- ☑ W13a one `fill()` TU; the GPU writes the packed matrix; CUDA launches chunk on an int64 pair offset (W13a e9216f4, ccc07db, c67dd3f, 71a26b6, 08db34d, 1affff2, ace7f07; merged 62d5822; no fill.cpp — the fill was already one function; FP32 L 100 fill 0.756× base time, host memory at N 20,000 L 1000 6.2 → 1.6 GB)
  (the N ≤ 65,536 refusal goes); a backend refuses before the N×N matrix is allocated — today `Problem` resizes first,
  so a huge N on a host without a GPU hits bad_alloc before DeviceError (W4d review)
- ☐ GPU assignment for CLARA (rectangular medoids × series on the pairwise kernels) — Q4: in 2.0, after C
- ☑ CUDA tuning, each behind its own band (W4a): a separately compiled preload wavefront for L 257–1024 (−15–17 %
  FP32 measured); a 64 KB carveout above L = 2048 (−18 % at L = 2049) (preload landed, C1 15b9143: FP32/FP64 L 257–512 at 0.82–0.89 of base; the carveout passed its band but needs the CUDA 12.5 API and the floor stays CUDA 12.0 — ARC loads 12.4 — so it was reverted, b803e47)
- ☑ CUDA: the global wavefront above L 2048 where fewer than 3 blocks fit an SM (C1's probe: FP32 L 6000/8000 at 0.64/0.66 of the shared route, FP64 L 2049–4000 at 0.76–0.84) — its own band; the Shared kernel's unreachable preload branch goes with it (C1) (C2 63d5d0a; merged dd33a0e; FP32 L 6000–8446 at 0.64–0.68 of base, FP64 L 2049–4223 at 0.76–0.84; deleting the Shared kernel's preload branch FALSIFIED — it raises occupancy and slows FP32 L 513–2757 by 3–23 %, so the branch stays)
- ☐ CUDA: the FP64 Shared kernel at 4 blocks per SM (C2 lead) — its own band
- ☑ CUDA floor: compute capability 8.0 (the A30's generation, Volkan 09-30); older devices get a typed DeviceError before any allocation (C1 c445089)
- ☑ CPU floor x86-64-v3 for release archives and wheels (Volkan 09-30); one `DTWC_ARCH_LEVEL` (native | v3 | v4) (V3) (V3 165e48d, 41a1be1; merged 169db40)
- ☐ cross-route checks (lanes vs per-pair) within a path-length bound; each compiler keeps its contraction (Volkan
  10-01); the GCC-only test failures explained (V4)
- ☐ ARC scripts follow the CUDA floor; a build on a GPU node is native (S1)
- ☐ lead: 16 double lanes to hide the min-then-add latency (V3: x86-64-v3 vs SSE2 ~1.0× unbanded, 1.17× banded) —
  its own band
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
- ☐ W7c one `validate(DistanceConfig)`; `core/dtw.*`, `DTWOptions`, selector validation go
- ☐ W7d one orientation helper replaces the copied preambles
- ☐ W7e WDTW weights at bind; Soft-DTW on the linear kernel; Interpolate thread_local buffers (WDTW weights at bind done in E1); the mutable
  `distance_matrix()` overload gets its own name, so a reader cannot clear `filled_` by accident (E1)
- ☐ W7f dead NaN functors and public helpers go
- ☐ W7g one `distance::dtw` per language
- ☐ W13b one finite scan at each matrix intake; read-only loops lose per-lookup checks
- ☐ W8a one reader entry (`read_data`); Parquet names and IPC nulls fixed; `load('x.parquet')` in Python
- ☐ W8b one writer (`write_result_files`); `Result::save` after streaming fixed
- ☐ W8c Python and MATLAB `compute_distance_matrix` through `Problem` (the binding's own failure-slot loop, which rethrows by thread number, goes with it — R1)
- ☐ W9a `Method` nine values; `ClusterMethod` goes; `run()` = apply, load, cluster, write; v1 CLI aliases
- ☐ W9b Python on `run(Config)`; one `DTWClustering` (matrix once, `score` never refits); `variant_params` / `cuda_settings` return read-only objects, so a nested write raises instead of
  silently editing a copy (E1) — Python `DTWClustering` refuses `max_iter = 0` like `sklearn.py` and MATLAB (B3)
- ☐ W9c `hpc` → `job.toml`; the positional transport goes; `device=hpc` takes `gpu_device=` (a100, a6000, l40s, h100, …) to pick the
  SLURM GPU and the build's CUDA arch; a build on the target node detects both itself (Volkan 09-30). S1 notes:
  `slurm_remote.sh build` submits with no `--gres` (always portable); `jobs/gpu_test.slurm` and
  `ucr_benchmark_gpu.slurm` ask for any GPU and can land on a refused V100; ARC documents `gpu:<type>:<n>` (P100,
  V100, RTX, RTX8000, A100) and constraints `gpu_sku`/`gpu_gen`/`gpu_cc`/`gpu_mem`/`nvlink`, no type for RTX A6000,
  H100 or L40S
- ☐ W9e MATLAB on the `run(Config)` MEX route; `cmd_cluster_legacy` and snake_case keys go here (DECISIONS 09-30); MATLAB
  regains read access to band, verbose, max_iter and n_repetitions under the Python names, and its own metric lists
  (`DTWClustering.resolve_metric`, `validate_metric.m`) give way to the C++ table (W6m)
- ☐ W9f Python test and example trims
- ☐ W10a `DistanceMatrixStrategy`, `CUDASettings` → `set_device` + `set_gpu_precision`; the fingerprint
  hashes the resolved backend
- ☐ W10b `gpu_available()` / `gpu_info()`; `gpu:1` on Metal refused; `system_info`, `check_system`,
  `*_AVAILABLE` go (`test.parallelisation()` / `test.gpu()` stay the diagnostics)

## F — tests to their oracles (W12)

- ☑ W12a unrun and wave/phase test files go (rescued cases named) (W12a 3368bc9, 4c0a163, 1be6b73, 0dc3480, 58568f9, ed61a1a, 7b99ce1, da1739d, 11a468a, c7111cf, cce3398, 2d02451; merged 6c6f3d4; −4,217/+243; the hidden benches that records cite stay)
- ☐ W12b `tests/unit/adversarial/` dissolved per subject
- ☐ W12c repeated DTW property tests → one table-driven `core/test_dtw.cpp` against a new `tests/support/dtw_oracle.hpp`
- ☑ W12d `test_contract_parity.py` existence lists → one table (W12d 4d49609; merged eeea331; 129 rows)
- ☑ every test writes to its own temp dir (a fixed `%TEMP%/dtwc_test` collides under concurrent runs:
  `unit_test_variant_distmat`) (F1 992d51f, 50d2c46; merged b5c7048; the 3 FIXTURE_ROOT tests and 6 cmake -P CLI tests keep one directory inside the build tree, so only two runs of the same build tree collide)
- ☑ `test_hpc` and `test_api` honour `DTWC_CL_PATH`; today `find_dtwc_binary` takes the newest `dtwc_cl` under
  `build*/`, e.g. an Arrow build that cannot load its DLLs (F2 2a3ae53; merged a965ad5)
- ☐ `.github/workflows/python-tests.yml` runs pytest with no dtwc_cl and no `DTWC_CL_PATH`; `test_api`'s two
  CLI-parity cases assert a binary exists (F2 note; CI not run here)

## G — docs and release prep (W14)

- ☐ W14a hand-written tier pages and a v1.0.0 → 2.0 migration page; `api-contract-2.0.md` deleted
- ☐ W14b CMake `FATAL_ERROR` for an explicit `ON` it cannot honour; CUDA CI asserts CUDA built;
  `test_conformance.py` collected
- ☐ W14c CHANGELOG → one `2.0.0 (unreleased)` section vs v1.0.0; MAP regenerated; audit folder deleted
- ☑ `scripts/generate_docs.py` keeps each page's line endings: on Windows it rewrites untouched pages with LF, so `git status` shows them modified (W6e) (H1 e5c7436; merged 1a079cd; the msvc preset names no generator either, f010a9a)

## After G (each behind a registered benchmark band)

- ☐ one `kmedoids_pp` (W13c) · ☐ HiGHS model built row-wise (W13d) · ☐ barycenter workspace (W13e)
- ☑ OneBatchPAM's final exact assignment (N·k DTW calls) runs in parallel (P4 9168ed9; merged 69be49c; that step 21.9× at 24 threads, the whole call 1.89× at N 2000, k 10, L 200 — identical labels, medoids, cost)
- ☑ OneBatchPAM's batch table fill (m·(N−1) DTW calls, 95 % of the call after P4) on the P1 lanes kernel (P5 1b69614; merged bac9120; table fill 4.61×, whole call 3.87× at N 2000, k 10, L 200, 24 threads; bitwise identical over 24 configurations)
- ☐ `check_docs.py` also checks the reverse direction (every live, non-hidden flag documented) — with W9's flag changes
- ☑ PF-5 probe: SIMD lanes across pairs PASS, 3.7–7.9× single-thread f64, bit-identical (2026-09-29; P1 in phase B
  integrates it)

## Blocked on another machine or on Volkan

- Metal: every Metal step runs on macOS CI (a push is Volkan's).
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
