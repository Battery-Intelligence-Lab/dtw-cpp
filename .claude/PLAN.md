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
- ◐ W6b enum validator tails → `-Werror=switch` (X3 `8c73029` on pb/X2; merges after X2)
- ☑ W6c `run_openmp` captures failures in per-thread slots (no critical, no atomic); `parse_ram_limit` shrinks;
  `GpuPrecision{Auto, FP32, FP64}` (Y3 `36c7883`, `dc55d13`, `b88ae0c`, `119216d`; merged `b260415`)
- ☑ W6d `cluster_by_kMedoidsPAM` shim restored; non-v1 root forwarders, D2/D3/F57 markers, tracker ids in
  comments go (Y3 `9c0b8cb`, `94ae95f`, `8b48caa`, `119216d`; merged `b260415`; tracker ids in comments move to a
  later sweep)
- ☐ W6e never-released Python and MATLAB aliases and the bindings of deleted surface go
- ☐ W6f C++ tests of deleted surface trimmed
- ☐ `test_run_resolution` runs MIP and LR-core on the CPU without a HiGHS guard: 2 of 7 cases fail in a build with
  `DTWC_ENABLE_HIGHS=OFF` (as `build/arrow-pyarrow-23`); guard them (Y3 merge report)
- ☐ Race-free sweep (DECISIONS §2 rule 6), after X2: failure capture in `fast_pam`, `fast_clara` and
  `one_batch_pam` through `run_openmp`; the FasterPAM and TADPole reductions → per-thread slots combined serially;
  OneBatchPAM's warning mutex → a serial warning. Proof: TSan in WSL (LLVM libomp + Archer)
- ☑ K1 (2026-09-29) the DP cell makes no library call: `std::min({…})` is `__std_min_d` on the MSVC STL, 7.2 ns/cell
  vs 1.06 on the Mac; nested two-argument min, `dp[i-1, j]` carried in a register; digit-identical
  (`baselines/2026-09-29-windows-kernel-msvc-stl-min.md`) (K1 `8bd6881`, `123146b`, `d114677`, `f705329`; merged
  `4441969`); the fill band is FALSIFIED — the unbanded fill runs the EAPruned kernel, which made no call; P1 measures
  lanes against it
- ☐ Y4 `bindings/matlab` and `tests/matlab` follow Y1, Y2, Y3 and X2 in one unit, then `matlab_suite`; `dtwc_mex` does not
  compile since Y1 (`7eb928b`: deleted checkpoint and storage-policy functions)
- ☑ P1 lanes in the CPU fill, after K1 and Y2: `dtw_kernel_lanes<T, W, Cell>` beside `_linear` / `_banded`, W one cache
  line of T; the fill steps a row by W columns of equal length, per-pair kernel otherwise; Standard DTW, L1 and
  squared L2, full and banded first; bitwise equal to the per-pair fill; band ≥ 2× on the 24-thread fill
  (P1 `94ef14b`, `e34c37f`, `940cd8a`, `3c95dc7`, `b930ff8`; merged `d61c499`; fill 14.5× unbanded, 5.1× band 50,
  15.3× ECG5000; cl builds get unpacked lanes — the Windows wheel is built by cl)
- ◐ P3 unbanded per-pair Standard DTW runs the linear kernel and EAPruned goes: after K1 the linear kernel is
  1.5–3.6× faster on 7 of 7 UCR datasets, bitwise equal (`baselines/2026-09-29-p2-eap-vs-linear.md`)

## C — GPU to one fill (W4 + W13's GPU half)

- ☑ W4a kernel A/B through `KernelOverride` (warp vs regtile, 2- vs 3-buffer), ±5 % band registered first: no variant
  within 5 % of its replacement, every kernel stays (`035e70d`, `baselines/2026-09-29-w4a-cuda-kernel-ab.md`)
- ☑ W4b delete MPI (Z1 `b3041d5`)
- ☑ W4c delete the 1-vs-N / K-vs-N kernels and GPU LB_Keogh (Z1 `f2b1cfe`, `8fbb7c9`; merged `741ca3b`)
- ☐ W4d CUDA: `KernelOverride` and fallback flags go; `gpu_config.cuh` reads attributes once at bind (sm_120 FP64
  fixed; its mutex and atomics go); FP32 L = 4095–4096 "invalid argument" fixed (the 48 KiB check ignores static
  shared memory; W4a)
- ☐ W4e Metal: one pipeline, one wavefront template, scratch failure → `DeviceError` (macOS CI)
- ☐ W13a one `fill()` TU; the GPU writes the packed matrix; CUDA launches chunk on an int64 pair offset
  (the N ≤ 65,536 refusal goes)
- ☐ GPU assignment for CLARA (rectangular medoids × series on the pairwise kernels) — Q4: in 2.0, after C
- ☐ CUDA tuning, each behind its own band (W4a): a separately compiled preload wavefront for L 257–1024 (−15–17 %
  FP32 measured); a 64 KB carveout above L = 2048 (−18 % at L = 2049)

## D — `index_t` in public counts (W11)

- ☐ W11a `Problem`, `Data`, `ClusteringResult`, `Config`, algorithm signatures and loops; `[[deprecated]]`
  `set_clusters(std::vector<int>)` for one release
- ☐ W11b Python `np.int64`, MATLAB double 1-based labels

## E — interface (W7 → W8 ‖ W9 + W10)

- ☐ W7a `DistanceConfig`; `set_distance / set_band / set_metric / set_variant / set_missing_strategy`
  invalidate the matrix; `bool filled_`
- ☐ W7b `resolve_dtw_fn(const DistanceConfig&)`; O(1) `dist_by_ind`; the preflight machinery goes. Acceptance:
  `dist_by_ind`'s parallel read path has no critical, atomic, validation flag or lazy allocation; a method that
  needs the matrix prepares it serially at entry
- ☐ W7c one `validate(DistanceConfig)`; `core/dtw.*`, `DTWOptions`, selector validation go
- ☐ W7d one orientation helper replaces the copied preambles
- ☐ W7e WDTW weights at bind; Soft-DTW on the linear kernel; Interpolate thread_local buffers
- ☐ W7f dead NaN functors and public helpers go
- ☐ W7g one `distance::dtw` per language
- ☐ W13b one finite scan at each matrix intake; read-only loops lose per-lookup checks
- ☐ W8a one reader entry (`read_data`); Parquet names and IPC nulls fixed; `load('x.parquet')` in Python
- ☐ W8b one writer (`write_result_files`); `Result::save` after streaming fixed
- ☐ W8c Python and MATLAB `compute_distance_matrix` through `Problem`
- ☐ W9a `Method` nine values; `ClusterMethod` goes; `run()` = apply, load, cluster, write; v1 CLI aliases
- ☐ W9b Python on `run(Config)`; one `DTWClustering` (matrix once, `score` never refits)
- ☐ W9c `hpc` → `job.toml`; the positional transport goes
- ☐ W9e MATLAB on the `run(Config)` MEX route
- ☐ W9f Python test and example trims
- ☐ W10a `DistanceMatrixStrategy`, `CUDASettings` → `set_device` + `set_gpu_precision`; the fingerprint
  hashes the resolved backend
- ☐ W10b `gpu_available()` / `gpu_info()`; `gpu:1` on Metal refused; `system_info`, `check_system`,
  `*_AVAILABLE` go (`test.parallelisation()` / `test.gpu()` stay the diagnostics)

## F — tests to their oracles (W12)

- ☐ W12a unrun and wave/phase test files go (rescued cases named)
- ☐ W12b `tests/unit/adversarial/` dissolved per subject
- ☐ W12c repeated DTW property tests → one table-driven `core/test_dtw.cpp` against a new `tests/support/dtw_oracle.hpp`
- ☐ W12d `test_contract_parity.py` existence lists → one table
- ☐ every test writes to its own temp dir (a fixed `%TEMP%/dtwc_test` collides under concurrent runs:
  `unit_test_variant_distmat`)
- ☐ `test_hpc` and `test_api` honour `DTWC_CL_PATH`; today `find_dtwc_binary` takes the newest `dtwc_cl` under
  `build*/`, e.g. an Arrow build that cannot load its DLLs

## G — docs and release prep (W14)

- ☐ W14a hand-written tier pages and a v1.0.0 → 2.0 migration page; `api-contract-2.0.md` deleted
- ☐ W14b CMake `FATAL_ERROR` for an explicit `ON` it cannot honour; CUDA CI asserts CUDA built;
  `test_conformance.py` collected
- ☐ W14c CHANGELOG → one `2.0.0 (unreleased)` section vs v1.0.0; MAP regenerated; audit folder deleted

## After G (each behind a registered benchmark band)

- ☐ one `kmedoids_pp` (W13c) · ☐ HiGHS model built row-wise (W13d) · ☐ barycenter workspace (W13e)
- ☐ OneBatchPAM's final exact assignment (N·k DTW calls) runs in parallel; it is serial today
- ☐ `check_docs.py` also checks the reverse direction (every live, non-hidden flag documented) — with W9's flag changes
- ☑ PF-5 probe: SIMD lanes across pairs PASS, 3.7–7.9× single-thread f64, bit-identical (2026-09-29; P1 in phase B
  integrates it)

## Blocked on another machine or on Volkan

- Metal: every Metal step runs on macOS CI (a push is Volkan's).
- Release archives: `cpack` + `scripts/smoke_release_archive.py` on Linux and Windows (Windows needs a
  `dumpbin /dependents` leg).
- `cpp_conformance` under GCC and MSVC Release, `strict` and `fast`: the same 17 significant figures.
- Linux wheel: whether `libgomp` ships, and the notice says so.
- Two crashes recorded on Windows and never re-run: CUDA `Auto` precision through `Problem` (F42); an
  llfio-ON MEX under R2024b in `std::mutex` (F43).
- X2 merged `4de2ce9` on 2026-09-30; its Release `dtwc_cl.exe` ran without a Sophos event

## Records

Charter `CHARTER.md` · map `MAP.md` · rules and rulings `DECISIONS.md` · lessons `LESSONS.md` · measurements
`baselines/` · session state `summaries/` (newest two, via `session-handoff`) · literature `CITATIONS.md` ·
LR-core maths `UNIMODULAR.md`. Anything superseded is deleted; git keeps it.
