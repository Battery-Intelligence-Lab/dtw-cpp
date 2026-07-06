# DTWC++ Development TODO

**Last Updated:** 2026-07-06

> Refreshed from stale 2026-04-13 state. Reconciled against CHANGELOG (Unreleased),
> the 2026-06-30 device/HPC handoff, and the 2026-06-01 full-repo adversarial audit.
> Known-bug file:line references below were **re-verified against source on 2026-07-06**
> (line numbers ±a few; the Critical items and the L2/fast_pam/L2-metric findings were
> read directly, the rest spot-checked).

## Known bugs — confirmed & unfixed

Surfaced by the 2026-06-01 60-agent adversarial audit (95 CONFIRMED findings,
citation-checked). The two fixes actually applied that session were the dependabot
`pip` ecosystem and the `scripts/slurm/env.example` cleanup; **everything below is
still open**. See `.claude/summaries/handoff-2026-06-01-adversarial-audit.md` for the
full list and proposed patches.

### Critical (silent-wrong results or memory-safety)

- [ ] **CUDA wavefront drops anti-diagonal cells when `max_L > 2048`** → silent wrong DTW on the 8K-sample target. `cuda/cuda_dtw.cu:280` (`MAX_SI=8` × block 256 = 2048 cap); 3-buffer path is correct.
- [ ] **Metal `decode_pair` uses FP32 `sqrt`** → wrong/OOB pair decode for `N > ~4096` (CUDA/MPI use FP64). `metal/metal_dtw.mm:57`.
- [ ] **Metal `num_pairs = N*(N-1)/2` int32 overflow** for `N ≥ ~46341`. `metal/metal_dtw.mm:102,220,341,761`.
- [ ] **`mmap_distance_matrix.hpp:120` — no overflow guard on `packed_size(n)`** (`n*(n+1)/2`) before the truncation check → attacker-chosen `n` wraps `expected` small, check passes, OOB read.
- [ ] **`mmap_data_store.hpp:229` — interior offsets never validated** (only the sentinel is) → OOB / underflow.
- [ ] **`bindings/matlab/dtwc_mex.cpp` — no `mxIsDouble` guard** → non-double input NULL-derefs and crashes MATLAB (verified absent 2026-07-06).
- [ ] **`mip/mip_Highs.cpp:199` — status check is `assert()`** (NDEBUG no-op) → a non-optimal solve extracts garbage / empty `centroids_ind` → UB. Same Gurobi catch path `mip_Gurobi.cpp:120`.

### High

- [ ] **CPU dispatch: SoftDTW `[[fallthrough]]` → Standard L1 silently.** `core/dtw.cpp:56` returns a Standard distance for `DTWVariant::SoftDTW` instead of erroring; `gamma>0` guard also skipped on this path (NaN poison).
- [ ] **`MetricType::L2` silently computes L1.** `case MetricType::L2:` has no body and falls through to the `default` in both dispatchers — `L1Dist` (scalar, `core/dtw_cost.hpp:83`) and `MVL1Dist` (multivariate, `:92`). Requesting an L2 metric gives a per-element `|diff|` sum, not an L2 norm. Verified 2026-07-06.
- [ ] **`default_data_t = float` on public helpers** halves precision. `settings.hpp:29` — see open question below.
- [ ] **CUDA int32 index `result_matrix[si*N+sj]` overflow** for `N > 46341` (`cuda/cuda_dtw.cu:202`); `decode_pair` `row_start` int32 (`:73`). Adjacent-series math already uses `long long`; the index was left `int`.
- [ ] **I/O readers cast to `DoubleArray` with no Float64 check** and no bounds on list offsets; `ndim=0` div-by-zero. `io/arrow_ipc_reader.hpp`, `io/parquet_reader.hpp`. (crc32 is integrity-only, not tamper-proof.)
- [ ] **`fast_pam` swap phase is O(N²k)** — a genuine triple loop (`for x(N) → for p(N) → for m(k)`, `algorithms/fast_pam.cpp:182`) contradicts the header's "O(N) per swap candidate"; the FastPAM1 O(1)-per-medoid-update trick is not implemented. Verified 2026-07-06.
- [ ] **`fast_clara` RAM/chunked seed divergence** — `mt19937` (RAM) vs `mt19937_64` (chunked) diverge on the same seed; in-RAM assign is serial (no OpenMP).
- [ ] **`--metric` silently ignored on the CPU path** (only CUDA consumes it); `std::stoi(device.substr(5))` uncaught → `terminate` on a bad `cuda:N`; `--device` match is case-sensitive with silent CPU fallback.
- [ ] **`TimeSeries::view()` drops `ndim`** → multivariate round-trip corruption. `core/time_series.hpp:66` (view built from `{data.data(), data.size()}` only).
- [ ] Build supply chain: `llfio` `GIT_TAG develop` (moving branch) + `REQUIRED` (violates optional-deps rule); quickcpplib clone of HEAD `--depth1` patched+executed at configure; CPM URL tarballs have no `URL_HASH`; codecov bash uploader `curl <()` on a PR with a token.

### Dead code / cleanup (safe removals for a refactor pass)

- [ ] `types/types_util.hpp` `is_integer`/`is_zero`/`is_one` — dead (no callers) **and** misclassify negatives.
- [ ] `core::dispatch_metric` (`core/dtw_cost.hpp`) — zero call sites; duplicates the live `dtwc::detail` cost functors in `warping.hpp`.
- [ ] `DTWC_ENABLE_SIMD` — referenced in README/benchmarks but never `option()`'d and no Highway CPM dep → dead, unbuildable branch. Either wire it up or delete the references.
- [ ] SSOT candidates flagged by the audit: `decode_pair` (3 divergent copies; MPI is the correct one), DTW band-bounds formula (4 incompatible forms), nearest-medoid scan (hand-copied 4×; `detail/medoid_utils.hpp` helpers exist), CSV `setprecision(15)` write loop (4×), bench `random_series` + CPU oracle (byte-identical across 8 files → `test_util.hpp`).

## Active Work

### Performance

- [ ] OpenMP scheduling sweep: `schedule(dynamic,1)` vs `dynamic,16` vs `guided` on the DTW outer loop

### Streaming CLARA

- [ ] Smart row-group ordering: sort access by Parquet row group to minimise decompression
- [ ] Sample size scaling: `sqrt(N)` for large N (current `max(40+2k, 10k+100)` too small at 100M)
- [ ] CLARA checkpointing: save/resume assignment state for long runs
- [ ] Integration test for chunked CLARA with a small synthetic Parquet file

### CUDA

- [ ] Architecture-aware dispatch by compute capability (target H100)
- [ ] Wire `compute_dtw_k_vs_all` kernel into streaming CLARA assignment
- [ ] Float32 GPU path: series data in 80GB HBM3, DTW on-device
- [ ] Wavefront kernel cleanup: remove dead preload branch
- [ ] Multi-stream pipelining for N > 5000

### Bindings

- [ ] Python: PyPI first release — CI ready, needs GitHub trusted publisher
- [ ] MATLAB Phase 2: MIPSettings, CUDA dispatch, checkpointing
- [ ] MEX binary hygiene: `bindings/matlab/dtwc_mex.mexw64` (4.6 MB) is tracked in git — decide whether to keep or `git rm --cached` + gitignore (deferred by user 2026-07-06)

### MIP Solver

- [ ] Odd-cycle cutting planes — instrument Benders gap first (see `.claude/UNIMODULAR.md` for the TU analysis motivating this)

### Algorithms & Scale

- [ ] Two-phase clustering (within-group + cross-group)
- [ ] Algorithm auto-selection: improve cost model

### Platform

- [ ] ARM Mac Studio: test CPU path on Apple Silicon
- [ ] Arrow CPM build on Windows+Clang: blocked by ExternalProject flag quoting (Arrow upstream)
- [ ] Arrow CPM build on Windows+MSVC: untested, should work
- [ ] `CMakePresets.json:20` hardcodes `C:/Program Files/LLVM/bin/clang++.exe`; cmake floor mismatch 3.26 (listfiles) vs 3.21 (preset/pyproject)

### Documentation

- [ ] Add `data-conversion.md` Hugo page for the `dtwc-convert` tool
- [ ] Add a Mermaid architecture diagram to the website
- [ ] `docs/docs_logo.png` and `docs/static/docs_logo.png` are byte-identical — dedup once the referencing pages are checked

## Deferred (explicit non-goals for now)

- [ ] DDTW kernel fusion — derivative on-the-fly in the DTW recurrence
- [ ] Stale cache detection — hash input filenames + sizes in the mmap header
- [ ] nanoarrow C Data Interface — eliminate the Arrow C++ dependency entirely
- [ ] HIPify for AMD GPU — accept community PRs only

## Blocked

- [ ] `device="hpc"` end-to-end on a real SLURM cluster (Oxford ARC) — the submit → poll → download → `NAME_labels.csv` chain is unverified end-to-end; needs one real ARC run (cannot test from a laptop). See handoff 2026-06-30.

## Open questions

- OPEN: is `default_data_t = float` (`settings.hpp:29`) deliberate for the Parquet f32 path, or an accident? Determines whether the precision fix is a change or a no-op.
- OPEN: is the `L2`-falls-through-to-`L1` behavior (`dtw_cost.hpp:83,92`, confirmed above) intentional — i.e. is DTW meant to only ever use L1/SquaredL2 and `L2` is a deliberate alias — or should `MetricType::L2` get its own norm?
- OPEN: pin `llfio` to which SHA? Needs a maintainer decision + network.

## Needs a PR / upstream nudge (nice to have)

- [ ] File an upstream issue against [quickcpplib](https://github.com/ned14/quickcpplib) for the `QuickCppLibUtils.cmake:download_build_install` + `find_quickcpplib_library:cmakeargs` lack of `-DCMAKE_MAKE_PROGRAM` forwarding. We carry a local patch (see `cmake/Dependencies.cmake`); once upstream fixes it, our sentinel-guarded patch self-retires.

## Completed (reverse-chron, one line each)

- **2026-06-30** — **PyTorch-style device API + HPC auto-dispatch**: `dtwcpp.device()` global setter with `cpu`/`gpu`/`hpc` names; unified `device()` → `load()` → `cluster()` → `result.plot()` flow (`python/dtwcpp/_api.py`); `device="hpc"` serialises + submits to SLURM and maps labels back to input order (`_hpc.py`); generic env-parametrised `cluster_generic.slurm` job. +23 tests (201 passed / 10 skipped via overlay). **Not yet run end-to-end on a real cluster** (see Blocked).
- **~2026-05** — **UCR benchmark suite**: 128-dataset cross-architecture benchmark; results in `benchmarks/ucr_benchmark_results.pdf` and `benchmarks/plots/`.
- **2026-06-01** — **Full-repo adversarial audit** (60 agents, ~3M tokens): 95 CONFIRMED findings, citation-checked. Applied: dependabot `pip` ecosystem + `scripts/slurm/env.example` dead-var labelling. All other findings logged above under Known bugs.
- **2026-04-13** — **Python wheel build unblocked**: patched `QuickCppLibUtils.cmake` via pre-clone in `cmake/Dependencies.cmake` to forward `-G` + `-DCMAKE_MAKE_PROGRAM` at both CMake spawn sites. Verified end-to-end on macOS arm64 with no system ninja.
- **2026-04-13** — **Phase 4 (standalone API fold)**: `warping_missing_arow.hpp` + `soft_dtw.hpp` forward pass delegate to `core::dtw_kernel_{full,linear,banded}`. ~305 LOC net removed. `soft_dtw_gradient()` stays separate (owns its forward matrix).
- **2026-04-13** — Audit hardening: `DTWC_REPRODUCIBLE_BUILD` option, AROW `std::isnan` cleanup, `Problem::{write,read}DistanceMatrix` roundtrip test, mmap `c_str()` lint fix.
- **2026-04-13** — Audit follow-ups: MIP Benders test coverage, `LoadOptions` struct for `load_folder`/`load_batch_file`, `.clang-tidy` config.
- **2026-04-13** — **Phase 3 (kernel unification)**: templated `resolve_dtw_fn<T>` replaces the 130-line switch; AROW / Soft-DTW folded via `AROWCell` + `SoftCell`; MV AROW first-class. Shipped with the f32 dispatch bug fix.
- **2026-04-12** — Phase 2: fold `warping_missing` into the unified DTW kernel; 1.54–2.83× speedup on banded paths.
- **Earlier** — Phase 4 (data access + I/O + f32): `Data::series(i)` span accessor, CLARA zero-copy views (48× subsample), `StoragePolicy` enum, Arrow IPC + Parquet readers, `dtwc-convert` CLI, runtime `Precision::Float32`/`Float64`.
- **Earlier** — Phase 2 (CPU perf): `adtwBanded` rolling column + early abandon + pruned.
- **Earlier** — Phase 0 (CPU throughput): `-march=native`, `std::isnan`-safe fast-math subset, O(n) Lemire envelope. +45–103% DTW throughput.
- **Earlier** — RAM-aware chunked CLARA: `--ram-limit`, `ParquetChunkReader` row-group streaming, medoid pinning, float32 chunked Parquet path with OpenMP.
