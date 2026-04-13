# DTWC++ Development TODO

**Last Updated:** 2026-04-13

## Active Work

### Performance
- [ ] OpenMP scheduling sweep: `schedule(dynamic,1)` vs `dynamic,16` vs `guided` on DTW outer loop

### Streaming CLARA
- [ ] Smart row-group ordering: sort access by Parquet row group to minimise decompression
- [ ] Sample size scaling: `sqrt(N)` for large N (current `max(40+2k, 10k+100)` too small at 100M)
- [ ] CLARA checkpointing: save/resume assignment state for long runs
- [ ] Integration test for chunked CLARA with small synthetic Parquet file

### CUDA
- [ ] Architecture-aware dispatch by compute capability (target H100)
- [ ] Wire `compute_dtw_k_vs_all` kernel into streaming CLARA assignment
- [ ] Float32 GPU path: series data in 80GB HBM3, DTW on-device
- [ ] Wavefront kernel cleanup: remove dead preload branch
- [ ] Multi-stream pipelining for N > 5000

### Bindings
- [ ] Python: PyPI first release — CI ready, needs GitHub trusted publisher
- [ ] MATLAB Phase 2: MIPSettings, CUDA dispatch, checkpointing

### MIP Solver
- [ ] Odd-cycle cutting planes — instrument Benders gap first

### Algorithms & Scale
- [ ] Two-phase clustering (within-group + cross-group)
- [ ] Algorithm auto-selection: improve cost model

### Platform
- [ ] ARM Mac Studio: test CPU path on Apple Silicon
- [ ] Arrow CPM build on Windows+Clang: blocked by ExternalProject flag quoting (Arrow upstream)
- [ ] Arrow CPM build on Windows+MSVC: untested, should work

### Documentation
- [ ] Add `data-conversion.md` Hugo page for dtwc-convert tool
- [ ] Add Mermaid architecture diagram to website

## Deferred (explicit non-goals for now)

- [ ] DDTW kernel fusion — derivative on-the-fly in DTW recurrence
- [ ] Stale cache detection — hash input filenames + sizes in mmap header
- [ ] nanoarrow C Data Interface — eliminate Arrow C++ dependency entirely
- [ ] HIPify for AMD GPU — accept community PRs only

## Blocked

- [ ] **Python wheel cross-language parity test** (`tests/integration/test_cross_language.py`). `uv pip install -e .` fails at llfio's quickcpplib sub-CMake configure: `no such file or directory '.../ninja'`. Root cause: quickcpplib spawns a sub-CMake that does not inherit scikit-build-core's ninja-path injection; passing `CMAKE_MAKE_PROGRAM` via `[tool.scikit-build.cmake.define]` propagates to the top-level but not the sub-build. Workarounds: (a) `brew install ninja` system-wide before `uv pip install -e .` — simplest local fix, doesn't help CI; (b) make llfio truly optional via `DTWC_ENABLE_MMAP` and guard `MmapDistanceMatrix` behind `#ifdef DTWC_HAS_MMAP` — non-trivial, touches `Problem::distMat` `std::variant` + `visit_distmat` + CUDA/Metal backends; (c) file upstream issue against quickcpplib (`QuickCppLibUtils.cmake:79`) for `CMAKE_MAKE_PROGRAM` propagation.

## Completed (reverse-chron, one line each)

- **2026-04-13** — **Phase 4 (standalone API fold)**: `warping_missing_arow.hpp` (dtwAROW / dtwAROW_L / dtwAROW_banded) and `soft_dtw.hpp` forward pass now delegate to `core::dtw_kernel_{full,linear,banded}`. ~305 LOC net removed. `soft_dtw_gradient()` stays separate (Cuturi–Blondel backward pass owns its forward matrix).
- **2026-04-13** — Audit hardening: `DTWC_REPRODUCIBLE_BUILD` option (`-ffile-prefix-map`), AROW `std::isnan` cleanup, `Problem::{write,read}DistanceMatrix` roundtrip test, mmap `c_str()` lint fix.
- **2026-04-13** — Audit follow-ups: MIP Benders test coverage (`[mip][highs][benders]`), `LoadOptions` struct for `load_folder`/`load_batch_file`, `.clang-tidy` config with `dtwc/` + `tests/` header filter.
- **2026-04-13** — **Phase 3 (kernel unification)**: templated `resolve_dtw_fn<T>` replaces 130-line switch; AROW / Soft-DTW folded into unified kernel via `AROWCell` + `SoftCell`; MV AROW first-class via per-channel-skip cost. Shipped with f32 dispatch bug fix (`dtw_function_f32()` was hardwired to Standard DTW).
- **2026-04-12** — Phase 2: fold `warping_missing` into unified DTW kernel; 1.54-2.83× speedup on banded paths.
- **Earlier** — Phase 4 (data access + I/O + f32): `Data::series(i)` span accessor, CLARA zero-copy views (48× subsample), `StoragePolicy` enum, Arrow IPC + Parquet readers, `dtwc-convert` CLI, runtime `Precision::Float32`/`Float64`.
- **Earlier** — Phase 2 (CPU perf): `adtwBanded` rolling column + early abandon + pruned.
- **Earlier** — Phase 0 (CPU throughput): `-march=native`, `std::isnan`-safe fast-math subset, O(n) Lemire envelope. +45-103% DTW throughput.
- **Earlier** — RAM-aware chunked CLARA: `--ram-limit`, `ParquetChunkReader` row-group streaming, medoid pinning, float32 chunked Parquet path with OpenMP.
