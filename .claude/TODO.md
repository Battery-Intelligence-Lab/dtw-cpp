# DTWC++ Development TODO

**Last Updated:** 2026-04-08

## Active Work

### Performance (Phase 0 — complete)
- [x] `-march=native` / `/arch:AVX2` CMake option — **+45-103% DTW throughput**
- [x] Replaced `-ffast-math` with explicit sub-flags — `std::isnan` now reliable
- [x] O(n) envelope computation via Lemire ring-buffer deque

### Data Access (Phase 4 — complete)
- [x] `Data::series(i)` → `span<const data_t>` uniform accessor (heap, mmap, view modes)
- [x] CLARA zero-copy views via `set_view_data()` — **48x subsample speedup**
- [x] `StoragePolicy` enum (Auto/Heap/Mmap)
- [x] Span overloads for `compute_summary`, `compute_envelope`, `lb_keogh`, `lb_keogh_symmetric`
- [x] Migrated distance layer (`distByInd`, `fillDistanceMatrix_BruteForce`, pruned) from `p_vec(i)` to `series(i)`

### I/O Formats (Phase 4 — complete)
- [x] Arrow IPC reader (`dtwc/io/arrow_ipc_reader.hpp`) — zero-copy mmap, List + LargeList
- [x] Parquet reader (`dtwc/io/parquet_reader.hpp`) — scalar + list columns, directory loading
- [x] Python `dtwc-convert` CLI tool — Parquet/CSV/HDF5 → Arrow IPC or .dtws
- [x] CMake: `DTWC_ENABLE_ARROW` via `find_package` or CPM from source (static, ~20MB CLI)
- [x] CLI: auto-detects `.parquet`, `.arrow`, `.ipc`, `.feather`, `.dtws`, `.csv` input

### Float32 (Phase 4 — complete)
- [x] Runtime precision selection (`Precision::Float32` / `Float64`)
- [x] `Data` float32 constructor + `series_f32(i)` accessor
- [x] `dtw_fn_f32_t` dispatch in `distByInd` and `fillDistanceMatrix_BruteForce`
- [x] CLI: `--precision float32|float64` (default float32)
- [x] Benchmarked: identical speed (latency-bound), 2x memory, 0.003% DTW error

### Performance (Phase 2 — next CPU work)
- [x] `adtwBanded`: rolling column vector + early abandon + pruned strategy
- [ ] OpenMP scheduling: benchmark `schedule(dynamic,1)` vs `schedule(dynamic,16)` vs `schedule(guided)`

## Remaining Work

### RAM-Aware Chunked Processing
- [x] Wire `--ram-limit` into chunked CLARA — stream Parquet row groups within budget
- [x] CLARA float32 view-mode (float32 + views via `p_spans_f32_`)
- [x] `ParquetChunkReader` for row-group streaming with RAM budget calculation
- [x] Medoid pinning: k medoid series loaded on demand from Parquet
- [ ] Smart row-group ordering: sort access by Parquet row group to minimize decompression

### Streaming CLARA
- [x] Assignment pass: load chunks from Parquet, compute N×k distances, discard chunk
- [x] Float32 chunked Parquet path: `read_row_groups_f32` + `assign_all_points_chunked_f32`
- [x] OpenMP parallelism in chunked assignment inner loop (`schedule(dynamic)`, reader called outside loop)
- [ ] Sample size scaling: `sqrt(N)` for large N (current `max(40+2k, 10k+100)` too small at 100M)
- [ ] CLARA checkpointing: save/resume assignment state for long runs
- [ ] Integration test for chunked CLARA with small synthetic Parquet file

### CUDA (Phase 3)
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
- [x] Update CHANGELOG.md for Phase 4 changes
- [x] Update README.md feature list (Parquet, Arrow IPC, float32)
- [ ] Add `data-conversion.md` Hugo page for dtwc-convert tool
- [ ] Add Mermaid architecture diagram to website

### Deferred
- [ ] DDTW kernel fusion — derivative on-the-fly in DTW recurrence
- [ ] Stale cache detection — hash input filenames + sizes in mmap header
- [ ] nanoarrow C Data Interface — eliminate Arrow C++ dependency entirely
- [ ] HIPify for AMD GPU — accept community PRs only

### Blocked
- [ ] Python wheel cross-language parity test (`tests/integration/test_cross_language.py`). `uv pip install -e .` fails with `CMake Error at ... QuickCppLibUtils.cmake:79 (message): FATAL: Configure download, build and install of quickcpplib ... stderr was: no such file or directory '.../ninja' --version`. Root cause: llfio's quickcpplib dependency spawns a **sub-CMake** configure that doesn't inherit scikit-build-core's ninja-path injection. Passing `CMAKE_MAKE_PROGRAM` at the top level (via `[tool.scikit-build.cmake.define]`) does not propagate to the sub-build. Workarounds to investigate: (a) `brew install ninja` system-wide before `uv pip install -e .`, (b) make llfio truly optional via `DTWC_ENABLE_MMAP` and guard `MmapDistanceMatrix` with `#ifdef DTWC_HAS_MMAP` (non-trivial — type lives in `Problem::distMat` `std::variant`), (c) file upstream issue against quickcpplib for make-program hint propagation.