# DTWC++ Development TODO

**Last Updated:** 2026-04-04

## Active Work

### Performance (Phase 0 — complete)
- [x] `-march=native` / `/arch:AVX2` CMake option (`DTWC_ENABLE_NATIVE_ARCH`) — **+45-103% DTW throughput**
- [x] Replaced `-ffast-math` with explicit sub-flags (omits `-ffinite-math-only`) — `std::isnan` now reliable
- [x] Simplified `missing_utils.hpp` `is_missing()` to `std::isnan()` wrapper
- [x] O(n) envelope computation via Lemire ring-buffer deque (was O(n×band))

### Performance (Phase 2 — next CPU work)
- [x] `adtwBanded`: replaced full Eigen/ScratchMatrix (O(n*m)) with rolling column vector (O(n)); added `early_abandon` parameter; pruned strategy now enabled for ADTW variant
- [x] Early abandon for `adtwFull_L` — `early_abandon` parameter added; full ADTW pruned path now fully wired
- [ ] OpenMP scheduling: benchmark `schedule(dynamic,1)` vs `schedule(dynamic,16)` vs `schedule(guided)` for `fillDistanceMatrix_BruteForce` (`Problem.cpp:364`)

### Architecture (Phase 1 — out-of-core, critical for 5TB datasets)
- [ ] `DataSource` interface (`dtwc/core/data_source.hpp`) — index-based, identical API for in-memory and disk-backed data; builds on existing `distByInd` pattern
- [ ] Binary format with in-memory index (`.dtwi`/`.dtwd`) + memory-mapped reader (`MappedDataSource`)
  - Evaluate `mio` (MIT, header-only) vs custom ~100-line `mmap`/`CreateFileMapping` wrapper for 5TB file support
- [ ] Streaming CLARA assignment pass (`fast_clara.cpp`) — load blocks, compute k distances, discard
- [ ] Sample size default: scale with `sqrt(N)` for large N (current `max(40+2k, 10k+100)` too small at 100M series)
- [ ] CLARA checkpointing: save/resume assignment state (labels + best cost) for long runs
- [ ] CPU/GPU heuristic in `DistanceMatrixStrategy::Auto`: in-memory → full matrix; disk-backed + short series → GPU K-vs-N streaming; disk-backed + long series → CPU streaming

### CUDA (Phase 3)
- [ ] Architecture-aware dispatch (`DispatchProfile` by compute capability)
- [ ] Wire `compute_dtw_k_vs_all` kernel into streaming CLARA assignment (GPU double-buffered path)
- [ ] Wavefront kernel cleanup: document/remove dead preload branch (reachable only for L ∈ (256, 512])
- [ ] Unify kernel dispatch logic across `launch_dtw_kernel`, `launch_dtw_one_vs_all_kernel`, `launch_dtw_k_vs_all_kernel`
- [ ] Multi-stream pipelining for N > 5000

### Bindings
- [ ] Python: PyPI first release — CI ready (`python-wheels.yml`), needs GitHub trusted publisher setup
- [ ] MATLAB Phase 2: MIPSettings, CUDA dispatch, checkpointing, I/O utilities in `dtwc_mex.cpp`
- [ ] MATLAB: `compile.m` standalone build script (no CMake required)

### MIP Solver
- [ ] Odd-cycle cutting planes ({0,1/2}-CG cuts) as lazy constraints — **instrument Benders gap first** (>50 iterations needed to justify)

### Algorithms & Scale
- [ ] Two-phase clustering (within-group + cross-group) — after Phase 1 streaming infra proven
- [ ] Algorithm auto-selection: improve `DistanceMatrixStrategy::Auto` cost model (`N^2 * min(L, band) * ndim` threshold)

### Platform
- [ ] ARM Mac Studio investigation: test CPU path on Apple Silicon, evaluate Metal compute for GPU path
- [ ] HIPify for AMD GPU — **accept community PRs only**, do not invest core developer time

## Guidelines (not TODOs — current conventions)
- Buffer > thread_local >> heap allocation: already enforced everywhere
- No naked `new`/`delete` in core: already enforced
- Contiguous arrays in hot paths: `Data::p_vec` as `vector<vector<data_t>>` is correct for variable-length series

## Removed (completed or cut)
- ~~NaN/-ffast-math robustification~~ — **DONE**: explicit fast-math sub-flags + `std::isnan()` wrapper
- ~~Eigen 5.0.1 exploration~~ — **CUT**: Eigen used only as aligned allocator, no gap identified
- ~~Condensed distance matrix~~ — **DONE**: `DenseDistanceMatrix` already uses packed triangular N*(N+1)/2
- ~~Unnecessary memory allocations audit~~ — **CUT**: 30+ `thread_local` declarations already in place
- ~~Device-side LB pruning (pair-level)~~ — **DONE**: `compact_active_pairs` in `cuda_dtw.cu`
