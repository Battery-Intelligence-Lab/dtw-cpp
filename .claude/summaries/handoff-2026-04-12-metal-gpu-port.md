---
name: handoff-2026-04-12-metal-gpu-port
description: Session handoff — Metal GPU backend for DTWC++ on Apple Silicon. 11–28× speedup over 12-thread CPU. Global-memory kernel for 10k series. Python + MATLAB bindings both exposed with <1% wrapper overhead. 69/69 tests pass.
type: project
---

# Session Handoff — 2026-04-12 — Metal GPU Port

Branch: **Claude**.  Working tree: substantial uncommitted changes (see "Files changed" below).

## Goal

User asked "are we really using the full power and parallelisation of the mac studio?" and subsequently directed "WE CAN ACHIEVE SO MANY THINGS!" + "Don't be lazy." The session turned into a from-scratch implementation of a Metal GPU backend for DTWC++, addressing the previously-empty macOS GPU story.

## Headline numbers (Apple M2 Max, 38-core GPU)

| Workload | CPU 12-thread | Metal | Speedup |
|---|---|---|---|
| 100 × 1000 | 1 648 ms | 139 ms | **11.9×** |
| 75 × 2500 | 5 914 ms | 342 ms | **17.3×** |
| 10 × 10 000 (global-mem kernel) | 3 058 ms | 108 ms | **28.3×** |
| 30 × 10 000 | 16 061 ms | 929 ms | **17.3×** |
| **75 × 10 000** | **92.5 s** | **5.80 s** | **15.9×** |

User performance targets met:
- 75 × 10 000 unbanded on CPU: **92.5 s** (target <100 s ✓)
- 75 × 10 000 with band=100 on CPU: **1.75 s** (target <4 s ✓)

Python wrapper overhead: **−0.5% to +0.9%** (measured apples-to-apples). MATLAB wrapper overhead: **<1%** (same path). Both meet the ≤10% target.

## Accomplishments

### 1. Metal backend (from zero to working)

- **[dtwc/metal/metal_dtw.hpp](../../dtwc/metal/metal_dtw.hpp)** — public C++ API, mirrors `cuda_dtw.cuh` shape (MetalDistMatOptions, MetalDistMatResult, `metal_available`, `metal_device_info`, `compute_distance_matrix_metal`).
- **[dtwc/metal/metal_dtw.mm](../../dtwc/metal/metal_dtw.mm)** — Objective-C++ dispatcher + embedded MSL kernel source. Two kernels:
  - `dtw_wavefront` — threadgroup-memory anti-diagonal wavefront (max_L ≤ 2730 at 32 KB threadgroup cap on M1/M2/M3).
  - `dtw_wavefront_global` — same algorithm but with 3 rotating anti-diagonal buffers in device (unified) memory. Dispatcher picks automatically based on `max_L * 3 * sizeof(float)` vs the device's `maxThreadgroupMemoryLength`.
- **Command-buffer chunking** — single dispatches that take >~2 s trigger macOS's GPU watchdog (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). The dispatcher now chunks pairs across multiple command buffers (budget ≈ 5 × 10⁹ DTW cells per buffer) with a `pair_offset` kernel argument. Scratch memory is reused across chunks (requires `waitUntilCompleted` between chunks in the global path).
- **Band-range iteration** — the kernel computes `band_lo = (k - band + 1) / 2`, `band_hi = (k + band) / 2` and only iterates cells inside the Sakoe-Chiba band, instead of iterating the full diagonal and writing INF for out-of-band cells. Correctness preserved because: if `(i,j)` is in the band at diagonal k, the cells read (prev[i-1], prev[i], prev2[i-1]) are all in the band at diagonals k-1, k-2. Out-of-band cells retain their INF initialization. Verified by `test_metal_correctness` (tolerance 1e-4 rel / 1e-3 abs against CPU). **Yielded 1.6–2× speedup on banded workloads** (best case: 75×10000 band=100 went 2722 ms → 1395 ms).
- **Build system**: [CMakeLists.txt](../../CMakeLists.txt) adds `DTWC_ENABLE_METAL` option (default ON on APPLE, silently disabled elsewhere), `enable_language(OBJCXX)`, Metal/Foundation frameworks, `DTWC_HAS_METAL` compile define, Metal row in Configuration Summary.

### 2. Problem.cpp dispatch

- Added `DistanceMatrixStrategy::Metal` to [Problem.hpp](../../dtwc/Problem.hpp).
- [Problem.cpp](../../dtwc/Problem.cpp) gained a `case Metal:` dispatch block paralleling the CUDA block, with graceful CPU fallback when (a) Metal is unavailable or (b) the kernel returned empty (`pairs_computed == 0`) due to scratch-allocation failure on very large workloads.
- Works with both `DenseDistanceMatrix` and the memory-mapped distance-matrix storage (verified by `test_metal_mmap.cpp` — 72 assertions pass on both paths).
- Crucially: the Metal include was **initially trapped inside an `#ifdef DTWC_HAS_CUDA` block** at line 22 — since CUDA isn't compiled on macOS, the Metal header was silently skipped. Fixed by giving Metal its own `#ifdef DTWC_HAS_METAL` guard. ([`dtwc/Problem.cpp:21-26`](../../dtwc/Problem.cpp#L21-L26))

### 3. Python bindings

- **[python/src/_dtwcpp_core.cpp](../../python/src/_dtwcpp_core.cpp)** — added `DistanceMatrixStrategy.Metal`, `metal_available()`, `metal_device_info()`, `compute_distance_matrix_metal()`, `METAL_AVAILABLE` module attribute, Metal line in `system_info()`.
- Also fixed a pre-existing bug at line 270-271 where `DenseDistanceMatrix::write_csv` / `read_csv` were wrong — the functions live in `dtwc::io::` namespace, not `dtwc::core::DenseDistanceMatrix::`. Replaced with lambdas calling the free functions.
- Python binding build on this macOS box had a pre-existing `llfio → quickcpplib` fetch issue via `uv pip install`. Workaround: use a venv, install `nanobind` in it, point CMake at it directly:
  ```sh
  source /tmp/dtw-venv/bin/activate
  cmake --preset clang-macos -DDTWC_BUILD_PYTHON=ON \
        -Dnanobind_DIR=/private/tmp/dtw-venv/lib/python3.13/site-packages/nanobind/cmake
  ```

### 4. MATLAB bindings

- **[bindings/matlab/dtwc_mex.cpp](../../bindings/matlab/dtwc_mex.cpp)** — `parse_distance_strategy()` now accepts `"metal"` (also `"cuda"` as alias for `"gpu"`); `system_check` struct gained `metal` + `metal_info` fields.
- **[bindings/matlab/+dtwc/check_system.m](../../bindings/matlab/+dtwc/check_system.m)** — reports Metal status with emoji ticks.
- **[bindings/matlab/+dtwc/Problem.m](../../bindings/matlab/+dtwc/Problem.m)** — `set_distance_strategy` docstring updated with `'metal'`.
- **R2024b MEX linker fix** ([bindings/matlab/CMakeLists.txt](../../bindings/matlab/CMakeLists.txt)): CMake's `FindMatlab.cmake` unconditionally adds `cppMexFunction.map` to the exported-symbols list on APPLE when `Matlab_HAS_CPP_API` is true. That map requires `_mexCreateMexFunction`, `_mexDestroyMexFunction`, `_mexFunctionAdapter` — symbols only relevant for the new `matlab::mex::Function` API. We use legacy `mexFunction()`. Solution: clear `Matlab_HAS_CPP_API` (both cache and regular variable) before calling `matlab_add_mex()`. The regular variable is set by FindMatlab.cmake at its line ~1777, and matlab_add_mex's check at line 1290 sees it; `CACHE FORCE` alone doesn't work because regular-variable lookup shadows cache.
- **Do NOT** add stub implementations of the three missing symbols — it compiles, but MATLAB then crashes at MEX load because it detects the symbols exist and tries to use them as the new C++ API adapter, getting nullptrs. Verified empirically.

### 5. Benchmark infrastructure

- **[benchmarks/bench_metal_dtw.cpp](../../benchmarks/bench_metal_dtw.cpp)** — Google Benchmark harness. `BM_metal_distanceMatrix` (unbanded), `BM_cpu_distanceMatrix` (CPU baseline for speedup), `BM_metal_distanceMatrix_banded` (band=L/10), `BM_metal_distanceMatrix_tightband` (band=L/100), `BM_cpu_distanceMatrix_b100` / `BM_cpu_distanceMatrix_banded`, `BM_metal_flops`. Sizes cover 50×500 through 75×10000.
- **[benchmarks/mac_metal_benchmarks.md](../../benchmarks/mac_metal_benchmarks.md)** — human-readable benchmark summary (NEW). Captures CPU thread scaling, Metal vs CPU, long-series numbers, band optimization before/after, wrapper overhead, theoretical vs achieved FLOPs, and follow-on gaps.
- **[benchmarks/ucr_benchmark_results.md](../../benchmarks/ucr_benchmark_results.md)** — added Apple M2 Max row to hardware table and a full Apple Silicon spot-check section.
- Raw JSON under [benchmarks/results/mac_m2max/](../../benchmarks/results/mac_m2max/) (CPU thread sweeps, pinning sweeps, metal_vs_cpu).

### 6. Tests

- **[tests/unit/test_metal_correctness.cpp](../../tests/unit/test_metal_correctness.cpp)** — 6 cases, 69 assertions. Covers N=2 smallest case, L=32 small, L=200 medium, L=2000 near-cap, and L=10000 global-memory kernel. Tolerance is FP32-realistic (1e-4 relative or 1e-3 absolute; wider at L=10000).
- **[tests/unit/test_metal_mmap.cpp](../../tests/unit/test_metal_mmap.cpp)** — 2 cases, 72 assertions. Confirms the Metal strategy routes correctly through both `DenseDistanceMatrix` and the memory-mapped distance-matrix paths via `Problem::fillDistanceMatrix`.
- **Full test suite: 69/69 pass** (2 CUDA tests correctly skipped on macOS).

### 7. Housekeeping

- **E/P-core lesson** added to [.claude/LESSONS.md](../../.claude/LESSONS.md) — don't cap to P-cores (E-cores help +33%), `OMP_PROC_BIND` / `OMP_PLACES` are no-ops on Darwin.
- **macOS CI thread cap removed** ([.github/workflows/macos-unit.yml](../../.github/workflows/macos-unit.yml)) — dropped `--parallel 4` / `-j4`.
- **[CHANGELOG.md](../../CHANGELOG.md)** — new "Added (Metal GPU backend — Apple Silicon)" subsection with full speedup table, kernel variants, macOS GPU-watchdog avoidance note, wrapper-overhead numbers.

## Decisions (with rationale)

1. **Runtime MSL compilation** over `.metallib` build artifact. Simpler build (no `xcrun metal` step), no runtime-path discovery needed. Apple's runtime shader compiler caches internally so the one-time cost is amortized.

2. **Two kernel functions, host-side dispatch** over one parameterized kernel. Metal doesn't have a clean analogue for "conditional threadgroup vs global memory in the same kernel" (threadgroup memory sizing is a dispatch-time decision). Two kernels is cleaner and the dispatcher picks based on `max_L * 3 * sizeof(float)` against the device cap.

3. **Band correctness via no-clobber-outside-band** rather than explicit INF wipes. The proof (written in the kernel comment) is subtle but sound: out-of-band cells in prev/prev2 buffers retain INF from init or from previous writes; since in-band cells at diagonal k only read from in-band cells at k-1, k-2, there's no need to keep out-of-band cells fresh. Saves one store per out-of-band cell per diagonal — the dominant cost on tight-band workloads.

4. **FP32 by default** rather than FP64. Apple GPUs emulate FP64 slowly (~1/32 throughput). The wrapper accepts `MetalPrecision::Auto/FP32/FP64` for API symmetry with CUDA, but currently FP64 falls back to FP32 with a warning. Users needing bit-identical FP64 should stay on CPU or CUDA. Accuracy tested: max FP32 error vs FP64 CPU is within 1e-4 relative across the full UCR-sized workloads.

5. **MATLAB: no stub symbols for the C++ MEX API.** Stubs compile but cause runtime crashes because MATLAB detects them. Only `Matlab_HAS_CPP_API = FALSE` before `matlab_add_mex()` works.

6. **`Problem::fillDistanceMatrix` has ~15% wrapper overhead** over direct `compute_distance_matrix_metal` — seen in both Python and MATLAB. This is because `visit_distmat` lambdas, storage setup, and verbose-logging checks add real work. Not worth optimizing (it's the "canonical" entry point) — users can call `compute_distance_matrix_metal` directly for 15% more.

7. **Long-series chunking at 5 × 10⁹ cells/buffer.** Empirically 30×10000 (4.35 × 10¹⁰ cells in one buffer) triggers the macOS watchdog; 10×10000 (4.5 × 10⁹ cells) doesn't. The 5e9 budget leaves headroom. Scratch memory is reused across chunks so peak RAM is bounded by `chunk × 3 × max_L × 4` bytes.

## Open questions / known gaps

- **Tight-band Metal is only 1.25× CPU** (1.40 s vs 1.75 s on 75×10000 band=100). Anti-diagonal wavefront pays a `threadgroup_barrier` per diagonal; for tight bands the per-diagonal math is tiny and barriers dominate. A **row-major banded kernel** (iterate i ∈ [0, La), j ∈ [max(0, i-band), min(Lb-1, i+band)]) would remove the 2·L barriers — expected 5–10× over CPU at band=100. Not attempted in this pass.

- **FLOP utilization is 1.6–2.4%** of M2 Max's 13.6 TFLOPS FP32 peak. Register-tile kernel (per-thread stripe of 4–8 columns, inter-thread communication via `simd_shuffle`) would lift to ~8–12% (2–4× faster on short/medium series). cuDTW++'s `dtw_regtile_kernel` is the reference. Non-trivial MSL + warp-shuffle work.

- **No LB_Keogh Metal kernel yet.** CUDA path has envelope computation + LB_Keogh pairwise + active-pair compaction. For Metal, the benefit is workload-specific — only helps when user supplies a meaningful skip threshold. Needs API design work.

- **No 1-vs-all / k-vs-all Metal variants.** CUDA has `compute_dtw_one_vs_all` + `compute_dtw_k_vs_all`. Useful in k-medoids cluster-assignment loop.

- **Python `uv pip install .` still fails on this box** (llfio → quickcpplib fetch via uv temp-dir ninja). Worked around for benchmarking via a persistent venv + direct CMake. Does not affect CI (cibuildwheel uses a different build env).

- **`Problem::fillDistanceMatrix` 15% wrapper cost** vs direct `compute_distance_matrix_metal`. Noticeable for short workloads (adds ~1 ms on 50×500 → 6.9 ms → ~8 ms). Not fixed — `Problem` is the user-facing API and the wrapper work is necessary.

## Next steps (if continuing)

1. **Commit** (suggested groupings):
   - Commit A: `.claude/LESSONS.md` (E/P-core lesson) + `.github/workflows/macos-unit.yml` (CI cap removal)
   - Commit B: Metal backend core — `dtwc/metal/*`, `dtwc/CMakeLists.txt`, `CMakeLists.txt`, `dtwc/dtwc.hpp`, `dtwc/Problem.hpp`, `dtwc/Problem.cpp`
   - Commit C: Metal tests — `tests/unit/test_metal_correctness.cpp`, `tests/unit/test_metal_mmap.cpp`
   - Commit D: Metal benchmarks — `benchmarks/bench_metal_dtw.cpp`, `benchmarks/CMakeLists.txt`, `benchmarks/mac_metal_benchmarks.md`, `benchmarks/ucr_benchmark_results.md`, `benchmarks/results/mac_m2max/`
   - Commit E: Python binding — `python/src/_dtwcpp_core.cpp`
   - Commit F: MATLAB binding — `bindings/matlab/dtwc_mex.cpp`, `bindings/matlab/CMakeLists.txt`, `bindings/matlab/+dtwc/check_system.m`, `bindings/matlab/+dtwc/Problem.m`
   - Commit G: `CHANGELOG.md`

2. **Row-major banded kernel** for tight-band speedup on long series. Target: 75×10000 band=100 from 1.40 s to ~200 ms. Complexity: medium (new MSL kernel, no new dispatcher shape).

3. **Register-tile kernel** for short/medium series (max_L ≤ 256). Target: 2–4× speedup on those sizes, pushing FLOP utilization from ~2% to ~8%. Complexity: high (warp-shuffle patterns, careful register pressure).

4. **LB_Keogh Metal kernel** + pruning threshold API in `MetalDistMatOptions`. Complexity: medium. Only worth pursuing with a specific workload that benefits (k-medoids cluster assignment probably does).

5. **1-vs-all / k-vs-all Metal kernels** to fill out parity with `cuda_dtw`. Needed if someone starts using the k-medoids path on GPU.

6. **v3.0.0 tag**. The Unreleased CHANGELOG is now very substantial (macOS support, AI commands, Metal backend, plus the prior C++20 / std::span / Arrow I/O / float32 entries). Breaking-change items (std::span API, C++20 minimum) warrant a major bump.

## Files changed (summary)

```
Modified (15):
  .claude/LESSONS.md
  .github/workflows/macos-unit.yml
  CHANGELOG.md
  CMakeLists.txt
  benchmarks/CMakeLists.txt
  benchmarks/ucr_benchmark_results.md
  bindings/matlab/+dtwc/Problem.m
  bindings/matlab/+dtwc/check_system.m
  bindings/matlab/CMakeLists.txt
  bindings/matlab/dtwc_mex.cpp
  dtwc/CMakeLists.txt
  dtwc/Problem.cpp
  dtwc/Problem.hpp
  dtwc/dtwc.hpp
  python/src/_dtwcpp_core.cpp

New (9 files + dirs):
  benchmarks/bench_metal_dtw.cpp
  benchmarks/mac_metal_benchmarks.md
  benchmarks/results/                      (directory + JSONs)
  dtwc/metal/metal_dtw.hpp
  dtwc/metal/metal_dtw.mm
  tests/unit/test_metal_correctness.cpp
  tests/unit/test_metal_mmap.cpp
  .claude/summaries/handoff-2026-04-12-metal-gpu-port.md  (this file)
```

Plan file: `/Users/engs2321/.claude/plans/steady-wandering-tulip.md`.

## Reproducing results

```sh
# One-time Python venv for benchmarks
python3 -m venv /tmp/dtw-venv
source /tmp/dtw-venv/bin/activate
uv pip install numpy nanobind scikit-build-core ninja

# Configure + build
cmake --preset clang-macos \
    -DDTWC_BUILD_BENCHMARK=ON -DDTWC_BUILD_TESTING=ON \
    -DDTWC_BUILD_PYTHON=ON -DDTWC_BUILD_MATLAB=ON \
    -DDTWC_ENABLE_HIGHS=ON -DDTWC_ENABLE_METAL=ON \
    -DMatlab_ROOT_DIR=/Applications/MATLAB_R2024b.app \
    -Dnanobind_DIR=/private/tmp/dtw-venv/lib/python3.13/site-packages/nanobind/cmake
cmake --build build --config Release

# Run tests
ctest --test-dir build -C Release -j

# Run Metal benchmarks
./build/bin/bench_metal_dtw --benchmark_min_time=1s --benchmark_out=benchmarks/results/mac_m2max/summary.json

# Python bench
python3 /tmp/py_metal_bench.py

# MATLAB bench
/Applications/MATLAB_R2024b.app/bin/matlab -batch "run('/tmp/matlab_metal_bench.m'); exit"
```
