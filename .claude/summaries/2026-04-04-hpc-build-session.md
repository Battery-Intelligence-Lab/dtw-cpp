# Session Handoff — 2026-04-04 (HPC build config + production SIMD wiring)

## What Was Done

### 1. Wired lb_keogh_highway into production dispatch

**File:** `dtwc/core/lower_bound_impl.hpp`

Added a forward declaration of `dtwc::simd::lb_keogh_highway()` under `#ifdef DTWC_HAS_HIGHWAY` (lines 29–35), and added an `if constexpr (std::is_same_v<T, double>)` dispatch inside `lb_keogh<T>()` (lines 167–170). When `DTWC_ENABLE_SIMD=ON`, all `double` LB_Keogh calls now route through the Highway SIMD path (2.7–3.3× speedup). `float` and other types fall through to the scalar loop unchanged.

### 2. Confirmed NaN safety — updated LESSONS.md

`std::isnan()` is safe because `-ffinite-math-only` is not set. `is_missing()` in `missing_utils.hpp` is a thin wrapper over `std::isnan()`. LESSONS.md updated to remove the stale `memcpy` advice.

### 3. Completed GCC/Clang fast-math flag set

Added `-fno-rounding-math` and `-fno-signaling-nans` to `cmake/StandardProjectSettings.cmake`. These are the two remaining components of `-ffast-math` that were missing (everything except `-ffinite-math-only` is now enabled). Also added MSVC `/Gy` (function-level linking / COMDAT elimination).

### 4. Added DTWC_ARCH_LEVEL for HPC portability

**File:** `cmake/StandardProjectSettings.cmake`

New `DTWC_ARCH_LEVEL` CMake cache string (`""` / `"v3"` / `"v4"`). When set:
- `v3` → `-march=x86-64-v3` (AVX2+FMA) — safe for the entire HPC fleet: Broadwell, Haswell, Cascade Lake, Sapphire/Emerald Rapids, AMD Rome, Genoa, Turin
- `v4` → `-march=x86-64-v4` / `/arch:AVX512` — AVX-512 nodes only (Cascade Lake Xeon, Sapphire/Emerald Rapids, Genoa, Turin)
- `""` (default) — unchanged behaviour (`-march=native` / `/arch:AVX2`)

Default is unchanged so desktop builds are unaffected.

### 5. Changed DTWC_ENABLE_SIMD default to ON for top-level builds

**File:** `CMakeLists.txt`

Changed from `option(DTWC_ENABLE_SIMD ... OFF)` to `cmake_dependent_option(... ON "PROJECT_IS_TOP_LEVEL;NOT DTWC_BUILD_PYTHON" OFF)`. Highway SIMD is now on by default for standalone builds. Highway compiles SSE4/AVX2/AVX-512 targets into one binary and dispatches at runtime — ideal for heterogeneous HPC clusters. Sub-project consumers and Python wheels remain unaffected (still OFF).

### 6. Added DTWC_CUDA_ARCH_LIST with HPC-fleet default

**File:** `CMakeLists.txt`

Added `DTWC_CUDA_ARCH_LIST` cache string defaulting to `"70;80;86;89;90"` (V100, A100, RTX Ampere, L40s, H100). Inserted before `enable_language(CUDA)` so CMake reads it at language-enable time. `CMAKE_CUDA_ARCHITECTURES` is only set from this default if not already specified by the user. P100 (sm_60) deliberately excluded — EOL at most HPC centres, doubles compile time.

### 7. Updated README with HPC section and options table

Added "HPC / Supercomputer builds" subsection under Installation with:
- Portable CPU build example (`DTWC_ARCH_LEVEL=v3`)
- AVX-512 build example (`DTWC_ARCH_LEVEL=v4`)
- CUDA multi-arch examples (default, single-arch, P100 addition)
- SIMD explanation (Highway, runtime dispatch, no rebuild needed per node)
- OpenMP NUMA hints (`OMP_PROC_BIND=close OMP_PLACES=cores`)

Updated "All CMake options" table with the three new options and corrected SIMD default/description.

---

## Hardware Context (for future sessions)

Target HPC hardware:
- **CPU mix:** Broadwell, Haswell, Cascade Lake (48-core, Xeon Platinum 8268, 384 GB), Sapphire/Emerald Rapids, AMD Rome/Genoa, AMD Turin (288-core, EPYC 9825, 2.3 TB)
- **GPU mix:** P100 (sm_60), V100 (sm_70), A100 (sm_80), RTX various (sm_75/86/89), L40s (sm_89), H100 (sm_90)
- All CPUs support at least AVX2; Cascade Lake+, Genoa, Turin support AVX-512

---

## Files Changed This Session

| File | Change |
|------|--------|
| `dtwc/core/lower_bound_impl.hpp` | Added Highway dispatch for `lb_keogh<double>` |
| `cmake/StandardProjectSettings.cmake` | Added `DTWC_ARCH_LEVEL`; `/Gy`; `-fno-rounding-math`; `-fno-signaling-nans` |
| `CMakeLists.txt` | `DTWC_ENABLE_SIMD` default ON; `DTWC_CUDA_ARCH_LIST`; `include(CMakeDependentOption)` |
| `README.md` | HPC section + updated options table |
| `CHANGELOG.md` | SIMD, HPC build, fast-math, `/Gy` entries added |
| `.claude/LESSONS.md` | NaN check updated: `std::isnan()` safe, `memcpy` advice removed |

---

## Open Questions / Next Steps

1. **Phase 2C (OpenMP scheduling):** Compare `schedule(dynamic,1)` vs `schedule(dynamic,16)` vs `schedule(guided)` for `fillDistanceMatrix_BruteForce` (`Problem.cpp` main OMP loop) at N=100, 500, 1000. Especially relevant at 288-core Turin scale.

2. **multi_pair_dtw equal-length fast path:** `imask` is still computed inside the inner loop even for equal-length pairs (always all-false). A maskless fast path for uniform lengths would push n=1000 to ~1.3× faster than scalar. Not implemented yet.

3. **multi_pair_dtw AVX-512 width:** `FixedTag<double, 4>` limits to 4 SIMD lanes even on AVX-512 (which supports 8). For Cascade Lake/Genoa/Turin nodes a `FixedTag<double, 8>` path or `ScalableTag<double>` would double throughput. Worth benchmarking on AVX-512 hardware.

4. **Phase 1 (out-of-core):** DataSource interface + binary format (`.dtwi`/`.dtwd`) + streaming CLARA assignment pass for the 5 TB dataset. Still the highest-priority feature.

5. **Intel compiler (icpx):** Many HPC clusters use Intel's LLVM-based compiler (`icpx`). It identifies as Clang-compatible in CMake so the GCC/Clang path should work, but this has not been tested. Worth verifying on a cluster.

6. **MPI default:** `DTWC_ENABLE_MPI=OFF` by default. For multi-node HPC jobs at 288-core scale, MPI becomes relevant. Consider changing to `cmake_dependent_option` like SIMD (ON when MPI libraries are found and `PROJECT_IS_TOP_LEVEL`).
