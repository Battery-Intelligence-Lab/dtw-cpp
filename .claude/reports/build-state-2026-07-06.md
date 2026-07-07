# DTWC++ Build / CI / Packaging / Parallelisation State — 2026-07-06

Audit for the plan: (a) automated executables + MEX + wheels on Windows / macOS-Intel / macOS-arm / Linux; (b) out-of-the-box parallelisation with no silent sequential fallback; (c) user-facing `dtwc.test.parallelisation()` / `dtwc.test.gpu()` introspection API.

All claims tagged **[confirmed]** cite file:line read on branch `Claude` (clean tree). **[inferred]** items name what would confirm them.

---

## 1. Build graph

### Targets
| Target | Kind | Defined | Condition |
|---|---|---|---|
| `dtwc++` | STATIC lib, PIC ON | dtwc/CMakeLists.txt:3-6 | always |
| `mip-solvers` | lib (dtwc/mip) | dtwc/CMakeLists.txt:8 | always |
| `dtwc_main` | exe | CMakeLists.txt:209 | `PROJECT_IS_TOP_LEVEL` only |
| `dtwc_cl` | exe (CLI11, optional yaml-cpp) | CMakeLists.txt:220-236 | `PROJECT_IS_TOP_LEVEL` only |
| `_dtwcpp_core` | nanobind module (STABLE_ABI if Py>=3.12) | python/CMakeLists.txt:27-31 | `DTWC_BUILD_PYTHON` |
| `dtwc_mex` | MEX via `matlab_add_mex` | bindings/matlab/CMakeLists.txt:30-35 | `DTWC_BUILD_MATLAB` + MATLAB found |
| tests / benchmarks / examples | subdirs | CMakeLists.txt:241-252 | respective options |

**No `install(TARGETS ...)` for `dtwc_main`/`dtwc_cl` anywhere; no CPack config** [confirmed: grep `CPack|install(TARGETS` hits only bindings/matlab/CMakeLists.txt:40 and python/CMakeLists.txt:41]. Executable distribution today = copy binaries out of the build tree by hand.

### Options (exact names + defaults, CMakeLists.txt:20-35)
| Option | Default | Notes |
|---|---|---|
| `DTWC_BUILD_EXAMPLES` | OFF | |
| `DTWC_BUILD_TESTING` | OFF | |
| `DTWC_BUILD_BENCHMARK` | OFF | |
| `DTWC_BUILD_PYTHON` | OFF | nanobind |
| `DTWC_BUILD_MATLAB` | OFF | soft-fails (WARNING + return) if MATLAB not found (bindings/matlab/CMakeLists.txt:8-11) |
| `DTWC_DEV_MODE` | OFF | |
| `DTWC_ENABLE_GUROBI` | **ON** | warns if not found (CMakeLists.txt:326-334) |
| `DTWC_ENABLE_HIGHS` | **ON** | CPM v1.14.0, NDEBUG forced on target (Dependencies.cmake:28-44) |
| `DTWC_ENABLE_MPI` | OFF | MS-MPI hint logic Dependencies.cmake:350-365; auto-disables with WARNING if missing |
| `DTWC_ENABLE_CUDA` | OFF | auto-disable on Apple w/ WARNING (CMakeLists.txt:40-45); auto-disable w/ WARNING if nvcc missing (:174-177); default arch list `60;70;75;80;86;89;90` (:123) |
| `DTWC_ENABLE_METAL` | **ON** | Apple-only; *silently* set OFF on non-Apple (CMakeLists.txt:188-190) |
| `DTWC_ENABLE_YAML` | OFF | yaml-cpp 0.9.0 via CPM |
| `DTWC_ENABLE_ARROW` | OFF | find_package first, else CPM Arrow 19.0.1 static minimal build (Dependencies.cmake:248-344) |

There is **no `DTWC_ENABLE_OPENMP` option** — OpenMP is unconditionally probed and used if found (dtwc/CMakeLists.txt:108-123); if absent, build proceeds with a configure-time `message(WARNING)` only.

### Required vs optional deps (Dependencies.cmake)
- Required, all via CPM at configure time: CPMLicenses 0.0.7, Catch2 3.13.0 (always fetched even when testing OFF — Dependencies.cmake:18-25), CLI11 2.6.2, rapidcsv 8.92, Eigen 5.0.1, **llfio `GIT_TAG develop` (unpinned!)** — llfio is `FATAL_ERROR` if absent (Dependencies.cmake:238).
- llfio bootstrap **pre-clones quickcpplib from GitHub at configure time and text-patches `QuickCppLibUtils.cmake`** (ninja `-G`/`-DCMAKE_MAKE_PROGRAM` propagation for sandboxed wheel builds) — Dependencies.cmake:126-231. Configure therefore requires network + git; patch is regex-fragile against upstream drift (WARNING path at :211-217).
- Optional: OpenMP (find), Gurobi (cmake/FindGUROBI.cmake), HiGHS (CPM), MPI (find), CUDA (enable_language at directory scope, CMakeLists.txt:39-178), Metal (OBJCXX + frameworks), Arrow/Parquet, yaml-cpp.

### Known-broken / fragile combos
1. **llfio `GIT_TAG develop`** — unpinned moving target; any upstream change can break all builds simultaneously [confirmed: Dependencies.cmake:130].
2. **Arrow CPM build on Windows+Clang** — hard-disabled with WARNING (`-Xclang --dependent-lib=msvcrt` space-in-flags breaks ExternalProject) [confirmed: Dependencies.cmake:276-283]. Workarounds documented in-file: conda arrow-cpp, MSVC, or Linux.
3. Wheel builds disable both MIP solvers (`DTWC_ENABLE_GUROBI=OFF`, `DTWC_ENABLE_HIGHS=OFF` in pyproject.toml:60 and python-wheels.yml:33-34) → `Method::MIP` fails at runtime inside every published wheel; configure prints the "No MIP solver available" WARNING (CMakeLists.txt:336-340).
4. Windows CUDA + VS generator: extensive Directory.Build.props / `CUDA_PATH_Vxx_y` workaround (CMakeLists.txt:46-172) — works but is generator-sensitive; Ninja path untested in CI.
5. MSVC OpenMP uses `/openmp:experimental` applied to **`dtwc_options` INTERFACE** (dtwc/CMakeLists.txt:113) — any consumer that doesn't link `project_options` compiles `#pragma omp` TUs serially; python/CMakeLists.txt:33-40 documents this exact bug class for the Python module.
6. A prebuilt `bindings/matlab/dtwc_mex.mexw64` is **committed to the repo** [confirmed: Glob] — stale-binary hazard.

### Presets (CMakePresets.json)
`clang-win` (Ninja, hardcoded `C:/Program Files/LLVM/bin/clang++.exe`), `clang-win-debug`, `msvc` (VS 17 2022), `gcc-linux` (Ninja), `clang-macos` (forces `/usr/bin/clang++` to avoid Homebrew LLVM libc++ mismatch with HiGHS — CMakePresets.json:55). All inherit `DTWC_BUILD_TESTING=ON`. No preset for wheels, MEX, CUDA, or MPI.

---

## 2. CI matrix (.github/workflows/)

| Workflow | Trigger | OS | Compiler | Builds/Tests | Publishes |
|---|---|---|---|---|---|
| ubuntu-unit.yml | push:develop, PR:main | ubuntu-latest | gcc-11, gcc-12, clang-14..17, + gcc-12 ASan/UBSan (7 jobs) | Debug, `DTWC_BUILD_TESTING=ON`, ctest | — |
| windows-unit.yml | push:develop, PR:main | windows-latest | default (MSVC/VS gen) | Debug + ctest | — |
| macos-unit.yml | push:develop, PR:main | macos-latest (arm64) | preset clang-macos + brew libomp | Release + HiGHS + ctest | — |
| python-tests.yml | push:develop, PR:main | ubuntu, macos, windows | default | uv pip install `.[test]` (source build), pytest, `dtwcpp.check_system()` — 18 jobs (Py 3.9–3.14) | — |
| python-wheels.yml | push:develop, tags v*, PR:main | ubuntu, macos, windows | cibuildwheel defaults | `pypa/cibuildwheel@v2.21`, CIBW_BUILD cp39–cp314, skip musllinux/win32/i686, `CIBW_ARCHS_MACOS: x86_64 arm64`, MACOSX_DEPLOYMENT_TARGET=10.15, brew libomp; sdist via `uv build` | wheel/sdist artifacts; on v* tags → Test PyPI (`environment: test-pypi`) then PyPI (`environment: pypi`), both **OIDC trusted publishing** (`id-token: write`, python-wheels.yml:74-103) |
| cuda-mpi-detect.yml | push:main,develop,**Claude**, PR:main | ubuntu (nvidia/cuda:12.6 container), macos, windows | gcc / clang / MSVC | CUDA compile (no GPU, arch 80), Linux MPI build + `mpiexec -n 2 unit_test_mpi`, macOS CUDA graceful-reject grep, macOS MPI build, Windows MS-MPI configure-only | — |
| documentation.yml | push:main,develop, PR:main | ubuntu | gcc-13 | Hugo + Doxygen + coverage (lcov→Codecov) | GitHub Pages |
| draft-pdf.yml | push/PR paths joss/** | ubuntu | — | JOSS paper PDF | artifact |

**Zero CI touches MATLAB/MEX** [confirmed: grep `matlab|mex` over .github → no matches]. **No workflow packages or releases executables.** Most workflows do NOT run on push to the working branch `Claude` (only cuda-mpi-detect does).

---

## 3. Wheel story

Exists [confirmed python-wheels.yml, pyproject.toml]:
- scikit-build-core backend (`build-dir = build_python/{wheel_tag}`), nanobind, cibuildwheel across all 3 runners, macOS dual-arch via `CIBW_ARCHS_MACOS: x86_64 arm64`, sdist, Test-PyPI→PyPI trusted publishing on tags. Version duplicated: pyproject.toml:7 `2.0.0` vs `VERSION` file read by CMake (CMakeLists.txt:10) — two sources of truth.

Missing / at risk for the 4 platform targets:
1. **cibuildwheel pin `v2.21` predates cp314** yet `CIBW_BUILD` requests `cp314-*` — cp314 wheels are silently not produced (unknown identifier is skipped) [inferred: pin at python-wheels.yml:24 vs :26; confirm from a CI run log].
2. **macOS-Intel wheels likely ship without OpenMP**: runner is arm64, `brew install libomp` installs arm64 libomp only; the x86_64 cross-build's `find_package(OpenMP)` fails → configure WARNING → serial wheel, silently (dtwc/CMakeLists.txt:118-123) [inferred: no CI assertion on `OPENMP_AVAILABLE` inside cibuildwheel; python-tests' check_system prints but does not fail]. Direct conflict with goal (b).
3. **No `CIBW_TEST_COMMAND`** — wheels are never imported/tested inside cibuildwheel; a wheel that imports but is serial or missing symbols passes CI.
4. **No Linux aarch64** (manylinux arm) target; skip list only excludes musllinux/i686/win32.
5. Wheels have **no MIP solver** (HiGHS OFF — pyproject.toml:60); no OpenMP runtime bundling strategy documented for Windows (LLVM libomp / vcomp via delvewheel) or macOS (delocate picks up libomp.dylib only if linked).
6. llfio configure-time git clone runs inside every cibuildwheel container — network dependency, and the quickcpplib patch is the load-bearing fix (Dependencies.cmake:134-217).
7. Metal is compiled into macOS wheels by default (`DTWC_ENABLE_METAL` default ON; not disabled in pyproject/cibw args) [confirmed by absence of `-DDTWC_ENABLE_METAL` in pyproject.toml:60 / python-wheels.yml:31-37].

---

## 4. MEX story

- Build path today: `cmake -DDTWC_BUILD_MATLAB=ON` → `find_package(Matlab COMPONENTS MX_LIBRARY)` → `matlab_add_mex(NAME dtwc_mex SRC dtwc_mex.cpp R2018a LINK_TO dtwc++)` (bindings/matlab/CMakeLists.txt:6,30-35). C++20 forced (:37). Install rules put the MEX + `+dtwc` package under `<prefix>/matlab` (:40-47).
- macOS R2024b workaround: clears `Matlab_HAS_CPP_API` so only the legacy C export map is used (:23-28).
- MATLAB-side introspection already exists: `dtwc.check_system()` calls `dtwc_mex('system_check')` returning struct with `openmp/openmp_threads/cuda/cuda_info/metal/metal_info/mpi` (bindings/matlab/+dtwc/check_system.m:24-72).
- **Missing automation**: no CI job builds the MEX on any platform (no MATLAB on runners; `matlab-actions/setup-matlab` not used); no packaging (mltbx or zip per platform); no `test_mex.m` execution in CI; committed `dtwc_mex.mexw64` is the de-facto distribution for Windows and can silently go stale.

---

## 5. Parallelisation story

Backend: **OpenMP only** — no std::execution, no std::thread, no TBB [confirmed: grep across dtwc/ → 0 hits].

Parallel regions (grep `#pragma omp`, dtwc/):
- `parallelisation.hpp:87` — generic `run_openmp` (`parallel for schedule(dynamic, chunk)`)
- `Problem.cpp:311` (fill distance matrix), `:242` (`critical(distByInd_init)`)
- `core/pruned_distance_matrix.cpp:172,184,216/225,375` + criticals `:101,305,456`
- `algorithms/fast_pam.cpp:58,174/181,213(critical)`
- `algorithms/fast_clara.cpp:144,198` (`if(chunk_size > 64)` clauses)
- `mpi/mpi_distance_matrix.cpp:134` (hybrid MPI+OpenMP)
- `omp simd` (vectorisation only, no threading): `core/lower_bound_impl.hpp:165,506`, `core/z_normalize.hpp:48,61,72,76`

Fallback & warning behaviour:
- No-OpenMP build: every `#pragma omp` becomes a no-op; the only runtime warning is in `get_max_threads()` (parallelisation.hpp:33-47) — one-time stderr message, **fires only if that function is called on the executed code path**. Regions in fast_pam/fast_clara/pruned_distance_matrix that don't route through `run_openmp`/`get_max_threads` run serially with no runtime warning [confirmed by reading parallelisation.hpp + grep — no other `_OPENMP`-absent warning sites].
- Configure-time: `message(WARNING)` when OpenMP not found (dtwc/CMakeLists.txt:118-123) — visible in logs only, build still succeeds. This is precisely the "silent sequential fallback" the user forbids: a wheel or exe built without OpenMP installs and runs quietly serial unless the user calls `check_system()`.
- OpenMP-present-but-1-thread (e.g. `OMP_NUM_THREADS=1`) produces no warning at all.
- `run()` (parallelisation.hpp:115-133) calls `omp_set_num_threads()` globally as a side effect of a per-call worker cap — process-wide mutation landmine for library embedding.

---

## 6. GPU gating story

- **CUDA**: gated by `DTWC_ENABLE_CUDA` (default OFF). Detection at directory scope in root CMakeLists (:39-178): Apple→hard off w/ WARNING; Windows→CUDA_PATH_Vxx_y env repair + Directory.Build.props pin for MSBuild; Linux→/usr/local/cuda hint; `check_language(CUDA)`→`enable_language`→`find_package(CUDAToolkit REQUIRED)`; missing nvcc → WARNING + auto-off. Source `dtwc/cuda/cuda_dtw.cu` added and `DTWC_HAS_CUDA` PUBLIC define set in dtwc/CMakeLists.txt:167-173.
- **Metal**: `DTWC_ENABLE_METAL` default ON, Apple-only (`enable_language(OBJCXX)`, frameworks Foundation+Metal; CMakeLists.txt:181-191, dtwc/CMakeLists.txt:176-186); silently disabled elsewhere. `metal_dtw.mm` compiled `-fno-objc-arc`.
- Runtime dispatch (Problem.cpp:397-499): `DistanceMatrixStrategy::CUDA/Metal` → runtime `cuda_available()` / `metal_available()` check → CPU brute-force fallback. **All fallback messages are `if (verbose)`-gated (Problem.cpp:404,450,470,477,493) — with verbose=false, GPU→CPU fallback is completely silent.** Goal (b) violation, second instance.
- Metal runtime emits its own stderr CPU-fallback notices for alloc failures (metal_dtw.mm:1392,1443,2031) — inconsistent with the verbose-gated Problem.cpp policy.
- CI covers CUDA compile-only (no GPU) on Linux and macOS graceful rejection (cuda-mpi-detect.yml:10-28,48-60). No GPU execution test in CI.

---

## 7. Existing introspection (base for goal (c))

- C++: `dtwc::cuda` `cudaGetDeviceCount`/`cudaGetDeviceProperties` (cuda/cuda_dtw.cu:919,926; cuda/gpu_config.cuh:54); `dtwc::metal::metal_available()` (metal/metal_dtw.hpp:71, .mm:1006); `get_max_threads()` (parallelisation.hpp:33).
- Python `_dtwcpp_core`: `CUDA_AVAILABLE`/`METAL_AVAILABLE`/`OPENMP_AVAILABLE`/`MPI_AVAILABLE` compile-time attrs, `cuda_available()`, `cuda_device_info()`, `metal_available()`, `openmp_max_threads()`, `system_info()` (python/src/_dtwcpp_core.cpp:813-939). Python `dtwcpp.check_system()` prints human-readable report (python/dtwcpp/__init__.py:219-262); `dtwcpp.device()` global default with warn-on-fallback in `_resolve_device` (`RuntimeWarning`, __init__.py:98-117 — note: Python layer DOES warn; C++ layer does not).
- MATLAB: `dtwc.check_system()` via `dtwc_mex('system_check')` (see §4).
- **Missing for goal (c)**: no `dtwc.test.*` namespace anywhere; no machine-readable pass/fail API (current check_system prints, returns None); no "prove parallelism actually engaged" test (e.g. timed 2-thread vs 1-thread run, or omp_get_num_threads inside a region); no GPU self-test that executes a kernel and validates a known result; Metal absent from Python `check_system()` output (only OpenMP/CUDA/MPI printed, __init__.py:236-259) though `METAL_AVAILABLE` is bound.

---

## 8. Gap list vs goals

### (a) Automated executables + MEX + wheels, 4 platforms
- No install/CPack/release workflow for `dtwc_main`/`dtwc_cl`; exes only built as top-level side effect, never published.
- No MEX CI at all; no `matlab-actions/setup-matlab` usage; no per-platform MEX artifacts; committed `.mexw64` binary is the only distribution.
- Wheels: cibuildwheel pin too old for cp314; no wheel import-test (`CIBW_TEST_COMMAND`); macOS x86_64 cross-build OpenMP unverified; no linux-aarch64; no repair-step audit (delocate/delvewheel handling of libomp); version string duplicated (VERSION file vs pyproject.toml).
- llfio unpinned + configure-time network (quickcpplib clone) makes every packaging job network-fragile.

### (b) Out-of-the-box parallelisation, no silent fallback
- OpenMP is optional-by-detection with warning-only configure behaviour; needs a `DTWC_REQUIRE_OPENMP`-style hard gate (default ON for shipped artifacts).
- Runtime warning exists only in `get_max_threads()`; most parallel regions never call it. Needs a single startup-time capability check.
- GPU→CPU fallbacks in Problem.cpp are `verbose`-gated (silent by default) — contradicts stated user requirement; Python layer warns but C++/CLI/MEX layers do not.
- No CI assertion that produced artifacts are parallel (e.g. fail wheel build if `OPENMP_AVAILABLE == False`).

### (c) `dtwc.test.parallelisation()` / `dtwc.test.gpu()`
- Building blocks exist in all three languages (§7) but: no `test` namespace, no structured return values, no exercised-kernel GPU validation, no engaged-threads proof, Metal missing from Python check_system, MPI rank/size not surfaced. Cross-language parity (CasADi-style) requires the same API shape in C++ (`dtwc::test::...`), Python, MATLAB.

### Landmines a refactor must not trip
1. quickcpplib text-patch (Dependencies.cmake:189-217) — regex depends on exact upstream text; any change to llfio fetch strategy must preserve or replace this for sandboxed wheel builds.
2. `project_options`→`/openmp:experimental` propagation to the nanobind module (python/CMakeLists.txt:33-40) — dropping `project_options` from any OMP-using consumer silently serialises it on MSVC.
3. macOS MEX `Matlab_HAS_CPP_API` clear (bindings/matlab/CMakeLists.txt:23-28) — required for R2024b legacy C API link.
4. clang-macos preset pins `/usr/bin/clang++` to avoid HiGHS libc++ mismatch (CMakePresets.json:55).
5. HiGHS needs forced `NDEBUG` (Dependencies.cmake:37-44) — Debug builds without it hit a false assertion.
6. CUDA on Windows relies on the generated Directory.Build.props + env repair (CMakeLists.txt:46-172); switching generator or moving CUDA detection into a function breaks `enable_language`.
7. STABLE_ABI only for Py≥3.12 (python/CMakeLists.txt:27-31) — a naive "abi3 everywhere" change breaks 3.9-3.11.
8. Arrow must stay OFF (or find_package-only) on Windows+Clang (Dependencies.cmake:276-283).
9. Most CI does not run on branch `Claude` — pushes there validate almost nothing except cuda-mpi-detect.
