# 2026-10-07 — WM: the Windows wheel's fill, MSVC against clang-cl — clang-cl recommended by the registered rule (run 2026-10-08)

**Result [confirmed], run 2026-10-08 01:58–02:01 by the orchestrator (3 rounds; load records 01:58:16 and 02:01:33,
machine quiet): by the registered rule, clang-cl is recommended** ("clang-cl recommended (>= 1.2x at 24 threads on (a) or
(b), no shape below 0.95x)"). clang-cl / MSVC, median Gcell/s over 3 rounds: the lanes fill (a) 3.14× and (b) 3.33× at 24
threads (4.40× and 4.68× at one thread); the per-pair fill (c) 1.24× at 24 threads (1.09× at one); one
`dtwcpp.distance.dtw` call (d) 1.00× unbanded and 1.04× at band 100, so the smallest ratio is 1.00 (the rule's floor is
0.95). Every shape returns bitwise the same result from both wheels (`same` in all 8 rows) and `problems: none`. Volkan
decides the wheels' compiler: the shipping costs he weighs against these ratios are below the table. The wheels were
built from `8a63cda3` and checked 2026-10-07 23:52 – 10-08 00:10 (agent WM); HEAD at the run, `75d61cfe`, changed
`Problem.cpp/.hpp`, `core/distance_matrix.*`, `core/matrix_io.hpp` and the binding (`git diff --stat 8a63cda3 75d61cfe`)
but none of the kernel files (`dtw_kernel.hpp`, `dtw_lanes.cpp`, `dtw_dispatch.cpp`) [confirmed].

Question (PLAN phase G row WM; Volkan 10-02, DECISIONS §3: "Measure first"): is a clang-cl Windows wheel faster than
the MSVC wheel CI builds today on the fill, enough to pay for shipping LLVM's `libomp.dll`? Background: cl packs no
lanes (DECISIONS 09-29, `/Qvec-report 1200`), clang does; a clang-cl wheel ships `libomp.dll`, which can abort with OMP
Error #15 beside Intel's OpenMP (DECISIONS 10-02).

## Registered (the orchestrator's, fixed before any run), verbatim

> this is a measurement for Volkan, not a pass/fail band. The rule for a recommendation: clang-cl is recommended if,
> at 24 threads, it is ≥ 1.2× on shape (a) or (b) and no shape at either thread setting is below 0.95×; MSVC stays
> recommended if every shape at both settings is within 0.95–1.05×; otherwise no recommendation, and the table goes to
> Volkan as it is. Shipping costs (libomp.dll in the wheel, licence notice) are reported beside the numbers.

How `driver.py report` reads it: ratio = clang-cl / MSVC of the median Gcell/s over 3 rounds; (d)'s two variants
(unbanded, band 100) count as two shapes. Before any timing the orchestrator changed the kit so that (d) (one pair per
`dtwcpp.distance.dtw` call, single-threaded whatever OpenMP is set to) is timed pinned only (`driver.py` `threads_for`;
the original is `driver.orig.py`): unpinned it could land on an E-core and swing the ratio by itself. So the rule's table
has 8 rows (a, b, c at 1 and 24 threads; both (d) variants at one thread), not 10; the rule and its text are unchanged,
and (d) has no 24-thread row that could fall below 0.95×.

## Load (`load.txt`; its first entry, 2026-10-08 00:08:16, is the kit check on the shared machine, total 8.4–18.8 %, not a run)

| when | total CPU (`typeperf`, 10 × 1 s) | busiest processes, % of all 24 logical processors |
| --- | --- | --- |
| 2026-10-08 01:58:16 (before the first round) | 3.4–6.5 % | `system` 1.62 %, `serviceshell` 0.82 %, `spotify` 0.63 %, `sspservice` 0.54 %, `wmiprvse` 0.43 %, `tpython` 0.22 %; COMSOL none |
| 2026-10-08 02:01:33 (after the last round) | 2.7–6.9 % | `system` 1.69 %, `serviceshell` 0.69 %, `spotify` 0.69 %, `wmiprvse` 0.26 %, `sspservice` 0.15 %; COMSOL none |

No load sample lies between the two records. `run_full.out` holds the whole run's output (load, rounds, load, report).

## Results (`tables.md`, verbatim; the 48 measurement lines of 3 rounds × 8 rows × 2 wheels are in `raw.txt`)

Medians over 3 round(s), Gcell/s; ratio = clang-cl / MSVC (above 1: clang-cl faster).

| shape | threads | MSVC Gcell/s | clang-cl Gcell/s | clang-cl/MSVC | shortest timed s | result |
| --- | --- | --- | --- | --- | --- | --- |
| (a) N 2000, L 100, unbanded: lanes fill | 1 | 1.482 | 6.519 | **4.40** | 3.07 | same |
| (a) N 2000, L 100, unbanded: lanes fill | 24 | 30.789 | 96.740 | **3.14** | 0.60 | same |
| (b) N 500, L 1000, band 100: lanes fill | 1 | 1.319 | 6.175 | **4.68** | 3.82 | same |
| (b) N 500, L 1000, band 100: lanes fill | 24 | 30.970 | 103.170 | **3.33** | 0.69 | same |
| (c) N 1000, L 90-110, unbanded: per-pair fill | 1 | 1.479 | 1.607 | **1.09** | 3.12 | same |
| (c) N 1000, L 90-110, unbanded: per-pair fill | 24 | 29.502 | 36.636 | **1.24** | 0.51 | same |
| (d) dtw, one pair, L 1000, unbanded | 1 | 1.383 | 1.380 | **1.00** | 0.50 | same |
| (d) dtw, one pair, L 1000, band 100 | 1 | 0.854 | 0.885 | **1.04** | 0.50 | same |

rule: clang-cl recommended (>= 1.2x at 24 threads on (a) or (b), no shape below 0.95x)
problems: none

From the table, not investigated: the ratio falls from one to 24 threads on (a) and (b) because MSVC scales further (a:
1.482 → 30.789 Gcell/s, 20.8×; clang-cl 6.519 → 96.740, 14.8×), while on (c) it rises (1.09 → 1.24).

## Shipping costs (beside the numbers)

- The wheel would carry `libomp.dll` (777,728 B, each CPython wheel its own copy). CI would need a pinned LLVM on the
  Windows runner, the clang-cl OpenMP link (the workaround above, or a fix in `dtwc/CMakeLists.txt`), the FP flags in a
  spelling clang-cl takes (`/clang:` prefixes; not tried), and a repair step that copies the DLL into the wheel (e.g.
  delvewheel; the workflow sets no Windows repair command today).
- Licence: LLVM's OpenMP runtime is Apache-2.0 with LLVM exceptions; the exception covers code a compilation embeds,
  not a redistributed DLL, so the wheel ships LLVM's licence text (`license-files`, `THIRD_PARTY_LICENSES.md`)
  [inferred from the licence text; Volkan to confirm].
- Two OpenMP runtimes in one process (MKL's or PyTorch's `libiomp5md.dll` beside `libomp.dll`): OMP Error #15
  (DECISIONS 10-02); a renamed copy does not avoid it, the check is at run time [inferred, not tested here].
- Today's MSVC wheel takes `VCOMP140.DLL` and `MSVCP140.dll` from the system's VC++ redistributable (loaded from
  System32 here); the clang-cl wheel needs `MSVCP140.dll` the same way. Neither wheel bundles them.

## Recipes (all under `C:/D/git/wt/tmp/WM/`; worktree `C:/D/git/wt/WM`, detached at `8a63cda3`, no tracked file changed)

Common: uv 0.10.2, CPython 3.12.12 (uv-managed), `uv pip install -v --no-cache` from the worktree; build isolation
pulled scikit-build-core 1.1.1 and nanobind 3.1.0; CMake 4.2.3. CI's `CIBW_CONFIG_SETTINGS` cmake.args verbatim
(`DTWC_BUILD_PYTHON=ON`, `GUROBI OFF`, `HIGHS OFF`, `LLFIO ON`, `BUILD_TESTING OFF`, `BUILD_BENCHMARK OFF`,
`CMAKE_BUILD_TYPE=Release`), passed as repeated `-C cmake.args=`; the backend received them as one list (log line
`build_wheel(..., {"cmake.args": [...]})`). Additions that change no compile or link flag: `-C build-dir=` (tree kept),
`-C build.verbose=true` (command lines in the log), `CMAKE_BUILD_PARALLEL_LEVEL=8`, the main checkout's
`.venv/Scripts` taken off PATH.

- **MSVC** (`build_msvc.sh`, log `build-msvc.log`, tree `build-msvc/`): `CMAKE_GENERATOR` unset → "Building for:
  Visual Studio 18 2026" (MSBuild 18.0.5, `/m:8 /v:n`), "CXX compiler identification is MSVC 19.50.35723.0" (toolset
  14.50.35717), SDK 10.0.26100.0. This is the local Visual Studio's cl; CI's runner image has its own MSVC version
  (not checked).
- **clang-cl** (`build_clang.bat`, log `build-clang.log`, tree `build-clang-omp/`): `vcvars64.bat` (VS 18, MSVC 14.50
  headers and libraries), `CMAKE_GENERATOR=Ninja` (Ninja 1.13.2, chocolatey), `-DCMAKE_C_COMPILER` /
  `-DCMAKE_CXX_COMPILER=C:/Program Files/LLVM/bin/clang-cl.exe` → "Clang 21.1.8 with MSVC-like command-line";
  CMakeCache: `CMAKE_LINKER` lld-link, `CMAKE_AR` llvm-lib. Not the VS generator with `-T ClangCL`: this VS 18 has no
  ClangCL toolset (`VC/Tools/Llvm/x64/bin` holds clang-format and clang-tidy only), and it would bring VS's LLVM.
  **Workaround** (the one addition to CI's cmake.args): `-DCMAKE_MODULE_LINKER_FLAGS=C:/PROGRA~1/LLVM/lib/libomp.lib`.
  Without it the module's link fails (`build-clang-1-link-failed.log`: `lld-link: error: undefined symbol:
  omp_get_max_threads`, `__kmpc_fork_call`, …, 12 symbols). Cause: CMake's `MSVC` is true for clang-cl, so
  `dtwc/CMakeLists.txt`'s `if(MSVC)` OpenMP branch gives `dtwc_core` only `/openmp:experimental` and links no runtime;
  cl's objects name `vcomp.lib` themselves, clang-cl's name none (a probe object's `.drectve` holds only msvcrt.lib and
  oldnames.lib; clang-cl adds `libomp.lib` only when it drives the link, and CMake calls lld-link directly). FindOpenMP
  "found" OpenMP from the cache values the Clang-on-Windows block seeds (`-fopenmp`, which clang-cl ignores). Side
  effect of the -D: CMake's default `/machine:x64` leaves the module's link line; lld-link takes x64 from the objects.

## Flags (from the logs; `-D`/`-I` dropped)

| TU / step | MSVC wheel | clang-cl wheel |
| --- | --- | --- |
| lanes `dtwc/core/dtw_lanes.cpp` | one CL batch for all of `dtwc_core`: `/O2 /Ob2 /GL /EHsc /MD /Gy /arch:AVX2 /fp:precise /std:c++20 /we4062 /openmp:experimental /fp:contract` | `/O2 /Ob2 -std:c++20 -MD -flto=thin -march=x86-64-v3 /openmp:experimental -Werror=switch … -fno-lto` (last, so native code: the object is COFF x64, magic `64 86`) |
| per-pair entry `dtwc/core/dtw_dispatch.cpp` (`resolve_dtw_fn`: the fill's per-pair kernels) | the same batch | the lanes line without `-fno-lto`: ThinLTO bitcode (magic `42 43 c0 de`) |
| binding `python/src/_dtwcpp_core.cpp` (`dtwcpp.distance.dtw` → `dtwBanded`; `NOMINSIZE`) | `/MP /O2 /Ob2 /GL /EHsc /MD /GS- /Gy /arch:AVX2 /fp:precise /std:c++20 /bigobj /openmp:experimental /fp:contract` | `/O2 /Ob2 -flto=thin -march=x86-64-v3 /bigobj /GS- /openmp:experimental`; bitcode |
| FP model | `/fp:precise /fp:contract` | the project's GNU spellings (`-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math`, `-fno-finite-math-only`) are on every line and **ignored**: 178 "unknown argument ignored in clang-cl" warnings (26 TUs); clang-cl's defaults apply (`-fmath-errno -ffp-contract=on`, a `-###` probe) |
| link | `link.exe … /OPT:REF /OPT:ICF /LTCG:incremental /DLL` (LTCG) | `lld-link /dll libomp.lib …` (ThinLTO of the bitcode objects in lld-link) |
| `dumpbin /dependents` | `VCOMP140.DLL`, MSVCP140, VCRUNTIME140(_1), python312, KERNEL32, USER32, ADVAPI32, api-ms-win-crt-* | `libomp.dll` + the same set without VCOMP140 |
| module | 1,497,600 B, sha256 `699cd379…43f1f5` | 1,795,584 B, sha256 `ee218c85…fc720b` |

AVX2 in the clang-cl lanes: the f64 L1 kernel's inner loop (`lanes-clang-f64-l1-inner-loop.txt`, `llvm-objdump -d -C`
of `build-clang-omp/bin/CMakeFiles/dtwc_core.dir/core/dtw_lanes.cpp.obj`) is packed on ymm: `vbroadcastsd` x[i], then
per 4-lane half `vsubpd`, `vandpd` (|·|), two `vminpd`, `vaddpd`, `vmovupd`. Whole-module census (`dumpbin /disasm`):
`vminpd ymm` 4 (MSVC) / 20 (clang-cl), `vaddpd ymm` 0 / 18 — consistent with cl packing no lanes; not checked per
function (the MSVC build writes no PDB).

## Checks (`check_wheel.py` → `check-msvc.txt`, `check-clang.txt`)

Each `dtwcpp` loads from its own venv; `OPENMP_AVAILABLE` True; a 40 × 100 fill (the first OpenMP use) takes the
process from 27 to 50 threads (a team of 24); `dtwcpp.test.parallelisation()` engages 24; the fill equals
`dtwcpp.distance.dtw` exactly. OpenMP DLL held by the process: MSVC `C:\windows\SYSTEM32\VCOMP140.DLL`; clang-cl
`C:\D\git\wt\venv\wm-clang\Lib\site-packages\dtwcpp\libomp.dll`. The two modules differ (sha256 above).

libomp for this measurement: `C:\Program Files\LLVM\bin\libomp.dll` (LLVM 21.1.8; 777,728 B; sha256 `5e6ac41e…94466`;
FileVersion 20140926, ProductVersion 5.0) copied next to the module in the clang venv. Without it the import fails
("DLL load failed while importing _dtwcpp_core: The specified module could not be found") although LLVM's bin is on
PATH: Python ≥ 3.8 does not search PATH for an extension's DLLs.

## Equality (`equality.py` → `equality.txt`; fills at the OpenMP default)

Every timing shape is **bitwise identical** between the wheels: (a) 4,000,000 entries, (b) 250,000, (c) 1,000,000, (d)
both values; max abs and rel difference 0. `fast_pam(prob, 5, 100, 42)` on (a): the same labels, medoids
[1304, 1324, 73, 1110, 586] and total cost 265345.7524956819. (L1 has no multiply to contract, so this is expected.)

## The kit

- `gen_data.py` (run once; numpy 2.5.3, seed 20261007): random walks, f64, `data/*.npy`, `data/manifest.json`
  (sha256 per file; every process checks them). (a) 2000 × 100; (b) 500 × 1000; (c) 1000 series, lengths uniform in
  90–110; (d) 2 × 1000.
- `measure.py <shape> <threads> [--smoke]`, one process per measurement, in a wheel's own python, public API only:
  `Problem.set_distance(band=, metric="l1")`, `set_data`, then `refresh_distance_matrix()` + timed
  `fill_distance_matrix()` repeated until ≥ 0.5 s, after a ≥ 0.2 s warm-up on the first quarter of the series; (d)
  `dtwcpp.distance.dtw(x, y, band=)` in a loop, ≥ 0.5 s after 0.1 s. Threads 1: `OMP_NUM_THREADS=1` and
  `SetProcessAffinityMask` to logical CPU 12 (a P-core, TB's record) before dtwcpp loads, read back. Threads 24: no
  `OMP_*`/`KMP_*` variable. Asserts the module is the venv's and the engaged thread count (1 or 24). Prints
  Gcell/s with cells = N(N−1)/2 × cells per pair (L² unbanded; L(2b+1) − b(b+1) for band b; (c) Σ_{i<j} L_i L_j),
  the module's hash and a hash of the result.
- `driver.py run | smoke | report [smoke]`: 3 rounds; each round runs every shape × its thread settings ({1, 24}; (d)
  {1} only, see Registered) once per wheel, MSVC first in rounds 0 and 2, clang-cl first in round 1 → `raw.txt`;
  `report` → medians, ratios, the rule's outcome, checks (3 rounds each, same result and module across rounds, exit 0,
  engaged threads) → `tables.md`.
- `load.sh` (TB's, writing `load.txt`); `run.sh` = load record, `driver.py run`, load record, `driver.py report`.
- Smoke (before the (d) change, by file times [inferred]): two tiny rounds (the second after a one-line warm-up edit): 20
  of 20 processes exit 0, engaged 1/24 as set, affinity `0x1000`, results the same in both wheels, "problems: none".
  Its numbers go in no table.
- Duration of `run.sh` (Git Bash): predicted about 5 min (±50 %), observed 3.5 min (01:58:16 to 02:01:44).

Caveat: the fill's timed interval includes the NaN initialisation of the packed matrix that `refresh_distance_matrix`
defers to the fill (16 MB for (a), small against a ≥ 0.5 s fill; not measured).
