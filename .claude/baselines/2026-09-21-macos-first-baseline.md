# 2026-09-21 — first macOS baseline on the new development machine

Branch `design-2.0`, HEAD `9c08074`. Read-only run: nothing in the tree was changed to obtain it.

## Machine and toolchain

| Item | Value |
| --- | --- |
| Host | Apple Silicon Mac, macOS (Darwin 25.6.0), Command Line Tools only (no full Xcode) |
| Compiler | Apple clang 21.0.0 (`/usr/bin/clang++`), libc++ |
| Installed today | `brew install cmake ninja doxygen graphviz libomp` → CMake 4.4.3, Ninja 1.13.2, Doxygen 1.18.0, Graphviz 16.1.0, libomp |
| Python | `uv` present (`~/.local/bin/uv`); system `python3` used only for stdlib scripts |

## Commands

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp
cmake --build --preset clang-macos          # 434 targets, 2 min 31 s wall
cd build && ctest -C Release -j1 --output-on-failure --timeout 1200   # 129 s wall
```

Configuration that resulted: Release, IPO ON, `DTWC_ENABLE_NATIVE_ARCH=ON`, OpenMP ON
(`-Xclang -fopenmp`, libomp), HiGHS ON, llfio ON, **Metal ON**, YAML ON, Arrow OFF, CUDA OFF,
MPI OFF, Gurobi probed and absent. 131 tests registered.

## Result: 126 / 131 passed, 2 skipped (CUDA), 5 failed — all five on the floor regex only

Every failing binary printed `All tests passed`. What failed is the W0 floor
(`tests/floors.cmake`, commits `d21ffee` / `22c4c5f`), which was measured from Windows logs only:

| Test | Floor `assertions;cases` | macOS actual |
| --- | --- | --- |
| `test_multivariate_adversarial` | 12646;25 | 9998 in 25 |
| `test_lower_bounds_adversarial` | 36509;29 | 33653 in 29 |
| `test_env_device` | 22;8 | 18 in 8 |
| `test_runtime_loudness_gpu` | 4;2 | 3 in 2 |
| `unit_test_cli_checkpoint` | 33;2 | 31 in 2 |

**Finding.** Case counts are identical across platforms; assertion counts are not (data-dependent
loops fed by `std::uniform_*_distribution`, which differs between standard libraries, and
platform-conditional assertions such as the Metal-present branch). A floor must be the minimum
over every supported platform, or it turns a passing platform red. `macos-unit.yml` builds this
same preset, so **macOS CI is red on this branch** until the floors are re-measured.

**Fix (W0, not done here).** `scripts/measure_test_floors.py` already takes the minimum over the
logs it is given. Re-run it with a `ctest -V` log from each of: Windows clang, MSVC Debug,
macOS Apple clang, Linux GCC. Record the four logs in the W0 run-log. Consider flooring the
assertion count at a fraction of the measured minimum for tests whose count is data-dependent;
the case floor can stay exact.

## Second regression found the same day (verified by running the script)

`python3 scripts/check_docs_contract.py` fails at HEAD:

```text
AssertionError: D2 CTest drift: expected one test_lb_keogh_derivation policy block
```

`d21ffee` replaced the `if(TARGET test_lb_keogh_derivation) … endif()` blocks in
`tests/CMakeLists.txt` with `dtwc_add_test(...)`, but `check_docs_contract.py:1195-1203` (D2) and
its D3 twin near `:1843` still pin the old block text. This script runs in
`.github/workflows/documentation.yml`, so **the docs workflow is red on this branch**.
`check_record_hygiene.py` and `check_repo_hygiene.py` both pass.

## Timing (serial, Release) — slowest tests

| Test | Seconds |
| --- | --- |
| `test_mip_backend_guards` | 35.6 |
| `unit_test_distance_matrix_properties` | 14.1 |
| `unit_test_clustering_algorithms` | 14.1 |
| `unit_test_checkpoint` | 12.7 |
| `test_problem_api_2_0` | 5.5 |
| `unit_test_variant_distmat` | 4.5 |

The whole suite takes about two minutes here. The Windows figures in ledger rows T-04 / T-12
(218 s, 116 s, 57 s, 52 s) are roughly 6–8× larger, so re-measure on the target CI runner before
spending effort shrinking fixtures.

## Codegen probe (feasibility of the "inspect the ASM, not only the clock" check)

A throw-away translation unit wrapped five hot functions in `extern "C" noinline` entry points and
was compiled with the project's real Release flags (taken from `build/compile_commands.json`:
`-O3 -march=native -fassociative-math -freciprocal-math -fno-signed-zeros -fno-trapping-math
-fno-math-errno -fno-finite-math-only …`) plus Clang's
`-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize`.

| Function (loop) | Verdict on Apple clang 21 / arm64 | Compiler's reason |
| --- | --- | --- |
| `lb_keogh` (`core/lower_bound_impl.hpp:189`) | **vectorised**, width 2 × interleave 4 (NEON) | — |
| `z_normalize`, 3 passes (`core/z_normalize.hpp:48,61,72`) | **vectorised**, width 2 × interleave 4 | — |
| `compute_envelopes` (`core/lower_bound_impl.hpp:102,103,118,119`) | not vectorised | early-exit loop with operations that cannot be speculated (monotonic-deque sliding min/max) |
| `dtw_kernel_linear` inner recurrence (`core/dtw_kernel.hpp:275`) | not vectorised | value used outside the loop is not a reduction — `short_side[i-1]` was written by the previous iteration (loop-carried dependency) |
| `dtw_kernel_linear` column loop (`:266`), first column (`:260`), banded loops (`:455,:489`) | not vectorised | early-exit loop with writes to memory / same dependency |

Reading: the elementwise helpers already get SIMD from the compiler and that is now checkable; the
DP recurrence cannot be vectorised along a row by construction, which is the mechanism behind the
recorded "SIMD killed" verdict. Any SIMD win for the recurrence has to come from a different axis
(several pairs in lockstep, or anti-diagonals) and goes through the R2-D17 door as a measured
prototype. Gotcha met on the way: compiling with `-I<repo root>` on a case-insensitive filesystem
makes the root `VERSION` file shadow the C++ header `<version>`; never put the repo root on an
include path.

## Environment consequences for the campaign

- Metal is now locally testable (`test_metal_correctness` ran, 1.6 s). The `[BLOCKED-ENV]` tag the
  spec puts on Metal moves to **CUDA** on this machine; CUDA work needs CI, the RTX box or ARC.
- MATLAB and MPI are not installed here; `matlab_suite` and `unit_test_mpi` did not exercise them.
