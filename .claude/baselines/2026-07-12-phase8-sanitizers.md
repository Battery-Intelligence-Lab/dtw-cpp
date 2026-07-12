# Phase 8.2 sanitizer and capability-gate evidence — 2026-07-12

## Registered bands

- Every DTWC++ and test translation unit in the ASan build is compiled with
  address instrumentation.
- Full CTest: zero failures. Optional capabilities may skip with an explicit
  reason; capability absence must not be reported as a product failure.
- `ASAN_OPTIONS=halt_on_error=1:detect_leaks=0:strict_string_checks=1`.
  LeakSanitizer is handled separately on Linux because it is unavailable in
  the Windows runtime.

## Toolchain arbitration

LLVM 21.1.8's Windows ASan runtime was rejected as an execution environment.
After adding its runtime directory to `PATH`, a standalone program that only
throws, catches, and reads `std::runtime_error::what()` reproducibly terminated
with an ASan access violation. Both `clang++ -fsanitize=address` and
`clang-cl /fsanitize=address /EHsc` reproduced it, with and without ThinLTO.
The same standalone probe printed `expected` and exited zero under MSVC
19.50.35723 `/fsanitize=address /EHsc`.

The accepted build is therefore:

```text
build/asan-msvc
MSVC 19.50.35723.0
RelWithDebInfo, Ninja, /fsanitize=address, IPO OFF
OpenMP /openmp:experimental ON
HiGHS/Gurobi/CUDA/Metal/MPI/Arrow/YAML/llfio OFF
```

`compile_commands.json` contained 112 test translation units and all 112 used
`/fsanitize=address`; the DTWC++ production translation units were instrumented
as well. Catch2 itself is an external dependency and was not scored as project
coverage.

## Confirmed finding F3 — optional HiGHS test contract

The first valid MSVC ASan full run reached all tests and reported:

```text
97% tests passed, 3 tests failed out of 112
The following tests FAILED:
  42 - test_lagrangian_root (Failed)
  74 - unit_test_benders (Failed)
  86 - unit_test_mip (Failed)
```

Every failure was an unguarded solver-required case receiving the intended
typed message `HiGHS solver is unavailable; rebuild with
-DDTWC_ENABLE_HIGHS=ON`. No sanitizer diagnostic was present. The existing
no-solver state-preservation test remained active.

After solver-required cases were made to query the public
`highs_solver_available()` capability before setup, the focused no-solver gate
was:

```text
100% tests passed, 0 tests failed out of 3
test_lagrangian_root  Passed
unit_test_benders    Skipped
unit_test_mip        Passed
```

The positive control rebuilt the same three targets with
`DTWC_ENABLE_HIGHS=ON`; all three executed and passed, so the guard does not
weaken solver-enabled coverage.

## Full gates after F3

MSVC ASan, optional solvers off:

```text
ctest --test-dir build/asan-msvc -j 4 --output-on-failure
100% tests passed, 0 tests failed out of 112
Total Test time (real) = 106.08 sec
9 explicit capability skips: llfio 2, CUDA 2, Arrow/Parquet 1, Metal 3,
and the all-HiGHS Benders target 1.
```

Canonical Clang 21.1.8 Release, HiGHS + llfio on:

```text
ctest --test-dir build/highs-1151 -j 4 --output-on-failure
100% tests passed, 0 tests failed out of 112
Total Test time (real) = 35.56 sec
6 explicit capability skips: CUDA 2, Arrow/Parquet 1, Metal 3.
```

UBSan and the TSan/Archer fallback remain open in the parent sanitizer lens.
