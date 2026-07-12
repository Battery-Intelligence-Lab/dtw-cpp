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

## UBSan arbitration and confirmed finding F4

The accepted undefined-behaviour build was Clang 18.1.3 under Ubuntu 24.04
WSL, RelWithDebInfo, IPO and optional dependencies off, and
`-fsanitize=undefined`. OpenMP used Ubuntu's exact Clang-18 `libomp` packages,
extracted locally under the ignored build tree rather than installed globally.

The first full UBSan run emitted no sanitizer diagnostic, but two literal
oracles disagreed with Windows: SSG barycenters and OneBatchPAM's one-sweep
medoids. Both failures reproduced at `OMP_NUM_THREADS=1` and 4. The cause was
the C++ standard's deliberately unspecified mapping in `std::shuffle`,
`std::sample`, and the standard distributions: the same `mt19937_64` stream
was translated differently by MSVC STL and libstdc++.

F4 establishes one `portable-v1` seeded mapping:

- Lemire multiply-high bounded integers with an unbiased rejection interval;
- an ascending Fisher-Yates shuffle;
- an exact 53-bit binary64 draw and strict cumulative weighted selection, so a
  zero draw cannot select a zero-weight/already-selected medoid;
- stable selection sampling with a fixed full-population bounded-call schedule
  and O(sample-size) sampling scratch, removing FastCLARA's avoidable 8*N-byte
  index pool (rejection may consume extra raw engine words).

Seeded barycenter, FastPAM, Lloyd initializers, OneBatchPAM, CLARANS, and both
FastCLARA paths use that map. The unseeded Tier-2 `std::mt19937`
initializer/FastPAM path is intentionally unchanged. Permanent golden tests
pin bounded rejection and draw consumption, weighted zero boundaries, two
consecutive samples, int/int64 identity, full/empty sampling, shuffle order,
and floating draws. On compilers with `unsigned __int128`, one million
independently generated products compare the portable four-limb multiply-high
against the native 128-bit arbiter.

The non-degenerate barycenter fingerprint is literal-equal on Windows Clang and
Ubuntu Clang, including first center
`{-7.0173908226487498,-6.10105517828847255,-5.73290735028454357,
-6.01109573627146609,-6.67454216978708459}` and total cost
`3.9239742509803337`. The OneBatch tolerance discriminator now first pins its
portable initial `{3,2}` state, proves 20% accepts the 33.3% improvement to
`{3,1}`, and proves 34% rejects it. FastCLARA's six vendor-dependent medians
are replaced by literal portable-v1 medians `{103,86,121,108,106,105}`.
CLARANS seed 42 pins medoids `{5,14}`, the ten-zero/ten-one label partition,
and cost `10.0`; Windows and Ubuntu produce the same literals.

## ThreadSanitizer toolchain preflight

GCC 13 TSan plus `libgomp` was rejected: it reported a race in a race-free
OpenMP reduction and then terminated with `unexpected memory mapping`. Clang
18 plus the locally extracted `libomp` passed the same race-free reduction and
reported the deliberate-race control at its exact source line. The accepted
project build was:

```text
build/tsan-wsl
Clang 18.1.3, Ubuntu 24.04 WSL, RelWithDebInfo, IPO OFF
libomp 18.1.3-1ubuntu1, optional dependencies OFF
-fsanitize=thread -fopenmp
TSAN_OPTIONS=halt_on_error=1:exitcode=66:ignore_noninstrumented_modules=1
OMP_NUM_THREADS=4
```

`ldd build/tsan-wsl/bin/unit_test_fast_clara` resolved `libomp.so.5` from the
local Clang-18 directory. The following required concurrency classes ran
serially as processes and with four OpenMP workers internally:

```text
unit_test_distance_matrix_properties
unit_test_parallelisation
unit_test_fast_pam
unit_test_faster_pam
test_fast_pam_adversarial
unit_test_pruned_distance_matrix
test_pruned_openmp_contract
unit_test_clustering_algorithms
unit_test_fast_clara
unit_test_clarans
test_tier1_cpp_api
```

The original ten-target batch exited zero in 180.4 seconds. After CLARANS was
added to the portable contract, its focused TSan rebuild and run also exited
zero (31.1 seconds including relinking). All eleven targets emitted no report
and never used TSan's registered exit code 66.

## Final sanitizer and canonical gates after F4

MSVC ASan was rebuilt after the portable-map change. `compile_commands.json`
contained 113/113 instrumented test translation units and 29/29 instrumented
DTWC++ production translation units:

```text
ASAN_OPTIONS=halt_on_error=1:detect_leaks=0:strict_string_checks=1
OMP_NUM_THREADS=4
ctest --test-dir build/asan-msvc --output-on-failure -j 4
100% tests passed, 0 tests failed out of 113
Post-review serial rerun: 113/113 passed, 0 failed
CTest log: 05:26:52--05:32:42 BST (shared three-gate run)
9 explicit capability skips
```

Clang UBSan, four OpenMP workers:

```text
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
ctest --test-dir build/ubsan-wsl --output-on-failure
100% tests passed, 0 tests failed out of 113
Total Test time (real) = 148.75 sec
9 explicit capability skips
```

Canonical Windows Clang 21.1.8 Release, HiGHS and llfio on:

```text
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 113
CTest log: 05:31:05--05:33:20 BST (shared three-gate run)
6 explicit capability skips
```

Result: AddressSanitizer, UndefinedBehaviorSanitizer, and the pre-authorized WSL
ThreadSanitizer fallback are CLEAN. No suppression was needed.
