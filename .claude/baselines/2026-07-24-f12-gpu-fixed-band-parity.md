# F12 GPU fixed-band and sentinel parity - 2026-07-24

## Scope and base

- Branch: `Claude`
- Base commit: `e529127111cdc4c01f1212d64ca5aec4eeca9d5d`
  (`docs: record archive parser falsification`)
- Subject: canonical fixed-band geometry in every CUDA DTW kernel family,
  overflow-safe `INT_MAX` handling, and exact translation of a device
  `FLT_MAX` no-path value to the public double `DBL_MAX` sentinel in CUDA and
  Metal result APIs.
- Real CUDA build: `build/cuda-verify` (Ninja, Release, MSVC host, CUDA 13.0).
- Canonical regression build: `build/highs-1151` (clang, Ninja, Release,
  HiGHS ON, llfio ON, Arrow OFF).
- This registration precedes the new F12 tests and every decisive F12
  execution. The existing D1 oracle and scratch recomputations used to select
  the fixture are prior/exploratory evidence, not a substitute for the new
  executable backend gates.

The killed-ideas section, plan archive, and `.claude/LESSONS.md` were searched
before choosing the repair. No killed fixed-band or sentinel design is being
reopened. The binding lessons require a finite exact no-path sentinel, an
independent unequal-length full-matrix oracle, and a non-degenerate fixture.

## Confirmed inherited discrepancies

Source inspection at the base commit found endpoint-scaled `slope`/`window`
geometry and signed `band + 1` in six CUDA kernels:

- pairwise wavefront, warp, and register-tile:
  `dtwc/cuda/cuda_dtw.cu:223-269,444-454,614-636`;
- one/K-vs-N wavefront, warp, and register-tile:
  `dtwc/cuda/cuda_dtw.cu:1796-1818,1949-1957,2057-2072`.

Both CUDA FP32 result-copy loops and both Metal result-copy loops blindly cast
device floats to public doubles. Therefore an exact device `FLT_MAX` sentinel
becomes approximately `3.4e38`, not the required
`numeric_limits<double>::max()`. Metal's ordinary fixed-band geometry is
already canonical, but `band * 20` in host selection and `k + band` in four
wavefront kernels are not source-safe for `INT_MAX`.

Existing GPU tests do not arbitrate this discrepancy: the CUDA unequal-length
case is random, uses a single band, and calls the production CPU banded route
as its reference; Metal's banded cases are equal-length.

## Preregistered independent oracle

The discriminator is reused from the independently validated D1 derivation:

```text
x = [0, 1, 0, 2, 0]       n = 5 samples
y = [0, 0, 0, 0, 0, 0, 2] m = 7 samples
|n-m| = 2 samples
```

A local full `(n+1)*(m+1)` dynamic-programming matrix evaluates only cells
with `|i-j| <= band`; it shares no production row-bound, rolling-buffer, CUDA,
or Metal helper. A third arbiter recursively enumerates admissible monotone
paths. The exact registered ledger is:

| band | position | admissible paths | L1 | squared L2 |
|---:|---|---:|---:|---:|
| 1 | below endpoint gap | 0 | no path / `DBL_MAX` | no path / `DBL_MAX` |
| 2 | at endpoint gap | 696 | 5 | 9 |
| 3 | above endpoint gap | 1,143 | 3 | 5 |
| `INT_MAX` | full coverage | 1,289 | 3 | 5 |

All finite costs are small integers and exactly representable in FP32 and
FP64. No tolerance is permitted. A no-path assertion compares exact bit-level
public value identity with `numeric_limits<double>::max()`; `isfinite`,
positivity, widened `FLT_MAX`, and approximate equality are false-greens.

The supplementary singleton discriminator is also inherited from D1 and
registered here before the GPU run:

```text
singleton_x = [0]
singleton_y = [1, 2, 3]
```

At band 1 it has no path and must return exact public `DBL_MAX`; at band 2 and
`INT_MAX` it must return exactly `6` for L1 and `14` for squared L2, in both
orientations. This specifically kills the inherited CUDA `M > 1` condition
that otherwise disables banding for a one-sample shorter series.

The inherited endpoint-scaled CUDA fingerprint is diagnostic:

| band | L1 | squared L2 |
|---:|---:|---:|
| 1 | 5 | 9 |
| 2 | 3 | 5 |
| 3 | 3 | 5 |

## Registered CUDA route matrix

The permanent FP64 test must drive the real public pairwise and external-query
one-vs-N APIs through every distinct device-kernel family:

| requested route | fixture inventory | required reported kernel |
|---|---|---|
| Auto | `{x,y}` | `warp` |
| RegTile | `{x,y}` | `regtile_w4` |
| RegTile | `{x,y,filler_129}` | `regtile_w8` |
| Wavefront | `{x,y}` | `wavefront` |

The length-129 filler selects W8 but is not an oracle operand. Each route must
return the exact ledger for both metrics and all four bands. Pairwise results
must match in both symmetric cells. The external-query one-vs-N route must
match its `x`-versus-`y` cell. A public K-vs-N smoke case must also reach the
separate one/K-vs-N launcher. At least one FP32 pairwise and one FP32
external-query below-gap call must return exact public `DBL_MAX`. The Warp
pairwise and external-query routes must additionally reproduce the singleton
ledger for both metrics.

`INT_MAX` is a full-coverage request for this fixture. The implementation must
not evaluate overflowing `band+1`, `abs(i-j)`, `k+band`, `2*band+1`, or
`band*20` arithmetic.

## Registered Metal route matrix

The same oracle must be compiled into `test_metal_correctness`, with
`use_lb_keogh=false` explicitly pinned so F12 judges the DTW kernels rather
than the separately open GPU lower-bound findings. On a real Metal device it
must drive pairwise Wavefront, WavefrontGlobal, and BandedRow where supported,
plus K-vs-N wavefront/global copy routes, assert the reported kernel, and
compare exact costs and exact `DBL_MAX`. Source inspection or a portable helper
test cannot close this executable requirement.

The K-vs-N global route uses targets `{y,filler_16385}` with a deterministic
16,385-sample non-oracle filler. Its `3*max_L*sizeof(float) = 196,620`-byte
candidate footprint exceeds the 32-KB M1/M2/M3 cap documented in the existing
Metal source and therefore must select `kvn_wavefront_global` before any such
request is issued; only the `x`-versus-`y` result is an oracle operand. The
short `{y}` inventory must report `kvn_wavefront`. A future device whose
runtime-reported cap reaches this footprint will falsify that registered route
fixture and require a larger non-oracle filler, not a relaxed route assertion.

The current host probe is:

```text
OS_PROBE=Microsoft Windows 11 Enterprise 10.0.26200
XCRUN_PROBE=NOT_FOUND
METAL_COMPILER_PROBE=NOT_FOUND
```

Verdict before implementation: **[BLOCKED-ENV]** for the real-Metal execution
on this host. The source fix and permanent test still proceed; the finding
remains open until operator-owned real-device evidence exists.

This registration does not claim an `INT_MAX` LB-envelope route. With Metal LB
explicitly enabled, `env_band=opts.band` can still reach the separate envelope
kernel's signed `k+w+1` expression. F28-F30 own GPU LB admissibility,
integer-width, and loudness; their recorded risk remains open.

## Acceptance band

F12's local implementation gate passes only if all of the following hold:

1. The full-matrix oracle and explicit path enumerator reproduce every literal
   in the registered ledger before judging a GPU result.
2. A freshly rebuilt `build/cuda-verify/bin/test_cuda_correctness.exe "[F12]"`
   runs on the local RTX, prints no CUDA skip, executes at least 100 assertions,
   and passes every registered route exactly.
3. The unfiltered freshly rebuilt CUDA correctness binary passes with no
   skipped test case; its own Catch2 summary is recorded verbatim.
4. The public float-to-double result normalizer maps exact `FLT_MAX` to exact
   `DBL_MAX`, preserves ordinary finite floats exactly as a C++ cast, and
   leaves double values unchanged. A non-CUDA canonical test pins this
   contract.
5. No endpoint-scaled fixed-band formula or signed `band+1` remains in any of
   the six CUDA kernels. Shared-memory launch sizing is updated consistently
   with any removed boundary arrays.
6. Metal result matrices and K-vs-N distances use the same sentinel
   normalizer; the no-LB fixed-band DTW dispatch and its four wavefront bound
   calculations are overflow-safe by source inspection; existing stale
   widened-`FLT_MAX` expectations are updated. No broader LB claim is made.
7. The canonical build and complete CTest gate retain the inherited floor:
   114/114 tests, zero failed, with exactly the six registered capability
   skips.
8. A real Metal device run is reported separately. Until it passes with no
   skip and an executed-assertion floor of 100, F12 and D1's cross-backend
   discrepancy remain open.

The canonical geometry ledger is immutable. There are at most two repair
attempts; a failing registered band is recorded as **FALSIFIED**, never
relaxed or rescue-tuned.

## Executions and verdicts

### Fresh inherited CUDA suite

The target was rebuilt at the registered base through the Visual Studio 18
`vcvars64.bat` environment:

```text
[0/2] Re-checking globbed directories...
[1/4] Building CUDA object bin\CMakeFiles\dtwc++.dir\cuda\cuda_dtw.cu.obj
cuda_dtw.cu
tmpxft_00019d4c_00000000-7_cuda_dtw.compute_90.cudafe1.cpp
[2/4] Linking CXX static library bin\dtwc++.lib
[3/4] Linking CXX executable bin\test_cuda_correctness.exe
```

The unfiltered real binary then produced:

```text
RNG seed: 617014223
C:\D\git\dtw-cpp\tests\unit\test_cuda_correctness.cpp(514): failed: gpu_result.matrix[i * 6 + j], WithinRel(cpu_mat[i * 6 + j], 1e-10) for: 87.32076646498155981 and 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0 are within 1e-08% of each other with 1 message: 'i=0 j=1'
test cases:   55 |   54 passed | 1 failed
assertions: 7278 | 7277 passed | 1 failed


[CUDA] max_L=4096 > 2048: using the 3-buffer wavefront path (the double-buffer register cache would drop cells).
```

Verdict: **FALSIFIED [confirmed]**. On the first random pair in the inherited
unequal-length test, CUDA returned finite
`87.32076646498155981` while the repaired canonical CPU route returned exact
`DBL_MAX`. The failing binary was freshly linked from the registered base and
ran on the local RTX; this is the named pre-change baseline.

### Registered F12 CUDA red

After adding the permanent oracle and GPU route tests, but before any
production edit, the focused real-RTX gate produced:

```text
Filters: [F12]
RNG seed: 3127847899
C:\D\git\dtw-cpp\tests\unit\test_cuda_correctness.cpp(355): failed: result.distances[1] == at_gap.l1 for: 3.0 == 5.0
C:\D\git\dtw-cpp\tests\unit\test_cuda_correctness.cpp(236): failed: result.matrix[1] == expected for: 5.0
==
179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0 with 3 messages: 'route.expected_kernel := "warp"' and 'squared := false' and 'row.band := 1'
C:\D\git\dtw-cpp\tests\unit\test_cuda_correctness.cpp(324): failed: pairwise.matrix[1] == oracle::public_no_path_sentinel for: 5.0
==
179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0
C:\D\git\dtw-cpp\tests\unit\test_cuda_correctness.cpp(296): failed: pairwise.matrix[1] == expected for: 6.0
==
179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0 with 2 messages: 'squared := false' and 'row.band := 1'
C:\D\git\dtw-cpp\tests\unit\test_cuda_correctness.cpp(264): failed: result.distances[0] == expected for: 5.0
==
179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0 with 3 messages: 'route.expected_kernel := "warp"' and 'squared := false' and 'row.band := 1'
test cases:  6 |  1 passed | 5 failed
assertions: 64 | 59 passed | 5 failed
```

The independent DP/path-enumeration case was the one passing case. All five
public CUDA subjects failed with the preregistered endpoint-scaled or bypass
fingerprints: principal band 1 returned `5`, singleton band 1 returned `6`,
and forced-Wavefront K-vs-N band 2 returned `3` instead of `5`. FP32 returned
the same finite geometric value before sentinel conversion could be tested.
Verdict: **FALSIFIED [confirmed]**. Repair attempt 1 now begins; the registered
ledger and assertion floor are unchanged.

### Repair attempt 1: real CUDA

The implementation replaced all six endpoint-scaled CUDA corridors with one
ordered-subtraction `|i-j| <= band` predicate, removed the obsolete boundary
arrays from launch sizing, and translated device FP32 no-path values at the
public double result boundary. After a fresh MSVC/CUDA rebuild, the first green
focused run printed 470 assertions. Independent audit then found that the
registered singleton phrase "in both orientations" was not yet driving the
external-query `query_length > target_length` branch. The test was amended
without changing a band, cost, sentinel, route, or assertion floor, rebuilt,
and rerun.

Final focused output:

```text
Filters: [F12]
RNG seed: 2957813384
All tests passed (515 assertions in 6 test cases)
```

Final unfiltered output:

```text
RNG seed: 1106685694
All tests passed (7827 assertions in 61 test cases)


[CUDA] max_L=3072 > 2048: using the 3-buffer wavefront path (the double-buffer register cache would drop cells).
```

The CUDA correctness and LB regression targets also passed together:

```text
Test project C:/D/git/dtw-cpp/build/cuda-verify
    Start 49: test_cuda_correctness
1/2 Test #49: test_cuda_correctness ............   Passed    7.96 sec
    Start 51: test_cuda_lb_keogh
2/2 Test #51: test_cuda_lb_keogh ...............   Passed    0.26 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   8.24 sec
```

Verdict: **PASS [confirmed]** for every locally executable CUDA band. The
focused binary ran 515 assertions, exceeded the registered floor of 100,
printed no skip, reproduced the independent literal ledger in both singleton
orientations, and asserted the reported kernel for every registered route.
The unfiltered binary ran every CUDA correctness case without a skip.

### Public sentinel normalizer

The fresh canonical host executable printed:

```text
RNG seed: 3602055020
All tests passed (8 assertions in 1 test case)
```

Verdict: **PASS [confirmed]**. Exact `FLT_MAX` maps to exact `DBL_MAX`;
ordinary finite floats use the ordinary C++ cast; the adjacent
`nextafter(FLT_MAX, 0)` value is not mistaken for a sentinel; double,
infinity, and NaN behavior is pinned separately. This new permanent target
raises the canonical test inventory from the inherited 114 to 115.

### Source audit

The final current-tree token audit printed:

```text
cuda_slope                    : 0
cuda_window                   : 0
cuda_band_plus_one            : 0
cuda_band_smem                : 0
cuda_s_j_low                  : 0
cuda_my_j_low                 : 0
cuda_use_band_flag            : 0
cuda_fixed_band_contains      : 9
cuda_normalizer_calls         : 3
metal_long_bounds             : 8
metal_normalizer_calls        : 2
metal_legacy_copy_casts       : 0
metal_unsafe_wavefront_bounds : 0
metal_unsafe_host_multiplies  : 0
```

The nine CUDA predicate occurrences are one definition plus eight live cell
checks across the six pairwise and one/K-vs-N kernels. The eight Metal widened
bounds are two bounds in each of four wavefront kernels. The Metal host keeps
the requested band for route selection, uses widened multiplication, and
normalizes only full-coverage wavefront requests before device arithmetic.
Metal BandedRow retains its public `band <= 512` limit. GPU LB integer
arithmetic was neither repaired nor claimed here and remains assigned to
F28-F30.

Two independent post-fix reviews found no remaining CUDA or source-level Metal
blocker. Both specifically audited shared-memory sizes, all kernel families,
the singleton orientations, K-vs-N global routing, exact sentinel copies, and
the explicit no-LB scope.

### Metal environment result

The freshly rebuilt canonical Metal-OFF test binary printed:

```text
RNG seed: 2716726404
C:/D/git/dtw-cpp/tests/unit/test_metal_correctness.cpp(35): skipped: 'DTWC_HAS_METAL not defined; Metal tests skipped'
test cases: 1 | 1 skipped
assertions: - none -


METAL_EXIT=4
```

Verdict: **[BLOCKED-ENV] [confirmed]** for executable Metal behavior. The
source and permanent gate cover pairwise Wavefront, WavefrontGlobal, and
BandedRow plus short/global K-vs-N routes, with `use_lb_keogh=false`, but
Windows did not compile the Objective-C++/Metal-only bodies and the binary
executed zero assertions. F12 and D1 therefore remain open pending an Apple
build and real-device run above the registered assertion floor.

### Canonical regression gate

After every final test amendment, `build/highs-1151` was rebuilt and the
complete canonical gate printed:

```text
100% tests passed, 0 tests failed out of 115

Total Test time (real) =  81.49 sec

The following tests did not run:
	 49 - test_cuda_correctness (Skipped)
	 51 - test_cuda_lb_keogh (Skipped)
	 55 - test_io_readers (Skipped)
	 56 - test_metal_correctness (Skipped)
	 57 - test_metal_lb_keogh (Skipped)
	 58 - test_metal_mmap (Skipped)
```

Verdict: **PASS [confirmed]**. The actual inventory is 115/115, zero failed,
and exactly the six registered capability skips. This exceeds the
preregistered inherited 114-test floor.

### llfio-OFF regression gate

The existing `build/nollfio` tree had testing disabled, so it was reconfigured
with `-DDTWC_BUILD_TESTING=ON`, rebuilt after the final test amendment, and its
complete gate printed:

```text
100% tests passed, 0 tests failed out of 115

Total Test time (real) =  75.38 sec

The following tests did not run:
	 34 - unit_test_mmap_data_store (Skipped)
	 35 - unit_test_mmap_distance_matrix (Skipped)
	 49 - test_cuda_correctness (Skipped)
	 51 - test_cuda_lb_keogh (Skipped)
	 55 - test_io_readers (Skipped)
	 56 - test_metal_correctness (Skipped)
	 57 - test_metal_lb_keogh (Skipped)
	 58 - test_metal_mmap (Skipped)
	 77 - unit_test_benders (Skipped)
```

Verdict: **PASS [confirmed]**. The core compiles and all 115 targets pass with
llfio disabled; the nine skips are the two mmap capabilities, two CUDA
capabilities, Arrow I/O, three Metal capabilities, and optional Benders.

## Final local verdict

Repair attempt 1 is **KEEP [confirmed]** in implementation commit `4583443`.
CUDA exact geometry and public-sentinel parity are locally closed. The common
host contract, canonical configuration, and llfio-OFF configuration all pass.
F12 itself remains **OPEN [BLOCKED-ENV]** solely because the registered Metal
Objective-C++ build and real-device execution cannot run on this Windows host.

Rollback: revert `4583443` if a future Apple run rejects the source-level
repair; retain this run-log and the permanent gate so that such a rejection is
visible rather than silently restoring the inherited divergence.

The claim most likely to be wrong is the source-inspection inference that the
amended Metal MSL compiles and selects `kvn_wavefront_global` on a real Apple
device. Only the registered Apple executable run can confirm it.
