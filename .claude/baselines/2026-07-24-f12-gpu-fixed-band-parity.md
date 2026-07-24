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
external-query below-gap call must return exact public `DBL_MAX`.

`INT_MAX` is a full-coverage request for this fixture. The implementation must
not evaluate overflowing `band+1`, `abs(i-j)`, `k+band`, `2*band+1`, or
`band*20` arithmetic.

## Registered Metal route matrix

The same oracle must be compiled into `test_metal_correctness`. On a real
Metal device it must drive pairwise Wavefront, WavefrontGlobal, and BandedRow
where supported, plus K-vs-N wavefront/global copy routes, assert the reported
kernel, and compare exact costs and exact `DBL_MAX`. Source inspection or a
portable helper test cannot close this executable requirement.

The current host probe is:

```text
OS_PROBE=Microsoft Windows 11 Enterprise 10.0.26200
XCRUN_PROBE=NOT_FOUND
METAL_COMPILER_PROBE=NOT_FOUND
```

Verdict before implementation: **[BLOCKED-ENV]** for the real-Metal execution
on this host. The source fix and permanent test still proceed; the finding
remains open until operator-owned real-device evidence exists.

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
   normalizer; Metal host/kernel band arithmetic is overflow-safe by source
   inspection; existing stale widened-`FLT_MAX` expectations are updated.
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

Pending.
