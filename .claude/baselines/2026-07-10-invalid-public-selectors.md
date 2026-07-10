# Task 8.2 F1 — non-distance selector audit

Date: 2026-07-10  
Configuration: Windows, Clang 21 Release, OpenMP enabled, HiGHS enabled  
Verdict: **PASS**

## Scope and complete enum census

A grep-driven census found 24 project-owned `enum class` declarations under
`dtwc/` (vendored nanoarrow declarations excluded):

- F1 caller-controlled selectors: `Method`, `Solver`, `Linkage`, `PAMVariant`,
  `OneBatchWeighting`, `BarycenterMethod`, `Device`, and
  `AssignmentMatrixLayout`.
- Already closed by M47/M50: `ConstraintType`, `MetricType`, `DTWVariant`,
  `MVMode`, `MissingStrategy`, `DistanceMatrixStrategy`,
  `LowerBoundStrategy`, `StoragePolicy`, `Precision`, `KernelOverride`,
  `CUDAPrecision`, and `MetalPrecision`. `CUDASettings::precision` is an
  integer selector rather than an enum and remains covered by M47.
- Derived/internal values: `SeqCause`, `Tier1ExecutionTarget`, `KernelPath`,
  and `FP64Rate`.

Every switch, comparison chain, and enum-dependent index in the project-owned
source was then traced from its public root. The F1 set is deliberately wider
than the six symptoms in the registration: barycenter method and exact-solver
assignment layout are public function/options parameters and therefore require
the same fail-closed contract.

## Preregistered red

`7b48561` added one permanent Catch2 target before production changes. For
each F1 enum it probes `-1`, the first ordinal above the declared domain,
`INT_MIN`, and `INT_MAX`. It snapshots caller-visible `Problem` data,
configuration, medoids, labels, matrix alternative, matrix size, computed
state, and every packed value bit. Separate sentinels pin OneBatchPAM stats and
exact-assignment output. `4033664` corrected the snapshot oracle to compare NaN
payloads bitwise rather than through floating-point equality.

The prerepair result was:

```text
invalid selectors: 464 assertions | 328 passed | 136 failed | 8/8 cases failed
valid controls:      59 assertions |  59 passed |   0 failed | 8/8 cases passed
```

The failures separated eight distinct permissive paths:

1. invalid `Method` could be stored and made `Problem::cluster()` return
   without selecting an algorithm;
2. invalid `Solver` could reach a no-op switch, while the Benders shortcut did
   not inspect it at all;
3. invalid `Linkage` left Lance–Williams distances at their initialized zero,
   with the one-point shortcut bypassing the switch entirely;
4. invalid `PAMVariant` skipped the swap implementation, and `k=1` bypassed
   membership checking;
5. invalid `OneBatchWeighting` executed an undocumented mixture of weighting
   formulas, while `k=N` bypassed it after observable setup work;
6. invalid `BarycenterMethod` was rejected only after unrelated series/options
   validation and work;
7. `to_string(invalid Device)` reported `cpu`; and
8. invalid `AssignmentMatrixLayout` decoded as point-major.

All declared values remained live in the prerepair control matrix, so the
repair could not shrink a legitimate domain to make the red pass.

## Repair and effect ordering

Production commit `6622834` adds small membership validators beside each enum
definition and invokes them at every operation-bearing root. Invalid values
raise `dtwc::InvalidInput` with one stable selector-specific diagnostic.
Validation now precedes setter publication, degenerate-size shortcuts,
distance-matrix fill, allocation, input scanning, backend selection, table
construction, stats writes, and clustering output publication.

All algorithm switches have explicit declared-value arms and an unreachable
`logic_error` terminal after validation. OneBatchPAM's formerly permissive
condition chains are explicit switches. `Problem::cluster()` and direct MIP
retain defensive terminals even though their public roots validate first.
Valid numerical behavior and enum ordinals are unchanged.

The four internal enums do not accept caller input at an operation boundary:

- `sequential_cause` is the only production `SeqCause` producer and returns
  exactly its three declared members before `sequential_warning_text` consumes
  it;
- `Tier1ExecutionTarget` is constructed only by the CPU/GPU ternary in
  `api.cpp` (the Python implementation independently uses validated backend
  strings);
- `KernelPath` is returned only by the exhaustive validated CUDA selector, and
  its name and both launch dispatches have explicit unknown-value terminals;
- `FP64Rate` defaults to `Slow` and is assigned only `Full` or `Slow` by the
  private GPU capability query before its sole comparison.

Thus none needs a new public validator or an artificial raw binding.

## Binding boundary

Fresh binding artifacts were built after the C++ change.

Python nanobind exposes `Method`, `Solver`, `Linkage`, `OneBatchWeighting`,
`BarycenterMethod`, and `Device`. Construction with each of the four invalid
integers raises `ValueError`; assigning raw integers to typed option/problem
fields raises `TypeError` before C++. `PAMVariant` and
`AssignmentMatrixLayout` are not exposed. A custom matrix passed 133/133
checks: every exported member, 24 invalid constructions, 36 raw-integer typed
boundaries, 28 no-mutation observations, and the two non-exposure checks. The
fresh-core contract/device suites then passed 237 cases with one expected skip.

MATLAB exposes method, solver, linkage, and device as strings. The three enum
parsers have explicit unknown-token terminals; device uses the validated
environment parser. The other four F1 selectors have no MATLAB surface.
Direct numeric selector inputs are rejected as `dtwc:invalidArgument`, never
cast to C++ enums. A fresh MEX passed 50/50 valid, invalid-string,
invalid-numeric, and no-mutation checks.

No ordinary Python or MATLAB call can transmit an arbitrary enum integer to a
C++ selector. Native C++ validation remains necessary defense for direct C++
callers and future/raw bindings.

## Green gates

```text
LLFIO ON focused F1:                559/559 assertions, 16 cases
  invalid ordinals:                 500/500 assertions,  8 cases
  every declared value:              59/59 assertions,  8 cases
LLFIO OFF focused F1:               559/559 assertions, 16 cases
M47 distance-selector regression: 1,533/1,533 assertions in each mode

hierarchical:                         59/59 assertions, 10 cases
FasterPAM:                           240/240 assertions, 7 cases
OneBatchPAM:                     10,237/10,237 assertions, 8 cases
barycenter:                          113/113 assertions, 14 cases
MIP/Benders:                         194/194 assertions, 17 cases
environment device:                  22/22 assertions, 8 cases
Problem core:                         18/18 assertions, 3 cases
Tier-1 C++ API:                       50/50 assertions, 9 cases

fresh Python boundary matrix:       133/133 checks
fresh Python contract/device:       237 passed, 1 capability skip
fresh MATLAB selector matrix:        50/50 checks

canonical serial CTest:             112/112 tests, 6 expected skips
canonical parallel CTest after F2:  112/112 tests, 6 expected skips
```

The first four-way full CTest run exposed an orthogonal test-infrastructure
finding: the wall-clock scalar/MV parity assertion ran alongside three heavy
targets and missed its threshold by 0.685 ms. All 12,645 correctness
assertions in that target passed. Five isolated repetitions passed
63,230/63,230 assertions with MV/scalar ratios from 0.69 to 1.00, falsifying a
kernel regression. Commit `b61001e` marks that timing target `RUN_SERIAL`; the
same four-way command then passed and scheduled the measurement alone. This is
registered as F2 rather than hidden from the full-gate evidence.

## Commits

- `7b48561` — preregister invalid clustering/environment selectors
- `4033664` — make selector state snapshots bit-exact
- `6622834` — reject invalid clustering/environment selectors before effects
- `b61001e` — isolate the wall-clock multivariate parity gate under CTest

F1 verdict: **PASS.** Every caller-controlled non-distance enum now fails
closed before observable work, every declared value remains operational, the
ordinary bindings cannot manufacture invalid values, and both storage modes
plus the complete canonical suite are green.
