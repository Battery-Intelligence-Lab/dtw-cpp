# R2-D3 LB_Enhanced and LB_Webb_NoLR — 2026-07-30

## Scope and base

- Branch: `Claude`.
- Clean base: `28904d7` (`docs: close F23 Python checkpoint bindings`).
- Subject: equal-length scalar LB_Enhanced and the production
  LB_Webb_NoLR-plus-tail-cap implementation under the same fixed
  Sakoe–Chiba radius as DTW, for L1 and unrooted squared-L2 point costs.
- Closure prerequisites: F54 live Enhanced/Keogh cascade reachability, F55
  Webb variant/provenance correction, and F57 CPU extreme-radius arithmetic.
- Decisive production build: `build/highs-1151` (clang, Ninja, Release,
  HiGHS ON, llfio ON, Arrow OFF).

This registration was written before adding or executing either new oracle.
The inherited clean-base run, red-first runs, and product attempts are appended
verbatim after this commit. Hand calculations and source inspection are
preflight evidence, not evidence that the new gates pass.

## Primary-source boundary

Tan, Petitjean, and Webb, *SDM 2019*, DOI
`10.1137/1.9781611975673.59`, Theorem 3.1, Eq. 3.7, and Theorem 3.2 provide
the elastic-cut construction and admissibility proof for LB_Enhanced. The
paper explicitly describes `LB_Enhanced^1` as uniformly tighter than
LB_Keogh. The repository's exact counterexamples, rather than that paper,
establish no universal ordering for effective `V >= 2`.

Webb and Petitjean, *Pattern Recognition* 115 (2021) 107895, DOI
`10.1016/j.patcog.2021.107895`, Theorem 2 and Eqs. 26–43 provide the
four-point correction argument. Full Algorithm 2 contains `MinLRPaths`.
The production function instead matches the later all-index
`LB_Webb_NoLR` formula and conservatively caps a trailing free flag. The
paper's Wafer table reports tightness 0.96904 for NoLR and 0.96891 for full
Webb, so F55 rejects the existing claim that omission only loosens.

## Preregistered proof obligations

For LB_Enhanced, every selected left elastic cut, every middle vertical
envelope set, and every reflected right cut is crossed by every admissible
path. The selected sets are mutually disjoint for
`v=min(V,floor(n/2))`. The sum of their minimum nonnegative point costs is
therefore no greater than any path cost. L1 has amplitude unit `U`;
unrooted squared L2 has unit `U^2`. There is no modelling approximation in
the exact-integer oracle.

For effective `V=1`, only the two endpoint Keogh contributions are replaced
by forced-corner costs, which are at least their interval projections.
Therefore directional Enhanced-1 is at least directional Keogh. For
effective `V>=2`, no ordering is claimed; both strict directions must execute.
The production cascade must take their maximum.

For Webb's ordered four-point condition, write
`p=x-a`, `q=y-x`, `r=b-y` for `a<=x<=y<=b`. L1 satisfies

```text
p+q+r = (p+q) + (q+r) - q.
```

Squared cost leaves slack

```text
(p+q+r)^2 - ((p+q)^2 + (q+r)^2 - q^2) = 2pr >= 0.
```

The first production pass is directional Keogh. Every full or
overlap-subtracted second-pass correction is nonnegative under that
condition, so production directional Webb is at least matching-direction
Keogh; taking the maximum of both directions dominates symmetric Keogh. A
single directional Webb value is not claimed to dominate symmetric Keogh.

Let a stored trailing flag at index `t` certify every actual safe predicate
in `[max(0,t-2w),t]`. Away from the tail, `t=j+w` is exactly the required
centered window. At the tail, `t=n-1` checks a superset of the exact clipped
window, so a production true flag implies an exact true flag. Replacing an
exact true flag by false selects either zero or a no-larger
overlap-subtracted correction. Thus

```text
production tail-cap <= exact-predicate NoLR <= DTW.
```

This tail argument is independent of `MinLRPaths`; no full-Webb/NoLR
ordering is asserted.

Assumptions enter here: both finite scalar series are equal and nonempty in
length, paths use the standard monotone/continuous steps and forced corners,
the same nonnegative effective radius and point cost define envelopes,
bounds, and DTW, costs are nonnegative and additive, and exact arithmetic is
representable. NaN/Inf, mutable envelope provenance/shape, lengths not
representable by current low-level indexing, and last-ULP pruning decisions
remain F46/D17 scope.

## Preregistered independent D3 oracle

Envelope alphabet is `{-2,0,3}`, lengths 1 through 5, and radii `0..n`:

```text
sum((n+1)*3^n, n=1..5) = 2,004 envelope cases.
```

Production `U`, `L`, `L(U)`, and `U(L)` are compared elementwise with direct
clipped-window scans.

Bound alphabet is `{-1,0,2}`, all ordered pairs, lengths 1 through 4, and
radii `0..n`:

```text
sum((n+1)*3^(2n), n=1..4) = 35,982 path/Webb/tail cases.
sum(3^(2n), n=1..4)       =  7,380 full-cover cases at w=n.
```

The DTW oracle recursively enumerates monotone paths and never calls
production DTW or duplicates its dynamic-programming recurrence. It must
recover full-cover Delannoy path counts `1,3,13,63`, diagonal-only count 1,
and a non-diagonal optimum.

Enhanced evaluates each `V=1..max(1,floor(n/2))`, giving:

```text
18 + 243 + 2,916 + 65,610 = 68,787 Enhanced configurations.
```

Production Enhanced is compared exactly with independently constructed
left/right elastic-cut and middle projection sets, then bounded by enumerated
DTW for both metrics and both orientations. Four deterministic length-10/11
checks exercise requested default `V=5` without exponentially enumerating
those lengths; their DTW comparator is an independent full-matrix recurrence.

Production Webb is compared exactly with an independent direct-predicate
NoLR reference for both metrics and both orientations. All four additive
branches (full upper/lower and overlap upper/lower) must execute. Every case
checks directional Webb at least directional Keogh and Webb no greater than
enumerated DTW. Every tail case checks production no greater than the exact
predicate reference and both no greater than DTW.

Five asymmetric scalar values generate all 70 nondecreasing four-element
multisets; L1 and squared costs give 140 metric cases, each also checked in
reverse orientation.

The required exact marker is:

```text
D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 webb_cases=35982 webb_branches=4/4 webb_strict=2/2 tail_cases=35982 tail_strict=2/2 metric_cases=140 order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS
```

It prints only after all violation ledgers are zero. CTest clears
`SKIP_RETURN_CODE`, rejects skip text, requires that exact marker followed by
at least 40 assertions in exactly one test case, sets `OMP_NUM_THREADS=1`,
is `RUN_SERIAL`, and has a finite timeout.

## Preregistered discriminators and live routes

- Webb strictness: `A=(-1,-1)`, `B=(-1,2)`, `w=1` gives Webb L1/squared
  `3/9` and matching-direction Keogh `0/0`.
- Enhanced greater at effective `V=2`: `A=(10,0,0,0)`,
  `B=(0,10,0,0)`, `w=1` gives symmetric Enhanced 10 and Keogh 0.
- Keogh greater: `A=(-1,-1,-1,-1)`,
  `B=(-1,-1,0,-1)`, `w=1` gives symmetric Keogh 1 and Enhanced 0 at
  effective `V=2`.
- Strict upper/lower tail pair:
  `A=(0,0,20,5,5,5,5)`, `B=(0,0,0,0,0,10,0)`, `w=2` gives production
  L1/squared `20/400` and exact-predicate NoLR `30/450`; negating both gives
  the same values through the lower branch. `tail_strict=2/2` counts these two
  branch orientations, and each orientation must pass both metrics.
- Supplementary binary tail:
  `A=(0,0,1,1)`, `B=(1,1,2,0)`, `w=1` gives production L1/squared `3/3`
  and exact-predicate NoLR `4/4`. Every nonzero difference is one, so it
  independently distinguishes the predicate without metric scaling.
- Default `V=5`: the fixed length-10 series in the committed oracle must
  distinguish `w=0` from `w=1`; an additional odd-length case prevents
  accidental even-length-only coverage.

F54 uses

```text
A=(0,10,0,0,0,0)
B=(0,0,0,10,0,0)
C=(0,0,10,0,0,0)
ordering={C,A,B}, w=1.
```

Exact distances are `C-A=0`, `C-B=0`, and `A-B=20`; Kim and Enhanced for the
last pair are zero and symmetric Keogh is 10. Both the direct
`fill_distance_matrix_pruned` route and configured public
`Problem::fill_distance_matrix` route must preserve all matrix entries and
prove exactly:

```text
total=3 kim=0 envelope=1 early=1 full=2.
```

The public route must emit the exact live pruning summary. OpenMP dynamic
scheduling is disabled and one thread is selected in-process as well as by
CTest. Exact-matrix recomputation means this is reachability/correctness
evidence, not a speedup claim.

## Preregistered F57 gate

F57 has a separate non-skippable executable and marker:

```text
F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 admissible=2/2 skips=0 verdict=PASS
```

For `A=(-2,-2)`, `B=(0,0)`, a global radius gives exact L1/squared results
4/8. The gate compares `n-1`, `n`, and `INT_MAX`, checks Enhanced saturation
parity as a supporting radius invariant, and requires both Webb results to
remain admissible. CTest clears the skip code, rejects skip text, requires the
exact marker followed by at least 12 assertions in one case, and runs serially.
The same target must pass in `build/ubsan-wsl` with
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`; environment impossibility
is recorded verbatim as `[BLOCKED-ENV]`, never converted to a pass.

Normalize an effective radius to `min(max(band,0),n-1)` before arithmetic and
use unsigned or explicitly saturated counters. F46 retains envelope
provenance/shape/aliasing and unrepresentable-length API work; F50 retains GPU
overflow.

## Execution protocol and immutable band

1. Commit this registration.
2. Rebuild and run the inherited focused lower-bound/pruned-matrix tests on
   clean base; append output verbatim.
3. Add the two permanent tests and their CTest policies without product
   changes; commit and execute the expected-red gates.
4. Permit at most two product attempts across F54/F55/F57. A failed second
   product attempt is recorded `FALSIFIED`; it is not rescue-tuned.
5. After focused green, run the inherited focused subjects, both new targets,
   then serial canonical, llfio-OFF, and Arrow-ON matrices. Because two
   executables are added, the prospective floors are 125/125, 125/125, and
   127/127; they become factual only after execution. Require the existing
   exact 6/9/8 skip sets, Arrow reader 390 assertions/11 cases, and all four
   real-CLI markers.
6. Run documentation-contract, record-hygiene, repository-hygiene, and diff
   checks on a clean tree.

Every numeric verdict must name its artifact and output. Wall time is
advisory. The claim most expected to be wrong is exact correspondence of the
production recurrence with the direct-predicate NoLR reference at clipped
tails; the all-case ledger, strict-tail witnesses, and branch counters are
the arbiters.

Preregistration correction before either new test was executed: the initial
two-point strict-tail candidate used `w=n`. F57's already registered effective
radius maps that to `n-1`, where the candidate is no longer strict. The stable
`tail_strict=2/2` counter now means the nondegenerate upper/lower pair above,
with both metrics checked for each orientation; the four-point `w=1` case is
retained as a supplementary metric-independent discriminator.

The same pre-execution audit replaced the original Enhanced-greater witness:
its effective `V=1` belongs to the separately proved uniformly tighter case
and could not establish no ordering for larger `V`. The length-four
effective-`V=2` witness above now pairs with the existing Keogh-greater
effective-`V=2` witness; `order_witnesses=2/2` is unchanged.

## Inherited clean-base baseline

After registration commit `8f8e7e5`, the working tree was clean. The canonical
rebuild printed:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

The registered inherited command was:

```text
ctest --test-dir build/highs-1151 -C Release -R '^(test_lb_enhanced_webb|unit_test_lower_bounds|unit_test_pruned_distance_matrix)$' --output-on-failure --no-tests=error -j 1
```

Its complete terminal output was:

```text
Test project C:/D/git/dtw-cpp/build/highs-1151
    Start  8: test_lb_enhanced_webb
1/3 Test  #8: test_lb_enhanced_webb ..............   Passed    0.13 sec
    Start 36: unit_test_lower_bounds
2/3 Test #36: unit_test_lower_bounds .............   Passed    0.19 sec
    Start 42: unit_test_pruned_distance_matrix
3/3 Test #42: unit_test_pruned_distance_matrix ...   Passed    4.06 sec

100% tests passed, 0 tests failed out of 3

Total Test time (real) =   4.43 sec
```

Baseline verdict: **PASS [confirmed]** for the three inherited executables.
This does not contradict F54/F55/F57: the inherited tests contain neither the
live take-max discriminator nor an independent NoLR/tail oracle nor the
`INT_MAX` arithmetic case.

## Permanent-gate expected red

Commit `244adf7` added both new executable gates and their fail-closed CTest
policies without changing product code. The focused build regenerated the
CMake graph, compiled and linked both targets, and exited zero. Its only
compiler diagnostics were the inherited unsupported
`-fno-signaling-nans` warning and the inherited llfio header-only pragma
warning.

The serial verbose command was:

```text
ctest --test-dir build/highs-1151 -C Release -R '^(test_lb_enhanced_webb_derivation|test_lb_webb_intmax)$' --output-on-failure --no-tests=error -V -j 1
```

D3 executed the complete finite oracle before reaching F54. Its exact failure
block was:

```text
C:/D/git/dtw-cpp/tests/unit/adversarial/test_lb_enhanced_webb_derivation.cpp(1061): FAILED:
  REQUIRE( stats.pruned_by_lb_keogh == 1 )
with expansion:
  0 == 1
with messages:
  first envelope violation:
  first Enhanced structure violation:
  first path violation:
  first Enhanced violation:
  first Webb violation:
  first tail violation:
  first predicate violation:

===============================================================================
test cases:  1 |  0 passed | 1 failed
assertions: 89 | 88 passed | 1 failed
```

The empty violation messages plus 88 preceding passing assertions confirm
that the independent envelopes, cut structure, explicit paths, Enhanced
formula/admissibility, Webb direct predicates/dominance/admissibility, tail
implication, metric condition, default `V=5`, strict order/tail witnesses, and
primitive F54 values all passed. The sole red is the live cascade's missing
Keogh maximum.

F57's exact failure block was:

```text
C:/D/git/dtw-cpp/tests/unit/adversarial/test_lb_webb_intmax.cpp(67): FAILED:
  REQUIRE( at_intmax.webb_l1 == exact_global_l1 )
with expansion:
  8.0 == 4.0

===============================================================================
test cases: 1 | 1 failed
assertions: 3 | 2 passed | 1 failed
```

The aggregate verdict was:

```text
0% tests passed, 2 tests failed out of 2

Total Test time (real) =   0.54 sec

The following tests FAILED:
	  9 - test_lb_enhanced_webb_derivation (Failed)
	 11 - test_lb_webb_intmax (Failed)
```

Expected-red verdict: **CONFIRMED**. F54 is observed as envelope-prune count
zero after every mathematical prerequisite passes. F57 is observed as the
inadmissible doubled L1 value 8 against exact global value 4. Product attempts
remain `0 / 2`.

CTest also emitted this post-run configure-glob diagnostic:

```text
-- GLOB mismatch!
The following files were added:
  +unit/adversarial/test_lb_enhanced_webb_derivation.cpp
  +unit/adversarial/test_lb_webb_intmax.cpp
Errors while running CTest
```

Both named targets had already compiled, linked, and executed in this run, so
the subject red is not inferred from that diagnostic. A settled rebuild is
required before a green verdict.

## Product attempt 1 — F54

Commit `d09cf9c` changed only F54's product route and directly coupled public
contracts: `LowerBoundStrategy::Enhanced` now activates both Keogh and
Enhanced and retains their maximum after Kim. The focused target rebuilt
successfully.

The unchanged D3 gate passed all mathematical and direct/public F54 state
checks, then exposed an incomplete whole-output test oracle:

```text
C:/D/git/dtw-cpp/tests/unit/adversarial/test_lb_enhanced_webb_derivation.cpp(1102): FAILED:
  REQUIRE( public_output.str() == "Distance matrix is being filled!\n" "Pruned strategy: 3 pairs, 1 early-abandoned, pruning ratio: " "0.333333\n" )
with expansion:
  "Distance matrix is being filled!
  Pruned strategy: 3 pairs, 1 early-abandoned, pruning ratio: 0.333333
  Distance matrix has been filled!
  "
  ==
  "Distance matrix is being filled!
  Pruned strategy: 3 pairs, 1 early-abandoned, pruning ratio: 0.333333
  "
with messages:
  first envelope violation:
  first Enhanced structure violation:
  first path violation:
  first Enhanced violation:
  first Webb violation:
  first tail violation:
  first predicate violation:

===============================================================================
test cases:   1 |   0 passed | 1 failed
assertions: 114 | 113 passed | 1 failed
```

CTest's exact summary was:

```text
0% tests passed, 1 tests failed out of 1

Total Test time (real) =   0.48 sec

The following tests FAILED:
	  9 - test_lb_enhanced_webb_derivation (Failed)
```

Attempt-1 verdict: **FALSIFIED by test-harness expectation, product behavior
confirmed through the preceding 113 assertions**. In particular, the repaired
direct ledger, both complete exact matrices, public route, and exact pruning
summary all passed before the final string comparison. The captured output
proves the public method also emits its existing completion line. Add exactly
`Distance matrix has been filled!\n` to the expected string; no counter,
marker, subject behavior, or pass/fail band changes. Product attempts are now
`1 / 2`.

## Product attempt 2 — final focused adjudication

Commit `53506a9` corrected only the incomplete public-output oracle. Before
the final execution, a fourth read-only adversarial audit found that the
original constant-series F57 fixture would distinguish signed overflow but
would not distinguish radius saturation from merely widening the arithmetic.
Commit `13cd4f6` therefore added the nonconstant analytic discriminator
`A={0,0}`, `B={-1,1}`. Its exact global L1 and squared bound is 2; radii
`n-1`, `n`, and `INT_MAX` must all return 2. Without geometric saturation,
the latter two return zero even with wide counters. The case exercises both
the full upper and full lower correction branches. It does not change the
registered marker or band.

The same audit confirmed before execution that commit `29f9103`:

- resolves the shared effective radius as
  `min(max(band,0),n-1)`;
- preserves every representable in-band result algebraically;
- uses unsigned saturated doubled-radius and free-run arithmetic; and
- computes the shifted flag index as
  `j + min(w,n-1-j)`, which cannot exceed `n-1`.

The separate F46 ownership of unrepresentable series lengths, envelope
shape/provenance, and `2*n` scratch sizing is unchanged.

The focused build command was:

```text
cmake --build build/highs-1151 --target test_lb_enhanced_webb_derivation test_lb_webb_intmax
```

It exited zero after compiling the changed product and both targets. Its
diagnostics were only the inherited unsupported `-fno-signaling-nans` warning
and inherited llfio header-only pragma warning.

The final product-attempt command was unchanged:

```text
ctest --test-dir build/highs-1151 -C Release -R '^(test_lb_enhanced_webb_derivation|test_lb_webb_intmax)$' --output-on-failure --no-tests=error -V -j 1
```

Its complete terminal output was:

```text
UpdateCTestConfiguration  from :C:/D/git/dtw-cpp/build/highs-1151/DartConfiguration.tcl
Parse Config file:C:/D/git/dtw-cpp/build/highs-1151/DartConfiguration.tcl
Test project C:/D/git/dtw-cpp/build/highs-1151
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 9
    Start  9: test_lb_enhanced_webb_derivation

9: Test command: C:\D\git\dtw-cpp\build\highs-1151\bin\test_lb_enhanced_webb_derivation.exe
9: Working Directory: C:/D/git/dtw-cpp
9: Environment variables:
9:  OMP_NUM_THREADS=1
9: Test timeout computed to be: 60
9: Randomness seeded to: 3457699623
9: [DTWC++ WARNING] OpenMP is available but only 1 thread is usable — DTWC++ is running SINGLE-THREADED.
9:   Distance-matrix computation will be extremely slow for large datasets.
9:   Raise the thread count (unset OMP_NUM_THREADS, or set OMP_NUM_THREADS>1) to use all CPU cores.
9: D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 webb_cases=35982 webb_branches=4/4 webb_strict=2/2 tail_cases=35982 tail_strict=2/2 metric_cases=140 order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS
9: ===============================================================================
9: All tests passed (115 assertions in 1 test case)
9:
1/2 Test  #9: test_lb_enhanced_webb_derivation ...   Passed    0.50 sec
test 11
    Start 11: test_lb_webb_intmax

11: Test command: C:\D\git\dtw-cpp\build\highs-1151\bin\test_lb_webb_intmax.exe
11: Working Directory: C:/D/git/dtw-cpp
11: Test timeout computed to be: 30
11: Randomness seeded to: 3905802112
11: F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 admissible=2/2 skips=0 verdict=PASS
11: ===============================================================================
11: All tests passed (24 assertions in 1 test case)
11:
2/2 Test #11: test_lb_webb_intmax ................   Passed    0.07 sec

The following tests passed:
	test_lb_enhanced_webb_derivation
	test_lb_webb_intmax

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.68 sec
```

Attempt-2 verdict: **PASS [confirmed]** against both preregistered exact
markers. Both subjects ran with zero skips. D3 passed 115 assertions and F57
passed 24 assertions. The two-attempt product budget is exhausted with a green
final result. Focused F54/F57 behavior is confirmed; D3 closure still requires
the registered provenance/derivation corrections, WSL UBSan execution, and
the serial integration matrices.

## Derivation and fail-closed documentation contract

Commit `7dce222` added the complete D3 derivation, primary-source scope,
assumptions and units, independent witnesses, and code-conformance map. A
post-draft adversarial audit replaced the informal Webb collision argument
with the stronger cellwise inequality
`row_bridge(i) + column_correction(j) <= point_cost(i,j)` for every admissible
path cell. Its four branch cases cover the exact all-index NoLR predicates,
the paper's explicit negated flags, and production's conservative equality
extension.

Commit `639e1c4` added the fail-closed D3 documentation contract. The direct
checker result was:

```text
generated documentation is current
documentation contract checks passed
```

An independent read-only review first found that inventory counts alone did
not pin the eight zero-violation verdicts, that only 8/24 F57 assertions were
pinned, that comments/literals could satisfy code markers, and that CTest's
marker and assertion floor were not required in one expression. After those
defects were repaired, these six in-memory mutations were all rejected:

```text
D3_CHECKER_MUTATION d3_zero_verdict=REJECTED
D3_CHECKER_MUTATION f57_admissibility=REJECTED
D3_CHECKER_MUTATION production_commented=REJECTED
D3_CHECKER_MUTATION cascade_commented=REJECTED
D3_CHECKER_MUTATION ctest_split_pass=REJECTED
D3_CHECKER_MUTATION ctest_commented_composite=REJECTED
```

The final independent narrow re-audit reported `CLEAN`.

## WSL UBSan F57 gate

Before execution, the existing build was inspected without rebuilding. The
exact environment probe was:

```text
Ubuntu clang version 18.1.3 (1ubuntu1)
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/bin
cmake version 3.28.3
1.11.1
CMAKE_BUILD_TYPE:STRING=RelWithDebInfo
DTWC_ENABLE_ARROW:BOOL=OFF
DTWC_ENABLE_CUDA:BOOL=OFF
DTWC_ENABLE_HIGHS:BOOL=OFF
DTWC_ENABLE_LLFIO:BOOL=OFF
DTWC_ENABLE_METAL:BOOL=OFF
DTWC_ENABLE_MPI:BOOL=OFF
dtwc_ENABLE_SANITIZER_UNDEFINED:BOOL=ON
```

The registered target was then rebuilt with:

```text
wsl.exe --cd /mnt/c/D/git/dtw-cpp env UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 cmake --build build/ubsan-wsl --target test_lb_webb_intmax -j 2
```

The reconfigure reported `Compiler: Clang 18.1.3`, `Build type:
RelWithDebInfo`, `Testing: ON`, every optional backend OFF, and the build
finished by linking `bin/test_lb_webb_intmax`. Diagnostics were the inherited
unsupported `-fno-signaling-nans` warning, two inherited requested-loop
vectorization warnings, dirty ignored CPM-cache warnings, and the expected
no-MIP-solver warning. No UBSan diagnostic appeared.

The decisive command was:

```text
wsl.exe --cd /mnt/c/D/git/dtw-cpp env UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --test-dir build/ubsan-wsl -R ^test_lb_webb_intmax$ --output-on-failure --no-tests=error -V -j 1
```

Its complete output was:

```text
Internal ctest changing into directory: /mnt/c/D/git/dtw-cpp/build/ubsan-wsl
UpdateCTestConfiguration  from :/mnt/c/D/git/dtw-cpp/build/ubsan-wsl/DartConfiguration.tcl
Parse Config file:/mnt/c/D/git/dtw-cpp/build/ubsan-wsl/DartConfiguration.tcl
UpdateCTestConfiguration  from :/mnt/c/D/git/dtw-cpp/build/ubsan-wsl/DartConfiguration.tcl
Parse Config file:/mnt/c/D/git/dtw-cpp/build/ubsan-wsl/DartConfiguration.tcl
Test project /mnt/c/D/git/dtw-cpp/build/ubsan-wsl
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 11
    Start 11: test_lb_webb_intmax

11: Test command: /mnt/c/D/git/dtw-cpp/build/ubsan-wsl/bin/test_lb_webb_intmax
11: Working Directory: /mnt/c/D/git/dtw-cpp
11: Test timeout computed to be: 30
11: Randomness seeded to: 466888232
11: F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 admissible=2/2 skips=0 verdict=PASS
11: ===============================================================================
11: All tests passed (24 assertions in 1 test case)
11:
1/1 Test #11: test_lb_webb_intmax ..............   Passed    0.03 sec

The following tests passed:
	test_lb_webb_intmax

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.06 sec
```

WSL UBSan verdict: **PASS [confirmed]**. The target executed 24 assertions,
printed the exact registered marker, incurred zero skips, and produced no
undefined-behavior diagnostic under `halt_on_error=1`.

## 2026-08-09 closure continuation — focused and canonical gates

The closure resumed from committed adjudicator base `a62e285`. Before any
runtime execution, `scripts/adjudicate_d3_closure.py self-test` rejected all
25 transcript mutations, and direct inspection confirmed the exact CTest
marker/floor policies, both product routes, saturated CPU arithmetic, and the
registered six-name canonical skip set.

The first canonical rebuild was not a no-work build: it rebuilt 227 of 228
reported Ninja edges because the retained build tree's dependency timestamps
were stale. It completed successfully; the diagnostic classes were the
already-recorded unsupported `-fno-signaling-nans` warning and LLFIO's
header-only error-category pragma. A second rebuild printed verbatim:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

The focused closure command was:

```text
uv run python scripts/adjudicate_d3_closure.py focused
```

Its load-bearing output was:

```text
8: All tests passed (39273 assertions in 14 test cases)
1/5 Test  #8: test_lb_enhanced_webb ..............   Passed    0.14 sec
9: D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 webb_cases=35982 webb_branches=4/4 webb_strict=2/2 tail_cases=35982 tail_strict=2/2 metric_cases=140 order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS
9: All tests passed (115 assertions in 1 test case)
2/5 Test  #9: test_lb_enhanced_webb_derivation ...   Passed    0.62 sec
11: F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 admissible=2/2 skips=0 verdict=PASS
11: All tests passed (24 assertions in 1 test case)
3/5 Test #11: test_lb_webb_intmax ................   Passed    0.11 sec
38: All tests passed (63 assertions in 11 test cases)
4/5 Test #38: unit_test_lower_bounds .............   Passed    0.11 sec
44: All tests passed (5584 assertions in 26 test cases)
5/5 Test #44: unit_test_pruned_distance_matrix ...   Passed    3.62 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =   4.65 sec
D3_CLOSURE_FOCUSED rc=0 subjects=5/5 markers=2/2 summary_exact=True skip_free=True verdict=PASS
```

Focused verdict: **PASS [confirmed]**. All five named executables ran; neither
D3 nor F57 could skip, and both exact registered markers and assertion floors
were observed.

After the settled build, the documentation-contract and record-hygiene gates
both passed. The decisive canonical command was:

```text
uv run python scripts/adjudicate_d3_closure.py canonical
```

Its load-bearing terminal output was:

```text
  8/125 Test   #8: test_lb_enhanced_webb .....................   Passed    0.09 sec
  9/125 Test   #9: test_lb_enhanced_webb_derivation ..........   Passed    0.39 sec
 11/125 Test  #11: test_lb_webb_intmax .......................   Passed    0.04 sec

100% tests passed, 0 tests failed out of 125

Label Time Summary:
f14            =   0.96 sec*proc (1 test)
f17            =   2.83 sec*proc (1 test)
integration    =   3.78 sec*proc (2 tests)

Total Test time (real) = 111.78 sec

The following tests did not run:
	 54 - test_cuda_correctness (Skipped)
	 56 - test_cuda_lb_keogh (Skipped)
	 60 - test_io_readers (Skipped)
	 61 - test_metal_correctness (Skipped)
	 62 - test_metal_lb_keogh (Skipped)
	 63 - test_metal_mmap (Skipped)
D3_CLOSURE_CANONICAL rc=0 inventory=125/125 subjects=1/1 skips=6/6 skip_set_match=True summary_exact=True verdict=PASS
```

Canonical verdict: **PASS [confirmed]** against the prospective 125/125
floor. D3 and F57 both executed as ordinary passes, zero tests failed, and the
six capability skips match the preregistered set exactly. This promotes only
the canonical floor; D3/F54/F55/F57 remain open until the llfio-OFF and
Arrow-ON gates, Arrow runtime subjects, documentation, and hygiene close.

### llfio-OFF matrix

`cmake --build build/nollfio` regenerated the retained llfio-OFF tree with
Clang 21.1.8, Release, OpenMP ON, and llfio/CUDA/Metal/MPI/HiGHS OFF. The
regeneration's configure-dependent glob diagnostic named exactly the two
expected additions:

```text
-- GLOB mismatch!
The following files were added:
  +unit/adversarial/test_lb_enhanced_webb_derivation.cpp
  +unit/adversarial/test_lb_webb_intmax.cpp
```

The build exited zero. A mandatory settling rebuild then printed:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

Generated CTest metadata contained exactly 125 tests and one non-skippable,
serial, marker-pinned entry for each D3 target. The adjudicator self-test again
rejected 25/25 mutations before runtime. The decisive command was:

```text
uv run python scripts/adjudicate_d3_closure.py nollfio
```

Its load-bearing terminal output was:

```text
  9/125 Test   #9: test_lb_enhanced_webb_derivation ..........   Passed    0.43 sec
 11/125 Test  #11: test_lb_webb_intmax .......................   Passed    0.07 sec

100% tests passed, 0 tests failed out of 125

Label Time Summary:
f14            =   0.55 sec*proc (1 test)
f17            =   2.43 sec*proc (1 test)
integration    =   2.98 sec*proc (2 tests)

Total Test time (real) = 102.05 sec

The following tests did not run:
	 39 - unit_test_mmap_data_store (Skipped)
	 40 - unit_test_mmap_distance_matrix (Skipped)
	 54 - test_cuda_correctness (Skipped)
	 56 - test_cuda_lb_keogh (Skipped)
	 60 - test_io_readers (Skipped)
	 61 - test_metal_correctness (Skipped)
	 62 - test_metal_lb_keogh (Skipped)
	 63 - test_metal_mmap (Skipped)
	 82 - unit_test_benders (Skipped)
D3_CLOSURE_NOLLFIO rc=0 inventory=125/125 subjects=1/1 skips=9/9 skip_set_match=True summary_exact=True verdict=PASS
```

llfio-OFF verdict: **PASS [confirmed]** against the prospective 125/125
floor. Zero tests failed, D3 and F57 executed as ordinary passes, and the nine
capability skips match the preregistered set exactly. D3/F54/F55/F57 remain
open pending Arrow-ON full-matrix and runtime-subject gates plus closure
bookkeeping and hygiene.
