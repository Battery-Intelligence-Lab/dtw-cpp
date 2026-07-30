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
- Enhanced greater: `A=(-1,0)`, `B=(0,-1)`, `w=1` gives symmetric
  Enhanced 2 and Keogh 0.
- Keogh greater: `A=(-1,-1,-1,-1)`,
  `B=(-1,-1,0,-1)`, `w=1` gives symmetric Keogh 1 and Enhanced 0 at
  effective `V=2`.
- Strict tail cap: `A=(0,0,1,1)`, `B=(1,1,2,0)`, `w=1` gives production
  L1/squared `3/3` and exact-predicate NoLR `4/4`. Every nonzero difference is
  one, so the witness distinguishes the tail predicate under both metrics
  without conflicting with F57's required `w>=n-1` saturation.
- Nondegenerate upper tail:
  `A=(0,0,20,5,5,5,5)`, `B=(0,0,0,0,0,10,0)`, `w=2`; negating both
  exercises the lower tail.
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
radius maps that to `n-1`, where the candidate is no longer strict. The
four-point `w=1` witness above preserves the unchanged `tail_strict=2/2`
counter and tests the intended tail-cap fact under both metrics.

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
