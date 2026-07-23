# FastPAM point-index width adjudication

Date: 2026-07-23

Base commit: `15fe168` (`docs: record F10 closure`)

Build: `build/highs-1151` (canonical clang + Ninja + Release gate)

## Finding

**[confirmed]** The interrupted edit in `dtwc/algorithms/fast_pam.cpp`
guards `fast_pam_swap` above `INT_MAX`, but leaves `fast_pam_seeded` with an
unchecked `static_cast<int>(prob.size())`. It also widens internal loop
counters only to cast every point index back to the library's `int`-indexed
distance/result boundary. The intended repair is one checked narrowing at each
public entry, before matrix materialisation or result mutation, followed by
plain `int` internals.

## Bands registered before the runs

- **BAND-RED [HARD]:** an extracted production dimension checker retaining the
  old unchecked narrowing must fail a focused `INT_MAX + 1` test. A compile-only
  failure is not sufficient; the test binary must run and report the missing
  exception.
- **BAND-BOUNDARY [HARD]:** the repaired checker accepts `1` and exactly
  `INT_MAX`, rejects zero, and rejects `INT_MAX + 1` with the caller-qualified
  typed message `N exceeds the int-indexed clustering result limit.` The
  dimension-tagged test binary must report zero failures and no skips.
- **BAND-EFFECTS [HARD]:** `fast_pam`, `fast_pam_seeded`, and `fast_pam_swap`
  all resolve the checked point count before `fillDistanceMatrix()` or any
  result-state mutation.
- **BAND-PARITY [HARD]:** the live C++ conformance route retains the recorded
  digit-identical labels `0×9,1×9,2×9` and medoids `4,13,22`; its three recorded
  scores remain within the permanent `1e-12` relative gate.
- **BAND-FOCUSED [HARD]:** the complete `unit_test_faster_pam` and
  `unit_test_fast_pam` binaries run with zero failures and no skips.
- **BAND-CANONICAL [HARD]:** rebuild succeeds and CTest reports `114/114`,
  zero failed, with exactly the six registered capability skips
  (`cuda_correctness`, `cuda_lb_keogh`, `io_readers`, `metal_correctness`,
  `metal_lb_keogh`, `metal_mmap`).

## Runs

### Valid-input baseline before repair

Commands:

```text
.\build\highs-1151\bin\cpp_conformance.exe
.\build\highs-1151\bin\unit_test_faster_pam.exe
.\build\highs-1151\bin\unit_test_fast_pam.exe
```

Verbatim summaries:

```text
All tests passed (7 assertions in 1 test case)
All tests passed (240 assertions in 7 test cases)
All tests passed (76 assertions in 14 test cases)
```

The conformance oracle in
`tests/conformance/conformance_reference.txt` records:

```text
labels,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2
medoids,4,13,22
silhouette,0.96894972764334841
davies_bouldin,0.038333333333333337
dunn,11.5
```

### Deliberate red run

The new production checker initially retained the old unchecked
`static_cast<int>(n_points)`. Command:

```text
.\build\highs-1151\bin\unit_test_faster_pam.exe "[dimensions]"
```

Verbatim decisive output:

```text
C:/D/git/dtw-cpp/tests/unit/algorithms/unit_test_faster_pam.cpp(176): FAILED:
  CHECK_THROWS_AS( (void)checked_fast_pam_point_count(size_int_max + 1, "fast_pam"), InvalidInput )
because no exception was thrown where one was expected:

C:/D/git/dtw-cpp/tests/unit/algorithms/unit_test_faster_pam.cpp(179): FAILED:
  CHECK_THROWS_WITH( (void)checked_fast_pam_point_count(size_int_max + 1, "fast_pam"), "fast_pam: N exceeds the int-indexed clustering result limit." )
because no exception was thrown where one was expected:

===============================================================================
test cases:  2 |  1 passed | 1 failed
assertions: 18 | 16 passed | 2 failed
```

**BAND-RED: PASS.**

### Repaired boundary and focused gates

After comparing `n_points` to `INT_MAX` in its original unsigned type:

```text
Filters: [dimensions]
Randomness seeded to: 770587042
===============================================================================
All tests passed (18 assertions in 2 test cases)
```

Complete focused suites and live conformance:

```text
All tests passed (258 assertions in 9 test cases)
All tests passed (76 assertions in 14 test cases)
All tests passed (7 assertions in 1 test case)
```

**BAND-BOUNDARY: PASS.**

**BAND-FOCUSED: PASS.**

**BAND-PARITY: PASS.**

### Effect ordering

**[confirmed]** `checked_fast_pam_point_count` or
`resolve_fast_pam_plan` precedes `fillDistanceMatrix()` in all three production
routes in `dtwc/algorithms/fast_pam.cpp`. The focused effect test also executes
all three empty-problem routes and proves invalid cluster counts/medoids leave
the matrix unfilled.

**BAND-EFFECTS: PASS.**

### Canonical gate

Commands:

```text
cmake --build build/highs-1151
ctest --test-dir build/highs-1151 -j4 -C Release --output-on-failure
```

The rebuild exited zero. Verbatim CTest verdict:

```text
100% tests passed, 0 tests failed out of 114

Total Test time (real) =  33.35 sec

The following tests did not run:
	 48 - test_cuda_correctness (Skipped)
	 50 - test_cuda_lb_keogh (Skipped)
	 54 - test_io_readers (Skipped)
	 55 - test_metal_correctness (Skipped)
	 56 - test_metal_lb_keogh (Skipped)
	 57 - test_metal_mmap (Skipped)
```

**BAND-CANONICAL: PASS.**

## Verdict

**REPAIR / PASS.** The supported point-index domain is explicitly
`1..INT_MAX`, matching `Problem::dist_by_ind(int, int)` and the
`vector<int>` clustering result ABI. Every public FastPAM entry now validates
that domain before effects and then uses one checked `size_t -> int`
conversion; widening internal loops no longer obscures the real boundary.

Rollback: revert the finding commit; no data migration or external state is
involved.

The claim most likely to be wrong is that public-route composition is fully
proved for an oversized concrete `Problem`: constructing more than `INT_MAX`
resident series is not a responsible test. The allocation-free production
checker proves the boundary arithmetic, and source/order plus executable
empty/invalid cases prove reachability before effects; an injectable
size-provider would be needed to execute the oversized public object path
without allocating the dataset.
