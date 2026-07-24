# F13 nearest-medoid assignment contract - 2026-07-24

## Scope and base

- Branch: `Claude`
- Base commit: `1af0aa85acb1add898a3556e6ad93dcfd92f9f42`
  (`docs: record F12 local parity verdict`)
- Canonical build: `build/highs-1151` (clang, Ninja, Release, HiGHS ON,
  llfio ON, Arrow OFF).
- Arrow build: `build/arrow-pyarrow-23` (clang, Ninja, Release, Arrow and
  Parquet supplied by PyArrow 23).
- Subject: first-medoid-slot tie behavior, index-ordered objective
  accumulation, rejection of computed non-finite assignment state, exact
  CPU-f32 no-path translation, and publication of a valid finite `DBL_MAX`
  objective.

This registration precedes every new F13 regression test, production edit,
and decisive F13 execution. Scratch seed probes and the existing Parquet
fixture runs used to select discriminators are exploratory only.

The killed-ideas section, the archived plan, and `.claude/LESSONS.md` were
searched before selecting the repair boundary. In particular, FastCLARA must
continue to assign through its bound resident/chunk distance functions and
must not allocate or populate the parent O(N^2) cache killed by F6. R4 retains
ownership of full scan consolidation; F13 may introduce only the minimum
shared validation and ordered-sum policy required to fix the registered
behavior.

## Confirmed inherited inventory

Source inspection found six algorithm assignment bodies:

1. FastPAM nearest and second-nearest scan
   (`dtwc/algorithms/fast_pam.cpp:94-127`);
2. CLARANS initial assignment (`clarans.cpp:98-112`);
3. CLARANS accepted-swap refresh (`clarans.cpp:190-203`);
4. resident FastCLARA, instantiated for f64 and f32
   (`fast_clara.cpp:159-190`);
5. streamed FastCLARA f64 (`fast_clara.cpp:225-284`);
6. streamed FastCLARA f32 (`fast_clara.cpp:287-344`).

The audit additionally found the public Lloyd medoid assignment body in
`Problem::assign_clusters` (`dtwc/Problem.cpp:1003-1023`). It is included in
F13 rather than left as an unrecorded seventh copy.

All seven use a strict comparison, so an exact tie selects the first medoid
**slot**, not the smallest global medoid index. None explicitly rejects a
non-finite computed distance. FastPAM totals with `std::reduce`, CLARANS uses
an inline left fold, and FastCLARA uses its own forced global-index left fold.
The purported common scan helpers in
`dtwc/algorithms/detail/medoid_utils.hpp` are not called by production.

CLARANS and FastCLARA initialize the best result objective to `DBL_MAX` and
update only for a strict decrease. A valid result whose finite objective is
exactly `DBL_MAX` therefore leaves their published result empty.

CPU float32 DTW returns its finite no-path marker as a double widened from
`FLT_MAX`. Its exact binary64 bits are `0x47efffffe0000000`, while the public
no-path contract is `DBL_MAX`, bits `0x7fefffffffffffff`. F12 repaired this
same compute-type/public-type boundary for GPU results only; CPU-f32 remains
an independent confirmed defect.

## Independent assignment oracle

The permanent test-local oracle accepts a literal point-by-medoid-slot table.
It does not call `Problem`, DTW, or a production medoid helper.

For each row it:

1. validates candidates in ascending slot order and rejects every NaN or
   infinity;
2. ranks `(distance, slot)` lexicographically, so exact and signed-zero ties
   select the first slot;
3. tracks candidate presence with booleans rather than a floating sentinel;
4. folds selected distances in increasing point order through a forced store;
5. rejects the first non-finite accumulated objective; and
6. canonicalizes an exact-zero public objective to positive zero.

Negative finite values remain valid because Soft-DTW objectives may be
negative. Exact `DBL_MAX` also remains a valid finite no-path value.

The literal oracle discriminators are:

| discriminator | medoid slots / selected distances | required result |
|---|---|---|
| midpoint tie | medoids `{0,2}` over singleton values `{0,1,2}` | labels `{0,0,1}`, objective `1`, bits `0x3ff0000000000000` |
| slot-not-index tie | medoids `{2,0}` over `{0,1,2}` | labels `{1,0,0}`, objective `1` |
| ordered cancellation | medoids `{0,4}` over `{0,2^53,1,1,0}` | labels `{0,0,0,0,0}`, objective `2^53`, bits `0x4340000000000000` |
| signed zero | row `{-0.0,+0.0}` | slot `0`, positive-zero objective bits `0x0000000000000000` |
| exact finite sentinel | row `{DBL_MAX,DBL_MAX}` | slot `0`, nearest/second/objective bits `0x7fefffffffffffff` |
| finite objective overflow | selected distances `{0x1.8p+1023,0x1.8p+1023}` | reject after point `1` |

The ordered-cancellation left fold is:

```text
((((0 + 0) + 2^53) + 1) + 1) + 0 = 2^53
```

Grouping both unit contributions before the large term instead produces the
next binary64 value `0x4340000000000001`. The inherited MSVC-STL
`std::reduce` happened to match the left fold in a scratch five-element probe;
that exploratory pass is not a contractual ordering guarantee.

For each of qNaN, `+inf`, and `-inf`, the oracle places the poison once in
slot 0 and once in slot 1 beside finite `1.0`. All six cases must reject the
poisoned coordinate even when it would not win the nearest comparison.

## Registered live-route fixtures

### A. Common exact tie and slot ordering

The midpoint fixture uses singleton series `{0,1,2}`.

- FastPAM: fixed medoids `{0,2}`, `max_iter=0`.
- CLARANS: `k=2`, `num_local=1`, `max_neighbor=0`, portable-v1 seed `4`;
  the initial sorted medoids are `{0,2}`.
- resident FastCLARA f64 and f32: `k=2`, `sample_size=2`, `n_samples=1`,
  seed `0`; the sample and mapped medoids are `{0,2}`.
- Lloyd: `centroids_ind={0,2}` followed by the public
  `Problem::assign_clusters`.

Every route must publish medoids `{0,2}`, labels `{0,0,1}`, and objective
exactly `1` where the route publishes an objective.

FastPAM repeats with caller medoid order `{2,0}` and resident FastCLARA repeats
with seed `10`, whose mapped medoid order is `{2,0}`. Both must publish labels
`{1,0,0}`. This kills an incorrect “smallest global index wins” repair.

The CLARANS accepted-swap copy uses singleton values
`{4,0,0,10,0,5,0,10}`, `k=2`, `num_local=1`, `max_neighbor=1`,
`max_dtw_evals=24`, and portable-v1 seed `317`. The registered schedule starts
at `{0,3}`, proposes remove-slot-0/add-point-6, and stops at the budget after
acceptance. The result is medoids `{6,3}`, labels
`{0,0,0,1,0,0,0,1}`, and objective exactly `9`. Point 5 is equidistant from
global medoids 6 and 3 and must select slot 0 even though its global index is
larger.

### B. Ordered objective

Singleton values `{0,2^53,1,1,0}` use duplicate-valued but distinct medoid
points `{0,4}`.

- FastPAM: fixed medoids `{0,4}`, `max_iter=0`.
- CLARANS: `k=2`, `num_local=1`, `max_neighbor=0`, seed `20`.
- resident FastCLARA f64 and f32: `k=2`, `sample_size=2`, `n_samples=1`,
  seed `11`.

All labels are slot 0. Every objective must have exact bits
`0x4340000000000000`. No relative or absolute tolerance is permitted.

### C. Non-finite distance and objective

FastPAM and CLARANS receive a complete symmetric three-point dense matrix
whose point 1 distance to medoid slot 0 is `+inf` while its distance to slot 1
is finite. They must reject the non-winning infinity rather than silently
choose slot 1. A `-inf` variant must also reject. Lloyd drives the same cached
matrix through `Problem::assign_clusters`.

Resident FastCLARA uses 65 singleton series: point 0 is `-DBL_MAX` (or
`-FLT_MAX`) and points 1..64 are the corresponding positive maximum. With
`k=1`, `sample_size=2`, `n_samples=1`, seed `0`, portable sampling selects
points 25 and 59, so the subproblem is finite and the full resident assignment
computes the poison at point 0. `N=65` forces the OpenMP branch. Both f64 and
f32 must throw `dtwc::InvalidInput` after the join, never terminate inside the
structured block.

The exact diagnostics are:

```text
<caller>: non-finite nearest-medoid distance at point P, medoid slot S (index I).
<caller>: nearest-medoid objective became non-finite after point P.
```

Registered callers are `fast_pam`, `clarans`, `fast_clara`, and
`kmedoids_lloyd`. Floating values are deliberately absent from messages
because NaN spelling and payload rendering are not portable.

Finite-distance objective overflow uses f64 singleton values
`{0,H,H,0}`, `H=0x1.8p+1023`, and medoids `{0,3}`. Each selected distance is
finite, but the point-ordered objective first becomes non-finite after point
2. FastPAM uses fixed medoids, CLARANS uses seed `0`, and resident FastCLARA
uses seed `8`; all must reject with the exact objective diagnostic.

### D. CPU-f32 no-path and finite-`DBL_MAX` publication

Two all-zero series of lengths 1 and 3 use `band=0`, `k=1`. The exact DTW
result between them is the finite no-path sentinel.

FastPAM, CLARANS, and resident FastCLARA must each publish:

```text
labels          = {0,0}
medoid count    = 1
objective bits  = 0x7fefffffffffffff
```

The result must be populated in both f64 and f32. A direct CPU-f32 DTW call
must also translate exact compute `FLT_MAX` to public `DBL_MAX`, preserve
`nextafter(FLT_MAX,0)` as an ordinary finite value, and leave infinities/NaNs
available for the algorithm-level rejection policy.

CPU-f32 public normalization is one implementation commit. Best-result
presence and assignment validation are a separate F13 implementation commit.

### E. Real Arrow CLI copies

No tracked data file is modified. The existing read-only fixture
`tests/fixtures/fast_clara_streaming_8x4.parquet` remains pinned to:

```text
size   = 1451
SHA256 = 2F259F418A6BB9C62CA0004CB334C05E8309213F5A76DC890F83C15D5BDA3CA8
```

The real CLI runs four configurations with `k=2`, `sample_size=2`,
`n_samples=1`, seed `40`: resident and forced-by-budget streaming for each of
Float64 and Float32. Every process must prove its mutually exclusive route
marker and execute FastCLARA. Required medoids and labels are:

```text
medoids = {0,3}
labels  = {0,0,0,1,1,1,1,1}
```

Series 1 is an exact tie between both medoids and must select slot 0. Resident
and stream artifacts must be byte-identical within each precision. Exact
checkpoint objective bytes at offset 24 are:

| precision | decimal | little endian | big endian |
|---|---:|---|---|
| f64 | `156.30000000000001` | `9A99999999896340` | `406389999999999A` |
| f32 | `156.30000233650208` | `0000809E99896340` | `406389999E800000` |

The f64 and f32 objectives are not required to equal one another; each
resident/stream pair is required to be digit-identical, while medoids and
labels are identical across all four routes.

### F. Streamed non-finite mutation discriminator

Independent gate audit found that fixture E is entirely finite, so it cannot
by itself kill removal of validation from either streamed scan. Before that
mutation is run, extend the Arrow-only gate with an Arrow-linked helper that
generates two **build-local** Parquet files under the test-owned work root;
no tracked data file is added or modified.

The first registered topology (65 rows in one row group, 900-byte cap) is
FALSIFIED below: sample loading must materialize that whole group and fails
before assignment. The second and final topology contains 129 singleton rows
split into row groups of 65 and 64. Row 0 is the negative maximum finite value
and rows 1..128 are the positive maximum finite value, using binary64 values
for the f64 file and binary32 values for the f32 file. With `k=1`,
`sample_size=2`, `n_samples=1`, portable-v1 seed `0`, and a 5000-byte cap, the
selected medoid at global index 65 is a finite positive, full resident loading
is rejected by the cap, and the first streamed assignment chunk has 65
points. It therefore enters the `chunk_size > 64` OpenMP branch and computes
a non-finite distance for point 0.

The real CLI must run both files in forced streaming mode and fail with the
exact diagnostic:

```text
fast_clara: non-finite nearest-medoid distance at point 0, medoid slot 0 (index 65).
```

Both processes must prove the streaming route and FastCLARA execution before
the expected failure. The gate reports `stream_rejections=2/2`; bypassing
validation in either streamed precision must fail this sub-band.

## Acceptance band

F13 passes locally only if every item below holds:

1. The independent oracle reproduces every literal label, floating bit
   pattern, poison coordinate, and overflow point above before judging
   production.
2. The CPU-f32 sentinel test is run red on the registered base, then passes
   after one dedicated implementation commit. Exact `FLT_MAX` maps to exact
   `DBL_MAX`; the adjacent finite float is not mistaken for a sentinel.
3. A new public-route C++ assignment target runs at least 50 assertions in at
   least 8 Catch2 cases, with no skip. It covers FastPAM, both CLARANS
   assignment copies, resident FastCLARA f64/f32 including its parallel
   failure path, Lloyd, finite `DBL_MAX`, non-winning infinities, and
   finite-distance objective overflow.
4. The new Arrow-only real-CLI target exists only when Parquet is linked. It
   runs four successful processes, proves 8/8 required/forbidden route
   markers, compares 6/6 resident/stream artifact pairs, and validates all
   four exact assignment payloads and objective bytes. It additionally runs
   the two registered build-local poison files and proves 2/2 exact streamed
   rejection paths. Skip text or a missing subject is failure.
5. Mutation probes must make the focused gate fail when strict `<` becomes
   `<=`, a production finite check is removed, ordered accumulation is
   replaced by a grouping-permitted reduction, or either streamed scan
   bypasses validation.
6. Source audits find no assignment-distance read in the seven registered
   bodies that can silently consume NaN or infinity, no grouping-permitted
   published objective reduction, and no floating sentinel used as
   best-result presence.
7. Fresh canonical and llfio-OFF builds pass 116/116 with zero failures and
   their existing capability skips. The reconfigured Arrow build passes
   118/118, including both non-skippable real-CLI targets and the real reader.
8. A fresh Python extension imports one current symbol before the full pytest
   floor is judged; the inherited floor remains 407 passed / 11 skipped
   (418 collected).

There are at most two repair attempts. A missed band is recorded as
**FALSIFIED** with its verbatim output and is not relaxed. Rollback for either
implementation is a local `git revert` of its dedicated commit. The claim
most likely to be wrong is that the five-element cancellation fixture will
distinguish the inherited `std::reduce` on this specific standard-library
implementation; its value is still a permanent ordering contract, while the
registered non-finite and sentinel failures provide the inherited red.

## Executions and verdicts

### CPU-f32 inherited red

The freshly rebuilt canonical target at registered base `f744d87` produced:

```text
Filters: [F13]
Randomness seeded to: 326414885

C:/D/git/dtw-cpp/tests/unit/core/unit_test_distance_semantics.cpp(169): FAILED:
  REQUIRE( f32_distance(short_f32, long_f32) == std::numeric_limits<double>::max() )
with expansion:
  340282346638528859811704183484516925440.0
  ==
  1797693134862315708145274237317043567980705675258449965989174768031572607800-
  2853876058955863276687817154045895351438246423432132688946418276846754670353-
  7516986049910576551282076245490090389328944075868508455133942304583236903222-
  9481658085593321233482747978262041447231687381771809192998812504040261841248-
  58368.0

===============================================================================
test cases: 1 | 1 failed
assertions: 1 | 1 failed
```

Verdict: **FALSIFIED [confirmed]**. The live CPU-f32 resolver exposed widened
`FLT_MAX` on the preregistered unequal-length no-path call.

### CPU-f32 repair attempt 1

Commit `62c6f26` introduces one shared compute/public distance normalizer and
uses it at every CPU resolver return; the F12 GPU detail seam imports the same
policy. The focused repaired gate produced:

```text
Filters: [F13]
Randomness seeded to: 3588231790
===============================================================================
All tests passed (5 assertions in 1 test case)
```

The unchanged GPU host contract produced:

```text
Randomness seeded to: 2097743251
===============================================================================
All tests passed (8 assertions in 1 test case)
```

The complete distance-semantics target produced:

```text
Randomness seeded to: 4030018866
===============================================================================
All tests passed (53 assertions in 4 test cases)
```

Verdict: **PASS [confirmed]** for the dedicated CPU-f32 boundary subtask.
Exact compute `FLT_MAX` maps to exact public `DBL_MAX`; the adjacent finite
float is preserved, and infinities/NaNs remain visible for the registered
algorithm-level rejection. The assignment/best-result half of F13 remains
open.

### Streamed rejection discriminator attempt 1

The registered 65-row, one-row-group fixture with a 900-byte cap did not
reach assignment:

```text
Parquet metadata selected streaming: 65 series, ~0 MB resident estimate exceeds the series-data cap [0:0.0055627 min:sec]
Series-data RAM limit: 900 bytes
Running FastCLARA (k=1) ...
FastCLARA: streaming from Parquet (65 rows, 1 row groups, ~0 MB resident estimate)
Error: ParquetChunkReader::read_rows: selected row group needs 4680 bytes in addition to 112 retained sample bytes, exceeding --ram-limit=900; rewrite with smaller row groups or raise the limit
```

Verdict: **FALSIFIED [confirmed]**. The assignment subject did not run.
The second and final registered topology above separates total resident size
from one materialized row group while retaining a 65-point first assignment
chunk.

### Assignment repair attempt 1 full-gate falsification

The repaired focused gate passed 113 assertions / 8 cases and the strengthened
real Arrow CLI gate passed its registered successful and rejection routes.
The first canonical full gate then exposed one stale contradictory test:

```text
C:/D/git/dtw-cpp/tests/unit/test_tier1_cpp_api.cpp(259): FAILED:
due to unexpected exception with message:
  kmedoids_lloyd: non-finite nearest-medoid distance at point 1, medoid slot 0
  (index 0).

===============================================================================
test cases:  9 |  8 passed | 1 failed
assertions: 50 | 49 passed | 1 failed

99% tests passed, 1 tests failed out of 116

The following tests FAILED:
	 67 - test_tier1_cpp_api (Failed)
```

Verdict: **FALSIFIED [confirmed]** at 115/116. The failed case explicitly
required Lloyd to publish an infinite objective, contradicting F13's
preregistered rejection of every non-finite assignment distance. Repair
attempt 2 changes only that stale expectation: require `InvalidInput` and
digit-identical preservation of the pre-call labels. The production policy is
not relaxed.

### Fresh-Python gate falsification and exact-base arbiter

The fresh current extension was built from the F13 tree and installed only
after its output timestamp advanced. The built and loaded extension hashes
were identical:

```text
5139A6745C325C8614A1191605E85227564910E49D70DBF41CE2FDDDFC58BF50
```

Its live FastCLARA smoke returned medoids `[0,2]`, labels `[0,0,1]`, and cost
`1.0`. With the Windows Arrow runtime directories prepended, the full Python
suite then produced:

```text
FAILED tests/python/test_api.py::TestClusterLocal::test_default_lloyd_seed_is_reproducible_across_calls
FAILED tests/python/test_api.py::TestClusterLocal::test_default_lloyd_seed_isolated_from_legacy_tier2_rng
FAILED tests/python/test_api.py::TestClusterLocal::test_lloyd_honors_nondefault_iteration_cap_and_keeps_default
3 failed, 1007 passed, 12 skipped in 70.10s (0:01:10)
```

The first two failures were exact `24.0 == 20.0` mismatches. The third
reported:

```text
ACTUAL: array([5, 2, 0])
DESIRED: array([6, 1, 4])
```

An isolated detached worktree at the exact registered F13 base
`1af0aa85acb1add898a3556e6ad93dcfd92f9f42` built a distinct extension with
SHA-256
`25F8D6C51B9229EC3CAFD670F52B8F13320BDD32C7FB5F84CBFDFF656E31BEF5`.
The accepted process removed the editable-install finder and current source
path, then asserted that both package and extension came from the retained
base stage. Its exact three-node run produced:

```text
collected 3 items
3 failed in 0.33s
```

The base process reproduced cost `24.0`, capped medoids `[5,2,0]`, labels
`[2,1,1,1,0,0,0,0]`, and portable-v1 initial medoids `[6,2,1]`.

Verdict: **FALSIFIED [confirmed]** for the inherited Python suite, and
**pre-existing [confirmed]** against exact base rather than inferred from
source inspection. Commit `24ef4e5` changed seeded initialization to
portable-v1 and added the current C++ initializer oracle, but these three
Python literals retained the earlier standard-library RNG trajectory.

### Registered Python-oracle repair

This registration precedes any edit to `tests/python/test_api.py`.
Production code is out of scope.

The two invocation-local default-seed cases keep their existing eight-series
fixture and must additionally pin the portable-v1 result:

```text
medoids = [5,2,0]
labels  = [2,1,1,1,0,0,0,0]
cost    = 24.0
```

Simply changing the old iteration-cap literals to those values would erase
the test's subject because `max_iter=1` and convergence now agree on that
fixture. Replace only that case's data with singleton series whose values in
point order are:

```text
[0,1,2,3,5,4]
```

Portable-v1 seed 42 selects initial medoids `[4,2]`. Independent literal L1
arithmetic gives this progression:

| state | medoids | labels after publication | objective |
|---|---|---|---:|
| `max_iter=1` | `[4,1]` | `[1,1,1,0,0,0]` | `5.0` |
| converged | `[5,1]` | `[1,1,1,0,0,0]` | `4.0` |

For the capped result, distances to values 5 and 1 are
`[1,0,1,2,0,1]`; for the converged result, distances to values 4 and 1
are `[1,0,1,1,1,0]`. Both sums and the unique second update are therefore
independent of the production objective accumulator.

Acceptance band:

1. The three focused nodes pass with the exact medoids, labels, and objectives
   above.
2. A live mutation that replaces the forwarded `max_iter` with the default
   `100` makes the cap case fail on its exact capped medoids and objective.
3. The same fresh extension provenance remains loaded. The complete current
   collection is exactly 1,022 tests and passes **1010 passed / 12 skipped /
   0 failed** with the Arrow runtime path supplied.
4. No production or binding source changes; `git diff --check` is clean.

There are at most two repair attempts. Rollback is a local revert of the
dedicated test commit. The claim most likely to be wrong is that the six-point
fixture remains discriminatory after the test enters through the full Python
dispatch; the focused real-binding run, not the hand calculation, judges it.

### Registered F13 manifest-inventory increment

The first full Python run after the portable-Lloyd test repair produced:

```text
________________ test_live_tracked_cmake_inventory_is_complete ________________

    def test_live_tracked_cmake_inventory_is_complete():
        archive_pins, manifest_total = pins.tracked_cmake_archive_pins(ROOT)
>       assert manifest_total == 25
E       assert 26 == 25

tests\python\test_supply_chain_pins.py:493: AssertionError
1 failed, 1009 passed, 12 skipped in 69.19s (0:01:09)
```

This registration precedes the inventory edit. A main-index comparison against
pre-assignment commit `e37b71a` produced:

```text
CURRENT=26
PRE_F13_CODE=25

InputObject                                                 SideIndicator
-----------                                                 -------------
tests/integration/test_fast_clara_assignment_contract.cmake =>
```

The retained exact-base worktree is not involved: `tracked_cmake_files` calls
`git ls-files`, and the one new tracked F13 gate is the entire delta.

Repair band:

1. Change only `REGISTERED_CMAKE_MANIFEST_TOTAL` and its direct unit-test
   literal from 25 to 26. Do not alter the F11/F36 parser or its seven archive
   identities.
2. The live checker reports 39/39 workflow actions, 7/7 immutable hashed
   archives, one Arrow pin, `TRACKED_CMAKE_MANIFESTS total=26`, and
   `supply-chain pins verified`.
3. Restoring either inventory constant to 25 makes the focused live-inventory
   test fail with exact `26 == 25`.
4. The complete fresh-extension Python gate then meets the registered
   1010-passed / 12-skipped / zero-failed floor.

There are at most two repair attempts. Rollback is a local revert of the
dedicated inventory commit. The claim most likely to be wrong is that no other
tracked manifest entered with F13; the exact index set comparison above, not a
filesystem walk, judges it.
