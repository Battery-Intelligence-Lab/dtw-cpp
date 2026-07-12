# Phase 8.2 F6 -- FastCLARA parent-memory contract

Date: 2026-07-12 (Europe/London)

## Confirmed failure

The non-full resident FastCLARA path computed only N*k assignment distances,
but routed them through `Problem::dist_by_ind`. Its first non-diagonal lookup
resized `DenseDistanceMatrix`, allocating all `N*(N+1)/2` packed doubles. The
existing regression checked only `count_computed()`, so it certified low work
while missing quadratic storage.

The strengthened N=500 regression failed before the fix:

```text
REQUIRE(parent_matrix.size() == 0)
with expansion: 500 == 0
unit_test_wave2a_integration: 951 passed, 1 failed
```

At that dimension the hidden backing vector contained 125,250 slots even
though only a small fraction held computed distances.

## Resolution and contract decision

Resident assignment now binds `Problem::dtw_function()` or
`dtw_function_f32()` once, serially, then evaluates that read-only dispatcher
directly inside the OpenMP point loop. DTW kernel scratch is thread-local.
Diagonal zero, strict medoid-order tie breaking, and the final index-ordered
cost reduction are unchanged. Parent distance storage is never accessed.

Peak non-full FastCLARA storage is now O(N + s^2): O(N) labels and nearest
costs plus the O(s^2) matrix of the current PAM subsample. A full-data fallback
remains PAM and intentionally owns O(N^2) storage.

The explicit semantic contract is the configured bound-DTW function, not
arbitrary values injected into a parent cache. A sentinel test preloads two
impossible cache values, proves results equal a fresh Problem, and verifies the
cache's size, packed count, computed count, and values remain byte-stable.
This also covers valid pre-existing dense/mmap caches: they are ignored and
unchanged. Existing stale raw configuration repair may detach a cache before
assignment, as it did before this finding.

## Validation

Focused canonical runs with four OpenMP workers:

```text
unit_test_fast_clara:        838 assertions, 20 cases
unit_test_wave2a_integration: 954 assertions, 15 cases
```

The same two executables passed under all three accepted sanitizer toolchains:

- MSVC 19.50 `/fsanitize=address`, no report;
- Clang 18.1.3 UBSan on Ubuntu 24.04 WSL, no report;
- Clang 18.1.3 plus libomp TSan, four workers, no report and exit code 0.

Coverage includes float64, float32, multivariate, ZeroCost/missing-data,
parallel determinism, exact cost recomputation, an initially empty parent, and
the sentinel-filled parent. Independent adversarial re-review returned PASS:
dispatcher repair precedes OpenMP, callables are read-only/thread-local, and
the cache-independence behavior is explicitly mutation-pinned.

Final canonical Windows Clang 21.1.8 Release gate (HiGHS, Gurobi, llfio ON):

```text
100% tests passed, 0 tests failed out of 113
Total Test time (real) = 124.74 sec
6 explicit capability skips
```
