# Session: Adversarial Audit + Stress Test (2026-04-07)

## What was done

### Bug fixes (commits a7fec14, 84eb9db, d3571e4, 4f17cc2)

Fixed 16 bugs total across 10 files. Key changes:

1. **DenseDistanceMatrix sentinel: -1.0 → NaN** — Soft-DTW returns negative distances; old sentinel broke caching, `is_computed()`, `all_computed()`. Now uses `std::numeric_limits<double>::quiet_NaN()`.
2. **ignoreBOM rewrite** — was OOB read at `BOMchars[3]` + multi-putback (non-portable). Now uses `tellg()`/`seekg()`.
3. **readFile** — unchecked `iss >> p_i` pushed garbage on parse failure. Now guarded.
4. **dtwc_cl YAML** — values bypassed CLI11 validators (silent no-op). Added post-parse `to_lower` + alias mapping + `input_file`/`n_clusters` validation. Moved help check before CLI11_PARSE. Removed `->required()` from `--input`.
5. **dtwAROW_banded** — no length-1 guard; returned maxValue. Added guard in both overloads.
6. **adtwBanded** — fallback paths dropped `early_abandon` parameter. Fixed all 3 fallback sites.
7. **scores.cpp** — silhouette div-by-zero on empty clusters. CH returns +inf when W==0.
8. **distByInd** — lazy-alloc race condition. Now uses `omp critical` double-check pattern.
9. **Review fixes** — `max()` init -inf with 0.0 fallback; YAML alias mappings for `hclust`, `sqeuclidean`, `l2sq`, `soft-dtw`.
10. **CI** — gcc-11 → gcc-13 to match system gcov. Added `#include <cstddef>` for GCC Linux.
11. **Hierarchical CLI** — missing `prob.fillDistanceMatrix()` before `build_dendrogram()`.

### Tests added

- `tests/unit/unit_test_accuracy.cpp` — 39 accuracy tests (cross-variant, cross-metric, early-abandon, numerical stability, lower bounds, multivariate, fuzz)
- `tests/integration/stress_test_cli.sh` — 44-test CLI stress suite
- `.claude/openmp-crashcourse.md` — OpenMP reference doc

### Stress test results (Windows MSVC Release)

```
PASS:  40
FAIL:  1   (hierarchical + SoftDTW on dummy data — crashes, needs investigation)
SKIP:  3   (SoftDTW + CLARA/kmedoids on dummy too slow; ARI python check failed)
TOTAL: 44
```

**All Phase 2 (Coffee UCR correctness) and Phase 3 (checkpoint, dist-matrix reload, banded, SoftDTW sentinel, k=1, k=N, ADTW banded, WDTW sweep) passed.**

## Uncommitted changes

| File | Status | Description |
|------|--------|-------------|
| `dtwc/dtwc_cl.cpp` | Modified | hierarchical fillDistanceMatrix fix |
| `tests/integration/stress_test_cli.sh` | New | CLI stress test script |
| `.claude/reports/snuggly-yawning-twilight.md` | New | Plan file (can delete) |

**Action needed:** commit these, then push branch Claude.

## Open bugs (remaining)

1. **hierarchical + SoftDTW crashes** — `build_dendrogram` or the distance fill with SoftDTW on large series (5000 pts) fails. Need to reproduce and debug. Might be the `max()` call in hierarchical expecting non-negative distances.
2. **`set_if_unset` in YAML** — unconditionally overrides CLI values (known, not yet fixed).
3. **MV banded DTW silently ignores band** for Missing/WDTW/ADTW variants.

## What to do next (on Mac Studio or any machine)

### Immediate
1. `git checkout Claude && git pull`
2. Commit the uncommitted changes (hierarchical fix + stress test)
3. Run: `bash tests/integration/stress_test_cli.sh` — verify same results
4. Investigate hierarchical+SoftDTW crash
5. Fix ARI check in stress test (Python path issue on Windows)

### Next phase (from TODO.md)
- **OpenMP scheduling benchmark** — `schedule(dynamic,1)` vs `dynamic,16` vs `guided`
- **CLARA sample size scaling** — `sqrt(N)` for large N
- **Algorithm auto-selection** — cost model improvement
- **Phase 1 architecture** — `DataSource` interface for out-of-core (if targeting 5TB)

### Build instructions (any platform)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_TESTING=ON
cmake --build build --config Release --parallel $(nproc)
ctest --test-dir build -C Release -j4  # expect 60/63 pass (CUDA x2 + Benders)
bash tests/integration/stress_test_cli.sh  # expect 40/44
```

## Key architectural decisions made this session
- NaN sentinel is safe under MSVC `/fp:precise` and GCC/Clang default. Would break under `-ffast-math` or MSVC `/fp:fast` (but we removed those in previous session).
- `omp critical` double-check for distByInd lazy-alloc is preferred over requiring callers to pre-fill (public API contract was already violated by existing code).
- CLARA intentionally does NOT compute full distance matrix or silhouettes — this is correct behavior, not a bug.
