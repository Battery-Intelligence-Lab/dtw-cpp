# Adversarial review — batch 3, verification of the two fix rounds (2026-09-02)

Read-only. Nothing built or run. Line numbers are the NEW tree.
**Observed** = read in the tree/diff. **Inferred** = reasoned from observed code.

## Ranked defects

### D1. (High) The GIL policy covers 5 of ~15 mutating entry points, and mixing held/released bindings cannot prevent the race anyway
**Observed.** `dist_by_ind` (`python/src/_dtwcpp_core.cpp:933`), `find_total_cost` (`:1020`),
`read_distance_matrix` (`:1001`), `write_clusters` (`:1035`) and `write_silhouettes` (`:1045`)
now hold the GIL, with docstrings saying so. These still RELEASE it and reach the same mutating
lazy path: `read_distance_matrix_np` (`:797-812`, bound as `distance_matrix` at `:979`, calls
`prob.fill_distance_matrix()` inside the release), `fill_distance_matrix` (`:975`),
`print_distance_matrix` (`:1005`), `cluster` (`:1015`), `assign_clusters` (`:1024`),
`calculate_medoids` (`:1028`), `write_distance_matrix` (`:1040`), and every score free function
— `silhouette` (`:1348`), `davies_bouldin` (`:1353`, `:1357`), `dunn` (`:1364`, `:1369`),
`inertia` (`:1375`), `calinski_harabasz` (`:1381`, `:1385`) — each of which calls
`prob.fill_distance_matrix()` / `dist_by_ind` (`dtwc/scores.cpp:120`, `:225`).
**Inferred.** Holding the GIL in binding A gives no mutual exclusion against binding B that
released it. Thread A in `p.distance_matrix()` resizes `distMat` and rebinds `dtw_fn_` with the
GIL released while thread B runs the GIL-holding `p.dist_by_ind()` on the same object: the
original race (`distMat` reallocation under a live reader) is untouched. The five docstrings
assert a safety property the module does not have.

**Residual `mutable` member — real, but the minor part.**
`Problem::validate_mmap_cache_identity()` writes `mmap_cache_data_validated_ = true`
(`dtwc/Problem.cpp:632`) from a `const` method; `print_distance_matrix` (`Problem.cpp:222`) and
`write_distance_matrix` (`Problem.cpp:~640`) are `const`, are called under a released GIL, and
both reach it. Two Python threads therefore perform unsynchronised writes to one `bool` — a data
race by the C++ memory model (plus a duplicated O(N·L) `distance_cache_identity` hash) — but they
are same-value byte stores, benign on all real ISAs. It is not the sharp edge; `distMat` is.

**Minimal fix.** GIL policy is the wrong tool. Give the binding layer one lock per `Problem`
(a `std::shared_ptr<std::recursive_mutex>` in the nanobind holder), acquire it in every `Problem`
method *before* `nb::gil_scoped_release`, and revert the five bindings to releasing. If that is
out of scope, delete the "holds the GIL … so it is not thread-safe" claims and document `Problem`
as single-thread-only. For the `mutable` flag alone: `std::atomic<bool>` at `Problem.hpp:149`.

### D2. (Medium) The silhouette catch is over-broad: genuine data and labelling errors are swallowed as warnings
**Observed.** `dtwc/Problem_IO.cpp:143-148` and `dtwc/api.cpp:275-280` catch `const InvalidInput &`.
`scores::silhouette` raises `InvalidInput` from four distinct causes: `cluster_counts_checked`
size disagreement (`scores.cpp:50`), out-of-range label (`:58`), `require_two_realised` (`:81`),
and — via `prob.fill_distance_matrix()` at `scores.cpp:120` — the NaN pre-scan
(`Problem.cpp:855`), the new all-NaN Interpolate scan (`:871`) and the new Pruned+f32 guard
(`:918`). `InvalidInput` is a leaf sibling of `IOError`/`DeviceError` (`error.hpp:51-75`), so
those still propagate.
**Inferred.** A corrupt `clusters_ind` ("labels disagree") or NaN input now prints
`Warning: silhouettes skipped: …` and `cluster_and_process()` / `Result::save()` report success
with a missing file. This is batch-2 item 5 (unify to one type) applied without a distinct type
for the undefined-score case.
**Minimal fix.** `class UndefinedScore : public InvalidInput` in `dtwc/error.hpp`; throw it only
from `require_two_realised` (`scores.cpp:81`); catch that in both save paths.
Warning destination is correct: both use `std::cerr` (Observed), so stdout-identity tests are safe.

### D3. (Low–Medium) Explicit `Pruned` + Float32 + mmap distance storage now throws where it used to work
**Observed.** The f32 guard is at `dtwc/Problem.cpp:911-922`, **above** the
`effective == Pruned && has_mmap_storage → BruteForce` downgrade at `:925`, which routes to the
exact generic row fill and handles f32 correctly.
**Minimal fix.** Move the f32 guard below the `has_mmap_storage` downgrade block.

### D4. (Low) Batch-2 item 5 only half-applied
`scores.cpp:181` (DBI), `:255` (dunn), `:303` (inertia), `:329`, `:340`, `:342`
(calinski_harabasz) still throw bare `std::runtime_error`. Harmless today, goal unmet.

### D5. (Low) `Problem.hpp` dropped `<type_traits>`; `Problem_IO.cpp` still uses it
`Problem_IO.cpp:196` and `:247` use `std::is_same_v<std::decay_t<…>>` and include neither
`<type_traits>` nor a header that guarantees it — it compiles only through `<variant>`.
Add `#include <type_traits>` to `Problem_IO.cpp`.

## Confirmed fixed — the 15 prior items

**Batch 1**
1. GIL on mutating bindings — **partially fixed (D1)**. Five entry points changed; ten
   equivalent ones still release.
2. `Data::series()` f32 throw reachable via Pruned — **fixed**. `!prob.data().is_f32()` in
   `pruned_strategy_applicable` (`Problem.cpp:751`) plus an explicit named guard (`:918`). Both
   sit before any OpenMP region; the earlier NaN/Interpolate pre-scan (`:846-876`) is a separate,
   non-conflicting block. No f64 false positive (`is_f32()` only). Covered non-vacuously by
   `unit_test_pruned_distance_matrix.cpp:1109-1152`. Caveat D3.
3. Checkpoint metric not reaching Python — **fixed**. `metric` added to both bindings
   (`_dtwcpp_core.cpp:1278-1307`, default `L1`); `distance_checkpoint_identity(metric)`
   (`Problem.cpp:579`); friend/decl updated (`Problem.hpp:47,215`);
   `tests/python/test_distance_matrix.py:324-354` asserts SquaredL2/L1 rejection.
4. `decode_pair` per-pair cost + clamp order — **fixed**. Hand-verified: `row_start` hoisted once
   (`decode_pair.hpp:71`), down-loop costs one comparison on the fast path (`:77`). N=2,k=0→(0,1);
   N=3: k=0→(0,1), k=1→(0,2), k=2→(1,2) (seed row=1, `row_start=2`, both loops no-op). Clamps now
   high-then-low (`:60-62`): N=1,k=0 → seed 0 → high −1 → low 0; N=0 → seed −1 → high −2 → low 0.
   Both terminate with `i ≥ 0` (the old up-loop spun forever at N=1: width `N−row−1 = 0` kept the
   `<= k` test true). The `row + 1 < N` bound can never block a needed correction: the true row is
   ≤ N−2, so `row+1 ≤ N−1 < N` whenever an increment is required. MSL mirror (`:123-133`) matches
   statement-for-statement; it needs no high clamp because its exact isqrt gives row ≤ N−2
   (k_max ⇒ disc = 9, s = 3, row = N−2).
5. `test_io_readers` count band — **fixed**. `tests/CMakeLists.txt:617-637` keeps the skip
   rejection, drops the upper bounds, keeps a one-sided ≥300 floor and an unbounded test-case
   count. Guard predicate is sound: `DTWC_HAS_ARROW` is `PUBLIC` (`dtwc/CMakeLists.txt:200,204`)
   so it does reach `INTERFACE_COMPILE_DEFINITIONS`.
6. `query_gpu_config` unpublished slot — **fixed**. Returns `static const GPUConfig kUnavailable{}`
   on failure (`gpu_config.cuh:72-75`); cache hit is an acquire load; all three call sites
   (`cuda_dtw.cu:81,90,1254`) bind by value or `const&`, so the new return type compiles.
7. OpenMP error-slot sizing — **fixed**. `num_threads(n_error_slots)` (`_dtwcpp_core.cpp:1103`).
8. mmap CSV double scan — **fixed** (see item 15 / matrix_io below).
9. Temp-path collision — **not addressed** (filed as documentation drift only).
10. Pruned-checkpoint test scenario — **fixed**. New `unit_test_pruned_distance_matrix.cpp:1160+`
    builds the N=64 / band≥0 configuration `Auto` really resolves to `Pruned` and drives it
    through `Problem::fill_distance_matrix()`.

**Batch 2**
11. Silhouette throw breaking `cluster_and_process`/`Result::save` — **fixed, with D2**.
    `Problem_IO.cpp:143-148`, `api.cpp:275-280`; the `return` in `save()` is safe (silhouettes is
    the last block, `api.cpp:281-289`). Regression test `unit_test_Problem.cpp:109-137` asserts
    `REQUIRE_NOTHROW(cluster_and_process())` plus file present/absent.
12. `one_batch_pam.total_cost` accumulator undocumented — **out of this scope** (CHANGELOG-only).
13. TADPole f32 + stale rationale + no signal — **fixed**. `bounds_valid` gains `!is_f32()`
    (`tadpole.cpp:73`), comment rewritten for post-A12 semantics (`:59-71`),
    `TADPoleStats::pruning_enabled` added (`tadpole.hpp:68`, set at `tadpole.cpp:324`),
    `@param prune` documents "NO PRUNING ON FLOAT32" (`tadpole.hpp:88-93`).
    **`can_prune` hoist adds no per-pair cost (Observed):** density loop — the `!can_prune` branch
    makes zero `series()` calls (was one per pair); delta loop — `prob.series(q)` is now called
    once per candidate instead of up to twice (`tadpole.cpp:243-247`). The f64 path is
    structurally identical: same order, same `si.size()==sj.size()` gate, same LB/UB/exact
    cascade, same `rho_local` updates.
14. Vacuous tadpole assertions — **fixed**. `unit_test_tadpole.cpp:363-431` asserts
    `pruning_enabled == false`, `pruned_by_lb == pruned_by_ub == 0`,
    `dtw_calls == brute dtw_calls`, plus an f64 control asserting `pruned_by_lb + pruned_by_ub > 0`
    — non-vacuous. The `one_batch_pam` `returned + threw == 64` tautology was left (harmless).
15. Score taxonomy — **partially fixed (D4)**. Silhouette, DBI and Dunn now throw `InvalidInput`;
    Calinski-Harabasz and the "cluster first" guards still throw `std::runtime_error`.

**Also verified in scope, no defect**
- `envelope_covers` (`lower_bound_impl.hpp:337`, `>=`) vs `envelope_sizes_ok` (`:347`, `==`;
  `WebbEnvelope` overload `:805`). Overload resolution is exact — `WebbEnvelope` is an unrelated
  aggregate with no conversion to `Envelope`, and its overload is declared before `lb_webb`
  (`:853`). No caller passes a `WebbEnvelope` to an `Envelope` helper: the only production sites
  are `pruned_distance_matrix.cpp:237` (`lb_keogh_symmetric`), `:243` (`lb_enhanced_symmetric`),
  `:249` (`lb_webb_symmetric`, `WebbEnvelope`) and `tadpole.cpp:196,243` (`Envelope`). The
  `std::min` prefix truncation is preserved and IS the live path (span overload `:378-382`, with
  the vector overload now delegating to it so both share one contract);
  `unit_test_lower_bounds.cpp:317-331` exercises `upper` shorter than `query` and asserts the
  non-zero prefix bound 2.0.
- Preflighted emitter split is byte-identical: `matrix_io.hpp:180-201` is the previous loop
  verbatim (same `distance_matrix_csv_token`, same `,`/`\n`, same empty-token skip), now templated
  and called by both `operator<<` overloads after their own preflight. No-partial-output ordering
  is preserved in `Problem_IO.cpp`: `preflight_distance_matrix_csv(m)` at `:200` precedes the
  `std::ios::trunc` open at `:202`.
- `parallelisation.hpp` `failure = current;` (`:142`) — behaviour-identical to the moved-from
  version (`current` has no later use; the copy is one refcount increment inside a critical region
  entered only on failure).
- `settings.hpp` did **not** drop `<iostream>`/`<string>`; both are kept with an explicit
  justification (`settings.hpp:18-30`), so the four TUs using `std::cout`/`std::cerr` without
  `<iostream>` (`tests/unit/core/unit_test_pruned_distance_matrix.cpp`,
  `tests/unit/test_storage_policy.cpp`, `benchmarks/UCR_dtwc.cpp`,
  `examples/cpp/example_project/main.cpp`) still compile via `Problem.hpp:18 → settings.hpp`.
  The removal that did happen is `#include "settings.hpp"` from `parallelisation.hpp`; all 12
  in-tree includers were checked and none relies on it (`barycenter.cpp` and `initialisation.cpp`
  use `data_t` but reach `settings.hpp` through `Problem.hpp` / a direct include;
  `test_runtime_loudness_compute.cpp` through `<dtwc.hpp>`). `types/Range.hpp` is correctly
  retained for `scores.cpp`.

**Tests that would now pass vacuously:** none found among the new cases. The two that could have
been (f32 pruned routing, f32 TADPole) each carry an explicit non-vacuity control.
`test_io_readers`' guard is attached only when `DTWC_HAS_ARROW` is published — correct, but inert
in the canonical `DTWC_ENABLE_ARROW=OFF` gate, as its own comment states.
