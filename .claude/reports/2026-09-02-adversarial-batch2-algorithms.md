# Adversarial review — batch 2, clustering-algorithm layer (2026-09-02)

Read-only review of the uncommitted diff in `dtwc/algorithms/**`, `dtwc/scores.cpp`,
`dtwc/core/medoid_assignment_policy.hpp`, the `dist_by_ind` hunks of `dtwc/Problem.{hpp,cpp}`,
and the associated tests / CHANGELOG / LESSONS. Nothing was built or run; claims are labelled
Observed (read from the new tree) or Inferred.

## Ranked defects

### 1. HIGH — the silhouette throw breaks `cluster_and_process()` and Tier-1 `Result::save()`

**Where.** `dtwc/scores.cpp:112-118` (`require_two_realised`, `std::invalid_argument`);
unguarded callers `dtwc/Problem_IO.cpp:136` (`write_silhouettes`), `dtwc/Problem.cpp:1099`
(`cluster_and_process`), `dtwc/api.cpp:267` (`Result::save`).

**Scenario (Observed).** `dtwc/api.cpp:61` `validate_common` accepts `k >= 1`, so
`dtwc::cluster(ds, 1, ...)` is legal; `Result::save()` writes `_labels.csv`, `_medoids.csv`
and `_distance_matrix.csv` (`api.cpp:235-264`) and *afterwards* throws at `:267`. The user is
left with a partially populated output directory plus an exception. Same for
`Problem::cluster_and_process()`, a shipped example path (`examples/cpp/MIP_single.cpp:44`).

**Inferred, and worse than k=1.** The guard is on *realised* clusters, so a legitimate k>=2
request that collapses also fails: with duplicate or constant series two medoids can be at
d = 0, ties resolve to the lower slot (`d < best`), and one declared cluster ends up empty.
No k-medoids route in `Problem.cpp` has empty-cluster repair (Observed: none in that file;
`barycenter.cpp:713-724` has it, k-medoids does not). A run that succeeded yesterday on a
dataset with repeated series now aborts at save time.

`dtwc_cl.cpp:1948-1962` is already guarded (`n_clusters > 1` **and** a try/catch that warns) —
the pattern was known and not applied to the library paths. The source review
(`.claude/reports/2026-09-02-review-algorithms.md`) never mentions a silhouette caller, so no
caller audit was done.

**Minimal fix.** Keep the throw in `scores::silhouette()` and `Result::score("silhouette")`.
In `Problem::write_silhouettes()` and in `api.cpp Result::save()`, wrap the call in
`try { ... } catch (const std::exception &e) { std::cerr << "Warning: silhouettes skipped: " << e.what(); }`
and skip the file (or write a header-only one). Two try blocks, no API change.

### 2. MEDIUM — `one_batch_pam.total_cost` is not digit-identical to the previous release, undocumented

**Where.** `dtwc/algorithms/one_batch_pam.cpp:395` replaces
`std::accumulate(point_cost.begin(), point_cost.end(), 0.0)` with
`core::detail::ordered_medoid_objective`.

**Scenario (Inferred, high confidence).** `OrderedMedoidObjective` accumulates into a
`volatile double total_` (`dtwc/core/medoid_assignment_policy.hpp:125`) *precisely* to defeat
reassociation, and the build enables `-fassociative-math`
(`cmake/StandardProjectSettings.cmake:59-70`). The old `std::accumulate` was therefore
reassociable/vectorisable, so the published objective can differ in the last ulp. The
implementer's "digit-identical" claim covers only the C1 hoist (correctly — see below), but
the CHANGELOG entry mentions only the finiteness rejection, not the changed accumulator.
Second undocumented change: `add()` throws `InvalidInput` when the *running total* overflows
to inf even though every element is finite (`medoid_assignment_policy.hpp:113-118`); a legal
large-magnitude dataset that used to return `inf` now aborts.

**Minimal fix.** One CHANGELOG line ("one_batch_pam's `total_cost` is now accumulated in point
order; last-ulp differences from 2.0.x are expected, and an overflowing objective is now
rejected"), and regenerate any golden artifact pinning that number.

### 3. MEDIUM — TADPole Float32 now silently degrades; its rationale is stale inside its own diff

**Where.** `dtwc/algorithms/tadpole.cpp:70` (`!prob.data().is_f32()`), comment `:55-66`,
`.claude/LESSONS.md:123`.

The guard is correct and the admissibility argument (bounds from f64 storage vs exact
distances routed through `dist_by_ind`, which does branch on `is_f32()`) is sound. Two problems:

- The stated justification — `series()` "indexes the empty float64 storage (UB) … reproduced
  as a segfault" — describes the *old* tree. In the new tree `Data::series()` throws
  (`dtwc/Data.hpp:57`, the A12 change in this same diff). Post-A12 f32+prune would have been a
  loud error; it is now a silent 2-10x slowdown. Observed.
- No runtime signal. `TADPoleStats` is optional (`tadpole.hpp:62`, `stats = nullptr` by
  default) and `tadpole.hpp:85` documents only "where the live predicate permits". Only the
  CHANGELOG records it. This conflicts with the standing rule against silent capability
  fallbacks.

**Minimal fix.** Add "no pruning on Float32 data" to `@param prune` in `tadpole.hpp`, and a
`bool pruning_enabled` field to `TADPoleStats`.

Related, not a defect but fragile: `auto si = can_prune ? prob.series(i) : std::span<const data_t>{}`
(`tadpole.cpp:182`, `:186`, `:224`) leaves an empty-span placeholder live in the hot loop. It
is safe only because every use sits behind a short-circuiting `can_prune &&`; a future edit
that reads `si`/`sj` outside that guard gets a silent empty span instead of a throw.

### 4. LOW — vacuous / weak assertions in the new tests

- `tests/unit/algorithms/unit_test_one_batch_pam.cpp:665` — `REQUIRE(returned + threw == 64)`
  is tautological: each of the 64 iterations increments exactly one counter, and any other
  exception escapes the loop and fails the test regardless. (`REQUIRE(threw > 0)` on `:664`
  *is* the real non-vacuity check and is good.)
- `tests/unit/algorithms/unit_test_tadpole.cpp:363-394` — asserts `prune=true` equals
  `prune=false` on f32. Because the fix makes `can_prune` unconditionally false there, the two
  calls execute literally the same code; the test would still pass if `bounds_valid` returned
  false for *every* input. It pins the fix but proves nothing about admissibility. Fix: pass a
  `TADPoleStats`, assert `pruned_by_lb == 0 && pruned_by_ub == 0`, and add an f64 control
  asserting pruning actually happened.

### 5. LOW — score errors bypass the project taxonomy; one function throws two types

`scores.cpp:38` and `:47` throw `std::runtime_error`; `:62` throws `std::invalid_argument`.
So `silhouette()` reports "labels disagree with the Problem" and "fewer than 2 realised
clusters" as unrelated exception types, which any save-path `catch` must enumerate, while the
neighbouring layer uses `dtwc::InvalidInput` (`medoid_assignment_policy.hpp:33`). Cheap fix:
use `dtwc::InvalidInput` for both — it also makes the fix for defect 1 a one-type catch.

## Confirmed sound

1. **Silhouette (Rousseeuw 1987).** a(i) sums d over C_i including the zero self-term and
   divides by |C_i|-1 — correct; b(i) is the min over other *non-empty* clusters of the
   |C_j|-mean; s = (b-a)/max(a,b); singleton -> 0; a=b=0 -> 0. Hand check, 4 points,
   C0={0,1}, C1={2,3}, d(0,1)=2, d(0,2)=10, d(0,3)=12: a = 2/(2-1) = 2, b = 22/2 = 11,
   s = 9/11 = 0.8182 — exactly what `scores.cpp:143-152` computes.
2. **Davies-Bouldin (1979).** S_i divides by |C_i| (not |C_i|-1) — correct, and deliberately
   different from silhouette's divisor. DBI = (1/k_realised) sum_i max_{j != i} R_ij; M_ij = 0
   gives +inf iff S_i + S_j > 0 and 0 otherwise, the correct limit under the paper's
   monotonicity in M_ij plus axiom 3. Empty clusters are `continue`d on both i and j
   (`:215`, `:217`) so they neither contribute nor divide. Hand check of the new
   padded-vs-tight test: both give S = 0, M = 20, R = 0, DBI = 0/2 = 0 — the equality
   assertion is real, not accidental.
3. **Calinski-Harabasz (1974).** k = realised sets both (k-1) and (N-k) consistently; the
   `continue` at `:367` removes an empty cluster's |c|.d^2 term and its share of k together.
   No off-by-one; W runs over all points via their own cluster medoid, so no double counting.
4. **Dunn (1974).** min inter / max intra over `j = i+1` pairs — no double counting; the
   realised-cluster guard removes the previous DBL_MAX/max_intra ~ 1.8e308 result.
5. **`cut_dendrogram` component check cannot reject a genuine dendrogram.**
   `hierarchical.cpp:123` deactivates `best_b` at every step, so the N-1 recorded `cluster_b`
   values are pairwise distinct and every merge is effective; replaying any prefix of length
   N-k performs exactly N-k unions, leaving exactly k components. Verified for N=1 (0 merges,
   k=1), N=2, k=N (0 merges) and k=1 (all merges). Ties are irrelevant — ids are original
   point indices in [0,N), never SciPy-style >= N. The `N < 1` guard is correctly ordered
   *before* the `merges.size() != N-1` comparison, so `size_t(N-1)` never underflows.
6. **`refresh_swap_state()` — the invariant holds; digit-identity is credible.**
   `base_removal_gain` and `tolerance` depend only on `nearest`, `nearest_distance`,
   `second_distance`; those are written *only* by `nearest_two` (`one_batch_pam.cpp:287`,
   `:305`, `:356`), and the candidate loop reads but never writes them (Observed, `:331-360`).
   The j-accumulation order is unchanged, `removal_gain = base_removal_gain` is an exact
   same-size copy with no reallocation, and `estimated_cost` (`:230`) is pure. No new lock,
   atomic, allocation or `std::function` in the SWAP loop; the helper is a by-reference
   `const auto` lambda.
7. **`dist_by_ind` single preflight.** `preflight_current_distance_semantics()`
   (`Problem.cpp:427`) is `const` and reads only `variant_params` / `missing_strategy` /
   `data_` / `distance_strategy` / `cuda_settings`; the only call between the two former
   preflights was `validate_mmap_cache_identity()`, also `const`.
   `repair_dtw_binding_after_relocation()` still runs, so post-relocation rebinding is
   preserved, and every other caller of `ensure_dense_cache_configuration_current()` keeps its
   own preflight. No path loses a preflight.
8. **`fast_clara` template.** `F32` picks reader entry point, accessor and element size at
   compile time; per-chunk accumulation is the same `OrderedMedoidObjective`, the
   `if constexpr` branches only select messages/accessors, and
   `resident_data_bytes(..., F32 ? sizeof(float) : sizeof(data_t))` reproduces both originals.
   No branch changes rounding order or accumulation type — byte-identical artifacts credible.
9. **`medoid_utils` deletion.** The three removed helpers were function templates with no
   remaining reference anywhere (`fast_pam.cpp:95` defines its *own*
   `compute_nearest_and_second` with a different signature; all other hits are comments/docs).
   Never instantiated, so no code was emitted for them — byte-identical objects credible. The
   removed `#include <limits>` breaks nothing: `fast_pam.cpp:46` includes `<limits>` directly
   and is the only production consumer.
10. **`barycenter_kmeans` `!converged` guard.** On the converged break
    (`barycenter.cpp:693-698`) `result.labels` / `result.total_cost` come from the `assign()`
    at the top of the same iteration (`:682`), and `centers` is untouched before the break
    (the centre update follows it) — the removed unconditional re-assign was provably
    redundant. The `max_iter`-exhausted path and the `max_iter == 0` path both still call it.
11. **Four named `omp critical` regions** — `dtwc_one_batch_table_failure`,
    `dtwc_pam1_naive_swap_reduce`, `dtwc_fastpam1_swap_reduce`,
    `dtwc_fast_pam_single_medoid_reduce` — all distinct, and distinct from the pre-existing
    `tadpole_density_reduce` / `dtwc_medoid_candidate_failure`.
