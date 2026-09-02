# Read-only audit — clustering algorithm layer (2026-09-02)

Scope: `dtwc/algorithms/**`, `dtwc/scores.*`, `dtwc/initialisation.*`, `dtwc/enums/**`, plus the tests
that judge their coverage. Nothing was built or run. **Observed** = traced in source, certain;
**Inferred** = plausible mechanism, needs a test. Section G covers the added lock-free/hot-path
constraint.

---

## A. Bugs

### Critical

**A1 · TADPole ignores Float32 → UB read + inadmissible pruning.** `tadpole.cpp:59-64` —
`bounds_valid` tests variant, `ndim` and `missing_strategy` but not `is_f32()`. Under
`Precision::Float32`, `Data::series(i)` (`Data.hpp:52-57`) indexes the empty `p_vec` → UB at
`tadpole.cpp:156-159, 174, 176, 214, 219-220`. Even setting the UB aside, `exact()` routes through
`Problem::dist_by_ind`, which *does* branch on `is_f32()` (`Problem.cpp:718-720`), so LB/UB and the
exact distance would come from different data and the prune stops being admissible.
`barycenter.cpp:150` handles this correctly — an inconsistency, not an unbuilt mode. **Observed.**

**A2 · `cut_dendrogram` trusts a caller-supplied `Dendrogram`.** `hierarchical.cpp:181-187` — `N` comes
from `dend.n_points`, never compared with `prob.size()`, and `dend.merges` is never checked to hold
`N-1` entries. `Dendrogram` is default-constructible with both fields writable from Python
(`_dtwcpp_core.cpp:426-431`) and `cut_dendrogram` is exported (`:1300-1303`). So
`Dendrogram(n_points=10, merges=[])` + `cut_dendrogram(d, prob, 1)` reads `merges[0]` out of bounds;
unvalidated `cluster_a/b` then index `UF::parent` OOB. **Observed.**

### High

**A3 · `silhouette` returns ≈ +1.0 for a single realised cluster.** `scores.cpp:73, 79, 82` — `min` is
updated only in the `else if (i != i_c)` branch, so with one cluster it stays `DBL_MAX` and
`(DBL_MAX − a)/DBL_MAX ≈ 1`. Three distinct series with `clusters_ind = {0,0,0}` score "perfect".
sklearn raises; `davies_bouldin` (`:116`) and `dunn` (`:173`) were hardened, `silhouette` was not.
**Observed.**

**A4 · `davies_bouldin` drops zero-distance medoid pairs instead of treating `R_ij` as +∞.**
`scores.cpp:144` `if (d_ij > 0) {`. Two clusters with duplicate medoid series but real internal spread
— the worst possible configuration — skip every pair, `max_ratio` keeps its `0.0` initialiser (`:140`),
DB reports 0.0 = perfect. **Observed.**

**A5 · `silhouette` yields NaN when a = b = 0.** `scores.cpp:82`. Two clusters of mutually identical
series → `0.0/0.0`; convention is `s(i) = 0`. The guard at `:70` covers only `n_c == 1`. The NaN
poisons any downstream mean. **Observed.**

**A6 · `one_batch_pam`'s final assignment has no finite guard.** `one_batch_pam.cpp:355-365` —
`distances.exact()` (`:185-198`) is the only distance call in the layer that skips
`require_finite_medoid_distance`, and the objective uses `std::accumulate` (`:365`) rather than
`ordered_medoid_objective` (cf. `fast_pam.cpp:157`, `clarans.cpp:117`, `fast_clara.cpp:192`). A NaN
makes `d < best` false, so `label` stays `0`: every point lands silently in cluster 0 with
`total_cost = NaN`, where every other algorithm throws. **Observed.**

### Medium

**A7 · Scores validate the *declared* `n_clusters`, not the realised partition.** `scores.cpp:116, 173`.
With `Nc = 3` but labels using only id 0, `dunn` returns `DBL_MAX / max_intra` (~1.8e308) as a finite
number. Same class as A3. **Observed.**

**A8 · CLARANS' DTW budget counts cached lookups.** `clarans.cpp:107, 175` increment `dtw_evals` on
every `dist_by_ind`, while `:156` states those are cache hits. After the first pass `max_dtw_evals` is
a lookup budget, so the search stops far earlier than a user asking for a DTW budget intends.
**Observed.**

**A9 · CLARANS can return an empty result.** `opts.num_local` is unvalidated (contrast
`fast_clara.cpp:54-60`, `one_batch_pam.cpp:62-67`). With `num_local <= 0` the restart loop never runs,
`best_present` stays false, and `:249-251` writes empty `clusters_ind`/`centroids_ind` into `prob`
alongside `set_n_clusters(k)` — a silently inconsistent Problem for every downstream score.
`max_neighbor` and `max_dtw_evals` are likewise unvalidated. **Observed.**

**A10 · The CLARANS 64-bit-N fix is cosmetic.** `:49-50` widens `N` and asserts
`static_assert(sizeof(N) >= 8, ...)` — a tautology testing nothing — but every loop counter stays `int`
(`:100, 172, 207`), `labels` is `vector<int>`, and `:145` narrows back for the RNG. Beyond `INT_MAX`
this is signed overflow, not truncation. The comment claims the audit finding is closed; the severity
here is the false assurance, since >2³¹ is unreachable in practice. **Observed.**

**A11 · TADPole allocates the N×N matrix it documents avoiding.** `tadpole.cpp:131-132`
(`vector<char> computed(M)`) plus `:147` `exact(0,1)`, which triggers `m.resize(N)` → `packed_size(N)`
doubles (`Problem.cpp:696`). `tadpole.hpp:10-11, 81-82` promise the opposite; `computed` also
duplicates `DenseDistanceMatrix::count_computed()`. Related: `pruned_fraction()` over-reports when
`tadpole_auto_dc` (`:99-101`) pre-warmed the cache — those pairs are free later but never entered in
`computed`. **Observed.**

**A12 · Unseeded BUILD uses a process-global mutable engine.** `settings.hpp:46`, consumed at
`initialisation.cpp:140, 172, 184`, reached from `fast_pam.cpp:555`. See G4.

**A13 · `init_with_seed` silently degrades for a custom initialiser.** `Problem.cpp:1170-1184` recovers
the `void(*)(Problem&)` target and special-cases only `&init::random` / `&init::Kmeanspp`; anything
else — including a lambda wrapping `init::random`, as `test_tier1_cpp_api.cpp:205` does — falls through
to the unseeded `init()` with no diagnostic. **Observed.**

**A14 · `barycenter` accepts and ignores `random_seed`.** `barycenter.hpp:47`; honoured only by `ssg`
(`barycenter.cpp:289`). `dba` and `soft_barycenter` are deterministic, yet `barycenter_kmeans` still
derives per-cluster seeds for them (`:742-745`). **Observed.**

**A15 · `dist_by_ind` is not a flat lookup.** Every call runs `preflight_current_distance_semantics()`
**twice** (`Problem.cpp:675`, and again inside `ensure_dense_cache_configuration_current` at `:467`),
plus `validate_mmap_cache_identity()` and `repair_dtw_binding_after_relocation()`. The SWAP kernel
issues N² of these per iteration (`fast_pam.cpp:288`). Per LESSONS the SWAP is memory-bound; here each
element read carries several out-of-line validation calls. **Observed.** See also G2.

### Low — doc/claim defects that would seed wrong tests

- `fast_pam.hpp:41` "Digit-identical to `FastPAM1Naive`" is false and is contradicted by the test
  file's own comment (`unit_test_faster_pam.cpp:280-287`): `acc + ploss[m]` and the naive nested sum
  group the same terms in different FP order; only the objective is invariant.
- `fast_pam.hpp:44` "Never worse in objective than FastPAM1" has no proof — eager and best-first reach
  *different* local optima of the same neighbourhood — yet it is a hard `REQUIRE`
  (`unit_test_faster_pam.cpp:253, 440`). Latent flaky test. **Inferred.**
- `hierarchical.hpp:80` says the "**last** N-k merges"; the code replays the **first**
  (`hierarchical.cpp:179-187`). `medoid_utils.hpp:10` claims three callers it does not have (D1).

---

## B. Duplication

`medoid_assignment_policy.hpp:5-7` already records the intent: *"Full assignment scans deliberately
remain owned by their algorithms until the behavior-frozen R4 consolidation."* That consolidation is
still open and its target module is dead (D1).

| Logic | Copies |
| --- | --- |
| nearest + second-nearest | `fast_pam.cpp:95-150`, `medoid_utils.hpp:62-92`, `one_batch_pam.cpp:201-224` |
| assign-to-nearest | `medoid_utils.hpp:31-53`, `clarans.cpp:100-116` **and** `:207-223` (same loop twice in one function), `fast_clara.cpp:152-194`, `:229-301`, `:304-374`, `barycenter.cpp:551-584` |
| f32 / f64 chunked assignment | `fast_clara.cpp:229-301` vs `:304-374` — identical but for `series_f32`/`dtw_fn_f32`; one template |
| ΔTD / removal-loss decomposition | `fast_pam.cpp:256-304` vs `one_batch_pam.cpp:311-331`. Checked algebraically: `removal_gain[m] = −ploss[m]`, `add_gain = −acc`, `max_element` ⇔ `argmin`, ties both to the lowest slot. Equivalent — but derived twice, so a fix to one will not reach the other |
| per-cluster medoid scan | `hierarchical.cpp:212-227` vs `medoid_utils.hpp:102-121` (identical tie-break) |
| contingency table (ARI/NMI) | `scores.cpp:307-317` vs `:370-378` |

---

## C. Simplifications / error-prone constructs

1. `one_batch_pam.cpp:311-314` allocates `vector<double> removal_gain(k)` and recomputes it in O(m)
   *per candidate* — N heap allocations per sweep — but it changes only when a swap is accepted.
   `fast_pam.cpp:396-398` hoists exactly this. Same for the `estimated_cost(...)` tolerance at
   `:332-333`. Strictly less work.
2. `one_batch_pam.cpp:107` calls `prob.data().is_f32()` inside the innermost N×m loop; loop-invariant.
3. **`default:` in exhaustive scoped-enum switches defeats `-Wswitch`** (enabled at
   `cmake/CompilerWarnings.cmake:43`): `fast_pam.cpp:517`, `one_batch_pam.cpp:160, 178`,
   `Problem.cpp:1032, 1078`, `core/pruned_distance_matrix.cpp:116`, `metal_dtw.mm:1362`. A fifth
   `Method` would compile clean and throw at runtime. Move the throw *after* the switch.
4. Dead tie-break clauses: `hierarchical.cpp:78-79, 222-223` and `medoid_utils.hpp:115` add
   `(cost == best && i < best_i)` to scans already in ascending index order — the first minimum is
   already lex-smallest.
5. `fast_pam.cpp:550-560` saves/restores `prob`'s cluster state around `Kmeanspp`, then
   `fast_pam_swap` overwrites all three fields at `:532-534`. The restore is dead.
6. `fast_clara.cpp:255` — `rg += rg_per_batch` with `rg_per_batch` unchecked; a 0 from
   `row_groups_per_batch` on a tight budget is an infinite loop. **Inferred.**
7. `hierarchical.hpp:62` — bare magic `max_points = 2000` guarding an `N*N` double allocation
   (`hierarchical.cpp:49`); raising it to 100k silently requests 80 GB.
8. `barycenter.cpp:595` copies every series when `series_indices` names two; `:355, 371` allocate
   `nx * ny` with no overflow guard, bypassing the file's own `checked_matrix_cells` (`:36-43`).
9. `initialisation.cpp:50` `first_unselected` is O(N·k) per call, O(N·k²) over BUILD — and is the path
   taken exactly when the data is mostly duplicates.
10. `scores.cpp:394-402` sums MI over an `unordered_map`, whose iteration order is unspecified, so
    NMI's last ULPs differ across libstdc++/libc++/MSVC — a reproducibility hazard for a project with
    cross-language conformance tests. **Inferred.**

---

## D. Dead / obsolete code

**D1 · `medoid_utils.hpp` is dead apart from one function.** `assign_to_nearest` (`:31`),
`compute_nearest_and_second` (`:62`) and `find_cluster_medoid` (`:102`) have **no production caller** —
only `tests/unit/algorithms/unit_test_medoid_utils.cpp`. Only `validate_medoids` is used
(`fast_pam.cpp:431`). The header is nonetheless pulled into the public umbrella `dtwc.hpp:34` and its
`@details` claims three callers. Its 20 unit tests give false coverage confidence for logic that ships
in five hand-rolled copies (B). **Observed.**

**D2 · `test_scores_adversarial.cpp:451-526`** sits inside `#if 0` behind a stale comment, "DBI and CH
index not yet implemented" — both *are* implemented. ~5 adversarial cases dead.

**D3 · `initialisation.cpp:185-188, 207-210`** duplicate-centroid guards are unreachable:
`distance_sampling_weights` zeroes selected indices (`distance_sampling_weights.hpp:56-58`), and
neither `discrete_distribution` nor `portable_weighted_index` can return a zero-weight index. Copied
again into `fast_pam.cpp:601-604`.

**D4 · `unit_test_fast_clara.cpp:452`** — `#ifndef DTWC_HAS_PARQUET` around the SECTION *"the missing
capability is loud"*. Inverted guard: the check vanishes in every Parquet build. The F9-class trap
already in memory.

**D5 · Hidden `[.]` tests hold the only hard quality/prune floors**: `unit_test_faster_pam.cpp:385`
(`REQUIRE(fp1_identical)`, `REQUIRE(faster_never_worse)` at large N), `unit_test_one_batch_pam.cpp:576`
(the only ≤1.05× oracle band), `unit_test_tadpole.cpp:342` (`REQUIRE(frac >= 0.50)`). None run by
default. `cmake/Coverage.cmake:33` also sets `SKIP_RETURN_CODE 4`, so a Catch2 `SKIP()` reports green.

**D6 · `test_fast_pam_adversarial.cpp:52`** — the helper sets `Method::Kmedoids` and calls
`prob.cluster()`. All 22 cases exercise the legacy Lloyd path, not `dtwc::fast_pam*`, despite the
filename. Do not count them as FastPAM coverage.

**D7 · `barycenter.cpp:758-759`** runs a full N·k `assign()` unconditionally after the loop, including
on the converged-`break` path (`:693-698`) that already stored the identical cost.

---

## E. Missing / forgotten

**E1 · `Method` has no value for the algorithms in this layer.** `Method` is
`{Kmedoids, MIP, LRCore, TADPole}` (`Method.hpp:15-25`); FastPAM, FastCLARA, CLARANS, OneBatchPAM and
hierarchical are reachable only as free functions plus the CLI's string dispatch. So `Problem::cluster()`
can never reach FastPAM (hence D6), and the MATLAB binding accepts only 2 of the 4 values it *does*
have — `dtwc_mex.cpp:415`: *"Valid: 'kmedoids', 'mip'"* — leaving `LRCore` and `TADPole` unreachable
from MATLAB.

**E2 · Silent string→enum fallbacks.** `api.cpp:351-354`: any unrecognised method string becomes
`Kmedoids`. `dtwc_cl.cpp:1758-1763`: an unrecognised `linkage` becomes `Average`. The CLI flag is
guarded by `CheckedTransformer` (`:832-836`), but the TOML path writes the raw string via
`set_if_unset` (`:943-946`), bypassing it — and `:1173` then prints the *requested* linkage while
running Average. Known-but-open (comment at `:942`).

**E3 · Enums with no wiring.** `KernelOverride` — all 5 values — has no parser, no CLI/TOML key and no
`nb::enum_` export, so its documented benchmarking purpose (`KernelOverride.hpp:8`) is unreachable
outside C++; Metal silently ignores an unsatisfiable override (`metal_dtw.mm:1351, 1356`) where CUDA
sets `fell_back_to_auto` (`cuda/kernel_selection.hpp:29`). `LowerBoundStrategy` has no CLI/TOML key at
all. No enum→string direction exists for any of the four enums, so the five parse tables
(`dtwc_cl.cpp:750`, `api.cpp:74`, `_api.py:246`, `_hpc.py:37`, `dtwc_mex.cpp:413-426`) are untestable
for round-trip — and already disagree (E1).

**E4 · `Solver::Gurobi` silently downgrades to HiGHS** when `DTWC_ENABLE_GUROBI` is off:
`Problem.cpp:207-215` prints and returns `false`; `dtwc_cl.cpp:1548` discards the return. Exit code 0.

**E5 · Coverage gaps** (Observed): no algorithm-level test that an emptied cluster is repaired in
fast_pam / clarans / fast_clara / one_batch_pam (only the unit-level `find_cluster_medoid` returns −1,
`unit_test_medoid_utils.cpp:189`); no `k > N` rejection for tadpole, hierarchical or one_batch_pam; no
`k == N` for fast_clara; Complete and Average linkage tested only through dendrogram distances, never
through `cut_dendrogram` (`unit_test_hierarchical.cpp:84, 106`); none of A3–A7 has a test.

---

## F. Top 5 actions

Perf constraint applied: none of the five adds a lock, a hot-loop allocation, or indirect dispatch;
4 removes some. Only 5 carries real risk — see its note.

| # | Action | Size |
| --- | --- | --- |
| 1 | **A1** — add `!prob.data().is_f32()` to `bounds_valid` (`tadpole.cpp:59-64`), mirroring `barycenter.cpp:150`, with a Float32 regression test. Removes UB and restores prune admissibility. | **S** |
| 2 | **A2** — validate `dend.n_points == prob.size()`, `dend.merges.size() == n_points - 1`, and each `cluster_a/b`, at the top of `cut_dendrogram`. Python-reachable OOB today. | **S** |
| 3 | **A3–A7** — fix the degenerate score cases: silhouette on one realised cluster (throw, as DB/dunn do) and on `a = b = 0` (return 0); DB `M_ij == 0` → +∞ and skip empty clusters; move all three from declared `n_clusters` to the realised label set. Then delete the `#if 0` at `test_scores_adversarial.cpp:451` and add the five missing degenerate tests. | **M** |
| 4 | **A6 + C1 + G1** — route `one_batch_pam`'s final assignment through `require_finite_medoid_distance` and `ordered_medoid_objective`; hoist `removal_gain`/`tolerance` out of the candidate loop (`:311, 332`); name the four unnamed `omp critical` regions. *Perf: strictly negative cost — removes N allocations per sweep and decouples a global lock.* | **S** |
| 5 | **B + D1** — land the deferred R4 consolidation: either route the four algorithms through `medoid_utils`' helpers (adding the OpenMP + finite-check policy there once) or delete the three dead helpers, their 20 tests and the false `@details` claim. Do **not** leave five copies with one tested-but-unused reference. Template the `fast_clara` f32/f64 chunk pair while there. **⚠ The one perf-risky action.** Three hard constraints, else just delete: (a) helpers stay **templated on `DistFn`** so the call site inlines the lambda — routing them through `std::function` would turn every N·k/N² distance read into an indirect call, i.e. G3 made universal; (b) `fast_pam`'s parallel `compute_nearest_and_second` must not regress to `medoid_utils`' serial version; (c) per-thread scratch stays hoisted per thread per iteration (`fast_pam.cpp:197, 332, 398`), never per candidate. | **M** |

Runner-ups (need a benchmark to size): **A15 + G2** — hoist a raw accessor to the filled dense matrix
for the duration of a SWAP; **G5** — use the existing `core::portable_sample_indices` in
`clarans.cpp:87-91` and `one_batch_pam.cpp:268-275`.

**Rejected on the perf constraint:** folding the five assignment loops behind a runtime-polymorphic
distance interface, or behind `dist_by_ind` itself — either adds indirection or the G2 per-element
validation to loops that are currently direct reads; and replacing the FastPAM per-thread reduction
(`fast_pam.cpp:233-240`) with an atomic or lock-based tracker — that pattern is already right, it only
needs a name.

---

## G. Lock-free / hot-path audit (added constraint)

Every lock, critical section, atomic and shared-mutable read on the SWAP, assignment and sampling
paths. Note `clarans.cpp` has no OpenMP at all — its O(N) swap evaluation (`:172-197`) and both
assignment loops (`:100`, `:207`) are serial while the equivalent FastPAM/CLARA loops are parallel.

**G1 · Unnamed `#pragma omp critical` — four sites, one global lock.** `fast_pam.cpp:233, 357, 488` and
`one_batch_pam.cpp:125`. In OpenMP an *unnamed* critical region shares a single unspecified name across
the whole program, so these four — and any unnamed critical in any other TU linked in — serialize
against one another. The FastPAM ones are the per-thread SWAP reduction, entered once per thread per
iteration, so measured cost is small; the defect is unbounded coupling to code this layer does not own.
The neighbouring failure-path criticals *are* named (`dtwc_medoid_candidate_failure`,
`dtwc_medoid_assignment_failure`, `tadpole_density_reduce`), so this is an oversight, not a design.
Fix: give each of the four a distinct name — S, zero perf risk. **Observed.**

**G2 · `dist_by_ind` reads shared mutable state on every hot-loop element.** Beyond A15's double
preflight, the `needs_init` probe (`Problem.cpp:686`) is an unsynchronized read of `distMat`, and the
function's own doc (`:666-671`) states the lazy path is *not* thread-safe and needs a serial priming
call. The SWAP kernel issues N² of these per iteration. Largest hot-path cost in the layer.
**Observed.**

**G3 · `dtw_fn_` is a `std::function` *and* a `mutable` member.** `Problem.hpp:118-119, 125-126`. Every
distance in the assignment paths is a type-erased indirect call — N·k in `fast_clara.cpp:174`, N·m in
`one_batch_pam.cpp:108/113`. Worse, the getter can rebind the member
(`repair_dtw_binding_after_relocation`), which is exactly why `one_batch_pam.cpp:91-95` resolves both
getters serially *before* its OpenMP region. But `exact()` (`:192-197`) calls the rebinding getter
**inside** the final assignment loop (`:355-364`), N·k times. That loop is serial by design
(`:352-354`), so no race today — a latent one the moment anybody parallelizes it, plus needless
indirection. **Observed.**

**G4 · Process-global mutable RNG in the sampling path.** `settings.hpp:46`
`inline std::mt19937 randGenerator(29);`, consumed at `initialisation.cpp:140, 172, 184`, reached from
`fast_pam.cpp:555`. Unsynchronized shared mutable state: two concurrent unseeded `fast_pam()` calls are
a data race, and any BUILD result depends on prior consumption elsewhere in the process. The `*_seeded`
twins are clean (invocation-local `std::mt19937_64` + `core::portable_*`). Fix: have the unseeded entry
points derive an invocation-local engine, as `fast_pam_seeded` already does — S. **Observed.**

**G5 · Heap allocation inside hot loops**, by call frequency: `one_batch_pam.cpp:311` — one allocation
*per candidate*, N per sweep (C1); `scores.cpp:62` — one `vector<pair<int,double>>` per point inside
the parallel silhouette task; `barycenter.cpp:300-301` — inside the innermost SSG loop, per series per
epoch; `clarans.cpp:87-91` and `one_batch_pam.cpp:268-275` — an O(N) `iota` + full `portable_shuffle`
of an N-vector merely to draw k (resp. m and k) indices, when `core::portable_sample_indices`
(`portable_random.hpp:130`) does this in O(sample-size) scratch and is already used by
`fast_clara.cpp:402, 555`. By contrast `fast_pam.cpp:197, 332, 398` allocate scratch **once per thread
per iteration** and reuse it across candidates — the pattern to copy. **Observed.**

**G6 · `static std::mutex` in a library.** `one_batch_pam.cpp:41-42`, in `warn_batch_size_adjustment`.
Off the hot path (at most one call per invocation, before the table build) and the serialization is
deliberate, so acceptable — but it is a process-wide lock plus a `std::cerr` write inside a core
algorithm. **Observed, low.**

**Correct as-is, do not "fix":** `fast_pam.cpp:279-284` documents `find_best_swap` as deliberately
sequential with a *measured* justification (~10× slower parallel at N=1000 — the O(N) body cannot
amortise fork/join plus a k-wide reduction per candidate). `compute_nearest_and_second`
(`fast_pam.cpp:107-108`) is genuinely lock-free: each iteration writes only index `p`.

---

*Note:* the brief stated `.claude/reports/` is gitignored. It is not — the rule at `.gitignore:87` is
commented out (`# .claude/reports/`), and `git check-ignore` confirms this file is tracked-eligible.
