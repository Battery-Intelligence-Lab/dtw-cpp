# AS-IS map — `dtwc/algorithms/` + `dtwc/mip/` (+ `initialisation.*`)

Repository `C:\D\git\dtw-cpp`, branch `Claude`, HEAD `a31956e`. Read-only pass; every line of
the 20 files in scope was read. Nothing was built or run — no claim below rests on execution.

Evidence tags: `[confirmed path:line]` = read in the source at HEAD. `[inferred]` = concluded
from stated evidence, with the confirming test named. `[not established]` = unknown.

Prior-review status uses **fixed / open / changed** against
`.claude/reports/2026-09-02-review-algorithms.md` (A–G) and
`.claude/reports/2026-09-02-review-backends-bindings.md` (A1–A9, B1–B3). Two commits did most of
the fixing: `0af4cbd` "fix(algorithms): TADPole f32 guard, dendrogram validation, degenerate
scores" and `84fc5e5` "fix(mip,matlab): valid Benders bounds and cuts, certified LR-core"
[confirmed `git log --since=2026-09-01 -- dtwc/algorithms/ dtwc/mip/`].

---

## 1. Module map

`LoC` from `wc -l`. "Needs from `Problem`" lists the exact members with the line of first use.

### 1a. `dtwc/algorithms/`

| File | LoC | Responsibility | Public entry points | Needs from `Problem` | Other includes |
|---|---|---|---|---|---|
| `fast_pam.hpp` | 116 | FastPAM1/FasterPAM contract + `PAMVariant` | `fast_pam`, `fast_pam_seeded`, `fast_pam_swap`, `PAMVariant`, `validate_pam_variant` | fwd-decl only | `core/clustering_result.hpp`, `error.hpp` |
| `fast_pam.cpp` | 616 | 3 SWAP variants + k=1 special case + BUILD | (impl) + `detail::checked_fast_pam_point_count`, `detail::resolve_fast_pam_plan` | `size()` :434; `fill_distance_matrix()` :437; `dist_by_ind()` :120,208,292,474,591; `n_clusters()` :555; `centroids_ind` :538,556,561,564; `clusters_ind` :539,557,565; `set_n_clusters()` :537,559,563 | `detail/fast_pam_plan.hpp`, `detail/medoid_utils.hpp`, `Problem.hpp`, `core/medoid_assignment_policy.hpp`, `core/portable_random.hpp`, `core/distance_sampling_weights.hpp`, `initialisation.hpp`, `parallelisation.hpp` |
| `fast_clara.hpp` | 72 | CLARA options | `fast_clara`, `CLARAOptions` | fwd-decl only | `core/clustering_result.hpp`, `settings.hpp` |
| `fast_clara.cpp` | 575 | Subsample → FastPAM → full assign; Parquet-streaming variant | (impl) + `detail::validate_clara_controls`, `resolve_clara_plan`, `validate_streaming_clara_plan` | `size()` :201; `data()` (`is_f32`,`series_f32`,`ndim`) :205,208,527,529; `series()` :212,534; `series_name()` :513; `dtw_function()`/`_f32()` :206,210,397,401; `band` :368,517; `variant_params` :369,518; `missing_strategy` :370,519; `distance_strategy` :371,520; `verbose()` :472,521; `set_verbose()` :372; `set_data()` :373; `set_view_data()` :528,535; `set_n_clusters()` :419,568; `centroids_ind` :420,569; `clusters_ind` :421,570; ctor `Problem(name)` :367,515 | `detail/fast_clara_plan.hpp`, `fast_pam.hpp`, `Problem.hpp`, `core/medoid_assignment_policy.hpp`, `core/portable_random.hpp`, `error.hpp`, `io/parquet_chunk_reader.hpp` (guarded) |
| `one_batch_pam.hpp` | 73 | OneBatchPAM options/stats/weighting | `one_batch_pam`, `OneBatchPAMOptions`, `OneBatchPAMStats`, `OneBatchWeighting`, `validate_one_batch_weighting` | fwd-decl only | `core/clustering_result.hpp`, `error.hpp`, `settings.hpp` |
| `one_batch_pam.cpp` | 413 | Fixed N×m batch table, eager swap on the estimate, exact final assignment | (impl) | `size()` :85,241; `data()` :108,110,195,197; `series()` :114,199; `dtw_function()`/`_f32()` :95,96,196,199; `set_n_clusters()` :252,396; `centroids_ind` :254,397; `clusters_ind` :253,398 | `Problem.hpp`, `core/medoid_assignment_policy.hpp`, `core/portable_random.hpp`, `error.hpp`, `<mutex>`, `<iostream>` |
| `clarans.hpp` | 66 | CLARANS options; **`@warning Experimental — not exposed in CLI`** :15 | `clarans`, `CLARANSOptions` | fwd-decl only | `core/clustering_result.hpp`, `settings.hpp` |
| `clarans.cpp` | 256 | Randomized neighbourhood search, `num_local` restarts | (impl) | `size()` :49; `dist_by_ind()` :106,174,185,213; `set_n_clusters()` :249; `centroids_ind` :250; `clusters_ind` :251 | `Problem.hpp`, `core/medoid_assignment_policy.hpp`, `core/portable_random.hpp` |
| `hierarchical.hpp` | 102 | `Linkage`, `DendrogramStep`, `Dendrogram`, `HierarchicalOptions` | `build_dendrogram`, `cut_dendrogram`, `validate_linkage` | fwd-decl only | `core/clustering_result.hpp`, `error.hpp` |
| `hierarchical.cpp` | 286 | O(N³) agglomerative + Lance-Williams; union-find cut | (impl) | `size()` :32,178; `is_distance_matrix_filled()` :39; `dist_by_ind()` :52,252,265; `set_n_clusters()` :279; `centroids_ind` :280; `clusters_ind` :281 | `Problem.hpp` only |
| `tadpole.hpp` | 125 | TADPole contract + `TADPoleStats` pruning ledger | `tadpole`, `tadpole_auto_dc`, `TADPoleStats` | fwd-decl only | `core/clustering_result.hpp` |
| `tadpole.cpp` | 337 | Density peaks with LB_Keogh/diagonal-UB pruning | (impl) | `size()` :96,127; `data()` (`ndim`,`is_f32`) :68,69; `variant_params` :67; `missing_strategy` :70; `band` :135; `series()` :164,185,187,232,235; `dist_by_ind()` :108,147; `set_n_clusters()` :330; `centroids_ind` :331; `clusters_ind` :332 | `Problem.hpp`, `core/lower_bound_impl.hpp`, `core/distance_matrix.hpp`, `core/dtw_options.hpp` |
| `barycenter.hpp` | 109 | `BarycenterMethod`, `BarycenterOptions`, `BarycenterClusteringOptions`, `BarycenterClusteringResult`, `detail::SoftDtwValueGradient` | `dtw_barycenter`, `barycenter_kmeans`, `detail::soft_dtw_squared_value_gradient`, `validate_barycenter_method` | fwd-decl only | `core/clustering_result.hpp`, `error.hpp`, `settings.hpp` |
| `barycenter.cpp` | 771 | Own squared-DTW DP + DBA/SSG/soft-DTW + sequence-centroid k-means | (impl) | **`const Problem&` only**: `variant_params` :133; `band` :137; `data()` :145,150,151; `size()` :148,149; `series()` :154. **No write-back.** | `Problem.hpp`, `core/portable_random.hpp`, `error.hpp`, `parallelisation.hpp`, `<omp.h>` |
| `detail/fast_pam_plan.hpp` | 27 | Checked int-indexed dimensions for FastPAM | `FastPamPlan`, `checked_fast_pam_point_count`, `resolve_fast_pam_plan` | — | `<cstddef>`, `<string_view>` |
| `detail/fast_clara_plan.hpp` | 34 | Allocation-free CLARA dimension planning | `ClaraPlan`, `validate_clara_controls`, `resolve_clara_plan`, `validate_streaming_clara_plan` | — | `../fast_clara.hpp` |
| `detail/medoid_utils.hpp` | 44 | **Only** `validate_medoids` (rest deleted, see §7) | `validate_medoids` | — | `<algorithm> <stdexcept> <string> <vector>` |

### 1b. `dtwc/mip/`

| File | LoC | Responsibility | Public entry points | Needs from `Problem` | Other includes |
|---|---|---|---|---|---|
| `mip.hpp` | 28 | The four MIP entry points + capability query | `MIP_clustering_byGurobi/byHiGHS/byBenders`, `LR_core_clustering`, `highs_solver_available` | fwd-decl only | — |
| `mip_Highs.cpp` | 241 | Compact Balinski N²-var MIP on HiGHS, **FacilityMajor** | `MIP_clustering_byHiGHS`, `highs_solver_available` | `data().size()` :52; `n_clusters()` :53; `mip_settings` :47,169-174,182; `verbose()` :47; `fill_distance_matrix()` :81; `max_distance()` :82; `dist_by_ind()` :86; `random_seed()` :183 | `highs_support.hpp`, `index_guard.hpp`, `solution_transaction.hpp`, `warm_start.hpp`, `Data.hpp`, `error.hpp`, `types/types.hpp`, `Problem.hpp`, `settings.hpp`, `timing.hpp` |
| `mip_Gurobi.cpp` | 165 | Same model on Gurobi, **PointMajor**, branch priority on the diagonal | `MIP_clustering_byGurobi` | `size()` :39; `n_clusters()` :40; `fill_distance_matrix()` :82; `max_distance()` :83; `dist_by_ind()` :88; `mip_settings` :93-102; `verbose()` :118; `random_seed()` :103 | `index_guard.hpp`, `solution_transaction.hpp`, `warm_start.hpp`, `Problem.hpp`, `error.hpp`, `settings.hpp`, `types/types.hpp` |
| `benders.hpp` | 91 | Scaled tolerances + the dual-bound accessor as inline free functions | `benders_abs_eps`, `benders_cut_coefficient_threshold`, `benders_master_lower_bound<>`, `MIP_clustering_byBenders` | fwd-decl only | `<algorithm>` |
| `benders.cpp` | 442 | Disaggregated Benders cut loop over an N-binary master | `MIP_clustering_byBenders` | `size()` :83; `n_clusters()` :84; `verbose()` :82; `mip_settings` :82,130,225-233,266,328; `fill_distance_matrix()` :124; `max_distance()` :269; `dist_by_ind()` :108,260,311,373,427; `centroids_ind` :93,114,138,150,157; `clusters_ind` :95,115,139,151; **private `method_`** :135,147,153; **private `last_iterations_`** :137,149; `n_repetitions()`/`set_n_repetitions()` :136,148,154; **private `cluster_by_kmedoids_lloyd_impl()`** :155; `find_total_cost()` :158 | `benders.hpp`, `mip.hpp`, `highs_support.hpp`, `nearest_medoid.hpp`, `solution_transaction.hpp`, `core/clustering_result.hpp`, `Problem.hpp`, `error.hpp`, `settings.hpp`, `timing.hpp` |
| `lagrangian_root.hpp` | 151 | `LagrangianParams` (10 knobs), `LagrangianResult` | `lagrangian_root(D…)`, `lagrangian_root_kelley`, `lagrangian_root_exact`, `lagrangian_root(Problem&)` | fwd-decl only | `<cstdint> <vector>` |
| `lagrangian_root.cpp` | 756 | Subgradient + Kelley duals, primal repair, B&B, `LR_core_clustering` | (impl) | `size()` :668; `n_clusters()` :669; `is_distance_matrix_filled()` :672; `fill_distance_matrix()` :672; `dist_by_ind()` :684,693; `centroid_of()` :684; `centroids_ind` :681; `clusters_ind` :682; `cluster_by_kmedoids_lloyd()` :680; `mip_settings` :712,721,733 | `nearest_medoid.hpp`, `reduced_cost_fixing.hpp`, `solution_transaction.hpp`, `core/clustering_result.hpp`, `error.hpp`, `parallelisation.hpp`, `Problem.hpp` |
| `pdlp_lp.hpp` | 89 | `PdlpParams`, `PdlpResult` | `pdlp_lp_bound`, `pdlp_gpu_available` | — (raw `const double*` only) | `<string>` |
| `pdlp_lp.cpp` | 204 | Explicit p-median LP relaxation for the PDLP arbiter | (impl) | none | `highs_support.hpp`, `index_guard.hpp`, `error.hpp` |
| `reduced_cost_fixing.hpp` | 65 | `FixingResult`, Beasley bound derivation | `reduced_cost_fixing` | — | `<vector>` |
| `reduced_cost_fixing.cpp` | 93 | Beasley conditional-bound fixing with an index tie-break | (impl) | none | `error.hpp` |
| `nearest_medoid.hpp` | 54 | **SSOT** nearest-open-medoid scan; header-inline template | `NearestMedoid`, `nearest_medoid<Dist>` | — | `<limits>` |
| `solution_transaction.hpp/.cpp` | 70/173 | `AssignmentMatrixLayout`, decode+validate, transactional publish | `extract_exact_clustering`, `ExactClusteringTransaction`, `validate_assignment_matrix_layout` | `size()` :43; `n_clusters()` :44; `centroids_ind` :150,157,168; `clusters_ind` :151,158,169 | `core/clustering_result.hpp`, `error.hpp`, `Problem.hpp` |
| `warm_start.hpp/.cpp` | 22/49 | Invocation-local FastPAM incumbent + input validation | `make_warm_start` | `size()` :17; `n_clusters()` :18 (+ everything `fast_pam_seeded` needs) | `solution_transaction.hpp`, `Problem.hpp`, `algorithms/fast_pam.hpp`, `error.hpp`, `settings.hpp` |
| `highs_support.hpp` | 58 | `setOptionValue` status guards | `set_highs_option`, `set_highs_option_best_effort` | — | `error.hpp`, `<Highs.h>` (guarded) |
| `index_guard.hpp` | 42 | Backend-neutral N² index-range guard | `require_index_range` | — | `error.hpp` |
| `CMakeLists.txt` | 74 | `mip-solvers` OBJECT lib; guarded llfio / Gurobi / HiGHS / `DTWC_HIGHS_GPU` | — | — | — |

`mip/CMakeLists.txt` notes: `mip-solvers` is an OBJECT library with `POSITION_INDEPENDENT_CODE ON`
:2-3. The llfio link is guarded on `TARGET llfio_hl` :55-58 with a 20-line comment explaining that
`mip-solvers` needs llfio **transitively only** — every mip source includes `Problem.hpp`, whose
`distMat_t` variant member embeds `MmapDistanceMatrix`, so `DTWC_HAS_MMAP` must match `dtwc++`'s or
`Problem`'s layout differs across TUs (an ODR violation) :42-49 [confirmed]. Gurobi :60-64 and
HiGHS :66-73 are likewise `TARGET`-guarded, and `DTWC_HIGHS_GPU` is a separate opt-in :71-73. This
file satisfies the project's optional-dependency rule cleanly and is the best-documented CMake unit
in the scope.

### 1c. `dtwc/initialisation.*` (only where algorithms depend on it)

| File | LoC | Responsibility | Entry points | Needs from `Problem` |
|---|---|---|---|---|
| `initialisation.hpp` | 26 | 4 initialiser declarations | `init::random`, `init::Kmeanspp`, `init::random_seeded`, `init::Kmeanspp_seeded` | fwd-decl only |
| `initialisation.cpp` | 217 | Shuffle-based and k-means++ BUILD, templated on the RNG hook | (impl) | `n_clusters()` :60,80; `size()` :64,67,84,94,101,112; `centroids_ind` :87; `set_clusters()` :72,119; `is_distance_matrix_filled()` :101; `dist_by_ind()` :104,108 |

`fast_pam` is the only algorithm in scope that calls into this layer, at `fast_pam.cpp:560`
(`init::Kmeanspp(prob)`) [confirmed]. `fast_pam_seeded` deliberately does **not** — it inlines its
own k-median++ D-sampling at `fast_pam.cpp:582-612`, with a 5-line comment explaining why the
weights are **not** squared (PAM minimises a sum of distances, so the sampling weight is the raw
nearest contribution; squaring would bias a different objective) `:593-597` [confirmed].

---

## 2. The `Problem` dependency — one consolidated table

**33 distinct members** are read or written by this scope: 30 public, 3 private reached through
`friend void MIP_clustering_byBenders(Problem &prob);` (`Problem.hpp:256`) [confirmed].

| `Problem` member | Kind | Readers | Writers | Why |
|---|---|---|---|---|
| `size()` | read | fast_pam :434,547,578; fast_clara :201,204,440,459,482,486; one_batch_pam :85,241; clarans :49; hierarchical :32,178; tadpole :96,127; barycenter :148,149; benders :83; mip_Gurobi :39; warm_start :17; solution_transaction :43; lagrangian_root :668; initialisation :64,67,84,94,101,112 | — | N; every dimension plan |
| `data()` | read | fast_clara :205,208,523,527,529,536; one_batch_pam :108,110,112,195,197,198; tadpole :68,69; barycenter :145,150,151; mip_Highs :52 | — | `is_f32()`, `ndim`, `series_f32()`, `size()` |
| `series(i)` | read | fast_clara :212,534; one_batch_pam :114,115,199,200; tadpole :164,185,187,232,235; barycenter :154 | — | zero-copy f64 span for the direct-DTW routes |
| `series_name(i)` | read | fast_clara :513 | — | build the sub-`Problem` name view |
| `centroid_of(i)` | read | lagrangian_root :684 | — | seed-UB cost |
| `dist_by_ind(i,j)` | read (**mutating**) | fast_pam :120,208,292,474,591; clarans :106,174,185,213; hierarchical :52,252,265; tadpole :108,147; benders :108,260,311,373,427; mip_Highs :86; mip_Gurobi :88; lagrangian_root :684,693; initialisation :104,108 | mutates `distMat` lazily (`Problem.cpp:694-728`) | **the** distance oracle; N² per SWAP iteration |
| `max_distance()` | read | benders :269; mip_Highs :82; mip_Gurobi :83 | — | objective / tolerance scaling |
| `fill_distance_matrix()` | call | fast_pam :437,549,580; benders :102,124; mip_Highs :81; mip_Gurobi :82; lagrangian_root :672 | fills `distMat` | prime the cache before parallel lookup |
| `is_distance_matrix_filled()` | read | hierarchical :39; lagrangian_root :672; initialisation :101 | may set `mmap_cache_data_validated_` | precondition check |
| `dtw_function()` | read | fast_clara :210,401; one_batch_pam :96,199 | non-const overload may rebind `dtw_fn_` | direct f64 DTW without the N² cache |
| `dtw_function_f32()` | read | fast_clara :206,397; one_batch_pam :95,196 | ditto for `dtw_fn_f32_` | direct f32 DTW |
| `band` (public field) | read | fast_clara :368,517; tadpole :135; barycenter :137 | — | envelope radius; barycenter rejects `!= -1` |
| `variant_params` (public field) | read | fast_clara :369,518; tadpole :67; barycenter :133 | — | prune / barycenter admissibility predicate |
| `missing_strategy` (public field) | read | fast_clara :370,519; tadpole :70 | — | prune admissibility |
| `distance_strategy` (public field) | read | fast_clara :371,520 | — | copied into the sub-`Problem` |
| `mip_settings` (public field) | read | benders :82,130,225-233,266,328,413; mip_Highs :47,169-174,182; mip_Gurobi :93-102; lagrangian_root :712,721,733 | — | solver tuning; see §6 D4 |
| `verbose()` | read | fast_clara :472,521; benders :82; mip_Highs :47,216; mip_Gurobi :118 | — | progress output gate |
| `random_seed()` | read | mip_Highs :183; mip_Gurobi :103 | — | warm-start determinism |
| `n_clusters()` | read | fast_pam :555; benders :84; mip_Highs :53; mip_Gurobi :40; warm_start :18; solution_transaction :44; lagrangian_root :669; initialisation :60,80 | — | k |
| `n_repetitions()` | read | benders :136 | — | saved for the ScopeExit restore |
| `set_n_clusters()` | — | fast_pam :537,559,563; fast_clara :419,568; one_batch_pam :252,396; clarans :249; hierarchical :279; tadpole :330 | writes `Nc` | result write-back (Task 1.6) |
| `set_n_repetitions()` | — | benders :148,154 | writes `N_repetition` | force 1 rep for the nested Lloyd |
| `set_verbose()` | — | fast_clara :372 | writes `verbose_` | silence the sub-`Problem` |
| `set_view_data()` | — | fast_clara :528,535 | writes `data_`, resizes `distMat` | zero-copy subsample |
| `set_data()` | — | fast_clara :373 | writes `data_`, refreshes matrix | Parquet subsample (owning) |
| `set_clusters()` | — | initialisation :72,119 | writes `centroids_ind` (+ resize) | publish BUILD medoids |
| `centroids_ind` (public field) | read+write | read: fast_pam :556,561; benders :138,157; lagrangian_root :681; solution_transaction :150 | write: fast_pam :538,564; fast_clara :420,569; one_batch_pam :254,397; clarans :250; hierarchical :280; tadpole :331; benders :93,114,150; solution_transaction :157,168; initialisation :87 | medoid point indices |
| `clusters_ind` (public field) | read+write | read: fast_pam :557; benders :139; lagrangian_root :682; solution_transaction :151 | write: fast_pam :539,565; fast_clara :421,570; one_batch_pam :253,398; clarans :251; hierarchical :281; tadpole :332; benders :95,115,151; solution_transaction :158,169 | labels in `[0,k)` |
| `find_total_cost()` | call | benders :158 | reads labels/medoids, may fill `distMat` | warm-start UB |
| `cluster_by_kmedoids_lloyd()` | call | lagrangian_root :680 | writes `centroids_ind`, `clusters_ind`, `last_iterations_` | seed UB |
| `method_` (**private**) | read+write | benders :135 | benders :147,153 | force `Kmedoids` for the nested run |
| `last_iterations_` (**private**) | read+write | benders :137 | benders :149 | restore after the nested run |
| `cluster_by_kmedoids_lloyd_impl(bool)` (**private**) | call | benders :155 | same as above, artifacts suppressed | run Lloyd without artifact files |

Coupling **not** captured by a member name: `fast_clara.cpp:367,515` constructs a whole `Problem`
(`Problem sub_prob("clara_subsample_" + …)`), so the algorithm layer depends on `Problem`'s
constructor, seven of its setters and its move semantics, not merely on a read interface
[confirmed]. `barycenter_kmeans` / `dtw_barycenter` take `const Problem&` and copy every series out
at `barycenter.cpp:143-160`, using `Problem` purely as a data + configuration bag [confirmed].
`mip::make_warm_start` and `mip::prepare_dense_D` are the only places where a `Problem&` is passed
*through* the MIP layer into an algorithm.

---

## 3. Algorithm anatomy

### 3.1 `fast_pam` / `fast_pam_seeded` / `fast_pam_swap`

**Oracle.** Schubert & Rousseeuw (2021), *Information Systems* 101:101804, arXiv:2008.05171
[confirmed `fast_pam.cpp:5-9`; `.claude/CITATIONS.md:90`]. The ΔTD decomposition is derived from
first principles in the file header `fast_pam.cpp:11-27` and cross-checked "against the paper
(Alg. 3–4) and the Rust `kmedoids` reference" `:27` (that reference implementation is catalogued at
`.claude/CITATIONS.md:157`). Independent test arbiter: a brute-force best-swap ΔTD scan,
`tests/unit/algorithms/unit_test_faster_pam.cpp:139,228,271-272` [confirmed].

**Phases.** BUILD (k-means++ or k-median++ D-sampling) → `compute_nearest_and_second` → SWAP
(one of three variants) → result write-back.

**Hot loops.**
- `compute_nearest_and_second` `:109-147` — `#pragma omp parallel for schedule(static)` over N
  points, k `dist_by_ind` lookups each. O(N·k); called once per accepted swap.
- `pam1_naive_swap_impl` `:201-232` — `omp for schedule(dynamic, swap_chunk)` over candidates,
  inner `for p` × `for m` ⇒ O(N²·k) per iteration.
- `find_best_swap` `:290-300` — the O(N) Eq.-11 pass; **deliberately sequential**, see §4 P1.
- `fastpam1_swap_impl` `:339-359` — parallel over candidates, one `find_best_swap` each ⇒ O(N²)
  per iteration.
- `fasterpam_swap_impl` `:410-421` — **fully serial** eager sweep; eagerness is the algorithm
  `:383-388`.
- k=1 branch `:463-503` — parallel O(N²) direct argmin, required because `d₂=+inf` makes the
  decomposition NaN `:452-456`.

**Memory access.** Every distance is `prob.dist_by_ind(i,j)` — a *random* packed-triangular lookup
through `core::tri_index` (`core/distance_matrix.hpp:29,60`). `find_best_swap` walks
`dist_by_ind(xj, o)` for `o = 0..N-1`: one row of the logical matrix, but a strided walk of the
packed triangle. Auxiliary state (`nearest`, `nearest_dist`, `second_dist`, `rho`, `ploss`) is flat
`std::vector`, accessed sequentially.

**Parallel decomposition.** Per-thread: `local_delta_m` / `ploss` scratch allocated **once per
thread per iteration** `:197,335` plus `local_best_*` scalars. Shared read-only: `nearest`,
`nearest_dist`, `second_dist`, `rho`, `is_medoid`. Merge: one named critical per site —
`dtwc_pam1_naive_swap_reduce` :236, `dtwc_fastpam1_swap_reduce` :361,
`dtwc_fast_pam_single_medoid_reduce` :493 — entered once per thread per iteration. Failure capture:
`dtwc_medoid_candidate_failure` :224,351,483 and `dtwc_medoid_assignment_failure` :139,
lowest-index-wins so scheduling cannot change which error surfaces.

**RNG / seed portability.** `fast_pam` (unseeded) → `init::Kmeanspp` → **process-global
`dtwc::randGenerator`** (`settings.hpp:55`, consumed at `initialisation.cpp:140,172,184`)
[confirmed]. `fast_pam_seeded` uses an invocation-local `std::mt19937_64` plus
`core::portable_bounded` / `core::portable_weighted_index` `:582-612` [confirmed] — portable by
construction.

**Complexity.** Naive O(N²k)/iter; FastPAM1 O(N²)/iter; FasterPAM O(N²)/sweep.

### 3.2 `fast_clara`

**Oracle.** Kaufman & Rousseeuw (1990) Ch. 3 + Schubert & Rousseeuw (2021) [confirmed
`fast_clara.hpp:6-11`; `.claude/CITATIONS.md:85,90`]. The auto sample size
`max(40+2k, min(N, 10k+100))` `fast_clara.cpp:85-87` is pinned by
`tests/unit/algorithms/unit_test_fast_clara.cpp:465,493` [confirmed].

**Phases.** plan → for each of `n_samples`: `portable_sample_indices` → build a view-mode
sub-`Problem` → `fast_pam_seeded` → map medoids to global indices → assign all N → keep-best →
write-back.

**Hot loops.** `assign_all_points_direct` `:163-189` — `omp parallel for schedule(static)
if (n_points > 64)`, N×k **direct DTW calls** (not cache lookups): `distance(point,
series_at(medoid))` :174, with a `p == medoid ? 0.0` self-distance shortcut.
`assign_all_points_chunked<F32, DtwFn>` `:290-318` — `schedule(dynamic) if (chunk_size > 64)` over
one Parquet batch.

**Memory access.** No N² matrix is ever allocated on the CLARA path; the doc says so explicitly —
"an existing parent distance cache is ignored and left unchanged" `fast_clara.hpp:63-64`
[confirmed]. The sub-`Problem` uses `set_view_data` with `std::span` views into the parent, so no
series is copied `:509,534` ("O(1), no data copy"). Only the s×s sub-matrix is materialised, inside
`fast_pam_seeded`.

**Parallel decomposition.** Per-point disjoint writes to `labels[p]` and `best_dists[p]`; the
`std::function` is resolved **serially before** the region `:206,210` and passed by `const&`. Merge
is `core::detail::ordered_medoid_objective` after the join `:192`; the chunked path accumulates
into a single `OrderedMedoidObjective` across chunks `:274,320`. Failure: one shared
`dtwc_medoid_assignment_failure` critical via `capture_assignment_failure` `:111-122`.

**RNG.** `std::mt19937_64 rng(opts.random_seed)` :343,498; `core::portable_sample_indices`
:353,506 — "Stable O(N)-time selection with O(sample_size) sampling scratch" :350-352; per-subsample
PAM seed `clara_pam_seed = seed + s` with both operands widened to `uint64_t` so the addition cannot
overflow `:146-150` [confirmed]. Portable.

**Complexity.** `n_samples · (O(s²) fill + O(s²) PAM + O(N·k) DTW)`.

### 3.3 `one_batch_pam`

**Oracle.** de Mathelin et al., AAAI 2025, doi:10.1609/aaai.v39i15.33776 [confirmed
`one_batch_pam.hpp:5-6`; `.claude/CITATIONS.md:160`]. The estimator implemented is explicitly a
**hybrid** — count/mean NNIW (Loog 2012) plus the experiment code's finite-table-maximum diagonal
correction — and the source says so, noting that "the paper's literal +infinity and maintained
OneBatchPAM v0.1.0 do not describe this exact hybrid estimator" `:142-148`, with the same caveat
repeated at the point of use `:177-179`. Provenance for both upstream variants (pinned commit
hashes and line ranges) is recorded at `.claude/CITATIONS.md:161-163`. This is the strongest
provenance note anywhere in the scope.

**Phases.** validate → batch and candidate sampling → build the N×m table (`FixedBatchDistances`) →
`nearest_two` over the batch → eager swap on the *estimate* → exact final assignment over all N.

**Hot loops.** Table build `:100-131` — `omp parallel for schedule(dynamic) if(n > 64)`, N×m DTW
calls, row-disjoint writes into `raw`. Swap sweep `:329-359` — **serial**, N candidates × m columns.
Final assignment `:378-388` — **serial by design**, because `exact()` increments the `evaluations`
counter and "correctness and an exact observable count are preferable to an atomic hot path here"
`:369-371` [confirmed].

**Memory access.** `raw` is one flat `n*m` row-major array; `raw[i*m + j]` in the inner `j` loop
`:152-156, 335-345` is a **contiguous scan**. This is the only algorithm in scope with a genuinely
cache-friendly innermost loop.

**Parallel decomposition.** Only the table build is parallel; both getters are resolved serially
first `:92-96` with the reason stated ("a legacy raw semantic mutation may require the mutable
getter to rebind once"), and that behaviour is pinned by
`tests/unit/algorithms/unit_test_one_batch_pam.cpp:257` [confirmed]. Per-row outputs
`row_evaluations`, `row_maxima`; failure via `dtwc_one_batch_table_failure` `:128`.

**RNG.** `std::mt19937_64 rng(options.random_seed)` :270, two `core::portable_shuffle` draws
:273,277. The second is deliberate — "The paper draws the candidate initialization independently of
the fixed batch" `:275-276` [confirmed]. Portable.

**Complexity.** O(N·m) distances, O(N·m) per sweep, plus at most N·k extra distances at the end.

### 3.4 `clarans`

**Oracle.** Ng & Han (2002), IEEE TKDE 14(5) [confirmed `clarans.cpp:5-8`;
`.claude/CITATIONS.md:91`]. The `max(250, 0.0125·k·(N−k))` auto-neighbour formula is attributed to
the paper `:18-19, 60-63`.

**Phases.** for each of `num_local` restarts: sample k medoids → full assignment → randomized swap
loop (strictly improving only, `delta < -1e-12` :199) → keep-best → write-back.

**Hot loops.** Initial assignment `:100-116` O(N·k); swap evaluation `:172-197` O(N) plus O(k) for
points served by the removed medoid; post-swap full reassignment `:207-223` O(N·k).

**Parallel decomposition.** **None.** `clarans.cpp` contains no `#pragma omp` at all [confirmed —
the file appears in no line of `grep -rn "#pragma omp" dtwc/algorithms/`]. Everything is serial
while the equivalent FastPAM / FastCLARA loops are parallel. `medoid_set` is an
`std::unordered_set<int>` probed in a rejection loop `:143-147` — the only hash container in any hot
path in this scope.

**RNG.** `std::mt19937_64 rng(opts.random_seed + restart)` :77-79 ("avoids inter-restart correlation
while keeping the whole run reproducible"), `core::portable_shuffle` :89, `core::portable_bounded`
:137,145. Portable.

**Complexity.** `num_local · (O(N·k) + max_nb · O(N))`, hard-capped by `max_total_swaps = max_nb*10`
:131.

### 3.5 `hierarchical` (`build_dendrogram` + `cut_dendrogram`)

**Oracle.** Generic Lance-Williams; Ward is deliberately excluded because DTW does not satisfy the
squared-Euclidean identity Ward's formula requires `hierarchical.hpp:5-8` [confirmed]. No paper
citation in the file, and none in `.claude/CITATIONS.md` for this algorithm — the weakest oracle
in the scope.

**Phases.** copy the full N×N matrix into `work` → N−1 merges (argmin over active pairs, then
Lance-Williams update) → union-find replay of the first N−k merges → per-cluster medoid scan.

**Hot loops.** `:73-85` — the O(N²) argmin per step ⇒ **O(N³) total**, fully serial. `:95-122` — the
O(N) Lance-Williams update, written symmetrically :120-121. `:249-259` — per-cluster medoid scan,
O(Σ|C|²).

**Memory access.** `work` is a **dense row-major N×N `double`** `:49` — 8N² bytes — populated by N²
`dist_by_ind` calls `:50-52`. Scans are row-major and contiguous, the best access pattern of any
algorithm here; the `max_points = 2000` guard (`hierarchical.hpp:62`) is what keeps it at 32 MB.

**Parallel decomposition.** None; no OpenMP anywhere in the file [confirmed].

**RNG.** None — fully deterministic, and the tie-break policy is documented `:5-8`.

**Complexity.** O(N³) time, O(N²) memory.

### 3.6 `tadpole`

**Oracle.** Begum, Ulanova, Wang & Keogh, KDD 2015 / arXiv:1612.00637, plus Rodriguez & Laio,
*Science* 344:1492 [confirmed `tadpole.cpp:5-7`; `.claude/CITATIONS.md:165-166`, which records the
cases A–D table, the cutoff kernel, and the δ(densest) convention that differs from Rodriguez–Laio].
Admissibility is derived inline `:9-27`, with the one deviation from the paper (`≥ dc` rather than
`> dc` for the LB case) argued explicitly `:18-19`. Open scope limits are named in the source:
**D17** (bit-level identity when a floating reduction lands near `dc` or `best`), **F46**
(finiteness / integer-length representability not validated by the predicate), **F48** (empty
series) `:24-27`.

**Phases.** `bounds_valid` predicate → serial pre-trigger `exact(0,1)` → per-series envelopes →
Stage 1 density → Stage 2 separation/parent → γ ranking → single-pass assignment → objective.

**Hot loops.** Stage 1 `:182-212` — `omp for schedule(dynamic,8) nowait` over i, inner j>i;
`can_prune` selects the entire i-body **outside** the pair loop `:179-181`. Stage 2 `:227-252` —
`omp parallel for schedule(dynamic,8)` over i, inner q over all N with an LB early-skip `:237-239`.

**Memory access.** The LB path reads two `std::span<const double>` sequentially per pair
(`lb_keogh_symmetric` + `diagonal_ub_l1` `:76-81`); the envelope array `envs` is computed once and
reused across all pairs — Begum's cached-envelope idea `:156-168`, with a correctness note that
band<0 must pass the series length as the window or the LB becomes invalid `:157-159`. `exact()`
goes through `dist_by_ind` — random packed lookup.

**Parallel decomposition.** Per-thread `rho_local(N)` plus two counters, merged under the named
critical `tadpole_density_reduce` `:213-218`. `computed` (a shared `vector<char>`) is written
**without synchronisation**, justified at `:141-144`: each unordered pair has exactly one owner per
stage and the stages are barrier-separated. Owner-uniqueness holds because `higher_density` is a
**strict total order** `:86-89`, so for a pair (i,q) exactly one of the two i-iterations examines it
[confirmed by reading `:233-234` against `:86-89`].

**RNG.** None. `tadpole_auto_dc` is explicitly RNG-free — "all pairs among the first min(N, cap)
series (a fixed, reproducible subset — no RNG)" `tadpole.hpp:113-116`, implemented `:101-108`
[confirmed].

**Complexity.** O(N²) pair decisions; exact DTW calls = unpruned pairs only.

### 3.7 `barycenter` (`dtw_barycenter`, `barycenter_kmeans`)

**Oracle.** DBA = Petitjean et al. (2011); SSG = Schultz & Jain (2018); soft-DTW = Cuturi & Blondel
(2017) [confirmed `barycenter.hpp:22-24`; `.claude/CITATIONS.md:79,174,175`]. The SSG step cap
`min(η, 0.5 / max V_ii)` is derived in-source from the fixed-path Hessian `2V` `:310-315`, and the
deviation from the paper's Algorithm 3 (raw direction + one scalar cap, instead of coordinate
preconditioning) is recorded in CITATIONS:175 [confirmed].

**Phases.** copy all series → k-means++ over squared DTW → Lloyd: assign → per-cluster barycenter
update (DBA / SSG / soft-DTW) → converge on stable labels or stable cost.

**Hot loops.** `align_squared` `:193-206` — the **file's own** O(nx·ny) squared-cost DP with an
optional traceback `:210-224`, entirely separate from `dtwc/core`'s DTW kernels. `assign` `:565-577`
— `omp parallel for schedule(static) num_threads(worker_count) if(parallel_assignment)`, N×k DP
alignments. Cluster update `:736-752` — `omp parallel for` over clusters.
`soft_dtw_squared_value_gradient` `:359-413` — forward DP plus reverse adjoint, both `nx*ny`.

**Memory access.** `AlignmentWorkspace::matrix` is one flat row-major DP array `:191`, reserved once
per worker `:664-665` and only `resize`d inside the region — the documented invariant that "no
exception can escape an OpenMP iteration" `:661-663`. An allocation regression gate exists
(`tests/unit/algorithms/unit_test_barycenter_allocations.cpp:104`, a global-`operator new` probe
counting ≥500 KiB allocations) [confirmed].

**Parallel decomposition.** A worker-indexed workspace pool `workspaces[omp_get_thread_num()]`
:567-568, 740. Nested parallelism is explicitly refused: `openmp_region_active()` forces
`max_workers = 1` `:635-639`. Failures are captured per cluster into `failures[cluster_index]` and
rethrown **in cluster order** `:730-731, 749-754` — no critical needed, the pattern
`.claude/LESSONS.md:1007-1015` prescribes.

**RNG.** `std::mt19937_64 rng(options.random_seed)` :667 (k-means++) and :289 (SSG epoch shuffle),
via `core::portable_shuffle` / `portable_weighted_index`. Portable. Per-cluster seeds are derived
`seed + iteration*k + cluster` `:742-745`.

**Complexity.** O(max_iter · N · k · L²) with L the series length.

### 3.8 MIP: Balinski compact, Benders, LR-core, PDLP

**Oracle.** Balinski (1965) for the disaggregated linking inequalities, with the repo explicitly
declining to credit him with the first complete p-median MILP and pointing at Marín & Pelegrín
(2020) for the attribution chain [confirmed `.claude/CITATIONS.md:101,104`]; Benders (1962) +
Magnanti & Wong (1981) `benders.cpp:16-21`; Beasley reduced-cost fixing, with both conditional
bounds derived `reduced_cost_fixing.hpp:6-25`; Geoffrion's theorem for LR-core's LP-equivalence,
derived in `lagrangian_root.hpp:6-25` and referred to `.claude/UNIMODULAR.md §8.2–8.5`. Registered
predictions **P1** (root certifies the heuristic optimal on clustered data,
`lagrangian_root.hpp:22-23`) and **P2** (≥80% candidate elimination when the root gap ≤1%,
`reduced_cost_fixing.hpp:27-29`) are both stated in-source with their verifying tests.

**Compact model** (`mip_Highs.cpp`, `mip_Gurobi.cpp`): N² binaries, cardinality + assignment +
linking, cost scaled by `max(max_distance()/2, 1)` :82 / :83. HiGHS emits `Nb·(Nb-1)` linking rows
:135-139; Gurobi emits `Nb·Nb` :70-72 (the `i == j` rows are `w[ii] <= w[ii]`, trivially redundant)
[confirmed]. Gurobi additionally sets `GRB_IntAttr_BranchPriority = 100` on the diagonal :58-59
with the TU argument inline :56-57.

**Benders** (`benders.cpp`): master = N binaries + N continuous θ :182-183. Loop `:275-396`: solve
master → extract medoids → assign in O(N·k) via `mip::nearest_medoid` :311 → LB = the **dual bound**
(`benders.hpp:67-71`) → converge or add up to N disaggregated cuts, each an O(N) coefficient scan
:372-379. The coefficient floor is `1e-12·max(1, d_nearest)` and the header explains at length why
it must **not** be the cost tolerance — dropping a positive coefficient makes the cut *stricter*
than the valid Benders cut and can remove the optimum `benders.hpp:42-56` [confirmed]. Total loop
cost O(N²) per iteration.

**LR-core** (`lagrangian_root.cpp`): `evaluate_dual` `:115-158` streams D once —
`omp parallel for schedule(static)` for ρ `:120-130` (row-contiguous `const double *Di = D + i*Nz`,
the best memory pattern in the MIP layer) and again for the subgradient `:146-156`; the k smallest ρ
come from `nth_element` :133. Primal repair runs every iteration (`try_primal` :163-192, O(N·k)) and
an O(N²) polish every `polish_period` :288-290 — "throttling the O(N²) polish is the dominant
large-N saving" :287. The Kelley variant `:442-480` adds a BOXSTEP trust region around the best-L
centre, with the reason stated: "unstabilized Kelley throws μ to box corners where L is terrible and
the LB never rises" `:427-429`. Exact B&B `:591-623` is a DFS on y over the core, branching
OPEN-first `:615-616`.

**PDLP** (`pdlp_lp.cpp`): forms the explicit N²-column LP `:86-151` (CSR, with the column-ascending
requirement handled by an explicit `i < j` branch `:130-148`) and solves it with `solver = "pdlp"`.
Explicitly an **arbiter only** — PLAN.md:1123 records "PDLP as production p-median solver:
falsified — matrix-free Kelley dominates on either device (945× CPU / 126× GPU at N=400)"
[confirmed], and `pdlp_lp.hpp:18-21` says the same in the header.

**RNG.** The MIP layer has no RNG of its own. `make_warm_start(prob, prob.random_seed())` threads
the `Problem` seed into `fast_pam_seeded` [confirmed `mip_Highs.cpp:183`, `mip_Gurobi.cpp:103`].
Benders instead runs `cluster_by_kmedoids_lloyd_impl`, which goes through `init_with_seed(random_seed_
+ rep)` (`Problem.cpp:1331`) [confirmed]. So the two warm-start routes have different seed
derivations and different initialisers.

---

## 4. Performance-critical structures and their rationale — DO-NOT-BREAK

| # | Structure | Evidence | Rationale | Cost of a naive "cleanup" |
|---|---|---|---|---|
| **P1** | `find_best_swap` is **sequential** | `fast_pam.cpp:275-281` | Measured: "~10× SLOWER than the sequential loop at N=1000". Fork/join plus a k-wide reduction per candidate cannot be amortised by an O(N) body. | Parallelising it is a *measured* ~10× regression. |
| **P2** | FasterPAM's sweep is serial | `fast_pam.cpp:383-388` | Eagerness is the algorithm: a swap changes state for later candidates in the same sweep. | Parallelising changes the algorithm, not just its speed. |
| **P3** | Per-thread scratch hoisted **once per thread per iteration** | `fast_pam.cpp:197,335,402` (all three labelled "no per-candidate alloc" / "reused across candidates") | Avoids one heap allocation per candidate, i.e. N per sweep. | Moving it inside the candidate loop reproduces exactly the defect that `0af4cbd` fixed in `one_batch_pam` (§5 D-E note). |
| **P4** | Every `omp critical` in a `.cpp` is **named**; the one in `parallelisation.hpp` is **unnamed** | `.cpp` names: `fast_pam.cpp:139,224,236,351,361,483,493`, `one_batch_pam.cpp:128`, `fast_clara.cpp:115`, `tadpole.cpp:213`. Unnamed header critical: `parallelisation.hpp:148` with a 12-line justification `:136-147` | Two opposing recorded lessons: `.claude/LESSONS.md:1062-1069` (an unnamed critical shares one implementation-defined name program-wide ⇒ name every one in a `.cpp`) and `.claude/LESSONS.md:1340-1351` (a *named* critical in a **header** emits `.gomp_critical_user_<name>` as a COMMON symbol and breaks GCC LTO + static-archive links — binutils PR ld/32083, GCC PR lto/116361, observed on GCC 13.3 / binutils 2.42). | Naming the header one re-breaks the Linux MPI CI link. Un-naming the `.cpp` ones re-couples them to every unnamed critical in the process. |
| **P5** | `mip::nearest_medoid` is a **header-inline function template on `Dist`** | `nearest_medoid.hpp:11-14, 40-52` | Stated contract: "never `std::function`, never virtual, and it allocates nothing, so the caller's accessor inlines into the scan." It sits inside the LR subgradient primal repair, run every iteration. | Routing it through `std::function` turns every O(N·k) repair distance into an indirect call. |
| **P6** | `OrderedMedoidObjective::total_` is `volatile double` | `core/medoid_assignment_policy.hpp:69-76,121` | The build enables FP reassociation (`-fassociative-math`), but published objectives are a **cross-route byte contract**. The volatile store is "narrow and intentional". | Removing it lets the compiler reassociate and silently breaks cross-language byte parity. |
| **P7** | Full assignment scans stay **owned by their algorithms** | `core/medoid_assignment_policy.hpp:9-12`, `detail/medoid_utils.hpp:5-13` | They differ in parallel-vs-serial, index space, distance signature and per-element side effects, so a single helper could absorb them only via runtime switches in the library's hottest loops. | A "DRY" merge inserts a runtime policy branch into every N·k / N² element. |
| **P8** | `tadpole`'s `can_prune` selects the whole i-body, not per pair | `tadpole.cpp:179-181, 184, 208` | Loop-invariant hoisting **and** it keeps `prob.series()` — which throws under Float32 — strictly inside the pruning branch. | Re-testing per pair costs a branch per element and re-introduces the f32 throw path. |
| **P9** | `barycenter` refuses nested OpenMP teams | `barycenter.cpp:635-639, 561, 732` | Nested parallelism oversubscribes the host **and** makes the worker-indexed scratch pool unsafe (`omp_get_thread_num()` would collide across teams). | Removing the guard is a correctness bug, not just a slowdown. |
| **P10** | `barycenter` reserves all DP capacity on the caller thread | `barycenter.cpp:661-665` | "Once a parallel region starts, resize/emplace stay within these capacities and no exception can escape an OpenMP iteration." | An in-region allocation that throws out of an OpenMP structured block is UB. |
| **P11** | `fast_clara` uses **view-mode** sub-`Problem`s | `fast_clara.cpp:509, 528, 535` | The point of CLARA is avoiding the O(N²) matrix; copying the subsample would also copy the series. | `set_data` instead of `set_view_data` copies s series per subsample, `n_samples` times. |
| **P12** | `dtw_fn_` is a `std::function` | `Problem.hpp:130-131, 137-138`; rationale `.claude/design.md:16` | Recorded decision: "`std::function` dispatch cost is negligible next to DTW runtime." True on the *direct-DTW* routes (`fast_clara`, `one_batch_pam`) and on `dist_by_ind`'s cache-**miss** path. | De-erasing it means templating `Problem` on the variant — a redesign, not a cleanup. |
| **P13** | Both DTW getters resolved **serially before** an OpenMP region | `one_batch_pam.cpp:92-96`, `fast_clara.cpp:206,210` | The non-const getter can rebind the member (`repair_dtw_binding_after_relocation`); calling it inside a region is a data race. Pinned by `unit_test_one_batch_pam.cpp:257`. | Calling the getter inside the region is a latent race — see §6 D2 for the one place it still happens. |
| **P14** | `dist_by_ind` runs **one** dense-cache refresh, not two | `Problem.cpp:680-684` plus the dedicated `ensure_dense_cache_configuration_current_preflighted()` (`Problem.hpp:237-241`, `Problem.cpp:468-479`) | Explicit: "The SWAP kernel issues N^2 dist_by_ind() calls per iteration, so paying for that preflight twice per element is not free." | Re-collapsing the two helpers restores a per-element cost. (A second preflight still survives by a different route — §6 D6.) |
| **P15** | `tadpole`'s serial pre-trigger `exact(0,1)` before any parallel region | `tadpole.cpp:150-154`; same pattern `initialisation.cpp:96-105`, `Problem.cpp:1225-1230` | `dist_by_ind`'s lazy alloc + `rebind_dtw_fn` is documented not-thread-safe (`Problem.cpp:671-676`). The pre-trigger is honestly counted as one real DTW so pruning is never over-claimed :152-153. | Removing it races the lazy rebind. Making it uncounted would inflate `pruned_fraction()`. |
| **P16** | Known-falsified speed claims must not be re-asserted | PLAN.md:1118-1121 (FastPAM decomposition on a **cached** matrix is **2.95×–8.06×**, not ≥10×, and non-monotone; both variants keep N² lookups/iteration); PLAN.md:1117 (LB pruning cannot reduce work on an *exact* matrix — `Pruned` is a pessimisation there); PLAN.md:1122 ("LB-pruning inside PAM/MIP/LRCore: not admissible — those consumers read the whole matrix"); PLAN.md:1109 (BanditPAM dominated by FasterPAM on precomputed matrices); PLAN.md:1110 (Elkan/triangle pruning invalid — DTW is not a metric); PLAN.md:1123 (PDLP as a production solver falsified); PLAN.md:1005 ("SIMD stays killed") | Registered kills with recorded evidence. | Reopening any of them without overturning the evidence violates PLAN.md:13. |

---

## 5. Duplication

The 2026-09-02 reports counted **seven** nearest-medoid copies in `mip/` and **seven** assignment
scans in `algorithms/`. Both counts have moved.

| Group | Copies at HEAD | Classification | Status vs 2026-09-02 |
|---|---|---|---|
| **D-A. Nearest-medoid scan, MIP side** | **1** definition (`nearest_medoid.hpp:40-52`), used at `benders.cpp:260,311,427` and `lagrangian_root.cpp:71,170,573,633,746` | consolidated | **fixed** (B2). Introduced by `84fc5e5`; the header fixes the tie-break as "FIRST minimum wins" :7-9. |
| **D-B. Nearest-medoid scan, algorithm side** | **7** distinct loops: `fast_pam.cpp:118-133` (nearest+second, parallel, `dist_by_ind`); `one_batch_pam.cpp:215-226` (nearest+second over the *batch column* space, serial, `estimate`) and `:378-388` (final, serial, `exact`); `clarans.cpp:100-116` and `:207-223`; `fast_clara.cpp:171-181` (parallel, direct DTW) and `:299-310` (chunked, global index) | **semantic** — they differ in index space (point vs batch column), distance signature (`dist_by_ind` / `estimate` / `exact` / a bound `DtwFn`), parallel-vs-serial, and per-element side effects (eval counting) | **open by design.** Documented at `core/medoid_assignment_policy.hpp:9-12` and `detail/medoid_utils.hpp:5-13`; see DO-NOT-BREAK **P7**. What *is* shared is the numeric contract (`require_finite_medoid_distance` + `ordered_medoid_objective`), not the loop. |
| **D-C. `clarans.cpp:100-116` vs `:207-223`** | **2**, inside one function | **near-identical**: the loop bodies match line for line; `:100` additionally does `++dtw_evals` :107 while `:207` does not (documented: post-swap lookups are cache hits :205-206) | **open.** The cheapest genuine DRY win in the scope — a file-local lambda, no interface change, no hot-path indirection. |
| **D-D. f32 / f64 chunked assignment** | **1** template `assign_all_points_chunked<F32, DtwFn>` `fast_clara.cpp:237-325`, plus `assign_all_points_direct<Distance, SeriesAt>` `:152-194` | consolidated | **fixed** (was `:229-301` vs `:304-374`). `0af4cbd` records "fast_clara f32/f64 chunk path templated (artifacts byte-identical)". The header states the perf invariant: `F32` selects the reader entry point and accessor **at compile time**, "no runtime branch enters the per-element inner loop" `:232-235`. |
| **D-E. ΔTD / removal-loss decomposition** | **2**: `fast_pam.cpp:288-306` (`ploss`, argmin, `acc + best`) vs `one_batch_pam.cpp:332-349` (`removal_gain`, `max_element`, `add_gain + *best_it`) | **semantic**, sign-mirrored (verified algebraically in the prior report). Different index space (N points vs m batch columns) and different distance source | **open.** A fix to one still will not reach the other. Note `one_batch_pam` has since acquired the hoisting `fast_pam` always had (`:309-325` vs `fast_pam.cpp:322-323,400-402`), so the two are now structurally closer, not further apart. |
| **D-F. Balinski model builders** | **3**: `mip_Highs.cpp:84-162` (FacilityMajor `f*N+p`, cost `d(point, facility)`, `Nb(Nb-1)` linking rows, all N² vars `kInteger` :162); `mip_Gurobi.cpp:54-90` (PointMajor `f+p*N`, cost `d(facility, point)`, `Nb²` linking constraints including `Nb` trivially-redundant `w[ii] <= w[ii]`, all vars `GRB_BINARY` :54); `pdlp_lp.cpp:86-151` (FacilityMajor `i*N+j`, cost `d(facility, point)`, `Nz(Nz-1)` linking rows, continuous) | **semantic** — three different (layout, cost-orientation, row-count) combinations of the same LP. `AssignmentMatrixLayout` (`solution_transaction.hpp:21-24`) papers over the *decode*, not the *build*. The three are equivalent **only while D is symmetric** | **open** (B3), and now **three** copies rather than two — PDLP is the third. |
| **D-G. "run a heuristic without publishing it"** | **3**: `mip::make_warm_start` (`warm_start.cpp:14-47`) = `ExactClusteringTransaction` + `fast_pam_seeded` + four input checks; `prepare_dense_D` (`lagrangian_root.cpp:666-695`) = `ExactClusteringTransaction` + `cluster_by_kmedoids_lloyd` + a two-condition check :681-682; `benders.cpp:134-159` = a hand-rolled `ScopeExit` over **five** fields | **semantic**; all three are now protective | **changed.** B1's "no protection at all" in `prepare_dense_D` is **fixed** (transaction added at :679). The count is still 3, but Benders' divergence is now justified in-source: "NOT `mip::make_warm_start`: this nested Lloyd must also restore `method_`, `n_repetitions` and `last_iterations_`, which the exact-clustering transaction does not own" `:131-133`. |
| **D-H. Cost recomputation** | **4** shapes: `ordered_medoid_objective` (fast_pam :157; clarans :117,224; fast_clara :192,274; one_batch_pam :392); plain `+=` with **no** finite guard (hierarchical :252,265; tadpole :307; benders :108,316); `std::accumulate` (one_batch_pam `estimated_cost` :231 — an *estimate*, not published); `Problem::find_total_cost` (`Problem.cpp:1439-1458`) | **semantic** | **open.** `hierarchical` and `tadpole` publish a `total_cost` computed by unguarded, reassociable `+=` while the other four use the byte-contract accumulator (see §6 D9). |
| **D-I. k=1 / k=N trivial branches** | **4 sites, 6 branches**: `fast_pam.cpp:452-507` (k=1); `one_batch_pam.cpp:245-257` (k=N) and `:293-307` (k=1); `clarans.cpp:126-128` (`all_medoids`); `benders.cpp:92-98` (k=N) and `:101-117` (k=1) | **semantic** | **open.** `lagrangian_root_exact` deliberately has none — "no special-casing needed, the B&B certifies them at the root" `lagrangian_root.cpp:718-719`. That is the design precedent for removing the others. |
| **D-J. A second DTW implementation** | `barycenter.cpp:184-226` is an independent squared-cost, unbanded DP with its own traceback, never touching `dtwc/core`'s kernel family; `soft_dtw_squared_value_gradient` `:350-415` is a third | **semantic** | **open, and arguably necessary** — barycenters need the alignment *path*, which `dist_by_ind` does not return. But `align_squared` gets none of core's banding, variant dispatch, missing-data handling or EAP work. |
| **D-K. Nearest-and-second computation** | **3**: `fast_pam.cpp:95-150` (parallel, over medoid slots, `dist_by_ind`); `one_batch_pam.cpp:204-227` (serial, over batch columns, `estimate`); the inline pair inside `find_best_swap`'s consumer state | **semantic** | **open**, subsumed by D-B / P7. |

---

## 6. Design problems

Severity: **H**igh / **M**edium / **L**ow. Blast radius = what else changes if it is fixed.

### D1 · `Method` cannot name most of the layer — H, blast radius: CLI + both bindings + docs
`Method` is `{Kmedoids, MIP, LRCore, TADPole}` [confirmed `Problem.cpp:1121-1140`]. FastPAM,
FastCLARA, CLARANS, OneBatchPAM, hierarchical and barycenter are reachable **only** as free
functions plus string dispatch. Consequence: `Problem::cluster()` can never reach FastPAM. Two
independent dispatch tables now exist — the 4-value `Method` enum, and a 9-string table in
`api.cpp:81-84` dispatched at `:379-402`. Neither is derivable from the other. **Status: open**
(E1). Candidate action: one enum shared by both routes.

### D2 · Algorithms mutate `Problem` as a result side channel — H, blast radius: every algorithm
Six algorithms end with the same three unguarded writes — `set_n_clusters(k)`, `centroids_ind = …`,
`clusters_ind = …` [confirmed fast_pam :537-539; fast_clara :419-421 and :568-570; one_batch_pam
:252-254 and :396-398; clarans :249-251; hierarchical :279-281; tadpole :330-332] — while
`barycenter_kmeans` takes `const Problem&` and writes nothing (`barycenter.cpp:611`), and all four
MIP backends go through `ExactClusteringTransaction`, which validates *and* rolls back on unwind
(`solution_transaction.cpp:148-171`). Three different publication policies in one library. The
algorithm writes are not transactional: an exception between `set_n_clusters` and the two vector
assignments leaves `Nc` inconsistent with `centroids_ind.size()`. `clarans` can publish **empty**
vectors (D5). Candidate action: route the algorithm write-back through the same transaction.

### D3 · `Problem&` is far wider than any algorithm needs — M, blast radius: whole layer
§2 counts 33 members. Eight (`method_`, `last_iterations_`, `cluster_by_kmedoids_lloyd_impl`,
`set_n_repetitions`, `n_repetitions`, `find_total_cost`, `cluster_by_kmedoids_lloyd`,
`centroid_of`) exist **only** to serve the two nested-heuristic call sites in `mip/`.
`hierarchical` needs 7; `clarans` 6; `barycenter` 6 and read-only. But `fast_clara` *constructs* a
`Problem` and calls seven setters on it (`:367-373, 515-536`), so the layer depends on `Problem`'s
constructor, move semantics and lifecycle — not on a distance interface. Candidate action: separate
the four genuinely-needed capabilities (distance oracle, series accessor, configuration snapshot,
labels sink) from the sub-problem factory.

### D4 · Settings sprawl — M, blast radius: CLI, both bindings, docs
Eight option structs, none composable, with validation in five different places
(`barycenter.cpp:97-109`, `fast_clara.cpp:50-61`, `one_batch_pam.cpp:52-69`, inline
`clarans.cpp:51-56`, `Problem.hpp:79-90`):

| Struct | Fields | Where each field is consumed | Reachable from `Problem`/CLI? |
|---|---|---|---|
| `MIPSettings` (`Problem.hpp:64-74`) | 8 | `mip_gap` benders :229,328 / Highs :169 / Gurobi :95; `time_limit_sec` ×3; `warm_start` ×3; `numeric_focus`, `mip_focus` **Gurobi only** :93,94; `verbose_solver` ×4; `max_benders_iter` benders :266; `benders` `Problem.cpp:1178`; `lr_max_nodes` LR :721 | yes, and validated (`validate_mip_settings`) |
| `LagrangianParams` (`lagrangian_root.hpp:43-54`) | **10** | `max_iters` :273; `rel_gap_tol` :223,295,474; `lambda0` :264; `stall_halve` :307; `lambda_min` :310; `deflect` :318,321; `polish_period` :289; `kelley_max_major` :442; `max_nodes` :592 | **only `max_nodes`** (`lagrangian_root.cpp:721`). 9 of 10 unreachable outside C++ |
| `PdlpParams` (`pdlp_lp.hpp:42-52`) | 5 | `variant` :160,176; `tol` :164; `iteration_limit` :165; `use_gpu` :177; `verbose` :158 | Python / MATLAB only |
| `CLARAOptions` (`fast_clara.hpp:36-50`) | 10 | all consumed; `force_parquet_streaming` :435,440,445,457 | `api.cpp:389-393` sets 3 of 10 |
| `OneBatchPAMOptions` (`one_batch_pam.hpp:42-49`) | 6 | all consumed | `api.cpp:383-386` sets 3 of 6 |
| `CLARANSOptions` (`clarans.hpp:33-40`) | 5 | all consumed | **not reachable from CLI/api** at all |
| `HierarchicalOptions` (`hierarchical.hpp:60-63`) | 2 | both consumed | `api.cpp:395` uses defaults only |
| `BarycenterOptions` / `BarycenterClusteringOptions` (`barycenter.hpp:38-63`) | 7 / 10 | all consumed | Python only |

`BarycenterClusteringOptions` is `BarycenterOptions` plus 3 fields, hand-copied field-by-field at
`barycenter.cpp:625-632` — composition expressed as duplication, with a `validate_options` call on
the copy `:633` because the copy is the only validated object.

### D5 · Unvalidated `CLARANSOptions` fields publish an empty clustering — M, blast radius: clarans + every downstream score
`opts.num_local` is never validated [confirmed `clarans.cpp:72` is its only use]. With
`num_local <= 0` the restart loop never runs, `best_present` stays false, and `:249-251` writes an
**empty** `centroids_ind` / `clusters_ind` alongside `set_n_clusters(k)`. `max_neighbor` and
`max_dtw_evals` are likewise unvalidated. Contrast `fast_clara.cpp:54-60` and
`one_batch_pam.cpp:52-69`, which validate every control. **Status: open** (A9). Candidate action:
a `validate_clarans_options` mirroring `validate_clara_controls`.

### D6 · `dist_by_ind` still runs **two** preflights per element, and its comment says one — M, blast radius: every N² kernel
`Problem.cpp:680-681` states "Exactly ONE preflight per call". But `:682` calls
`preflight_current_distance_semantics()` and `:683` calls `validate_mmap_cache_identity()`, whose
**first statement** is `preflight_current_distance_semantics()` (`Problem.cpp:598`) [confirmed].
Each preflight is `validate_precision` + `validate_distance_matrix_strategy` +
`validate_cuda_settings_precision` + `validate_problem_distance_semantics`
(`core/distance_semantics.hpp:91-117`, itself six-plus checks). The SWAP kernel issues N² of these
per iteration. **Status: A15 half-fixed** — the *dense-cache* double preflight is gone (P14), the
mmap one is not, and the comment is now inaccurate. Mitigating factor: IPO/LTO is ON by default
(`cmake/ProjectOptions.cmake:21`, `cmake/InterproceduralOptimization.cmake:5`), so cross-TU
inlining is possible — but not confirmed here (see §10 Q10). Candidate action: measure before
touching; a `filled && validated` fast path is the obvious shape.

### D7 · Silent no-op on an invalid Benders problem — M, blast radius: `Method::MIP`
`benders.cpp:86-89`: `if (Nb <= 0 || Nc <= 0 || Nc > Nb) { std::cerr << …; return; }` — prints and
**returns**, leaving `centroids_ind`/`clusters_ind` untouched and the process exit status success.
Every sibling throws (`fast_clara.cpp:74-79`, `clarans.cpp:53-56`, `lagrangian_root.cpp:670`,
`one_batch_pam.cpp:58-62`). This is the last silent-success path in the MIP layer after `84fc5e5`
removed the others. Candidate action: throw `InvalidInput`.

### D8 · String-typed choices where every sibling is an enum — M, blast radius: CLI + bindings
`MIPSettings::benders` is `std::string` (`Problem.hpp:72`), dispatched by string comparison at
`Problem.cpp:1178`. It is now *validated* (`validate_mip_settings`, `Problem.hpp:88-89`) and
`cluster_by_mip` validates before dispatching `Problem.cpp:1177`, so `"On"` is a loud error rather
than a silent "off" — **C1 changed from silent-wrong to loud**. Still the only string selector where
every sibling (`DistanceMatrixStrategy`, `Solver`, `Method`, `Linkage`, `PAMVariant`,
`OneBatchWeighting`, `BarycenterMethod`, `AssignmentMatrixLayout`) is an enum with a `validate_*`
free function. Same shape for `PdlpParams::variant` (`pdlp_lp.hpp:44`) — unvalidated, but now
hard-failed downstream by `set_highs_option` (`pdlp_lp.cpp:160`). Candidate action:
`enum class BendersMode { Auto, On, Off }`.

### D9 · `hierarchical` and `tadpole` skip the numeric contract every other algorithm enforces — M
`hierarchical.cpp:250-252, 264-265` and `tadpole.cpp:304-307` accumulate `total_cost` with plain
`+=` and never call `require_finite_medoid_distance` or `ordered_medoid_objective`. In
`hierarchical`'s medoid scan a NaN makes `cost < best_cost` false for every candidate, so `best_idx`
silently stays `mem[0]` (`:247`) — exactly the failure mode `0af4cbd` fixed in `one_batch_pam` and
described at `one_batch_pam.cpp:373-377`. **Status: new finding, open.** Candidate action: route both
through `OrderedMedoidObjective` (accepting the P6 volatile cost).

### D10 · `int` truncation of `prob.size()` in two algorithms — L (needs N > INT_MAX)
`hierarchical.cpp:32` `const int N = static_cast<int>(prob.size());` runs **before** the
`N > opts.max_points` check `:34`, so `size() == 2³²+100` yields `N == 100` and passes the guard.
Same pattern at `tadpole.cpp:96, 127`. Contrast `fast_pam` / `fast_clara` / `one_batch_pam`, which
all use a checked converter (`detail::checked_fast_pam_point_count` `fast_pam.cpp:54-64`,
`resolve_clara_plan:71`, `validate_options:56`). **Status: new finding, open.** [inferred —
reproducing needs >2³¹ series.] Candidate action: reuse `checked_fast_pam_point_count`.

### D11 · CLARANS' 64-bit fix is cosmetic, and one line is latent UB — L/M
`clarans.cpp:49-50` widens `N` to `int64_t` and asserts `static_assert(sizeof(N) >= 8, …)` — a
tautology testing nothing, whose comment claims an audit finding is closed. Every loop counter stays
`int` (`:100, 172, 207`), `labels` is `vector<int>`, and `:145` narrows back for the RNG. Beyond
`INT_MAX` this is signed overflow, not truncation. Separately, `:62`
`static_cast<int>(0.0125 * k * (N - k))` converts a `double` that can exceed `INT_MAX` (k=1000,
N=10⁸ ⇒ 1.25e9; then `max_total_swaps = max_nb * 10` at `:131` overflows `int`) — `static_cast<int>`
of an out-of-range `double` is UB. **Status: A10 open, plus a new UB line.**

### D12 · Error handling mixes typed and untyped failures — M, blast radius: every caller's catch block
Typed failures in the layer are `InvalidInput`, `SolverError`, `DeviceError`. Exceptions to that:
`clarans.cpp:52, 54` throw bare `std::runtime_error` while `clarans.cpp:165` throws `InvalidInput`
inside the *same function*; `hierarchical.cpp:35, 40, 179, 185, 192, 199, 229` all throw
`std::runtime_error`; `detail::validate_medoids` (`medoid_utils.hpp:32, 35, 41`) throws
`std::runtime_error`, documented at `fast_pam.hpp:111`; `benders.cpp:87` writes to `std::cerr` and
returns (D7). A caller catching `InvalidInput` therefore misses `clarans`' size errors,
`hierarchical` entirely, and `fast_pam_swap`'s medoid errors. Note `mip_Gurobi.cpp:146-158` has a
deliberate, documented handler ordering (`GRBException` derives from `std::runtime_error` in this
Gurobi version, so it *must* precede the `std::runtime_error` handler) — an example of the
right treatment.

### D13 · The B&B copies a `std::vector<int>` per node — M, blast radius: LR-core at large gaps
`lagrangian_root.cpp:585` `struct Frame { int pos; int opened; double opened_rho; std::vector<int> S; };`,
pushed twice per expansion (`:617` copies `fr.S`; `:618-619` copies it again into `S_open`). With
`max_nodes` defaulting to 2 000 000 (`Problem.hpp:73`) that is up to ~4M vector allocations on the
adversarial regime the header explicitly anticipates (`lagrangian_root.hpp:128-132`). Candidate
action: one shared path array plus a depth index, or a small-vector.

### D14 · `prepare_dense_D` duplicates the whole distance matrix — M, blast radius: LR-core memory
`lagrangian_root.cpp:689-693` allocates a full **N×N** `double` array (`N²·8` bytes) with a serial
O(N²) `dist_by_ind` copy, *on top of* `Problem`'s packed `N(N+1)/2` matrix. At the
`require_index_range` ceiling (N = 46 340) that is 17 GB plus 8.6 GB. The header says
"materializes a dense copy of D" `lagrangian_root.hpp:145` — accurate, but the cost is not stated,
and the LR-core route is precisely the one sold as "matrix-free" (`lagrangian_root.hpp:19-20`
"never forming the N²-column LP"). The *LP* is matrix-free; the *distance matrix* is doubled.

### D15 · A `detail` header is in the public umbrella; a shipped algorithm is not — L
`dtwc.hpp:34` includes `algorithms/detail/medoid_utils.hpp`, while `algorithms/tadpole.hpp` is
**not** in the umbrella at all [confirmed `grep -n tadpole dtwc/dtwc.hpp` → no match]. The public
surface exports an internal helper and omits a shipped algorithm.

### D16 · Two `.cpp` files share one critical name across TUs — L
`dtwc_medoid_assignment_failure` is used in both `fast_pam.cpp:139` and `fast_clara.cpp:115`, so
those two algorithms share one runtime lock. Both are exception paths (cold), so this is
acceptable — but it is a *policy* name, not a per-site name, and `.claude/LESSONS.md:1068` says
"Every `critical` in the codebase must carry a distinct name". Worth clarifying the lesson rather
than changing code.

### D17 · Two unreachable duplicate-centroid guards — L
`initialisation.cpp:185-188` and `fast_pam.cpp:606-609` re-check that the sampled index is not
already selected. `core::distance_sampling_weights` leaves every selected index at weight `0.0`
(`distance_sampling_weights.hpp:57-58`: `if (is_selected[i]) continue;` — the slot is never written,
so it keeps its value-initialised `0.0`), and `core::portable_weighted_index` cannot return an index
with weight 0 (`portable_random.hpp:105-125`: a zero weight leaves `cumulative` unchanged, so
`threshold < cumulative` cannot newly become true at that index, and the `last_positive` fallback is
by definition a positive-weight index). `std::discrete_distribution` assigns a zero weight
probability zero. **Status: D3 open** [inferred from those two sources plus [rand.dist.samp.discrete]].

### D18 · `pruned_fraction()` over-reports after `tadpole_auto_dc` — L
`tadpole_auto_dc` calls `prob.dist_by_ind(i,j)` **directly** (`tadpole.cpp:108`), not through
`exact()`, so up to 2016 warmed pairs are never flagged in `computed`. A later `tadpole()` on the
same `Problem` finds them free and never counts them, inflating `pruned_fraction()`.
`Problem::cluster()` performs exactly this sequence (`Problem.cpp:1132-1135`). **Status: open.**

### D19 · `fast_pam`'s BUILD state save/restore is dead on the success path — L
`fast_pam.cpp:555-557` saves `Nc` / `centroids_ind` / `clusters_ind`; `:563-565` restores them; then
`fast_pam_swap` overwrites all three at `:537-539`. The restore has value **only** on the throwing
path — it gives `fast_pam` a strong exception guarantee if `fast_pam_swap` throws after
`init::Kmeanspp` has already mutated `prob`. That is neither documented nor tested. **Status: C5
open, with a nuance the prior report missed** [inferred].

### D20 · Overclaiming header docs — L
- `fast_pam.hpp:43` "Digit-identical to `FastPAM1Naive`". The test comparing them uses
  `WithinAbs(…, 1e-9)` and its own comment says "Exact medoid identity is NOT required … The
  objective is the invariant" (`unit_test_faster_pam.cpp:277-295`) [confirmed]. **Open.**
- `fast_pam.hpp:47` "Never worse in objective than FastPAM1" — still a hard `REQUIRE`
  (`unit_test_faster_pam.cpp:442`) with no proof; eager and best-first reach *different* local optima
  of the same neighbourhood. It now sits inside a hidden `[.]` bench, so it cannot flake the default
  suite. **Open, de-risked.**
- `tadpole.hpp:10-11, 82-83` "the one method that need NOT materialise the full N×N matrix" /
  "Does NOT call `fill_distance_matrix()`". True for *computation*; false for *allocation* — §7.

---

## 7. Obsolete and dead code, stale comments

| Item | Proof | Status |
|---|---|---|
| **The three dead `medoid_utils` helpers** (`assign_to_nearest`, `compute_nearest_and_second`, `find_cluster_medoid`) and their 20 unit tests | `wc -l dtwc/algorithms/detail/medoid_utils.hpp` → **44** (was 115+). `grep -n TEST_CASE tests/unit/algorithms/unit_test_medoid_utils.cpp` → **5 cases, all `validate_medoids`**. `0af4cbd` records "Dead medoid_utils helpers deleted (library objects byte-identical)". | **fixed (D1).** The header now carries a 9-line rationale for why they will not return `:5-13`. |
| **`test_scores_adversarial.cpp` `#if 0` block** | `grep -rn "#if 0" tests/ dtwc/` → one hit, `tests/unit/adversarial/test_scores_adversarial.cpp:451`, now a **comment** citing D2, not a preprocessor block. | **fixed (D2).** |
| **`benders.hpp` stale "prints a diagnostic and returns"** | `benders.hpp:83-85` now reads "throws `dtwc::SolverError` if HiGHS is not compiled in, and also if the cut loop reaches its iteration cap without closing the bound gap"; matches `benders.cpp:403-416, 437-438`. | **fixed.** |
| **`mip_Gurobi.cpp` missing / unused includes** | `<iostream>` :21 and `<algorithm>` :19 are present and used (:119, :83); `<limits>` :22 is used (:50); `<string_view>` is gone. `grep -n cassert dtwc/mip/benders.cpp` → no match. | **fixed.** |
| **`tadpole.hpp` "need NOT materialise the full N×N matrix"** (`:10-11`, `:82-83`) | `tadpole.cpp:139` allocates `vector<char> computed(packed_size(N))` = N(N+1)/2 bytes, and `:154` `exact(0,1)` → `dist_by_ind` → `DenseDistanceMatrix::resize(N)` → `data_.assign(packed_size(n), NaN)` (`core/distance_matrix.hpp:36,51-55`), i.e. **8·N(N+1)/2 bytes**. Peak ≈ 4.5·N² bytes. | **open (A11).** The *computation* claim holds; the *materialisation* claim does not. |
| **`Problem.cpp:680` "Exactly ONE preflight per call"** | `:682` plus `:683`→`:598` = two. | **open** — stale comment introduced by the partial A15 fix (D6). |
| **`hierarchical.cpp:78-79` and `:254-255` dead tie-break clauses** | `:73-75` iterates `i`, then `j = i+1`, ascending, so the first minimum is already lexicographically smallest and `(d == best_dist && (i < best_a || …))` can never fire. Same at `:254-255`: `mem` is built ascending `:236-237`. | **open (C4).** Harmless, but they advertise a determinism mechanism that does nothing. |
| **`initialisation.cpp:185-188`, `:207-210`; `fast_pam.cpp:606-609` duplicate guards** | See D17. | **open (D3).** |
| **`clarans.cpp:50` tautological `static_assert`** | `static_assert(sizeof(N) >= 8, …)` on an `int64_t`. Cannot fail. Its comment claims audit finding R4 is closed. | **open (A10).** |
| **`fast_clara.cpp:271,278` "infinite loop if `rg_per_batch == 0`"** (prior C6) | `ParquetChunkReader::row_groups_per_batch` returns 0 **only** when `num_row_groups_ == 0` (`io/parquet_chunk_reader.hpp:355`), and otherwise returns `max(1, …)` (`:378`), throwing if even one row group exceeds the budget (`:367-374`). With `num_row_groups_ == 0`, `total_rg == 0` and `for (rg = 0; rg < 0; …)` never executes. | **not a defect — prior C6 withdrawn** on this evidence. |
| **`lagrangian_root(Problem&)` has no production caller** | `grep -rn "lagrangian_root(" dtwc/ python/ bindings/ tests/` → the only non-test callers are `lagrangian_root.cpp:499` (the `D*` overload, from `lagrangian_root_exact` when HiGHS is off) and `:703`. The `Problem&` overload is reached only from `tests/unit/mip/test_lagrangian_root.cpp:494` and `test_mip_backend_guards.cpp:193`. | **open, low.** A documented public bound-only API with a dedicated no-side-effect guard test; unexercised in production, not dead. |
| **`clarans` has no CLI or Tier-1 caller** | `grep -n "clarans\|CLARANS" dtwc/dtwc_cl.cpp dtwc/api.cpp` → **no match**. Reachable only via `dtwc.hpp:33` and the Python / MATLAB bindings. | **consistent with its own `@warning` (`clarans.hpp:15`).** Not dead. |
| **Prune-rate and identity floors still hidden** | `grep -rn '"\[\.\]"' tests/unit/algorithms tests/unit/mip` → `unit_test_tadpole.cpp:342` (≥50% prune), `unit_test_faster_pam.cpp:385` (`fp1_identical` + `faster_never_worse`), `test_lagrangian_root.cpp:403,453`, `test_pdlp_lp.cpp:247`. None run by default. | **D5 partially open.** The OneBatchPAM quality band was promoted out of `[.]` (`unit_test_one_batch_pam.cpp:291`); TADPole's prune floor and FastPAM's identity claim were not. |
| **Killed ideas lingering in code** | Grepped PLAN.md:1108-1123's terms (FastDTW, BanditPAM, Elkan, triangle pruning, OSLP, SIMD) against both directories → no residue. `pdlp_lp` is retained *and* correctly labelled an arbiter (`pdlp_lp.hpp:14-21`, PLAN.md:1123). | **clean.** |
| **Options fields never read** | Every field of all eight option structs has ≥1 consumer [confirmed by a per-field grep over `dtwc/`, `python/src/`, `bindings/`]. | **clean** — the problem is reachability (D4), not deadness. |

---

## 8. Lock, atomic, critical, and shared-state inventory

Complete for the scope [confirmed by `grep -rn "#pragma omp\|std::atomic\|std::mutex\|thread_local\|omp_get_thread_num\|omp_in_parallel" dtwc/algorithms/ dtwc/mip/ dtwc/initialisation.cpp`].

| Site | Construct | Hot / cold | Invariant protected | Notes |
|---|---|---|---|---|
| `fast_pam.cpp:109` | `omp parallel for schedule(static)` | **hot** (O(N·k)/call) | none — disjoint writes to `nearest[p]`, `nearest_dist[p]`, `second_dist[p]` | lock-free by construction, documented `:107-108` |
| `fast_pam.cpp:139` | `critical(dtwc_medoid_assignment_failure)` | cold (throw only) | lowest-index `exception_ptr` wins | name shared with `fast_clara.cpp:115` (D16) |
| `fast_pam.cpp:195/201`, `333/339`, `463/468` | `omp parallel` + `omp for schedule(dynamic, chunk)` (`nowait` at :468) | **hot** | per-thread scratch and locals | chunk from `dtwc::omp_chunk_size(N)` (`parallelisation.hpp:74`) |
| `fast_pam.cpp:224, 351, 483` | `critical(dtwc_medoid_candidate_failure)` | cold | same as above | three sites, one name, never concurrent |
| `fast_pam.cpp:236` | `critical(dtwc_pam1_naive_swap_reduce)` | warm — once/thread/iteration | best-swap reduction under the `better_swap` total order `:165-170` | named by `0af4cbd`; **G1 fixed**, with the reason inline `:233-235` |
| `fast_pam.cpp:361` | `critical(dtwc_fastpam1_swap_reduce)` | warm | ditto | **G1 fixed** |
| `fast_pam.cpp:493` | `critical(dtwc_fast_pam_single_medoid_reduce)` | warm | k=1 argmin reduction | **G1 fixed** |
| `fast_clara.cpp:115` | `critical(dtwc_medoid_assignment_failure)` | cold | lowest-index failure | see D16 |
| `fast_clara.cpp:163` | `omp parallel for schedule(static) if (n_points > 64)` | **hot** (N·k DTW) | disjoint `labels[p]`, `best_dists[p]` | `std::function` resolved before the region (P13) |
| `fast_clara.cpp:290` | `omp parallel for schedule(dynamic) if (chunk_size > 64)` | **hot** | disjoint within one chunk; the reader is **not** called inside `:288-289` | |
| `one_batch_pam.cpp:42-43` | `static std::mutex` + `lock_guard` | cold (≤1/invocation, before the table build) | one uninterleaved `std::cerr` line | **G6 open.** Deliberate and documented `:38-41`; still a process-wide lock plus a `std::cerr` write inside a core algorithm |
| `one_batch_pam.cpp:100` | `omp parallel for schedule(dynamic) if(n > 64)` | **hot** (N·m DTW) | row-disjoint writes to `raw`, `row_evaluations`, `row_maxima` | |
| `one_batch_pam.cpp:128` | `critical(dtwc_one_batch_table_failure)` | cold | first-wins `exception_ptr` | named by `0af4cbd`; **G1 fixed** |
| `tadpole.cpp:174/182` | `omp parallel` + `omp for schedule(dynamic,8) nowait` | **hot** (N²/2 pairs) | per-thread `rho_local(N)`; **unsynchronised** shared `vector<char> computed` | owner-uniqueness argued `:141-144`, verified against the strict order `:86-89` |
| `tadpole.cpp:213` | `critical(tadpole_density_reduce)` | warm — once/thread | ρ and the two prune counters | already named before 2026-09-02 |
| `tadpole.cpp:227` | `omp parallel for schedule(dynamic,8)` | **hot** | disjoint `delta[i]`, `parent[i]`; shared `computed` as above | |
| `barycenter.cpp:563` | `omp parallel for schedule(static) num_threads(worker_count) if(parallel_assignment)` | **hot** (N·k DP) | `workspaces[omp_get_thread_num()]`; disjoint `labels[i]`, `local_costs[i]` | `parallel_assignment` is false inside an outer region `:561` |
| `barycenter.cpp:734` | `omp parallel for schedule(static) num_threads(update_workers) if(parallel_updates)` | **hot** | per-cluster `failures[]` slots — **no critical**, the pattern `.claude/LESSONS.md:1007-1015` prescribes | `update_workers <= assignment_workers` holds only because `k <= n` (`:618-619`) — an implicit invariant |
| `barycenter.cpp:68-84` | `omp_get_thread_num()` / `omp_in_parallel()` | — | worker-indexed scratch; nested-team refusal | see P9 |
| `lagrangian_root.cpp:120, 146` | `omp parallel for schedule(static)` | **hot** (two O(N²) streams per dual evaluation) | disjoint `rho[i]` / `g[j]`; the bodies are pure arithmetic on `const double*` and cannot throw | no exception machinery needed — the cleanest parallel region in the scope |
| **outside scope, reached from it** — `parallelisation.hpp:148` | **unnamed** `#pragma omp critical` | cold (exception path) | first-failure `exception_ptr`, with an `std::atomic<int>` cutoff at `:123-135` | **deliberate.** 12-line justification `:136-147`; `.claude/LESSONS.md:1340-1351`. Reached from `initialisation.cpp:112`, `Problem::assign_clusters`, `Problem::calculate_medoids`. |
| **outside scope, on the hot path** — `Problem.cpp:697` | `critical(distByInd_init)` | cold-ish, guarded by an unsynchronised `needs_init` read `:694` | one-shot `DenseDistanceMatrix::resize` + `rebind_dtw_fn` | **G2 open.** `dist_by_ind`'s own doc says the lazy path is *not* thread-safe and needs a serial priming call `:671-676`; `tadpole.cpp:150-154`, `initialisation.cpp:96-105` and `Problem.cpp:1225-1230` each re-implement that priming separately. |
| — | `thread_local` | — | — | **none in scope** (design.md:26 prescribes it for the matrix-fill path, outside this scope) |
| — | `std::atomic` | — | — | **none in scope** |

**Shared mutable state without a lock:** `dtwc::randGenerator` (`settings.hpp:55`), consumed at
`initialisation.cpp:140, 172, 184`, reachable from `fast_pam.cpp:560`. **G4 open.** It is now
documented as a "legacy mutable Tier-2 facility" and `settings.hpp:122-123` asserts that "Tier-1
entry points do not read, reseed, or otherwise consume that process-global engine" — but the
unseeded `dtwc::fast_pam` does, and it is exported to Python (`python/src/_dtwcpp_core.cpp:1216`)
and MATLAB (`bindings/matlab/dtwc_mex.cpp:1269, 1583`) [confirmed]. Two concurrent unseeded
`fast_pam` calls are a data race, and any unseeded BUILD result depends on prior consumption
elsewhere in the process. `api.cpp:380` correctly uses `fast_pam_seeded`, so the Tier-1 C++ API is
clean; the bindings are not.

---

## 9. Extension points today

| Extension | Mechanism | Compile-time / runtime | Reach | Notes |
|---|---|---|---|---|
| **Initialisation** | `std::function<void(Problem&)> init_fun{ init::random }` (`Problem.hpp:319`), invoked by `Problem::init()` :682 | runtime | Lloyd only (`Problem.cpp:1331` → `init_with_seed`) | `init_with_seed` (`Problem.cpp:1281-1298`) recovers the `void(*)(Problem&)` target and special-cases **only** `&init::random` and `&init::Kmeanspp`; anything else — including a lambda wrapping `init::random` — falls through to the unseeded `init()`. That fallthrough is now *documented as deliberate* :1294-1296 ("An arbitrary callback has no seed parameter, so retain its exact legacy invocation semantics rather than silently replacing it with the default initializer"). **A13 changed from silent bug to documented limitation.** No algorithm in `dtwc/algorithms/` consumes `init_fun`; `fast_pam` hard-codes `init::Kmeanspp` (`:560`), `fast_pam_seeded` hard-codes its own inline D-sampling (`:582-612`). |
| **Custom distance** | `Problem::set_variant(core::DTWVariant / DTWVariantParams)` (`Problem.hpp:539-540`) → `resolve_dtw_fn` rebinds `dtw_fn_` / `dtw_fn_f32_` | runtime, but a **closed set** | all algorithms | There is no hook to install an arbitrary `dtw_fn_t`: `dtw_fn_` is private (`Problem.hpp:137`) and both getters return `const&`. "Custom distance" means "one of the built-in variants". |
| **Assignment policy** | `core/medoid_assignment_policy.hpp` exports `require_finite_medoid_distance`, `require_finite_candidate_distance`, `ordered_medoid_objective`, `OrderedMedoidObjective` | compile-time, header-inline | opt-in per algorithm | **Not a policy object** — free functions each algorithm calls by hand, and two do not (D9). The header states the scans are *not* candidates for a shared loop `:9-12`. |
| **Solver** | `Problem::set_solver(Solver)` (`Problem.hpp:421`) + the `mip_settings.benders` string; dispatch `Problem.cpp:1177-1194` | runtime, closed set `{Gurobi, HiGHS}` | `Method::MIP` only | Backend *availability* is compile-time (`DTWC_ENABLE_GUROBI`, `DTWC_ENABLE_HIGHS`, `mip/CMakeLists.txt:60-73`); an unavailable backend throws `SolverError` from the `#else` arm (`mip_Highs.cpp:236`, `mip_Gurobi.cpp:160`, `benders.cpp:437`, `lagrangian_root.cpp:355`, `pdlp_lp.cpp:46`). Consistent across all five. |
| **PAM variant** | the `PAMVariant` parameter of `fast_pam_swap` (`fast_pam.hpp:114`), dispatched `fast_pam.cpp:509-524` | runtime enum | `fast_pam_swap` only — `fast_pam` and `fast_pam_seeded` hard-code `FastPAM1` (`:571, 613`) | the enum's stated purpose is "the bench A/B them on an identical BUILD (isolating swap cost)" `:50` |
| **Templating** | `assign_all_points_direct<Distance, SeriesAt>` `fast_clara.cpp:152`; `assign_all_points_chunked<bool F32, DtwFn>` `:237`; `mip::nearest_medoid<Dist>` `nearest_medoid.hpp:41`; `mip::set_highs_option<Value>` `highs_support.hpp:30`; `benders_master_lower_bound<MasterSolver>` `benders.hpp:67`; `init::random_with<Shuffle>` / `kmeanspp_with<FirstIndex, WeightedIndex>` `initialisation.cpp:57, 75` | compile-time | — | All six exist to **avoid an indirect call or a solver-header dependency**, not to offer user extension. `benders_master_lower_bound` is templated purely so `benders.hpp` pulls in no HiGHS declaration (HiGHS is a PRIVATE dependency of `mip-solvers`) `:63-65`. f32-vs-f64 is `if constexpr (F32)`, not a user-visible parameter. |
| **Not extensible** | linkage rule, barycenter method, weighting scheme, TADPole's bound choice, the Balinski model itself, `LagrangianParams`' 9 non-`max_nodes` knobs | — | — | All closed enums, hard-coded, or unreachable from any public surface (D4). |

---

## 10. Open questions for the designer

1. **What is the minimal interface the algorithms actually need?** §2 gives the exact 33-member
   dependency. Eight members exist **only** for the two nested-heuristic call sites in `mip/`. A
   distance oracle plus a labels sink would cover 12 of the remaining 25 — but `fast_clara`
   *constructs* a `Problem` (D3), so it needs a factory too. Is that construction essential, or
   could the CLARA subsample route call `fast_pam_swap` against a sub-matrix directly?

2. **Should result publication be one policy?** Three exist: raw triple-write (six algorithms), no
   write (barycenter), and `ExactClusteringTransaction` (all four MIP backends). Is barycenter's
   difference a signal that `ClusteringResult` is simply the wrong output type for sequence
   centroids, or that the write-back contract should be optional?

3. **Three Balinski builders (D-F) — consolidate or delete?** PDLP is a declared arbiter
   (PLAN.md:1123). Gurobi's extra `Nb` redundant linking rows and its all-binary declaration
   (against design.md:71's TU argument) suggest the two compact builders have drifted
   independently. Consolidating requires a decision on whether `d(point, facility)` vs
   `d(facility, point)` is allowed to depend on D's symmetry.

4. **What is the real cost of `dist_by_ind` in the SWAP kernel?** D6/P14 identify a residual double
   preflight. Nobody has measured it. The decisive test is cheap: a filled-matrix N=2000 FastPAM1
   sweep with and without the `validate_mmap_cache_identity` preflight. Until that exists, any
   "hoist a raw accessor" work is speculative — and PLAN.md:1118-1121 is the standing precedent for
   an unmeasured swap-speed claim being falsified.

5. **Is `LagrangianParams` (10 fields, 1 reachable) a surface gap or dead configuration?** Either
   expose it (a `Problem::lr_settings` mirroring `mip_settings`) or shrink it to what LR-core's
   callers can actually vary.

6. **Should `hierarchical` and `tadpole` adopt the numeric contract (D9)?** They are the only two
   algorithms publishing a `total_cost` from unguarded, reassociable `+=`. Adopting
   `OrderedMedoidObjective` costs one volatile store per point (P6) — real on tadpole's O(N) final
   loop, negligible against hierarchical's O(N³).

7. **Is `clarans` worth keeping?** No CLI or Tier-1 caller, no OpenMP anywhere in the file, two
   unvalidated options that can publish an empty clustering (D5), a latent `int` UB (D11), and a
   header that already says "Experimental … Use FastCLARA for large N. Promote only after
   benchmarks justify" (`clarans.hpp:15-16`). Has that benchmark ever been run?

8. **Does `barycenter`'s private DTW (D-J) need to converge with `dtwc/core`?** It is the only
   consumer that needs the alignment *path*. Exposing a path-returning core kernel would remove
   ~250 lines of duplicated DP, but drags the barycenter objective (squared local cost, unbanded)
   into the variant dispatch that `barycenter.cpp:131-141` currently rejects outright.

9. **What binds a TADPole `dc` to the ledger it reports?** D18 makes `pruned_fraction()`
   path-dependent on whether `tadpole_auto_dc` ran first — which `Problem::cluster()` does
   (`Problem.cpp:1132-1135`). Is the ledger a *per-call* or a *per-Problem* quantity?

10. **[not established]** Whether IPO/LTO actually inlines `Problem::dist_by_ind` into the
    algorithm TUs in the canonical build. `cmake/ProjectOptions.cmake:21` defaults
    `dtwc_ENABLE_IPO` to `PROJECT_IS_TOP_LEVEL` and
    `cmake/InterproceduralOptimization.cmake:5` sets `CMAKE_INTERPROCEDURAL_OPTIMIZATION ON`, but
    nothing in this pass confirms the resulting inlining. Since `dist_by_ind` is the single
    distance oracle for six of the eight algorithms and every MIP backend, this is the largest
    unknown behind any performance reasoning about the SWAP kernel — and it interacts directly with
    P4 (the named-critical/LTO constraint), so an "IPO off" build is not a free experiment.

11. **Should `k == 1` / `k == N` be a shared precondition layer (D-I) or a per-algorithm branch?**
    `lagrangian_root_exact` deliberately handles neither, arguing the B&B certifies them at the
    root (`lagrangian_root.cpp:718-719`). Four other sites hand-code them. One of the two positions
    is redundant work.
