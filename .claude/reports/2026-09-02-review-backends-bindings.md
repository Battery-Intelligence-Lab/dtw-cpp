# Read-only audit: solver backends, accelerators, bindings — 2026-09-02

Scope: `dtwc/mip/`, `dtwc/cuda/`, `dtwc/metal/` (parity only), `python/src/_dtwcpp_core.cpp` + `python/dtwcpp/*.py`, `bindings/matlab/`.
No file modified, nothing built, no tests run.

**Provenance.** MIP and Python findings were read directly by me. CUDA/Metal and MATLAB findings come from two
delegated read-only reviewers with exact line citations; I did not independently re-open every cited line.
Those are marked `[del]`.

---

## (A) Bugs

### A1. Benders reports a possibly-suboptimal answer as complete — HIGH — `dtwc/mip/benders.cpp:282-406,430`
The cut loop `for (iter = 0; iter < max_benders_iter; ...)` has **no post-loop convergence check**. When the
iteration cap (default 200, `Problem.hpp:76`) is exhausted without the gap closing, control falls to line 414 and
prints `"Benders decomposition complete: cost = ..."`. Scenario: N=500, k=10, slow cut convergence → user selects
`Method::MIP` (exact), auto-dispatch sends N>200 to Benders (`Problem.cpp:1064`), and receives a heuristic PAM-quality
answer labelled complete, with no exception and no status flag. Contrast `mip_Highs.cpp:264` and `mip_Gurobi.cpp:120`,
which both `throw SolverError` on non-optimality. **Observed.**

### A2. Benders lower bound is not a lower bound — HIGH — `dtwc/mip/benders.cpp:286-299,343`
The master accepts `kObjectiveBound` and `kSolutionLimit` as success, and `mip_rel_gap` is set on the master
(line 247). `theta_sum` is then the **incumbent** objective of an early-terminated master, not its dual bound, yet it
is used as the LB in `bound_gap = best_cost - theta_sum` (line 340). Scenario: master stops at 1e-5 relative gap;
`theta_sum` sits above the true master optimum; the test at line 343 fires and Benders declares convergence while a
strictly better medoid set exists. The correct quantity is `highs.getInfo().mip_dual_bound`. **Observed** (code);
**Inferred** (HiGHS status semantics).

### A3. Benders tolerance is absolute and unscaled — MEDIUM-HIGH — `dtwc/mip/benders.cpp:280,343,372`
`abs_eps = 1e-6` is compared against raw, **unscaled** distance sums, while `mip_Highs.cpp:137` and
`mip_Gurobi.cpp:74` both divide the objective by `max(max_distance()/2, 1)`. Scenario A: distances ~1e6 (unnormalised
power/energy series) → the cut-skip test at line 372 (`d_nearest <= theta_j + abs_eps`) becomes vacuous and every
point regenerates a cut every iteration. Scenario B: distances ~1e-8 → line 372 suppresses **all** cuts while
`bound_gap` (up to `N*abs_eps`) still exceeds the tolerance, so the loop re-solves an unchanged master for all 200
iterations doing nothing. **Observed** (constant + comparisons); **Inferred** (regimes).

### A4. `LR_core_clustering` publishes an uncertified result and skips all validation — HIGH — `dtwc/mip/lagrangian_root.cpp:696-712`
`lagrangian_root_exact` correctly sets `certified_optimal=false` when `max_nodes` is hit (line 631) and warns on
stderr. `LR_core_clustering` **ignores that flag** and writes `prob.centroids_ind = r.medoids` directly. It also
does not use `mip::ExactClusteringTransaction` / `extract_exact_clustering`, so unlike HiGHS/Gurobi there is no check
that `r.medoids.size() == k`, no duplicate-medoid check, and no self-assignment check. If `r.medoids` is empty,
`centroids_ind` is left empty while `clusters_ind` is all-zero — downstream `silhouette` / `centroid_of` then index
an empty vector. `mip.hpp:21` calls this the "EXACT clustering entry point". **Observed.**

### A5. Negative `remaining_need` → `size_t` underflow OOB read — MEDIUM — `dtwc/mip/lagrangian_root.cpp:543,581,594-596`
`need = k - forced.size()` is never checked for `>= 0`. If reduced-cost fixing over-fixes (possible when the
LB/UB pair is inconsistent, e.g. a stale UB), `need < 0`; the leaf test `remaining_need == 0` never fires, the
guard `rem < remaining_need` is false for negative values, and line 595 evaluates
`csum[static_cast<std::size_t>(fr.pos) + static_cast<std::size_t>(remaining_need)]` — a wrapped index far past the
end of `csum`. **Observed** (missing guard); **Inferred** (trigger).

### A6. `pmedian_local_search` returns cost/labels for a different medoid set — MEDIUM — `dtwc/mip/lagrangian_root.cpp:60-96`
The medoid update (lines 77-92) runs **after** the cost/label computation of the same sweep. When the loop exits by
exhausting `max_sweeps=32` with `changed == true`, the returned `cost` and `labels` describe the pre-update medoids
while `medoids` holds the post-update set. `try_primal` (line 178) and `finalize` (line 198) then store
`best_labels` containing point indices that are **not members of `best_medoids`**. The UB stays valid (Lloyd is
monotone), so this is a label-consistency bug, not a bound bug. **Observed.**

### A7. Warm-start indices used unchecked — MEDIUM — `mip_Highs.cpp:241-248`, `mip_Gurobi.cpp:99-106`
`pam_result.medoid_indices[pam_result.labels[j]]` and `sol.col_value[med*(Nb+1)]` use `operator[]` with no bound
check. A FastPAM result with fewer than k medoids or an out-of-range label writes out of bounds into a
`Nb*Nb` vector. `solution_transaction.cpp` validates the *output* thoroughly; the *input* warm start is unvalidated.
**Observed.**

### A8. HighsInt truncation is silent — MEDIUM — `mip_Highs.cpp:128`, `pdlp_lp.cpp:69-70,94-95`
`num_col_ = Nvar` where `Nvar = Nb*Nb` (size_t), and `static_cast<HighsInt>(nvar)` with `nvar = N*N`. `HighsInt` is
int32 in a default HiGHS build, so N > 46340 silently truncates to a wrong-sized model rather than erroring.
`extract_exact_clustering` guards `n_points > INT_MAX` (`solution_transaction.cpp:93`) but not `N*N`. **Observed.**

### A9. Every `setOptionValue` return status is discarded — MEDIUM — `pdlp_lp.cpp:145-148`, `mip_Highs.cpp:222-226`, `benders.cpp:244-251`
`highs.setOptionValue("solver", params.variant)` takes an unvalidated user string (`pdlp_lp.hpp:44`). A typo
(`"PDLP"`) or an option absent from the linked HiGHS version (`"kkt_tolerance"`) returns `kError`, is ignored, and
the solve proceeds on the **default simplex solver** — while `PdlpResult::iterations` reads `pdlp_iteration_count`
(0) and the result is still reported as a PDLP arbiter value. That silently invalidates the LR-core cross-check.
**Observed.**

### A10. Raw `new` leaks / unbounded copy in the Python binding — MEDIUM — `_dtwcpp_core.cpp:991`, `1351-1353`, `1437-1438`
`double *ptr = new double[n*n]()` is only handed to a `nb::capsule` at line 1022. Anything throwing in between
(`compute_distance_matrix_pruned` on a NaN series) leaks the whole N² buffer. In the CUDA/Metal lambdas
`std::copy(result.matrix.begin(), result.matrix.end(), data)` copies `result.matrix.size()` elements into a
`n*n` buffer with **no size assertion**. **Observed.**

### A11. Exceptions can escape an OpenMP region — MEDIUM — `_dtwcpp_core.cpp:1007-1018`
`#pragma omp parallel for` around `dtwc::dtwBanded` / `dtwFull_L`, both of which can throw on NaN input. An
exception escaping a parallel region is undefined behaviour and terminates the process — from Python this is a
hard interpreter crash rather than an `InvalidInput`. **Observed** (structure); **Inferred** (throw path).

### A12. `dtwc.cluster(..., 'method','mip')` silently ignores `k` — CRITICAL `[del]` — `bindings/matlab/+dtwc/cluster.m:75-78`
The MIP branch calls `prob.set_method('mip'); prob.cluster();` without `prob.set_n_clusters(k)`. `Problem::Nc`
defaults to 1 (`Problem.hpp:122`), so MATLAB MIP clustering returns one cluster for any requested k. Every sibling
branch passes k. **Observed by delegate.**

### A13. Unchecked column count in `mx_to_dendrogram` — CRITICAL `[del]` — `bindings/matlab/dtwc_mex.cpp:369-377`
Reads `data[i + 3*n_merges]` after checking only `mxGetM`; no `mxGetN == 4` check. An Nx3 or transposed `merges`
field over-reads the heap. `cut_dendrogram` has no MATLAB test. **Observed by delegate.**

### A14. Non-finite MATLAB labels → UB float→int — CRITICAL `[del]` — `bindings/matlab/dtwc_mex.cpp:123-136, 927, 1272, 1293`
`require_label_vector` checks class/complex/sparse/empty but not finiteness; `static_cast<int>(p[i] - 1)` on
NaN/Inf is UB. The file already has `get_exact_int` (line 255) doing exactly this check, unused here.
**Observed by delegate.**

### A15. `num_pairs` truncated to `int` before the LB pre-pass — CRITICAL `[del]` — `dtwc/cuda/cuda_dtw.cu:1347, 1405, 1599`
`launch_dtw_kernel` guards `num_pairs > INT_MAX` at `:1136`, but the LB_Keogh pre-pass runs first and is unguarded,
and `compute_lb_keogh_cuda` has no guard at all. N ≥ 65536 truncates (possibly negative) and the in-kernel
`pid >= num_pairs` test then drops or over-runs work before the later throw can fire. Directly relevant to the
100M-series target. **Observed by delegate.**

### A16. CUDA returns an all-zero matrix when no device is present — HIGH `[del]` — `dtwc/cuda/cuda_dtw.cu:1445, 2277, 2342, 2405`
`!cuda_available()` returns N×N zeros with `kernel_used="none"` — a valid-looking distance matrix. `Problem.cpp:944`
guards, but `_dtwcpp_core.cpp:1349` and `test_api.hpp:213` call the API directly. Violates the project's
no-silent-fallback rule; the adjacent `cudaSetDevice` failure is loud. **Observed by delegate.**

### A17. `decode_pair` corrects the row in one direction only — HIGH `[del]` — `dtwc/detail/decode_pair.hpp:68-71`
The C++/CUDA variant has an up-only `while` loop; its MSL sibling at `:106-107` corrects both directions and
documents that an up-only loop "cannot recover from an overestimate" of the FP64 seed. Latent: no triggering input
found, but the two SSOT variants disagree. **Inferred by delegate.**

---

## (B) Duplication

- **B1. Three incompatible "run a heuristic without publishing it" mechanisms.** `mip::make_warm_start` +
  `ExactClusteringTransaction` (`warm_start.cpp:60-66`, used by HiGHS/Gurobi); a hand-rolled `ScopeExit` snapshot of
  five `Problem` fields (`benders.cpp:154-179`); and **no protection at all** in
  `prepare_dense_D` (`lagrangian_root.cpp:659`), which calls `cluster_by_kmedoids_lloyd()` and permanently clobbers
  the caller's `centroids_ind`/`clusters_ind` — a visible side effect on the bound-only public entry point
  `lagrangian_root(Problem&)` (line 676).
- **B2. Nearest-medoid assignment written seven times.** `benders.cpp:267-274, 319-327, 417-428`;
  `lagrangian_root.cpp:63-74, 158-168, 555-559, 614-622, 703-712`. Each is a nested min-scan with its own tie-break.
- **B3. The Balinski model is built twice with mirrored index conventions.** `mip_Highs.cpp:139-195` uses
  facility-major (`facility*Nb + point`); `mip_Gurobi.cpp:52-79` uses point-major. `AssignmentMatrixLayout`
  (`solution_transaction.hpp:21-24`) papers over it correctly, but the two objective loops assign `d(point,facility)`
  and `d(facility,point)` respectively — equivalent only while D is symmetric. A third copy of the same LP exists in
  `pdlp_lp.cpp:104-140`.
- **B4. CUDA wavefront config duplicated verbatim** `[del]` — `cuda_dtw.cu:1221-1250` vs `2196-2216`; the
  "scan lengths for max_L" loop appears five times (`:1449, 1591, 2286, 2345, 2418`).
- **B5. MATLAB deprecated aliases re-validate what the MEX layer already validates** `[del]`
  (`adjusted_rand_index.m:28-30` vs `dtwc_mex.cpp:1264-1268`), while the canonical names validate nothing.

## (C) Simplifications / error-prone constructs

- **C1. `MIPSettings::benders` is a string** (`Problem.hpp:77`, dispatched at `Problem.cpp:1064`). `"On"`, `"true"`,
  or any typo silently means off. Every sibling selector in the file is an enum with a `validate_*` function
  (`DistanceMatrixStrategy`, `Solver`, `Method`). Replace with `enum class BendersMode { Auto, On, Off }`.
- **C2. Magic tolerances.** `benders.cpp:280` `abs_eps=1e-6`; `benders.cpp:220,393` `1e20` instead of `kHighsInf`;
  `lagrangian_root.cpp:566` `tol` computed once from the initial incumbent and never refreshed;
  `reduced_cost_fixing.cpp:68` `1e-9`. Only the last is documented and magnitude-scaled. Note that
  `lagrangian_root_exact` prunes at `node_lb >= best_cost - tol` (line 597) yet still reports `gap = 0.0` and
  `certified_optimal = true` (lines 638-640) — the certificate is epsilon-optimal, not exact.
- **C3. All `Nb*Nb` variables are declared integer** (`mip_Highs.cpp:216`). `design.md:71` states the assignment
  block is TU once the medoid variables are fixed, so only the `Nb` diagonal columns need `kInteger`. Gurobi already
  exploits this via branch priority (`mip_Gurobi.cpp:47-50`) but still declares all vars binary. Perf win, no
  correctness risk; needs a benchmark before claiming a number.
- **C4. `evaluate_dual` uses `nth_element` without an index tie-break** (`lagrangian_root.cpp:121-123`) while
  `reduced_cost_fixing.cpp:51-54` uses one. Self-consistent today, but the determinism guarantee is asymmetric.
- **C5. MATLAB integer validation is inconsistent** `[del]` — `get_exact_int` exists (`dtwc_mex.cpp:255`) but is
  used only for CUDA `device_id`; band/max_iter/k/fast_pam all use bare `static_cast<int>(get_scalar(...))`.
  Same for `require_char` before `get_string` (used for 4 setters, omitted for 4 others).
- **C6. `GPUConfig::max_shared_per_block` is computed and never read** `[del]` (`gpu_config.cuh:72-75`), so an
  over-large shared-memory request surfaces as a raw CUDA error (`cuda_dtw.cu:1240-1250`).

## (D) Dead / obsolete code

- `benders.hpp:26-27` documents "prints a diagnostic and returns without modifying `prob` if HiGHS is not compiled
  in" — the code throws `SolverError` (`benders.cpp:434`). Stale doc.
- `mip_Gurobi.cpp` uses `std::cout` (line 110) and `std::max` (line 74) without including `<iostream>` or
  `<algorithm>`; includes `<limits>` and `<string_view>` that are unused.
- `benders.cpp:74` includes `<cassert>`; no assert in the file.
- `cuda_dtw.cu:34,36,39` include `<chrono>`, `<numeric>`, `<climits>` — unused `[del]`.
- `pdlp_lp.cpp:157` computes `gpu_used` before `passModel`; `params.use_gpu` governs only a warning — documented,
  but the flag is otherwise inert.
- MATLAB: `dtwc.CheckpointOptions` is consumed by nothing `[del]`; `cmd_cluster_legacy` has an unreachable
  `prhs[4]` slot and no `.m` wrapper (`dtwc_mex.cpp:1310-1317`); `Problem_get_band` is in the dispatch table
  (`:1400`) and called by no `.m` file.
- **Design-rule check passes:** no root-level `dtwc.dtw_distance` in MATLAB; Python keeps `dtw_distance` private as
  `_dtw_distance_raw` (`python/dtwcpp/__init__.py:55`) and `__all__` exposes only `distance.*`.

## (E) Missing / forgotten

- **E1. LR-core ignores every `MIPSettings` knob.** `lagrangian_root.cpp:696` calls `lagrangian_root_exact(D, N, k, ub)`
  with **default** `LagrangianParams`. `mip_gap`, `time_limit_sec`, `warm_start`, `verbose_solver` and the
  `Solver` selection all have no effect on `Method::LRCore`; `max_nodes`, `rel_gap_tol`, `kelley_max_major` are
  unreachable from C++ `Problem`, Python, MATLAB, or the CLI.
- **E2. `pdlp_lp_bound`, `lagrangian_root`, `reduced_cost_fixing` are not exposed in any binding.** The PDLP arbiter
  and the LR bound are C++-only; `pdlp_gpu_available()` is likewise unbound, so the Python `HIGHS_AVAILABLE`
  capability set has no GPU-LP counterpart.
- **E3. Benders ignores `prob.random_seed()`** — it uses `cluster_by_kmedoids_lloyd_impl(false)` rather than the
  seeded `mip::make_warm_start`, so its warm start is not reproducible the way HiGHS/Gurobi's is.
- **E4. Benders prints unconditionally to `std::cout`** (lines 115, 181, 255, 289, 310, 344, 410, 430) regardless of
  `verbose_solver` / `prob.verbose()`; both other backends gate their output.
- **E5. CUDA silently ignores `max_length_hint`** `[del]` (declared `core/gpu_dtw_common.hpp:45`, re-advertised
  `cuda_dtw.cuh:58`, zero uses in `dtwc/cuda/`); Metal honours it (`metal_dtw.mm:1317-1319`). Metal ignores
  `kernel_override` in its K-vs-N path; CUDA honours it (`cuda_dtw.cu:2432`). Metal has no FP64 path and silently
  downgrades to FP32 (`metal_dtw.mm:1263-1265`).
- **E6. Python/MATLAB parity gaps** `[del]`: absent from MATLAB — `one_batch_pam(_with_stats)`, `dtw_barycenter`,
  `barycenter_kmeans`, `data_from_arrow_c_array`, `save_/load_dataset_{csv,hdf5,parquet}`, submodules
  `preprocess`/`diagnose`/`features`/`io`, `sklearn.DTWCKMedoids`, `Env`/`env`, `fast_pam_seeded` by name.
  Nothing functional is MATLAB-only.
- **E7. Test coverage holes.** No test references `extract_exact_clustering`, `ExactClusteringTransaction`, or
  `make_warm_start` outside `unit_test_mip.cpp` and one selector test; `lagrangian_root_kelley` and
  `LR_core_clustering` have no direct test. `unit_test_benders.cpp` has a "convergence within iteration limit" case
  but nothing asserting behaviour **past** the limit (A1). CUDA device tests are bare `SKIP` with no
  `FAIL_REGULAR_EXPRESSION` gate, so a GPU-less host reports the whole CUDA suite green — the finding-F9 pattern
  recurring `[del]`. MATLAB `cut_dendrogram`, `clarans`, `fast_clara`, binary-checkpoint round-trip and the mmap
  path are untested `[del]`.

## (G) Lock-free / concurrency & global-state audit

Constraint checked: no host-side mutex, atomic, or mutable global state on any orchestration or binding path;
no serialisation of parallel work.

### G1. `query_gpu_config` takes a process-global mutex on **every** call, including cache hits — HIGH — `dtwc/cuda/gpu_config.cuh:44,50`
```
static std::mutex mtx;   ... 
std::lock_guard<std::mutex> lock(mtx);
if (initialized[device_id]) return configs[device_id];
```
The lock is acquired **before** the `initialized` test, so the steady-state cached path serialises too. It is called
once per kernel launch (`cuda_dtw.cu:1251`) and from `prefer_fp32` (`cuda_dtw.cu:83`). The `thread_local`
workspaces at `cuda_dtw.cu:966` and `:2072` show multi-threaded host dispatch is the intended model — so every host
thread's every launch funnels through one mutex. **Observed.** Smallest lock-free fix: cache the `GPUConfig` in the
`thread_local` workspace, or make `initialized[]` a `std::atomic<bool>` with acquire/release and keep the mutex only
for the one-time fill (double-checked, lock-free on the hit path).

### G2. Racy `static bool logged` warning latches — MEDIUM — `dtwc/cuda/cuda_dtw.cu:1230-1232`, `:2203-2205`
`static bool logged = false; if (!logged) { logged = true; ... }` is a non-atomic read-modify-write of
process-global state from any launching host thread. Formally a data race (UB); practically a duplicated or lost
warning. **Observed.** Fix: `static std::atomic<bool>` with `exchange(true)`, or `std::call_once`. No hot-path cost —
the block is already gated behind `max_L > 2048`.

### G3. MATLAB handle table is unsynchronised process-global mutable state — MEDIUM `[del]` — `bindings/matlab/dtwc_mex.cpp:57, 1347`
`static std::unordered_map<uint64_t, std::shared_ptr<T>> map_` plus `static bool first_call`. Safe today only
because MATLAB drives MEX from one thread; nothing in the file documents or enforces that invariant, and any
threaded caller corrupts the map. Related: `mexLock()` is called once (`:1357`) and never unlocked, so the lock
count does not track live handles as the file header claims.

### G4. GIL held across long C++ calls — MEDIUM — `python/src/_dtwcpp_core.cpp`
Most heavy entry points do release (`fill_distance_matrix` :918, `cluster` :952, `fast_pam` :1039, CLARA :1081,
barycenter, `clarans`, `build_dendrogram`, `cut_dendrogram`, CUDA/Metal). These do **not**:
`find_total_cost` (:955) and `dist_by_ind` (:880) — both reach `Problem::dist_by_ind`, which computes DTW **lazily**
(`Problem.cpp:718-721`), so a single Python call can run O(N) DTW kernels with the GIL held;
`read_distance_matrix` (:942), `write_distance_matrix` (:968), `write_clusters` (:966), `write_silhouettes` (:969),
`print_distance_matrix` (:944) — N² file I/O under the GIL; `save_checkpoint`/`load_checkpoint` (:1167,:1172) —
bound as raw function pointers with no release, while their `*_binary_*` siblings (:1178,:1189) **do** release;
`DenseDistanceMatrix.to_numpy` (:515-528) — packed→dense N² expansion plus memcpy under the GIL. This blocks every
other Python thread in the process for the duration. **Observed.**

### G5. Nested `gil_scoped_acquire` inside a live `gil_scoped_release` — LOW — `_dtwcpp_core.cpp:1349-1355`, `:1371-1377`, `:1434-1441`
The returned `nb::ndarray` is built under the inner `acquire`, then survives that object's destructor (GIL dropped)
until the outer `release` destructor re-takes it. No refcount operation happens in that window today, so it is
correct — but it is a fragile shape; closing the release in its own block before allocating is equivalent and
obviously safe. **Observed.**

### G6. Redundant per-element dispatch in the Benders hot loop — MEDIUM (perf, not a lock) — `dtwc/mip/benders.cpp:322, 383, 421`
Benders calls `prob.dist_by_ind(j, i)` O(N²) times **per cut iteration** (line 383 is inside a double loop inside
the iteration loop). Each call runs `preflight_current_distance_semantics()`,
`validate_mmap_cache_identity()`, `ensure_dense_cache_configuration_current()` and three `std::variant` visits
(`Problem.cpp:673-708`) before returning an already-computed value. `prepare_dense_D`
(`lagrangian_root.cpp:667-671`) shows the right pattern — materialise once, then index raw memory. **Observed.**
The `omp critical(distByInd_init)` at `Problem.cpp:690` is correctly one-shot (guarded by `needs_init`), so the
steady state is lock-free; the cost here is dispatch, not synchronisation.

### G7. Conforming — no action
`dtwc/mip/` contains **no** mutex, atomic, `thread_local`, or mutable global state (grep clean across all 16 files).
The two OpenMP regions in `lagrangian_root.cpp:107-118` and `:133-144` are lock-free by construction — iteration `i`
writes only `rho[i]`, iteration `j` only `g[j]`; no `reduction`, `critical`, or `atomic` clause. The O(N) scalar
reductions at `:126-129`, `:292-293`, `:311-321` are deliberately left serial, which is right next to the O(N²)
parallel loop. The CUDA memory deleters (`cuda_memory.cuh:33,57,96,117`) swallow errors rather than throwing from
destructors — correct. Metal's `std::call_once` context init (`metal_dtw.mm:1018-1019`) is one-shot process setup,
not a per-dispatch lock.

---

## (F) Top 5 recommended actions

1. **(S) Make Benders honest.** Read the LB from `getInfo().mip_dual_bound`; reject `kObjectiveBound`/`kSolutionLimit`
   as master success; add a post-loop check that throws `SolverError` (or sets a documented status) when
   `max_benders_iter` is exhausted without convergence; route output through `verbose_solver`. Fixes A1, A2, E4.
2. **(S) Route LR-core through the transaction.** Have `LR_core_clustering` build a `core::ClusteringResult`,
   check `r.certified_optimal`, and publish via `mip::ExactClusteringTransaction`; add a `need >= 0` guard at
   `lagrangian_root.cpp:543`. Fixes A4, A5 and removes the third warm-start mechanism (B1).
3. **(S) MATLAB `cluster.m` MIP branch: pass `k`.** One line (`prob.set_n_clusters(k)`), plus a regression test
   asserting `numel(unique(labels)) == k` for the MIP path. Fixes A12 — currently a wrong-answer bug on a
   documented public entry point.
4. **(S) Guard `num_pairs` before the CUDA LB pre-pass** and make `!cuda_available()` throw `DeviceError` instead of
   returning zeros. Fixes A15, A16. Perf risk: none. Add the CUDA suite to the ctest skip-detection gate so the
   fix is actually exercised.
5. **(S) Remove the mutex from the CUDA launch path (G1).** Cache `GPUConfig` in the existing `thread_local`
   workspace, or make `initialized[]` `std::atomic<bool>` with a lock-free double-checked read; switch the two
   `static bool logged` latches to `std::atomic<bool>::exchange` (G2). Pure win — removes serialisation, adds none.

**Perf-risk labelling of every proposal above (constraint: lock-free, no added serialisation):**

| Proposal | Adds lock / serialises? | Note |
|---|---|---|
| F1 Benders status + `mip_dual_bound` | No | Outside all loops; `getInfo()` is a struct read. |
| F2 LR-core transaction + `need >= 0` guard | No | `ExactClusteringTransaction` is two `vector::swap`s, no synchronisation. |
| F3 MATLAB `set_n_clusters(k)` | No | One call. |
| F4 CUDA `num_pairs` guard + throw on no device | No | Host-side comparison before launch. |
| F5 CUDA config caching (G1/G2) | **Removes** a lock | Must be done with atomics/`thread_local`, never by widening the mutex. |
| B2 shared `assign_to_nearest_medoid` | No | **Perf-risky by inlining, not by locking**: called every LR subgradient iteration (`try_primal`). Keep it a header-inline free function on a raw `const double*`; do not make it virtual, do not take a `Problem&`, do not add an OpenMP `reduction` (the loop is O(Nk), already inside a parallel-friendly caller). Re-benchmark `lagrangian_root` before merging. |
| A7 `validate_warm_start` | No | O(k) once per solve. |
| A10 `unique_ptr` instead of raw `new` | No | Zero cost. |
| A11 exceptions escaping OpenMP | **Must not** use `omp critical` | Lock-free fix: each thread writes a failure index to its **own** slot in a pre-sized `vector<int>`; inspect after the region. A `critical` section around a shared error flag would serialise the hot loop and is the wrong fix. |
| C1 `benders` string → enum | No | Removes a string compare per dispatch. |
| C3 relax off-diagonal integrality | No | Strictly reduces solver work; still benchmark before quoting a factor. |
| G4 release the GIL on the remaining defs | No | **Improves** Python-level concurrency; check each callee is genuinely thread-safe w.r.t. `Problem` first — `dist_by_ind` mutates `distMat`, so releasing the GIL there means concurrent Python threads on the same `Problem` race. Release it on the I/O and checkpoint defs unconditionally; for `find_total_cost`/`dist_by_ind` the correct fix is to prime the matrix, not to drop the GIL over a mutating call. |
| G6 hoist `dist_by_ind` out of the Benders loop | No | Reuse `prepare_dense_D`'s materialise-once pattern; the existing `omp critical(distByInd_init)` then never runs at all. |

**Deliberately not raised:** formatter nits; the `scaling_factor` role divergence between backends (correct while D is
symmetric — noted under B3 as a latent constraint, not a bug).
