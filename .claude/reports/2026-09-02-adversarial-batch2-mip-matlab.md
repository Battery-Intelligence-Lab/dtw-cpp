# Adversarial review — batch 2: MIP solver layer + MATLAB bindings (2026-09-02)

Read-only. Line numbers are the NEW (working-tree) files.

## Ranked defects

### 1. HIGH — the scaled `abs_eps` was also applied to the Benders cut-coefficient filter, making the cuts invalid
`dtwc/mip/benders.cpp:371` — `if (coeff > abs_eps)` now drops coefficients up to
`1e-6 * max(max_distance()/2, 1)` instead of `1e-6`.
Observed: the cut is `theta_j + sum_i c_i y_i >= d_nearest`, `c_i = max(0, d_nearest - d_ji) >= 0`.
Dropping a positive `c_i` *shrinks the LHS*, i.e. makes the constraint **stricter** than the valid
Benders cut. Inferred: the master's `mip_dual_bound` is then inflated by up to `N * abs_eps` and can
exceed the true p-median optimum, so the optimum can be cut off and the new `converged` branch
(`:328`) can certify a suboptimal incumbent — under a message that now claims optimality was
*proved*. Pre-fix the error was bounded by `1e-6 * N`; today's change multiplies it by
`max_distance()/2` (on DTW distances routinely 1e3–1e7).
Minimal fix: keep the coefficient filter unscaled and relative to the cut itself, e.g.
`if (coeff > 1e-12 * std::max(1.0, d_nearest))`, and scale only the two tests the claim names
(`:328` convergence, `:358` cut-skip).

### 2. HIGH — `Method::MIP` now throws by default on exactly the size range where Benders auto-engages
`dtwc/Problem.cpp:1115` selects Benders for `benders == "auto" && N > 200`; defaults are
`max_benders_iter = 200`, `mip_gap = 1e-5` (`dtwc/Problem.hpp:69,75,76`); `benders.cpp:399-408`
now converts an exhausted cut loop into `SolverError`.
Scenario: any `N > 200` instance on default settings — the historical, documented default route —
where 200 disaggregated-cut rounds do not close a 1e-5 relative gap. Previously it printed a
diagnostic and returned the PAM-quality incumbent; now it aborts the call. Defect 1 makes this
*more* likely (a looser cut-skip test at `:358` emits fewer cuts per round).
Observed: no new or existing test exercises `N > 200`; the largest case in
`tests/unit/mip/test_mip_backend_guards.cpp` is `N = 12`, and A1 (`:90-100`) forces the throw with
`max_benders_iter = 1`, which says nothing about the default.
Minimal fix: raise the default cap (or derive it from `N`) and/or record the unconverged state on
the result and throw only when the caller asked for a certificate — plus one `N = 250`
default-settings regression case.

### 3. HIGH — the LR-core throw is not actionable: there is no node-cap knob
`dtwc/mip/lagrangian_root.cpp:721` calls `lagrangian_root_exact(D.data(), N, k, ub)` with a
**default-constructed** `LagrangianParams` (`max_nodes = 2000000`, `lagrangian_root.hpp:52`).
`MIPSettings` (`Problem.hpp:68-77`) has no LR fields. The new message
(`lagrangian_root.cpp:735-739`) tells the user to "raise the node cap", which `Method::LRCore` gives
them no way to do — the method becomes unusable on any capped instance.
Minimal fix: add `long lr_max_nodes` (or a `LagrangianParams` member) to `MIPSettings` and forward it
at `:721`; then the message is actionable and A4 gets a discriminating test (see §12).

### 4. MED — Gurobi got no index guard, contrary to the claim
`dtwc/mip/mip_Gurobi.cpp:45` `model.addVars(Nb * Nb, GRB_BINARY)` narrows `std::size_t` to `int`;
`:125` `std::vector<double> solution(Nb * Nb)`. The diff to this file is includes only. So
`N > 46340` still builds a wrong-sized Gurobi model silently, while HiGHS/PDLP now refuse.
Minimal fix: a backend-neutral `require_index_range(Nb * Nb, INT_MAX, "column count N*N", "Gurobi")`
before `addVars`.

### 5. MED — the HiGHS guard checks `HighsInt` but the code casts to `int`
`dtwc/mip/mip_Highs.cpp:64-67,105` bound the dimensions by `numeric_limits<HighsInt>::max()`, but the
triplets are built with `static_cast<int>` (`:120,123,128-135`). On a `HIGHSINT64` build the guard's
limit is 2^63-1, so an `N > 46340` model passes and then truncates in the `int` triplet fields — the
exact failure the guard was added to prevent.
Minimal fix: guard against `std::min<std::size_t>(HighsInt max, INT_MAX)`.

### 6. MED — `set_highs_option` turns version skew and a bad user setting into hard failures on default paths
`dtwc/mip/pdlp_lp.cpp:161` sets `kkt_tolerance`, which is absent from older HiGHS builds → `kError`
→ `pdlp_lp_bound` now throws on a build where it previously produced a (correctly-solved) bound.
`dtwc/mip/mip_Highs.cpp:166` passes `prob.mip_settings.mip_gap` unconditionally; a negative value is
outside the option domain → `kError` → the compact backend throws a HiGHS-worded error instead of an
input error.
(The `kWarning`-vs-`kError` split itself is correct — HiGHS uses `kWarning` for deprecated-but-honoured
names and clamped values.)
Minimal fix: validate `mip_gap >= 0` in `MIPSettings` (`InvalidInput`); treat `kkt_tolerance` as
best-effort (probe `getOptionValue` first, or warn) rather than fatal.

### 7. MED — the stated parity rationale for scaling `abs_eps` is not supported by the compact backend
`dtwc/mip/benders.cpp:229-233` and CHANGELOG:19 claim the scaling matches "the same scaling the
compact HiGHS backend applies to its [objective] before applying a 1e-6 tolerance".
Observed: `mip_Highs.cpp:78` divides the objective by `scaling_factor` purely for conditioning; there
is **no** `1e-6` anywhere in `mip_Highs.cpp`, and its only tolerances (`mip_rel_gap`, the `> 0.5`
binary threshold in `solution_transaction.cpp:110,131`) are scale-invariant. Scaling a Benders
tolerance is defensible on its own terms, but the justification as written is wrong, and it is what
licensed defect 1.

### 8. LOW-MED — Benders' own warm start is still cardinality-unvalidated
`benders.cpp:129-133` deliberately bypasses `mip::make_warm_start`; `:243-250` validates only that
each medoid is in `[0, N)`. If the nested Lloyd returns fewer than `Nc` medoids, `best_medoids` keeps
that size and the run prints "converged" and then fails inside `publish` with an
"expected k medoids but decoded …" `SolverError`. The claim "warm-start validated in
`make_warm_start`" does not cover this backend.
Minimal fix: also check `best_medoids.size() == Nc` at `:243`.

### 9. LOW — `total_cost` is written and silently discarded
`benders.cpp:417` and `lagrangian_root.cpp:741` set `result.total_cost`;
`ExactClusteringTransaction::publish` (`solution_transaction.cpp:166-172`) swaps only
`medoid_indices` and `labels`. Dead stores that read as if the cost is published.

### 10. LOW — `exact_int_from_double` reintroduces signed overflow at its own boundary
`bindings/matlab/dtwc_mex.cpp:258-269` accepts `value == INT_MIN` (`int_min` is exactly representable),
and `label_vector_to_0based:288,290` then evaluates `INT_MIN - 1` — signed overflow, UB, in the helper
added to eliminate UB. Both branches (int32 and double) have it.
Checked and sound: `-0.0` → `0` (`floor(-0.0) == -0.0`, cast → `0`); `2^53` is `> int_max` so rejected;
`int_max` is exactly representable so there is no off-by-one at the top.
Minimal fix: reject `value <= int_min` in the 1-based path, or subtract before the conversion.

### 11. LOW — the MATLAB tests cannot run in CI, and the MIP one cannot run here at all
`tests/CMakeLists.txt:1` globs `*.cpp` only; `tests/matlab/test_cluster_mip.m` and
`test_pdlp_binding.m` are registered nowhere (same as the pre-existing `test_mex_input_validation.m`).
`.claude/LESSONS.md:225-232` records `[BLOCKED-ENV]`: every recorded MEX build is HiGHS-OFF and a
HiGHS-enabled MEX crashes MATLAB on this box — so `test_cluster_mip.m`'s capability probe always takes
the `assumeTrue` skip, and the A12 fix (`+dtwc/cluster.m:76`) has **never been executed**. That is
honestly guarded in the test, but CHANGELOG:37-38 reads as a verified fix.

### 12. Tests that are outcome locks (pass pre-fix), and a cheap discriminating replacement for each
- **A2** `test_mip_backend_guards.cpp:121-135`. Compares Benders cost to compact-MIP cost on `N=12,
  k=4`; the old `theta_sum` LB also converges to the same partition there. Discriminating: extract the
  LB read into `benders_master_lower_bound(Highs&)` and assert it equals `getInfo().mip_dual_bound`
  and is `<= sum(theta)` on a master deliberately stopped with `mip_rel_gap = 0.5`.
- **A3** `:140-156`. With the *old* absolute `1e-6` the same partition also results (a tighter
  tolerance only adds cuts more eagerly), so this passes pre-fix. Discriminating: expose
  `benders_abs_eps(double max_distance)` and unit-test `benders_abs_eps(2e6) == 1e-6 * 1e6` — three
  lines, no solver. It would also have surfaced defect 1, by forcing the author to name every site
  that consumes the value.
- **A4** `:161-179`. The deleted direct-write code already produced k unique medoids with in-range
  labels on this separable instance. Discriminating: with the knob from defect 3, set a node cap of 1
  on an instance that does not certify at the root and `REQUIRE_THROWS_AS(prob.cluster(), SolverError)`.
- Genuinely discriminating already: **A1** (`:90-100`), **A9** (`:201-219`), **B1** (`:184-196`), and
  the rollback case (`:102-116`).

### 13. CHANGELOG lines that overstate
- `:19` "the same scaling the compact HiGHS backend applies" — see §7.
- `:33-34` "reject an `N*N` model dimension that does not fit `HighsInt` instead of truncating it
  silently" — not true for Gurobi (§4) and incomplete for the `int` casts (§5).
- `:29-31` "the Lagrangian primal repair now returns a cost and labels that describe the medoid set it
  returns" — true only because `labels` hold *point* indices. `lagrangian_root.cpp:109` sorts `medoids`
  after the final `assign()`, leaving `cluster_of` (positions) stale, which contradicts the in-file
  comment at `:64-66` that claims `cluster_of` too.
- `:9-16` present the Benders throw as pure honesty; §2 shows it is also a default-path behaviour break
  that no test covers.

## Confirmed sound

- **`need < 0` guard** (`lagrangian_root.cpp:553-561`): correct, and the complementary `need > C` case
  is already handled by `rem < remaining_need` at `:608`. The message accurately names the cause.
- **No off-by-one in the Benders post-loop throw**: `converged` is set before the only success `break`
  (`:329`); the `failure_reason` breaks (`:277`, `:299`) leave it false; a completed loop leaves
  `iter == max_benders_iter` and false. `max_benders_iter <= 0` also throws, with an accurate message.
- **Transaction rollback**: `~ExactClusteringTransaction` is `noexcept` and swaps both vectors back
  (`solution_transaction.cpp:154-161`), so `Problem` is left exactly as the caller had it on every new
  Benders/LR throw; `test_mip_backend_guards.cpp:102-116` pins it. Constructing it *after* the
  trivial-`k` branches (`benders.cpp:120`) is deliberate and correct — those branches write directly
  and return.
- **`nearest_medoid` semantics** match all seven replaced scans: strict `<`, first-minimum tie-break,
  default position 0, `int` position width. The only difference is the sentinel — `benders.cpp`'s two
  former scans used `DBL_MAX`, the header uses `+inf`; with a non-finite distance the reported cost is
  now `inf` rather than a finite-looking `1.8e308`, which is strictly better. NaN behaves identically
  (`d < best` is false, keeps the incumbent), matching the old `std::min(best, d)` in `cost_of`.
  Header-inline template over the accessor, no `std::function`, no allocation — meets the hot-loop rule.
- **The eighth scan is correctly not replaced**: `lagrangian_root.cpp:153` counts `D[m][j] < mu_j` over
  `S_k`; it is a subgradient cardinality count, not an argmin.
- **`pmedian_local_search`**: the extra `assign()` (`:108`) runs at most once per call and is O(N·k),
  not per sweep; both exit paths now leave `cost` and `labels` consistent with the returned `medoids`.
- **`prepare_dense_D`'s seed transaction** (`:678-687`) correctly makes `lagrangian_root(Problem&)`
  side-effect-free and does not disturb the outer transaction in `LR_core_clustering`.
  `test_lagrangian_root.cpp:508-530` (the only `Method::LRCore` case) runs on instances that certify,
  so the new throw does not break it.
- **`set_highs_option`'s `kError`-only policy** is the right split; `kWarning` (deprecated-but-honoured
  names, clamped values) is correctly tolerated.
- **`mx_to_dendrogram`** (`dtwc_mex.cpp:392-411`): the 4-column check precedes every `data[...]` read,
  the empty single-point case is exempted correctly, and `mxGetDoubles` on an empty array is never
  dereferenced. `n_points` now goes through `get_exact_int`.
- **1-based shift applied exactly once** in `label_vector_to_0based` and in `mx_to_dendrogram`'s
  `cluster_a`/`cluster_b` (and correctly *not* to `new_size` or `distance`).
- **`FindMatlab` change** (`bindings/matlab/CMakeLists.txt:39-50`) is guarded `WIN32 AND NOT MSVC`, so
  the MSVC path is byte-identical; `LINKER:` yields the right driver syntax for the clang GNU frontend.
- **`cmd_pdlp_lp_bound`** (`dtwc_mex.cpp:1300-1355`): the column-major→row-major copy is correct, the
  odd-option-count check fires before the pair loop, unknown options are rejected by name, and the
  identifiers (`dtwc:solverError`, `dtwc:invalidArgument`) match the mapping at `:1562-1573` and the
  identifiers the `.m` docs and `test_pdlp_binding.m` assert
  (`MATLAB:pdlp_lp_bound:expectedSquare` is what `validateattributes` actually raises).
- **`cluster.m:76`** places `set_n_clusters(k)` before `set_method`/`cluster`, matching the sibling
  branches' handling of `k`. No sibling forwards method-specific options either, so the MIP branch is
  at parity — though `set_max_iter` at `:59` is inert for this route.
- **Optional deps stay optional**: `require_highs_index_range` sits outside `#ifdef DTWC_ENABLE_HIGHS`
  and pulls in no HiGHS header on that path; `set_highs_option` is inside it; Benders and
  `pdlp_lp_bound` keep their `#else` `SolverError` arms (`benders.cpp:428-431`, `pdlp_lp.cpp:40-46`).
  No naked `new`/`delete` introduced.
