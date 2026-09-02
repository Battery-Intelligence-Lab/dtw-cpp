# Adversarial review — MATLAB Tier-1 re-route (uncommitted, branch `Claude`)

Reviewer: adversarial agent, 2026-09-02. Read-only on the repo. Build under test:
`build/mex-highs-fix/bin/dtwc_mex.mexw64` (Sep 2 14:29, HiGHS ON, OpenMP ON);
source `bindings/matlab/dtwc_mex.cpp` is Sep 2 14:28 — binary is newer, not rebuilt.

## Measured counts

* R2025b direct, `runtests('tests/matlab')`:
  **TOTAL=118 PASSED=117 FAILED=0 INCOMPLETE=1**, the single incomplete being
  `test_test_api/test_parallelisation_serial_is_honest` (the allow-listed
  capability skip). Matches the implementer's report exactly.
* `ctest --test-dir build/mex-highs-fix -C Release -R matlab_suite` →
  `1/1 Test #131: matlab_suite ... Passed 30.14 sec`, exit 0.
* `DTWC_MATLAB_SUITE_MIN_PASSED` is **115** (`tests/CMakeLists.txt:871`), not 60 as
  briefed. 115 vs measured 117 = 2 head-room. Consistent; no action.
* `test_cluster_mip`'s three HiGHS cases really ran — they are not in the
  incomplete set, and the only incomplete is the OpenMP capability pair member.

---

## S1 — CONFIRMED, HIGH. Tier-1 `kmedoids` and `lrcore` now need a CWD-relative `./results`

`bindings/matlab/+dtwc/cluster.m:59` → `dtwc_mex('tier1_cluster',...)` →
`dtwc/api.cpp:380-386` (`problem->cluster()` with `Method::Kmedoids`/`LRCore`),
which persists Lloyd artifacts to a process-CWD-relative `./results` it does
not create. This violates non-negotiable #1 and is a *new* user-facing failure:
before the re-route MATLAB `'kmedoids'` collapsed into `fast_pam` and wrote nothing.

Repro (from any directory without `results/`):

    sc = tempname; mkdir(sc); cd(sc);
    dtwc.cluster([ones(3,8); 10*ones(3,8)], 2, 'method','kmedoids')
    % dtwc:runtime | Failed to open medoids output file: .\results/datasetmedoids_rep_0.csv

Per-method sweep from a clean CWD (observed): `kmedoids` FAIL, `lrcore` FAIL;
`mip`, `tadpole`, `onebatch`, `clara`, `pam`, `hierarchical` OK.
The implementer's report (line 145) over-claims `mip` as affected — it is not.

The suite hides this twice: `tests/matlab/test_tier1_route_parity.m`'s
`scratch_working_directory()` pre-creates `results/`, and `matlab_suite` runs with
`WORKING_DIRECTORY = ${CMAKE_SOURCE_DIR}`, where a tracked `results/.gitignore`
exists. Also note the malformed name `datasetmedoids_rep_0.csv` (missing separator).

## S2 — CONFIRMED, MEDIUM. Stale F18/F40 statements outside the contract

`docs/api-contract-2.0.md` was updated consistently (§1.2/§1.3/§2.2/§2.7 all read
correctly against observed behaviour), and `docs/content/api/tier-1.md` is its
generated mirror. Now-false statements elsewhere (listed, not edited):

* `PLAN.md:477-482` — F18 still `- [ ]`, "ATTEMPTS EXHAUSTED / FALSIFIED".
* `PLAN.md:707-719` — F40 still `- [ ]`, with a source-line citation
  (`cluster.m:36-89`) that no longer describes the file.
* `PLAN.md:31, 37` — "residuals → F40/F41/F42", "the registered F18 red".
* `PLAN.md:528` — "the expected F18/F39 reds".
* `AGENTS.md:160-162` — "**85 collected, 82 passed, 2 expected F18 failures, 3
  incomplete**". Measured today: 118/117/0/1.
* `docs/api-contract-2.0.md:145` (§1.3 `device` row) — "neither reads nor mutates
  the process device" is half-false: with `device=''` `dtwc::cluster` deliberately
  *reads* `env().device()` (`api.cpp:308-310`). Only the override case is read-free.

## S3 — CONFIRMED (code), MEDIUM. `Device='gpu:N'` silently drops the index

`bindings/matlab/+dtwc/DTWClustering.m:250-263` (`apply_device_strategy`) sets
`prob.set_distance_strategy('cuda'|'metal')` but never
`prob.set_cuda_settings(device_id, ...)`. C++ `configure_device`
(`dtwc/api.cpp:231`) does set `prob.cuda_settings.device_id = index`. So a MATLAB
estimator asked for `gpu:1` would execute on GPU 0 — a silent wrong-device, not a
wrong answer. Unmeasurable on this no-CUDA build; the code path is unambiguous.

## S4 — CONFIRMED, MEDIUM. The `tier1_cluster` cell/ragged branch is dead

`dtwc_mex.cpp:1473` handles `mxIsCell(prhs[1])` and the doc comment at 1457 claims
"or a cell of series", but `cluster.m:44-52` rejects a cell before the gateway:

    dtwc.cluster({[1 2 3],[9 1 1]}, 2)
    % dtwc:invalidArgument | cluster: data must be a dtwc.Dataset, a numeric matrix, or a file path.

Python Tier-1 accepts ragged sequences (contract §1.2 `source` row). Tier-2
`Problem.set_data` accepts a cell. So MATLAB Tier-1 is the only ragged-blind route,
and the MEX carries unreachable code plus an untrue comment.

## S5 — CONFIRMED, LOW/MED. Test name over-claims; two near-tautological tests

* `test_every_cpp_method_name_is_reachable` (test_tier1_route_parity.m:53) lists 9
  names and **omits `mip`** — one of the ten C++ names. `mip` is covered by
  `test_cluster_mip`, so coverage is fine; the test's *name* is not.
* `test_per_call_device_is_validated_and_does_not_leak` is vacuous on this build
  (only `cpu` is settable, so the old global mutation is unobservable). The
  implementer already registers this; recording it as confirmed.
* `test_tier1_load_negative_skip_rows_rejected` asserts only that the two
  identifiers are equal. Both are MATLAB's own
  `MATLAB:InputParser:ArgumentFailedValidation` (verified: `dtwc.cluster(X,NaN)`
  yields the same class of id), so the test never reaches C++ `validate_skips` and
  would still pass if `skip_rows` were ignored downstream.

## S6 — CONFIRMED, LOW. `inputParser` errors escape the §6 taxonomy

`dtwc.cluster(X, NaN)` and `dtwc.cluster(X, "2")` raise
`MATLAB:InputParser:ArgumentFailedValidation`, not `dtwc:invalidArgument`.
Pre-existing pattern, unchanged by this diff. Everything that reaches C++ maps
correctly (all verbatim, observed): unknown method → `dtwc:invalidArgument`
"cluster: unknown method 'bogus'. Valid methods: …"; `k>N` → "cluster: k must not
exceed the number of series."; bad device → `dtwc:deviceError` "[dtwc] unknown
device 'nope'. …"; missing file → `dtwc:ioError` "load: failed to read …".
`k=Inf/1.5`, `band=1.5/NaN/-2^40`, `max_iter=1.5`, `skip_cols=1.5` all →
`dtwc:invalidArgument` "… must be a finite integer in the C++ int range."
2-char delimiter → "delimiter must be a single character." String (`"pam"`) inputs work.

## REFUTED attacks

* **Handle lifetime (attack 1).** `mexLock()` fires unconditionally on first call
  (`dtwc_mex.cpp:1599`) and is never released, so `clear mex` cannot unload the
  gateway: `mislocked('dtwc_mex')` is `true` before *and* after `clear mex`, and
  `r.score('inertia')` still returned `0.22530755` afterwards. `clear all`
  survived. `delete(r)` twice is a no-op; use-after-delete raises
  `MATLAB:class:InvalidHandle`. 50 sequential Results, no crash. No dangling
  possible. Two notes: the lock is intentionally unbalanced (pre-existing), and
  the new drain-order comment at `dtwc_mex.cpp:1584` is inaccurate — Problems
  created inside `dtwc::cluster` are never registered in
  `HandleManager<dtwc::Problem>`, so the ordering is moot (harmless).
* **Column-major marshalling (attack 2).** `matrix_to_series`
  (`dtwc_mex.cpp:153-166`) reads `data[i + j*N]`, i.e. row-per-series. Verified on
  the non-symmetric fixture `X = [1 2 3; 9 1 1]`: the new
  `DTWClustering_compute_distance_matrix` and the Tier-2 `Problem` route both give
  `D(1,2) = 11`, equal to the hand DTW-L1 DP for those two rows. Labels `[2 1]`.
  `delimiter ''→0` and `device ''→global` behave as documented; `int32` input is
  accepted (cluster.m casts to `double`).
* **F18 estimator (attack 4).** The exact matrix really is the requested metric:
  `TotalCost` = 0.002137761294 for `squared_euclidean` vs 0.2253075454 for `l1`,
  and recomputing the cost from `DTWClustering_compute_distance_matrix(X,-1,
  'squared_euclidean')` with the returned labels/medoids reproduces
  0.002137761294 to <1e-9. `Band` reaches the precompute (band=2 differs from
  full). `resolve_metric` runs *before* every input/device/Problem effect;
  unknown metric and the Variant/MissingStrategy cross-products raise
  `dtwc:invalidArgument` with the Python-parallel wording. Device restore is
  correct: `onCleanup` is registered *before* `dtwc.device(obj.Device)`
  (DTWClustering.m:122-124), so the failing `Device='gpu'` fit left
  `dtwc.device()` = `'cpu'`. No path found that leaks a changed global.
* **`Result.plot()` scratch save (attack 5).** `[tempname '_dtwc_mds']` is an
  absolute `%TEMP%` path — no repo- or CWD-relative dependence — and `onCleanup`
  removes it; a before/after `dir(tempdir)` diff around a `plot()` showed no
  `*_dtwc_mds` residue (only MATLAB's own `mathworks-*`/`.tmp` churn). The written
  matrix CSV is header-free (`core::matrix_io.hpp:210`), matching
  `readmatrix(...,'NumHeaderLines',0)`. Correct, but it costs a full
  `fill_distance_matrix()` + 4 CSVs per plotted Result; the implementer's proposed
  `Result::distance_matrix()` accessor is the right fix.
* `checkcode` on all 8 changed `.m` files: only the two pre-existing
  `DTWClustering.m:207` `predict()` warnings. Report's claim holds.
* CHANGELOG.md carries the MATLAB bullets (non-negotiable #2 satisfied).

---

## Verdict — MERGE after two cheap MATLAB-only fixes; register S1

The re-route does what it claims: one gateway call, C++ owns routing, the guards
and messages are verbatim, F18 is genuinely executed (numerically verified), and
no handle-lifetime hazard exists. Nothing here is a silent wrong answer.

Fix before merge (both MATLAB-only, minutes): **S3** (pass the GPU index through
`set_cuda_settings`) and **S4** (either let `cluster.m` forward a cell, or delete
the dead MEX branch and its comment). **S1** is pre-existing `dtwc/` behaviour that
this diff newly exposes — it must be registered as its own finding with a
CHANGELOG/known-issues line, not left to be discovered by a user running
`method='kmedoids'` outside the repo root. **S2** is a docs-hygiene follow-up
(PLAN.md/AGENTS.md); **S5**/**S6** are test-quality debt, not blockers.
