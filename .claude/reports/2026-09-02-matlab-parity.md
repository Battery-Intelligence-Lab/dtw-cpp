# MATLAB Tier-1 parity: one C++ route, no `.m` re-implementation

Date 2026-09-02. Branch `Claude`. Build `build/mex-highs-fix` (MSVC, HiGHS ON,
llfio ON). MEX at `build/mex-highs-fix/bin/dtwc_mex.mexw64`.

## Outcome

All nine drift items are closed. `bindings/matlab/+dtwc/cluster.m` is now
argument parsing plus **one** gateway call; every routing decision lives in
C++ `dtwc::cluster`. F18 and F40 are closed with their pinned red tests green.

Baseline before the work (R2025b, `runtests('tests/matlab')`):
**103 run, 100 passed, 2 failed (the two registered F18 tests), 3 incomplete**.
After: **118 run, 117 passed, 0 failed, 1 incomplete** on R2025b *and* R2024b.

## Route decision

`dtwc_mex('tier1_cluster', source, k, method, band, device, max_iter,
skip_cols, skip_rows, delimiter, name)` builds a `dtwc::Dataset` with
`dtwc::load(...)` and calls `dtwc::cluster(...)`. The returned `dtwc::Result`
is registered in a `HandleManager<dtwc::Result>` (drained before the Problem
manager in `mexAtExit`, because a Result owns its Problem). The command
returns `{handle, labels, medoid_indices, total_cost, device, name}`.
Command name is `tier1_cluster`, not `cluster`, because `cluster` is the
pre-existing stateless legacy command; renaming it was out of scope.

**Deleted from `.m`:** the whole `switch method` block, the `dtwc.device(dev)`
global mutation, the `Dataset.materialize` + `dtwc.Problem` + `set_data`
construction, the four per-branch `labels/medoids/cost` extractions, and the
hand-written CSV writer and score dispatch in `Result.m` (`Result.save` and
`Result.score` are now C++ `Result::save` / `Result::score`). `cluster.m` went
92 → 68 lines; `Result.m` lost its 40-line writer.

**One residual, named:** `dtwc::Result` exposes no accessor for its Problem or
its distance matrix (`api.hpp`), and `dtwc/` was off-limits this session, so
`Result.plot()` reads the matrix back from a scratch `Result::save` into a
temp directory (cached, private `distance_matrix`). A one-line
`const Problem& Result::problem()` or `distance_matrix()` accessor in
`dtwc/api.hpp` would remove it.

## Per item

| # | Item | Fix | Test | Before → after |
|---|---|---|---|---|
| 1 | `pam`/`kmedoids`/`auto` collapsed into one `fast_pam` | C++ routes | `test_kmedoids_routes_to_problem_cluster_not_fast_pam` (oracle: `Problem::cluster` with `Method::Kmedoids`), `test_pam_and_auto_match_seeded_fast_pam` | errored/mismatched → pass |
| 2 | `onebatch`/`lrcore`/`tadpole` absent | C++ routes | `test_every_cpp_method_name_is_reachable` (9 names, prints `TIER1_METHODS routed=9/9 skips=0`) | `dtwc:invalidArgument` → all 9 route |
| 3 | `device=` mutated the global registry | per-call override inside `dtwc::cluster` (local `Env`) | `test_per_call_device_is_validated_and_does_not_leak` | pass before and after — **this build cannot distinguish it**: only `cpu` is settable (no CUDA), so the mutation is unobservable. Closed by construction, not by measurement. |
| 3b | estimator `Device`/`Metric` not routed (F18/F40) | `DTWClustering.fit` resolves `Metric` before any effect, precomputes the exact matrix for `squared_euclidean` via the new `DTWClustering_compute_distance_matrix` MEX command, and treats `Device` as a save/restore per-call override that sets the Problem's distance strategy | the two pre-existing red tests in `test_contract_parity.m` | 2 failed → 2 passed, printing `F18_MATLAB_METRIC ... skips=0` and `F18_MATLAB_VALIDATION ... skips=0` |
| 4 | `Result.save` hard-coded ordinal names | C++ `Result::save` through `Result_save` | `test_result_save_writes_dataset_series_names` (batch CSV; `DataLoader` names rows `1..N`) | wrote `0,1,2,3` → writes `1,2,3,4` |
| 5 | `clara` ignored `max_iter` | C++ routes | `test_clara_honours_max_iter` (oracle `fast_clara` at `MaxIter` 1 and 100) | mismatch at `max_iter=1` → pass |
| 6 | `k <= N` guard missing | C++ guard | `test_k_above_n_is_rejected_with_the_cpp_message` (message asserted verbatim) | no guard → `dtwc:invalidArgument`, `cluster: k must not exceed the number of series.` |
| 7 | in-memory `skip_cols` ignored | C++ materialisation; **also** fixed `Dataset.materialize` itself so the lazy handle agrees | `test_in_memory_skip_cols_is_honoured`, `test_dataset_materialize_honours_in_memory_skip_cols`, `test_in_memory_skip_rows_is_honoured` | ignored → honoured, and `skip_cols > L` raises the C++ message |
| 8 | `Problem::checkpoint` not exposed | `Problem_set_checkpoint` / `Problem_get_checkpoint` MEX (mirroring `set_mip_settings`, using `exact_int_from_double`) + `Problem.set_checkpoint/get_checkpoint` | `test_checkpoint_options_round_trip`, `test_checkpoint_mid_fill_publishes_and_resumes`, `test_checkpoint_invalid_settings_surface_cpp_messages` | absent → round-trips; N=4/interval=1 publishes a generation whose manifest reads `pairs_computed=10` (packed cells, `n(n+1)/2`, **not** `n(n-1)/2` — the band I registered was wrong and the run falsified it), and a fresh Problem + `load_checkpoint` restores a bit-identical matrix; two of the three `InvalidInput` messages assert verbatim |
| 9 | `save/load_checkpoint` lacked `metric` | optional trailing token → `core::parse_metric_token` | `test_checkpoint_metric_is_part_of_the_identity` | 3rd arg errored → save as `squared_euclidean`, `load(...,'l1')` is `false`, `load(...,'squared_euclidean')` is `true`, unknown token is `dtwc:invalidArgument` |

Not asserted: the mmap-storage checkpoint `InvalidInput`. It needs
`set_storage_policy('mmap')` plus an owning `set_data`, and I did not want a
llfio-dependent case in a suite that must also pass a llfio-OFF MEX.

## Gates (verbatim)

* R2025b `runtests('tests/matlab')`: `R2025b TOTAL=118 PASSED=117 FAILED=0 INCOMPLETE=1`
* R2024b `runtests('tests/matlab')`: `R2024b TOTAL=118 PASSED=117 FAILED=0 INCOMPLETE=1`
  (the single incomplete is the allow-listed
  `test_test_api/test_parallelisation_serial_is_honest`, the opposite-flavour
  member of the capability pair on this OpenMP-ON MEX)
* `ctest --test-dir build/mex-highs-fix -C Release -R matlab_suite`:
  `1/1 Test #131: matlab_suite ..... Passed`, with
  `matlab_suite: 118 run, 117 passed, 0 failed, 1 incomplete`.
  HiGHS-only routes RAN: `test_cluster_mip`'s three cases are not among the
  incomplete set, and the two F18 markers print `skips=0`.
* `uv run python scripts/check_docs_contract.py` → `generated documentation is
  current` / `documentation contract checks passed`.
  `uv run python scripts/generate_docs.py` → `generated documentation updated`.
  No contract pin needed updating. `generate_docs.py` also rewrote generated
  pages from *other* agents' in-flight `api-contract-2.0.md` edits; I
  hand-edited none of them.
* `checkcode` on every changed file: clean, except `DTWClustering.m`'s two
  pre-existing `predict()` warnings, which are byte-identical to the HEAD
  baseline I diffed against.

## Docs

`docs/api-contract-2.0.md`: §1.3 heading dropped `[MATLAB device gap F40]`; the
MATLAB cells for `k`, `method`, `auto`, `device` now say "same"; the "MATLAB F40
source anchors" paragraph is replaced by a description of the single gateway
route; §2.2's `Device`/`Metric` estimator rows and the F18 paragraph record the
executed behaviour; §2.7's options-struct row and both checkpoint rows gain the
MATLAB bindings and the `metric` argument; the no-silent-fallback paragraph now
names only F24.

## Proposed CHANGELOG bullets (Unreleased) — not applied

```
- MATLAB `dtwc.cluster` now delegates to C++ `dtwc::cluster` through a single
  MEX call instead of re-implementing Tier-1 routing in `.m`. This makes
  `kmedoids` run Lloyd k-medoids rather than FastPAM, adds the missing
  `onebatch`, `lrcore` and `tadpole` methods, resolves `auto` by the C++ rule
  (CLARA above 5000 series), enforces `k <= N` with the C++ message, forwards
  `max_iter` to CLARA, honours `skip_cols`/`skip_rows` for in-memory sources,
  and makes `device=` a per-call override that no longer mutates the process
  device. `dtwc.Result.score`/`save` are now the C++ members, so the four
  output CSVs carry the dataset's series names and match the CLI byte for byte.
- MATLAB `dtwc.Dataset.materialize` honours `skip_cols` for an in-memory
  matrix source and rejects `skip_cols` beyond the series length with the C++
  message, matching `dtwc::Dataset::materialize_local`.
- MATLAB `dtwc.DTWClustering` executes `Metric` (`l1` / `squared_euclidean`,
  precomputing the exact matrix for the non-L1 case) and `Device` (validated
  through `dtwc::Env`, restored afterwards, and applied to the Problem's
  distance strategy) instead of only storing them. An unknown `Metric`, or a
  non-L1 metric combined with a non-standard `Variant` or a non-`error`
  `MissingStrategy`, raises `dtwc:invalidArgument` before any input, device or
  Problem effect. Closes gaps F18 and F40.
- MATLAB exposes automatic mid-fill checkpointing:
  `Problem.set_checkpoint(dtwc.CheckpointOptions(...))` /
  `Problem.get_checkpoint()` write and read `Problem::checkpoint`, and the
  three `fill_distance_matrix` validation errors surface verbatim.
- MATLAB `dtwc.save_checkpoint` / `dtwc.load_checkpoint` take the optional
  `metric` token (`'l1'` default, `'squared_euclidean'`) that C++ and Python
  have taken since 2.0, so a SquaredL2 matrix is no longer stamped and
  reloaded as L1.
```

## Recommended floor for `tests/CMakeLists.txt`

`sum([r.Passed]) >= 115` (observed 117 on both releases; 2 head-room for a
capability skip). The allow-list is unchanged.

## Unresolved / most-likely-wrong claim

* **Item 3 is unfalsifiable on this build.** Only `cpu` is a settable device
  here, so no test can observe the old global-mutation. The claim rests on
  reading `api.cpp:309-322` (per-call `Env local`), not on a measurement.
  A CUDA build would settle it.
* `Result.plot()` round-trips through a scratch `Result::save`. It is correct
  and cached, but it writes four files to a temp directory for what should be
  one accessor. Follow-up: add `Result::distance_matrix()` (or a Problem
  accessor) to `dtwc/api.hpp` and delete the scratch save.
* The Tier-1 `kmedoids`/`mip`/`lrcore` routes print C++ Lloyd progress to the
  MATLAB console. That is `Problem::cluster()`'s own unconditional output — the
  MATLAB Tier-2 route already did it — but MATLAB Tier-1 was previously quiet,
  so users will see new noise. Silencing it needs a `dtwc/` change.
* Those same routes write Lloyd artifacts to a **source-root-relative**
  `./results` that they do not create. The new tests `cd` into a scratch
  directory containing `results/` (F45 collision lesson); a Tier-1 `kmedoids`
  call from a directory without `results/` raises `dtwc:runtime`. That is
  pre-existing C++ behaviour now reachable from MATLAB Tier-1, and is worth its
  own finding.
