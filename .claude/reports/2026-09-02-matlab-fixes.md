# MATLAB adversarial fixes (S3, S4, S5/S6, contract, comment) — 2026-09-02

Source review: `.claude/reports/2026-09-02-adversarial-matlab.md`.
Build under test: `build/mex-highs-fix/bin/dtwc_mex.mexw64`, rebuilt from source
after every edit (MSVC Release, HiGHS ON, OpenMP ON, no CUDA/Metal).

## Outcome

| Release | TOTAL | PASSED | FAILED | INCOMPLETE |
|---|---|---|---|---|
| R2025b | 124 | 123 | **0** | 1 (allow-listed `test_test_api/test_parallelisation_serial_is_honest`) |
| R2024b | 124 | 123 | **0** | 1 (same) |

`ctest --test-dir build/mex-highs-fix -C Release -R matlab_suite` → **Passed**
(1/1, 28.74 s, exit 0). New measured passed count = **123** (was 117);
`DTWC_MATLAB_SUITE_MIN_PASSED` is 115 at `tests/CMakeLists.txt:871` — floor may
be re-pinned to 123 (owner: caller; I may not edit that file).
`uv run python scripts/check_docs_contract.py` → "documentation contract checks
passed" after `scripts/generate_docs.py` regenerated `docs/content/api/tier-1.md`.

## Per item

**S3 — GPU ordinal dropped.** `DTWClustering.m`: `apply_device_strategy` moved to
a `methods (Static, Hidden)` block next to a new `gpu_index(name)` resolver, and
now calls `prob.set_cuda_settings(gpu_index(activeDevice))` *before* choosing the
cuda/metal strategy — mirroring `detail::configure_device` (`dtwc/api.cpp:92-113`),
which sets `cuda_settings.device_id = index` ahead of the `#if` on the backend.
`Problem.m::set_cuda_settings` no longer forces `precision = 0` when precision is
omitted (the MEX already treats it as optional). Added
`Problem_get_cuda_settings` to the gateway + `Problem.m::get_cuda_settings`
(mirrors the existing checkpoint round-trip pair) so the id is observable.
Tests (`test_contract_parity.m`): `test_dtwclustering_gpu_index_is_parsed_from_the_canonical_name`
(unconditional: cpu→0, gpu→0, gpu:1→1, cuda:3→3, 'gpu:x'→`dtwc:deviceError`),
`test_problem_cuda_settings_round_trip`, and
`test_dtwclustering_forwards_the_gpu_ordinal_to_cuda_settings`.
**Deviation from the brief:** the last one branches on `system_check` instead of
`assumeTrue`. An assumption yields MATLAB "Incomplete", which the `matlab_suite`
gate rejects unless allow-listed in `tests/CMakeLists.txt` — a file I may not
edit. Both branches therefore assert and print: on this build
`S3_GPU_ORDINAL branch=no-gpu rejected-before-effect` (observed), on a GPU build
it asserts `get_cuda_settings().device_id == 1`. That GPU branch is
**[inferred], never executed here** — the one claim I most expect to be wrong.

**S4 — dead ragged branch.** `cluster.m` now accepts a cell array (doc + the
`elseif`), normalising through a new `dtwc.Dataset.normalise_cell_series` static
(1×N cell of real double row vectors, else `dtwc:invalidArgument`), so the
`mxIsCell` branch of `cmd_tier1_cluster` is live; `skip_cols`/`skip_rows` stay
C++-owned exactly as for a matrix. `Dataset.materialize` gained the matching
cell branch (drop leading series, then leading elements, same verbatim
`load: skip_cols exceeds an in-memory series length.`); `load.m` docs updated.
Tests (`test_tier1_route_parity.m`, mirroring
`tests/python/test_api.py::TestRaggedInMemorySource`):
`test_ragged_cell_source_matches_the_tier2_route` (4 series of lengths 4/2/3/5;
labels, medoids and cost equal the Tier-2 `Problem` + seeded FastPAM oracle),
`..._honours_skip_rows_and_skip_cols`, `..._rejects_a_non_numeric_element`.

**S5/S6 — test debt.** `test_every_cpp_method_name_is_reachable` now appends
`'mip'` behind the same probe `test_cluster_mip.m` uses and prints the routed
count: observed `TIER1_METHODS routed=10/10 skips=0` on both releases, so the
name no longer over-claims. `test_tier1_load_negative_skip_rows_rejected` keeps
its two inputParser assertions and now additionally drives
`dtwc_mex('tier1_cluster', ...)` directly with `skip_rows=-1` and `skip_cols=-1`,
pinning the verbatim C++ `detail::validate_skips` messages
`load: skip_rows must be non-negative.` / `load: skip_cols must be non-negative.`
and identifier `dtwc:invalidArgument`.

**Contract line.** `docs/api-contract-2.0.md` §1.3 `device` MATLAB cell now reads
"`''` reads the process device (`dtwc.device()`); a non-empty value is a per-call
override that configures the local `Problem`'s distance strategy (GPU ordinal
included) and never mutates the process device". §1.2 `source` and §1.3 `data`
MATLAB cells now list the cell/ragged source.

**Comment.** `dtwc_mex.cpp` drain-order comment replaced: Results own their
Problem through a `shared_ptr` and are never registered in
`HandleManager<Problem>`, so the two drains are independent.

## Lint

`checkcode` on all changed `.m` files: only the two pre-existing
`DTWClustering.m:207` `predict()` warnings, the pre-existing
`test_contract_parity.m:437` unused-`testCase`, and two pre-existing stale
suppressions (662, 1049). Zero new messages.

## Proposed CHANGELOG bullet (user-visible; not applied — CHANGELOG.md is off-limits)

- MATLAB: `dtwc.cluster`/`dtwc.load` accept a cell array of numeric vectors as a
  ragged in-memory source, matching C++ `load(series_type)` and the Python list
  route; `dtwc.DTWClustering` with `Device='gpu:N'` now forwards the GPU ordinal
  to `Problem.cuda_settings.device_id` instead of always using GPU 0;
  `dtwc.Problem.get_cuda_settings` added.

## Unresolved

- S1 (Tier-1 `kmedoids`/`lrcore` need a CWD-relative `./results`) is untouched —
  it lives in `dtwc/`, owned elsewhere, and still needs its own finding.
- S2 docs hygiene (`PLAN.md`, `AGENTS.md` MATLAB inventory line, now 124/123/0/1)
  is outside my ownership; `AGENTS.md:159-162` remains stale.
- The GPU branch of the S3 test has no local execution evidence (no CUDA/Metal in
  this MEX). Rollback for everything here: revert the seven files listed above.
