# Handoff — 2026-09-02 code-quality / correctness campaign (append-as-you-go)

Base commit: 47126d8 (branch Claude). Baseline gate before any change:
build/highs-1151 125/125, 6 skips (log: build/highs-1151/baseline-2026-09-02-ctest.log).

## Method
Four read-only Opus review agents (core / algorithms / IO-CLI / backends+bindings),
reports in `.claude/reports/2026-09-02-review-*.md`. Then Opus implementers per
area, each with failing-test-first, focused ctest only. Then adversarial
re-reviews (`2026-09-02-adversarial-batch{1,2-algorithms,2-mip-matlab,3-fixers}.md`)
and fixer rounds. Rule added by Volkan mid-session: design always lock-free and
high performance (saved to memory).

## Landed (uncommitted at time of writing)
- core: MV+Interpolate / MV+SoftDTW now rejected at bind time (were silently
  flattening channels); all-NaN rejected in serial pre-scan; MissingStrategy::Error
  enforced at pairwise API; lb_enhanced/lb_webb band<0 → 0; envelope size checks
  (lb_keogh keeps the D2 prefix contract via envelope_covers); WDTW weight-span
  precondition; pruned fill honours restored checkpoint entries; num_threads
  clause instead of omp_set_num_threads; atomic_min under if(use_lb); decode_pair
  two-way correction with hoisted row_start; dead core::*Dist functors removed.
- algorithms: TADPole f32 guard (was UB) + pruning_enabled stat; cut_dendrogram
  validation; silhouette/DB/Dunn/CH on realised clusters with derived degenerate
  semantics; one_batch_pam finite guard + refresh_swap_state on accepted swaps;
  four omp critical regions named; dist_by_ind single preflight; dead medoid_utils
  helpers deleted (byte-identical objects); fast_clara f32/f64 templated.
- IO/CLI: read_distance_matrix propagates errors; unknown method/solver/linkage
  rejected; Parquet/Arrow detection outside #ifdef (Arrow-OFF gives clear error);
  Parquet nulls rejected (one null_count per chunk); DataLoader atomic counter +
  entropy; --benders CheckedTransformer; checkpoint identity includes metric;
  sorted directory listing; Ndata contract unified; extension case-insensitive;
  ofstream checks; rapidcsv dependency removed; mid-fill checkpoint claim retracted.
- CUDA/Python: pair-count int truncation guarded (launch_prep.hpp); no-device →
  DeviceError not zeros; gpu_config lock-free cache hit; atomic log latches;
  adopt_as_ndarray (no raw new); per-thread exception slots; PDLP bound in Python
  and MATLAB; checkpoint metric arg in Python.
- MIP/MATLAB: MATLAB MIP passes k; dendrogram column check; non-finite labels
  rejected; Benders uses mip_dual_bound LB, kOptimal only, post-loop SolverError,
  scaled tolerance, transaction publish; LR-core respects certified_optimal;
  need<0 guard; warm-start validation; HighsInt range + option status checks;
  mip::nearest_medoid header-inline helper; Clang MEX gateway export fix.
- misc: supply-chain manifest count 27→28 (fb853eb root cause); parquet_test.slurm
  --skip-cols dropped for parquet/dtws inputs; docs-contract check green.

## Interim gate (after batches 1-2, before batch-3 fixer)
highs-1151 128/128 (6 skips), nollfio 128/128 (9 skips), arrow-pyarrow-23 130/130
(8 skips). Log: build/gate-interim-2026-09-02.log. Python 1045 pass / 0 fail.

## Open / for Volkan
- HiGHS-enabled MEX crashes MATLAB (0xc0000005 in Thrd_yield) on any MIP solve —
  [BLOCKED-ENV]; not fixed.
- Python `Problem` is single-thread-only per instance (GIL released consistently).
- Tier-1 `Dataset` lacks skip_rows (needs cross-language parity decision).
- Mid-fill interval checkpoint is NOT implemented (docs now say so); implementing
  it needs a serial phase between row blocks — feature decision.
- Problem move-assign is `noexcept = default` while wdtw_weights_cache_ (unordered_map)
  is not nothrow-movable under MSVC STL — latent terminate risk.
- F19 script flags a benign reinterpret_cast in the Python binding (pre-existing).
- docs/public generated output still mentions rapidcsv (regenerate site).

## In flight (appended)
- batch-3 fixer: consistent GIL release + single-thread-per-Problem doc; UndefinedScore
  exception; f32 Pruned guard moved below mmap downgrade; score taxonomy → InvalidInput;
  <type_traits> in Problem_IO.cpp.
- MIP fixer (adversarial-batch2-mip-matlab.md): unscaled cut-coefficient filter (scaled
  one made cuts INVALID); N=250 default-settings Benders regression + cap; lr_max_nodes in
  MIPSettings; Gurobi/HiGHS int guards; mip_gap validation; kkt_tolerance best-effort;
  Benders warm-start size check; total_cost publish consistency; INT_MIN in MEX;
  cluster_of stale after sort; CHANGELOG overstatements.
- Then: code-simplifier pass on new hunks; final serial gates (3 matrices) + CUDA +
  Python; AGENTS.md floors (expect 128/128, 128/128, 130/130 + new mip tests);
  commits per area; PLAN decision-log digest.

## Final state (appended at close)
All agents done. Final serial gates: highs-1151 128/128 (6 skips), nollfio 128/128
(9), arrow-pyarrow-23 130/130 (8), CUDA 4/4 ran (7827/532/25/688 assertions),
Python 1045 passed / 12 skipped / 0 failed, docs-contract + hygiene green.
Run-log: `.claude/baselines/2026-09-02-quality-campaign.md`. AGENTS.md floors
promoted. PLAN.md digest + progress entry added.
PR #32 (Kasper Westman): ported hunk-by-hunk (table in run-log), then merged
`pr-32` into `Claude` so GitHub credits/marks it merged when Volkan pushes.
Commits are per area; Volkan pushes (never Claude).
Resume point: PLAN cursor unchanged (GPU-LB CUDA cluster F27/F29/F47/F50, then D4).
Open items for Volkan listed above; also `Dataset.skip_rows` parity decision and
`lr_max_nodes` Python/MATLAB exposure.
