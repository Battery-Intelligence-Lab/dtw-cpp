# Handoff — 2026-07-23 — R1 reconciliation

## Current objective

Make the repository record trustworthy without changing program behavior.
The TODO reconciliation is closed. The active R1 task is the docs truth audit.

## Registered work

- R0 closed at `83a2048`; the tree was clean.
- The TODO inventory contains 53 records: 49 unchecked boxes, one checked box,
  and three open questions.
- The preregistered verdict and acceptance rules are in
  `.claude/baselines/2026-07-23-r1-todo-reconciliation.md`.
- All 53 records are in scope. Known defects/cleanup map to R3 when still open;
  open product/performance work maps to its campaign, operator, or community
  owner.
- The evidence ledger is committed in `512bbc4`; the rewritten live index is
  committed in `81cae08`.
- Final parser: 53 expected/actual/unique; 21 known-bug/cleanup plus 32
  remaining; zero missing, unexpected, duplicate, or `UNVERIFIED` records.
- New unique R3 routes are F11–F16. Existing F8 still owns the permanent
  resident-versus-stream Parquet fixture.
- Focused current-tree closure gate: nine direct test binaries, 5,809
  assertions in 142 cases, zero skip markers or failures.

## Exact resume point

Complete the R1 docs truth audit: correct source-of-truth drift in README,
website pages, API contract, CHANGELOG, and comments; run the CLI contract and
internal-link gates against the fresh canonical binary. Keep changes
non-behavioral and commit each green record task separately.

The audit band and local Hugo `[BLOCKED-ENV]` probe are registered in
`.claude/baselines/2026-07-23-r1-docs-truth.md`. The CLI contract gate remains
decisive; any check over the existing ignored `docs/public/` is advisory only.

## Docs-audit findings

- F17: CLI `--resume` reads a binary `ClusteringResult` into block-local
  `ckpt_result` at `dtwc/dtwc_cl.cpp:1398-1405`, but no later code consumes it.
  User documentation and CHANGELOG disclose the limitation; R3 owns a
  real-binary failing regression and behavioral repair.
- F18: MATLAB `DTWClustering` stores `Metric` without consuming it and changes
  global `Env` for `Device` without routing its newly created `Problem`.
  Non-degenerate metric and real CUDA-enabled MEX regressions are registered.
- F19: frozen `Problem` cleanup is incomplete: promised canonical accessors are
  missing, configuration/result fields remain public, and MATLAB retains
  redundant binding-side result writeback.
- F20: `Problem::set_storage_policy` stores an advisory enum while actual
  heap/mmap data routing belongs to `DataLoader`; the frozen override promise
  has no executable path.
- F21: C++ never acquired the frozen `DataLoader::start_column/start_row` and
  `settings::paths::set_data_path/set_results_path` canonical names.
- F22: retained C++/Python/MATLAB compatibility aliases do not consistently
  emit the deprecation diagnostics promised by the frozen policy.
- F23: Python never bound the frozen binary `ClusteringResult` checkpoint
  save/load pair; only directory distance checkpoints are exposed.
- F24: Python HPC preflight/submission raises wrapper-specific `RuntimeError`
  instead of the frozen `DeviceError` and verbatim Env-message contract.
- F25: public view-mode and bounds preconditions still use assertions, making
  invalid behavior differ between Debug and Release instead of raising typed
  errors.
- F26: Python `Problem.set_view_data` first converts to owning nested vectors;
  it is not the frozen non-owning view surface its name promises.
- F27: CUDA/Metal LB_Keogh always sums L1 excess, so squared-L2 threshold
  pruning can discard a pair whose true squared-DTW cost is below threshold.
- F28: Metal defaults/accepts a narrow LB envelope under full DTW, making the
  lower bound inadmissible for paths allowed by the actual warping window.
- F29: CUDA/Metal LB kernels truncate unequal series to equal-index prefixes;
  a CUDA counterexample has LB 0.25 but admissible banded-DTW cost 0.
- F30: explicit GPU options silently degrade: Metal LB path/resource no-ops,
  CUDA full-DTW LB no-op, unsupported kernel overrides, and Metal FP64→FP32.
- F31: operational Metal allocation/launch failures escape
  `Problem::fill_distance_matrix()` as `std::runtime_error` instead of the
  frozen public `DeviceError`.
- F32: dependent Soft-DTW and Interpolate accept `ndim > 1`, then run scalar
  recurrence/interpolation over the flat interleaved buffer instead of a
  channel-aware route or a typed rejection.

## Exact resume point (updated)

Finish the frozen API-contract source audit. Register every confirmed
implementation gap as its own R3 finding before correcting contract prose, then
regenerate derived pages and run the registered docs drift gate.

The frozen-contract reconciliation band is now registered as D6 in
`.claude/baselines/2026-07-23-r1-docs-truth.md`. Its PLAN decision preserves
every 2.0 promise, routes the nine confirmed implementation gaps to R3, and
limits the current task to documentation truth.

## Frozen-contract reconciliation result

- F18–F26 are each registered once in PLAN and named in the contract as
  unfulfilled 2.0 obligations.
- All eight former reviewer questions are adjudicated to current shipped
  behavior.
- Current precision, Env ownership, checkpoint-v2 directory layout,
  `CheckpointOptions`, matrix-copy/view, LR-core, and result-writeback status
  replace the pre-implementation narrative.
- The derived Tier-1, Tier-2, and migration pages were regenerated.
- The real-CLI documentation contract gate passes; fresh Hugo rendering remains
  `[BLOCKED-ENV]` under the registered probe.

## Exact resume point (updated again)

Repair `docs/content/method/gpu-backends.md` against the registered D7 band and
add a reachable docs-contract guard first. F27–F31 are already registered; do
not implement them during this non-behavioral R1 task. Then finish the remaining
R1 docs pages, source comments, and record hygiene.

## GPU-page reconciliation result

- The inherited guard failed on all 12 registered stale claim classes.
- The corrected page names F27–F31, thresholded `+inf` semantics,
  equal-length-L1/admissible-window scope, actual option/default differences,
  and CPU-only Auto/lower-bound routing.
- Unsupported regtile/LB timing tables and crossover advice were removed; the
  one retained historical/advisory table names its tracked raw artifact.
- The real-CLI documentation contract gate passes.

## Exact resume point (current)

Correct the remaining R1 docs truth items in examples, interface parity,
multivariate, scores, and source/build comments. Add one unconditional guard
covering their known stale phrases, demonstrate the inherited red, then repair
and rerun the real-CLI docs gate. After that, reconcile LESSONS/CITATIONS/
UNIMODULAR and finish tracked-junk/CHANGELOG/branch-state hygiene.

## Remaining-doc reconciliation result

- The inherited D8 guard failed on all 24 registered deprecated-name and
  overbroad-claim markers.
- Examples now use `set_n_clusters` and canonical score names; the multivariate
  page names the live dependent/independent routes and low-level-only lower
  bounds; the score page records current degenerate behavior.
- Source comments now describe the selected floating-point relaxations,
  scalar delegation, and finite TWE sentinel without session history or dead
  SIMD claims.
- The canonical rebuild completed; its verification rerun reported
  `ninja: no work to do.` The real-CLI docs gate passes after that rebuild.
- The ignored existing-site link check passes as advisory evidence only. A
  fresh Hugo render remains `[BLOCKED-ENV]` because both Hugo and Go are absent.

## Exact resume point (current)

Reconcile LESSONS/CITATIONS/UNIMODULAR, then finish the tracked-junk,
CHANGELOG-structure, and branch-state R1 tasks. F32 now owns the multivariate
Soft-DTW/Interpolate flat-buffer defect; do not implement it during R1.

## Record-hygiene result

- Added `scripts/check_record_hygiene.py` before corrections and recorded its
  deliberate inherited red.
- Added the sole current UNIMODULAR freshness header and reconciled the live
  Lagrangian root, tolerance-guarded fixing, capped y-only exact finish, and
  separate N-theta legacy Benders route. Historical cut/runtime proposals stay
  visibly historical.
- Corrected LESSONS evidence scope for Python import provenance, SIMD,
  FastPAM1, Float32, mmap/I/O, MATLAB, CMake, Arrow/F7/F9, and LR-core/PDLP.
  The verbatim FasterPAM table is now read as advisory, non-monotone
  2.95×–8.06×; memory-bandwidth causation remains inferred.
- Deduplicated canonical citation entries and URLs, replaced mutable
  "immutable" wording with versioned/tagged wording, and recorded every opened
  primary source plus the Loog secondary-metadata caveat.
- Preserved the deliberate `0449f7c` retirement of `.claude/MISSING.md` and
  `.claude/READ.md`; they were not recreated.
- Two independent read-only final reviews found no remaining content,
  citation, anchor, fence, or checker blockers after the residual corrections.
  Evidence: `.claude/baselines/2026-07-23-r1-record-hygiene.md`.

## Exact resume point (current)

Start the R1 tracked-file junk census from the preregistered read-only audit:
add `scripts/check_repo_hygiene.py` before mutations, demonstrate its inherited
red, then remove only the five verified non-data artifacts, expand
`.gitignore`, remove the Codecov badge query token, add both compatibility
disclosures to CHANGELOG, and record the branch-state/operator merge plan.
Retain `data/test/AllGestureWiimoteX_dist_50.csv` because the absolute data
read-only rule overrides its orphan evidence. Never touch the untracked build
roots or their CMake caches.

## Repository-hygiene result

- Registered `scripts/check_repo_hygiene.py` before mutation. Its inherited
  run failed with 5 banned paths, 2 unintended zero-byte files, 3 duplicate
  groups, 3/4 asset routes, 0/23 future ignore targets, one Codecov badge query,
  and 0/2 RNG compatibility disclosures.
- Commit `4797c97` removes exactly the two empty generated artifacts and three
  byte-identical duplicate assets, repoints Doxygen to the retained static
  logo, expands future generated-file ignores, removes the README badge query,
  and adds both breaking-compatibility disclosures. No data, core code, or
  build path changed.
- The orphaned data fixture remains byte-identical at SHA-256
  `FC8173362CD03DD4D13DBF32E4FD773DE1013DD1AF28E7C6E99B95A58019D283`.
  This is a deliberate safety-rule decision, not evidence that it has a live
  consumer.
- The permanent checker reads staged index blobs rather than trusting
  worktree existence, scans its own staged content, handles malformed headings
  as FAIL, and probes ordered Git ignore semantics. Four adversarial review
  rounds rejected earlier false greens before the final `GREEN`.
- The tokenless and former query-bearing Codecov badge endpoints both returned
  status 200 and identical 2,274-character SVG content. The query value remains
  reachable in history; operator revocation/rotation is the remedy if Codecov
  classifies it as scoped.
- Installed Doxygen, Hugo, Go, gitleaks, trufflehog, and detect-secrets probes
  all returned `NOT_FOUND`. Fresh rendering remains `[BLOCKED-ENV]`; asset and
  real-CLI documentation gates are the fallback. The independent batched
  history scan covered 4,559 reachable blobs and found zero registered
  high-confidence secret shapes.

## Preserved local builds

All three ignored roots remain untouched:

- `build/`: 29 repo-source configure caches, including `highs-1151`,
  `nollfio`, Arrow/PyArrow-23, CUDA, MEX, sanitizer, configuration, YAML, and
  historical verification recipes.
- `build_arrow_test/`: one standalone Debug Arrow+HiGHS configure.
- `build_python/`: two CPython 3.13/final-wheel MSVC configures.

The 32-cache inventory is
`.claude/baselines/2026-07-23-r1-repo-hygiene.md`; deletion remains an operator
decision.

## Branch state and operator merge plan

At committed hygiene snapshot `4797c97`, before this documentation-only
closeout:

```text
main_vs_Claude=0	548
origin_main_vs_Claude=0	548
origin_Claude_vs_Claude=0	51
main_vs_origin_main=0	0
origin_Claude_vs_origin_main=497	0
 640 files changed, 153326 insertions(+), 2144 deletions(-)
```

That first snapshot was invalidated during final review: at 18:14:56 +01:00,
the local `origin/Claude` reflog advanced from `c36ba27` to `4797c97` with the
exact message `update by push`. A read-only probe at 18:15:38 then reported
`origin_Claude_vs_Claude=0 0`; `origin/main` remained at `b0cb297` and 548
behind. This agent issued no remote mutation. Only inactive sample hooks were
present, but four GitHub Desktop processes (PIDs 5684, 40592, 50008, and
88212) had been running since 17:26. A follow-up found zero `git.exe`
processes; that does not exclude GitHub Desktop as an alternate client. The
actor/cause is **[inferred: unknown]**. The original global “no remote
operation occurs” sub-band is **FALSIFIED**; the narrower verified compliance
claim is that this campaign agent issued no remote command.

After R0–R6 are clean, an operator should re-read the server state, fetch/prune,
recompute ancestry/divergence, rerun gates against the fetched base, decide
whether the unexpected `4797c97` remote update should be retained or restored,
then use a review PR and hosted gates. Fast-forward, rebase, merge, force
update, tag, and publication remain operator actions; this campaign will not
attempt the rollback.

## Exact resume point (R1 closed)

Start R2-D1 by preregistering the DTW-recurrence/Sakoe–Chiba derivation and its
non-degenerate oracle band. In parallel, resume R3 at F8's resident/streaming
fixture; F9 and F10 are already closed. Search the killed-ideas record before
either branch, keep every discrepancy as a named finding, and commit each
green task separately.
