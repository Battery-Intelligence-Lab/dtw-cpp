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

## Exact resume point (updated)

Finish the frozen API-contract source audit. Register every confirmed
implementation gap as its own R3 finding before correcting contract prose, then
regenerate derived pages and run the registered docs drift gate.

The frozen-contract reconciliation band is now registered as D6 in
`.claude/baselines/2026-07-23-r1-docs-truth.md`. Its PLAN decision preserves
every 2.0 promise, routes the nine confirmed implementation gaps to R3, and
limits the current task to documentation truth.
