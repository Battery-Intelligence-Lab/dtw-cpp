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
