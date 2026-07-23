# Handoff — 2026-07-23 — R0 adjudication

## Current objective

Adjudicate every change left uncommitted by the 2026-07-20 interrupted run
before beginning any later campaign phase. Each coherent KEEP/REPAIR/REVERT
verdict is gated and committed separately.

## Accomplishments

- Read `AGENTS.md`, PLAN v2.0, LESSONS, CITATIONS, design, TODO, the prior-plan
  archive, and the current handoffs.
- Inventoried and classified every dirty path against `c36ba27`; the durable
  classification and preregistered bands are in
  `.claude/baselines/2026-07-23-r0-adjudication.md`.
- Verified the plan archive has the same 1,100 lines as the prior `PLAN.md`,
  differing only at the two R4 refinements explicitly named by R0.
- Committed the campaign record separately as `6a92e70`
  (`docs: establish the research and release campaign`).
- Reproduced F10 red (4/51 assertions failed because selected NaN/±Inf were
  accepted), repaired the validation order, and committed the complete
  signed/degenerate coverage as `20b894d`. The direct suite passes 51/51;
  changed caller suites pass; the canonical gate passes 114/114 with exactly
  six capability skips. Evidence:
  `.claude/baselines/2026-07-23-f10-sampling.md`.
- Reproduced the FastPAM point-count narrowing failure at `INT_MAX + 1`
  (2/18 assertions failed), replaced the partial widened-loop repair with one
  checked public-entry conversion, and committed it as `f8ff7d3`. The complete
  FasterPAM suite passes 258/258, FastPAM passes 76/76, live conformance passes
  7/7, and the canonical gate passes 114/114 with exactly six capability skips.
  Evidence: `.claude/baselines/2026-07-23-fast-pam-index-width.md`.
- Reclassified the mixed CLI/FastCLARA test as F7 guard coverage, not F8;
  separated its static CLI seam from the public algorithm tests; added both
  missing forced-stream prerequisites; and committed it as `7c71602`. Three
  production mutants went red. Final CLI/FastCLARA suites pass 149/149 and
  842/842; a fresh real CLI rejects capped CSV with exit 1 and accepts the same
  uncapped input with exit 0; canonical CTest remains 114/114 with six skips.
  Evidence: `.claude/baselines/2026-07-23-f7-routing-coverage.md`.
- Reproduced the F9 system-package failure in a fresh CMake directory: configure
  found Arrow and Parquet, but the generated test compile command omitted
  `DTWC_HAS_PARQUET`. Exported `Parquet_FOUND` from the dependency function and
  committed the repair as `833f570`; the newly reachable Parquet slice passes
  348/348 assertions in seven cases.
- Reproduced and repaired the independent Windows Arrow-test lifetime failure:
  the test attempted to unlink its fixture while `ArrowIPCDataSource` still
  owned the mmap. Commit `0c91c9b` scopes the reader before cleanup.
- Closed F9 with a fresh PyArrow-23-backed Arrow-ON build: the direct binary and
  CTest route run all 11 Arrow/Parquet cases and pass 390/390 assertions; the
  executable imports `arrow.dll` and `parquet.dll`. Commit `e323197` adds a
  pinned Ubuntu 24.04 job and a parser that requires ≥348 assertions and ≥11
  cases from Catch2's own summary. Six mutation probes pass. PyYAML validation
  passes; `actionlint` and `shellcheck` are `[BLOCKED-ENV]`; hosted CI remains
  operator-owned and unclaimed. Final canonical CTest passes 114/114 with
  exactly the expected six skips. Evidence:
  `.claude/baselines/2026-07-23-f9-arrow-gate.md`.
- Adjudicated every remaining scholarly record. `c627826` corrects the
  floating-point build description and keeps the historical EAP cause
  explicitly inferred; `8775156` marks TODO as a stale snapshot and closes
  only its verified FastPAM entry; `85eabcd` replaces the unsupported
  Balinski/Vinod same-program claim with a source-bounded p-median provenance
  record.
- Closed R0 with every inherited path committed and an empty
  `git status --porcelain=v1`. Final per-change verdicts are in
  `.claude/baselines/2026-07-23-r0-adjudication.md`.

## Decisions and findings

- The inherited F9 workflow was not a gate: its 12-case floor exceeded the 11
  cases in source, while it omitted the registered 348-assertion floor. It is
  repaired in `e323197`; the same parser was exercised against the exact floor,
  a skip, both one-below-floor mutations, a missing summary, and an ambiguous
  summary.
- CMake package results found inside `dtwc_setup_dependencies()` do not escape
  that function automatically. Configure messages are not evidence of the
  parent target's compile definitions; inspect the generated command and run
  the guarded binary.
- The interrupted CLI test did not close an “Arrow-OFF half” of F8: its planner
  is a static seam and its algorithm assertions were guard coverage. F8 still
  lacks every resident-versus-stream output/checkpoint comparison.
- The F10 direct test contains a false explanatory claim: adding a common
  shift preserves order and pairwise differences, not proportional sampling.
- The F10 implementation ignores a non-finite value at a selected index,
  contradicting PLAN's fail-closed “throws on non-finite” contract. R0 will
  repair the test first, reproduce red, then repair the seam.
- The FastPAM width edit was not F10. It was a partial R3 integer-width repair
  and is now closed by `f8ff7d3`; the extracted production boundary checker is
  executable without allocating more than `INT_MAX` series.
- The Vinod author-page abstract supports an early integer-programming
  treatment of partitional clustering, not DTWC++'s exact diagonal p-median
  constraint matrix. The inherited “same 0/1 program” and independent-lineage
  claims were removed rather than retained as load-bearing inference.
- The canonical build is not `-ffast-math`. It supplies an explicit
  Clang/GCC Release flag set that permits reassociation while preserving
  non-finite semantics; the surviving artifact does not isolate the historical
  EAP failure to one flag, and factor 16 remains an empirical regression
  constant pending R2-D4.
- The historical attribution split between Balinski linking inequalities and
  the ReVelle-Swain complete p-median model is the R0 claim most likely to need
  later scholarly refinement. It does not affect the constraint matrix read
  directly from the DTWC++ implementation.

## Exact resume point

R0 is complete. Begin Phase R1 with `.claude/TODO.md`'s full reconciliation:
preregister the CLOSED-BY / STILL-OPEN / NOT-REPRODUCIBLE evidence rules, then
adjudicate every remaining historical entry against the current tree. F8
remains open and belongs to R3.
