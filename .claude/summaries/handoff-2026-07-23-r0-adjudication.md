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

## Decisions and findings

- The proposed F9 workflow is not yet a gate: it asserts at least 12 Catch2
  cases but omits the registered 348-assertion floor.
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
- The Vinod bibliographic record is verified, but the full text was not read.
  Any “same 0/1 program” statement remains inferred and must be labelled as
  such rather than used as load-bearing mathematical evidence.

## Exact resume point

Repair and adjudicate F9 next, including a local Arrow-ON suite that proves the
subject ran; then adjudicate the scholarly records before checking R0 complete.
