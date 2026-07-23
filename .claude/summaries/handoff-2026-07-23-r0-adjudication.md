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

## Decisions and findings

- The proposed F9 workflow is not yet a gate: it asserts at least 12 Catch2
  cases but omits the registered 348-assertion floor.
- The F10 direct test contains a false explanatory claim: adding a common
  shift preserves order and pairwise differences, not proportional sampling.
- The F10 implementation ignores a non-finite value at a selected index,
  contradicting PLAN's fail-closed “throws on non-finite” contract. R0 will
  repair the test first, reproduce red, then repair the seam.
- The FastPAM width edit is not F10. It is a partial R3 integer-width repair:
  `fast_pam_seeded` still narrows `Problem::size()` to `int` unchecked, and
  `fast_pam` reaches matrix materialisation before the downstream width guard.
- The Vinod bibliographic record is verified, but the full text was not read.
  Any “same 0/1 program” statement remains inferred and must be labelled as
  such rather than used as load-bearing mathematical evidence.

## Exact resume point

Repair and gate the separate FastPAM width change, including the still-unchecked
`fast_pam_seeded` narrowing and pre-materialisation guard ordering. Then
adjudicate F8, F9, and scholarly records in that order before checking R0
complete.
