# Handoff — R2-D3 LB_Enhanced and LB_Webb_NoLR — 2026-07-30

## Accomplishments

- Read the campaign rules, current PLAN, killed ideas, LESSONS, CITATIONS,
  D2 proof/gate pattern, live lower-bound implementations, and both live
  pruned-matrix routes before any decisive execution.
- Verified the complete Tan–Petitjean–Webb SDM 2019 and Webb–Petitjean
  Pattern Recognition 2021 primary records and preprints.
- Independently re-derived Enhanced's disjoint-cut proof, Webb's four-point
  condition for L1 and squared costs, directional/symmetric Keogh dominance,
  and the conservative tail-cap implication.
- Three read-only adversarial audits independently confirmed F54 and F55.
  The arithmetic audit additionally confirmed F57: signed `2*w` and free
  counters overflow at a valid `INT_MAX` radius.
- Registered exact finite inventories, strict ordering/tail/branch
  discriminators, two live cascade routes, separate D3/F57 non-skippable
  markers, two-attempt caps, and full-matrix expectations in
  `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
- Updated the primary-source ledger and PLAN finding/decision records before
  adding tests or running binaries.
- Committed preregistration as `8f8e7e5`, rebuilt the canonical tree with
  exact `ninja: no work to do`, and ran the inherited serial three-target
  baseline. It passed 3/3 in 4.43 seconds; the output is recorded verbatim in
  the D3 baseline. None of the three new discriminators existed in that run.
- Before either new target executed, an adversarial consistency pass rejected
  the proposed two-point tail witness because its `w=n` radius is canonically
  `n-1` under F57. The unchanged `tail_strict=2/2` marker now counts stable
  nondegenerate upper/lower orientations: capped L1/squared `20/400`,
  direct-predicate NoLR `30/450` for the source pair and its negation. The
  exact four-point `w=1` witness `3/3 < 4/4` remains supplementary.
- The pre-execution math audit also rejected a `V=1` Enhanced-greater witness
  as evidence for the no-ordering claim. Replaced it without changing the
  counter by the effective-`V=2` exact pair with Enhanced 10 and Keogh 0; its
  partner remains the effective-`V=2` pair with Enhanced 0 and Keogh 1.
- Added and committed both permanent red-first targets as `244adf7`. Three
  final read-only audits checked the independent reference mathematics,
  compile/API routes, marker semantics, assertion floors, skip rejection, and
  F54 false-green hazards before execution.
- Expected red is **CONFIRMED**. D3 passed 88/89 assertions and every
  mathematical violation ledger, then failed only because the live Enhanced
  route reported envelope-prune count 0 instead of 1. F57 failed at assertion
  3 because raw `INT_MAX` Webb returned L1 8 instead of exact 4. CTest reported
  0/2 passed in 0.54 seconds. Product attempts remain 0/2.
- F54 product commit `d09cf9c` activates both Keogh and Enhanced and retains
  their maximum. Product attempt 1 passed 113/114 assertions, including every
  mathematical ledger, exact F54 primitive/counter value, and both full
  matrices. The sole red was the test's incomplete whole-output literal: the
  public method correctly appended its existing
  `Distance matrix has been filled!` line. Attempt 1 is conservatively consumed;
  correct only that literal and run the unchanged F54 product together with
  F57 in attempt 2.
- Commit `53506a9` corrected that exact output literal. A fresh read-only
  pre-run audit then found the F57 constant-series case could pass a
  widened-but-unsaturated implementation. Commit `13cd4f6` added an analytic
  nonconstant discriminator with exact global L1/squared value 2 at radii
  `n-1`, `n`, and `INT_MAX`; it exercises both full-correction directions
  without changing the registered marker.
- Commit `29f9103` saturates the CPU Webb/Enhanced radius, makes Webb's doubled
  radius and counters unsigned and saturating, and replaces `j+w` by bounded
  addition. The static audit found no objection within F57's registered
  normal domain; F46 retains unrepresentable lengths and envelope
  shape/provenance.
- Final product attempt 2 is **PASS [confirmed]**. The unchanged serial command
  printed the exact D3 marker and passed 115/115 assertions, then printed the
  exact F57 marker and passed 24/24 assertions. CTest reported 2/2, zero
  failures, zero skips, in 0.68 seconds. The complete terminal output is in
  `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
- Commit `6abff20` corrects F55 across source contracts, the public enum and
  metric page, the historical changelog, canonical lessons, legacy test
  commentary, and the 2026-07-08 run log's dated corrigendum. It preserves the
  API name while identifying local NoLR-plus-tail-cap, scopes metric and
  directional/symmetric ordering claims, and restores Enhanced's effective
  `V=1` dominance. `uv run python scripts/check_docs_contract.py` passed.
- PLAN's live D3 and fuzz wording now uses the same theorem domains. A binding
  decision explicitly supersedes matching stale claims in the immutable plan
  archives rather than rewriting those historical records.

## Decisions

- Name the production function truthfully as `LB_Webb_NoLR` plus a
  conservative tail cap. Full Algorithm 2 and NoLR have no universal
  ordering; the tail-cap proof is a distinct one-sided result.
- Enhanced with effective `V=1` dominates matching-direction Keogh. For
  effective `V>=2`, exact repository counterexamples establish both order
  directions, so the live cascade must take the maximum.
- Keep F57 separate from F46's API/provenance work and F50's device
  arithmetic. Its focused target must execute normally and under WSL UBSan.
- Use one independent path enumerator for small exact domains, direct
  envelope/cut/predicate references, and a separate full-matrix recurrence
  only for deterministic default-`V=5` cases.
- Product attempts are capped at two. Full test matrices remain serial.

## Exact resume point

Commit the F55 PLAN/handoff bookkeeping, then write the complete D3 derivation
with a fail-closed contract checker. Run F57 under the existing WSL UBSan build
before the focused inherited and three full serial integration matrices.
Close F54/F55/F57 and D3 only after those gates and the final hygiene pass are
recorded.

Rollback is the eventual local D3/F54/F55/F57 commits in reverse order. No
remote or operator state has changed. The claim most likely to be wrong is
exact clipped-tail equivalence between the production recurrence and the
direct-predicate NoLR oracle; adjudicate it from the registered exhaustive
ledger rather than changing the band.
