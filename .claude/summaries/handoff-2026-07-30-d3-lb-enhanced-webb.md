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

Commit the preregistration. Then run and append the clean-base inherited
focused baseline before adding either new test. No simulation or test binary
has run during D3 preflight.

Rollback is the eventual local D3/F54/F55/F57 commits in reverse order. No
remote or operator state has changed. The claim most likely to be wrong is
exact clipped-tail equivalence between the production recurrence and the
direct-predicate NoLR oracle; adjudicate it from the registered exhaustive
ledger rather than changing the band.
