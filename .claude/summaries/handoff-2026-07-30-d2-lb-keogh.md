# Handoff — R2-D2 envelopes and LB_Keogh — 2026-07-30

## Accomplishments

- Read the campaign rules, current PLAN, killed ideas, archive context,
  LESSONS, CITATIONS, D1 derivation, live lower-bound implementations, and
  every CPU/GPU call-site class relevant to D2.
- Verified the full Keogh–Ratanamahatana paper and Lemire source; separated
  their actual scope from the repository-specific L1, unrooted-squared, and
  unequal-prefix claims.
- Re-derived admissibility from path-row coverage in both orientations and
  unit-checked the L1 (`U`) and squared (`U^2`) forms.
- Independently hand-checked the exact 2,004 / 28,602 / 17,712 exhaustive
  inventories and the non-degenerate numeric discriminators.
- Ran and recorded the clean-base five-target baseline: 5/5 passed, zero
  failed, 4.61 seconds.
- Preregistered the decisive oracle, mutation discriminators, call-site
  checks, exact execution marker, and two-attempt cap in
  `.claude/baselines/2026-07-30-d2-lb-keogh.md` before adding or running the
  new test.
- Added and committed the non-skippable permanent executable oracle as
  `21ba41d` (`test: add exhaustive D2 LB Keogh oracle`).
- Decisive focused verdict: **PASS**. The exact marker reported
  `2004 / 28602 / 17712 / 2/2`, and Catch2 reported 40 assertions in one test
  case; CTest reported 1/1 passed, zero failed, 0.41 seconds.
- The executed unequal-length arbiter falsifies F29's old premise under the
  current fixed-window geometry. The negative-band helper discrepancy and
  both safe full-DTW callers were independently reached.
- An adversarial call-site review correctly observed that the first TADPole
  fixture proves safety but not positive LB-stage reachability. A
  separated-range full-DTW pair and its exact `pruned_by_lb=1` fingerprint are
  now preregistered as a supplementary gate before execution.
- Landed that strengthened gate as `f4bdd55`; it passed 51 assertions with
  the exact unchanged marker. The joint TADPole fingerprint was `(0,1)`, which
  proves both rejection of the unsafe radius-zero envelope and positive
  reachability of the intended global-envelope stage.

## Decisions

- The D2 oracle will recursively enumerate monotone paths rather than use a
  second DP implementation; production envelope generation is compared with
  a direct-window scan.
- Unequal-length prefix LB_Keogh is admissible for the current fixed
  `|i-j| <= radius` geometry when the radius is feasible and covers the DTW
  window. This is a repository-derived theorem, not attributed to the
  equal-length primary-source proposition.
- The inherited F29 fixture is infeasible at radius zero after D1. F29 is not
  changed until the registered 17,712-case arbiter executes.
- Unsafe envelope construction/size/provenance and the squared-Kim support
  declaration are independent prospective findings. D2 will name them; each
  repair gets its own failing gate and commit.
- Floating threshold roundoff is deferred to D17; D2 uses exactly
  representable integer fixtures.

## Exact resume point

Write `docs/derivations/02-envelopes-lb-keogh.md`, update the derivation index
and verified citations, add documentation-drift guards, and register the
confirmed envelope/Kim/TADPole/API discrepancies as separate R3 findings.
Then run the focused five inherited targets, docs gate, and full canonical,
llfio-OFF, and Arrow-ON matrices; update the test-inventory floors.
