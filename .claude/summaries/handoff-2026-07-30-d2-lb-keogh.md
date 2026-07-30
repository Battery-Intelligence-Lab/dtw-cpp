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
- Wrote the complete D2 derivation, synchronized the derivation index,
  primary-source ledger, public method pages, Python docs/docstrings, and
  source contracts, and added a permanent documentation-drift assertion that
  pins the exact oracle and non-skipping CTest policy.
- The first integrated docs run exposed only a source-token wrapping
  false-red for the exact `band=-1 disables LB_Keogh` marker. Rewrapping that
  same compiled docstring made the marker source-auditable; the immediate
  rerun passed with `generated documentation is current` and
  `documentation contract checks passed`. Record/repository hygiene and
  `git diff --check` also pass.
- An integrated adversarial review rejected the draft exclusion of independent
  multivariate DTW. The proof now applies the scalar bound to each separately
  minimized channel path, while an opposite-warp fixture distinguishes the
  independent objective from shared-path dependent DTW.
- Preregistered and executed the supplementary multivariate discriminator:
  production bound `3/3`, independent DTW `4/4`, dependent DTW `8/20`.
  The decisive canonical target passed the exact marker with **65 assertions
  in one case**, 1/1 passed, zero failed, zero skips; commit `39e9a92`
  contains only that test and its permanent CTest floor.
- Corrected six pre-closure false-greens: radius-zero complexity, FP32
  selected-precision wording, the `w`/`r` existence condition, omitted
  F48/F49 summaries, finite/int-representable TADPole scope, and comment-only
  GPU sentinel checking. The guard now inspects the actual CUDA/Metal compact
  writes and public Float32-to-Float64 normalization.
- Removed the last universal floating-point overclaim from TADPole's public
  and source contracts. Exact arithmetic and the executed exactly
  representable fixture are confirmed; prune/brute identity at a floating
  threshold remains explicitly open under D17.
- Final independent guard/math audits corrected TADPole decision-counter and
  prune-rate semantics, pinned both GPU survivor predicates and CUDA's
  nonnegative-band condition, mapped F30, repaired live anchors, and stated
  the commensurate/scaled-channel assumption behind multivariate units.
- Rebuilt and ran the integrated six-executable focused gate serially.
  Verdict: **PASS**, 6/6 Passed, zero failed, no CTest Skipped result. The D2
  oracle printed its exact registered marker and 65 assertions in one case.
  The five inherited binaries each printed their own Catch2 pass summary:
  `36509/29`, `2572/6`, `63/11`, `5584/26`, and `49/16`
  assertions/cases. These inherited counts are observations, not retroactive
  bands.

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
- The adversarial call-site inventory registered F46–F50 as distinct subjects:
  envelope representation/validation, squared Kim units, empty-series
  TADPole, direct-fill band/cache provenance, and GPU envelope `INT_MAX`
  arithmetic. F29 now carries a feasible replacement device gate rather than
  its falsified slope-window fixture.
- F46's radius-contract audit also owns TADPole's unchecked
  `size_t`-to-`int` full-envelope narrowing. D2 makes no pruning claim for
  non-finite samples; the current configuration-only predicate does not
  inspect that domain.

## Exact resume point

Commit the focused execution evidence, then run the complete canonical,
llfio-OFF, and Arrow-ON matrices serially at the registered 123/123, 123/123,
and 125/125 floors. Prove the exact capability-skip name sets rather than
trusting CTest's aggregate percentage, and prove Arrow's reader ran at least
390 assertions in 11 cases. Update AGENTS.md and PLAN.md only from those
observed inventories.
