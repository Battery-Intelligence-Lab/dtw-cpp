# LB cascade upgrade — LB_Enhanced + LB_Webb (Task 5.2)

**Date:** 2026-07-08 · **Machine:** dev box, shared/under load · **Build:** build/highs-1151 (clang 18).
**Base commit:** 8ca7354 (Task 5.1). **Baseline gate:** 90/90 passed, 0 failed (6 documented skips).

## The reframe (why the plan's headline band was not run)

PLAN Task 5.2 registered "**≥25% fewer full DTW calls on matrix build**, digit-identical
matrix." That band is **unachievable as written** and was NOT chased — it would require gaming
a counter. Proof, `[confirmed]` from source:

- `fill_distance_matrix_pruned` builds an EXACT N×N matrix; every consumer reads exact entries.
- A full exact matrix needs every DTW computed. A lower bound skips work only when the exact
  value is NOT needed (NN-search: skip if LB ≥ best-so-far). No LB can skip a DTW here.
- The pruned path uses the LB as an early-abandon THRESHOLD; the kernel's abandon returns the
  `maxValue` sentinel (dtw_kernel.hpp:239,342), never the exact value, so an abandoned pair is
  **recomputed fully** (pruned_distance_matrix.cpp:281-284). Since `lb ≤ dtw`, the entry test
  `lb > threshold` ⇒ `dtw > threshold` ⇒ abandon ALWAYS fires ⇒ ALWAYS recomputes. A "pruned"
  pair costs **partial + full > full**. So `work(pruned) ≥ work(brute)`; a tighter LB pushes
  MORE pairs into the worse bucket and drives `computed_full_dtw` DOWN while doing MORE work.
- The pre-existing pruned tests assert only digit-identity + stat-sum; none checks a speedup.

Exact-matrix DTW-work reduction lives in **Task 5.3 (TADPole — skip pairs that can't change the
clustering)** and **Task 5.4 (PrunedDTW/EAPruned — prune DP cells, return exact)**. Task 5.2
delivers the tighter LB **primitives** those consume, plus any NN path. User chose this scope
("primitives + honest reframe").

## What shipped

`dtwc::core::lb_enhanced` (Tan, Petitjean & Webb, SDM 2019) and `lb_webb` /
`lb_webb_symmetric` (Webb & Petitjean, Pattern Recognition 2021), header-only, templated on the
pointwise metric (L1Metric + SquaredL2Metric). LB_Webb is clean-room from Algorithm 2 (the
authors' Java is GPL-3.0 and was NOT ported); it omits the paper's MinLRPaths corner DP (a
tightness-only optimisation) and blanket-caps the tail free-flag — both loosen, never break,
the bound. New `WebbEnvelope` (4 arrays: U, L, LU=L(U), UL=U(L)) via reused `compute_envelopes`.
Wired into `LowerBoundStrategy { …, Enhanced, Webb }` and the Problem pruned cascade (SELECTABLE
primitives; matrix stays digit-identical — the bound only feeds the abandon threshold).

## Registered bands (fixed BEFORE the runs) + verdicts

- **VALIDITY [HARD] → CONFIRMED.** `LB_Enhanced ≤ DTW_w` and `LB_Webb ≤ DTW_w` (L1 and
  SquaredL2), random + adversarial (shared endpoints, query-just-outside-envelope, constant,
  extreme 1e12), bands {0,1,2,5,10,20, 10%-of-n}, lengths incl. edge {2..11} and n≈2V. Plus
  non-negativity and `LB(x,x)=0`. Test `test_lb_enhanced_webb.cpp`: **39273 assertions, 14
  cases, all pass.**
- **LB_Webb ≥ LB_Keogh [HARD, provable] → CONFIRMED.** Per instance, one-directional
  (`webb ≥ keogh(A,envB)`) and symmetric (`webb_sym ≥ keogh_sym`). Webb = one-dir LB_Keogh +
  non-negative Thm-2 corrections, so this holds by construction and is asserted.
- **LB_Enhanced ≥ LB_Keogh: NOT claimed.** SDM 2019 proves no such ordering (the column arm can
  undercut Keogh); only validity is asserted. The cascade takes max over active bounds, so a
  looser Enhanced never regresses the result.
- **DIGIT-IDENTICAL MATRIX [HARD] → CONFIRMED.** Pruned + {Enhanced, Webb, Keogh} strategies
  match BruteForce to 1e-10 (unit_test_pruned_distance_matrix.cpp `[strategy]`, 395 assertions).

## Tightness bench (ADVISORY) — mean(LB / DTW) over all pairs, 5 clustered sets, n=128

```
 band=12 (10% of 128)             kim     keogh_sym  enhanced_sym  webb_sym
   ALL (2480 pairs)             0.0157     0.5495      0.5678       0.6758
   verdict: webb >= keogh (rel gain 23.0%);  enhanced rel gain 3.3%

 band=51 (40% of 128)
   ALL (2480 pairs)             0.0188     0.4055      0.4497       0.5498
   verdict: webb >= keogh (rel gain 35.6%);  enhanced rel gain 10.9%
```

Findings:
1. **LB_Webb tightens the envelope cascade meaningfully at both bands** (+23% at 10%, +36% at
   40% vs symmetric LB_Keogh) — the genuine "tighter bound" deliverable, validity-proven.
2. **LB_Enhanced's gain GROWS with band width** (+3.3% → +10.9%), exactly the paper's regime
   ("optimal V increases with W"). At the narrow 10% band it is a minor win; it is the tool for
   WIDE bands. Both facts recorded honestly.
3. LB_Kim (O(1) endpoints/extrema) is ~0.016 — near-useless as a standalone bound, correct as
   the O(1) cascade pre-filter it already is.

## Gate

Full `ctest` (build/highs-1151): see the commit message / session note for the final count
(baseline 90/90 → +1 new suite `test_lb_enhanced_webb`). No regression: the pruned matrix stays
digit-identical, so all existing correctness tests are unaffected.

## F55 provenance corrigendum — 2026-07-30

The numerical outputs above remain verbatim evidence for the code that ran,
but the original “What shipped” identification is false. The public
`lb_webb` function used by every logged `webb` value is the paper's all-index
`LB_Webb_NoLR` bridge and corrections plus a conservative trailing-flag cap;
it is not full Algorithm 2, which contains `MinLRPaths`. Omitting
`MinLRPaths` has no universal loosening direction: the paper's Wafer table
reports NoLR tightness 0.96904 versus 0.96891 for full Webb. Only the separate
tail cap has the proved order
`production <= exact-predicate NoLR <= DTW`.

The original Enhanced attribution is also superseded. Directional
LB_Enhanced with effective `V=1` dominates matching-direction Keogh. For
effective `V>=2`, exact D3 repository witnesses establish both strict order
directions; Tan et al. are not the source of that no-ordering result. The
measured 3.3% and 10.9% gains above describe this one benchmark and do not
establish a universal growth law.

Confirmed evidence: `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`
and `tests/unit/adversarial/test_lb_enhanced_webb_derivation.cpp`.
