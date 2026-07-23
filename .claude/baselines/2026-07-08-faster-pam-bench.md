# FasterPAM / FastPAM1-decomposition SWAP — correctness + bench (Task 5.1)

**Date:** 2026-07-08 · **Machine:** dev box, shared/under load · **Build:** build/highs-1151 (clang 18).
**Status:** ADVISORY timings (CLAUDE.md — shared machine; read the RATIOs and TRENDs). The
correctness halves (brute-force local optimality, objective identity) are HARD gates.

## What shipped

`dtwc::algorithms::PAMVariant { FastPAM1Naive, FastPAM1, FasterPAM }`, dispatched by
`fast_pam_swap(prob, initial_medoids, max_iter, variant)`; `fast_pam` does K-means++ BUILD
then `fast_pam_swap(..., FastPAM1)`.

- **FastPAM1Naive** — the pre-5.1 baseline: single best swap per pass, naive nested O(N·k)
  per-candidate gain ⇒ **O(N²·k) per iteration**. Kept as bench baseline + oracle.
- **FastPAM1** (NEW DEFAULT) — the paper's FastPAM1: same single-best-swap result, but each
  candidate's ΔTD over all medoids from ONE O(N) pass (Eq. 11 decomposition) ⇒ **O(N²) per
  iteration**, parallel over candidates. Objective-identical to the naive.
- **FasterPAM** (NEW) — eager: performs swaps as found, shares removal loss across medoids ⇒
  **O(N²) per sweep**, converges in ≈1 sweep, but SEQUENTIAL (eager dependency).

Derivation of the ΔTD decomposition (acc + ploss[m]) is in the `fast_pam.cpp` header comment;
cross-checked against Schubert & Rousseeuw 2021 (arXiv:2008.05171) Alg. 3–4 and the Rust
`kmedoids` reference by a research agent — line-for-line agreement.

## Registered bands (stated BEFORE the runs) + verdicts

- **BAND-LOCALOPT [HARD] → CONFIRMED.** Every variant converges to a brute-force-verified
  local optimum (no improving (medoid_out, point_in) swap; ΔTD by full reassignment — tie-
  independent). 3 variants × {N=20,30,45} × {k=2,3,4}. Proves each ΔTD computation is correct.
- **DECOMP ≡ NAIVE (objective) [HARD] → CONFIRMED.** FastPAM1 decomposition reaches the same
  local-optimum objective as the naive to 1e-9 over {N=24,40,60,90}×{k=2,3,5,8}. (Exact medoid
  identity NOT required: acc+ploss[m] and the naive nested sum group terms in different order,
  so an EXACT tie — e.g. two points symmetric about a group median, equal cost — can tip to a
  different but equal-cost medoid. Objective is the invariant.)
- **NO-WORSE [HARD] → CONFIRMED.** FasterPAM final objective ≤ FastPAM1 + 1e-9 from an
  identical BUILD, {N=24,40,60,90}×{k=2,3,5}. Strictly BETTER when the best-swap variant hits
  max_iter (see k=200 below).
- **k=1 [HARD] → CONFIRMED (after fix).** All variants find the true argmin 1-medoid. The
  decomposition is special-cased at k=1: `second_dist = +inf` makes ρ=inf, corrections=−inf
  ⇒ NaN ⇒ (without the special case) it returned the BUILD medoid, not the optimum. Caught by
  the CLARA exact-median oracle (now `tests/unit/algorithms/unit_test_fast_clara.cpp:712`),
  not the weak k=1 unit test.
- **SPEED [ADVISORY] → PARTIALLY FALSIFIED (deliverable, see below).** The "≥10× faster" band
  is NOT met vs the already-parallel naive at k≤50. The retained N²-lookup structure suggests
  a memory-traffic explanation, but no PMU artifact proved the bottleneck.

## Bench — three variants, k-group data (N=1000), verbatim

```
     N    k |   naive ms (it) |    fp1 ms (it) | faster ms (sw) | naive/fp1 | naive/faster
    1000  10 |     24.2 ( 10) |     8.2 ( 10) |    39.7 (  1) |     2.95x |       0.61x
    1000  20 |     31.7 ( 20) |     9.2 ( 20) |    40.1 (  1) |     3.45x |       0.79x
    1000  50 |     72.2 ( 50) |    24.5 ( 50) |    40.0 (  1) |     2.95x |       1.81x
    1000 100 |    225.8 (100) |    52.5 (100) |    45.1 (  1) |     4.30x |       5.01x
    1000 200 |    380.6 (100*)|    47.2 (100*)|    62.9 (  1) |     8.06x |       6.05x   (* max_iter cap)
  large-N (naive omitted; O(N²·k) impractical):
    2000  50 | fp1  85.1 ms (50 it) | faster 149.9 ms (1 sw) | obj fp1=400.0  faster=400.0
    5000  50 | fp1 662.9 ms (50 it) | faster 1063.4 ms (1 sw) | obj fp1=2500.0 faster=2500.0
```

**Record correction (2026-07-23):** the verbatim table is authoritative. Its
naive/FastPAM1 ratios span **2.95×–8.06×** and are non-monotone (the k=50 ratio
returns to 2.95×). The earlier 2.3× narrative minimum and monotone-growth claim
were transcription/interpretation errors, not additional measurements.

## Findings

1. **The decomposition (fp1) beats the naive at every measured k:
   2.95×–8.06× in this advisory table.** The ratios are non-monotone. This is
   the measured O(N²·k)→O(N²) comparison, parallel, objective-identical, with
   the same iteration counts (same swaps).
   This is the genuine "replace the O(N²·k) swap" deliverable and the new default.

2. **Why NOT ≥10× remains inferred.** Both implementations retain N²
   distance-matrix lookups per iteration, while the decomposition removes O(k)
   arithmetic per point. That structure is consistent with memory traffic
   limiting the gain, and a scratch-buffer hoist moved less than benchmark
   noise, but no PMU run established bandwidth or cache behavior. **The ≥10×
   target is falsified by the table regardless of mechanism.**

3. **FasterPAM converges in ONE sweep at every k**, vs the best-swap variants' O(k) iterations.
   At k=200 both best-swap variants hit max_iter=100 WITHOUT converging (returning a worse
   objective); FasterPAM finished in 1 sweep. This is FasterPAM's real edge: far fewer passes
   over the matrix + robustness at large k.

4. **At large N the parallel fp1 beats sequential FasterPAM** (fp1 85 ms vs 150 ms at N=2000;
   663 ms vs 1063 ms at N=5000), identical objective. FasterPAM's O(N²)/sweep is sequential
   (eager dependency) + refreshes state O(N·k) per accepted swap (the paper's O(N) incremental
   do_swap was NOT implemented — a documented simplification; it would help large-k FasterPAM
   but not change that parallel fp1 won at the measured large N; the
   memory-bound explanation remains inferred). → **fp1 is
   the right default** (never regresses small-k/large-N, k× fewer arithmetic ops than naive).

## Scope notes (evidence-based)

- **LAB init: intentionally SKIPPED.** The paper (§3.3) states LAB's head-start is "largely
  nullified" by FasterPAM's eager swapping and recommends uniform/k-means++-style init instead
  — which is exactly the existing K-means++ BUILD. Implementing LAB would add code for a
  head-start the authors say the eager swap erases. Recorded, not built.
- **FasterCLARA: inherited, carry-over deferred.** CLARA already calls `fast_pam` and so runs
  on the faster decomposition FastPAM1 automatically. The paper's FasterCLARA refinement
  (carry best medoids across subsamples) is a separate enhancement — deferred with a note.

## Gate

CPU build `build/highs-1151`: full `ctest` → **100% passed, 0 failed / 90** (new suite
`unit_test_faster_pam`: 240 assertions / 7 cases; `unit_test_fast_clara` 813/18;
`unit_test_fast_pam` 64/11). No regression.
