# LB_Webb vs LB_Keogh in TADPole, N = 200 (W2a)

**Date:** 2026-09-28 · **Base:** `0e3b969` (worktree `pb/X1`) · **Build:** `clang-win` Release,
`-DDTWC_BUILD_BENCHMARK=ON`, Windows 11 (shared box; counters only, no timing).

## Question and pre-registered band

Does `lb_webb_symmetric` prune more TADPole pairs than `lb_keogh_symmetric`? Pre-registered: Webb
stays only if its pruned fraction is **≥ 83 %** (Keogh's 78 % + 5 points); otherwise it is deleted.

## Fixture and method

The hidden `[bench]` case of `tests/unit/algorithms/unit_test_tadpole.cpp` ("TADPole: >=50% of
brute-force DTW calls pruned"): `make_clusters(N = 200, k_true = 5, len = 64, band = 6, seed = 1)`,
`dc = tadpole_auto_dc(prob, 2.0)`, `tadpole(prob, 5, dc, prune = true, &stats)`. Deterministic.

- **A (Keogh):** unmodified `dtwc/algorithms/tadpole.cpp`.
- **B (Webb):** the same file with the four lines that build `Envelope` / call `lb_keogh_symmetric`
  switched to `WebbEnvelope` (`compute_webb_envelope(s, env_band)`) and
  `lb_webb_symmetric(..., band)` in both stages. Not committed; reverted after the run.
- A temporary print of an FNV hash of the labels, the medoids and the cost was added to the case for
  both runs, then reverted.

Command: `build/bin/unit_test_tadpole.exe "[bench]"`.

## Result `[confirmed]`

```
A Keogh: N=200 len=64 band=6 dc=24.74 | dtw_calls=4372 / 19900 pairs | pruned=78.0% (LB=16000 UB=247)
         labels_fnv=231efea9709afee3 medoids=180,79,126,2,63, cost=4964.3421861653278
B Webb:  N=200 len=64 band=6 dc=24.74 | dtw_calls=4372 / 19900 pairs | pruned=78.0% (LB=16000 UB=247)
         labels_fnv=231efea9709afee3 medoids=180,79,126,2,63, cost=4964.3421861653278
```

With Webb the full `unit_test_tadpole` suite (admissibility vs prune-off, vs the brute oracle) also
passed: 2584 assertions in 7 test cases.

## Interpretation

Webb prunes exactly the pairs Keogh prunes: 78.0 % in both, identical clustering. `LB = 16000` is
exactly the number of inter-cluster pairs (19900 − 5 · C(40, 2) = 16000), so Keogh already rejects
every cross-cluster pair; the 3,900 within-cluster pairs, where a tighter bound would have to clear
`dc`, stay below it for both bounds. The +23 % mean tightness of `webb_sym` over `keogh_sym`
(2026-07-08-lb-cascade.md) does not move a single density or parent decision here.

**Verdict: FALSIFIED (0.0 points < 5).** LB_Webb is deleted in W2c.
