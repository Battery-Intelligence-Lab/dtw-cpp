# Independent multivariate DTW (Task 5.6)

**Date:** 2026-07-08 · **Machine:** dev box, shared/under load · **Build:** build/highs-1151 (clang 18, fast-math).
**Base commit:** 1a1279d (Task 5.5). **Baseline gate:** 94/94 passed, 0 failed (6 documented skips).

## What shipped

Independent multivariate DTW (**DTW_I**; Shokoohi-Yekta et al., *Generalizing DTW
to the multi-dimensional case requires an adaptive approach*, DMKD 31(1) 2017) as
an orthogonal **mode**, not a new variant:

- `dtwc::dtw_independent_mv<T>` (dtwc/warping.hpp) — runs an independent
  univariate DTW on each channel and sums: `DTW_I = Σ_c DTW(x[:,c], y[:,c])`.
  De-interleaves the interleaved MV layout (`x[t*ndim + c]`) one channel at a
  time into `thread_local` scratch, then calls the univariate kernel: the exact
  EAPruned kernel (`dtwFull_eap`) when unbanded (matches the scalar Standard
  default, Task 5.4), `dtwBanded` when banded. L1 / SquaredL2 only.
- `core::MVMode { Dependent, Independent }` + `DTWVariantParams::mv_mode`
  (default **Dependent** → existing DTW_D behaviour, zero regression).

The pre-existing multivariate path (`dtwFull_L_mv` / `dtwBanded_mv`) is the
**dependent** DTW_D: one warping path shared by all channels, per-cell cost
summed over channels. Neither mode dominates for clustering accuracy — that is
the paper's central finding, so both are needed.

**Wiring (CasADi-style):** `MVMode` + `mv_mode` (dtw_options.hpp); dispatch
interception at the top of `resolve_dtw_fn` (dtw_dispatch.cpp, `make_independent`);
CLI `--mv-mode dependent|independent` (+ TOML key); Python `MVMode` enum,
`DTWVariantParams.mv_mode`, and the `KMedoids(mv_mode=...)` wrapper. Checkpoint
needs no change (it serialises only the variant name, like msm_c / adtw_penalty).

**v1 scope (documented, not silent):** Independent mode is Standard-variant +
`MissingStrategy::Error` only. Any other combination (DDTW/WDTW/ADTW/MSM/TWE, or
a missing-data strategy) is **rejected at bind time** — `set_variant()` rebinds
eagerly, so the `InvalidInput` fires there, serial, before any parallel fill —
never silently collapsed to dependent mode. Univariate (`ndim == 1`) ignores the
flag entirely (falls through to the scalar Standard path, verified no-op).

**TC-DTW LB (arXiv:2101.07731) deferred.** The plan marks it "if time"; it is a
lower-bound tightening that benefits only the pruning/NN path (Tasks 5.2 LB
cascade + 5.3 TADPole already shipped there), orthogonal to the DTW_I distance
deliverable. Left OPEN for a later LB-focused task.

## Oracle reference (aeon 1.5.0)

aeon's `dtw_distance` uses a **squared-Euclidean** local cost, matching our
`MetricType::SquaredL2`. DTW_I reference = `Σ_c dtw_distance(channel_c)`.
Generated on reproducible LCG series (embedded verbatim in the test). Regenerate:

```python
# uv run --no-project --with aeon --with numpy python thisfile.py
import numpy as np; from aeon.distances import dtw_distance
def lcg(seed, n):
    s=seed & 0xFFFFFFFF; out=[]
    for _ in range(n):
        s=(1103515245*s+12345)&0x7FFFFFFF; out.append(round(-5.0+10.0*(s/0x7FFFFFFF),6))
    return out
# cases (ndim,nx,ny): (2,5,5),(2,8,8),(2,7,11),(3,6,6),(3,10,10),(3,9,13),
#                     (2,12,12),(2,4,4),(3,15,15),(2,20,16)
# channel c of x: lcg(1000+100*k+c, nx); of y: lcg(7000+100*k+c, ny)
# DTW_I = sum_c dtw_distance(X[c], Y[c]); interleave x[t*ndim+c] for the C++ test.
```

## Registered bands (fixed BEFORE the runs) + verdicts

- **BAND-ORACLE [HARD] → CONFIRMED.** `dtw_independent_mv(..., SquaredL2)` == aeon
  1.5.0 `Σ_c dtw_distance(channel_c)` to rel ≤ 1e-9 on 10 non-degenerate 2-/3-
  channel pairs (equal + unequal lengths).
- **BAND-INEQ [HARD] → CONFIRMED.** `DTW_I ≤ DTW_D` on 200 random MV pairs
  (ndim 2–4, lengths 3–20), for L1 **and** SquaredL2, unbanded **and** banded
  (window 2). Independent-math arbiter: with an additive per-channel local cost,
  for the dependent-optimal shared path P, `DTW_D = Σ_c cost_c(P) ≥ Σ_c min_{P_c}
  cost_c = DTW_I` — each channel is free to pick its own path.
- **BAND-DEINTERLEAVE [HARD] → CONFIRMED.** `dtw_independent_mv` == manual
  `Σ_c dtwFull_eap(channel_c)` where the test de-interleaves channels
  independently of the library — guards the strided channel copy (L1 + SquaredL2,
  all 10 pairs).
- **BAND-WIRING [HARD] → CONFIRMED.** A `Problem` with `mv_mode=Independent`,
  ndim=2 fills a matrix equal to the direct kernel (rel 1e-10) and different from
  the Dependent matrix on ≥1 entry (genuinely distinct modes). CLI
  `--mv-mode independent` runs end-to-end (converged) on the dummy dataset
  (univariate → no-op Standard path).
- **BAND-REJECT [HARD] → CONFIRMED.** Independent + DDTW and Independent + a
  missing-data strategy each throw `dtwc::InvalidInput` at `set_variant()`.
- **Univariate no-op → CONFIRMED.** ndim=1 Independent == scalar `dtwFull_eap`
  (L1 + SquaredL2, 20 random pairs).

Hand-computed 2-channel example (L1): x=[(0,0),(1,1)], y=[(0,0),(3,3)] →
per-channel univariate DTW = 2 each → DTW_I = 4 (asserted).

Test `tests/unit/core/unit_test_independent_mv.cpp`: **7 cases, 889 assertions**.

## Gate

Full `ctest` (build/highs-1151): **95/95 passed, 0 failed** (baseline 94 → +1
suite `unit_test_independent_mv`; 6 documented skips: CUDA×2, Metal×3,
io_readers). No regression — Independent mode is opt-in (default Dependent).

## Honesty notes / open

- **Python runtime parity DEFERRED (not verified).** Same pre-existing wheel-build
  environment failure as Task 5.5 (scikit-build "VS 18 2026" generator; Ninja
  llfio/quickcpplib ExternalProject error). The binding edits (`MVMode` enum,
  `mv_mode` `def_rw`, `KMedoids(mv_mode=...)`) are additive and compile-consistent;
  C++ core + CLI + aeon oracle are fully gated.
- Independent mode is **Standard-only** in v1 (matches the paper's DTW scope).
  Independent DDTW/WDTW/ADTW would each need a per-channel univariate kernel call
  with the variant's own preprocessing/weights — a clean future extension.
- **TC-DTW lower bound (arXiv:2101.07731) not implemented** — deferred (see above).
