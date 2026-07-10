# Phase 8.1 remediation ledger — 2026-07-10

Each confirmed review finding records its pre-fix discriminator and focused
post-fix gate here. Full configuration floors remain in
`2026-07-10-phase67-floors.md` until the Phase 8 exit gate re-records them.

## H4 — frozen-contract governance (`8562104`)

Registered band: after adding an executable governance assertion, the current
contract must fail because its STATUS line lacks the required decision-entry
clause. After remediation, the docs generator/contract/CLI drift gate and the
Python contract-parity suite must pass.

Red command:

```powershell
uv run --no-sync python scripts/check_docs_contract.py
```

Decisive output (verbatim):

```text
generated documentation is current
AssertionError: frozen API contract status must require a PLAN.md decision entry
```

Green commands and decisive output:

```powershell
uv run --no-sync python scripts/check_docs_contract.py --cli build/highs-1151/bin/dtwc_cl.exe
uv run --no-sync pytest tests/python/test_contract_parity.py -q
```

```text
generated documentation is current
documentation contract checks passed
153 passed in 4.09s
```

Decision: restore executable governance; explicitly authorize rather than hide
the MATLAB post-freeze method exception and the C++ HPC throwing-beta boundary.
The dated compatibility/rationale/owner decisions are in `PLAN.md`.

## H2 — sklearn precomputed pairwise tag

Registered band: both new tests must fail before the fix. The native tag must be
`True` only for `metric="precomputed"`; a real `GridSearchCV` fold must receive
a square train-by-train matrix. After the one-line tag fix, the complete
estimator test file must pass.

Red commands:

```powershell
.venv/Scripts/python.exe -m pytest tests/python/test_sklearn_estimator.py::test_precomputed_native_pairwise_tag -q
.venv/Scripts/python.exe -m pytest tests/python/test_sklearn_estimator.py::test_grid_search_slices_precomputed_matrix_on_both_axes -q
```

Decisive output (verbatim):

```text
AssertionError: assert False is True
 +  where False = InputTags(..., pairwise=False).pairwise
FAILED tests/python/test_sklearn_estimator.py::test_precomputed_native_pairwise_tag
1 failed in 5.96s

X = array([[ 0. ,  0.2,  0.4, 10. , 10.2, 10.4],
           [ 0.2,  0. ,  0.2,  9.8, 10. , 10.2],
           [10. ,  9.8,  9.6,  0. ,  0.2,  0.4],
           [10.4, 10.2, 10. ,  0.4,  0.2,  0. ]])
ValueError: fit with metric='precomputed' requires a square distance matrix
FAILED tests/python/test_sklearn_estimator.py::test_grid_search_slices_precomputed_matrix_on_both_axes
1 failed in 5.06s
```

The 4×6 fold is the bug: sklearn sliced rows only instead of producing the
required 4×4 train-by-train matrix.

Green command:

```powershell
uv run --no-sync pytest tests/python/test_sklearn_estimator.py -q
```

Decisive output (verbatim):

```text
...........                                                              [100%]
11 passed in 4.46s
```

Verdict: **PASS.** Native/get_tags coverage also proves raw mode remains
non-pairwise.

## H1 — Python matrix-free band propagation

Registered band: on a non-degenerate pair with an eight-step warp, every real
matrix-free Tier-1 method (`onebatch`, `clara`, `tadpole`) must retain
`distance_matrix is None`, produce cost `8.6` for full DTW, and produce cost
`63.85` for a Sakoe–Chiba band of 5. No setter spy is accepted as the oracle.

Red command:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/python/test_api.py::TestMatrixFreeBand --tb=short
```

Decisive output:

```text
banded.cost = 8.6
expected     = 63.85
3 failed
```

All three algorithms silently used the unbanded callable before the fix.

Green commands and decisive output:

```powershell
uv run --no-sync pytest tests/python/test_api.py::TestMatrixFreeBand -q
uv run --no-sync pytest tests/python/test_api.py tests/python/test_problem.py -q
```

```text
...                                                                      [100%]
3 passed in 2.73s
..................................................                       [100%]
50 passed in 4.07s
```

Verdict: **PASS.** The production fix sets the band before `set_data()` performs
the normal refresh/rebind; no matrix is materialized and no unsupported
capability is silently substituted.

## H3 — non-trivial soft-DTW adjoint validation

Registered band: central finite differences on a non-degenerate 5×7 pair must
agree with the production squared-cost soft-DTW adjoint at every center
coordinate within `1e-5` relative error for gamma 0.1 and 1.0. The perturbation
is `1e-6 * max(1, |x_i|)`.

Initial result: **coverage gap confirmed, arithmetic bug falsified.** The new
test passed on its first execution without modifying the forward or adjoint
recurrences. This item therefore has no artificial red result; the old 1×1
test was insufficient, but the production math was correct.

Focused command:

```powershell
build/highs-1151/bin/unit_test_barycenter.exe \
  "soft-DTW production adjoint matches finite differences" \
  --success --reporter compact
```

Decisive output (verbatim):

```text
All tests passed (12 assertions in 1 test case)
```

The smallest-magnitude checked component was
`0.00071988044417052` versus finite difference
`0.00071988043731225`; both gamma regimes and all ten gradient components
passed.

## First remediation-wave full gates (H1/H2/H3/H4 cumulative)

The initial combined build-and-test shell reached its 180-second wrapper limit
and was **not counted**. The build and test were rerun as separate commands.

Native decisive output:

```text
100% tests passed, 0 tests failed out of 99
Total Test time (real) =  28.29 sec
```

Six unchanged capability skips: CUDA ×2, Arrow/Parquet reader ×1, Metal ×3.

Python decisive output after the five new H1/H2 tests:

```text
412 passed, 11 skipped in 18.57s
```

Verdict: **PASS.** This editable Python wave gate uses the already-verified
HiGHS-enabled release core plus live pure-Python H1/H2 sources; the Phase 8 exit
gate will rebuild a fresh wheel after all native changes.
