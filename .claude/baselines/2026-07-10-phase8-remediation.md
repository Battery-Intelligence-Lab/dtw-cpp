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

## M2 — matrix-free `Result.distance_matrix` migration contract

Registered band: the documentation drift gate must fail while the generated
migration guide omits the behavior, then pass only when the generator source,
tracked guide, CHANGELOG, and live `Result` docstring agree.

Red command and decisive output:

```powershell
uv run --no-sync python scripts/check_docs_contract.py
```

```text
generated documentation is current
AssertionError: migration guide omits matrix-free Result.distance_matrix behavior
```

The runtime behavior already had direct tests; this finding was a migration
contract omission, not a new implementation change.

Green commands and decisive output:

```powershell
uv run --no-sync python scripts/generate_docs.py
uv run --no-sync python scripts/check_docs_contract.py --cli build/highs-1151/bin/dtwc_cl.exe
uv run --no-sync pytest tests/python/test_api.py -q
```

```text
generated documentation updated
generated documentation is current
documentation contract checks passed
36 passed in 5.20s
```

Verdict: **PASS.** The tracked migration page is generated from the same source
that the drift check validates, so a future regeneration cannot silently erase
the behavior note.

## M8 — rc1 behavior and migration ledger

Registered band: the docs drift gate must name both missing migration promises
before the fix. Afterward the generated guide must explicitly contrast the 1.x
GPU fallback and solver print/return paths with typed 2.0 errors; CHANGELOG must
also contain the `dist_by_ind` rebind-race fix and a separate development-history
heading.

Red decisive output:

```text
AssertionError: migration guide omits behaviors: ['Explicit GPU requests no longer warn and run on CPU', 'Requesting an unavailable MIP solver no longer prints and returns']
```

An independent assertion against the pre-fix `HEAD:CHANGELOG.md` also rejected
all four required summary/structure markers:

```text
AssertionError: HEAD rc1 changelog omits behaviors/structure: ['raises `DeviceError`', 'raises\n  `SolverError`', '`dist_by_ind` rebind race', '# Development history absorbed into 2.0.0rc1']
```

Green decisive commands:

```text
uv run --no-sync python scripts/generate_docs.py
  generated documentation updated
uv run --no-sync python scripts/check_docs_contract.py --cli build/highs-1151/bin/dtwc_cl.exe
  generated documentation is current
  documentation contract checks passed
git diff --check
  exit 0
```

Verdict: **PASS.** The generated migration SSOT and its drift gate now enforce
both breaking behaviors. The same gate also scopes the three rc1 behavior notes
to the release summary, requires the separate absorbed-history heading, and
rejects the old stray contract bullet.

## M3 — barycenter Problem-configuration contract

Registered band: both public barycenter entry points must reject all six
non-Standard variants and every finite Problem band before silently computing
their intrinsic Standard, unbanded, squared-local-cost objective. Error text
must identify the entry point and the supported setting.

Red decisive output before the guard:

```text
unit_test_barycenter.exe "[configuration]" --reporter compact
  1 test case failed, 2 assertions failed
  dtw_barycenter returned normally for DDTW and band=1
```

Green decisive output:

```text
unit_test_barycenter.exe "[configuration]" --reporter compact
  All tests passed (28 assertions in 1 test case)
unit_test_barycenter.exe
  All tests passed (56 assertions in 7 test cases)
ctest --test-dir build -R "^unit_test_barycenter$" --output-on-failure
  1/1 passed, 0 failed
```

Adversarial premise check:

```text
rg -n "set_metric|MetricType|metric|variant_params|int band" \
  dtwc/Problem.hpp dtwc/Problem.cpp dtwc/core/dtw_options.hpp
  Problem.hpp:134: int band{ settings::DEFAULT_BAND };
  Problem.hpp:141: core::DTWVariantParams variant_params;
  dtw_options.hpp:77: MetricType metric = MetricType::L1;
```

Verdict: **PASS, metric clause FALSIFIED/N/A.** `Problem` exposes variant and
band configuration but no metric state or setter. Adding one merely to satisfy
the finding would expand the frozen API and imply semantics that the distance
dispatcher does not own. The supported squared-cost barycenter objective is now
documented explicitly instead.

## M1 — Python/C++ device-alias parity

Registered band: Python must agree with live `dtwc::Env` on 62 valid and
adversarial GPU spellings, preserve `gpu:N` through both CUDA and Metal
resolution, and keep explicit CUDA from falling through to Metal. C++ whitespace
and 32-bit ordinal boundaries are part of the grammar.

Red progression before the final parser:

```text
gpu:0, gpu:07, and gpu:2147483647: rejected by Python, accepted by Env
1 failed, 75 passed, 1 skipped
  Metal resolved gpu:7 as ('metal', 0), expected ('metal', 7)
adversarial follow-up: vertical-tab-wrapped aliases were accepted by Python
  strip() but rejected by C++ trim()
```

Green decisive commands:

```text
uv run --no-sync pytest \
  tests/python/test_device.py tests/python/test_cuda.py \
  tests/python/test_contract_parity.py -q
  244 passed, 10 skipped in 2.80s
uv run --no-sync pytest tests/python -q
  480 passed, 11 skipped in 11.20s
git diff --check
  exit 0
```

Verdict: **PASS.** Python now delegates syntactic truth to the same grammar
contract as `Env`: four explicit trim characters, ASCII decimal ordinals, and a
C++ `int` maximum. Syntax failures are `DeviceError`; non-string inputs are
`InvalidInput`; operational GPU failures remain loud and never change the
stored global selection.

## L2 — LR-core canonical derivation drift gate

Registered band: a clean clone with the canonical derivation removed must fail
the generator check. A clean clone with the source present must pass, and a
source-only perturbation must name `docs/content/math/lr-core.md` as stale.

The premise that the report was still gitignored was itself stale:

```text
git ls-files --error-unmatch .claude/reports/solver-math-2026-07-06.md
  .claude/reports/solver-math-2026-07-06.md
git check-ignore -v .claude/reports/solver-math-2026-07-06.md
  no match
```

Nevertheless, the degradation was real. In an isolated clean clone, deleting
that tracked file before the repair returned the false green:

```text
uv run --directory build/phase8-l2-red --no-sync \
  python scripts/generate_docs.py --check
  generated documentation is current
```

After promoting the byte-identical source to
`docs/sources/lr-core-derivation.md` and making the read mandatory:

```text
source present: generated documentation is current
source deleted: RuntimeError: missing canonical LR-core derivation source:
                docs\sources\lr-core-derivation.md
source perturbed: stale or missing generated documentation:
                    docs\content\math\lr-core.md
```

Verdict: **PASS.** The gate cannot degrade to an existence-only check. The full
derivation has a permanent documentation-owned source, missing-source output is
actionable, and content changes require explicit regeneration.

## M10 — Metal chunk-offset production seam

Registered band: the host test must call arithmetic used by both production
Metal dispatch loops, cross `INT32_MAX`, and drive the real shared pair decoder.
Changing the seam's return type to `int32_t` must fail independently of any
locally mirrored constant.

Red before the seam existed:

```text
test_decode_pair.cpp: no member named 'pair_chunk_offset' in
namespace 'dtwc::metal::detail'
```

Adversarial mutation after extraction (`int64_t` to `int32_t`) produced:

```text
static assertion failed: returned -2147471303, expected 2147495993
static assertion failed: return type is not std::int64_t
```

Green decisive output after restoring the production type:

```text
cmake --build build/highs-1151 --config Release --target test_decode_pair -j 4
ctest --test-dir build/highs-1151 -C Release \
  -R "^test_decode_pair$" --output-on-failure
  1/1 passed, 0 failed
ctest --test-dir build/highs-1151 -C Release --output-on-failure
  99 registered, 0 failed, 6 documented capability skips, 26.71s
```

Verdict: **PASS (coverage defect, arithmetic already correct).** The new helper
owns the `size_t` to Metal-ABI `int64_t` boundary for NxN and K-vs-N dispatch.
The regression uses its result in `decode_pair` and proves a real encode/decode
round trip beyond the old overflow boundary. The all-target build wrapper that
timed out before CTest emitted no result and is deliberately not counted.

## M7 — OneBatchPAM estimator provenance and Dmax correction

Primary-source verdict (pinned 2026-07-10):

1. **CONFIRMED:** NNIW counts each fixed-batch point's Voronoi assignments and
   divides by mean count `n/m`. This is proportional to Loog's raw counts and is
   used by both authors' implementations.
2. **MIXED PROVENANCE:** arXiv:2501.19285 states a literal +∞ diagonal and
   presents Debias/NNIW separately. `obpam@ee823101...` instead normalizes by
   finite Dmax, writes normalized diagonal 1, and combines that correction with
   NNIW. Maintained `onebatch` v0.1.0 keeps count/mean NNIW but drops diagonal
   replacement. DTWC++ deliberately matches the paper-linked experiment hybrid.
3. **CORRECTED DEFECT:** DTWC++ used `scale=max(1,Dmax)` both for normalization
   and diagonal replacement. It matched the experiment only for Dmax≥1; below
   one it changed candidate ordering.

Registered counterexample with fixed batch `{0, 0.01}` and candidate `0.1` has
Dmax=0.1. Before the fix, the nonsampled candidate won because its off-diagonal
costs remained below the substituted raw diagonal 1:

```text
unit_test_one_batch_pam.exe "[debiasing]" --reporter compact
  expected medoid {0}; actual medoid {2}
  13 assertions passed, 1 failed
```

Green output after separating actual table maximum from the all-zero fallback:

```text
unit_test_one_batch_pam.exe "[debiasing]" --reporter compact
  All tests passed (16 assertions in 1 test case)
unit_test_one_batch_pam.exe
  All tests passed (587 assertions in 4 test cases)
ctest --test-dir build/highs-1151 -C Release \
  -R "^unit_test_one_batch_pam$" --output-on-failure
  1/1 passed, 0 failed
```

The all-zero table remains finite with exact cost zero. Verdict: **PASS after
correction.** A separate source audit confirmed another real issue:
`relative_tolerance * max(1,cost)` rejects a 33.3% improvement when cost<1.
That independent behavior is registered as M9 rather than hidden in this fix.

## M9 — sub-unit relative stopping tolerance

Registered counterexample: values `[0, .01, .02, 1]`, k=2, full uniform batch,
seed 0, one sweep, and `relative_tolerance=.2`. Initial medoids `{3,2}` have
estimated/exact objective .03; swapping 2→1 gains .01 and reaches .02. The
relative gain is 33.3%, so the registered .2 threshold must accept it.

Red with the old `tolerance * max(1, cost)` formula:

```text
unit_test_one_batch_pam.exe "[tolerance]" --reporter compact
  failed: medoids {3,2} == expected {3,1}
  1 test case, 1 assertion, 1 failure
```

Green after using the actual current estimate:

```text
unit_test_one_batch_pam.exe "[tolerance]" --reporter compact
  All tests passed (4 assertions in 1 test case)
ctest --test-dir build/highs-1151 -C Release \
  -R "^unit_test_one_batch_pam$" --output-on-failure
  1/1 passed, 0 failed
```

Verdict: **PASS.** For zero objective the threshold is zero, which is safe:
nonnegative distances have no genuine positive improvement left. Both pinned
authors' implementations likewise multiply tolerance by the live loss without
an absolute unit floor.

## L1 — workflow and Arrow supply-chain pins

Registered band: every non-local `uses:` reference in every workflow must be a
40-hex commit; the Arrow 19.0.1 URL must carry its independently reproduced
SHA-256. A standard-library script enforces both and runs in CI.

Red before remediation:

```text
mutable GitHub Action references:
  36 occurrences across 10 workflow files
  (checkout, artifact, Python/Go/uv/MATLAB/Hugo/Doxygen/JOSS/cibuildwheel,
   Pages deployment; codecov was the sole existing full-SHA pin)
Arrow archive is missing URL_HASH SHA256
```

`git ls-remote` resolved tag refs directly from each upstream repository; for
annotated tags the peeled commit (`^{}`), not the tag object, is pinned:

```text
actions/checkout              v4       34e114876b0b11c390a56381ad16ebd13914f8d5
actions/checkout              v6       df4cb1c069e1874edd31b4311f1884172cec0e10
actions/deploy-pages          v5       cd2ce8fcbc39b97be8ca5fce6e763baed58fa128
actions/download-artifact     v4       d3f86a106a0bac45b974a628896c90dbdf5c8093
actions/setup-go              v5       40f1582b2485089dde7abd97c1529aa768e1baff
actions/setup-python          v6       ece7cb06caefa5fff74198d8649806c4678c61a1
actions/upload-artifact       v7       043fb46d1a93c77aae656e7c1c64a875d1fc6a0a
actions/upload-pages-artifact v4       7b1f4a764d45c48632c6b24a0339c27f5614fb0b
astral-sh/setup-uv            v6       d0cc045d04ccac9d8b7881df0226f9e82c39688e
matlab-actions/run-command    v3       bcd446a219949c24051c1d04a4b4e0274c42f23b
matlab-actions/setup-matlab   v3.0.1   a0180c939fb1a28de13f44f7b778b912384ced1f
mattnotmitt/doxygen-action    v1.12    b84fe17600245bb5db3d6c247cc274ea98c15a3b
openjournals/draft-action     master   85a18372e48f551d8af9ddb7a747de685fbbb01c
peaceiris/actions-hugo        v3       2752ce1d29631191ea3f27c23495fa06139a5b78
pypa/cibuildwheel             v3.4.1   8d2b08b68458a16aeb24b64e68a09ab1c8e82084
```

The prior Doxygen reference `v1.12.0` returned no tag; v1.12 is the actual
upstream release. The pre-existing Codecov v5 commit remains pinned.

Arrow was downloaded twice from the exact CPM URL:

```text
first =4C898504958841CC86B6F8710ECB2919F96B5E10FA8989AC10AC4FCA8362D86A
repeat=4C898504958841CC86B6F8710ECB2919F96B5E10FA8989AC10AC4FCA8362D86A
```

Green gates:

```text
uv run --no-sync python scripts/check_supply_chain_pins.py
  supply-chain pins verified
PyYAML safe_load(all .github/workflows/*.yml)
  workflow YAML parsed
rg mutable uses-pattern .github/workflows
  no matches
git diff --check
  exit 0
```

Verdict: **PASS.** The new job runs on the active Python workflow and the docs
workflow invokes the same gate. Full SHAs are the immutable execution identity;
human-readable release refs remain comments for update tooling and review.

## M4 — SSG path multiplicity and stable true-gradient steps

Schultz–Jain define the averaged objective `F=(1/N) sum DTW²` and a sampled
component gradient `2(Vz-Wx)`. Uniformly sampling occurrences makes that an
unbiased gradient estimator, so no explicit `1/N` belongs in each step; `V_ii`
is the number of path matches for center coordinate i. Their pseudocode absorbs
the factor two into eta but does not divide coordinates by their valence.

Registered one-step oracle: resampling `{0,2,4,10}` to center `{0,10}` gives a
unique path whose first coordinate has valence 3 and aligned sum 6. At eta=.1,
the correct first coordinate is 1.2:

```text
old coordinate-mean update: actual 0.2, expected 1.2 (RED)
true component gradient:    actual 1.2, expected 1.2
```

Before accepting that fix, an orthogonal length-ratio attack found a second
failure in the proposed raw-gradient implementation:

```text
single ramp length 100 -> target length 2, default eta=.2
initial center {0,99}, first raw step {490,-391}
objective 80,850 -> 2.86e71 after 50 iterations (RED)
```

For a selected fixed path, the quadratic Hessian is diagonal `2V`, so its
gradient Lipschitz constant is `L=2 max(V_ii)`. The final implementation caps
the one scalar step at `1/L`; unlike the old coordinate-wise division, this
preserves the raw gradient direction and all relative multiplicities.

Green adversarial outcomes:

```text
length 100:  center {24.5,74.5}, objective 80,850 -> 20,825
length 1000: center {249.5,749.5}, objective 83,083,500 -> 20,833,250
unit_test_barycenter.exe --reporter compact
  All tests passed (77 assertions in 10 test cases)
ctest --test-dir build/highs-1151 -C Release \
  -R "^unit_test_barycenter$" --output-on-failure
  1/1 passed, 0 failed
```

Verdict: **PASS after adversarial refinement.** The original finding was real,
and naively restoring the paper gradient would have introduced a much larger
unequal-length bug. The scalar cap is direction-preserving, documented in both
public option structs, and pinned by 100:2 plus 1000:2 regressions.

## M5 — soft-DTW cross-implementation boundary

Registered band: use unequal-length binary series (5×3), so every local pair
satisfies `|x-y|=(x-y)^2`, and compare the independent public-L1 and
barycenter-squared forward passes at gamma .1, 1, and 2.5 within absolute
`1e-12`. A nonbinary case must remain separated by more than 1 to prove the
test has not erased the intended semantic distinction.

Green unchanged-production result:

```text
unit_test_barycenter.exe "[cross_implementation]" --reporter compact
  All tests passed (4 assertions in 1 test case)
ctest --test-dir build/highs-1151 -C Release \
  -R "^unit_test_barycenter$" --output-on-failure
  1/1 passed, 0 failed
```

Mutation check: changing the barycenter local cost from squared to absolute
left all three shared-cost comparisons green but made the sensitivity assertion
fail exactly:

```text
failed: abs(squared_value - l1_value) > 1.0
actual: 0.0 > 1.0
```

Verdict: **PASS (coverage/architecture finding, no arithmetic bug).** The 8.3
decision is to factor one recurrence and adjoint framework behind explicit
local-cost/value-derivative policies. Public `soft_dtw` remains L1; barycenter
remains squared-cost. This avoids duplicated dynamic programming without a
silent numerical API change.

## L4 — separate MATLAB serial/OpenMP honesty gates

Registered band: the OpenMP build must report available/pass, empty reason, and
at least two engaged threads. The explicit sequential build must separately
report unavailable/fail, one max/engaged thread, and a nonempty reason naming
the sequential build. Each flavor skips only the opposite case, and the MATLAB
wrapper must equal the raw MEX struct exactly.

Red on the fresh sequential artifact before restoring the separate case:

```text
test_test_api.m: 5 passed, 2 failed
OpenMP-only available/pass assertions rejected the truthful sequential report
```

Artifact provenance was established before any four-way run. A stale serial
MEX that predated the version command was rejected and rebuilt. Final binaries:

```text
serial: build/mex-verify/bin/dtwc_mex.mexw64
  Clang/Ninja, DTWC_ALLOW_SEQUENTIAL=ON, OpenMP discovery disabled
  SHA256 93A49471292CAFC7D9BB5A89B4829E495A42623ECAFB3B75A3BA6D04E5B9AD60
OpenMP: build/mex-verify-msvc/bin/dtwc_mex.mexw64
  MSVC Release /openmp:experimental, imports VCOMP140.DLL
  SHA256 595169152C9A0D4BC74061B371A90BCBB5300424D442DE150DA759C93A704E6D
```

Every command began with `restoredefaultpath`, added `bindings/matlab`, added
the selected build directory last, cleared `dtwc_mex`, and printed `which` so
only the intended artifact resolved. Results:

```text
R2024b serial:  7 passed, 0 failed, 1 OpenMP-case skip
R2024b OpenMP:  7 passed, 0 failed, 1 serial-case skip
R2025b serial:  7 passed, 0 failed, 1 OpenMP-case skip
R2025b OpenMP:  7 passed, 0 failed, 1 serial-case skip
serial report: available=0 pass=0 max=1 engaged=1 reason contains "sequential"
OpenMP report: available=1 pass=1 max=24 engaged=24 reason empty
```

Verdict: **PASS.** Flavor assumptions isolate capability-specific assertions
without converting an unsupported capability into a false pass. Exact wrapper
struct equality prevents the MATLAB facade from embellishing the MEX truth.

## M11 — TADPole CLI distance-storage routing

Registered band: at a fired threshold, TADPole must execute through an actual
`MmapDistanceMatrix` on LLFIO builds; without LLFIO it must throw before the
dense packed vector allocates. OneBatchPAM must remain unallocated and exempt.

Red-first compilation failed because the test named the not-yet-existing policy
seam `configure_cli_distance_storage`. After extraction, both configurations
exercise the same CLI function that main calls after method resolution.

Green evidence:

```text
LLFIO ON  unit_test_cli_args: 56 assertions / 11 cases passed
LLFIO OFF unit_test_cli_args: 54 assertions / 11 cases passed
both builds, ctest -R "unit_test_cli_args|unit_test_tadpole|unit_test_variant_distmat"
  3/3 passed, 0 failed
```

End-to-end CLI evidence:

```text
LLFIO ON:  exit 0; TADPole completed; mmap cache file created
LLFIO OFF: exit 1; error names mmap support, DTWC_ENABLE_LLFIO=ON,
           threshold/RAM tradeoff, and onebatch alternative;
           no cache or checkpoint created
```

Verdict: **PASS.** TADPole is matrix-free in scheduling/result-surface terms,
not in worst-case cache capacity: its lazy exact/fallback calls can touch O(N²)
pairs. The CLI now gives that cache the intended file backing at large N and
never silently substitutes a dangerous heap allocation when the capability is
unavailable.

Adjacent audit finding retained as M14: the mmap header validates magic,
version, CRC, and N but not dataset/configuration identity. Same-N stale cache
reuse is a separate correctness defect and is not disguised as part of this OOM
repair.

## M12 — device-aware Tier-1 auto method

Registered policy at the 5000/5001 boundary:

```text
CPU:        N=5000 -> pam; N=5001 -> clara
CUDA/Metal: N=5000 -> pam; N=5001 -> pam
Python HPC: preserve auto for remote post-materialisation resolution
C++ HPC:    existing loud transport-boundary DeviceError before local work
explicit method: unchanged (so explicit clara+GPU remains loud)
```

Red before backend-aware resolution:

```text
C++ policy N=5001/GPU: actual clara, expected pam (1 failed assertion)
Python: CUDA policy, Metal policy, and mocked full CUDA route all failed;
        the full route raised DeviceError after resolving auto to CLARA
```

Green evidence:

```text
uv run --no-sync pytest tests/python/test_api.py -q
  45 passed
uv run --no-sync pytest tests/python -q
  489 passed, 11 skipped
ctest --test-dir build/highs-1151 -C Release \
  -R "test_tier1_cpp_api|test_env_device|test_device_loudness|test_global_device|test_problem_device" \
  --output-on-failure
  5/5 passed, 0 failed
```

The Python full-route test uses 5001 real series objects but mocks GPU discovery,
matrix computation, and clustering: it asserts `gpu:3` survives resolution, a
matrix is produced, PAM receives it, and a poisoned CLARA path is never called.

Verdict: **PASS.** Resolution now precedes matrix-policy selection in both
languages. Task 8.3 will bind the pure C++ `resolve_tier1_method` seam privately
through nanobind and delete Python's mirror/threshold constant; this commit keeps
that later refactor behavior-neutral.

## L3 — loud OneBatchPAM batch-size correction

Registered behavior:

```text
explicit batch_size < k: exact warning per invocation; effective m=k
explicit batch_size >= k: no stderr
batch_size=-1 auto: no stderr, including internal max(auto,k)
```

Red before the warning:

```text
failed: stderr_output == expected
actual: ""
```

Green exact line (emitted twice for two calls in the regression):

```text
[dtwc] warning: one_batch_pam requested batch_size=2, but n_clusters=4 requires
batch_size >= 4; using effective batch_size=4. Set batch_size to at least
n_clusters to avoid this adjustment.
```

Focused and suite results:

```text
unit_test_one_batch_pam.exe "[loudness]" --reporter compact
  All tests passed (6 assertions in 1 test case)
unit_test_one_batch_pam.exe
  All tests passed (597 assertions in 6 test cases)
ctest --test-dir build/highs-1151 -C Release \
  -R "^unit_test_one_batch_pam$" --output-on-failure
  1/1 passed, 0 failed
```

Verdict: **PASS.** A process-once guard was rejected: different later calls can
request different invalid sizes, and suppressing them would recreate a silent
configuration change. A mutex serializes whole per-use lines for concurrent
callers without hiding information.

## L5 — objective-matched FastPAM BUILD sampling

The code-path audit replaced all five residual `distByInd` calls in
`fast_pam.cpp` with the canonical `dist_by_ind` API. The sampling policy is
intentionally not textually identical to barycenter k-means: PAM minimizes
`sum(DTW)`, so its current nearest objective contributions are sampled as D;
the barycenter implementation's `align_squared` values already are its
squared-local-cost objective contributions.

The registered discriminator uses singleton values `{0,1,3}`, `k=2`, and
seeds 0…4095. Conditional on point 0 being the first medoid, correct D weights
select point 3 with probability 3/4, while the wrong D² mutation targets 9/10.
The fixed seed census is far from either threshold:

```text
production D: conditioned=1357 selected_far=1017 fraction=0.749447
mutated D^2:  conditioned=1357 selected_far=1224 fraction=0.90198968312453942
registered production band: 0.70 < fraction < 0.80
```

The intentional production mutation `distance *= distance` failed exactly at
the upper assertion:

```text
far_fraction < 0.80
0.90198968312453942 < 0.80000000000000004
test cases: 1 | 1 failed
assertions: 3 | 2 passed | 1 failed
```

After restoring D weights:

```text
unit_test_fast_pam "[fast_pam][seeded][initialization]" --reporter compact
  All tests passed (3 assertions in 1 test case)
unit_test_fast_pam --reporter compact
  All tests passed (67 assertions in 12 test cases)
```

Verdict: **PASS.** The two initializers share the higher-level rule “sample
proportional to the current objective contribution”; squaring FastPAM's
already-objective distances would optimize the wrong seeding surrogate.
