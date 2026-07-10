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

## M6 — nondegenerate OneBatchPAM 50k validation

The replacement fixture has five groups, 100 unique warped profiles per group,
and 100 exact replicas of each profile: 50,000 series total. Profile lengths
cover every integer from 64 through 128 (mean 96.3); the nonlinear clock is
monotone because its derivative is bounded below by 0.59. Groups are translated
by 1,000 while every unshifted profile value is bounded by 3.1.

The oracle is globally exact. Omitting any group costs at least 6,360,320 per
replica, while a feasible one-medoid-per-group solution costs at most 790,500.
Thus every optimum represents all five groups, and equal multiplicities plus
translation invariance reduce the problem to an exhaustive search over the 100
unique within-group profiles. Variant 9 is optimal at
2,851.0245324230636 per replica. Deliberately choosing worst variant 19 costs
13,912.043591653444, or 4.8796646375503849× the oracle, so the registered 1.05×
quality band catches the mutation decisively.

The 5,000-series structural preflight ran before the large simulation:

```text
M6_PREFLIGHT n=5000 variants=100 replicas=10 lengths=64..128 band=8 batch=256
wall_s=1.0729183 evaluations=1304739 max_evaluations=1304739
fraction=0.052189560000000003 cost=28510.245324230687
exact_oracle=28510.245324230636 ratio=1.0000000000000018 accepted_swaps=30
All tests passed (10019 assertions in 1 test case)
```

Registered before the 50k run: the N×m table is 97.65625 MiB; series payload
is 36.7355 MiB; work is at most 13,049,739 DTWs (0.52198956% of N²) and
28,396,242,944 conservative banded DP cells; peak RSS must remain below the
advisory 300 MiB band; objective must be within 1.05× the exact oracle.

First run:

```text
M6_50K n=50000 variants=100 replicas=100 lengths=64..128 band=8 batch=256
wall_s=10.071664 evaluations=13049739 max_evaluations=13049739
fraction=0.0052198955999999998 cost=285102.45324230636
exact_oracle=285102.45324230636 ratio=1 accepted_swaps=35
All tests passed (100019 assertions in 1 test case)
```

Monitored deterministic repeat:

```text
M6_MONITOR external_wall_s=39.594735 peak_rss_bytes=163254272
peak_rss_MiB=155.691 exit_code=0
M6_50K wall_s=14.102904100000002 evaluations=13049739
fraction=0.0052198955999999998 cost=285102.45324230636
exact_oracle=285102.45324230636 ratio=1 accepted_swaps=35
medoids=10900,30900,900,20900,40900
All tests passed (100019 assertions in 1 test case)
```

The repeat began under unrelated host load, so timing remains advisory; its
internal time stayed inside the preregistered 5.4–21.5 second scaling interval.
Work, objective, ratio, swap count, and one-variant-9 medoid per group repeated
exactly. The non-hidden suite also passed 10,235 assertions in seven cases.

Verdict: **PASS.** The release gate now exercises real unequal-length DTW work,
has a proved exact global oracle, and measures a tight implementation-derived
work fraction rather than relying on the old length-1 special case.

## M14 — semantic mmap distance-cache identity

The M11 follow-up reproduced the defect before implementation: a v1 mmap header
identified only magic/version/endianness/element-size/N, so five same-N cases
reopened cached bits under changed data or distance semantics. A deliberately
constructed legacy-v1 file also opened under the old reader. The registered red
suite therefore had five stale-reuse failures plus the legacy-format failure.

The replacement 64-byte v2 header stores a SHA-256 identity and CRC-protects all
metadata. The identity includes raw IEEE values, series order/lengths, storage
precision and `ndim`, band, all variant parameters and multivariate mode,
missing-data policy, pointwise metric, backend, and backend precision. Names are
excluded because they cannot alter a distance. Exact file length, reserved bytes,
algorithm ID, version, endian marker, element size, and CRC are all checked before
the computed-bit region is exposed. CUDA `Auto` precision is rejected because
the runtime GPU would make persistence semantics ambiguous; non-L1 caches are
external-fill-only on the L1 CPU path.

Mutation review found that hashing the full data on every pair lookup would turn
warm access into O(series length). The final design validates the full identity
at bind and once at first use, then checks an immutable-width configuration
snapshot. Semantic setters detach the mapping; naked config edits are caught on
every access. Raw in-place data edits after first validation are explicitly
unsupported and documented to require refresh-before-edit or `set_data()`.

Green evidence from clean authoritative builds:

```text
LLFIO ON focused mmap/identity/CLI/SHA suites:
  4/4 tests passed, 0 failed (10.51 s)
LLFIO OFF focused suites:
  3 passed + 1 capability skip, 0 failed (4.14 s)

O(1) lookup discriminator, 100000 cached reads each:
  length=1:    3284600 ns
  length=4096:  734600 ns
  registered upper bound: 46276800 ns
  3 assertions passed
```

The CLI now applies band/variant/multivariate/backend/precision settings before
binding storage, carries the selected pointwise metric into the identity, writes
GPU results through either dense or mmap storage, and rejects dense CSV
`--checkpoint`/`--dist-matrix` combinations at the mmap threshold before opening
either path. A production Metal mmap test was reordered to bind only after its
backend setting. `git diff --check` was clean.

Verdict: **PASS.** Warm-start reuse is now conditional on exact distance
semantics rather than filename and N. Version-1 caches are intentionally invalid
and have actionable recomputation guidance; the frozen-format exception is
recorded in the PLAN decision log and migration documentation.

## L6 — reusable and deterministic parallel barycenter workspaces

Before editing, a mixed-length, three-cluster SSG fixture pinned labels, every
center value, total cost `3.92461431099334757`, iteration count, and convergence
with exact equality (5/5 assertions). A separate DBA allocation probe used three
series of lengths 257/255/253, target length 257, and one update. It counted heap
requests at least 500 KiB, exposing the original allocation schedule:

```text
pre-edit exact fingerprint: 5/5 assertions passed
pre-edit large DP allocations: 9
registered post-edit requirement: 1
allocation oracle: FAILED as intended
```

The implementation now passes one caller-owned matrix/path workspace through
hard objectives, DBA, SSG, k-means++ and assignment. `barycenter_kmeans` owns a
prewarmed worker pool; all capacity/overflow failures occur before OpenMP.
Assignments write disjoint point slots and retain a serial index-order cost sum.
Cluster membership and order-sensitive empty repair remain serial, while only
independent non-empty center updates run in parallel. Each update keeps the old
`seed + iteration*k + cluster` stream, exceptions are captured per cluster and
re-thrown in serial cluster order, and a nested OpenMP caller forces one worker.

The hard scratch bound is independent of cluster count:

```text
W * [8*Ld*max(Ld,Lt) + sizeof(index_pair)*(Ld+max(Ld,Lt)-1)]
W = min(max_threads, N)
```

Post-edit hard gates:

```text
large DP allocations: 1 (2/2 allocation assertions passed)
exact fingerprint, OMP_NUM_THREADS=1:  5/5
exact fingerprint, OMP_NUM_THREADS=2:  5/5
exact fingerprint, OMP_NUM_THREADS=24: 5/5
full barycenter suite: 86 assertions / 12 cases passed
```

The adversarial mutation reversed the cluster-to-RNG-stream mapping, simulating
the most likely unsafe scheduling refactor. Labels happened to stay fixed, but
the exact center fingerprint failed; restoring cluster-indexed streams returned
all thread counts to green.

Only after those correctness checks, the preregistered workload (N=128, k=8,
lengths 80–95, target 88, four outer and four barycenter iterations) ran one
warmup plus five timed samples per configuration. Registered bands were post
serial ≤1.25× pre serial, both parallel medians ≤1.25× post serial, and at least
one parallel median ≤0.95× post serial:

```text
pre-edit serial median: 0.179682100 s
post OMP=1 median:      0.184484300 s  (1.027x pre)
post OMP=2 median:      0.117212400 s  (0.635x post-1)
post OMP=24 median:     0.049031500 s  (0.266x post-1)
```

Labels, every center, cost `3.0910791478147139`, iterations, and convergence
were bit-identical between the post-edit 1/2/24-thread runs and the pre-edit
fingerprint. Timing remains advisory on the shared host, but every registered
band passed. The two direct-extension timing modules both emitted the same
nanobind teardown diagnostic. M18 later isolated this to the timing harness's
unregistered `importlib` module lifetime, not a DTWC binding leak; see below.

Verdict: **PASS.** The hot path removes repeated large allocation, provides real
parallel speedup, and preserves the serial numerical/RNG contract exactly.

## M13 — cross-language invocation-local random defaults

The pre-edit audit found four incompatible public stories: C++ Tier-1 PAM fixed
seed 29, Python and MATLAB PAM consumed the mutable process-global FastPAM
engine, sklearn treated `random_state=None` as 42, and CLI `--seed` affected
CLARA/OneBatchPAM but not PAM. The first red gates were structural rather than
well-separated-output tests: Python lacked `DEFAULT_RANDOM_SEED` in both public
modules (2/2 failures), and the C++ contract fixture did not compile because no
shared constant existed. A source audit then found all three FastCLARA branches
(in-memory sample, chunked sample, and full-data fallback) still called unseeded
FastPAM; a red regression proved those calls advanced the global engine.

The discriminating fixture is eight translated copies of the nonconstant
waveform `[0, .01, -.02, .03]`, clustered at k=3. It intentionally avoids both
the old length-one DTW shortcut and a well-separated case whose optimum would
hide initialization differences:

```text
seed 29: initial [4,2,7], final medoids [4,1,7], cost 20
seed 42: initial [6,2,5], final medoids [6,2,5], cost 24
seed 43: second restart reaches cost 20
```

`settings::DEFAULT_RANDOM_SEED`, `dtwcpp.DEFAULT_RANDOM_SEED`, and
`dtwc.default_random_seed()` now expose 42. C++/Python/MATLAB Tier-1 PAM and the
seed-aware OneBatchPAM/CLARA routes construct local engines. FastCLARA seeds its
internal PAM solve with `base_seed + sample_index` identically in chunked and
in-memory paths and uses the base seed for its full-data fallback. Estimator
restart `i` uses `42+i` and retains the strict best objective. MATLAB's optional
`Seed` preserves omitted-seed Tier-2 compatibility and validates finite integral
ranges before conversion; CLI accepts `[0, UINT_MAX]` and applies the same flag
to PAM, OneBatchPAM, and CLARA.

The legacy boundary is deliberate: `dtwc::randGenerator` remains a mutable
`std::mt19937` initially seeded 29, and the unseeded Tier-2 FastPAM overload
continues to consume it. New non-consumption tests place that engine in different
states before identical Tier-1/FastCLARA calls and verify both equal results and
bit-identical engine state afterward.

Final evidence after the FastCLARA closure:

```text
focused CTest (Tier-1 API, CLI args, FastCLARA): 3/3 passed
FastCLARA full executable: 816 assertions / 18 cases passed
isolated built-extension Python API/sklearn/cross-validation/HPC: 93 passed
MATLAB test_contract_parity.m: assertSuccess passed
fresh MEX: build/mex-verify-msvc/bin/dtwc_mex.mexw64
           501248 bytes, timestamp 2026-07-10 15:00:22
targeted py_compile: passed
git diff --check: clean
```

The audit also found two distinct follow-ups and kept them out of this claim:
M16 owns ignored CLI/HPC restart propagation; M17 owns Lloyd initialization and
MIP warm starts that still use the legacy engine.

Verdict: **PASS.** Every seed-aware route in M13's stated scope is reproducible
per invocation and cross-language, while the documented legacy overload remains
source- and behavior-compatible.

## M17 — Lloyd and direct-MIP invocation-local seeds

The M13 audit left two standard Tier-1 paths outside its claim. Lloyd called
the public one-argument `init::random` through `Problem::init_fun`, and both
direct solver backends called unseeded FastPAM for their MIP incumbent. A third
bug appeared while preregistering restarts: Lloyd wrote the best repetition
number but left the final repetition's labels/medoids in `Problem`.

The red fixture reused M13's eight translated copies of the nonconstant
waveform `[0, .01, -.02, .03]` at k=3. Different legacy-engine states changed
both Lloyd and the solver warm-start construction. The warm-start medoids were
`[3,6,0]` versus `[6,3,1]`, and four of five assertions failed. A controlled
two-repetition Lloyd initializer made repetition 0 converge to medoids
`[0,3,6]`, cost 20, then repetition 1 to `[0,2,5]`, cost 24. The implementation
printed `Best repetition: 0` but returned repetition 1's cost and medoids:

```text
problem.find_total_cost() == 20.0  -> 24.0 == 20.0
problem.medoids() == {0,3,6}       -> {0,2,5} == {0,3,6}
test cases: 1 | 1 failed
assertions: 4 | 2 passed | 2 failed
```

`Problem::random_seed` now defaults to `DEFAULT_RANDOM_SEED` (42). The private
Lloyd dispatch recognizes only the standard `init::random` and
`init::Kmeanspp` function-pointer targets and calls new local-engine overloads;
any arbitrary public `init_fun` callback is invoked unchanged once per
repetition. The restart range is checked before matrix work, restart `i` uses
`seed+i`, and repetition zero is snapshotted unconditionally so an infinite or
NaN comparison cannot leave the restored vectors empty. The actual strict-best
labels, medoids, and iteration count are restored at the end.

`mip::make_warm_start` is the single FastPAM incumbent seam used by HiGHS and
Gurobi. It takes the `Problem` seed explicitly and preserves the old 100-iteration
FastPAM warm-start limit. Solver models, tuning, objectives, and extraction were
untouched. The one-argument initializers and unseeded Tier-2 FastPAM still use
the mutable seed-29 `std::mt19937`.

Green evidence from fresh post-edit artifacts:

```text
build/phase8-m13 (Clang 21.1.8, Release, HiGHS/Gurobi/LLFIO OFF)
  [lloyd]:                 16 assertions / 4 cases passed
  [seed] MIP seam:          9 assertions / 1 case passed
  full Tier-1 C++ API:     50 assertions / 9 cases passed
  initializer suite:      56 assertions / 10 cases passed
  FastPAM suite:           67 assertions / 12 cases passed

build/highs-1151 (HiGHS 1.15.1 ON, Gurobi 13.0.1 ON, LLFIO ON)
  both backend sources and shared seam compiled and linked
  full unit_test_mip:      55 assertions / 11 cases passed

fresh isolated build/phase8-m13 Python extension
  tests/python/test_api.py: 48 passed
```

The solver-runtime gate includes exact cold/warm agreement, Benders, and a
global-engine non-consumption assertion. Gurobi's licensed runtime was not
needed for this finding; its enabled source compiled against 13.0.1 and the
static audit finds exactly the same shared call in both backend files.

The audit separately found Benders leaking its temporary `N_repetition=1`
configuration. That broader post-call state-corruption bug is registered as
M20 rather than folded into this deterministic direct-solver change.

Verdict: **PASS.** Lloyd and direct MIP warm starts are reproducible per
invocation, exact solver optima are unchanged, best-restart output is truthful,
and the explicit legacy/custom-initializer compatibility boundary is preserved.

## M21 — LF-safe shell entrypoints on Windows

The M16 SLURM last-mile syntax gate exposed a checkout-level failure before any
remote work could begin. The repository had no `.gitattributes`, the local Git
configuration used `core.autocrlf=true`, and tracked LF blobs were materialized
as CRLF or mixed worktree files. Git Bash rejected the documented entrypoints:

```text
scripts/slurm/slurm_remote.sh: line 49: syntax error near unexpected token `done'
scripts/slurm/jobs/cluster_generic.slurm: line 43: syntax error near unexpected token `$'{\r''
```

This was not a shell-logic defect: `git ls-files --eol` showed LF index blobs.
The fix is a repository checkout contract, not an environment workaround:

```gitattributes
*.sh text eol=lf
*.slurm text eol=lf
```

All tracked matching files were normalized before validation. The preregistered
gate enumerated the repository rather than sampling only the two M16 files:

```text
tracked *.sh + *.slurm: 11
git ls-files --eol:      11/11 i/lf, w/lf, attr/text eol=lf
bash -n:                 11/11 passed
```

No script content or runtime behavior belongs to M21; M16's separate wrapper and
job arguments remain unstaged for its own semantic commit.

Verdict: **PASS.** Fresh Windows checkouts now preserve parseable LF entrypoints,
and every current shell/SLURM script passes the same syntax gate.

## M20 — exception-safe Benders warm-start state

The M17 audit found that Benders runs a nested Lloyd solve after temporarily
changing `Problem::method` and `Problem::N_repetition`. The old success cleanup
restored method, medoids, and labels by assignment, but leaked the forced
restart count and Lloyd's `last_iterations`; any exception skipped all cleanup
and leaked the nested method, restart count, medoids, and labels as well.

The regression snapshots every caller-visible configuration/data field and the
fully materialized distance matrix before entering Benders. The success case
uses non-default sentinels and permits only the intended final exact
labels/medoids. The exception case points Lloyd's result writer at a unique
nonexistent directory, so the failure occurs only after Lloyd has converged and
mutated its working result. Before remediation, the focused gate failed exactly
the six leaked-state checks:

```text
build/highs-1151/bin/unit_test_mip.exe "[state]" --reporter compact
success: N_repetition 1 != 4; last_iterations 3 != 77
throw:   method Kmedoids != MIP; N_repetition 1 != 5
         labels {0,0,0,1,1,1,1,0} != the saved all-1 sentinel
         medoids {1,4} != the saved {6,7} sentinel
test cases: 2 | 0 passed | 2 failed
assertions: 87 | 81 passed | 6 failed
```

`BendersWarmStartStateGuard` now copies every field the nested Lloyd operation
can mutate before changing any of them. Its `noexcept` destructor restores the
three scalar fields and swaps the saved vectors, so unwinding performs no
allocation. The guard's scope ends before the exact Benders loop writes its
final medoids and labels.

Green evidence from rebuilt post-edit artifacts:

```text
build/highs-1151 (HiGHS 1.15.1 ON)
  [state]:              87 assertions / 2 cases passed
  full unit_test_mip:  142 assertions / 13 cases passed

build/phase8-m13 (HiGHS/Gurobi OFF)
  unit_test_mip target compiled and linked
  [seed]:                9 assertions / 1 case passed
```

The successful state test additionally checks that Benders returns two valid
medoids, one label per input, and a finite nonnegative final objective. The
forced exception is matched to the expected result-file open failure, proving
that restoration covers a real post-mutation unwind rather than an early
synthetic throw.

Verdict: **PASS.** Nested Lloyd configuration and working results no longer
escape Benders on success or failure, and the final exact clustering result is
preserved.

## M16 — effective CLI and Python-HPC PAM restarts

The M13 audit found that CLI11 parsed `--n-init` and main stored it in
`Problem::N_repetition`, but the PAM dispatch called seeded FastPAM exactly once.
The Python estimator's local route already used `42+i`; its HPC route discarded
the count before `_hpc.cluster_on_hpc`. The complete pre-edit transport trace was:

```text
DTWClustering.fit
  -> cluster_on_hpc
  -> SlurmRemoteRunner.submit_cluster
  -> slurm_remote.sh submit-cluster
  -> sbatch --export
  -> cluster_generic.slurm
  -> dtwc_cl                 (no n_init or seed after the first arrow)
```

The registered translated-waveform fixture failed red exactly where intended:
the seed-42 first run cost 24, seed 43 cost 20, but the two-restart CLI seam still
returned 24. Python red tests independently found the missing estimator, command
builder, and runner arguments.

CLI PAM now reads the seed and repetition count from `Problem`, validates the
entire uint64 schedule before distance work, runs invocation-local `seed+i`, and
replaces the incumbent only for a strictly smaller objective. Ties therefore
retain the earliest seed. CLI/YAML counts must be positive. The Python transport
rejects booleans, non-integral values, CLI int/unsigned range violations, and
uint64 schedule overflow before creating a run directory or contacting a runner.

`build_dtwc_command`, `cluster_on_hpc`, and `submit_cluster` accept the same
schedule. The wrapper validates decimal positionals without signed-shell
overflow, exports `DTWC_N_INIT` and an optional `DTWC_SEED`, and the job always
passes `--n-init` plus `--seed` only when present. `seed=None` deliberately omits
the flag, preserving the C++ CLI as the default's single source of truth. The
estimator resolves its public default to 42 before dispatch. Binary discovery now
also considers nested `build/*/bin` verification trees, preventing an older
top-level executable from falsifying local end-to-end tests.

Green evidence from the fresh HiGHS-enabled CLI and editable Python source:

```text
build/highs-1151
  focused [n_init]:              19 assertions / 1 case passed
  full unit_test_cli_args:       90 assertions / 16 cases passed

tests/python/test_hpc.py:        29 passed
  includes estimator -> _hpc -> runner schedule assertions
  includes executable job-script capture with seed=42 and seed omitted
  includes real local dtwc_cl cost 24 -> cost 20 and n_init=0 rejection

tests/python/test_cross_validation.py restart/range gate:
                                  6 passed / 15 deselected
targeted py_compile:              passed
bash -n wrapper + final job:      passed
git diff --check:                 clean
```

The C++ focused case repeats the two-restart result three times and matches seed
43's medoids, labels, cost, iteration count, and convergence exactly. It also
pins the `n_init=0` and `UINT64_MAX + 1 restart` error text. The executable job
uses a fake binary only for argument capture; SSH/rsync/sbatch are intentionally
not contacted on the development machine.

Verdict: **PASS.** `n_init` now changes PAM work and results consistently on the
local CLI and Python HPC route, with deterministic best-result retention and no
silent seed-default override.

## M18 — nanobind teardown diagnostic (FALSIFIED as a normal-use leak)

L6's pre/post timing processes both exited 0 after printing the same nanobind
shutdown warning: 3 instances, 33 types, and 325 functions. The three instances
were `BarycenterOptions`, `BarycenterClusteringOptions`, and
`HierarchicalOptions`. The exact archived commands proved that pre and post ran
in separate Python processes, each loading one extension as `_dtwcpp_core`:

```python
spec = importlib.util.spec_from_file_location("_dtwcpp_core", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
```

The harness did not place the module in `sys.modules`. A minimal load with no
DTWC call reproduced the full 3/33/325 diagnostic, so neither barycenter work
nor L6's parallel changes caused it. Controlled subprocesses then separated the
loader lifetime from the binding:

```text
case                                           exit          stderr
unregistered direct load, no DTWC call         0             3 instances / 33 types / 325 functions
same load with sys.modules[spec.name] = m       0             empty
normal package import + all 3 defaulted calls  0             empty (stdout: 3 3 2)
unregistered load, delete 3 defaulted funcs    0             empty
two extension copies in one interpreter        0xC0000409    duplicate Device/CPU registration abort
```

The normal-use case constructed a `Problem`, invoked `dtw_barycenter` and
`barycenter_kmeans` without explicit options, filled its distance matrix, and
invoked `build_dendrogram` without explicit options. It therefore exercised all
three objects named by the warning, not merely an import-only happy path.

The counts have an exact source explanation. `_dtwcpp_core.cpp` has exactly
three bound custom-object defaults:

```text
dtw_barycenter(..., options=BarycenterOptions{})
barycenter_kmeans(..., options=BarycenterClusteringOptions{})
build_dendrogram(..., opts=HierarchicalOptions{})
```

Nanobind converts each default into a Python instance stored by the bound
function. Its `nb_func_dealloc` decrements every `arg.value` during orderly
module teardown. The installed nanobind 2.12.0 `internals_cleanup()` runs from
`Py_AtExit`, counts still-live `inst_c2p` entries, and deliberately prints type
and function registries only when an instance/keep-alive record remains. Thus
the 33 types and 325 functions were consequences of those three module-owned
defaults still being alive, not 358 independent leaks. Registering the module
lets normal interpreter module cleanup release the functions/defaults before
the leak checker runs. The harness's unregistered local module instead remained
live through that check.

Both extensions were Release builds and `NB_ABORT_ON_LEAK` was absent, which
explains the warning with exit 0 and falsifies the debug-build hypothesis. The
two-copy control was also not the original scenario: two copies in one
interpreter fail earlier and loudly because the same nanobind types/enumerators
cannot be registered twice.

The ownership audit found no `keep_alive` policies. The only reference return
policy is `env()`, which exposes a process-static singleton, and every allocated
NumPy buffer has a deleting capsule owner. The custom exception references are
also process-lifetime translator state and do not appear in nanobind's instance
tracker.

Verdict: **FALSIFIED.** Normal one-module DTWC import/use exits cleanly with no
nanobind diagnostic. No production change, PLAN finding, or suppressing leak
checker call is warranted. Future direct-extension harnesses must insert the
module into `sys.modules` before `exec_module` (or use normal import machinery)
so interpreter teardown owns its lifetime.

## M22 — Benders nested-Lloyd persistence isolation

M20's forced-unwind discriminator exposed a separate behavior defect: Benders
used the public Lloyd entry point for its internal incumbent and therefore
wrote `medoids_rep_0.csv` plus `_bestRepetition_Nc_*.csv` into the caller's
result directory. Those files describe the heuristic intermediate result, not
the exact Benders solution, and an unwritable output path could abort an exact
solve before the solver started.

The registered discriminator uses independent unique empty directories for a
Benders solve and a direct public Lloyd solve on the same N=10, length-16,
k=2, seed-1234 fixture. It requires the nested files to be absent, while pinning
the existing `Best repetition: 0` line, warm-start cost prefix `22.498`, and
four-iteration Benders convergence. The direct call must still create both
documented artifacts and print its best-repetition line. Before remediation,
only the two Benders absence assertions failed:

```text
build/highs-1151/bin/unit_test_mip.exe "[io]" --reporter compact
failed: !exists(nested_medoids)   for: !true
failed: !exists(nested_best_rep)  for: !true
test cases: 1 | 1 failed
assertions: 9 | 7 passed | 2 failed
```

The public `Problem::cluster_by_kmedoids_lloyd()` signature and behavior remain
unchanged. It forwards to a private implementation with persistence enabled.
Only the single `MIP_clustering_byBenders(Problem&)` friend can select the
disabled mode. Both modes execute the same distance fill, seed/restart loop,
assignment/update arithmetic, best-state retention, and stdout; the disabled
mode gates only `writeMedoids` and the file-writing part of `writeBestRep`.

Because the old M20 unwind test deliberately threw from the nested file write,
M22 migrated that current oracle to a custom initializer that mutates method,
restart count, iteration count, medoids, and labels, then throws inside Lloyd.
The guard still restores the broad caller snapshot and all five changed fields.
M20's historical output-failure red evidence above remains accurate for the
pre-M22 implementation.

Green evidence from fresh rebuilt artifacts:

```text
build/highs-1151 (HiGHS 1.15.1 ON)
  [io]:                  9 assertions / 1 case passed
  [state]:              87 assertions / 2 cases passed
  full unit_test_mip:  151 assertions / 14 cases passed

build/phase8-m13 (HiGHS/Gurobi OFF)
  full unit_test_mip target rebuilt and linked
  [seed]:                9 assertions / 1 case passed
```

The call-graph audit finds one disabled-mode call in `benders.cpp`, one enabled
call in the public wrapper, and no other private implementation callers. Thus
LR-core and every direct/deprecated public Lloyd route retain persistence by
construction, while the no-solver build confirms the friend/private seam adds
no optional dependency.

Verdict: **PASS.** Benders exposes only its exact in-memory result, no longer
leaks or depends on intermediate Lloyd files, and preserves the incumbent and
solver trajectory exercised before the fix.

## M19 — lossless Python-HPC distance configuration

The M16 restart audit showed that the estimator's HPC branch still forwarded
only k, method, band, restart count, and seed. Non-default `max_iter`, all six
exposed variants and their five parameters, `mv_mode`, `missing_strategy`, and
`metric` disappeared before `_hpc.cluster_on_hpc`, so the remote executable ran
defaults without warning. The pre-edit trace ended at each missing boundary:

```text
DTWClustering.fit
  -> cluster_on_hpc
  -> SlurmRemoteRunner.submit_cluster
  -> slurm_remote.sh positionals
  -> sbatch --export
  -> cluster_generic.slurm
  -> dtwc_cl final argv
```

Fifteen red tests pinned the estimator call, runner tuple, wrapper exports, job
flags, final executable argv, pre-side-effect incompatibility checks, and a real
CLI missing-strategy option. Before implementation the focused run was exactly:

```text
15 failed, 24 deselected
```

`_validate_remote_configuration` is now the one Python normalization boundary
used by the pure command builder, runner, and orchestration function. It rejects
booleans and out-of-range `max_iter`, non-finite variant parameters, non-ASCII or
overflowing CUDA ordinals, unknown strings, CPU squared metric, every CUDA
variant except Standard, CUDA missing/multivariate modes, and the core's
unsupported non-Standard+missing or independent-mode combinations. Validation
runs before the result directory, serialization, runner preflight, or process
execution. The shell wrapper repeats the externally reachable grammar before
upload/submission; its numeric check rejects syntactically numeric overflow such
as `1e9999`, not only `nan`/`inf` spellings.

The normalized values occupy ten append-only wrapper positionals, become ten
`DTWC_*` SBATCH exports, and are passed as the corresponding ten CLI flags by
the job. Appending preserves M16's established positional schedule. Five
executable fake-job cases cover the unchanged defaults, CPU TWE with every
non-default parameter, CUDA squared distance, CPU ZeroCost missing handling,
and CPU independent multivariate mode. No impossible combined configuration is
used as a last-mile oracle.

The CLI now accepts `--missing-strategy` and the same TOML/YAML key, normalizes
aliases, maps the value onto `Problem` before variant binding, and prints it in
verbose diagnostics. One production helper validates unknown YAML values and
all backend/variant/missing/multivariate combinations before `Env`, output,
input, or cache work. A real rebuilt CLI run with `zero_cost` printed
`Missing:  zero_cost` and produced labels.

Green evidence:

```text
build/highs-1151 (HiGHS 1.15.1 ON, LLFIO ON)
  focused [config]:                 10 assertions / 1 case passed
  full unit_test_cli_args:         100 assertions / 17 cases passed

tests/python/test_hpc.py:           55 passed
  pre-side-effect rejection table: 20 passed / 35 deselected
  five executable final-job configurations included

targeted py_compile:                passed
bash -n tracked *.sh/*.slurm:      11/11 passed
git ls-files --eol shell scripts:  11/11 index/worktree LF
documentation contract/drift gate: passed
git diff --check:                  passed
```

SSH, rsync, and sbatch were intentionally not contacted. The executable job
test substitutes only the final binary while running the real job script.

An attempted NaN semantic contrast exposed a separate loader defect rather
than a missing-strategy defect: on this Windows build, formatted stream
extraction rejects textual `nan`, silently truncates the row, and discards its
remaining values. The CLI reported average length 2 for three intended length-3
rows under both PAM and hierarchical execution. M26 registers delimiter-aware
full-token parsing and requires the real Error-versus-ZeroCost PAM contrast
after NaN preservation; that loader fix is not hidden inside M19.

Verdict: **PASS.** Every estimator-exposed distance setting reaches the final
remote command or is rejected before side effects, defaults remain unchanged,
and unsupported execution is loud at Python, shell, and CLI boundaries.

## M24 — barycenter first-update convergence and finite-state checks

`barycenter_kmeans` initialized `previous_cost` to positive infinity and tested
relative cost convergence before its first center update. For every positive
tolerance, the finite first cost satisfied `inf <= tolerance * inf` under IEEE
arithmetic. The function therefore returned the random k-means++ initializer,
marked it converged with zero iterations, and never computed a barycenter.

The preregistered scalar discriminator uses `{0},{2}`, k=1, DBA. The correct
first update is the arithmetic mean 1 with hard squared-DTW cost 2 and one
completed update. It runs once at the unchanged default tolerance `1e-6` and
once at tolerance zero. The same test wave pins existing NaN/Inf input errors
and uses finite `DBL_MAX` values to exercise six previously silent computed
overflow routes: DBA/SSG hard cost, DBA accumulation/update, soft-DTW
value/gradient, k-means assignment, and k-means++ initialization weights.

Red evidence from the unfixed release build:

```text
build/highs-1151/bin/unit_test_barycenter.exe "[m24]" --reporter compact
positive tolerance:
  iterations 0 != 1
  center 2 != 1
  total_cost 4 != 2
computed overflow:
  expected exception, got none (DBA cost)
  expected exception, got none (SSG cost)
  expected exception, got none (DBA update)
  expected exception, got none (SoftDTW value/gradient)
  expected exception, got none (k-means assignment)
  expected exception, got none (k-means++ initialization)
test cases: 2 | 0 passed | 2 failed
assertions: 27 | 18 passed | 9 failed
```

The zero-tolerance center/cost/iteration assertions and all four explicit
NaN/Inf input-message assertions already passed red. No band was rescue-tuned.

Convergence is now eligible only after a completed prior assignment with a
finite stored cost. Hard barycenter paths validate each alignment, accumulated
objective, update, and convergence measure on serial or per-cluster caught
paths. Soft-DTW validates its initial value/gradient and gradient norm;
transient non-finite line-search trials still backtrack, but an unrecoverable
all-non-finite search is loud. K-means++ rejects a non-finite weight total
before constructing its random distribution. Assignment workers never throw:
their local costs are scanned and accumulated only after the OpenMP join.
Every new error instructs callers to rescale values to a smaller magnitude.

The Euclidean relative-change norm now uses chained `hypot` instead of
overflow-prone sums of squares. This changes no registered ordinary arithmetic:
the tolerance-zero mixed-length labels, all center bits, cost, iteration, and
convergence flag remain exactly equal to the pre-edit fingerprint.

Green evidence from fresh post-edit artifacts:

```text
build/highs-1151 (HiGHS 1.15.1 ON, LLFIO ON)
  [m24]:                              27 assertions / 2 cases passed
  [fingerprint]:                       5 assertions / 1 case passed
  full unit_test_barycenter:         113 assertions / 14 cases passed
  unit_test_barycenter_allocations:    2 assertions / 1 case passed
  ctest -R barycenter:                 2 tests / 0 failed

build/phase8-m13 (HiGHS/Gurobi/LLFIO OFF)
  unit_test_barycenter target rebuilt and linked
  [m24]:                              27 assertions / 2 cases passed
```

Verdict: **PASS.** Default-tolerance clustering performs real barycenter work,
zero-tolerance and ordinary digit-level behavior remain stable, and finite
inputs can no longer turn non-finite computation into a silent result.

## M27 — omitted HPC seed cannot leak from `--export=ALL`

M16 deliberately omitted `--seed` when Python callers passed `seed=None`, so
the remote CLI remained the single source of its default. The SLURM wrapper,
however, built `--export=ALL,...` and appended `DTWC_SEED` only for an explicit
seed. An ambient login-shell value could therefore reappear inside the job and
silently turn omission into an explicit seed.

The preregistered test runs the real wrapper from an isolated project under a
fake SSH/transfer boundary. Its parent environment contains `DTWC_SEED=29`;
the captured `sbatch` command must explicitly export an empty field when the
seed positional is omitted. The unfixed wrapper submitted no `DTWC_SEED` field:

```text
tests/python/test_hpc.py::TestSlurmLastMile::
  test_seed_export_overrides_inherited_slurm_environment[""-""]
FAILED: assert ',DTWC_SEED=' in submitted
1 failed
```

The wrapper now appends `DTWC_SEED=${SEED}` unconditionally. Empty means the
job script omits the CLI flag; a supplied value remains explicit. The same
executable boundary checks both ambient-29/omitted -> empty and
ambient-29/explicit-42 -> 42.

Green evidence:

```text
tests/python/test_hpc.py:              57 passed
targeted py_compile:                   passed
bash -n tracked *.sh/*.slurm:         11/11 passed
git ls-files --eol shell scripts:     11/11 index/worktree LF
```

Verdict: **PASS.** Neither login state nor `--export=ALL` can select a seed
when the API caller omitted it, and explicit schedules remain unchanged.

## M23 — one literal `DTWClustering` distance contract

The M19 closeout compared the sklearn-style estimator's public parameters with
its local fit and predict paths. Four independent defects shared one cause: the
estimator had no centralized statement of which distance semantics it could
execute.

First, `metric="squared_euclidean"` affected only the old Standard prediction
helper. CPU fit remained on `Problem`'s lazy L1 matrix and GPU fit omitted the
metric argument. The strict seed-42, k=2 fixture
`[[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,2,2]]` returned medoids `[4,2]`
and reported L1 inertia 4. Those medoids cost 6 under the requested squared
matrix; independent squared-matrix injection returns `[4,1]`, cost 5.

Second, `_dtw_fn` had explicit DDTW/WDTW/ADTW branches but fell through to
Standard DTW for MSM and TWE, and ignored every missing strategy. The registered
nearest-center discriminators reverse the Standard result:

```text
MSM(c=.7): x=000, c1=033, c2=113
  Standard: 6, 5 -> c2; MSM: 4.4, 5 -> c1
TWE(nu=.1, lambda=.8): x=000, c1=003, c2=020
  Standard: 3, 2 -> c2; TWE: 3, 4 -> c1
missing x=012, c1=0,NaN,100, c2=111
  ZeroCost/AROW: 98, 2 -> c2; Interpolate: 147, 2 -> c2
  old Standard fallback: NaN, 2 -> `argmin` c1
```

Third, unknown `metric`/`mv_mode` values and cross-products that the core cannot
represent were accepted and silently collapsed: non-Standard+non-L1,
non-Standard+non-Error missing handling, independent mode outside
Standard/Error/L1, squared missing handling, and CUDA/Metal missing or
independent modes. Fourth, if every restart objective was non-finite,
`best_result` either remained `None` or selected `-inf`, exposing an attribute
error or meaningless clustering instead of a numeric failure.

After correcting one test-oracle export typo, the preregistered production
suite failed exactly on the intended behavior:

```text
tests/python/test_clustering_semantics.py
  25 failed, 5 passed
```

The competing hypothesis that raw `Problem.variant_params` and
`missing_strategy` fields made local fit silently Standard was **FALSIFIED**
before editing. Dense storage is deferred; its first allocation rebinds the
distance function. The live pre-edit `_build_problem` produced MSM distances
`4.4, 5.0` and ZeroCost distances `98.0, 2.0` after fill. Local variant/missing
training was already correct; prediction and metric storage were not.

`_validate_semantics` now normalizes and validates the estimator's variant,
metric, missing strategy, multivariate mode, and resolved backend before any
distance work. Unsupported cross-products raise actionable `ValueError`
instead of substituting a recurrence. `_build_problem` applies the band through
its setter, writes missing strategy, then calls `set_variant_params`, making the
production rebind immediate and explicit.

Standard squared DTW computes one exact matrix and injects it into every CPU
restart; CUDA/Metal matrix construction receives the same metric. Standard-L1
CPU fit still uses the previous lazy `Problem` path, preserving its output and
storage/performance contract. Finite Standard/DDTW/WDTW/ADTW prediction keeps
the existing raw free-function fast path. MSM, TWE, non-error missing handling,
independent mode, and Error-on-NaN prediction use a two-series configured
`Problem`, which is the complete production dispatcher rather than mirrored
Python arithmetic.

Restarts now ignore non-finite candidates, snapshot the first finite result
unconditionally, replace it only for a strict improvement, and retain the
earliest finite tie. If every restart is non-finite, `FloatingPointError` names
each restart/seed/value and recommends checking data and parameters.

Green evidence:

```text
tests/python/test_clustering_semantics.py:                    35 passed
cross-validation + clustering + sklearn estimator suites:    49 passed
tests/python/test_hpc.py:                                     57 passed
tests/python/test_cuda.py:                     9 passed / 9 capability-skipped
tests/python/test_contract_parity.py:                         153 passed
targeted py_compile:                                          passed
```

The repository-wide Python run reached 576 passed / 12 capability-skipped but
also executed six concurrent, already-owned reds outside M23: four unrebuilt
M25 semantic-setter cases, the preregistered M28 unsafe-name case, and a newly
reported Tier-1 Lloyd reproducibility failure. None touches the M23 source or
focused gates; their owners must close them before the Phase-8 full-suite exit
claim.

Verdict: **PASS.** Every accepted estimator configuration uses the same
distance semantics for medoid selection, inertia, and prediction; unsupported
semantics and unusable restart results are loud; default Standard-L1 behavior
is unchanged.

## M28 — secure HPC submission envelope

Registered band: every job name, path, integer, method, and upload flag that
crosses Python, Bash, SSH, Slurm's comma-delimited export parser, or a transfer
tool must be normalized before any local/remote side effect. A real copied
wrapper must run behind an SSH executable that actually evaluates the remote
command. A crafted name must neither submit nor execute an extra command; an
accepted path containing otherwise sensitive-but-allowed bytes must arrive in
the captured `DTWC_INPUT` export unchanged.

The preregistered red set failed all ten new cases. Six direct-runner cases
invoked the wrapper, three high-level cases reached the runner and created the
run-directory boundary, and the real-wrapper malicious-name case returned
success instead of rejecting the remote-shell syntax.

The repaired Python boundary normalizes bounded CLI integers, method aliases,
the upload boolean, a 128-byte job-name grammar, and a conservative
comma/whitespace/metacharacter-free path grammar before creating a directory or
calling a runner. The Bash boundary repeats every check before its first SSH or
transfer. A follow-up adversarial pass found that a leading `-` source could
still be interpreted as a transfer option; that path is now rejected at both
layers and `rsync`/`scp` also receive `--` as defense in depth. An independent
rsync 3.2.7 discriminator then proved that `--` does not prevent `foo:bar` from
being treated as a remote source. Upload sources containing `:` are therefore
rejected at both layers, while pre-staged remote paths retain the character.
Missing paths and directories requested for upload are likewise rejected before
the banner and first SSH call; a fake-SSH sentinel mutation-pins that ordering.

The wrapper constructs `sbatch` as an array and shell-quotes each remote argv
element independently. The fake SSH executes with POSIX `sh`, and a hostile
optional cluster value `arc;touch <sentinel>;` arrives as one exact fake-Slurm
argument without creating the sentinel; restoring the old raw interpolation
would execute it. A real safe-upload capture also pins `rsync -az --` followed
by the unchanged relative source and remote target. The wrapper explicitly
exports dtype and seed in addition to all other job-consumed request fields, so
`--export=ALL` cannot supply an omitted configuration value. The executing
fake-SSH gate captures distinct fake-Slurm argv:
`/remote/input:v1+tag@host%=a.tsv` is present exactly as
`DTWC_INPUT=/remote/input:v1+tag@host%=a.tsv`, while an ambient seed 29 becomes
empty for omission or remains caller-selected 42.

Green commands and decisive output:

```powershell
.venv/Scripts/python.exe -m pytest tests/python/test_hpc.py -q
.venv/Scripts/python.exe -m py_compile python/dtwcpp/_hpc.py tests/python/test_hpc.py
# bash -n over every tracked *.sh; CR-byte scan over every *.sh/*.slurm
```

```text
82 passed in 35.26s
py_compile ok
shell syntax ok: 4
LF-only shell/slurm entrypoints: 11
```

Verdict: **PASS.** Unsafe requests are rejected before effects at both public
entry layers, the remote command is argv-quoted, transfer option ambiguity is
closed twice, and accepted export bytes retain their exact value.

## M30 — canonical `--missing-strategy` CLI reference

The M28 closeout ran the documentation contract checker against the live
HiGHS-enabled CLI and exposed one documentation-only mismatch. Generated pages
were already current, but the hand-maintained canonical CLI flag set omitted the
option added by M19. The exact retained red was:

```text
generated documentation is current
AssertionError: CLI reference drift:
  live but undocumented: ['--missing-strategy']
  documented but not live: []
```

The canonical CLI DTW-options table now records `--missing-strategy`, canonical
values `error`, `zero_cost`, `arow`, and `interpolate`, accepted aliases
`zero-cost` and `zerocost`, and the live `error` default. The full TOML and YAML
examples and their CLI-to-key table carry the same `missing-strategy` setting.
The existing missing-data method guide was retained unchanged. No production
code or drift-check arithmetic changed; `check_docs_contract.py` still compares
the exact live and documented flag sets in both directions.

Green commands and exact output:

```powershell
.venv/Scripts/python.exe scripts/generate_docs.py --check
# generated documentation is current

.venv/Scripts/python.exe scripts/check_docs_contract.py `
  --cli build/highs-1151/bin/dtwc_cl.exe
# generated documentation is current
# documentation contract checks passed
```

Verdict: **PASS.** The canonical references describe the live missing-data
option and configuration key, and the unchanged live-binary drift gate is exact
and green.

## M25 — dense distance-cache semantic identity

The confirmed defect was that a populated dense matrix had no identity beyond
its shape. After one distance had been computed—or after a full matrix had been
injected—changing `band`, `variant_params`, `missing_strategy`,
`distance_strategy`, or nested `cuda_settings` left the old computed bits in
place. `dist_by_ind()` then returned those bits without invoking the newly
requested recurrence.

The production fixtures were written and run red before the source edit. The
formal PLAN checkbox was inadvertently absent at that point; when this was
noticed, further gates paused and commit `02a9db3` recorded the exact M25 band
before work resumed. The retained pre-fix output was:

```text
unit_test_variant_distmat "[dense][semantic_mutation]"
  1 test case failed; 11 assertions: 7 passed, 4 failed
  band:     cached 0, expected 10
  variant:  cached 2, expected 1
  missing:  cached 0, expected 1
  backend:  injected matrix still reported filled

pytest tests/python/test_problem.py -k DenseSemanticMutation
  4 failed, 14 deselected
```

`Problem` now records a fixed-size `DistanceCacheConfiguration` whenever its
DTW function is rebound. The snapshot compares band, variant, all six numeric
variant parameters, multivariate mode, missing strategy, backend strategy,
CUDA device, and CUDA precision in constant time. A changed dense configuration
makes the read-only filled query false. Non-const compute/access paths clear the
matrix and rebind before use; const value accessors throw an actionable error
instead of exposing stale work. Direct dense I/O paths perform the same guard.

New C++ setters own missing strategy, distance strategy, and CUDA settings.
Existing band and variant setters share the invalidation rule, including a
follow-up audit correction that makes identical enum/whole-parameter assignments
true no-ops. Python whole-property bindings and MATLAB commands route through
these setters. Python deliberately retains nested mutation of `variant_params`
and `cuda_settings`; the snapshot detects it on the next query. Legacy public
C++ fields remain source-compatible under the same rule.

Mmap behavior remains stronger and unchanged in intent: raw mutation is
validated before any dense self-refresh and therefore fails its bound semantic
fingerprint loudly. An explicit setter detaches without rewriting the old file,
as before. Binding a mmap cache first normalizes any prior raw dense mutation so
its DTW function and fingerprint describe the same configuration.

Final green evidence against the committed implementation `d5a9659`:

```text
Clang + LLFIO ON focused semantic suite:   57/57 assertions, 2 cases
Clang + LLFIO ON full variant/distmat:     97/97 assertions, 13 cases
MSVC + LLFIO OFF full variant/distmat:     61/61 assertions, 10 mmap skips

fresh isolated Python extension:
  DenseSemanticMutation:                  4 passed / 14 deselected
  tests/python/test_problem.py:           18 passed

orthogonal C++ suites:
  variants / missing / Problem-missing /
  MV variants / MV missing / API 2.0:      183/183 assertions

fresh R2025b MEX:
  final setter + semantic subset:         2 passed
  full test_contract_parity:              24 passed
```

Verdict: **PASS.** Dense/precomputed work survives an identical configuration
and no semantic mutation can expose an old cached distance; legacy mutation is
detected in O(1), and persistent mmap identity remains loud.

## M29 — capped Lloyd final assignment coherence

Lloyd's loop assigned every point to the current medoids and then replaced the
medoids with the best members of those clusters. When the loop stopped because
it exhausted `max_iter`, it returned immediately: `centroids_ind` held the new
medoids, while `clusters_ind` still described the preceding medoids.
`find_total_cost()` follows the stored labels, so restart comparison and the
public result were internally inconsistent as well.

The preregistered scalar fixture is
`{0,40,40,46,49,51,51,51,100}`, with k=2, initial medoids `{0,8}`, and
`max_iter=1`. The single update selects medoid indices `{1,5}` (values 40 and
51). Against those final medoids, 46 and 49 both belong to cluster 1 and the
independent nearest-medoid objective is
`40 + 5 + 2 + 49 = 96`. The stale labels left both points in cluster 0 and
therefore produced 104.

Red command:

```powershell
build/phase8-capped-lloyd/bin/unit_test_clustering_algorithms.exe `
  "[m29]" --reporter compact
```

Decisive pre-fix output:

```text
final medoids:  {1,5}
stale labels:   {0,0,0,0,0,1,1,1,1}
nearest labels: {0,0,0,1,1,1,1,1,1}
reported cost:  104
nearest cost:   96
test cases: 1 | 1 failed
assertions: 11 | 8 passed | 3 failed
```

The regression also runs the same problem to convergence. It independently
recomputes every nearest label and distance from the returned medoids, then
requires the capped and converged states to match. The converged path reaches
the same medoids, labels, and cost after two iterations, which pins the
no-change branch as well as the capped correction.

The implementation performs one final `assign_clusters()` only for status -1,
immediately before cost calculation and restart snapshotting. Status 0 proves
that the update left medoids unchanged, so its preceding assignment is already
current and receives no extra work. The full distance matrix is materialized
before the loop, making the added capped assignment read-only and parallel-safe.
It consumes no RNG and does not change initialization or iteration counts.

Green evidence:

```text
Clang Release, HiGHS/LLFIO ON (build/highs-1151)
  focused [m29]:                         11 assertions / 1 case passed
  full unit_test_clustering_algorithms:  67 assertions / 11 cases passed
  Tier-1 [lloyd] contracts:              16 assertions / 4 cases passed
  k-means++ FastPAM quality contrast:      1 assertion  / 1 case passed
  Problem Phase-0 artifact failures:      50 assertions / 3 cases passed
  focused CTest group:                     5 tests / 0 failed

Clang Release, HiGHS/Gurobi/LLFIO OFF (build/phase8-capped-lloyd)
  focused [m29]:                         11 assertions / 1 case passed
  full unit_test_clustering_algorithms:  67 assertions / 11 cases passed
  Tier-1 [lloyd] contracts:              16 assertions / 4 cases passed
  direct initialization gates:            4 assertions / 2 cases passed
  k-means++ FastPAM quality contrast:      1 assertion  / 1 case passed
  Problem Phase-0 artifact failures:      50 assertions / 3 cases passed
```

The Tier-1 group covers best-restart restoration, local seed scheduling,
restart-count rejection, checked seed overflow, custom callbacks, and a first
infinite-cost result. The focused CTest group additionally runs the complete
FastPAM and adversarial FastPAM suites.

Verdict: **PASS.** Capped Lloyd results now expose one coherent final state;
converged arithmetic, seed/restart schedules, k-means++ behavior, non-finite
selection, artifact errors, and optional-dependency floors remain green.

## M35 — explicit Python Tier-1 Lloyd seed

The M23 full-suite closeout exposed a consecutive-call failure in Python's
functional `dtwcpp.cluster(..., method="kmedoids")` route. An independent run
against the editable environment reproduced two different results from the
same process and data:

```text
first medoids:  [6, 1, 3]
second medoids: [3, 6, 0]
costs:          20, 20
```

That editable extension was stale: it did not expose `Problem.random_seed` or
`set_random_seed`, so it predated M17's C++ seed dispatch. The competing claim
that current C++ Lloyd remained globally random was therefore tested against
the fresh M25-built extension (SHA-256
`2ffafbdeaeac29d199c3888574d40c64fe6902410f183f120c4c06fa9d4f8269`) and
**FALSIFIED**: before the Python edit it already returned `[6,1,3]`, cost 20,
twice. This distinction prevents a stale binary from being misreported as new
C++ arithmetic work.

The Python route still had a real contract omission. Unlike PAM, OneBatchPAM,
and CLARA, `_run_local_method` did not explicitly select the shared Tier-1
seed. A fake-Problem dispatch test registered the source-level red:

```text
actual events:   [('cluster', None)]
expected events: [('seed', 42), ('cluster', None)]
```

The repair calls `Problem.set_random_seed(DEFAULT_RANDOM_SEED)` only in the
Tier-1 `kmedoids` branch, immediately before `Problem.cluster()`. It does not
replace `init_fun`; explicit seeds and arbitrary callbacks remain owned by the
advanced `Problem` API, and the one-argument initializers plus unseeded Tier-2
FastPAM retain their deliberate mutable-global behavior. MIP, LR-core, TADPole,
PAM, OneBatchPAM, CLARA, and hierarchical dispatch are unchanged.

Two semantic regressions use the translated eight-waveform seed discriminator:
one calls Tier-1 Lloyd consecutively, and one consumes the legacy Tier-2 engine
between calls. Both return medoids `[6,1,3]`, identical labels, and cost 20. The
dispatch spy mutation-pins the explicit seed before cluster and the existing
MIP fake lacks a seed setter, proving the new call is kmedoids-only.

Green evidence using the fresh core with the edited source `_api` loaded
explicitly (avoiding the stale editable extension):

```text
tests/python/test_api.py:                                      49 passed
contract + clustering semantics + CV + sklearn estimator:    221 passed
test_tier1_cpp_api.exe "[lloyd]": 16 assertions / 4 cases passed
py_compile _api.py + test_api.py:                              passed
```

Verdict: **PASS.** Python's functional Lloyd boundary states its seed contract
explicitly, repeated and interleaved calls are reproducible, and no Tier-2,
custom-initializer, explicit-seed, estimator, or other-method behavior changed.

## M31 — transactional HPC job identity and completion

Registered band: two same-name submissions must never share a writable local
input, remote upload, or remote job script; a returned Slurm job ID must be the
only result identity. Nonzero submit/status/download exits, a configured-cluster
query failure, a hung status process, unsafe allocator output, or a missing
exact result must fail loudly without returning a stale label file. Polling
controls must be validated before local or remote effects and bound by elapsed
monotonic time rather than requested sleeps.

The preregistered red command selected the new transactional cases from
`tests/python/test_hpc.py` and produced 11 failures / 20 passes. It accepted a
nonzero submit containing `Job ID`, matched job `111` inside `1110`, invoked
status for invalid polling controls, had no job-aware download signature,
reused the same local input and remote basename, reached the runner before
poll validation, and submitted both uploads through the same destination.

The repaired Python path validates positive finite poll/timeout controls before
creating a run directory, uses atomic `submission-*` directories, rejects every
nonzero transport result, matches a status row's exact leading job ID, and
passes the monotonic deadline's remaining duration to each status subprocess.
`TimeoutExpired` and elapsed calls are typed `TimeoutError`s. Label retrieval
uses `download-cluster name job-id` and requires the exact managed
`results/slurm/<name>_<job-id>/<name>_labels.csv`.

The wrapper allocates one remote `${name}.XXXXXXXX` directory for every call,
including pre-staged inputs. Its exact input and copied `cluster_generic.slurm`
live there, and captured fake-Slurm argv proves `sbatch` receives that copied
path rather than a shared script. The allocator result must equal its root plus
one basename whose suffix is exactly eight alphanumerics; a prefix-matching
`/../../src` response is rejected. `status` no longer falls back from a failed
configured-cluster query to an unscoped empty queue. Exact download deletes the
old exact target first, so a failed or no-output transfer cannot expose stale
bytes.

Independent review initially found four production gaps: masked federated
status failure, sleep-only timeout accounting, a shared remote script, and
prefix-only allocator validation. After those repairs it found one test gap:
the unique copied scripts were captured but their final `sbatch` positions were
not. The final two-call test equates each captured copy destination with that
call's final Slurm argument and proves both differ. Final independent verdict:
**PASS**.

Green evidence:

```text
focused transaction/security selection: 46 passed, 47 deselected
post-review focused selection:          37 passed, 61 deselected
tests/python/test_hpc.py:                98 passed in 49.82s
targeted py_compile:                     passed
tracked shell syntax:                    4/4 passed
shell/Slurm LF scan:                     11/11 passed
git diff --check:                        passed
```

Verdict: **PASS.** Submission input, script, status, timeout, and result are now
bound to one job identity; transport ambiguity is loud and stale/cross-job
labels cannot be selected.

## M33 — transactional direct-MIP state publication

The direct HiGHS and Gurobi paths both called the shared seeded FastPAM helper
on the caller's `Problem`. FastPAM returns a value result but also writes its
heuristic medoids and labels back into that object. HiGHS later cleared those
vectors only while extracting a successful exact solution, so any solve or
extraction failure exposed the incumbent. Gurobi cleared medoids before setup,
then FastPAM repopulated them, and successful extraction appended exact medoids
without another clear. Its normal warm-start result could therefore contain 2k
medoids; setup, solve, or extraction failure exposed empty, heuristic, or
partially decoded state.

The first discriminator preloaded a valid, distinct clustering before calling
`mip::make_warm_start`. The returned seed-42 incumbent remained the registered
medoids `{6,2,5}` at cost 24, but the caller had to retain its original vectors.
The unfixed helper instead published the incumbent:

```text
unit_test_mip "MIP FastPAM warm-start medoids are invocation-local"
  first_problem.centroids_ind: {6,2,5} != {0,3,7}
  first_problem.clusters_ind:  {1,1,1,1,2,2,0,0}
                            != {0,0,0,1,1,1,2,2}
test cases: 1 | 1 failed
assertions: 11 | 9 passed | 2 failed
```

The Gurobi append was independently confirmed in the production source: its
entry clear preceded FastPAM, while the exact diagonal scan used `push_back`
against the now-populated vector. The corresponding runtime requires a licensed
solver, so the repair is pinned with a Gurobi-ON production compile plus the
same point-major extraction seam exercised without a license.

`ExactClusteringTransaction` snapshots the two caller-visible clustering
vectors and restores them with non-throwing swaps unless a result is published.
The warm-start helper uses this transaction solely as a restore guard, making
its returned incumbent private. Each direct backend owns another transaction
for its complete setup/solve/extract lifetime.

The shared extractor understands HiGHS facility-major and Gurobi point-major
layouts. Before returning local vectors it requires a finite N-by-N solution,
exactly k selected medoids, exactly one active assignment per point, assignment
only to selected medoids, and medoid self-assignment. Publication independently
checks k unique in-range medoids and N labels in `[0,k)`, then swaps both vectors
into `Problem`; no throwing work follows publication. Backend-neutral tests
mutation-pin both layouts, a forced solve failure after incumbent exposure, a
partial extraction with two active assignments, duplicate-medoid rejection,
and the compiled-out HiGHS error path.

Green evidence:

```text
Clang Release, HiGHS 1.15.1 + Gurobi 13.0.1 + LLFIO ON
  focused [m33]:                       36 assertions / 3 cases passed
  state-neutral seeded warm start:     11 assertions / 1 case passed
  live HiGHS warm exact invariants:    16 assertions / 1 case passed
  full unit_test_mip:                 194 assertions / 17 cases passed
  M20/M22 [benders] selection:        112 assertions / 6 cases passed
  full unit_test_benders:              42 assertions / 7 cases passed
  Tier-1 [lloyd] compatibility:        16 assertions / 4 cases passed
  focused CTest group:                  4 tests / 0 failed
  mip_Gurobi.cpp:                       compiled with DTWC_ENABLE_GUROBI=ON

Clang Release, HiGHS/Gurobi/LLFIO OFF
  focused [m33]:                       38 assertions / 3 cases passed
  state-neutral seeded warm start:     11 assertions / 1 case passed
  Tier-1 [lloyd] compatibility:        16 assertions / 4 cases passed
  unit_test_mip + direct-MIP objects:   compiled and linked
```

The focused CTest group comprises full `unit_test_mip`, `unit_test_benders`,
`test_tier1_cpp_api`, and `unit_test_clustering_algorithms`. Benders does not use
the shared direct FastPAM seam and retains its M20 state guard and M22 artifact
policy unchanged. The no-solver backend still raises the same typed
unavailable-HiGHS error and now explicitly proves the caller vectors unchanged.

Verdict: **PASS.** Direct exact solvers expose no heuristic or partially decoded
clustering state: failure restores the caller, and success publishes exactly k
unique medoids plus N valid labels only after complete validation.
