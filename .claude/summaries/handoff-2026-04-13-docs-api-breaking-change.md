---
name: handoff-2026-04-13-docs-api-breaking-change
description: Session handoff - docs/API parity work, Hugo+Doxygen local build, Python/MATLAB distance namespace breaking change, and default public template scalar alias introduced.
type: project
---

# Session Handoff - 2026-04-13 - Docs/API Parity + Breaking Distance Namespace Change

Branch: **Claude**.

This handoff covers the continuation after the adversarial audit:
- local Hugo + Doxygen docs build
- cross-language API parity review
- Python/MATLAB breaking change to `distance.*` pairwise APIs
- additive C++ `dtwc::distance::*` namespace
- new public default template scalar alias in `settings.hpp`

It is intended to supersede the later chat history for this branch, not the earlier audit handoff.

## Executive summary

The repository now has one documented pairwise-distance story across C++, Python, and MATLAB:

- C++: `dtwc::distance::*`
- Python: `dtwcpp.distance.*`
- MATLAB: `dtwc.distance.*`

Python and MATLAB were changed as an intentional breaking release:
- old root-level pairwise helpers such as `dtwcpp.dtw_distance(...)` were removed from the Python public package surface
- old MATLAB root wrapper files such as `+dtwc/dtw_distance.m` were deleted

The docs site now builds locally with Hugo and Doxygen, and the generated API tree is staged into `docs/public/Doxygen`. The previous machine-freeze issue during Doxygen was caused by Graphviz using all cores; `DOT_NUM_THREADS` is now capped to `2`.

Separately, hardcoded template defaults like `template <typename T = double>` were replaced with a canonical alias:
- `dtwc::settings::default_data_t = float`

Important boundary:
- **internal `dtwc::data_t` storage remains `double`**
- only the **default template argument** was changed to `settings::default_data_t`
- this avoids breaking `Data`, `Problem`, Python, and MATLAB internals in one risky step

## User intent for this phase

User asked to continue beyond the adversarial audit and specifically wanted:

1. the docs website compiled locally with Hugo
2. API docs with class/structure diagrams
3. inspection of divergence across C++, Python, and MATLAB
4. a more similar flow across bindings:
   - settings / YAML
   - load data
   - construct `Problem`
   - cluster
   - inspect scores/results
5. a namespace-based pairwise API such as:
   - `dtwcpp.distance.dtw(...)`
   - `dtwc::distance::dtw(...)`
6. later, an explicit breaking change:
   - no need for legacy compatibility aliases
7. latest follow-up:
   - replace `template <typename T = double>` defaults with a settings-defined default type, preferring float

## Current repository truth

### Build / test baseline

Verified earlier in the same session:
- `cmake --build build --config Release -j 8`
- `ctest --test-dir build -C Release --output-on-failure -j 4`

The Windows Release suite was green after fixing the lazy-distance race in `Problem::assignClusters()`.

### Docs build baseline

Local docs prerequisites now available on this machine:
- Hugo
- Go
- Doxygen
- Graphviz

Local docs commands that succeeded:
- `docs/`: `hugo mod get -u`
- `docs/`: `hugo --minify --baseURL "https://battery-intelligence-lab.github.io/dtw-cpp/"`
- repo root: `doxygen docs/Doxyfile`

Generated output now exists at:
- `docs/public/index.html`
- `docs/public/api/interface-parity/index.html`
- `docs/public/Doxygen/index.html`

### Doxygen machine-safety fix

The freeze root cause was:
- `NUM_PROC_THREADS = 1` already
- but `DOT_NUM_THREADS = 0`, which means Graphviz can fan out aggressively

Fix:
- `docs/Doxyfile`: `DOT_NUM_THREADS = 2`

Also enabled:
- `UML_LOOK = YES`

Still disabled:
- `CALL_GRAPH = NO`
- `CALLER_GRAPH = NO`

This keeps structure/class graphs without the previous pathological graph explosion.

## Files added or materially changed

### New / new public surfaces

- `dtwc/distance.hpp`
- `python/dtwcpp/distance.py`
- `bindings/matlab/+dtwc/+distance/dtw.m`
- `bindings/matlab/+dtwc/+distance/standard.m`
- `bindings/matlab/+dtwc/+distance/ddtw.m`
- `bindings/matlab/+dtwc/+distance/wdtw.m`
- `bindings/matlab/+dtwc/+distance/adtw.m`
- `bindings/matlab/+dtwc/+distance/soft_dtw.m`
- `bindings/matlab/+dtwc/+distance/missing.m`
- `bindings/matlab/+dtwc/+distance/arow.m`
- `docs/content/api/interface-parity.md`
- `docs/custom.css`
- `.claude/summaries/handoff-2026-04-13-docs-api-breaking-change.md` (this file)

### MATLAB wrappers deleted

Deleted root wrappers:
- `bindings/matlab/+dtwc/dtw_distance.m`
- `bindings/matlab/+dtwc/ddtw_distance.m`
- `bindings/matlab/+dtwc/wdtw_distance.m`
- `bindings/matlab/+dtwc/adtw_distance.m`
- `bindings/matlab/+dtwc/soft_dtw_distance.m`
- `bindings/matlab/+dtwc/dtw_distance_missing.m`
- `bindings/matlab/+dtwc/dtw_arow_distance.m`

### Other touched files

Representative major edits:
- `python/dtwcpp/__init__.py`
- `dtwc/dtwc.hpp`
- `dtwc/settings.hpp`
- `dtwc/core/dtw.hpp`
- `dtwc/core/time_series.hpp`
- `dtwc/warping.hpp`
- `dtwc/warping_adtw.hpp`
- `dtwc/warping_ddtw.hpp`
- `dtwc/warping_missing.hpp`
- `dtwc/warping_missing_arow.hpp`
- `dtwc/warping_wdtw.hpp`
- `dtwc/soft_dtw.hpp`
- `docs/Doxyfile`
- `docs/content/getting-started/python.md`
- `docs/content/getting-started/matlab.md`
- `docs/content/getting-started/examples.md`
- `docs/content/getting-started/configuration.md`
- `docs/content/getting-started/supported-data.md`
- `docs/content/method/dtw-variants.md`
- `docs/content/method/missing-data.md`
- `examples/python/01_quickstart.py`
- `examples/python/02_dtw_variants.py`
- `examples/python/04_missing_data.py`
- `examples/python/06_distance_matrix.py`
- `examples/matlab/example_quickstart.m`
- `tests/python/test_dtw.py`
- `tests/python/test_cross_validation.py`
- `tests/python/test_problem.py`
- `tests/integration/test_cross_language.py`
- `tests/matlab/test_dtwc.m`
- `bindings/matlab/test_mex.m`

## What changed

### 1. C++ gained a public `dtwc::distance::*` namespace

Added `dtwc/distance.hpp` and included it from `dtwc/dtwc.hpp`.

Exposed functions include:
- `dtwc::distance::dtw(...)`
- `dtwc::distance::ddtw(...)`
- `dtwc::distance::wdtw(...)`
- `dtwc::distance::adtw(...)`
- `dtwc::distance::soft_dtw(...)`
- `dtwc::distance::missing(...)`
- `dtwc::distance::arow(...)`

Also added a convenience dispatcher:
- `dtwc::distance::dtw(x, y, DTWVariantParams, band, metric, missing_strategy)`

Important limitation of that dispatcher:
- combined missing-strategy dispatch only supports `variant=Standard`
- `Interpolate` currently delegates by materializing interpolated vectors and then calling standard DTW

This is additive on the C++ side:
- older low-level free functions still exist in headers
- docs/examples now prefer `dtwc::distance::*`

### 2. Python is now namespace-only for pairwise distances

Added:
- `python/dtwcpp/distance.py`

Public Python pairwise API is now:
- `dtwcpp.distance.dtw`
- `dtwcpp.distance.ddtw`
- `dtwcpp.distance.wdtw`
- `dtwcpp.distance.adtw`
- `dtwcpp.distance.soft_dtw`
- `dtwcpp.distance.missing`
- `dtwcpp.distance.arow`

The dispatcher supports:
- `variant`
- `band`
- `metric`
- `g`
- `penalty`
- `gamma`
- `missing_strategy`

Removed from the public root package surface:
- `dtwcpp.dtw_distance`
- `dtwcpp.ddtw_distance`
- `dtwcpp.wdtw_distance`
- `dtwcpp.adtw_distance`
- `dtwcpp.soft_dtw_distance`
- `dtwcpp.dtw_distance_missing`
- `dtwcpp.dtw_arow_distance`

`python/dtwcpp/__init__.py` now exports `distance` and no longer lists the old pairwise helpers in `__all__`.

One explicit regression guard was added:
- `tests/python/test_dtw.py` now asserts the root alias is absent

### 3. MATLAB is now namespace-only for pairwise distances

The new public MATLAB pairwise API is:
- `dtwc.distance.dtw`
- `dtwc.distance.standard`
- `dtwc.distance.ddtw`
- `dtwc.distance.wdtw`
- `dtwc.distance.adtw`
- `dtwc.distance.soft_dtw`
- `dtwc.distance.missing`
- `dtwc.distance.arow`

Originally those namespace functions forwarded through the old root wrappers.
That is no longer true.

They now call `dtwc_mex(...)` directly.

This matters because the old root wrapper files were deleted:
- the new namespace is not just documentation-level sugar
- it is now the real MATLAB pairwise API surface

Important MATLAB limitation:
- the new namespace functions currently only support `Metric='l1'` for the standard / missing / arow stateless wrappers
- non-L1 metrics throw a targeted error in those wrappers
- this matches the current MEX command coverage rather than inventing fake support in MATLAB

### 4. Docs were rewritten around the shared flow

New docs/API parity page:
- `docs/content/api/interface-parity.md`

That page now includes:
- a cross-language interface-flow diagram
- links to Doxygen class/namespace pages
- embedded generated collaboration graphs for:
  - `dtwc::Problem`
  - `dtwcpp._clustering.DTWClustering`

The docs now present the intended flow as:

```text
settings -> data -> Problem -> clustering algorithm -> scores / outputs
```

Other docs changes:
- stale C++ examples replaced with `Problem` + `fast_pam` + `dtwc::distance::*`
- Python docs rewritten to use `dtwcpp.distance.*`
- MATLAB docs rewritten to use `dtwc.distance.*`
- configuration docs corrected to reflect current YAML/CLI precedence reality
- supported-data docs updated because the earlier “no multivariate support” statement was stale

### 5. Local docs build was made reproducible

`docs/Doxyfile` was changed to:
- include Python and MATLAB source trees
- retain class/collaboration/namespace structure views
- avoid expensive call/caller graphs
- cap Graphviz threads

One subtle Doxygen issue was also discovered:
- deleting source files is not enough
- incremental Doxygen builds can leave stale HTML pages behind

To remove stale deleted-wrapper pages, a clean rebuild of `build/Doxygen` was necessary before copying into `docs/public/Doxygen`.

### 6. Default public template type is now centralized in `settings.hpp`

Added:
- `dtwc::settings::default_data_t = float`

Then replaced hardcoded defaults like:
- `template <typename T = double>`
- `template <typename data_t = double>`

with:
- `template <typename T = dtwc::settings::default_data_t>`
- `template <typename data_t = dtwc::settings::default_data_t>`

This affected public/defaulted templates in:
- `dtwc/core/dtw.hpp`
- `dtwc/core/time_series.hpp`
- `dtwc/distance.hpp`
- `dtwc/warping.hpp`
- `dtwc/warping_adtw.hpp`
- `dtwc/warping_ddtw.hpp`
- `dtwc/warping_missing.hpp`
- `dtwc/warping_missing_arow.hpp`
- `dtwc/warping_wdtw.hpp`
- `dtwc/soft_dtw.hpp`

Also updated stale doc comments that still claimed “default: double”.

## Important non-change: internal storage type is still `double`

`dtwc::data_t` in `dtwc/settings.hpp` still equals `double`.

This was intentional.

Reason:
- `data_t` is not just a cosmetic alias
- it threads through `Data`, `Problem`, `Problem` distance binding, caches, loaders, and parts of the Python/MATLAB/native interface layer
- flipping `data_t` to `float` today would be a much broader precision/storage migration

Examples of code still explicitly built around double-backed internals/bindings:
- `python/src/_dtwcpp_core.cpp` uses `nb::ndarray<const double, ...>` and many `std::vector<double>` surfaces
- `bindings/matlab/dtwc_mex.cpp` is explicitly double-based (`mxGetDoubles`, `mxCreateDoubleScalar`, etc.)
- several algorithm and matrix/helper layers still use `double` directly

So the current state is:
- **public template defaults**: float
- **internal canonical storage alias (`dtwc::data_t`)**: double

This split is a real design compromise, not a finished precision unification.

## Verification performed

### Successful

1. C++ build:
   - `cmake --build build --config Release --target dtwc_cl -j 8`
   - succeeded after the `default_data_t` refactor

2. Python syntax:
   - `python -m compileall python\\dtwcpp`
   - succeeded

3. Hugo:
   - `hugo --minify --baseURL "https://battery-intelligence-lab.github.io/dtw-cpp/"`
   - succeeded

4. Doxygen:
   - `doxygen docs/Doxyfile`
   - succeeded
   - clean rebuild + restage also completed after the thread cap was introduced

5. Generated docs assets confirmed:
   - `docs/public/Doxygen/namespacedtwcpp_1_1distance.html`
   - `docs/public/Doxygen/namespacedtwc_1_1distance.html`
   - `docs/public/api/interface-parity/index.html`

### Not fully runtime-verified

The local Python package import was **not** runtime-verified directly from `python/` because:
- `dtwcpp._dtwcpp_core` is not present in-place in that source tree on this checkout
- importing `dtwcpp` with `PYTHONPATH=python` fails until the extension module is built/installed into the expected location

So for Python this session verified:
- syntax
- caller/doc/test migration
- not full runtime import/execution from the source tree

MATLAB runtime was also not exercised locally in this session.

## Remaining warnings / rough edges

### 1. YAML vs CLI precedence is still unresolved

Still a real drift issue:
- docs were adjusted to reflect current behavior
- code still wants a real precedence fix

This should still be treated as a priority issue.

### 2. The earlier FastPAM/Lloyd test naming issue still exists

The audit finding remains:
- `test_fast_pam_adversarial.cpp` overstates what it actually covers

That should still be split or renamed.

### 3. C++ has not yet taken the same “hard break” as Python/MATLAB

Current state:
- docs prefer `dtwc::distance::*`
- old lower-level free functions still exist in headers

If the project wants one consistent breaking philosophy, the next decision is:
- either keep old C++ free functions as low-level/internal helpers
- or formally deprecate/remove them too

### 4. `default_data_t=float` but `data_t=double` is internally inconsistent

This is acceptable as a transitional state, but it is not conceptually clean.

If the project wants “float by default” to be globally true, a larger follow-up is required:
- make `Data` / `Problem` / binding surfaces precision-aware in a cleaner way
- decide whether `data_t` should also become float
- audit all explicit `double` assumptions in:
  - Python bindings
  - MATLAB bindings
  - dense distance matrix and score/reporting layers
  - CUDA / Metal / MPI helpers

### 5. Doxygen still emits documentation warnings

Not blockers for HTML generation, but still present:
- obsolete config warnings
- several parameter-doc mismatches
- a few existing README / changelog ref warnings
- some MATLAB doc parsing warnings

These are doc-quality cleanup items, not runtime blockers.

### 6. MATLAB should live in a top-level `matlab/` folder, not under `bindings/`

Current layout:
- `bindings/matlab/...`

This works for the build, but it is not the best user-facing layout.

Recommended direction:
- move MATLAB sources to a top-level `matlab/` folder
- keep the package rooted there so users can simply `addpath('matlab')` and work
- avoid forcing users to know or care that MATLAB is implemented as one of several bindings

If this is done, the follow-up needs to update:
- CMake paths
- Doxygen input paths
- docs references
- examples/tests references
- any CI/build scripts that currently reference `bindings/matlab`

### 7. Tests are still able to pollute the repository root with CSV output

Root-level CSVs observed in this checkout:
- `test_clusteringmedoids_rep_0.csv`
- `test_clusteringmedoids_rep_1.csv`
- `test_clusteringmedoids_rep_2.csv`
- `test_clustering_bestRepetition_Nc_1.csv`
- `test_clustering_bestRepetition_Nc_10.csv`
- `test_clustering_bestRepetition_Nc_3.csv`

These are a symptom of tests still exercising legacy output/reporting paths that
write relative to the current working directory.

Required direction:
- every test that writes files should redirect output into:
  - `results/`, or
  - a temp directory under `results/`, or
  - a dedicated temp test directory
- no test should write CSVs or other generated artifacts into the repo root

This should be treated as a repository hygiene rule, not a one-off cleanup.

Likely relevant code paths:
- `dtwc/Problem_IO.cpp`
- `dtwc/Problem.cpp`
- tests that call legacy clustering/reporting functions without overriding output paths

### 8. Device selection should eventually include `hpc` as a first-class target

User expectation captured explicitly for future design work:

- device selection should not stop at:
  - `cpu`
  - `cuda`
- it should also support:
  - `hpc`

Desired behavior:
- when a user selects `device="hpc"` (or equivalent in C++ / MATLAB / CLI),
  the target should be an HPC execution path rather than local CPU/CUDA
- HPC credentials / connection settings should come from `.env` or environment
  variables, not hardcoded into source files
- `Problem` should eventually be able to accept HPC-related settings and submit
  or orchestrate work on the HPC target automatically
- the project should also expose HPC diagnostics / capability checks, analogous
  to existing local backend checks

Recommended direction:
- define a small `HPCSettings` or remote-execution config object
- feed it through the same shared settings/config system recommended elsewhere
- make device routing explicit:
  - `cpu` -> local CPU
  - `cuda` -> local GPU
  - `hpc` -> remote/HPC target configured via env
- add explicit helper/diagnostic functions such as:
  - `check_hpc()`
  - `hpc_available()`
  - `hpc_system_info()`

Important implementation note:
- this should be designed as a real execution target abstraction, not just a
  string alias layered on top of the current local-device code
- secret handling must remain outside tracked config files; `.env` / env vars
  are the right place for credentials and host-specific settings

## Recommended next steps

Priority order from here:

1. **Decide whether `dtwc::data_t` should remain double or also become float.**
   - This is the next real design decision after introducing `settings::default_data_t`.
   - Do not change it casually; bindings and `Problem/Data` need an explicit migration plan.

2. **Introduce a real cross-language settings/config object.**
   - Current docs now make the intended flow explicit.
   - The implementation still lags:
     - CLI is config-first
     - library bindings are still more `Problem`-first than config-first
   - Best target:
     - one shared `Settings` / `ProblemConfig` representation
     - one YAML/TOML ingestion strategy
     - project into Python and MATLAB from that shared core model

3. **Resolve YAML-vs-CLI precedence in code, not just docs.**
   - This remains a truthfulness issue.

4. **Decide whether the C++ free-function surface should also be broken/deprecated.**
   - Python/MATLAB already made the break.
   - C++ is the remaining mixed case.

5. **Fix or split the misleading FastPAM/Lloyd adversarial test.**
   - Still high-value because it affects trust in coverage claims.

6. **Move MATLAB from `bindings/matlab` to a top-level `matlab/` folder.**
   - Goal: users should be able to `addpath('matlab')` and work immediately.
   - Treat this as a user-experience and layout cleanup, not just a cosmetic rename.

7. **Enforce a test-output rule: tests must write under `results/` or temp dirs, never repo root.**
   - Clean up the existing clustering/reporting tests that still emit root-level CSVs.
   - Prefer a temp subdirectory under `results/` when the code path is expected to exercise real output generation.

8. **Design a first-class `hpc` device target alongside `cpu` and `cuda`.**
   - Credentials and remote target settings should come from `.env` / env vars.
   - `Problem` should eventually be able to carry HPC settings and orchestrate remote execution.
   - Add HPC diagnostics/check functions rather than treating HPC as an opaque string.

9. **Consider a proper precision strategy document.**
   - The project now has:
     - `settings::default_data_t`
     - `data_t`
     - float32 CLI data-loading modes
     - float64 bindings
     - GPU precision flags
   - These need one coherent documented story.

## Suggested continuation prompt

If continuing from this handoff, the next most useful prompt is:

> Decide whether `dtwc::data_t` should also become float, then either complete the precision migration end-to-end or document and enforce the split between public default template type and internal storage type.

If continuing on the cross-language flow instead:

> Introduce a shared `Settings` / `ProblemConfig` object with YAML loading, then migrate Python and MATLAB examples so all three bindings show the same sequence: settings -> data -> Problem -> algorithm -> scores/results.
