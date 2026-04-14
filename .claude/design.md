# DTWC++ Architecture & Design Decisions

## Layered Design

**Layer 1: Core** - DTW kernels, variant dispatch, constraints, and distance-matrix builders. Minimal stable headers.

**Layer 2: Algorithms** - PAM, CLARA, CLARANS, hierarchical clustering, and evaluation metrics. Mostly in `dtwc/algorithms/`.

**Layer 3: IO** - Data loaders, CSV parsing, and matrix/result persistence. Kept separate from core compute paths where possible.

**Layer 4: Bindings** - Python (nanobind + scikit-build-core) and MATLAB (legacy C MEX + handle classes). Both should call stable core APIs rather than reimplement logic.

## DTW Variant Architecture

- **Unified kernel family per topology** (`dtw_kernel_banded`, `dtw_kernel_full`, and MV equivalents). Variants are expressed through Cost/Cell policies, not copy-pasted recurrence loops.
- **`resolve_dtw_fn` binds once in `Problem`** and returns the correct f64/f32 callable for the active variant and missing-data strategy. `std::function` dispatch cost is negligible next to DTW runtime.
- **WDTW and ADTW are not metric swaps** - they change cost/recurrence semantics, but they now fit the shared kernel contract cleanly.
- **DDTW remains a preprocessing transform** followed by the shared DTW kernel family.
- **Soft-DTW is still a distinct recurrence (`softmin`)** but lives in the same kernel architecture via a dedicated Cell policy instead of a wholly separate implementation stack.
- **Missing-data handling is part of dispatch, not an afterthought** - standard error-on-NaN, zero-cost, AROW, and interpolation paths all need to stay aligned with the same kernel contracts.

## Performance Principles

- DTW is **latency-bound** (~10 cycles/cell recurrence). Rolling buffers fit in L1.
- **Lock-free by design** in the explicit matrix-fill path - parallel decomposition must guarantee non-overlapping writes.
- **`thread_local` scratch buffers** - resize, never shrink.
- **Flat containers only** in hot paths. No `std::deque`, no linked structures.
- Simplify orchestration and dispatch freely; do not reintroduce duplicate inner loops without benchmark evidence.

## Bindings Strategy

### Python
- nanobind + scikit-build-core. Zero-copy numpy via `nb::ndarray`.
- sklearn-compatible: `fit(X)`, `predict(X)`, `fit_predict(X)`.
- Release the GIL for C++ calls that can run materially longer than Python overhead.
- Publish via `uv publish` with OIDC trusted publishing.

### MATLAB
- Legacy C MEX API (`mex.h`). Handle-based object management with `mexLock`.
- `+dtwc` package: same OOP feel as Python where practical.
- snake_case function names for parity with Python. PascalCase properties remain acceptable on the MATLAB side.
- All indices are shifted at the MEX boundary for MATLAB's 1-based indexing.

## Cross-Language API Direction

- **Pairwise distances should converge on a `distance.*` namespace** across all bindings.
  - C++: `dtwc::distance::dtw(...)`
  - Python: `dtwcpp.distance.dtw(...)`
  - MATLAB: `dtwc.distance.dtw(...)`
- **Python and MATLAB should treat `distance.*` as the public pairwise surface.**
  - Root helpers such as `dtwcpp.dtw_distance(...)` and `dtwc.dtw_distance(...)` should not be preserved just for compatibility.
  - If lower-level helpers remain anywhere, they should be considered internal/back-end plumbing, not the documented API.
- **Keep direct specialized entry points alongside the generic dispatcher.**
  - Generic `dtw(..., variant=...)` is good for discoverability and docs parity.
  - Direct `soft_dtw(...)`, `wdtw(...)`, etc. remain important for repeated calls and explicitness.
- **Stateful workflows should converge on one documented flow:**
  - settings/config
  - data
  - `Problem`
  - clustering algorithm
  - scores / persisted outputs
- **Config-file loading is still a gap outside the CLI.**
  - The clean path is to add one shared settings/config representation, then expose it in Python and MATLAB.
  - Do not invent separate YAML schemas or binding-specific config loaders.

## MIP Solver

- Balinski p-median formulation. HiGHS (default) or Gurobi.
- FastPAM warm start provides a tight upper bound.
- Benders decomposition for large `N` (currently auto-dispatched above 200).
- Once medoid variables are fixed, assignment is TU and the LP relaxation is integral.
