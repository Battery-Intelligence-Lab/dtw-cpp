# DTWC++ Architecture & Design Decisions

## Layered Design

**Layer 1: Core** — DTW distances, metric abstraction, constraints, distance matrix builder. Minimal stable headers.

**Layer 2: Algorithms** — PAM, CLARA, CLARANS, hierarchical. Evaluation metrics (silhouette, DB, CH, Dunn, ARI, NMI). In `dtwc/algorithms/`.

**Layer 3: IO** — Data loaders, CSV parsing. Separate from core compute.

**Layer 4: Bindings** — Python (nanobind + scikit-build-core), MATLAB (legacy C MEX + handle classes). Both call stable core APIs.

## DTW Variant Architecture

- **Separate functions per variant** (WDTW, ADTW, DDTW, Soft-DTW). No parameterized uber-function.
- **std::function dispatch in Problem** — ~2ns overhead is negligible vs 1-100ms DTW.
- **WDTW/ADTW need recurrence changes** — they CANNOT be metric swaps.
- **DDTW is a preprocessing step** — derivative transform before standard DTW.
- **Soft-DTW is a separate algorithm** — softmin replaces std::min.

## Performance Principles

- DTW is **latency-bound** (~10 cycles/cell recurrence). Rolling buffer fits in L1.
- **Lock-free by design** — parallel decomposition guarantees non-overlapping writes.
- **thread_local scratch buffers** — resize, never shrink.
- **Flat containers only** in hot paths. No std::deque, no linked structures.
- Optimize only when benchmarks show wins. Record in `/benchmarks/README.md`.

## Bindings Strategy

### Python
- nanobind + scikit-build-core. Zero-copy numpy via `nb::ndarray`.
- sklearn-compatible: `fit(X)`, `predict(X)`, `fit_predict(X)`.
- GIL release for any C++ call >10ms.
- Publish via `uv publish` with OIDC trusted publishing.

### MATLAB
- Legacy C MEX API (`mex.h`). Handle-based object management with `mexLock`.
- `+dtwc` package: same OOP feel as Python (CasADi-style).
- snake_case function names (intentional Python parity). PascalCase properties.
- All indices +1 at MEX boundary (MATLAB 1-based).

## MIP Solver

- Balinski p-median formulation. HiGHS (default) or Gurobi.
- FastPAM warm start provides tight upper bound.
- Benders decomposition for N > 200 (auto-dispatch).
- Once medoid variables fixed, assignment is TU → LP-integral.
