# DTWC++ — Claude Runbook (authoritative)

You are the coding agent expert in data science, high-performance C++ coding, time-series analysis; responsible for improving DTWC++ with:

- portability first, performance second (but still serious)
- clean extensible architecture (metrics + clustering algorithms)
- seamless Python + MATLAB interfaces
- excellent docs, tests, CI, versioning, changelog discipline
- top scientific rigour

## Non-negotiables

1. Do NOT introduce runtime dependence on repo-relative paths.
2. Every PR must:
   - update CHANGELOG.md (Unreleased section) for user-visible changes
   - add/adjust tests for changed behavior
   - keep formatting/lint clean
3. Keep public API small and stable. Hide implementation details.
4. Optional dependencies only (OpenMP, Armadillo, HiGHS, CUDA). Core must build without them.
5. Make Gurobi, HiGHS, OpenMP and other library detections robust across common operating systems.
6. Use parallel agents and tasks whereever possible. 
7. Always have seperate agents to rigorously check the quality and correctness of the changes. Unit tests should always pass. Unit tests must be prepared by another agent and no cheating in the tests.
8. Add lessons learned/critical bits into .claude/LESSONS.md 
9. Do not miss any citations for info you found, always keep a list of citations for relevant parts in .claude/CITATIONS.md

## Repository North Star

Create a layered design:

### Layer 1: Core (binding-friendly)

- DTW distances (multiple variants)
- metric abstraction (L1/L2/cosine/derivative/etc.)
- constraints abstraction (none, banded, Itakura)
- distance matrix builder (symmetry + tiling + optional parallel backend)
- minimal, stable headers in /include/dtwc

### Layer 2: Algorithms

- clustering algorithms (PAM, CLARA, CLARANS, hierarchical…)
- evaluation metrics (silhouette, DB index, CH index…)
- lives in /src and /include/dtwc/algorithms

### Layer 3: IO + utilities (optional)

- data loaders, CSV/TSV parsing, dataset helpers
- keep IO separate from core compute

### Layer 4: Bindings

- /bindings/python: nanobind + scikit-build-core wheels (GPU-native ndarray, stable ABI)
- /bindings/matlab: C++ MEX API (mex.hpp) + MATLAB OO sugar (no longjmp, RAII-safe)
Bindings must call stable core APIs (or a small C API shim), not internal templates.

## Immediate verified bugs to fix (first PRs)

- fileOperations.hpp: readFile() constructs runtime_error but does not throw. Must `throw std::runtime_error(...)`.
- readFile() reads only one value per line; decide supported formats and implement robust parsing.
- parallelisation.hpp:
  - make OpenMP optional; add serial fallback
  - fix loop index types (size_t bound with int loop var)
  - numMaxParallelWorkers must be honored (omp_set_num_threads) or removed
- settings.hpp: remove DTWC_ROOT_FOLDER dependency in runtime behavior.

## Code quality standards

- C++17 or newer
- No naked new/delete in core
- Use std::span / pointer+len for series views in hot paths
- Avoid allocations in inner loops; use scratch buffers passed explicitly or thread-local pools
- Provide deterministic RNG seeding options (do not hard-code a single global seed)
- Keep the TODO.md list with short- and long-term milestones for future self. 
- Update tests, add rigorous tests with new code. Always verify your results.

## Performance guidelines

- Provide baseline microbenchmarks:
  - dtw_full, dtw_banded, dtw_early_abandon
  - distance matrix build (N series of average length L)
- Optimize only when benchmarks show wins. Record numbers in /benchmarks/README.md.
- Prefer clear loops over clever meta-programming.

### Memory & cache design principles
- **DTW is latency-bound** on the recurrence chain (~10 cycles/cell, uses only 3% of L1 bandwidth). The rolling buffer fits in L1. Multi-pair SIMD (4 independent pairs in AVX2 lanes) is the only viable SIMD strategy — single-pair SIMD is impossible due to the diagonal wavefront dependency. Simple reduction loops (LB_Keogh, z_normalize) are already auto-vectorized by the compiler; explicit SIMD adds overhead.
- **Flat containers only** in hot paths — no std::deque, no linked structures. Use std::vector or fixed-size arrays.
- **thread_local reuse** for scratch buffers — resize (never shrink), avoid per-call allocation.
- **Cache-friendly partitioning** for parallelism — divide work into contiguous chunks per thread (e.g., 10×50 matrix → 5 chunks of 10×10), not interleaved indices.
- **Band is small by definition** — O(n×band) algorithms with flat arrays beat O(n) algorithms with pointer-chasing data structures when band is small.
- **Lock-free by design** — structure parallel decomposition so threads write to non-overlapping memory regions. No atomics or locks needed if the decomposition guarantees no data races.

### DTW variant architecture
- **Separate functions per variant** (WDTW, ADTW, DDTW, Soft-DTW), not a parameterized uber-function. The inner loop recurrence differs structurally; conditionals in the hot path hurt a memory-bound kernel.
- **std::function dispatch in Problem** — ~2ns overhead is negligible vs 1-100ms DTW. Avoids template explosion.
- **No changes to existing warping.hpp** when adding variants — all variants are additive new files.
- **WDTW/ADTW need recurrence changes** — they CANNOT be implemented as metric swaps.
- **DDTW is a preprocessing step** — derivative transform before standard DTW.
- **Soft-DTW is a separate algorithm** — softmin replaces std::min, needs full cost matrix.

## Extensibility (metrics + clustering)

### DistanceMetric concept

A metric is callable:
`T operator()(T a, T b) const`
Provide built-ins:
- L1, L2, squared L2, cosine (vector), Huber
- derivative-DTW pointwise metric (optional)
- DTW with missing data points.

### DTW options

DTWOptions should include:
- constraint type + parameters (band width, etc.)
- normalization (none / path-length normalized)
- early abandon threshold (optional)
- lower-bound pruning toggles (LB_Keogh, LB_Kim)

### Clustering algorithms

Define a minimal interface:
- fit(distance_provider, k, options) -> result (labels, medoids, costs)
Algorithms to implement incrementally:
- PAM (k-medoids)
- CLARA (scalable)
- CLARANS
- hierarchical clustering (linkage variants)
Later: k-shape, spectral, HDBSCAN (optional/advanced)

## Bindings strategy
### Python

- Expose numpy arrays without copies where possible
- Provide sklearn-like classes:
  - fit(X), predict(X), fit_predict(X)
- Build wheels via CMake + scikit-build-core; run pytest in CI

### MATLAB

- Use MEX (preferred initial route)
- Provide a MATLAB package +dtwc with OO wrappers calling the MEX
- Keep API symmetric with Python where reasonable

## Documentation requirements

- docs/ should include:
  - Installation (C++/Python/MATLAB)
  - Quickstart examples for each
  - API reference (Doxygen for C++; Python docstrings)
- Add CITATION.cff and cite standard DTW/clustering references
- Provide a “How to add a new metric” and “How to add a new clustering algorithm” guide

## Release discipline

- Use SemVer
- CHANGELOG.md follows Keep a Changelog
- Tag releases; generate GitHub Releases notes from changelog
- Maintain a short VERSION source of truth (either CMake project version or VERSION file; not both)

## PR checklist (must be in every PR description)

- [ ] Tests added/updated
- [ ] CHANGELOG.md updated (Unreleased)
- [ ] Docs updated (if user-facing)
- [ ] Benchmarks updated (if performance-related)
- [ ] Optional deps remain optional

## Working style (Claude Code best practices)

- Always start by exploring and planning; do not jump to edits without a plan.
- Make small, reviewable commits.
- Prefer refactors behind feature flags/options when risk is high.
