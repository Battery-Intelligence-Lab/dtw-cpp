# Wave 2A: Clustering Algorithms - Adversarial Revised Plan

## Review History

1. Claude draft: feature-forward plan with CondensedDistanceMatrix, hierarchical clustering, CLARANS, improved FastCLARA, and CLI integration.
2. First adversarial pass: identified the real architectural blocker first - `Problem::set_data()` currently forces `O(N^2)` dense-matrix allocation through `refreshDistanceMatrix()`.
3. Expanded adversarial pass: correctly found existing FastCLARA correctness bugs and correctly rejected naive large-N CLARANS.

The expanded pass was useful, but it overcorrected on CLARANS. "Not viable as the main large-scale DTW algorithm" is a strong and defensible claim. "Drop it entirely" is weaker. There is still a plausible mid-ground role for CLARANS, but only as a bounded, benchmark-gated algorithm, not as a headline scalability story.

## Final Position

The wave should be split by regime instead of pretending one algorithm fits all scales.

- Small N, dense exact regime: FastPAM and hierarchical clustering.
- Mid-ground, budgeted local-search regime: CLARANS, but only if it earns its keep against FastCLARA and FastPAM.
- Large N, sample-based regime: FastCLARA.

That is the honest shape of the problem.

## What Was Right, What Was Wrong

### Correct from the adversarial expansion

- The `Problem` memory model is still the phase-0 blocker.
- FastCLARA has real existing correctness bugs:
  - `ndim` is not propagated into the subsample `Data`
  - `missing_strategy` is not propagated into the subsample `Problem`
- Hierarchical clustering needs a separate cut step that can see real distances.
- Hierarchical clustering must assert a fully computed matrix and must not silently read uncomputed zeros.

### Overstated from the adversarial expansion

- "Drop CLARANS" is too blunt.

The more accurate statement is:

- do not market CLARANS as the scalable DTW answer
- do not put it on the critical path for Wave 2A
- do keep it as an experimental mid-ground candidate if it is implemented with hard budgets and benchmark gates

## Why CLARANS Still Has A Plausible Role

Naive CLARANS on full DTW workloads is a bad idea. The expanded review was right about that. A textbook `num_local * max_neighbor * N` search with expensive DTW lookups blows up quickly.

But that does not imply CLARANS has no role at all.

There are two plausible uses:

1. A mid-ground algorithm for moderate `N`, where:
   - hierarchical is already too expensive
   - full FastPAM is too slow or too memory-hungry
   - FastCLARA may suffer from sample bias

2. A refinement stage seeded from FastCLARA or another initializer, where:
   - the starting medoids are already decent
   - CLARANS uses a strict swap budget
   - the goal is not global search, but "buy some quality improvement with bounded extra work"

That second role is the stronger one. A budgeted CLARANS-refine pass is much more defensible than selling standalone CLARANS-from-random-start as a large-scale DTW algorithm.

## Real Blockers And Design Constraints

### 1. `Problem` still allocates a dense matrix too early

Current path:

- `Problem::set_data()`
- `refreshDistanceMatrix()`
- `DenseDistanceMatrix::resize(size())`

Consequence:

- any algorithm that merely accepts a populated `Problem&` inherits `O(N^2)` memory before it asks for a single distance

That invalidates any large-scale claim until fixed.

### 2. `distByInd()` must stay safe under parallel callers

If distance access becomes lazy and non-dense by default, `distByInd()` must not perform unsafe shared writes from parallel code. The safe first step is simple:

- if the dense matrix exists and the entry is computed, return it
- if the dense matrix does not exist, compute through `dtw_fn_` and return without caching

That preserves correctness and thread safety. Fancy caching can come later.

### 3. Hierarchical clustering is a dense small-N feature

Do not pretend otherwise.

If the first implementation is generic agglomerative `O(N^3)`, it needs:

- a hard `max_points` guard
- explicit documentation that it is small-N only
- deterministic tie-breaking

### 4. CLARANS must be budgeted, not open-ended

If CLARANS survives the plan, it must not arrive as a textbook algorithm with only `num_local` and `max_neighbor` and a vague promise of scale.

It needs hard resource controls such as:

- `max_neighbor`
- optional `max_dtw_evals`
- optional `time_budget_ms`

If the implementation cannot be bounded, it does not belong in this wave.

## Revised Scope

## Track A: Required for Wave 2A

- defer dense distance-matrix allocation
- fix FastCLARA correctness bugs
- improve FastCLARA sample sizing and parent/subproblem behavior
- add hierarchical clustering as an explicitly dense small-N feature
- integrate the proven pieces into CLI and headers

## Track B: Experimental, benchmark-gated

- CLARANS, preferably as a bounded mid-ground or refinement algorithm

The distinction matters. Track A is what should ship. Track B is what should earn promotion through evidence.

## Revised Task Plan

## Task 0: Defer Dense Distance-Matrix Allocation

Goal: stop `Problem::set_data()` from forcing `O(N^2)` memory immediately.

Required changes:

- `refreshDistanceMatrix()` should reset state and rebind DTW, but not eagerly resize to `N x N`
- `fillDistanceMatrix()` should allocate the dense matrix on first real need
- `distByInd()` should:
  - return cached values when the dense matrix exists and the entry is computed
  - otherwise compute directly and return without mutating shared state

This is the real architectural blocker. Everything else depends on it.

Files:

- `dtwc/Problem.hpp`
- `dtwc/Problem.cpp`
- `dtwc/core/distance_matrix.hpp`

Tests:

- `set_data()` no longer allocates full dense storage by default
- `fillDistanceMatrix()` still materializes and fills the matrix correctly
- `distByInd()` still works before dense fill
- existing FastPAM path still works unchanged

## Task 1: Fix Existing FastCLARA Bugs First

Fix the bugs already found:

- propagate `prob.data.ndim` into the subsample `Data`
- propagate `prob.missing_strategy` into the subsample `Problem`
- also propagate:
  - `band`
  - `variant_params`
  - `distance_strategy`
  - `verbose`

This is not optional cleanup. These are correctness bugs.

Files:

- `dtwc/algorithms/fast_clara.cpp`
- `tests/unit/algorithms/unit_test_fast_clara.cpp`

Tests:

- multivariate FastCLARA uses correct `ndim`
- FastCLARA with missing-data strategy does not crash or silently change semantics
- existing FastCLARA tests still pass

## Task 2: Improve FastCLARA Properly

Required changes:

- move sample-size logic into a named helper
- use the improved rule from Schubert and Rousseeuw instead of the old fixed heuristic
- keep the parent `Problem` non-dense by default after Task 0
- keep subsample problems dense, because FastPAM on the sample really does want the local matrix

Optional but sensible follow-up:

- consider parallelizing parent-level assignment using direct DTW calls rather than unsafe cached writes

Why this matters:

- FastCLARA is still the primary large-N algorithm in this wave
- it needs to be correct before it needs to be clever

## Task 3: Factor Shared Medoid Utilities

Create an internal helper layer, for example:

- `dtwc/algorithms/detail/medoid_utils.hpp`

Factor reusable logic out of `fast_pam.cpp`:

- assign-to-nearest-medoid
- total-cost accumulation
- nearest / second-nearest support where needed

Important design rule:

- utilities should accept a distance function `(int, int) -> double`
- they should not be hardwired to a dense `Problem` assumption

This prepares both FastCLARA improvements and any later CLARANS work without duplicating brittle logic.

## Task 4: Hierarchical Clustering, Small-N Only

Create:

- `dtwc/algorithms/hierarchical.hpp`
- `dtwc/algorithms/hierarchical.cpp`
- `tests/unit/algorithms/unit_test_hierarchical.cpp`

API direction:

```cpp
namespace dtwc::algorithms {

enum class Linkage { Single, Complete, Average };

struct DendrogramStep {
  int cluster_a;
  int cluster_b;
  double distance;
  int new_size;
};

struct Dendrogram {
  std::vector<DendrogramStep> merges;
  int n_points = 0;
};

struct HierarchicalOptions {
  Linkage linkage = Linkage::Average;
  int max_points = 2000;
};

Dendrogram build_dendrogram(Problem& prob, const HierarchicalOptions& opts = {});
core::ClusteringResult cut_dendrogram(const Dendrogram& dendrogram, Problem& prob, int k);

} // namespace dtwc::algorithms
```

Required behavior:

- assert the dense matrix is fully computed before building the dendrogram
- keep a separate working inter-cluster distance structure
- do not mutate `Problem` inside `cut_dendrogram`
- use deterministic tie-breaking
- throw when `N > max_points`

Do not introduce a public `CondensedDistanceMatrix` yet. That is premature until a second real consumer exists.

## Task 5: Experimental CLARANS, Reframed

CLARANS stays in the plan only under a stricter contract.

### Positioning

- not the flagship scalable algorithm
- not a required Wave 2A deliverable
- not exposed in the CLI on day one

### Acceptable roles

- bounded mid-ground clustering for moderate `N`
- bounded refinement of medoids produced by FastCLARA or another initializer

### Preferred implementation order

1. Internal prototype first
2. Seed from a reasonable initializer
3. Add hard budgets
4. Benchmark against FastCLARA and FastPAM
5. Promote only if it wins somewhere meaningful

### API direction

If implemented, prefer an options struct that can actually bound runtime:

```cpp
struct CLARANSOptions {
  int n_clusters = 3;
  int num_local = 2;
  int max_neighbor = -1;
  int64_t max_dtw_evals = -1;
  int64_t time_budget_ms = -1;
  unsigned random_seed = 42;
};
```

Even better for the first prototype:

- implement an internal `clarans_refine(...)` helper that starts from initial medoids
- only expose a standalone public CLARANS API if benchmarks justify it

### Acceptance criteria

CLARANS earns promotion only if it satisfies at least one of these on a defined mid-ground benchmark set:

- materially better cost than FastCLARA at acceptable additional runtime
- materially lower runtime than FastPAM while staying close enough in cost

If it never lands on a useful Pareto point, delete it from the wave.

That is the correct adversarial standard.

## Task 6: CLI, Headers, And Changelog

Ship only the proven pieces.

Required:

- add hierarchical support to the CLI
- reuse `-k/--clusters` for hierarchical cut level
- add `--linkage {single,complete,average}`
- update `dtwc.hpp`
- update `CHANGELOG.md`

Do not expose CLARANS in the CLI until the experimental task passes its acceptance criteria.

## Execution Order

```text
T0 defer dense allocation
  ->
T1 fix FastCLARA correctness bugs
  ->
T2 improve FastCLARA
  ->
T3 factor shared medoid utilities
  ->
T4 hierarchical clustering (small-N only)
  ->
T6 CLI + headers + changelog

T5 experimental CLARANS runs beside or after T3, but does not block shipping Track A
```

## Verification Plan

### Correctness

- all existing FastPAM tests still pass
- all existing FastCLARA tests still pass
- new hierarchical tests pass
- new regression tests cover:
  - deferred dense allocation
  - multivariate FastCLARA
  - missing-data FastCLARA
  - hierarchical precondition failures

### Performance and scale claims

Do not claim scale from toy examples.

Required evidence:

- one smoke test showing `Problem::set_data()` no longer forces dense `N x N` allocation
- FastCLARA runtime and quality checks on medium and larger datasets
- hierarchical explicitly benchmarked only in the small-N regime

### CLARANS-specific evidence

If CLARANS is attempted, benchmark it in the regime where it is supposed to matter, not in a fantasy regime where it obviously fails.

At minimum compare against:

- FastCLARA
- FastPAM

Across:

- moderate `N`
- multiple `k`
- at least one expensive DTW setting

Report:

- wall time
- number of DTW evaluations
- final total cost
- initializer used

If the algorithm is only good when hidden behind cherry-picked settings, that is a rejection signal, not a success story.

## Definition Of Done

- `Problem` no longer forces dense `O(N^2)` allocation at data load time
- FastCLARA is corrected for multivariate and missing-data inputs
- FastCLARA remains the primary large-N algorithm
- hierarchical clustering ships as an explicitly dense small-N feature
- CLARANS, if kept, is positioned honestly as a benchmark-gated mid-ground or refinement algorithm
- only proven features reach the CLI

## Bottom Line

The expanded adversarial review was right to attack naive CLARANS. It was wrong to conclude that the only honest answer is deletion.

The stronger plan is:

- fix the memory model first
- fix the existing FastCLARA bugs second
- ship hierarchical honestly as small-N
- keep CLARANS on a short leash as an experimental mid-ground candidate

That preserves the adversarial standard without confusing "not the main answer" with "has no role at all."
