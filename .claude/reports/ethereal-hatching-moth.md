# Wave 2A: Clustering Algorithms - Adversarial Review and Revised Plan

## Verdict

Claude's draft has the right feature list, but the plan overclaims scale and under-specifies the architecture. The biggest problem is not CLARANS or hierarchical linkage logic. It is the current `Problem` contract.

Today, `Problem::set_data()` calls `refreshDistanceMatrix()`, which resizes a `DenseDistanceMatrix` to `N x N`. That means any algorithm that takes `Problem&` already inherits quadratic memory before it computes a single DTW distance. Under the current design, "100K-1M series" is not a serious claim. It is impossible on ordinary hardware.

The original plan also asks `Dendrogram::cut()` to compute medoids from merge history alone. That is not possible with the proposed type. Merge history is not enough to recover medoids without access to the underlying pairwise distances.

So the revised plan below starts with the real blocker instead of pretending the rest of the wave can be implemented independently.

## Findings Against The Original Draft

### 1. The scalability claim is false under the current `Problem` design

Relevant code path:

- `Problem::set_data()` -> `refreshDistanceMatrix()`
- `refreshDistanceMatrix()` -> `distMat.resize(size())`
- `DenseDistanceMatrix::resize(n)` allocates `n * n` doubles plus computed flags

Consequence:

- `fast_clara(prob, ...)` and any future `clarans(prob, ...)` already pay `O(N^2)` memory just by receiving a populated `Problem`.
- That directly contradicts the stated goal of supporting `100K-1M` time series.

This is the phase-0 blocker. Do not bury it under five feature tasks and call that a plan.

### 2. `CondensedDistanceMatrix` is premature as a public API

The draft creates a new public `core::CondensedDistanceMatrix`, but its first user is only hierarchical clustering. Worse, the proposed `from_dense()` path duplicates matrix storage at peak memory if the source is the existing `DenseDistanceMatrix`.

That is exactly backwards:

- first prove the algorithm and the ownership model
- then factor out a reusable matrix type if a second real consumer exists

For v1, hierarchical clustering can operate directly on the existing dense matrix and be explicitly scoped to small `N`.

### 3. The hierarchical design is internally inconsistent

The draft proposes:

- `Dendrogram` stores only `merges` and `n_points`
- `Dendrogram::cut(int k) const` returns `core::ClusteringResult`
- medoids are computed as within-cluster argmin of sum-of-distances

That does not work. `Dendrogram::cut()` has no access to:

- the original distance matrix
- the original `Problem`
- any cluster membership cost table

So it cannot compute medoids.

The plan also says a generic `O(N^3)` agglomerative implementation is "fine for N < 5000". That is too loose to be credible. If this ships as a naive all-linkage `O(N^3)` implementation, it needs a hard small-`N` limit and explicit documentation that it is not a large-scale algorithm.

### 4. The CLARANS task is too naive for a "large-scale" wave

The draft says CLARANS should use `prob.distByInd(i, j)` directly and rely on lazy caching in `DenseDistanceMatrix`.

That has two problems:

1. It is still coupled to the dense-cache architecture that breaks the scale claim.
2. It throws away reusable logic that already exists in `fast_pam.cpp` for nearest / second-nearest caching and total-cost accounting.

If CLARANS is supposed to matter, it should share the medoid-assignment machinery with FastPAM instead of starting over from a textbook implementation.

### 5. CLI integration is correct to leave last, but the proposed interface is clumsy

The draft adds:

- `--method hierarchical`
- `--linkage average`
- `--cut-k 5`

But the CLI already has `-k/--clusters`. There is no reason to invent a second cluster-count flag for hierarchical clustering. Reuse `-k` as the cut level and make `--linkage` method-specific.

### 6. The draft is missing deterministic tie-breaking rules

For hierarchical clustering and CLARANS, tests will become flaky unless the plan defines how ties break.

At minimum:

- hierarchical merge ties must break by stable cluster id ordering
- medoid ties must break by lowest original point index
- CLARANS swap ties must either reject neutral swaps or choose deterministically

If the plan does not state this, the tests will discover it the hard way.

## Revised Scope

Split Wave 2A into two tracks instead of pretending everything targets the same scale regime:

- Track A: scalable medoid algorithms (`CLARANS`, improved `FastCLARA`)
- Track B: small-`N` analytical clustering (`hierarchical`)

Those are different engineering problems and they should not share the same success criteria.

## Revised Task Plan

## Task 0: Fix The Distance-Access Contract First

Primary goal: make scalable algorithms possible without forcing `O(N^2)` memory at `Problem` construction time.

Minimal viable design:

- Add a cache policy to `Problem`, for example:
  - `Dense`
  - `Disabled`
- Keep `Dense` as the default for backward compatibility.
- When cache is `Disabled`, `refreshDistanceMatrix()` must not allocate an `N x N` matrix.
- In disabled mode, `distByInd(i, j)` computes directly via `dtw_fn_` and returns the value without storing it in the dense matrix.
- Add an explicit helper for algorithms that truly need the full matrix, for example:
  - `fillDistanceMatrix()`
  - or `ensure_dense_distance_matrix()`

Why this task comes first:

- without it, CLARANS and large-scale CLARA are still quadratic-memory algorithms
- without it, the "100K-1M" language is still nonsense

Files likely involved:

- `dtwc/Problem.hpp`
- `dtwc/Problem.cpp`
- `dtwc/core/distance_matrix.hpp`

Tests required:

- `Problem` in dense mode behaves exactly as before
- `Problem` in disabled-cache mode can still answer `distByInd()`
- `fast_clara()` with disabled cache does not call `fillDistanceMatrix()` on the full dataset
- `clarans()` with disabled cache leaves the dense matrix unfilled

Non-goal for Task 0:

- do not build a fancy LRU cache yet
- dense vs disabled is enough for the first pass

## Task 1: Factor Shared Medoid Utilities

Before adding CLARANS, stop copy-pasting medoid logic.

Create an internal helper header such as:

- `dtwc/algorithms/detail/medoid_utils.hpp`

Move or factor shared logic out of `fast_pam.cpp`:

- nearest medoid assignment
- second-nearest distance tracking
- total cost reduction
- medoid-index validation helpers

Why:

- FastPAM already contains the right building blocks
- CLARANS and improved CLARA should not each invent their own partial copy

Public API impact:

- none

## Task 2: Implement CLARANS On Top Of The Shared Utilities

Create:

- `dtwc/algorithms/clarans.hpp`
- `dtwc/algorithms/clarans.cpp`
- `tests/unit/algorithms/unit_test_clarans.cpp`

Suggested API:

```cpp
namespace dtwc::algorithms {

struct CLARANSOptions {
  int n_clusters = 3;
  int num_local = 2;
  int max_neighbor = -1;   // -1 = auto
  unsigned random_seed = 42;
};

core::ClusteringResult clarans(Problem& prob, const CLARANSOptions& opts);

} // namespace dtwc::algorithms
```

Implementation requirements:

- require Task 0 first if the wave still claims scalability
- do not call `fillDistanceMatrix()` on the full dataset
- use shared nearest / second-nearest style bookkeeping where it helps
- keep restart-level RNG deterministic
- reject neutral swaps by default; only strictly improving swaps should be accepted

Auto `max_neighbor`:

- keep the published heuristic if desired
- but clamp it explicitly and document the formula in one helper

Tests:

- valid labels and medoids
- deterministic with fixed seed
- `k = 1`
- `k = N`
- invalid inputs throw
- multiple local minima search (`num_local > 1`) is no worse than `num_local = 1`
- with disabled cache mode, algorithm does not require a dense distance matrix

## Task 3: Improve FastCLARA The Right Way

Modify:

- `dtwc/algorithms/fast_clara.cpp`
- `tests/unit/algorithms/unit_test_fast_clara.cpp`

Required changes:

1. Replace the hardcoded auto sample-size rule with a named helper and explicit clamp logic.
2. Propagate all relevant `Problem` settings into the subsample problem:
   - `band`
   - `variant_params`
   - `missing_strategy`
   - `distance_strategy`
   - `verbose` if needed
   - `data.ndim`
3. Keep the parent problem in disabled-cache mode for the global assignment step if Task 0 is implemented.
4. Keep the subsample problem in dense mode, because FastPAM does want a full local matrix.

Important correction to the original draft:

- the sample-size tweak is not the main event
- the real improvement is making CLARA large-dataset-friendly under the actual `Problem` architecture

Tests to add:

- missing-strategy propagation
- multivariate `ndim` propagation
- sample-size helper returns the documented value
- parent full problem does not become fully materialized just because CLARA ran

## Task 4: Add Hierarchical Clustering, But Admit It Is A Small-N Feature

Create:

- `dtwc/algorithms/hierarchical.hpp`
- `dtwc/algorithms/hierarchical.cpp`
- `tests/unit/algorithms/unit_test_hierarchical.cpp`

Do not create a public `CondensedDistanceMatrix` in this task.

First implementation should:

- use the existing `DenseDistanceMatrix`
- explicitly require a full matrix
- explicitly guard against large `N`

Suggested API split:

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
  int max_points = 2000;   // hard guard for the dense implementation
};

Dendrogram build_dendrogram(Problem& prob, const HierarchicalOptions& opts = {});
core::ClusteringResult cut_dendrogram(
  const Dendrogram& dendrogram,
  Problem& prob,
  int k);

} // namespace dtwc::algorithms
```

Why this API is better:

- `build_dendrogram()` only needs to describe the merge tree
- `cut_dendrogram()` receives `Problem&`, so it can compute medoids from real distances
- the type no longer pretends that merges alone are enough to recover a full clustering result

Implementation notes:

- if the implementation is still generic `O(N^3)`, enforce `max_points` hard
- define stable tie-breaking in the plan, not after the tests fail
- for medoid extraction after a cut, break ties by smallest original index

Tests:

- 4-point hand-checked examples for each linkage
- deterministic merge order under equal distances
- `cut(1)` and `cut(N)`
- medoid correctness on small synthetic problems
- throws when `N > max_points`

Future work, not this task:

- if a second real consumer appears, then consider an internal or public condensed matrix
- if hierarchical becomes performance critical, revisit algorithm choice (`SLINK`, nearest-neighbor chain, etc.)

## Task 5: CLI, Headers, Docs, And Changelog

Modify:

- `dtwc/dtwc.hpp`
- `dtwc/dtwc_cl.cpp`
- `CHANGELOG.md`

CLI rules:

- add `clarans`
- add `hierarchical`
- reuse `-k/--clusters` as the cut level for hierarchical
- add `--linkage {single,complete,average}` only for hierarchical

Do not add:

- `--cut-k`

That flag is redundant and makes the CLI worse.

Header policy:

- add new public headers only after the implementations and tests are stable

Documentation rule:

- changelog last
- do not mark the wave as "large-scale" unless Task 0 landed

## Execution Order

```text
T0 Distance-access contract
  |
  +--> T1 Shared medoid utilities
         |
         +--> T2 CLARANS
         |
         +--> T3 FastCLARA improvements
  |
  +--> T4 Hierarchical clustering (small-N only)
         |
         +--> T5 CLI + headers + changelog
```

Rationale:

- `T0` is the blocker for any honest scalability claim
- `T1` prevents duplicate medoid logic
- `T4` is independent in feature space, but it is not part of the scale story
- CLI integration belongs at the end

## Verification Plan

### Required unit tests

- existing FastPAM tests still pass
- existing FastCLARA tests still pass
- new CLARANS tests pass
- new hierarchical tests pass

### Required regression checks

- dense-cache mode preserves current behavior
- disabled-cache mode does not silently materialize the full matrix
- CLARANS and FastCLARA do not call full-matrix code paths on the full dataset

### Required benchmark or smoke checks

Do not benchmark only tiny toy data and call it scalable.

Add at least:

- small-N quality checks where FastPAM is the reference
- medium-N runtime checks for CLARANS and FastCLARA
- one large-N smoke run in disabled-cache mode to prove the wave no longer depends on dense `N x N` storage

The last item matters more than a pretty cost table. Without it, the scale claim is still unproven.

## Definition Of Done

- The plan no longer lies about `100K-1M` support under a dense-matrix `Problem`
- CLARANS and FastCLARA can run without forcing full `N x N` storage on the full dataset
- Hierarchical clustering ships as an explicitly dense, small-`N` feature
- CLI integration is clean and does not invent redundant flags
- tie-breaking is deterministic
- tests prove both correctness and the intended memory model

## Bottom Line

The original draft tried to schedule five feature tasks as if the architecture were already compatible with large-scale clustering. It is not.

The improved plan fixes that by doing the adult work first:

- repair the distance-access contract
- reuse medoid logic instead of cloning it
- separate genuinely scalable algorithms from dense small-`N` analytics

That is the difference between a feature list and an implementation plan.
