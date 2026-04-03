# Plan: Full OOP MATLAB Bindings (CasADi-style, Phased)

## Context

The Python bindings expose 49 public symbols (8 enums, 13 structs, ~25 functions) via
nanobind (790 lines). The MATLAB MEX has 3 commands (~12% coverage). Goal: CasADi-style
API parity — same class names, same methods, same feel.

Two adversarial reviews identified critical safety issues (mexLock, fast_pam side-effects)
and a ~60% API gap vs Python. The plan is phased: Phase 1 ships the core workflow,
Phase 2 fills remaining gaps.

## Phase 1 (this implementation): Core Workflow

**Scope:** Problem class, all DTW variants, all algorithms, all scoring, distance matrix.
This covers the "90% use case" — users can do everything they'd do in Python.

**Deferred to Phase 2:** Enums as standalone classes, DenseDistanceMatrix class, Data class,
MIPSettings/CUDASettings as MATLAB classes, checkpointing, CUDA functions, I/O utilities,
`compile.m` standalone build script.

## Architecture

### Handle-based MEX with `mexLock`

```
MATLAB                        MEX (C++)                     C++ Core
+dtwc/Problem.m  ------> dtwc_mex('Problem_new')  ------> new Problem()
  .set_data()    ------> dtwc_mex('Problem_set_data', h)   prob->set_data()
  .Band = 50     ------> dtwc_mex('Problem_set_band', h)   prob->band = 50
  delete(obj)    ------> dtwc_mex('Problem_delete', h)      shared_ptr released
```

**CRITICAL: `mexLock()`** on first MEX call. This prevents MATLAB from unloading the MEX
DLL while handle objects exist. Without it, `delete()` during garbage collection crashes.

**HandleManager** is a static `unordered_map<uint64, shared_ptr<T>>`. Thread safety:
MATLAB never calls `mexFunction` concurrently — the map is only accessed from the MATLAB
main thread. Document this assumption. Do NOT add mutex (unnecessary overhead).

**`mexAtExit` callback** must drain all HandleManagers to ensure C++ destructors run in
order before DLL teardown (avoids OpenMP static destructor crash).

### Error handling (longjmp-safe)

```cpp
std::string error_id, error_msg;
try {
    // ... all C++ work ...
} catch (const std::invalid_argument &e) {
    error_id = "dtwc:invalidArgument"; error_msg = e.what();
} catch (const std::out_of_range &e) {
    error_id = "dtwc:outOfRange"; error_msg = e.what();
} catch (const std::runtime_error &e) {
    error_id = "dtwc:runtime"; error_msg = e.what();
} catch (const std::exception &e) {
    error_id = "dtwc:internal"; error_msg = e.what();
} catch (...) {
    error_id = "dtwc:internal"; error_msg = "Unknown C++ exception";
}
if (!error_msg.empty())
    mexErrMsgIdAndTxt(error_id.c_str(), "%s", error_msg.c_str());
```

### Index conversion rules

- **labels** `[0, k)` → `[1, k]` (cluster IDs, MATLAB 1-based)
- **medoid_indices** `[0, N)` → `[1, N]` (data point indices, MATLAB 1-based)
- **dist_by_ind(i, j)** — MATLAB passes 1-based, MEX subtracts 1
- These are semantically different transformations that happen to both be "+1"

### Enums: string-based dispatch (MATLAB convention)

Instead of exposing C++ enum classes, MATLAB uses string arguments (standard MATLAB pattern):

```matlab
prob.MissingStrategy = 'arow';          % maps to MissingStrategy::AROW
prob.DistanceStrategy = 'pruned';       % maps to DistanceMatrixStrategy::Pruned
opts.linkage = 'complete';              % maps to Linkage::Complete
```

This is the standard MATLAB pattern (e.g., `optimoptions('fmincon', 'Algorithm', 'sqp')`).

### Naming convention: snake_case (intentional Python parity)

MATLAB convention is camelCase, but we intentionally use snake_case for function names
to match Python exactly. This is a deliberate trade-off documented in the help text.
CasADi uses camelCase; we prioritize cross-language code portability over MATLAB convention.

Property names use PascalCase (MATLAB convention): `Band`, `Verbose`, `MaxIter`.

### ClusteringResult: return as MATLAB struct (not a class)

```matlab
result = dtwc.fast_pam(prob, 3);
result.labels          % int32 row vector (1-based)
result.medoid_indices  % int32 row vector (1-based)
result.total_cost      % double scalar
result.iterations      % int32 scalar
result.converged       % logical scalar
```

Returned directly from MEX as a struct. No handle management needed.

### Dendrogram: return as MATLAB struct (not a handle)

```matlab
dend = dtwc.build_dendrogram(prob, 'Linkage', 'average');
dend.merges    % N-1 x 4 double matrix [cluster_a, cluster_b, distance, new_size]
dend.n_points  % int32 scalar
```

For `cut_dendrogram`, the MEX reconstructs the C++ Dendrogram from this struct.

### fast_pam must store results back into Problem

**CRITICAL:** After calling `dtwc::fast_pam()`, the MEX must store results back:

```cpp
prob->set_numberOfClusters(n_clusters);
prob->centroids_ind = result.medoid_indices;
prob->clusters_ind = result.labels;
```

This is required because `silhouette(prob)`, `davies_bouldin_index(prob)`, etc. read
from `prob.clusters_ind` and `prob.centroids_ind`. The Python binding does exactly this.

### Problem.m: handle class with cached + dependent properties

**Cached** (stored on MATLAB side, synced via MEX):
- `Band`, `Verbose`, `MaxIter`, `NRepetition` — scalar values, set via setter

**Dependent** (always read from C++ via MEX):
- `Size`, `ClusterSize`, `Name`, `CentroidsInd`, `ClustersInd`

**Custom `disp()` method** — single bulk MEX call `Problem_get_info` that returns a
struct with all display-worthy fields. Avoids 11+ individual MEX calls.

**Destructor safety:**

```matlab
function delete(obj)
    if obj.Handle > 0
        try
            dtwc_mex('Problem_delete', obj.Handle);
        catch
            % MEX may be unloaded during shutdown
        end
        obj.Handle = uint64(0);
    end
end
```

## Files

### MEX gateway: `bindings/matlab/dtwc_mex.cpp` (rewrite, ~600 lines)

MEX commands:

```
% Problem lifecycle
Problem_new(name) -> handle
Problem_delete(handle)
Problem_get_info(handle) -> struct (for disp)

% Problem properties (get/set)
Problem_set_data(handle, data_matrix)
Problem_set_band(handle, band)
Problem_get_band(handle) -> double
Problem_set_verbose(handle, bool)
Problem_set_max_iter(handle, int)
Problem_set_n_repetition(handle, int)
Problem_set_n_clusters(handle, k)
Problem_set_missing_strategy(handle, string)
Problem_set_distance_strategy(handle, string)
Problem_set_variant(handle, variant_string, [params...])
Problem_get_size(handle) -> double
Problem_get_cluster_size(handle) -> double
Problem_get_name(handle) -> string
Problem_get_centroids(handle) -> int32 row (1-based)
Problem_get_clusters(handle) -> int32 row (1-based)
Problem_is_distance_matrix_filled(handle) -> logical

% Problem methods
Problem_fill_distance_matrix(handle)
Problem_dist_by_ind(handle, i, j) -> double  (i,j 1-based)
Problem_cluster(handle)
Problem_find_total_cost(handle) -> double
Problem_get_distance_matrix(handle) -> NxN double
Problem_set_distance_matrix(handle, NxN double)

% DTW distance functions (stateless)
dtw_distance(x, y, band) -> double
ddtw_distance(x, y, band) -> double
wdtw_distance(x, y, band, g) -> double
adtw_distance(x, y, band, penalty) -> double
soft_dtw_distance(x, y, gamma) -> double
soft_dtw_gradient(x, y, gamma) -> vector
dtw_distance_missing(x, y, band) -> double
dtw_arow_distance(x, y, band) -> double
compute_distance_matrix(data, band) -> NxN double
derivative_transform(x) -> vector
z_normalize(x) -> vector

% Algorithms (take Problem handle, return struct)
fast_pam(handle, k, max_iter) -> ClusteringResult struct
fast_clara(handle, k, sample_size, n_samples, max_iter, seed) -> struct
clarans(handle, k, num_local, max_neighbor, max_dtw_evals, seed) -> struct
build_dendrogram(handle, linkage_string, max_points) -> Dendrogram struct
cut_dendrogram(merges, n_points, handle, k) -> ClusteringResult struct

% Scoring (take Problem handle)
silhouette(handle) -> vector
davies_bouldin_index(handle) -> double
dunn_index(handle) -> double
inertia(handle) -> double
calinski_harabasz_index(handle) -> double
adjusted_rand_index(labels1, labels2) -> double
normalized_mutual_information(labels1, labels2) -> double
```

### MATLAB +dtwc package files

| File | Type | Description |
|------|------|-------------|
| `Problem.m` | handle class | Full OOP wrapper with cached/dependent properties |
| `DTWClustering.m` | handle class | Update: add Variant, WdtwG, AdtwPenalty, MissingStrategy |
| `dtw_distance.m` | function | Existing (keep) |
| `ddtw_distance.m` | function | New |
| `wdtw_distance.m` | function | New |
| `adtw_distance.m` | function | New |
| `soft_dtw_distance.m` | function | New |
| `soft_dtw_gradient.m` | function | New |
| `dtw_distance_missing.m` | function | New |
| `dtw_arow_distance.m` | function | New |
| `compute_distance_matrix.m` | function | Existing (update: add Metric param) |
| `derivative_transform.m` | function | New |
| `z_normalize.m` | function | New |
| `fast_pam.m` | function | New — takes Problem, returns struct |
| `fast_clara.m` | function | New — takes Problem + Name-Value opts |
| `clarans.m` | function | New — takes Problem + Name-Value opts |
| `build_dendrogram.m` | function | New — takes Problem + Linkage/MaxPoints |
| `cut_dendrogram.m` | function | New — takes dendrogram struct + Problem + k |
| `silhouette.m` | function | New |
| `davies_bouldin_index.m` | function | New |
| `dunn_index.m` | function | New |
| `inertia.m` | function | New |
| `calinski_harabasz_index.m` | function | New |
| `adjusted_rand_index.m` | function | New |
| `normalized_mutual_information.m` | function | New |

### Test file: `bindings/matlab/test_mex.m` (expand to ~20 tests)

### CHANGELOG.md update

## Verification

1. Build: `cmake --build build --config Release --target dtwc_mex`
2. MATLAB: `matlab -batch "addpath('build/bin','bindings/matlab'); test_mex"`
3. Check output for "ALL TESTS PASSED" (ignore exit code — OpenMP teardown issue)
4. C++ tests: `ctest --test-dir build --build-config Release` (62/62)
5. API parity: every Phase 1 Python function has a MATLAB equivalent
