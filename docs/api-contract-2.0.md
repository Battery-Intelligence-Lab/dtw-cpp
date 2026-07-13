STATUS: FROZEN 2026-07-07; implementation-audited for 2.0.0rc1 on 2026-07-10 — changes require a PLAN.md decision entry before editing

# DTWC++ 2.0 — API Contract (freeze artifact)

> **Task 1.1 deliverable.** This document is the frozen cross-language API surface
> implemented by 2.0. The 2026-07-10 release audit replaced pre-implementation
> status notes with live 2.0.0rc1 status and added post-freeze algorithms without
> changing the original naming, indexing, error, or output contracts.
>
> **Provenance tags.** `[live]` = the symbol exists now at the cited location;
> `[new]` = introduced by 2.0; `[rename]` = a live symbol whose canonical name
> changes (old name kept per the deprecation policy in §4). Sources read for this
> contract: `dtwc/Problem.hpp`, `dtwc/scores.hpp`, `dtwc/settings.hpp`,
> `dtwc/Data.hpp`, `dtwc/distance.hpp`, `dtwc/dtwc_cl.cpp`,
> `python/src/_dtwcpp_core.cpp`, `python/dtwcpp/_api.py`,
> `python/dtwcpp/__init__.py`, `python/dtwcpp/_hpc.py`,
> `bindings/matlab/dtwc_mex.cpp`, `bindings/matlab/+dtwc/Problem.m`,
> `bindings/matlab/+dtwc/DTWClustering.m`, and the API-surface inventory
> `.claude/reports/api-surface-2026-07-06.md`.
>
> **Implementation anchors.** Tier 1 lives in `dtwc/api.hpp`,
> `python/dtwcpp/_api.py`, and `bindings/matlab/+dtwc/`; the permanent live
> conformance routes are under `tests/conformance/` and
> `tests/unit/test_tier1_cpp_api.cpp`.

## Freeze governance and approved addenda

The contract remains frozen. Any non-additive or parity-affecting edit requires
a dated decision in `PLAN.md` that states the old rule, the approved rule, the
compatibility effect, rationale, and owner. Removing this requirement is itself
a contract change and is not permitted without the same decision process.

The following post-freeze scope decisions are approved:

1. The original common Tier-1 MATLAB method set remains `auto`, `pam`, `clara`,
   `kmedoids`, `mip`, and `hierarchical`/`hclust`. OneBatchPAM, LR-core, and
   TADPole were added later to C++ and Python; MATLAB rejects those names at
   Tier 1 rather than silently substituting another method. No previously
   accepted MATLAB method was removed. Adding the three post-freeze methods to
   MATLAB Tier 1 is owned by the 2.1 parity milestone.
2. C++ continues to accept `device="hpc"` as a valid device name, but Tier-1
   `cluster()` raises the documented `DeviceError` because the C++ API has no
   authenticated remote-transport implementation. Python remains the tested
   SLURM transport. A local CPU fallback would violate the no-silent-fallback
   rule; enabling C++ submission is owned by the Oxford ARC / 2.1 HPC gate.
3. The 2026-07-12 F7 decision corrects two CLI-specific invariants. First,
   `--ram-limit` is the Parquet series decode/materialisation cap; it does not
   select mmap distance storage or promise a whole-process RSS ceiling. Second,
   matrix-free CLI runs always emit byte-stable labels and medoids, but do not
   materialise an O(N²) matrix merely to emit distance and silhouette CSVs.
   `Result::save(dir)` retains its four-file contract; its files and CLI files
   are byte-identical whenever the corresponding CLI artifact exists.

---

## 0. Conventions (read once, applied everywhere)

**Naming law (fixed decision).** *One name per concept.* Classes/structs are
`PascalCase` in every language. Methods and free functions are `snake_case` in
every language — including MATLAB, whose historical `PascalCase` settable
properties (`Band`, `MaxIter`, `NRepetition`) become deprecated in favour of
snake_case setter methods so the three columns stay textually alignable. This is
a deliberate 2.0 break from MATLAB property idiom, taken for cross-language
consistency (CasADi model, user preference in project memory).

**Indexing.** C++ and Python are 0-based. MATLAB is 1-based; the 0↔1 conversion
happens *only* at the MEX boundary (`ivec_to_mx_1based`, `dtwc_mex.cpp:160-166`)
and is a preserved invariant (§7 item 8). Every index-valued field below
(`labels`, `medoids`, `dist_by_ind(i,j)`) obeys this.

**Precision.** Distances and the distance matrix are **always** `double`,
independent of series storage precision. `data_t = double` is the storage
default; `float32` is an explicit opt-in. Full precision story in §8.

**Two tiers.** *Tier 1* is the high-level `device → load → cluster → Result`
flow (§1). *Tier 2* is the advanced object surface — `Problem`, `DataLoader`,
`scores::*`, and the algorithm free functions (§2). Both tiers exist in all
three languages; Tier 1 is what most users touch.

**No silent fallback.** Any requested capability that cannot be delivered
(device, GPU backend, method, metric) raises a typed error (§5) — it never
quietly degrades. The Python, C++, MATLAB, and CLI routes enforce this before
expensive work.

---

## 1. Tier 1 — high-level API

The canonical flow, identical in all three languages:

```
dtwc.device("gpu")            # set the compute device once (global)
data = dtwc.load("Crop.tsv")  # lazy handle — not read yet on hpc
res  = dtwc.cluster(data, k=3)# Result: labels, medoids, score(name), save(dir), plot()
```

### 1.1 `device(name)` — global device get/set  `[live in C++/Python/MATLAB]`

| Aspect | C++ `[live]` | Python `[live]` | MATLAB `[live]` |
|---|---|---|---|
| Set | `std::string dtwc::device(std::string_view name)` | `dtwcpp.device(name: str) -> str` | `dtwc.device(name)` |
| Get | `std::string dtwc::device()` | `dtwcpp.device() -> str` (`__init__.py:123`) | `name = dtwc.device()` |
| Accepts | `"cpu"`,`"gpu"`,`"gpu:N"`,`"cuda"`,`"cuda:N"`,`"hpc"` | same (`_parse_device`, `__init__.py:78`) | same |
| Returns | normalized name (lower-cased) | normalized name | normalized name |
| Errors | `DeviceError` on unknown name (§6) | `InvalidInput`/`ValueError` today; `DeviceError` in 2.0 | `dtwc:deviceError` |
| Delegates to | `dtwc::env().set_device(name)` (Task 1.3) | module global `_DEFAULT_DEVICE` → to be backed by `Env` | MEX `set_device` → `Env` |

All three front ends delegate device validation to `dtwc::Env`. The friendly
name `"gpu"` resolves to CUDA (or Metal on macOS) at call time. C++ Tier-1 HPC
job submission remains beta and fails loudly with transport instructions; the
Python route owns the tested SLURM orchestration until a real ARC run closes it.

### 1.2 `load(source, ...)` — lazy dataset handle  `[live in C++/Python/MATLAB]`

| Parameter | C++ `[live]` | Python `[live]` | MATLAB `[live]` |
|---|---|---|---|
| signature | `dtwc::Dataset dtwc::load(source, int skip_cols=0, char delimiter=0, std::string_view name="")` | `load(source, *, skip_cols=0, delimiter=None, name=None) -> Dataset` | `ds = dtwc.load(source, 'skip_cols',0, 'delimiter','', 'name','')` |
| `source` | `std::filesystem::path` **or** `std::vector<std::vector<double>>` (overloads) | path `str`/`os.PathLike` **or** array-like | char path **or** N×L double matrix |
| `skip_cols` | leading columns to drop (id columns) | same | same |
| `delimiter` | `0` = auto from extension (`.tsv/.txt`→`\t` else `,`) | `None` = auto | `''` = auto |
| `name` | `""` = derive from filename stem | `None` = filename stem, else `"dataset"` | `''` = stem |
| result type | `dtwc::Dataset` (lazy; materialises only for local backends) | `dtwc.Dataset` (`_api.py:25`) | `dtwc.Dataset` handle |

*Contract:* `load()` performs **no I/O** — on `device="hpc"` the path is forwarded
to the cluster and never read locally (preserves the 100M-series scaling story).

### 1.3 `cluster(data, k, ...) -> Result`  `[live in C++/Python/MATLAB]`

| Parameter | C++ `[live]` | Python `[live]` | MATLAB `[live]` |
|---|---|---|---|
| signature | `dtwc::Result dtwc::cluster(const Dataset& data, int k, std::string_view method="pam", int band=-1, std::string_view device="", int max_iter=100)` | `cluster(data, k, *, method="pam", band=-1, device=None, max_iter=100) -> Result` | `res = dtwc.cluster(data, k, 'method','pam', 'band',-1, 'device','', 'max_iter',100)` |
| `data` | `Dataset` (or path/array via `load`) | `Dataset`/path/array | `Dataset`/path/matrix |
| `k` | `int` clusters | `int` | `int` |
| `method` | `"auto"·"pam"·"onebatch"·"clara"·"kmedoids"·"mip"·"lrcore"·"tadpole"·"hierarchical"` (alias `"hclust"`) | same set | MATLAB Tier 1 supports `auto`, `pam`, `clara`, `kmedoids`, `mip`, and `hierarchical`; newer algorithms use Tier 2 where bound |
| `auto` resolution | local CPU: `pam` for N≤5000, else `clara`; local GPU: `pam`; C++ HPC reaches the documented transport error before local resolution | same local rule; HPC forwards `auto` for resolution after remote materialisation | CPU-only Tier-1 alias of `pam` |
| `band` | Sakoe-Chiba band, `-1` = full | `-1` | `-1` |
| `device` | `""` = global default; else per-call override | `None` = global | `''` = global |
| `max_iter` | `100` | `100` | `100` |
| unknown `method` | `InvalidInput` (never silently PAM) | `ValueError` (`_normalize_method`, `_api.py:151`) | `dtwc:invalidArgument` |

**Deterministic Tier-1 seed (2.0 addendum).** The cross-language
invocation-local default is 42, exposed as
`dtwc::settings::DEFAULT_RANDOM_SEED`, `dtwcpp.DEFAULT_RANDOM_SEED`, and
`dtwc.default_random_seed()`. The C++/Python/MATLAB Tier-1 PAM route and the
seed-aware OneBatchPAM/CLARA routes construct or receive a local engine from
this value; `auto` inherits the resolved method. `DTWClustering` restart `i`
uses `DEFAULT_RANDOM_SEED + i` and retains the lowest-cost result, with the
schedule range-checked. `DTWCKMedoids(random_state=None)` means the same default.
CLI `--seed` defaults to 42, applies to PAM, OneBatchPAM, and CLARA, and accepts
`[0, UINT_MAX]`. Lloyd k-medoids uses the same invocation-local default and a
checked `base_seed + i` schedule for its repetitions, restoring the actual
lowest-cost repetition rather than leaving the final run in `Problem`. Direct
HiGHS and Gurobi MIP warm starts use a shared seed-aware FastPAM incumbent;
their exact model and optimum are unchanged. These calls do not consume
`dtwc::randGenerator`. Maintained invocation-local seeded paths translate
`std::mt19937_64` output through DTWC++'s versioned `portable-v1` bounded,
shuffle, weighted, and selection maps rather than vendor-defined standard
distributions, so one seed has the same schedule under MSVC STL and libstdc++.

The unseeded Tier-2 `fast_pam` overload intentionally retains its legacy mutable
`std::mt19937` engine, initially seeded 29; use `fast_pam_seeded` or MATLAB's
`Seed` option for invocation-local reproducibility. The one-argument
`init::random` and `init::Kmeanspp` functions retain the same legacy engine.
Lloyd recognizes those standard function-pointer initializers and selects their
seeded counterparts; an arbitrary user-supplied `Problem::init_fun` callback is
still invoked unchanged once per repetition and owns its own RNG policy.

### 1.4 `Result` — clustering outcome  `[live in C++/Python/MATLAB]`

Canonical class name is **`Result`** in all three languages. Python keeps
`ClusterResult` as a deprecated alias (§4).

| Member | C++ `dtwc::Result` `[live]` | Python `dtwcpp.Result` `[live]` | MATLAB `dtwc.Result` `[live]` |
|---|---|---|---|
| `labels` | `const std::vector<int>& labels() const` | `res.labels` → `np.ndarray[int]` | `res.labels` → int32 row (1-based) |
| `medoids` | `const std::vector<int>& medoids() const` | `res.medoids` → `np.ndarray[int]` | `res.medoids` → int32 row (1-based) |
| `score(name)` | `double score(std::string_view name) const` | `res.score(name: str) -> float` | `s = res.score(name)` |
| `save(dir)` | `void save(const std::filesystem::path& dir) const` | `res.save(dir)` | `res.save(dir)` |
| `plot()` | **not provided** — C++ writes plottable CSV via `save()` | `res.plot(png="clusters_2d.png", show=True)` (`_api.py:93`) | `res.plot()` |
| (aux) `cost` | `double cost() const` | `res.cost` (`_api.py:85`) | `res.cost` |
| (aux) `device` | `std::string device() const` | `res.device` | `res.device` |

*`score(name)` names* (accepted in every language; resolve to the Tier-2 `scores::*`
functions in §2.4): `"silhouette"` (returns the **mean** silhouette),
`"davies_bouldin"`, `"dunn"`, `"calinski_harabasz"`, `"inertia"`. Unknown name →
`InvalidInput` listing the valid set.

*`save(dir)` contract* (identical bytes in every language — this is the preserved
output contract, §7 item 2): writes `<name>_labels.csv` (`"name,cluster"`),
`<name>_medoids.csv` (`"cluster,medoid_index,medoid_name"`),
`<name>_distance_matrix.csv`, and `<name>_silhouettes.csv`
(`"name,cluster,silhouette"`) into `dir`. These are exactly the corresponding
CLI outputs when present. A matrix-free CLI run emits labels and medoids without
forcing the matrix-only files; this approved exception is specified in §7 item
2.

*`plot()` is Python/MATLAB only.* It renders a classical-MDS 2D scatter of the
distance matrix coloured by cluster (`_api.py:93-130`). **C++ has no `plot()`**:
it calls `save(dir)` to emit the plottable CSVs above, which any plotting tool
(or the `dtwc.visualize` skill) consumes. On an `hpc` run only labels return, so
`plot()` prints cluster sizes and returns nothing (`_api.py:99-103`).

### 1.5 `DTWClustering` — sklearn-style estimator (Python + MATLAB)

A live public class exists in both bindings and 2.0 **retains it** (it is the
scikit-learn-idiomatic entry point, distinct from the functional Tier-1 `cluster()`):
Python `dtwcpp.DTWClustering` (`python/dtwcpp/_clustering.py:36`, `BaseEstimator,
ClusterMixin`) and MATLAB `dtwc.DTWClustering` (`bindings/matlab/+dtwc/DTWClustering.m`).
It has no C++ twin (sklearn estimator idiom is language-specific) and stays
Python/MATLAB-only.

**Known cross-language parameter divergence — resolved here (was surface report §5
row 14).** The two constructors disagree on one parameter:

| Parameter | Python (`_clustering.py:84-86`) | MATLAB (`DTWClustering.m:53/76`) | 2.0 resolution |
|---|---|---|---|
| `device` / `Device` | `device=None` (present) | **absent** | MATLAB **gains** `device` (`[new]`), delegating to `Env` (§6), for parity |
| `metric` / `Metric` | **absent** | `Metric='l1'` (present) | Python **gains** `metric='l1'` (`[new]`), matching the distance-fn `metric` arg (§2.6) |

Both estimators converge on the shared constructor set `{n_clusters, variant, band,
max_iter, n_init, wdtw_g, adtw_penalty, missing_strategy, metric, device}` in 2.0.
No `metric` on Python today and no `device` on MATLAB today are the only gaps; both
are additive (`[new]`), so neither breaks an existing call. Fixed decision, not an
open item.

---

## 2. Tier 2 — advanced object surface

Retained for power users. `Problem` stays a first-class object but is *cleaned*:
snake_case methods, invariant-preserving setters (no naked public field whose
write silently de-syncs bound state), and algorithms that **write results back
into `Problem`** in pure C++ (killing the C++-vs-bindings divergence #10 in the
surface report — the bindings' explicit auto-wire at `_dtwcpp_core.cpp:573-576`
and `dtwc_mex.cpp:222-227` then deletes in Phase 2).

### 2.1 `Problem` — configuration setters

Canonical config setters (all `snake_case`, all invariant-preserving). Naked
public fields present today become private with these setters in Task 1.6.

| Concept | C++ 2.0 `[rename]` | Python 2.0 | MATLAB 2.0 | Live source |
|---|---|---|---|---|
| k | `set_n_clusters(int)` | `set_n_clusters(n)` | `set_n_clusters(k)` | C++ `set_numberOfClusters` (Problem.hpp:185); Py `set_number_of_clusters` (`_dtwcpp_core.cpp:445`); MEX already `set_n_clusters` (Problem.m:113) |
| method (enum) | `set_method(Method)` | `set_method(Method)` / `method` prop | `set_method(str)` `[new]` | field `method{Method::Kmedoids}` (Problem.hpp:129); Py rw prop `method` (`_dtwcpp_core.cpp:422`); written live by the Tier-1 path `prob.method = dtwcpp.Method.MIP` (`_api.py:186`) |
| band | `set_band(int)` | `band` prop / `set_band` | `set_band(b)` | field `band` (Problem.hpp:133); MEX `set_band` |
| max iterations | `set_max_iter(int)` | `max_iter` prop | `set_max_iter(n)` | field `maxIter` (Problem.hpp:130) |
| repetitions | `set_n_repetitions(int)` | `n_repetitions` prop | `set_n_repetitions(n)` | field `N_repetition` (Problem.hpp:131) |
| random seed | `set_random_seed(uint64_t)` | `random_seed` prop / `set_random_seed` | Tier-1 default via `dtwc.default_random_seed()`; method-specific `Seed` where exposed | field `random_seed` (Problem.hpp), default `DEFAULT_RANDOM_SEED` |
| variant (enum) | `set_variant(core::DTWVariant)` | `set_variant(DTWVariant)` | `set_variant(name[,param])` | Problem.hpp:206; `_dtwcpp_core.cpp:446` |
| variant (params) | `set_variant(core::DTWVariantParams)` — **rebinds `dtw_fn_`** | `set_variant_params(DTWVariantParams)` | `set_variant(name, param)` | Problem.hpp:207 |
| missing strategy | `set_missing_strategy(core::MissingStrategy)` | `missing_strategy` prop | `set_missing_strategy(str)` | field (Problem.hpp:135) |
| distance strategy | `set_distance_strategy(DistanceMatrixStrategy)` | `distance_strategy` prop | `set_distance_strategy(str)` | field (Problem.hpp:136) |
| lower-bound strategy | `set_lb_strategy(LowerBoundStrategy)` | `lb_strategy` prop `[new bind]` | `set_lb_strategy(str)` `[new]` | field `lb_strategy` (Problem.hpp:137) — unbound today |
| storage policy | `set_storage_policy(core::StoragePolicy)` | `storage_policy` prop `[new bind]` | `set_storage_policy(str)` `[new]` | field (Problem.hpp:138) — unbound today |
| solver | `set_solver(Solver) -> bool` | `set_solver(Solver)` `[new bind]` | `set_solver(str)` `[new]` | Problem.hpp:187 — unbound in Py/MEX today |
| MIP settings | `mip_settings` field | `mip_settings` prop | `set_mip_settings(struct)` `[new]` | Problem.hpp:140 |
| CUDA settings | `cuda_settings` field | `cuda_settings` prop `[new bind]` | — `[new]` | Problem.hpp:139 — unbound today |
| output folder | `set_output_folder(path)` | `output_folder` prop `[new bind]` | `set_output_folder(dir)` `[new]` | field (Problem.hpp:145) — unbound today |
| verbose | `set_verbose(bool)` | `verbose` prop | `set_verbose(tf)` | field (Problem.hpp:141) |
| data (owning) | `set_data(Data)` | `set_data(series, names)` | `set_data(X)` | Problem.hpp:189; `_dtwcpp_core.cpp:447`; MEX `set_data` |
| data (view) | `set_view_data(Data)` | `set_view_data(...)` `[new bind]` | — | Problem.hpp:197 — unbound today |

### 2.2 `Problem` — distance-matrix & clustering methods `[rename: camelCase → snake_case]`

| C++ live (Problem.hpp) | C++ 2.0 canonical | Python 2.0 | MATLAB 2.0 |
|---|---|---|---|
| `refreshDistanceMatrix()` (:178) | `refresh_distance_matrix()` | `refresh_distance_matrix()` (live) | `refresh_distance_matrix()` `[new]` |
| `readDistanceMatrix(path)` (:184) | `read_distance_matrix(path)` | `read_distance_matrix(path)` `[new bind]` | `read_distance_matrix(path)` `[new]` |
| `maxDistance()` (:209) | `max_distance()` | `max_distance()` (live) | `max_distance()` `[new]` |
| `distByInd(i,j)` (:210) | `dist_by_ind(i,j)` | `dist_by_ind(i,j)` (live) | `dist_by_ind(i,j)` (1-based, live) |
| `isDistanceMatrixFilled()` (:225) | `is_distance_matrix_filled()` | `is_distance_matrix_filled()` (live) | `is_distance_matrix_filled()` (live) |
| `fillDistanceMatrix()` (:246) | `fill_distance_matrix()` | `fill_distance_matrix()` (live) | `fill_distance_matrix()` (live) |
| `printDistanceMatrix()` (:247) | `print_distance_matrix()` | `print_distance_matrix()` `[new bind]` | — |
| `writeDistanceMatrix([name])` (:249) | `write_distance_matrix([name])` | `write_distance_matrix()` (live) | — |
| `dense_distance_matrix()` (:236) | `dense_distance_matrix()` (unchanged) † | `distance_matrix()` ‡ (was live `distance_matrix_numpy()`, `_dtwcpp_core.cpp:457`; one name; zero-copy where safe, Phase 2.1) | `get_distance_matrix()` → **rename** `distance_matrix()` |
| — (writer) | `set_distance_matrix(...)` | `set_distance_matrix(...)` (was live `set_distance_matrix_from_numpy()`, `_dtwcpp_core.cpp:492`, used by `_api.py:224`) | `set_distance_matrix(D)` (live, Problem.m:164) |
| `use_mmap_distance_matrix(path)` (:244) | `use_mmap_distance_matrix(path)` | `use_mmap_distance_matrix(path)` `[new bind]` | — |
| `findTotalCost()` (:269) | `find_total_cost()` | `find_total_cost()` (live) | `find_total_cost()` (live) |
| `assignClusters()` (:270) | `assign_clusters()` | `assign_clusters()` (live) | — |
| `calculateMedoids()` (:272) | `calculate_medoids()` | `calculate_medoids()` (live) | — |
| `cluster()` (:262) | `cluster()` | `cluster()` (live) | `cluster()` `[new]` |
| `cluster_by_MIP()` (:263) | `cluster_by_mip()` | — | — |
| `cluster_by_kMedoidsLloyd()` (:264) | `cluster_by_kmedoids_lloyd()` | — | — |
| `printClusters()` (:252) | `print_clusters()` | `print_clusters()` (live) | — |
| `writeClusters()` (:253) | `write_clusters()` | `write_clusters()` (live) | — |
| `writeMedoidMembers(iter,rep=0)` (:255) | `write_medoid_members(iter, rep=0)` | `[new bind]` | — |
| `writeSilhouettes()` (:256) | `write_silhouettes()` | `write_silhouettes()` (live) | — |

**† Name collision (unresolved — flagged, not silently renamed).** C++ has a
*second*, live `Problem::distance_matrix()` (Problem.hpp:231/233) that returns the
internal `std::variant<DenseDistanceMatrix, MmapDistanceMatrix>` by reference — a
different return type from the Python/MATLAB `distance_matrix()` NxN-array
accessor above. The token `distance_matrix()` therefore denotes two different
things across languages. 2.0 does **not** unify them here; the C++ variant
accessor keeps its name (it is the storage-level handle, not a user array) and the
divergence is recorded as an open item (§10 item 6). Task 1.6 confirms the final
C++ spelling (candidate: `distance_matrix_storage()`).

**‡ Python read/write rename.** The live Python read accessor is
`distance_matrix_numpy()` (`_dtwcpp_core.cpp:457`) and the live writer is
`set_distance_matrix_from_numpy()` (`_dtwcpp_core.cpp:492`, called by
`_api.py:224`). 2.0 renames the pair to `distance_matrix()` / `set_distance_matrix()`
so the read and write names match MATLAB's row 29/29a and the C++ dense accessor.
Both old Python names survive one cycle as deprecated aliases (§4). See §3 rows 29a
and 29b.

Read accessors (canonical, all languages): `size()`, `n_clusters()` (was
`cluster_size()`), `name()`, `series(i)`, `series_name(i)`, `labels()` (reads
`clusters_ind`), `medoids()` (reads `centroids_ind`), `centroid_of(i)`.
`clusters_ind`/`centroids_ind` stay as raw fields in C++ but the canonical
read path is `labels()`/`medoids()` for cross-language parity with `Result`.

**Remaining live C++ `Problem` members — explicit fate (so Phase 2 has a complete
surface).** These live symbols are neither renamed above nor bound in Python/MATLAB
today; 2.0 fixes their status as follows:

| Live C++ member | Source | 2.0 fate |
|---|---|---|
| `set_clusters(std::vector<int>&)` | Problem.hpp:186 | **stays C++-only**, canonical `set_clusters` (already snake_case); seeds candidate medoids. Not bound (internal seeding hook). |
| `cluster_and_process()` | Problem.hpp:266 | **stays C++-only** convenience (cluster + write outputs). The Tier-1 `cluster()` free function (§1.3) is its cross-language successor; not bound. |
| `resize()` | Problem.hpp:179 | **becomes private** in Task 1.6 (internal invariant maintenance; called by `set_view_data`). Not a public entry point. |
| `init()` | Problem.hpp:259 | **stays C++-only**, canonical `init` (runs `init_fun`); not bound. |
| `last_iterations` (field) | Problem.hpp:132 | **becomes private with a read accessor** `last_iterations()` in Task 1.6 (diagnostic count written by clustering). Read-only; no setter. |
| `init_fun` (`std::function`) | Problem.hpp:143 | **stays a public C++ field** (the initialisation-strategy hook `init::random` by default); C++-only, not bound (no cross-language callable-injection contract in 2.0). Task 1.6 may wrap it behind `set_init_strategy(...)` — recorded as §10 item 7. |

### 2.3 `DataLoader` (C++ Tier-2 only) `[rename: camelCase → snake_case]`

CSV/TSV builder. Bindings do **not** expose `DataLoader` — Tier-1 `load()`
covers the binding use case; the multi-format (Parquet/Arrow/.dtws) loading in
the CLI (`dtwc_cl.cpp:471-555`) is the other path. Chained setters return
`DataLoader&`.

| C++ live (DataLoader.hpp) | C++ 2.0 canonical |
|---|---|
| `startColumn(int)` (:46-110) | `start_column(int)` |
| `startRow(int)` | `start_row(int)` |
| `n_data(int)` | `n_data(int)` (unchanged) |
| `delimiter(char)` | `delimiter(char)` (unchanged) |
| `path(fs::path)` | `path(fs::path)` (unchanged) |
| `verbosity(int)` | `verbosity(int)` (unchanged) |
| `load() -> Data` / `count()` | `load()` / `count()` (unchanged) |

### 2.4 `scores::*` free functions `[rename: camelCase → snake_case]`

Canonical scheme drops the redundant `Index`/`Information` noun (the fixed
decision `daviesBouldinIndex → davies_bouldin` sets the pattern; applied
uniformly). Same name in all three languages.

| Concept | C++ live (scores.hpp) | C++ 2.0 canonical | Python 2.0 | MATLAB 2.0 |
|---|---|---|---|---|
| silhouette | `silhouette(prob)` (:22) | `silhouette(prob)` | `silhouette(prob)` (live) | `silhouette(prob)` (live) |
| Davies–Bouldin | `daviesBouldinIndex(prob)` (:23) | **`davies_bouldin(prob)`** *(fixed)* | `davies_bouldin(prob)` | `davies_bouldin(prob)` |
| Dunn | `dunnIndex(prob)` (:25) | `dunn(prob)` | `dunn(prob)` | `dunn(prob)` |
| inertia | `inertia(prob)` (:26) | `inertia(prob)` | `inertia(prob)` (live) | `inertia(prob)` (live) |
| Calinski–Harabasz | `calinskiHarabaszIndex(prob)` (:27) | `calinski_harabasz(prob)` | `calinski_harabasz(prob)` | `calinski_harabasz(prob)` |
| Adjusted Rand | `adjustedRandIndex(l1,l2)` (:29) | `adjusted_rand(l1,l2)` ‡ | `adjusted_rand(l1,l2)` | `adjusted_rand(l1,l2)` |
| Normalized MI | `normalizedMutualInformation(l1,l2)` (:31) | `normalized_mutual_info(l1,l2)` ‡ | `normalized_mutual_info(l1,l2)` | `normalized_mutual_info(l1,l2)` |

Deprecated aliases retained one cycle (§4): Python `davies_bouldin_index`,
`dunn_index`, `calinski_harabasz_index`, `adjusted_rand_index`,
`normalized_mutual_information` (`_dtwcpp_core.cpp:671-709`); MATLAB the same
(`dtwc_mex.cpp:792-874`). ‡ **OPEN for adversarial review:** the exact final
spellings of Adjusted-Rand and Normalized-MI (`adjusted_rand` vs
`adjusted_rand_index`; `normalized_mutual_info` vs `normalized_mutual_information`)
are the two least-settled names — `davies_bouldin` is fixed by decision, the
rest follow its pattern. Reviewer resolves before FROZEN.

### 2.5 Algorithm free functions (Tier-2, all languages)

| Function | C++ | Python (`_dtwcpp_core.cpp`) | MATLAB (`+dtwc/`) |
|---|---|---|---|
| FastPAM | `fast_pam(Problem&, int k, int max_iter=100)` | `fast_pam(prob, n_clusters, max_iter=100)` (:567) | `fast_pam(prob, k, 'max_iter',100)` |
| FastCLARA | `algorithms::fast_clara(Problem&, CLARAOptions)` | `fast_clara(prob, n_clusters, sample_size=-1, n_samples=5, max_iter=100, seed=42)` (:602) | `fast_clara(prob, k, ...)` |
| CLARANS | `algorithms::clarans(Problem&, CLARANSOptions)` | `clarans(prob, opts)` (:738) | `clarans(prob, k, ...)` |
| dendrogram build | `algorithms::build_dendrogram(Problem&, HierarchicalOptions)` | `build_dendrogram(prob, opts=HierarchicalOptions())` (:715) | `build_dendrogram(prob, ...)` |
| dendrogram cut | `algorithms::cut_dendrogram(Dendrogram, Problem&, int k)` | `cut_dendrogram(dend, prob, k)` (:725) | `cut_dendrogram(dend, prob, k)` |

**Result write-back (fixed decision).** In 2.0, `fast_pam`/`fast_clara`/`clarans`
write `labels`/`medoids`/`k` back into `Problem` **in C++** (Task 1.6). The
binding-side auto-wire that does this today (Python `_dtwcpp_core.cpp:573-576,
615-619, 744-747`; MATLAB `store_result_in_problem`, `dtwc_mex.cpp:222-227`) is
then deleted in Phase 2 — behaviour is unchanged, ownership moves to core.
`cut_dendrogram` also writes back in 2.0 (today it does not — `_dtwcpp_core.cpp:732`).

**Reserved:** `Method::LRCore` (Phase 4 exact solver; name registered here so
Phase 4 lands against it). Until Phase 4, the Tier-1 `method="mip"` maps to
solver-backed exact (`Method::MIP`).

### 2.6 Distance free functions (Tier-2, all languages)

Canonical namespace is `dtwc::distance::*` (`distance.hpp:29`), mirrored by
Python `dtwcpp.distance.*` (`distance.py`) and MATLAB `+dtwc/+distance/`.

| Variant | C++ `dtwc::distance::` | Python `dtwcpp.distance.` | MATLAB `dtwc.distance.` |
|---|---|---|---|
| standard | `dtw(x,y,band=-1,metric=L1)` | `standard(x,y,band=-1,metric='l1')` | `standard(x,y,'Band',-1,'Metric','l1')` |
| derivative | `ddtw(x,y,band=-1,metric=L1)` | `ddtw(x,y,band=-1)` | `ddtw(...)` |
| weighted | `wdtw(x,y,band=-1,g=0.05)` | `wdtw(x,y,band=-1,g=0.05)` | `wdtw(...)` |
| amerced | `adtw(x,y,band=-1,penalty=1.0)` | `adtw(x,y,band=-1,penalty=1.0)` | `adtw(...)` |
| soft | `soft_dtw(x,y,gamma=1.0)` | `soft_dtw(x,y,gamma=1.0)` | `soft_dtw(...)` |
| missing | `missing(x,y,band=-1,metric=L1)` | `missing(...)` | `missing(...)` |
| AROW | `arow(x,y,band=-1,metric=L1)` | `arow(...)` | `arow(...)` |
| dispatcher | `dtw(x,y,DTWVariantParams,band=-1,metric=L1,missing_strategy=Error)` | `dtw(x,y,*,variant='standard',band=-1,metric='l1',g=,penalty=,gamma=,missing_strategy='error')` | `dtw(x,y,'Variant',...,'Band',...,...)` |

**Naming-law carve-out for `dtw` (explicit exception to §0).** The token `dtw`
is **overloaded by design** and this is the one sanctioned break from "one name
per concept":

- In **C++**, `dtwc::distance::dtw` names *both* the standard single-pair function
  (`distance.hpp:31`, no `DTWVariantParams` arg) *and* the variant dispatcher
  (`distance.hpp:87`, with `DTWVariantParams`). The two are C++ overloads resolved
  by argument list, so there is no `dtwc::distance::standard`.
- In **Python/MATLAB**, the standard single-pair call is named `standard(x,y,…)`
  (there is no argument overloading), and `dtw(x,y,variant=…)` is the dispatcher
  only (`distance.py`).

Net: `dtw` = "standard DTW" in C++ but "dispatcher" in Python/MATLAB. This is
accepted rather than unified because C++ overloading and the Python/MATLAB
keyword-dispatch idiom cannot share one signature; unifying would force an
un-idiomatic name on one side. Recorded as an open item (§10 item 8) for the
reviewer to ratify before FROZEN.

**Precision-default fix.** All `dtwc::distance::*` templates default
`T = settings::default_data_t`, which is `float` today (`distance.hpp:31`,
`settings.hpp:29`) and becomes `double` in Task 1.5. See §8. Python/MATLAB are
always `double`. **Arg-type parity (Phase 2.1):** Python distance fns must all
take zero-copy ndarray uniformly — today `dtw/missing/arow` take ndarray but
`ddtw/wdtw/adtw/soft` take `std::vector` (copy) (`_dtwcpp_core.cpp:291-334`).

### 2.7 Checkpoint / resume (Tier-2, implements invariant #4)

The checkpoint/resume surface backing preserved invariant #4 (§7 item 4). These
symbols exist today and are frozen here so Phase 2.1 has them to implement against;
names are already snake_case, so no rename — only binding-parity gaps are marked.

| Concept | C++ live (`checkpoint.hpp`) | Python | MATLAB 2.0 |
|---|---|---|---|
| options struct | `CheckpointOptions` {`directory`,`save_interval`,`enabled`} (:31) | `CheckpointOptions` (live, `_dtwcpp_core.cpp:640`) | `CheckpointOptions` struct `[new]` |
| save dir checkpoint | `save_checkpoint(const Problem&, const std::string& path)` (:46) | `save_checkpoint(prob, path)` (live, `_dtwcpp_core.cpp:651`) | `save_checkpoint(prob, path)` `[new]` |
| load dir checkpoint | `load_checkpoint(Problem&, const std::string& path) -> bool` (:57) | `load_checkpoint(prob, path)` (live, `_dtwcpp_core.cpp:656`) | `load_checkpoint(prob, path)` `[new]` |
| save binary result | `save_binary_checkpoint(const core::ClusteringResult&, ...)` (:79) | — `[new bind]` | — `[new]` |
| load binary result | `load_binary_checkpoint(core::ClusteringResult&, ...) -> bool` (:90) | — `[new bind]` | — `[new]` |

*Contract:* directory checkpoint = `distances.csv` + `metadata.txt`; binary result
checkpoint = the `<name>_checkpoint.bin` of §7 item 2; the mmap distance-matrix
cache (`<name>_distmat.cache`) is the third leg of the invariant-#4 triple and is
driven via `use_mmap_distance_matrix(path)` (§2.2) + CLI `--resume`. MATLAB
checkpointing (`TODO.md:105`, "MATLAB Phase 2: checkpointing") is `[new]` and
lands in Phase 2.2 against this table.

**Persistent mmap identity (2.0 safety addendum).** The mmap cache uses the
64-byte version-2 header. Its SHA-256 identity covers the raw IEEE series values,
series order and lengths, storage precision, `ndim`, band, every DTW-variant
parameter, multivariate mode, missing-data strategy, pointwise metric, compute
backend, and backend precision. Series names are excluded because they do not
affect distance semantics. Header metadata has a CRC, reserved bytes are checked,
and the file length must match the packed matrix exactly. A mismatch is a hard
error before any cached value is exposed; callers must use the original semantics
or delete/rename the cache and recompute it.

Version-1 mmap caches are deliberately rejected because their N-only identity
cannot prove safe reuse. Semantic setters detach a bound cache without deleting
it. The complete data identity is checked at bind and once at first use; later
lookups compare a fixed-size configuration snapshot so warm access remains O(1).
Consequently raw in-place `Data` mutation after first use is unsupported: call
`refresh_distance_matrix()` before the edit, or replace the data through
`set_data()`. CUDA mmap caches require explicit FP32 or FP64 (not hardware-
dependent `Auto`), and non-L1 identities are external/GPU-fill-only because the
CPU lazy path computes L1.

---

## 3. Full 1.x → 2.0 rename table

Every current public camelCase / duplicate / divergent symbol found by reading
the headers and both bindings. Column **Shim** = how the old name survives:
`[[deprecated]]` C++ inline shim, or `alias` (Python/MATLAB deprecated alias),
or `removed` (dropped from bindings — 2.0 is the break point, surface report §7).

| # | Concept | 1.x name(s) | 2.0 canonical | Shim |
|---|---|---|---|---|
| 1 | set k (C++) | `Problem::set_numberOfClusters` (Problem.hpp:185) | `set_n_clusters` | C++ `[[deprecated]]` |
| 2 | set k (Python) | `Problem.set_number_of_clusters` (`_dtwcpp_core.cpp:445`) | `set_n_clusters` | removed (alias 1 cycle) |
| 3 | set k (MATLAB) | `Problem.set_n_clusters` (Problem.m:113) | `set_n_clusters` | already canonical |
| 4 | max iterations (C++ field) | `Problem::maxIter` (Problem.hpp:130) | `set_max_iter` / `max_iter` accessor | C++ `[[deprecated]]` field-name kept |
| 5 | max iterations (MATLAB prop) | `Problem.MaxIter` (Problem.m:27) | `set_max_iter` | alias (loud warn) |
| 6 | repetitions (C++ field) | `Problem::N_repetition` (Problem.hpp:131) | `set_n_repetitions` / `n_repetitions` | C++ `[[deprecated]]` |
| 7 | repetitions (Python prop) | `Problem.n_repetition` (`_dtwcpp_core.cpp:424`) | `n_repetitions` | alias 1 cycle |
| 8 | repetitions (MATLAB prop) | `Problem.NRepetition` (Problem.m:28) | `set_n_repetitions` | alias (loud warn) |
| 9 | band (MATLAB prop) | `Problem.Band` (Problem.m:25) | `set_band` | alias (loud warn) |
| 10 | verbose (MATLAB prop) | `Problem.Verbose` (Problem.m:26) | `set_verbose` | alias (loud warn) |
| 11 | refresh dist mat | `refreshDistanceMatrix` (Problem.hpp:178) | `refresh_distance_matrix` | C++ `[[deprecated]]` |
| 12 | read dist mat | `readDistanceMatrix` (Problem.hpp:184) | `read_distance_matrix` | C++ `[[deprecated]]` |
| 13 | max distance | `maxDistance` (Problem.hpp:209) | `max_distance` | C++ `[[deprecated]]` |
| 14 | dist by index | `distByInd` (Problem.hpp:210) | `dist_by_ind` | C++ `[[deprecated]]` |
| 15 | is filled | `isDistanceMatrixFilled` (Problem.hpp:225) | `is_distance_matrix_filled` | C++ `[[deprecated]]` |
| 16 | fill dist mat | `fillDistanceMatrix` (Problem.hpp:246) | `fill_distance_matrix` | C++ `[[deprecated]]` |
| 17 | print dist mat | `printDistanceMatrix` (Problem.hpp:247) | `print_distance_matrix` | C++ `[[deprecated]]` |
| 18 | write dist mat | `writeDistanceMatrix` (Problem.hpp:249) | `write_distance_matrix` | C++ `[[deprecated]]` |
| 19 | print clusters | `printClusters` (Problem.hpp:252) | `print_clusters` | C++ `[[deprecated]]` |
| 20 | write clusters | `writeClusters` (Problem.hpp:253) | `write_clusters` | C++ `[[deprecated]]` |
| 21 | write medoid members | `writeMedoidMembers` (Problem.hpp:255) | `write_medoid_members` | C++ `[[deprecated]]` |
| 22 | write silhouettes | `writeSilhouettes` (Problem.hpp:256) | `write_silhouettes` | C++ `[[deprecated]]` |
| 23 | total cost | `findTotalCost` (Problem.hpp:269) | `find_total_cost` | C++ `[[deprecated]]` |
| 24 | assign clusters | `assignClusters` (Problem.hpp:270) | `assign_clusters` | C++ `[[deprecated]]` |
| 25 | calc medoids | `calculateMedoids` (Problem.hpp:272) | `calculate_medoids` | C++ `[[deprecated]]` |
| 26 | cluster via MIP | `cluster_by_MIP` (Problem.hpp:263) | `cluster_by_mip` | C++ `[[deprecated]]` |
| 27 | cluster via Lloyd | `cluster_by_kMedoidsLloyd` (Problem.hpp:264) | `cluster_by_kmedoids_lloyd` | C++ `[[deprecated]]` |
| 28 | n clusters read | `cluster_size` (Problem.hpp:162) | `n_clusters` | C++ `[[deprecated]]` alias |
| 29 | dist mat access (MATLAB) | `get_distance_matrix`/`set_distance_matrix` (Problem.m:158,164) | `distance_matrix`/`set_distance_matrix` | alias (loud warn) |
| 29a | dist mat read (Python) | `Problem.distance_matrix_numpy` (`_dtwcpp_core.cpp:457`) | `distance_matrix` | alias 1 cycle |
| 29b | dist mat write (Python) | `Problem.set_distance_matrix_from_numpy` (`_dtwcpp_core.cpp:492`, used `_api.py:224`) | `set_distance_matrix` | alias 1 cycle |
| 29c | size read (MATLAB) | `Problem.Size` (dependent prop, Problem.m:32; getter :177) | `size()` | alias (loud warn) |
| 29d | n clusters read (MATLAB) | `Problem.ClusterSize` (dependent prop, Problem.m:33; getter :181) | `n_clusters()` | alias (loud warn) |
| 29e | name read (MATLAB) | `Problem.Name` (dependent prop, Problem.m:34; getter :185) | `name()` | alias (loud warn) |
| 29f | medoids read (MATLAB) | `Problem.CentroidsInd` (dependent prop, Problem.m:35; getter :189) | `medoids()` | alias (loud warn) |
| 29g | labels read (MATLAB) | `Problem.ClustersInd` (dependent prop, Problem.m:36; getter :193) | `labels()` | alias (loud warn) |
| 30 | Davies–Bouldin | `scores::daviesBouldinIndex` (scores.hpp:23) | `scores::davies_bouldin` **(fixed)** | C++ `[[deprecated]]`; Py/MEX alias |
| 31 | Dunn | `scores::dunnIndex` (scores.hpp:25) | `scores::dunn` | C++ `[[deprecated]]`; Py/MEX alias |
| 32 | Calinski–Harabasz | `scores::calinskiHarabaszIndex` (scores.hpp:27) | `scores::calinski_harabasz` | C++ `[[deprecated]]`; Py/MEX alias |
| 33 | Adjusted Rand | `scores::adjustedRandIndex` (scores.hpp:29) | `scores::adjusted_rand` ‡ | C++ `[[deprecated]]`; Py/MEX alias |
| 34 | Normalized MI | `scores::normalizedMutualInformation` (scores.hpp:31) | `scores::normalized_mutual_info` ‡ | C++ `[[deprecated]]`; Py/MEX alias |
| 35 | start column (loader) | `DataLoader::startColumn` (DataLoader.hpp) | `start_column` | C++ `[[deprecated]]` |
| 36 | start row (loader) | `DataLoader::startRow` | `start_row` | C++ `[[deprecated]]` |
| 37 | set data path | `settings::paths::setDataPath` (settings.hpp:71) | `set_data_path` | C++ `[[deprecated]]` |
| 38 | set results path | `settings::paths::setResultsPath` (settings.hpp:79) | `set_results_path` | C++ `[[deprecated]]` |
| 39 | Result class (Python) | `ClusterResult` (`_api.py:73`) | `Result` | alias 1 cycle |
| 40 | medoids field (Result) | `ClusterResult.medoid_indices` (`_api.py:83`) | `Result.medoids` | alias 1 cycle |
| 41 | default template scalar | `settings::default_data_t = float` (settings.hpp:29) | `= double` | behaviour change (§8), no name change |
| 42 | CLI dtype default | `--dtype float32` (dtwc_cl.cpp:226) | `--dtype float64` | old accepted, default flips (§8) |

**Mmap-cache migration.** The unsafe version-1 `<name>_distmat.cache` format is
not resumed by 2.0. Delete or rename that cache and rerun to create a fingerprinted
version-2 cache; source data and result checkpoints are unaffected. At the CLI
mmap threshold, legacy dense `--checkpoint` and `--dist-matrix` inputs cannot be
combined with the mmap cache and fail before either storage path is opened. Omit
the dense option to use automatic mmap resume, or raise the threshold only when
the dense matrix and CSV checkpoint fit in memory.

**Duplicate-elimination principle (surface report §7).** Where the same concept
had three different names (surface report inconsistency table rows 1, 5, 12), 2.0
collapses to one canonical and the bindings expose **only** that name. C++ keeps
`[[deprecated]]` shims for source compatibility; Python/MATLAB keep a
one-release deprecated alias, then removal.

---

## 4. Deprecation & removal policy

**2.0 is the break point** (surface report §7: "2.0 removes all duplicates from
bindings").

- **C++.** Every renamed method/function keeps a `[[deprecated("use <new>")]]`
  inline shim forwarding to the canonical implementation. Shims compile-warn,
  never change behaviour, and are scheduled for removal in 3.0. Renamed *fields*
  (`maxIter`, `N_repetition`) keep the old identifier as a `[[deprecated]]`
  reference/accessor where a field can't carry the attribute cleanly; the
  invariant-preserving setter is canonical.
- **Python.** Removed duplicate names (`set_number_of_clusters`,
  `n_repetition`, the `*_index`/`*_information` score aliases, `ClusterResult`,
  `medoid_indices`) survive **one** minor release as thin aliases that emit
  `DeprecationWarning` on use, then are deleted. New canonical names are the
  only ones documented.
- **MATLAB.** PascalCase settable properties (`Band`, `MaxIter`, `NRepetition`,
  `Verbose`) and `get_/set_distance_matrix` keep a deprecated shim that prints a
  one-line loud notice and forwards. The 1-based boundary conversion is
  untouched.
- **CLI.** Old flag spellings are accepted with a deprecation warning; the SLURM
  callers (`cluster_generic.slurm`, `_hpc.build_dtwc_command`) are updated in the
  same commit that renames a flag (Phase 2.3) — the CLI flag set is a de-facto
  API (§7 item 3).
- **Nothing silently disappears.** A removed binding name that a user calls must
  raise `AttributeError`/`Unknown command` — never resolve to a different
  behaviour.

---

## 5. Error taxonomy

**Fixed decision.** Base `dtwc::Error` (subclass of `std::runtime_error`) plus
four leaf types. No `assert`-as-validation, no `exit()` in library code
(enforced in Task 1.2, `dtwc/error.hpp`). Bindings translate to native
exceptions / `mexErrMsgIdAndTxt`.

| C++ type | Covers | Python class | MATLAB identifier |
|---|---|---|---|
| `dtwc::Error` (base) | anything DTWC-thrown not more specific | `dtwcpp.DtwcError(Exception)` | `dtwc:error` |
| `dtwc::InvalidInput` | bad argument: wrong shape/dtype/range, unknown method/metric/variant name, empty data, `ndim` mismatch, unknown `score()` name | `dtwcpp.InvalidInput(DtwcError, ValueError)` | **`dtwc:invalidArgument`** |
| `dtwc::SolverError` | MIP/LP solver failure: infeasible, iteration/time limit hit without optimum, solver returned non-optimal status (Task 0.5 migrates onto this) | `dtwcpp.SolverError(DtwcError, RuntimeError)` | `dtwc:solverError` |
| `dtwc::DeviceError` | device/backend problem: unknown device name, `gpu` on non-GPU build, `.env`/HPC credential failures (§6) | `dtwcpp.DeviceError(DtwcError, RuntimeError)` | `dtwc:deviceError` |
| `dtwc::IOError` | file/format failure: file not found, unreadable, bad Parquet/Arrow type, OOB offsets, checkpoint mismatch | `dtwcpp.IOError(DtwcError, OSError)` | `dtwc:ioError` |

**Binding-translation rules.**

- **Python (nanobind).** Register one exception translator per type. Each leaf
  subclasses both `DtwcError` and the closest built-in (`ValueError`/`OSError`/
  `RuntimeError`) so idiomatic `except ValueError:` and `except dtwcpp.InvalidInput:`
  both work. `dtwc::Error` maps to `DtwcError`.
- **MATLAB (MEX).** `mexFunction`'s catch ladder maps types to identifiers.
  `dtwc::InvalidInput` **must** map to `dtwc:invalidArgument` — this identifier
  is **already pinned verbatim** by `tests/matlab/test_mex_input_validation.m`
  (16 assertions at lines 63–178) and must not change. The current ladder
  (`dtwc_mex.cpp:1039-1049`) maps `std::invalid_argument → dtwc:invalidArgument`,
  `std::out_of_range → dtwc:outOfRange`, `std::runtime_error → dtwc:runtime`; in
  2.0 it maps the DTWC leaf types first (`InvalidInput → dtwc:invalidArgument`,
  `SolverError → dtwc:solverError`, `DeviceError → dtwc:deviceError`,
  `IOError → dtwc:ioError`), keeping the std fallbacks below them so
  `dtwc:invalidArgument` continues to fire for the pinned input-validation cases.

---

## 6. Device / `Env` semantics

**Fixed decision.** `dtwc::Env` (Task 1.3) owns device (`cpu`/`gpu`/`hpc`),
precision, and thread policy; `device()` in every language delegates to it.
Singleton accessor `dtwc::env()`. **No silent fallback anywhere.**

### 6.1 Device names

Canonical: `cpu`, `gpu`, `hpc`. Aliases accepted: `gpu:N`, `cuda`, `cuda:N`
(GPU device index N; `gpu` ≡ `cuda` on NVIDIA, ≡ Metal on macOS). Case-insensitive.

- **Unknown device name → `DeviceError`** listing valid names, verbatim:
  ```
  [dtwc] unknown device 'foo'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N), hpc.
  ```
- **`gpu` on a build with no GPU backend → `DeviceError`** naming the build
  flag (NOT silent CPU fallback), verbatim:
  ```
  [dtwc] device='gpu' requested but this build has no GPU backend compiled in.
  Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.
  This build will not silently fall back to CPU.
  ```

**Behaviour change (implemented).** Python 1.x warned and fell back to CPU when
CUDA was requested but absent. In 2.0 this is a hard `DeviceError` in every
front end. The user selects `cpu` explicitly if that is what they want.

### 6.2 `device="hpc"` — `.env` credential contract

`hpc` reads SLURM credentials from a `.env` file at the repository root. Required
keys (the names the live SLURM path already uses — `_hpc.py:122-126`,
`scripts/slurm/env.example`): **`SLURM_HOST`**, **`SLURM_USER`**,
**`SLURM_REMOTE_BASE`**. Each of the three failure modes produces a specific,
actionable `DeviceError` (Python `DeviceError`, MATLAB `dtwc:deviceError`), never
a stack trace and never a silent local fallback.

**Task 1.3 asserts these three messages VERBATIM.** They are authored here as the
exact C++ `DeviceError::what()` strings; Python/MATLAB reproduce them byte-for-byte.
The recommended test fixture uses `SLURM_HOST=arc-login.arc.ox.ac.uk`,
`SLURM_USER=abcd1234`.

**(1) No `.env` file** — host-independent, fully verbatim:
```
[dtwc] device='hpc' requires a .env file at the repository root, but none was found.
Copy scripts/slurm/env.example to .env and set SLURM_HOST, SLURM_USER, and SLURM_REMOTE_BASE.
Example .env:
  SLURM_HOST=arc-login.arc.ox.ac.uk
  SLURM_USER=abcd1234
  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs
```

**(2) `.env` present but a required key is missing** — the `{key}` slot is the
first missing key of `SLURM_HOST`/`SLURM_USER`/`SLURM_REMOTE_BASE`; the test pins
the `SLURM_HOST` case, so the fully-substituted verbatim string is:
```
[dtwc] device='hpc': the .env file is missing required key 'SLURM_HOST'.
Set it in .env at the repository root. Example .env:
  SLURM_HOST=arc-login.arc.ox.ac.uk
  SLURM_USER=abcd1234
  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs
```

**(3) Host authentication failure** — names the host and user it tried; with the
fixture values the fully-substituted verbatim string is:
```
[dtwc] device='hpc': could not authenticate to SLURM host 'arc-login.arc.ox.ac.uk' as user 'abcd1234'.
Check that your SSH key is authorized on that host (ssh abcd1234@arc-login.arc.ox.ac.uk must succeed without a password prompt) and that SLURM_HOST and SLURM_USER in .env are correct.
```

Message-template rule for (2) and (3) (so the implementation and the test agree
on substitution): `{key}` = the missing key name; `{host}` = value of
`SLURM_HOST`; `{user}` = value of `SLURM_USER`. Messages (1) and (2)'s example
block are constant text.

### 6.3 Lazy load & big-data policy (fixed decision, contract-level)

- `device="hpc"`: **metadata-only** local load — shapes/counts/names read
  locally; bulk series streamed to the cluster at submit (`load()` never reads
  the payload; `cluster_on_hpc` forwards a path, `_hpc.py:185-193`).
- Local library devices (`cpu`/`gpu`): `core::StoragePolicy::Auto` — mmap-backed
  store when estimated footprint exceeds a threshold (default 50% free RAM,
  overridable via `set_storage_policy`). The CLI controls parent distance
  storage separately with `--mmap-threshold`. Its `--ram-limit` is a conservative
  cap on Parquet selected-series decoding/materialisation, applied before payload
  I/O; it is not a whole-process RSS limit. Only a single list-per-row file can
  exceed that cap and continue, through non-full CPU FastCLARA row-group
  streaming. View-mode spans (48× CLARA subsample win, surface report §6 wart 6
  / §8 item 6) are preserved.

---

## 7. Preserved invariants (any 2.0 change must not break these)

The five load-bearing constraints from the API-surface report §8, plus the
determinism/index rules, restated as a checklist for the adversarial reviewer:

1. **File formats read.** CSV/TSV (start_row/start_col/delimiter/Ndata,
   folder-of-files), Parquet file+dir with optional `--column`, Arrow IPC
   (`.arrow/.ipc/.feather`), `.dtws` mmap + `.names` sidecar
   (`dtwc_cl.cpp:471-555`), Python Polars `large_list<float>` ragged ingest.
2. **Output contract (bit-identical).** Two distinct sets, and the equal-bytes
   guarantee applies to the **first set only**:
   - *Human-readable results (2 unconditional + 2 matrix-dependent files) —
     the equal-bytes contract.* `<name>_labels.csv` (`"name,cluster"`) and
     `<name>_medoids.csv` (`"cluster,medoid_index,medoid_name"`) are always
     emitted. `<name>_distance_matrix.csv` and `<name>_silhouettes.csv` are CLI
     outputs only when a full distance matrix is materialised. `Result::save(dir)`
     (§1.4) still requests and emits all four; every corresponding CLI file must
     be byte-for-byte identical. Matrix-free CLI routes, including RAM-limited
     Parquet FastCLARA, do not create O(N²) state solely for the latter two.
     Streamed list rows retain the eager names `series_0`, `series_1`, and so on.
     The SLURM path machine-parses `<name>_labels.csv` and maps 1-based
     lexically-sorted rows back to input order (`_hpc.py:45-65`).
   - *Run-time persistence artifacts (up to 2 files) — NOT part of the save()
     equal-bytes set.* `<name>_checkpoint.bin` is written by every successful CLI
     run, including streamed FastCLARA; `<name>_distmat.cache` is written when
     mapped distance storage is selected. They are produced **during the run**
     for resume (invariant 4), **not** by `Result::save(dir)`. Their format is
     preserved for `--resume` compatibility,
     but they are explicitly outside the `save()`↔CLI byte-identity claim. The
     mmap cache's safety-mandated v1→v2 invalidation is the authorized exception:
     v1 caches must be recomputed because they cannot identify their data/config.
3. **CLI flag set + TOML/YAML keys** (kebab-case) are a de-facto API:
   `cluster_generic.slurm` and `_hpc.build_dtwc_command` (`_hpc.py:68-84`)
   compose `dtwc_cl` command lines. Renames go through the accept-old-name
   deprecation path (§4) with those two callers updated in the same commit.
4. **Checkpoint/resume triple.** Directory checkpoint (distances.csv +
   metadata.txt), binary result checkpoint, mmap distance-matrix cache with
   `--resume` — long SLURM runs depend on all three. A non-full FastCLARA run
   has no parent distance matrix and therefore rejects the directory checkpoint
   and imported dense matrix paths; its automatic binary result checkpoint is
   still written. The full-sample PAM fallback retains the ordinary triple.
5. **Precision contract.** Distance matrix and returned distances are always
   `double`, even with `float32` series storage (§8).
6. **Zero-copy / perf paths.** nanobind ndarray zero-copy + GIL release on every
   long call; Data view-mode spans; interleaved multivariate layout
   (`[t0f0,t0f1,t1f0,…]`); lock-free row-partitioned matrix fill
   (`_dtwcpp_core.cpp:536-550`).
7. **Determinism.** Seed-aware Tier-1 PAM/OneBatchPAM/CLARA entry points use the
   invocation-local cross-language default 42 (§1.3); estimator restart `i` uses
   `42+i`. The unseeded Tier-2 FastPAM overload retains the legacy mutable
   `std::mt19937 randGenerator(29)`, and CLARANS retains its explicit option
   default 42. Scores read state from `Problem`, so result write-back (now in
   C++, §2.5) must run before any `score()`.
8. **MATLAB 1-based conversion at the MEX boundary only**; rectangular N×L
   matrix input still accepted. Optional deps (OpenMP/HiGHS/CUDA/Metal/Arrow)
   stay optional — core builds without them.

---

## 8. Precision story (existing contract, now documented)

**`data_t = double` is THE default everywhere.** (`settings.hpp:35`,
`using data_t = double`.)

- **Template default flip (Task 1.5).** `settings::default_data_t` is `float`
  today (`settings.hpp:29`) and is the default `T` on every
  `dtwc::distance::*` helper (`distance.hpp:31`). 2.0 changes it to `double`, so
  `dtwc::distance::dtw(x, y)` with no explicit `T` computes in `double`. This is
  a *default* change, not a numeric one — a `double`-typed call is byte-identical
  before and after (Task 1.5 registers a pre/post f64 digit-identity check).
- **CLI default flip (Task 1.5).** `--dtype` defaults to `float32` today
  (`dtwc_cl.cpp:226`); 2.0 defaults to `float64`. `float32`/`f32`/`fp32` remain
  accepted opt-ins (the `CheckedTransformer` map at `dtwc_cl.cpp:229-234` stays).
- **`float32` is an explicit opt-in only.** C++: build `Data` from
  `vector<vector<float>>` (`Data.hpp:113`) or set `Precision::Float32`; CLI:
  `--dtype f32`. Storage halves; nothing else changes semantically.
- **Accumulation is ALWAYS `double`.** Regardless of series storage precision,
  every DTW cost is accumulated and every distance-matrix entry stored in
  `double` — the f32 path reads `float` inputs but returns `double`
  (`Problem::dtw_fn_f32_t = std::function<double(...)>`, Problem.hpp:94;
  storage.hpp:19, surface report §8 item 5). Python/MATLAB are always `double`
  end to end.
- **Doc-bug fix (Task 1.5).** `storage.hpp:21` comment calls `Float32` the
  "Default"; it is not (`Data::precision = Float64`, `Data.hpp:38`). The comment
  is corrected in the same task.

---

## 9. Notes for Phase 2 / test authors (LESSONS compliance)

- **Tests must pin the LIVE code path.** Per `.claude/LESSONS.md`, a test name
  must match the algorithm path it actually exercises, and every parity test
  must drive a real public entry point — not a dead sibling. When Phase 2 /
  Task 1.3 add tests against this contract, each test states in a comment which
  public entry point it exercises (e.g. `// drives dtwc::cluster() Tier-1`,
  `// drives Env::set_device("hpc") .env-missing path`). The `.env` verbatim
  assertions (§6.2) and the `dtwc:invalidArgument` pin (§5) are the two
  highest-value live-path anchors.
- **Parity fixture (Phase 2.4).** One recorded dataset → banded DTW → `fast_pam`
  k=3, fixed seed → labels + medoids + 3 scores, run from C++, Python, MATLAB,
  and CLI, asserting digit-identical labels/medoids and scores equal to 1e-12
  rel. That fixture is the permanent cross-language gate for this contract.

---

## 10. Open items for the adversarial reviewer (before FROZEN)

1. **Score names ‡** (§2.4): confirm `adjusted_rand` / `normalized_mutual_info`
   vs keeping `_index`/`_information`. `davies_bouldin` is fixed.
2. **MATLAB PascalCase→snake_case property break** (§0, §3 rows 5–10, 29,
   29c–29g): confirm the deliberate idiom break is acceptable vs keeping
   PascalCase properties as first-class (with snake_case setters/getters as the
   alignable twin). This now also covers the read-only dependent props
   `Size`/`ClusterSize`/`Name`/`CentroidsInd`/`ClustersInd` (§3 rows 29c–29g).
3. **`Result.score("silhouette")` aggregation** (§1.4): confirm returning the
   **mean** silhouette (scalar) is the right Tier-1 contract, with the per-point
   vector available via Tier-2 `scores::silhouette(prob)`.
4. **`.env` key names** (§6.2): confirm `SLURM_HOST`/`SLURM_USER`/
   `SLURM_REMOTE_BASE` (matching the live wrapper) rather than a new
   `DTWC_HPC_*` scheme — the verbatim messages depend on this choice.
5. **Python `IOError` name clash** (§5): confirm exposing `dtwcpp.IOError`
   (shadows the builtin alias of `OSError`) vs naming it `dtwcpp.DtwcIOError`.
6. **`distance_matrix()` cross-language collision** (§2.2 †): confirm the C++
   variant-storage accessor `Problem::distance_matrix()` (Problem.hpp:231/233,
   returns `std::variant<Dense,Mmap>`) keeps its name while Python/MATLAB
   `distance_matrix()` returns an NxN array — or rename the C++ accessor (candidate
   `distance_matrix_storage()`, Task 1.6) to remove the overload.
7. **`Problem::init_fun` hook** (§2.2, "Remaining live members"): confirm the
   `std::function<void(Problem&)>` initialisation hook stays a public C++-only
   field vs wrapping it behind `set_init_strategy(...)` with a bound enum of named
   strategies (no callable injection across the binding boundary).
8. **`dtw` overload carve-out** (§2.6): confirm the sanctioned exception where
   `dtw` = standard DTW in C++ (overload) but = dispatcher in Python/MATLAB, vs
   forcing a single spelling (`standard`/`dtw`) on all three languages.

---

*End of contract. On adversarial sign-off, the header line becomes
`STATUS: FROZEN` and Phase 2 implements every signature above verbatim.*
