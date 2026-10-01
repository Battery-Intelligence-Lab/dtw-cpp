STATUS: FROZEN 2026-07-07; implementation re-audited on 2026-07-29 — changes require a PLAN.md decision entry before editing

# DTWC++ 2.0 — API Contract (freeze artifact)

> **Task 1.1 deliverable.** This document is the frozen cross-language API
> surface required for 2.0. The 2026-07-23 source audit distinguishes shipped
> behavior from unfulfilled frozen promises: a gap stays a 2.0 obligation unless
> the approved addenda below explicitly defer it to 2.1.
>
> **Provenance tags.** `[live]` = the symbol exists now at the cited location;
> `[introduced-2.0]` = historical provenance, not future work; `[rename]` = the
> frozen canonical spelling; `[gap Fxx]` = a confirmed unfulfilled 2.0 promise
> owned by that R3 finding. Sources read for this contract:
> `dtwc/Problem.hpp`, `dtwc/scores.hpp`, `dtwc/settings.hpp`,
> `dtwc/Data.hpp`, `dtwc/distance.hpp`, `dtwc/dtwc_cl.cpp`,
> `python/src/_dtwcpp_core.cpp`, `python/dtwcpp/_api.py`,
> `python/dtwcpp/__init__.py`, `python/dtwcpp/_hpc.py`,
> `bindings/matlab/dtwc_mex.cpp`, `bindings/matlab/+dtwc/Problem.m`,
> and `bindings/matlab/+dtwc/DTWClustering.m`.
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
2. C++ continues to accept `device="hpc"` as a valid device name, but
   `dtwc::run` (`dtwc_cl` and Tier-1 `cluster()`) raises the documented
   `DeviceError`, without reading `.env`: a run computes where it starts, and
   the C++ API has no authenticated remote-transport implementation. Python
   (`dtwcpp.cluster(..., device="hpc")`) and `slurm_remote.sh submit-cluster`
   remain the tested SLURM transport. A local CPU fallback would violate the no-silent-fallback
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
every language. MATLAB has no `PascalCase` property aliases: it was not in
v1.0.0, so its `Problem` offers the snake_case methods only.

**Indexing.** C++ and Python are 0-based. MATLAB is 1-based; the 0↔1 conversion
happens *only* at the MEX boundary (`ivec_to_mx_1based` in `dtwc_mex.cpp`)
and is a preserved invariant (§7 item 8). Every index-valued field below
(`labels`, `medoids`, `dist_by_ind(i,j)`) obeys this.

**Counts and indices are 64-bit in every language.** C++ takes and returns `index_t`
(`std::int64_t`); Python returns labels and medoids as `np.int64` arrays and takes any
int that fits 64 bits; MATLAB returns them as 1-based doubles, which are exact
to 2^53, and reads a count only if it is a finite integer in the 64-bit range.
`max_iter`, `n_init`, `n_samples` and `band` stay `int`, and seeds are `uint64`.

**Precision.** The default scalar and all `Problem`/`Result`/CLI
distance-matrix outputs are `double`. Explicit C++ `distance::*<float>` helpers
return `float`; Float32 `Problem` storage and recurrence widen only the final
distance to `double`. `data_t = double` is the storage default; Float32 is an
explicit opt-in. Full precision story in §8.

**Two tiers.** *Tier 1* is the high-level `device → load → cluster → Result`
flow (§1). *Tier 2* is the advanced object surface — `Problem`, `DataLoader`,
`scores::*`, and algorithm free functions (§2). Tier 1 is common to all three
languages. Tier 2 is shared only where its tables list a binding; `DataLoader`
and several extension hooks remain C++-only.

**No silent fallback.** Any requested capability that cannot be delivered
(device, GPU backend, method, metric) raises a typed error (§5) — it never
quietly degrades. This is the frozen rule. F24 (Python HPC exception
translation) names the one remaining violation rather than weakening it. F18
(MATLAB estimator routing) and F40 (the functional MATLAB `cluster` device
route) are closed: `dtwc.cluster` is one gateway call into C++ `dtwc::cluster`,
and `dtwc.DTWClustering.fit` resolves `Metric` and `Device` before any effect.

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
| Get | `std::string dtwc::device()` | `dtwcpp.device() -> str` (`__init__.py`) | `name = dtwc.device()` |
| Accepts | `"cpu"`,`"gpu"`,`"gpu:N"`,`"cuda"`,`"cuda:N"` | same, plus `"hpc"` and `"hpc:gpu"`; local names are parsed by the C++ grammar through its binding (`_dtwcpp_core.parse_device`) | same as C++ |
| Returns | canonical name from `to_string(Device)` (+`:N` for a non-zero GPU ordinal) | same: `device()` returns what `dtwc::device()` returns, so `"cuda:0"` comes back as `"gpu"` | same |
| Errors | `DeviceError` on an unknown name, and on `"hpc"`, naming Python's `dtwcpp.device("hpc")` and `slurm_remote.sh` (§6) | `DeviceError` on unknown/unavailable local device; HPC transport gap F24 | `dtwc:deviceError`, as C++ |
| Delegates to | one process-wide `{Device, index}` | `dtwc::device(name)` after local validation, so C++ holds the local selection; Python records `hpc` / `hpc:gpu` itself and checks credentials at submit | MEX `set_device` → `dtwc::device(name)` |

C++ and MATLAB validate through `dtwc::device(name)`; Python parses with the same
grammar (`dtwc::detail::parse_device`), checks that the GPU is live, and stores a
CPU/GPU selection in C++ through `dtwc::device(name)`; it records an `hpc` or
`hpc:gpu` selection of its own, which C++ refuses. The friendly name `"gpu"`, and
its alias `"cuda"`, resolves to CUDA (or Metal on macOS) at call time. C++
Tier-1 HPC submission remains the approved 2.1 transport defer. Python owns the
SLURM wrapper, but its current HPC errors violate the frozen taxonomy/messages
(F24).

### 1.2 `load(source, ...)` — lazy dataset handle  `[live in C++/Python/MATLAB]`

| Parameter | C++ `[live]` | Python `[live]` | MATLAB `[live]` |
|---|---|---|---|
| signature | `dtwc::Dataset dtwc::load(source, index_t skip_cols=0, index_t skip_rows=0, char delimiter=0, std::string_view name="")` | `load(source, *, skip_cols=0, skip_rows=0, delimiter=None, name=None) -> Dataset` | `ds = dtwc.load(source, 'skip_cols',0, 'skip_rows',0, 'delimiter','', 'name','')` |
| `source` | `std::filesystem::path` **or** `std::vector<std::vector<double>>` (overloads) | path `str`/`os.PathLike` **or** array-like; a sequence of 1-D sequences may be RAGGED, matching the C++ `series_type` overload (a rectangular array keeps the NumPy fast path) | char path, N×L double matrix, **or** a cell array of numeric vectors (RAGGED, the same `series_type` overload) |
| `skip_cols` | leading columns to drop (id columns), dropped as FIELDS before numeric parsing for a path and erased from each row in memory | same, via the same `DataLoader`: `Dataset.as_series()` parses a path with the C++ reader (text id columns and variable-length rows included) and erases leading in-memory columns, raising `InvalidInput` when `skip_cols` exceeds a series length | same |
| `skip_rows` | leading rows to drop, `>= 0`. Path source: header **lines** of the file (the `dtwc_cl --skip-rows` / `DataLoader::start_row` meaning). Directory source: the same count is applied **per file**, since `load_folder` forwards `start_row` to every `readFile` and one file is one series. In-memory source: leading **series**, since one memory row is one file line. Negative → `InvalidInput` | same, but negative/non-integer is rejected at `cluster()` like `skip_cols` (`ValueError`/`TypeError`); `device="hpc"` rejects a non-zero value (the SLURM wrapper has no `skip_rows` slot) | same; rejected by the `dtwc.load` input parser exactly as `skip_cols` is |
| `delimiter` | `0` = auto from extension (`.tsv/.txt`→`\t` else `,`) | `None` = auto | `''` = auto |
| `name` | `""` = derive from filename stem | `None` = filename stem, else `"dataset"` | `''` = stem |
| result type | `dtwc::Dataset` (lazy; materialises only for local backends) | `dtwc.Dataset` (`_api.py:42`) | `dtwc.Dataset` handle |

*Contract:* `load()` performs **no I/O** — on `device="hpc"` the path is forwarded
to the cluster and never read locally (preserves the 100M-series scaling story).

### 1.3 `cluster(data, k, ...) -> Result`  `[live in C++/Python/MATLAB]`

| Parameter | C++ `[live]` | Python `[live]` | MATLAB `[live]` |
|---|---|---|---|
| signature | `dtwc::Result dtwc::cluster(const Dataset& data, index_t k, std::string_view method="pam", int band=-1, std::string_view device="", int max_iter=100)` | `cluster(data, k, *, method="pam", band=-1, device=None, max_iter=100) -> Result` | `res = dtwc.cluster(data, k, 'method','pam', 'band',-1, 'device','', 'max_iter',100)` |
| `data` | `Dataset` (or path/array via `load`) | `Dataset`/path/array | `Dataset`/path/matrix/cell of numeric vectors (ragged) |
| `k` | `index_t` clusters; `k > N` → `InvalidInput("cluster: k must not exceed the number of series.")`, empty dataset → `InvalidInput("cluster: dataset is empty.")` | same guards, same messages | same guards, same messages, raised by C++ as `dtwc:invalidArgument` |
| `method` | `"auto"·"pam"·"onebatch"·"clara"·"kmedoids"·"mip"·"lrcore"·"tadpole"·"hierarchical"` (aliases `"hclust"`, `"obp"`, `"lr"`, as `dtwc_cl` reads them; ASCII case-insensitive) | same set | same set, routed by the same C++ code |
| `auto` resolution | local CPU: `pam` for N≤5000, else `clara`; local GPU: `pam` at any N; C++ HPC reaches the documented transport error before local resolution | same local rule; HPC forwards `auto` for resolution after remote materialisation | same local rule (the same `dtwc::run`) |
| `band` | Sakoe-Chiba band, `-1` = full | `-1` | `-1` |
| `device` | `""` = global default; else per-call override | `None` = global | `''` reads the process device (`dtwc.device()`); a non-empty value is a per-call override that configures the local `Problem`'s distance strategy (GPU ordinal included) and never mutates the process device |
| `max_iter` | `100` | `100` | `100` |
| unknown `method` | `InvalidInput` (never silently PAM) | `ValueError` (`_normalize_method`, `_api.py:413-433`) | `dtwc:invalidArgument` |

**MATLAB routes through C++, it does not re-implement.** `dtwc.cluster`
(`bindings/matlab/+dtwc/cluster.m`) parses arguments and makes one gateway call,
`dtwc_mex('tier1_cluster', ...)`, which builds a `dtwc::Dataset` with
`dtwc::load` and calls `dtwc::cluster`. Method routing, `auto` resolution, the
`k <= N` and empty-dataset guards, `max_iter`, the per-call device override and
the `skip_cols`/`skip_rows`/`delimiter`/`name` source semantics therefore have
exactly one implementation. `dtwc.Result` holds the returned `dtwc::Result`, so
`score()` and `save()` are the C++ members and the four CSVs carry the
dataset's series names. This closes F40.

**C++ `cluster()` is a `dtwc::run` call (IF-2 S3, `dtwc/cli/run.hpp`).** It builds
a `dtwc::Config` holding `dtwc_cl`'s default for every setting its signature does
not name, with `output` empty (nothing is written; `save()` writes), the dataset's
name, and the device `device` names (grammar only) or else the process device,
and calls `run(config)` for a path or `run(config, data)` for in-memory series.
`dtwc_cl` is the same `run` of the `Config` its flags and `--config` file build,
so the two share one method x device resolution: on `gpu` the matrix methods
(`pam`, `kmedoids`, `mip`, `lrcore`, `hierarchical`, and `clara` when its sample
covers every series) run with the GPU filling the matrix, while `onebatch`,
`tadpole` and a smaller `clara` sample, which compute on the CPU as they go,
raise `DeviceError`. A path dataset reads every format `dtwc_cl` reads (CSV/TSV,
a folder, Parquet, Arrow IPC).

**Deterministic Tier-1 seed (2.0 addendum).** The cross-language
invocation-local default is 42, exposed as
`dtwc::settings::DEFAULT_RANDOM_SEED`, `dtwcpp.DEFAULT_RANDOM_SEED`, and
`dtwc.default_random_seed()`. The C++/Python/MATLAB Tier-1 PAM route and the
seed-aware OneBatchPAM/CLARA routes construct or receive a local engine from
this value; `auto` inherits the resolved method. `DTWClustering` restart `i`
uses `DEFAULT_RANDOM_SEED + i` and retains the lowest-cost result, with the
schedule range-checked. `DTWCKMedoids(random_state=None)` means the same default.
CLI `--seed` defaults to 42, applies to PAM, OneBatchPAM, and CLARA, and accepts
`[0, UINT64_MAX]`. Lloyd k-medoids uses the same invocation-local default and a
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

Canonical class name is **`Result`** in all three languages.

| Member | C++ `dtwc::Result` `[live]` | Python `dtwcpp.Result` `[live]` | MATLAB `dtwc.Result` `[live]` |
|---|---|---|---|
| `labels` | `const std::vector<index_t>& labels() const` (`index_t` = `std::int64_t`) | `res.labels` → `np.ndarray[int64]` | `res.labels` → double row (1-based) |
| `medoids` | `const std::vector<index_t>& medoids() const` | `res.medoids` → `np.ndarray[int64]` | `res.medoids` → double row (1-based) |
| `score(name)` | `double score(std::string_view name) const`; fills the retained `Problem` on demand after a matrix-free run | `res.score(name: str) -> float`; same lazy fill, so `onebatch`/`clara`/`tadpole` results are scoreable and `save()` writes all four CSVs | `s = res.score(name)` |
| `distance_matrix` | `std::vector<double> distance_matrix() const` `[introduced-2.0]` — dense **row-major N x N**; fills the retained `Problem` on demand exactly as `score()` does, so a matrix-free run is still readable | `res.distance_matrix` → dense N x N `np.ndarray`; a matrix-free `onebatch`/`clara`/`tadpole` run leaves it unmaterialised and the property fills the retained `Problem` on first read, exactly as `score()`/`save()` do (`_api.py:160-172`). `None` only for an `hpc` run, which has no local `Problem` | private helper `Result.distance_matrix()` (`Result.m:107-121`), used by `plot()` |
| `save(dir)` | `void save(const std::filesystem::path& dir) const` | `res.save(dir)`; writes the loader's series names (not ordinals), C++'s line endings (the platform one for the three text-mode files, LF for the binary-mode distance matrix), `setprecision(8)` silhouettes and `to_chars(general, max_digits10)` matrix values, so a Python run and a CLI run on one file are byte-identical; an undefined silhouette warns (`RuntimeWarning`, stderr) and skips the file | `res.save(dir)` |
| `plot()` | **not provided** — C++ writes plottable CSV via `save()` | `res.plot(png="clusters_2d.png", show=True)` (`_api.py:330-367`) | `res.plot()` |
| (aux) `cost` | `double cost() const` | `res.cost` (`_api.py:153`) | `res.cost` |
| (aux) `device` | `std::string device() const` | `res.device` | `res.device` |
| (aux) run statistics | `ClusterMethod method() const` (`auto` resolved), `int iterations() const`, `bool converged() const` `[introduced-2.0]` (IF-2 S3; the first `RunStats` fields, IF-4) | — | — |

*`score(name)` names* (accepted in every language; resolve to the Tier-2 `scores::*`
functions in §2.4): `"silhouette"` (returns the **mean** silhouette),
`"davies_bouldin"`, `"dunn"`, `"calinski_harabasz"`, `"inertia"`. Unknown name →
`InvalidInput` listing the valid set.

*`save(dir)` contract* (identical bytes in every language — this is the preserved
output contract, §7 item 2): writes `<name>_labels.csv` (`"name,cluster"`),
`<name>_medoids.csv` (`"cluster,medoid_index,medoid_name"`),
`<name>_distance_matrix.csv`, and `<name>_silhouettes.csv`
(`"name,cluster,silhouette"`) into `dir`. These are exactly the corresponding
CLI outputs when present. When fewer than two clusters are realised (k = 1, or a
collapsed partition) the silhouette is undefined: `save` writes the other files,
warns on stderr, and skips `<name>_silhouettes.csv`; `score("silhouette")`
still raises `UndefinedScore`. A matrix-free CLI run emits labels and medoids without
forcing the matrix-only files; this approved exception is specified in §7 item
2.

*`plot()` is Python/MATLAB only.* It renders a classical-MDS 2D scatter of the
distance matrix coloured by cluster (`_api.py:330-367`). **C++ has no `plot()`**:
it calls `save(dir)` to emit the plottable CSVs above, which any plotting tool
(or the `dtwc.visualize` skill) consumes. On an `hpc` run only labels return, so
`plot()` prints cluster sizes and returns nothing (`_api.py:336-340`).

### 1.5 `DTWClustering` — sklearn-style estimator (Python + MATLAB)

A live public class exists in both bindings and 2.0 **retains it** (it is the
scikit-learn-idiomatic entry point, distinct from the functional Tier-1 `cluster()`):
Python `dtwcpp.DTWClustering` (`python/dtwcpp/_clustering.py:39`, `BaseEstimator,
ClusterMixin`) and MATLAB `dtwc.DTWClustering` (`bindings/matlab/+dtwc/DTWClustering.m`).
It has no C++ twin (sklearn estimator idiom is language-specific) and stays
Python/MATLAB-only.

The constructors now expose the shared parameter set, but exposure alone did
not prove execution:

| Parameter | Python | MATLAB | Current status |
|---|---|---|---|
| `device` / `Device` | `device=None` `[introduced-2.0]`, routed by `_clustering.py` | `Device=''` `[introduced-2.0]`, validated through `dtwc::device` and restored afterwards (a per-call override, never a global mutation), then applied to the estimator `Problem`'s distance strategy |
| `metric` / `Metric` | `metric='l1'` `[introduced-2.0]`, consumed by fit | `Metric='l1'` or `'squared_euclidean'`, consumed by fit: a non-L1 metric builds the exact matrix through `dtwc_mex('DTWClustering_compute_distance_matrix', X, band, metric)` and sets it on the `Problem`, as `_clustering.py` does |

Both estimators converge on the shared constructor set `{n_clusters, variant, band,
max_iter, n_init, wdtw_g, adtw_penalty, missing_strategy, metric, device}`.
Both are now executed, not merely exposed: `Metric` is normalised and
validated (including the `Variant`/`MissingStrategy` cross-products) before any
input, device, or `Problem` effect, and an unknown value raises
`dtwc:invalidArgument`. This closes F18.

---

## 2. Tier 2 — advanced object surface

Retained for power users. `Problem` stays a first-class object. Canonical
snake_case methods, core-owned algorithm result writeback, and the frozen
encapsulation/accessor split are live. The distance and device settings are
private; seven deliberately retained expert/result fields remain public. A live symbol
in the tables below does not imply that every other promised invariant or
deprecation diagnostic is complete.

### 2.1 `Problem` — configuration setters

Canonical config setters are snake_case. Seven expert/result fields remain
public: the v1.0.0 `int` fields `maxIter` and `N_repetition`, plus
`band`, `mip_settings`, `init_fun`, `clusters_ind`, and `centroids_ind`. The
setters check what they are given; a direct write to a field does not. The
device settings are read with `distance_strategy()` and `cuda_settings()`. The distance settings (variant and parameters,
metric, missing-data strategy, band) are one private `DistanceConfig`, read with
`distance()`, `variant_params()`, `missing_strategy()` and `metric()` and changed
with `set_distance()` or the per-setting setters; a change drops the distance
matrix and the clustering, which describe the old distances. `band` is also the
v1.0.0 field: a direct write takes effect at the next fill or `dtw_function()`
call, and until then `is_distance_matrix_filled()` is false.

`dist_by_ind(i, j)` reads the matrix: O(1), no check and no computation, so a
parallel loop reads it freely. It needs a matrix that holds the pair, which
`fill_distance_matrix()` provides; every method that reads the matrix fills it
first (FastPAM, Lloyd's steps, the MIP and LR-core, the dendrogram, the scores,
`init::Kmeanspp`), while OneBatchPAM, FastCLARA's assignment and TADPole compute
the pairs they need through `dtw_function()`. `is_distance_matrix_filled()` is a
flag, set by a fill and by a complete read, load or mapped bind. The v1.0.0
`distByInd`, which computed a pair on demand, fills the matrix on its first
call. Python binds `clusters_ind` and `centroids_ind` read-only
(v1.0.0's Python never bound them): a clustering reaches a Python `Problem`,
and the scores that read it, through `set_result(ClusteringResult)`, which
validates it as C++ `Problem::set_result` does.

The C++ accessors stay unchecked because they sit in hot loops, so the Python
and MATLAB boundaries own the range checks: `dist_by_ind(i, j)`, `series(i)`,
`series_name(i)` and `centroid_of(i)` raise `InvalidInput` for an index outside
`[0, N)`, naming the index and N, and `centroid_of(i)` raises it for a `Problem`
that holds no clustering. MATLAB exposes `dist_by_ind` only (1-based, checked
in the MEX); it has no series, name or medoid accessor.

A `Problem` holds a clustering when `clusters_ind` has one label per series and
`centroids_ind` one medoid per cluster. Both stay empty until a clustering writes
them (`set_result`, `set_clusters` for the medoids, the algorithms):
`set_n_clusters` sizes neither, and `set_data` and `set_view_data` empty both (the
labels describe the series they were computed on). `find_total_cost`, `write_clusters`,
`print_clusters`, `write_medoid_members`, `calculate_medoids`, the scores and
Python's `centroid_of` raise `InvalidInput` ("... cluster it first") on a
`Problem` that holds none, one check per call; C++ `centroid_of(i)` stays
unchecked, and `require_clustered(who)` is the check. `set_n_clusters(k)` refuses
k < 1 and `set_band(b)` refuses b < -1 with `InvalidInput`; k > N is refused by
`cluster()`, since the data may change after the setter.

| Concept | C++ 2.0 `[rename]` | Python 2.0 | MATLAB 2.0 | Live source |
|---|---|---|---|---|
| k | `set_n_clusters(index_t)` | `set_n_clusters(n)` | `set_n_clusters(k)` | canonical setters own behavior; the retained C++ `set_numberOfClusters` is a deprecated warning alias |
| method (enum) | `method()` / `set_method(Method)` | `set_method(Method)` / `method` prop | `set_method(str)` `[introduced-2.0]` | live in all three routes |
| band | `set_band(int)` | `band` prop / `set_band` | `set_band(b)` | retained field `band` (`Problem.hpp`); MEX `set_band` |
| max iterations | `set_max_iter(int)` | `max_iter` prop | `set_max_iter(n)` | public `int maxIter` field (v1.0.0) plus the canonical accessor (`Problem.hpp`/`Problem.cpp`); `n < 1` raises `InvalidInput` |
| repetitions | `set_n_repetitions(int)` | `n_repetitions` prop | `set_n_repetitions(n)` | public `int N_repetition` field (v1.0.0) plus the canonical accessor (`Problem.hpp`/`Problem.cpp`); `n < 1` raises `InvalidInput` |
| random seed | `random_seed()` / `set_random_seed(uint64_t)` | `random_seed` prop / `set_random_seed` | Tier-1 default via `dtwc.default_random_seed()`; method-specific `Seed` where exposed | private state, default `DEFAULT_RANDOM_SEED` |
| variant (enum) | `set_variant(core::DTWVariant)` | `set_variant(DTWVariant)` | `set_variant(name[,param])` | `Problem.hpp`; `_dtwcpp_core.cpp` |
| variant (params) | `set_variant(core::DTWVariantParams)` — **rebinds `dtw_fn_`** | `set_variant_params(DTWVariantParams)` | `set_variant(name, param)` | `Problem.hpp`; `_dtwcpp_core.cpp` |
| missing strategy | `missing_strategy()` / `set_missing_strategy(core::MissingStrategy)` | `missing_strategy` prop | `set_missing_strategy(str)` | private state (`Problem.hpp`) |
| distance settings | `distance()` / `set_distance(core::DistanceConfig)` `[introduced-2.0]` | — | — | private state: variant and parameters, metric, missing-data strategy and band in one struct; `ndim` is the series' |
| metric | `metric()` / `set_metric(core::MetricType)` `[introduced-2.0]` | — (IF-2 S4) | — (IF-2 S4) | private state, default `L1`: the pointwise cost of every distance the `Problem` computes (CPU fill, the bound function, GPU routes, mmap cache and checkpoint identities); a metric other than `L1` takes Standard DTW with `MissingStrategy::Error`, else `InvalidInput` |
| distance strategy | `distance_strategy()` / `set_distance_strategy(DistanceMatrixStrategy)` | `distance_strategy` prop | `set_distance_strategy(str)` | private state (`Problem.hpp`) |
| device | `set_device(Device, int index = 0)` `[introduced-2.0]` | `Problem(name, *, device="cpu")` / `set_device(name)` `[introduced-2.0]` | `dtwc.Problem(name, 'Device', d)` / `set_device(name)` `[introduced-2.0]` | `Problem.hpp`; names parsed by the one device grammar (§6.4) |
| TADPole cutoff | `tadpole_dc()` / `set_tadpole_dc(double)` | — | — | private C++ state; CLI exposes `--dc` |
| solver | `[[nodiscard]] set_solver(Solver) -> bool` | `set_solver(Solver) -> bool` `[introduced-2.0]` | `ok = set_solver(str)` `[introduced-2.0]` | live in all three routes; `false` when `Gurobi` is requested on a build without it, and the solver is then HiGHS |
| MIP settings | `mip_settings` field | `mip_settings` prop | `set_mip_settings(struct)` `[introduced-2.0]` | live in all three routes; fields `mip_gap`, `time_limit_sec`, `warm_start`, `numeric_focus`, `mip_focus`, `verbose_solver`, `lr_max_nodes` |
| CUDA settings | `cuda_settings()` / `set_cuda_settings(CUDASettings)` | `cuda_settings` prop (a copy) `[introduced-2.0]` | `set_cuda_settings(device_id, precision)` `[introduced-2.0]` | live in all three routes |
| output folder | `output_folder()` / `set_output_folder(path)` | `output_folder` prop `[introduced-2.0]` | `set_output_folder(dir)` `[introduced-2.0]` | live in all three routes; default `./results/`, relative to the working directory (the process-global `settings::paths` it replaced is removed, §3 rows 37-38) |
| verbose | `verbose()` / `set_verbose(bool)` | `verbose` prop | `set_verbose(tf)` | live in all three routes |
| problem name | `name()` / `set_name(std::string)` | `name` prop | `name()` / `Name` (read-only) | private C++ state with live binding reads |
| data (owning) | `data() const` / `set_data(Data)` | `set_data(series, names)` | `set_data(X)` | read-only C++ accessor plus live setters |
| data (view) | `set_view_data(Data)` | — | — | C++ only: the view path CLARA uses; no binding exposes it |

### 2.2 `Problem` — distance-matrix & clustering methods `[rename: camelCase → snake_case]`

| C++ retained 1.x alias (Problem.hpp) | C++ 2.0 canonical | Python 2.0 | MATLAB 2.0 |
|---|---|---|---|
| `refreshDistanceMatrix()` | `refresh_distance_matrix()` | `refresh_distance_matrix()` (live) | `refresh_distance_matrix()` `[introduced-2.0]` |
| `readDistanceMatrix(path)` | `read_distance_matrix(path)` | `read_distance_matrix(path)` `[introduced-2.0]` | `read_distance_matrix(path)` `[introduced-2.0]` |
| `maxDistance()` | `max_distance()` | `max_distance()` (live) | `max_distance()` `[introduced-2.0]` |
| `distByInd(i,j)` | `dist_by_ind(i,j)` | `dist_by_ind(i,j)` (live) | `dist_by_ind(i,j)` (1-based, live) |
| `isDistanceMatrixFilled()` | `is_distance_matrix_filled()` | `is_distance_matrix_filled()` (live) | `is_distance_matrix_filled()` (live) |
| `fillDistanceMatrix()` | `fill_distance_matrix()` | `fill_distance_matrix()` (live) | `fill_distance_matrix()` (live) |
| `printDistanceMatrix()` | `print_distance_matrix()` | `print_distance_matrix()` `[introduced-2.0]` | — |
| `writeDistanceMatrix([name])` | `write_distance_matrix([name])` | `write_distance_matrix()` (live) | — |
| — (reader) | `distance_matrix()` † | `distance_matrix()` ‡ (independent NumPy copy) | `distance_matrix()` |
| — (writer) | `set_distance_matrix(...)` | `set_distance_matrix(...)` (used by `_api.py`) | `set_distance_matrix(D)` (live in `Problem.m`) |
| `use_mmap_distance_matrix(path)` | `use_mmap_distance_matrix(path)`, for the `Problem`'s `metric()`; `use_mmap_distance_matrix(path, metric)` binds a cache for `metric`, which becomes the `Problem`'s metric (a bind that throws changes neither) | `use_mmap_distance_matrix(path)` `[introduced-2.0]` | — |
| `findTotalCost()` | `find_total_cost()` | `find_total_cost()` (live) | `find_total_cost()` (live) |
| `assignClusters()` | `assign_clusters()` | `assign_clusters()` (live) | — |
| `calculateMedoids()` | `calculate_medoids()` | `calculate_medoids()` (live) | — |
| `cluster()` | `cluster()` | `cluster()` (live) | `cluster()` `[introduced-2.0]` |
| `cluster_by_MIP()` | `cluster_by_mip()` | — | — |
| `cluster_by_kMedoidsPAM()` | `cluster_by_kmedoids_lloyd()` | — | — |
| `printClusters()` | `print_clusters()` | `print_clusters()` (live) | — |
| `writeClusters()` | `write_clusters()` | `write_clusters()` (live) | — |
| `writeMedoidMembers(iter,rep=0)` | `write_medoid_members(iter, rep=0)` | `write_medoid_members(...)` `[introduced-2.0]` | — |
| `writeSilhouettes()` | `write_silhouettes()` | `write_silhouettes()` (live) | — |

**† Name collision (adjudicated in §10 item 6).** C++
`Problem::distance_matrix()` returns the `core::DistanceMatrix` by reference:
the packed lower triangle, on the heap or mapped (`use_mmap_distance_matrix`),
read and written through the same `get` / `set`. The Python/MATLAB spelling
returns an NxN numeric matrix of either storage; Python returns an independent
copy. The language-specific semantics are retained.

**‡ Python read/write names.** `distance_matrix()` and
`set_distance_matrix()` are the only spellings; no `_numpy` variant exists.

**Reading and writing files (2026-09-24).** `read_distance_matrix(path)` takes
only a matrix of this `Problem`'s size: a file whose row count is not the series
count, an empty file included, raises `InvalidInput` and leaves the matrix
untouched (a `Problem` without series takes any matrix); a non-square or
asymmetric file raises `InvalidInput` naming the row. `write_clusters`,
`write_silhouettes`, `write_medoid_members`, `write_distance_matrix` and Tier-1
`Result::save` check each file after closing as well as after opening, so a
write lost after a successful open (a full disk, a file-size quota) raises
`IOError` instead of leaving a truncated file behind a success.

Read accessors required by the frozen contract are live: `size()`,
`n_clusters()` (was `cluster_size()`), `name()`, `series(i)`,
`series_name(i)`, `labels()`, `medoids()`, and `centroid_of(i)`.
The eight same-name reads for encapsulated state are `method()`, `random_seed()`,
`last_iterations()`, `tadpole_dc()`, `verbose()`, `output_folder()`,
`name()`, and `data()`.
`last_iterations()` is intentionally read-only, and
`data()` returns `const Data&`; the other six configuration values have
`set_*` mutators, while data replacement uses
`set_data()` or `set_view_data()`.

**Remaining live C++ `Problem` members — frozen fate and current status.**

| Live C++ member | Source | 2.0 fate |
|---|---|---|
| `set_clusters(const std::vector<index_t>&)` | `Problem.hpp` | **stays C++-only**, canonical `set_clusters` (already snake_case); seeds candidate medoids. Not bound (internal seeding hook). v1's `set_clusters(std::vector<int>&)` stays `[[deprecated]]` and converts. |
| `cluster_and_process()` | `Problem.hpp` | **stays C++-only** convenience (cluster + write outputs). The Tier-1 `cluster()` free function (§1.3) is its cross-language successor; not bound. |
| `resize()` | `Problem.hpp` | private invariant maintenance, as frozen |
| `init()` | `Problem.hpp` | **stays C++-only**, canonical `init` (runs `init_fun`); not bound. |
| `last_iterations()` | `Problem.hpp` | read-only accessor over private algorithm-owned state; no public setter |
| `init_fun` (`std::function`) | `Problem.hpp` | public C++-only callable extension point; no `set_init_strategy` enum (§10 item 7) |

### 2.3 `DataLoader` (C++ Tier-2 only) `[rename: camelCase → snake_case]`

CSV/TSV builder. Bindings do **not** expose `DataLoader` — Tier-1 `load()`
covers the binding use case. `dtwc::read_data(path, skip_cols, skip_rows,
delimiter, column)` (`io/read_data.hpp`) is the one multi-format entry (CSV/TSV,
a folder, Parquet, Arrow IPC) that `dtwc_cl`, Tier-1 `cluster()` and Python's
`load()` share; its text branch is a `DataLoader`. Chained setters return
`DataLoader&`.

| C++ live (DataLoader.hpp) | C++ 2.0 canonical |
|---|---|
| `startColumn(int)` (`DataLoader.hpp:291-297`) | `start_column(int)` |
| `startRow(int)` (`DataLoader.hpp:299-306`) | `start_row(int)` |
| `n_data(int)` | `n_data(int)` (unchanged) |
| `delimiter(char)` | `delimiter(char)` (unchanged) |
| `path(fs::path)` | `path(fs::path)` (unchanged) |
| `verbosity(int)` | `verbosity(int)` (unchanged) |
| `load() -> Data` / `count()` | `load()` / `count()` (unchanged) |

The canonical setter names in this section are implemented.
`start_column(int)` and `start_row(int)` own the loader mutations. The camelCase
spellings remain deprecated inline forwarders for the 2.x transition. The no-argument `startColumn()`/`startRow()`
getters were not renamed by the frozen table and remain unchanged.

### 2.4 `scores::*` free functions `[rename: camelCase → snake_case]`

Canonical scheme drops the redundant `Index`/`Information` noun (the fixed
decision `daviesBouldinIndex → davies_bouldin` sets the pattern; applied
uniformly). Same name in all three languages.

| Concept | C++ | Python | MATLAB |
|---|---|---|---|
| silhouette | `silhouette(prob)` | `silhouette(prob)` | `silhouette(prob)` |
| Davies–Bouldin | `davies_bouldin(prob)` | `davies_bouldin(prob)` | `davies_bouldin(prob)` |
| Dunn | `dunn(prob)` | `dunn(prob)` | `dunn(prob)` |
| inertia | `inertia(prob)` | `inertia(prob)` | `inertia(prob)` |
| Calinski–Harabasz | `calinski_harabasz(prob)` | `calinski_harabasz(prob)` | `calinski_harabasz(prob)` |
| Adjusted Rand | `adjusted_rand(l1,l2)` ‡ | `adjusted_rand(l1,l2)` | `adjusted_rand(l1,l2)` |
| Normalized MI | `normalized_mutual_info(l1,l2)` ‡ | `normalized_mutual_info(l1,l2)` | `normalized_mutual_info(l1,l2)` |

Only `silhouette` shipped in v1.0.0; the camelCase and `*_index` spellings were
never released and are gone. `silhouette` on an unclustered Problem raises
`InvalidInput`. The canonical Adjusted-Rand and Normalized-MI spellings are
adjudicated in §10 item 1.

### 2.5 Algorithm free functions (Tier-2, all languages)

| Function | C++ | Python (`_dtwcpp_core.cpp`) | MATLAB (`+dtwc/`) |
|---|---|---|---|
| FastPAM | `fast_pam(Problem&, index_t k, int max_iter=100)` | `fast_pam(prob, n_clusters, max_iter=100)` | `fast_pam(prob, k, 'max_iter',100)` |
| FastCLARA | `algorithms::fast_clara(Problem&, CLARAOptions)` | `fast_clara(prob, n_clusters, sample_size=-1, n_samples=5, max_iter=100, seed=42)` | `fast_clara(prob, k, ...)` |
| dendrogram build | `algorithms::build_dendrogram(Problem&, HierarchicalOptions)` | `build_dendrogram(prob, opts=HierarchicalOptions())` | `build_dendrogram(prob, ...)` |
| dendrogram cut | `algorithms::cut_dendrogram(Dendrogram, Problem&, index_t k)` | `cut_dendrogram(dend, prob, k)` | `cut_dendrogram(dend, prob, k)` |

**`max_iter` of `fast_pam`.** `0` returns the BUILD medoids without a SWAP
(`converged` false) in all three languages; a negative count raises `InvalidInput`
from the C++ core, and neither binding checks it again. `fast_clara`, the
`Problem` setter and the Tier-1 `cluster` give `0` no meaning and refuse it.

**Result write-back (implemented).** `fast_pam`/`fast_clara` and
`cut_dendrogram` write `labels`/`medoids`/`k` back into `Problem` in C++.
Python and MATLAB both rely on that core-owned writeback; neither binding
repeats the assignment.

**LR-core is live.** `Method::LRCore` dispatches the implemented exact
Lagrangian-relaxation solver. `Method::MIP` remains the solver-backed exact
route; neither name is reserved future work.

### 2.6 Distance free functions (Tier-2, all languages)

Canonical namespace is `dtwc::distance::*` (`distance.hpp:33`), mirrored by
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
  (`distance.hpp:35-42`, no `DTWVariantParams` arg) *and* the variant dispatcher
  (`distance.hpp:94-144`, with `DTWVariantParams`). The two are C++ overloads resolved
  by argument list, so there is no `dtwc::distance::standard`.
- In **Python/MATLAB**, the standard single-pair call is named `standard(x,y,…)`
  (there is no argument overloading), and `dtw(x,y,variant=…)` is the dispatcher
  only (`distance.py`).

Net: `dtw` = "standard DTW" in C++ but "dispatcher" in Python/MATLAB. This is
accepted rather than unified because C++ overloading and the Python/MATLAB
keyword-dispatch idiom cannot share one signature; unifying would force an
un-idiomatic name on one side. Section 10 item 8 adjudicates this carve-out.

**Input domain (2026-09-24, FX-15).** Every `dtwc::distance::*` function,
`core::dtw_runtime` and `soft_dtw_gradient` checks `x` and `y` once per call,
before any distance work, and raises `InvalidInput` naming the series, the
position and the fix for a NaN or ±inf value. `missing`, `arow` and the
dispatcher under a ZeroCost, AROW or Interpolate missing strategy read NaN as a
missing value and reject only ±inf. Python's distance functions and
`compute_distance_matrix`, and MATLAB's `dtwc.distance.*`, apply the same
check. The per-pair wrappers in `warping*.hpp` (with `soft_dtw()` and
`core::msm_distance` / `twe_distance`) are the documented unchecked layer the
matrix fills call: they require finite input (the missing-data wrappers also
take NaN) and return NaN, the unreachable `max()` or an ordinary-looking number
otherwise, so their caller checks first, once per call or per fill. A `Problem`
checks its series the same way before it computes (§6.4).

**Precision default.** All `dtwc::distance::*` templates default to
`T = settings::default_data_t`, which is `double`. An explicit `<float>`
instantiation computes and returns `float`; `Problem`/`Result`/matrix routes
return `double` as detailed in §8. Python's single-pair functions uniformly
accept contiguous NumPy array views; MATLAB's public numeric boundary is double.

### 2.7 Checkpoint / resume (Tier-2, implements invariant #4)

The checkpoint/resume surface backs preserved invariant #4 (§7 item 4). Names
are snake_case; current availability and gaps are explicit below.

| Concept | C++ live (`checkpoint.hpp`) | Python | MATLAB 2.0 |
|---|---|---|---|
| options struct | `CheckpointOptions` {`directory`,`save_interval`,`enabled`}, consumed through `Problem::checkpoint` | live: `dtwcpp.CheckpointOptions` and `Problem.checkpoint` (a view, so `prob.checkpoint.enabled = True` mutates the Problem) | live `[introduced-2.0]`; `dtwc.CheckpointOptions` round-trips through `Problem.set_checkpoint(opts)` / `Problem.get_checkpoint()` |
| save checkpoint | `save_checkpoint(const Problem&, path)` writes `checkpoint_path(prob, path)`, tagged with the `Problem`'s `metric()`; `save_checkpoint(prob, path, core::MetricType metric)` tags a matrix a producer outside the `Problem` filled | `save_checkpoint(prob, path, metric=MetricType.L1)` | `dtwc.save_checkpoint(prob, path, metric)`, `metric` a token (`'l1'` default, `'squared_euclidean'`) |
| load checkpoint | `[[nodiscard]] load_checkpoint(Problem&, path) -> bool`, expecting the `Problem`'s `metric()`; `load_checkpoint(prob, path, core::MetricType metric)` expects `metric`; `false` only when the file is absent; other series or settings `InvalidInput`, a file that is not a whole `.dtwm` file `IOError`, neither changing the `Problem` | `load_checkpoint(prob, path, metric=MetricType.L1) -> bool` | `dtwc.load_checkpoint(prob, path, metric) -> logical` |

A checkpoint is one `.dtwm` file, `checkpoint_path(prob, dir)` =
`<dir>/<name>.dtwm` (`distances.dtwm` for an unnamed `Problem`): the layout of a
mapped matrix (below), so a checkpoint maps with `use_mmap_distance_matrix` and a
mapped matrix loads as a checkpoint. A save writes `<name>.dtwm.tmp`, flushes it
to the device and renames it over the previous file; a `Problem` mapped to that
file is its own checkpoint and a save flushes the mapping in place.

`CheckpointOptions` is consumed by `Problem::fill_distance_matrix()` through
the public `Problem::checkpoint` member. With `enabled`, the fill runs the exact
BruteForce row schedule in consecutive blocks of `save_interval` completed
matrix rows and saves after each block, the last block included, so a completed
fill leaves a complete checkpoint. Each save runs on the calling thread after its
block has joined; a save that throws propagates out of `fill_distance_matrix()`,
leaving the computed cells resident and the previous file whole.
`save_interval >= 1` and a non-empty `directory` are required; either violation
raises `InvalidInput` before any distance is computed. `enabled` defaults to
`false`, in which case the fill is unchanged. A save of a matrix in RAM writes all
`N(N+1)/2` doubles, so an automatic fill writes `O(N^3 / save_interval)` bytes in
total; choose `save_interval` so a save is a small fraction of a block (a block
costs about `save_interval * N` DTWs). The CLI opts in with a non-zero
`--checkpoint-interval <rows>`, which requires `--checkpoint <dir>`; the default
`0` saves once, at the end. The CLI's mapped matrix is `<name>.dtwm` in the
`--checkpoint` directory when one is given, else in the output directory.

**Persistent mmap identity (2.0 safety addendum).** A mapped matrix is a `.dtwm`
file: a 48-byte header (magic `DTWM`, version 4, N, a SHA-256 identity) and the
packed lower triangle of doubles, NaN = not computed. The identity covers the raw
IEEE series values, series order and lengths, storage precision, `ndim`, band,
every DTW-variant parameter, multivariate mode, missing-data strategy, pointwise
metric, compute backend, and backend precision. Series names are excluded because
they do not affect distance semantics. Opening an existing file checks its magic,
version, exact length and N, then the identity, before any distance can be read:
a file for other series or settings raises `InvalidInput`; a short or foreign file,
or one of the earlier cache layouts (versions 1-3), raises `IOError`. Either way,
use the original semantics or delete/rename the file and recompute it. A new file
is filled with NaN and flushed to the device before its header is written, so a
power cut can leave a file that does not open, never one whose zeros read as
distances. Semantic setters detach a bound cache without deleting it. The complete
data identity is checked once, at bind; a lookup is a read. `Problem::data()` is
read-only, but heap values exposed by `p_vec()` and caller-owned backing storage
supplied through `set_view_data()` can still change in place. Before such an
edit, call `refresh_distance_matrix()`, or replace the data through
`set_data()`. CUDA
mmap caches require explicit FP32 or FP64 (not hardware-dependent `Auto`). The
metric in the identity is the `Problem`'s `metric()`, which the CPU fill, the
bound function and the GPU routes all compute.

---

## 3. Full 1.x → 2.0 rename table

Frozen registry of public camelCase/duplicate/divergent names. Column
**Compatibility requirement** states the live transition. Every retained
callable alias in this table emits its required C++ compile diagnostic while
forwarding to canonical behavior; the
v1.0.0 C++ fields `maxIter` and `N_repetition` stay plain public fields.
PLAN.md separately retains F22's evidence adjudication because its registered
C++ mutation band was falsified; that does not change the implemented public
policy.

| # | Concept | 1.x name(s) | 2.0 canonical | Compatibility requirement |
|---|---|---|---|---|
| 1 | set k (C++) | `Problem::set_numberOfClusters` (`Problem.hpp`) | `set_n_clusters` | C++ `[[deprecated]]` |
| 3 | set k (MATLAB) | `Problem.set_n_clusters` (Problem.m:148) | `set_n_clusters` | already canonical |
| 4 | max iterations (C++ field) | `Problem::maxIter` (`Problem.hpp`) | `set_max_iter` / `max_iter` accessor | plain field, as in v1.0.0 |
| 6 | repetitions (C++ field) | `Problem::N_repetition` (`Problem.hpp`) | `set_n_repetitions` / `n_repetitions` | plain field, as in v1.0.0 |
| 11 | refresh dist mat | `refreshDistanceMatrix` (`Problem.hpp`) | `refresh_distance_matrix` | C++ `[[deprecated]]` |
| 12 | read dist mat | `readDistanceMatrix` (`Problem.hpp`) | `read_distance_matrix` | C++ `[[deprecated]]` |
| 13 | max distance | `maxDistance` (`Problem.hpp`) | `max_distance` | C++ `[[deprecated]]` |
| 14 | dist by index | `distByInd` (`Problem.hpp`) | `dist_by_ind` | C++ `[[deprecated]]` |
| 15 | is filled | `isDistanceMatrixFilled` (`Problem.hpp`) | `is_distance_matrix_filled` | C++ `[[deprecated]]` |
| 16 | fill dist mat | `fillDistanceMatrix` (`Problem.hpp`) | `fill_distance_matrix` | C++ `[[deprecated]]` |
| 17 | print dist mat | `printDistanceMatrix` (`Problem.hpp`) | `print_distance_matrix` | C++ `[[deprecated]]` |
| 18 | write dist mat | `writeDistanceMatrix` (`Problem.hpp`) | `write_distance_matrix` | C++ `[[deprecated]]` |
| 19 | print clusters | `printClusters` (`Problem.hpp`) | `print_clusters` | C++ `[[deprecated]]` |
| 20 | write clusters | `writeClusters` (`Problem.hpp`) | `write_clusters` | C++ `[[deprecated]]` |
| 21 | write medoid members | `writeMedoidMembers` (`Problem.hpp`) | `write_medoid_members` | C++ `[[deprecated]]` |
| 22 | write silhouettes | `writeSilhouettes` (`Problem.hpp`) | `write_silhouettes` | C++ `[[deprecated]]` |
| 23 | total cost | `findTotalCost` (`Problem.hpp`) | `find_total_cost` | C++ `[[deprecated]]` |
| 24 | assign clusters | `assignClusters` (`Problem.hpp`) | `assign_clusters` | C++ `[[deprecated]]` |
| 25 | calc medoids | `calculateMedoids` (`Problem.hpp`) | `calculate_medoids` | C++ `[[deprecated]]` |
| 26 | cluster via MIP | `cluster_by_MIP` (`Problem.hpp`) | `cluster_by_mip` | C++ `[[deprecated]]` |
| 27 | cluster via Lloyd | `cluster_by_kMedoidsPAM` (`Problem.hpp`) | `cluster_by_kmedoids_lloyd` | C++ `[[deprecated]]` |
| 28 | n clusters read | `cluster_size` (`Problem.hpp`) | `n_clusters` | C++ `[[deprecated]]` alias |
| 28a | n clusters read (Python) | `Problem.cluster_size()` (`py_main.cpp`, a method) | `n_clusters` | kept, silent |
| 35 | start column (loader) | `DataLoader::startColumn` (DataLoader.hpp) | `start_column` | C++ `[[deprecated]]` |
| 36 | start row (loader) | `DataLoader::startRow` | `start_row` | C++ `[[deprecated]]` |
| 37 | set data path | `settings::paths::setDataPath` (2.0-born) | — | removed pre-tag with `set_data_path` (D-3): pass input paths explicitly (`load(path)`, `--input`) |
| 38 | set results path | `settings::paths::setResultsPath` (2.0-born) | `Problem::set_output_folder` | removed pre-tag with `set_results_path` (D-3); a `Problem`'s default output folder is `./results/` |
| 41 | default template scalar | `settings::default_data_t = float` (settings.hpp:30) | `= double` | behaviour change (§8), no name change |
| 42 | CLI dtype default | `--dtype float32` (dtwc_cl.cpp:717-725) | `--dtype float64` | old accepted, default flips (§8) |

**Checkpoint and mmap-cache migration.** A checkpoint directory in the earlier
CSV layout (`CURRENT`, `generations/`) or a cache in an earlier layout (versions
1-3 of `<name>_distmat.cache`) is not read by the `.dtwm` format (version 4):
delete it and rerun; source data is unaffected. At the CLI mmap threshold,
`--dist-matrix` (a CSV matrix) cannot be combined with mapped storage and fails
before either path is opened; `--checkpoint` maps `<dir>/<name>.dtwm`.

**Duplicate-elimination principle (surface report §7).** Documentation exposes
one canonical name per concept. Compatibility aliases remain callable for the
specified transition window; they do not become a second canonical spelling.
All retained aliases and fields in the compatibility inventory now satisfy
their applicable diagnostic and identity/forwarding rules.

---

## 4. Deprecation & removal policy

**2.0 is the break point** (surface report §7: "2.0 removes all duplicates from
bindings").

- **C++.** Every renamed method/function keeps a `[[deprecated("use <new>")]]`
  inline shim forwarding to the canonical implementation. Shims compile-warn,
  never change behaviour, and are scheduled for removal in 3.0. The v1.0.0
  fields `maxIter` and `N_repetition` stay plain public `int` members; the
  checking setters and the accessors are canonical.
- **Python.** The names that 2.0 development renamed were never released, so
  none is kept as an alias. The v1.0.0 method `Problem.cluster_size()` stays,
  silent, beside `n_clusters()`.
- **MATLAB.** No alias survives: MATLAB was not in v1.0.0, so `Problem` has the
  snake_case methods only. The 1-based boundary conversion is untouched.
- **CLI.** Old flag spellings are accepted with a deprecation warning; the SLURM
  callers (`cluster_generic.slurm`, `_hpc.build_dtwc_command`) are updated in the
  same change that renames a flag. The CLI flag set is a de-facto API (§7 item
  3).
- **Nothing silently disappears.** A removed binding name that a user calls must
  raise `AttributeError`/`Unknown command` — never resolve to a different
  behaviour.

This section is normative and implemented for the complete retained inventory:
22 C++ diagnostic entities; no Python or MATLAB alias remains. PLAN.md retains
F22's separate evidence verdict; the exhausted C++ mutation campaign was
falsified at 33/46 and is not described here as closure of that finding.

---

## 5. Error taxonomy

**Fixed decision.** Base `dtwc::Error` (subclass of `std::runtime_error`) plus
four leaf types. Maintained user-facing configuration/dispatch validators raise
typed exceptions; internal assertions are permitted only for preconditions
made unreachable by those validators. `Problem::get_name(i)` and `p_vec(i)`,
which return references into owned heap storage, raise `InvalidInput` on
storage they do not own (a view or memory-mapped series store; for `p_vec`, a
Float32 or metadata-only store too); `series_name(i)` and `series(i)`
(`data().series_f32(i)` for Float32) read heap, view and memory-mapped stores
alike (F25, closed 2026-09-24). Library code does not call `exit()`.
Bindings translate to native exceptions / `mexErrMsgIdAndTxt`.

| C++ type | Covers | Python class | MATLAB identifier |
|---|---|---|---|
| `dtwc::Error` (base) | anything DTWC-thrown not more specific | `dtwcpp.DtwcError(Exception)` | `dtwc:error` |
| `dtwc::InvalidInput` | bad argument: wrong shape/dtype/range, unknown method/metric/variant name, empty data, `ndim` mismatch, unknown `score()` name, a NaN or ±inf value a distance does not take (§2.6), `skip_cols` wider than a row, a matrix stored for other data (a CSV of another size, a `.dtwm` checkpoint or mmap cache of other data or configuration) | `dtwcpp.InvalidInput(DtwcError, ValueError)` | **`dtwc:invalidArgument`** |
| `dtwc::UndefinedScore` | (an `InvalidInput`) a quality score is mathematically undefined for the labelling supplied — fewer than two non-empty clusters. `save` catches it to skip the silhouette file; `score("silhouette")` propagates it | `dtwcpp.UndefinedScore(InvalidInput)` | `dtwc:invalidArgument` (inherited: the MEX ladder catches it as `InvalidInput`) |
| `dtwc::SolverError` | MIP/LP solver failure: infeasible, iteration/time limit hit without optimum, solver returned non-optimal status | `dtwcpp.SolverError(DtwcError, RuntimeError)` | `dtwc:solverError` |
| `dtwc::DeviceError` | device/backend problem: unknown device name; `gpu` on a non-GPU build; a request the device cannot honour (§6.4) and `hpc` asked of a local run (§1.3); `.env`/HPC credential failures (§6) | `dtwcpp.DeviceError(DtwcError, RuntimeError)` | `dtwc:deviceError` |
| `dtwc::IOError` | file/format failure: file not found, unreadable or unwritable (a directory that cannot be created, a quota), bad Parquet/Arrow type, OOB offsets | `dtwcpp.IOError(DtwcError, OSError)` | `dtwc:ioError` |

A failure no public entry point can cause — an unreachable branch, a broken invariant, a precondition every caller validates —
throws `std::logic_error` (Python `RuntimeError`, MATLAB `dtwc:internal`). Every other throw raises a type above. A file that
cannot be parsed (a bad field or row, a bad cache header, a Parquet / Arrow type or offset) is `IOError`; a
well-formed file the request cannot use (a non-square matrix, Parquet nulls, an unknown column, a matrix or cache made for
other data, `skip_cols` wider than a row) is `InvalidInput`. A format this build cannot read (Parquet or Arrow IPC without
Arrow, a memory-mapped matrix without llfio, a YAML `--config` file without fkYAML) is `IOError` too, and a GPU
backend it lacks (`gpu`) is `DeviceError`.

**Binding-translation rules.**

- **Python (nanobind).** Register one exception translator per type. Each leaf
  subclasses both `DtwcError` and the closest built-in (`ValueError`/`OSError`/
  `RuntimeError`) so idiomatic `except ValueError:` and `except dtwcpp.InvalidInput:`
  both work. `dtwc::Error` maps to `DtwcError`. `dtwcpp.UndefinedScore` is the
  one sub-leaf: it subclasses the bound `InvalidInput`, so `except InvalidInput`
  and `except ValueError` still catch it, and its translator branch runs first
  because `dtwc::UndefinedScore` derives from `dtwc::InvalidInput`. The public
  I/O name is `dtwcpp.IOError`; there is no `DtwcIOError` alias (§10 item 5).
- **MATLAB (MEX).** `mexFunction`'s catch ladder maps types to identifiers.
  `dtwc::InvalidInput` **must** map to `dtwc:invalidArgument` — this identifier
  is pinned verbatim by `tests/matlab/test_mex_input_validation.m` and must not
  change. The live ladder maps the DTWC leaf types first
  (`InvalidInput → dtwc:invalidArgument`,
  `SolverError → dtwc:solverError`, `DeviceError → dtwc:deviceError`,
  `IOError → dtwc:ioError`), keeping the std fallbacks below them so
  `dtwc:invalidArgument` continues to fire for the pinned input-validation cases.

---

## 6. Device / `Env` semantics

**Superseded 2026-09-28.** `dtwc::Env` is gone: `dtwc::device(name)` /
`dtwc::device()` hold one process-wide `{Device, index}`, and `Device` is
`{CPU, GPU}`. `hpc` / `hpc:gpu` belong to Python's `dtwcpp.device` and
`slurm_remote.sh`; C++, MATLAB and `dtwc_cl` raise a `DeviceError` naming them,
so the §6.2 `.env` messages live in `python/dtwcpp/_hpc.py` only, and
`Problem::set_device` has no `hpc` case. The text below is kept for history.

**Fixed decision.** `dtwc::Env` owns device (`cpu`/`gpu`/`hpc`) and thread
policy. Series/recurrence precision belongs to `Data`, `Problem`, and CLI
configuration, not Env. Singleton accessor: `dtwc::env()`. C++ and MATLAB
device calls delegate directly; Python stores CPU/GPU selections in Env through
`dtwc::device(name)` but defers HPC validation to its wrapper. **No silent
fallback anywhere.**

### 6.1 Device names

Canonical: `cpu`, `gpu`, `hpc`. Aliases accepted: `gpu:N`, `cuda`, `cuda:N`
(GPU device index N; `gpu` ≡ `cuda` on NVIDIA, ≡ Metal on macOS). Case-insensitive.
One grammar, `dtwc::detail::parse_device`, reads the device names of the C++,
Python and MATLAB APIs, so `cuda` is a spelling of `gpu` in each: on a Metal
build it selects Metal (Python, until 2026-09-24, read it as CUDA-only and
raised). `dtwc_cl --device` reads the same grammar since IF-2 S3; its own `cpu` /
`cuda` / `cuda:N` parser, which refused `gpu` and ran nothing on Metal, is gone.

- **Unknown device name → `DeviceError`** listing valid names, verbatim:
  ```
  [dtwc] unknown device 'foo'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N).
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
keys (the names the live SLURM path already uses,
`scripts/slurm/env.example`): **`SLURM_HOST`**, **`SLURM_USER`**,
**`SLURM_REMOTE_BASE`**. The frozen contract requires each failure mode to
produce the specific actionable `DeviceError` below, never a local fallback.
C++/MATLAB follow Env's messages; Python currently raises wrapper-specific
`RuntimeError` text instead (F24).

The C++/MATLAB tests assert these three messages verbatim. They are authored
here as the exact C++ `DeviceError::what()` strings; F24 requires Python to
reproduce them byte-for-byte.
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

**Superseded 2026-09-28.** Series always load into RAM (`DataLoader::load()`, as
in v1.0.0): the metadata-only load, `StoragePolicy` and its `.dtws` series store
are gone. `hpc` forwards a path (`_hpc.py`); data beyond RAM is list-per-row
Parquet streamed by CLARA / OneBatchPAM under `--ram-limit`. The first two
bullets below are kept for history.

- `device="hpc"`: **metadata-only** local load — shapes/counts/names read
  locally; bulk series streamed to the cluster at submit (`load()` never reads
  the payload; `cluster_on_hpc` forwards a path, `_hpc.py:527-601`).
- Local library series routing is shared by `DataLoader::storage_policy` and
  `Problem::set_storage_policy`. A `Problem` policy governs its next owning
  `set_data` call and is deliberately non-retroactive; `set_view_data` remains
  an explicit non-owning bypass. `Heap` retains owning vectors. `Mmap` writes a
  Float64 `.dtws` store and retains its mapped view for the lifetime of the
  `Problem`; explicit Mmap fails before publication for Float32 data or a build
  without llfio. `Auto` keeps its best-effort threshold behavior, including a
  loud in-memory fallback when the mapped route is unavailable.
- Series storage is independent of distance-matrix storage and both CLI RAM
  controls. `dtwc::run` (the CLI and Tier-1 `cluster()`) loads series `Heap` on
  `gpu`, whose kernels upload owned series, and `Auto` on `cpu`; it controls
  distance storage separately with `--mmap-threshold`, and maps the matrix only
  into an output directory it writes (`output` non-empty). Its
  `--ram-limit` is a conservative cap on Parquet selected-series
  decoding/materialisation, applied before payload I/O; it is not a
  whole-process RSS limit or a library series-storage threshold. Only a single
  list-per-row file can exceed that cap and continue, through non-full CPU
  FastCLARA row-group streaming. View-mode spans (48× CLARA subsample win,
  surface report §6 wart 6 / §8 item 6) are preserved.

### 6.4 A `Problem`'s device (added 2026-09-24, IF-1 / FX-1)

- `Problem::set_device(Device, int index = 0)` is the Tier-1 device choice
  moved onto the session: `dtwc::run` (`dtwc_cl`, Tier-1 `cluster()`) calls it. `cpu` keeps a CPU
  `distance_strategy` the caller chose (`BruteForce`) and moves a GPU
  one to `Auto`; `gpu` selects the build's backend (CUDA, else Metal — the
  `CUDA` / `Metal` strategies remain spellings of `gpu`) and records `index` in
  `cuda_settings().device_id`. `gpu` on a build without a GPU backend raises the
  §6.1 `DeviceError` at the call; `hpc` raises `InvalidInput` (a Tier-1 / CLI
  run option, not a `Problem` device); a negative index raises `InvalidInput`.
- A `Problem` never reads the process-wide `Env` device; one never told a
  device computes on the CPU, as in 1.x. Python's `Problem(device=...)` /
  `set_device(name)` and MATLAB's `'Device'` / `set_device(name)` parse names
  with the same grammar as `Env::set_device` (`dtwc::detail::parse_device`), as
  do Python's `dtwcpp.device()` and the `device=` of `compute_distance_matrix`,
  `cluster` and `DTWClustering` (2026-09-24).
- The request is validated before any pair is computed: by
  `fill_distance_matrix()` (unless the matrix already holds every pair) and by
  each `dtw_function()` / `dtw_function_f32()` call, the accessors OneBatchPAM,
  FastCLARA's assignment and TADPole compute through. On CUDA or
  Metal, a non-Standard variant, a missing-data strategy, `ndim > 1`, Float32,
  mmap-backed, view-mode or metadata-only series, and on Metal precision FP64
  or a GPU index other than 0, raise `DeviceError` naming the setting and value;
  squared L2 (a `SquaredL2` mapped cache) is computed on the GPU. On every
  device, a band `>= 0` narrower than the longest-minus-shortest series length
  (timesteps) raises `InvalidInput` naming both series and the smallest
  feasible band; Soft-DTW, MSM and TWE, which ignore the band, are exempt. A
  ±inf series value, or a NaN under `MissingStrategy::Error`, raises
  `InvalidInput` naming the series and position (§2.6), as does an
  entirely-NaN series under `Interpolate`. The matrix-free schedules compute
  on the CPU under a GPU device, so `dtwc::run` rejects that
  combination: on `gpu`, `onebatch`, `tadpole` and a `clara` sample smaller than
  N raise `DeviceError`, and the rules that need no series (variant, missing-data
  strategy, Float32, Metal index and precision: `validate_gpu_request`, which the
  fill shares) are raised before a series is read.

---

## 7. Preserved invariants (any 2.0 change must not break these)

The five load-bearing constraints from the API-surface report §8, plus the
determinism/index rules, restated as a checklist for the adversarial reviewer:

1. **File formats read.** CSV/TSV (start_row/start_col/delimiter/Ndata,
   folder-of-files), Parquet file+dir with optional `--column`, Arrow IPC
   (`.arrow/.ipc/.feather`)
   (`cli/run.cpp`, `dtwc::run`), Python Polars `large_list<float>` ragged ingest.
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
     lexically-sorted rows back to input order (`_hpc.py:284-304`).
   - *Run-time persistence artifact — NOT part of the save() equal-bytes set.*
     `<name>.dtwm` is written when mapped distance storage is selected, or by
     `--checkpoint`, **during a run** for resume (invariant 4), **not** by
     `Result::save(dir)`, and is outside the `save()`↔CLI byte-identity claim.
3. **CLI flag set + TOML/YAML config keys** (kebab-case, identical in both formats) are a de-facto API:
   `cluster_generic.slurm` and `_hpc.build_dtwc_command` (`_hpc.py:307-353`)
   compose `dtwc_cl` command lines. Renames go through the accept-old-name
   deprecation path (§4) with those two callers updated in the same commit.
   The keys are `dtwc::Config`'s: `cli::bind` is the one key table, and
   `dtwc_cl --print-config` writes every key back as a file `--config` reads
   (tests/conformance/`config_all_fields.toml`, `config_defaults.toml`).
4. **Checkpoint.** One `.dtwm` file, `<dir>/<name>.dtwm`, for the checkpoint
   and the mapped distance matrix alike. It loads (or maps) whenever
   `--checkpoint` is supplied, and a mapped cache reopens automatically. A
   non-full FastCLARA run has no parent distance matrix and therefore rejects
   the checkpoint and imported dense matrix paths. The full-sample PAM fallback
   retains them.
5. **Precision contract.** `Problem`/`Result`/CLI distance results and matrices
   are double, including Float32 storage; explicit C++ helper templates return
   their requested scalar type (§8).
6. **Zero-copy / perf paths.** Nanobind pairwise ndarray routes are zero-copy
   and release the GIL on long calls; Data view-mode spans and interleaved
   multivariate layout
   (`[t0f0,t0f1,t1f0,…]`); lock-free row-partitioned matrix fill
   are preserved.
7. **Determinism.** Seed-aware Tier-1 PAM/OneBatchPAM/CLARA entry points use the
   invocation-local cross-language default 42 (§1.3); estimator restart `i` uses
   `42+i`. The unseeded Tier-2 FastPAM overload retains the legacy mutable
   `std::mt19937 randGenerator(29)`. Scores read state from `Problem`, so result write-back (now in
   C++, §2.5) must run before any `score()`.
8. **MATLAB 1-based conversion at the MEX boundary only**; rectangular N×L
   matrix input still accepted. Optional deps (OpenMP/HiGHS/CUDA/Metal/Arrow)
   stay optional — core builds without them.

---

## 8. Precision story (existing contract, now documented)

**`data_t = double` is the default.** `settings.hpp` defines both internal
`data_t` and public-template `settings::default_data_t` as `double`.

- **Template default (changed in Task 1.5).** Every `dtwc::distance::*` helper
  defaults `T` to `settings::default_data_t`, so
  `dtwc::distance::dtw(x, y)` with no explicit scalar computes in double.
  The migration changed the old Float32 default; the registered pre/post
  fingerprint established that explicitly double-typed calls were
  digit-identical.
- **CLI default (changed in Task 1.5).** `--dtype` now defaults to `float64`
  (`Config::dtype`). `float32`/`f32`/`fp32` remain accepted opt-ins.
- **`float32` is an explicit opt-in only.** C++: build `Data` from
  `vector<vector<float>>` or set `Precision::Float32`; CLI: `--dtype f32`.
  Series storage halves, but inputs are rounded to Float32 and results may
  differ numerically from the Float64 route.
- **Recurrence precision follows series precision; result storage is double.**
  Float64 inputs use double recurrence buffers. The f32 dispatcher instantiates
  the unified kernels with `T=float`, then converts the final distance to the
  public `double` return (`Problem::dtw_fn_f32_t`); dense/mmap distance-matrix
  entries are double. A double result container does not restore precision
  discarded by Float32 inputs or recurrence arithmetic.
- **Explicit helper return type follows the template scalar.**
  `dtwc::distance::*<float>` returns `float`; the default template scalar is
  double. The “result storage is double” rule above is specific to
  `Problem`/`Result`/matrix routes.
- **Storage default.** `Data::precision` and the `Precision::Float64` enum
  comment identify Float64 as the default; Float32 is opt-in.

---

## 9. Permanent verification anchors

- **Tests pin the live path.** Test names must match the algorithm path they
  exercise, and parity tests drive public entry points rather than helpers. The
  Env message tests and MATLAB `dtwc:invalidArgument` tests pin §5/§6.
- **Cross-language conformance fixture.** `tests/conformance/` runs one recorded
  dataset through banded DTW, FastPAM k=3, and three scores from C++, Python,
  MATLAB, and CLI. Labels/medoids are digit-identical; the permanent reference
  values are enforced to relative tolerance 1e-12.
- **Tier-1 native gate.** `tests/unit/test_tier1_cpp_api.cpp` exercises the C++
  device→load→cluster→Result route. Binding parity gates live in
  `tests/python/test_contract_parity.py` and
  `tests/matlab/test_contract_parity.m`.
- **Known coverage limits are named findings.** Symbol-existence tests do not
  close F18–F26; each finding in PLAN.md registers its own behavioral or
  compile-time first gate.

---

## 10. Adjudicated reviewer decisions

1. **Resolved:** canonical score names are `adjusted_rand` and
   `normalized_mutual_info`. The `_index`/`_information` spellings were never
   released and are gone; no compatibility alias exists in Python or MATLAB.
2. **Resolved:** snake_case MATLAB methods are the only spelling; MATLAB has no
   `PascalCase` property aliases.
3. **Resolved:** `Result.score("silhouette")` returns the arithmetic mean of
   the per-series silhouette vector. Tier-2 `scores::silhouette(prob)` returns
   the vector.
4. **Resolved:** HPC credential keys remain `SLURM_HOST`, `SLURM_USER`, and
   `SLURM_REMOTE_BASE`. C++/MATLAB use Env's pinned messages; Python uses the
   same keys but has the F24 exception/message gap.
5. **Resolved:** the public Python I/O exception is `dtwcpp.IOError`,
   subclassing both `DtwcError` and `OSError`. No `DtwcIOError` alias exists.
6. **Resolved:** retain language-specific `distance_matrix()` semantics. C++
   returns the storage variant; Python/MATLAB return an NxN numeric matrix, and
   Python returns an independent copy.
7. **Resolved:** retain `Problem::init_fun` as a public C++-only callable
   extension point. No `set_init_strategy` enum is introduced; arbitrary
   callbacks own their RNG policy.
8. **Resolved:** retain the `dtw` naming carve-out. C++ overloads `dtw` for
   standard DTW and variant dispatch; Python/MATLAB expose `standard` for the
   direct call and `dtw` for dispatch.

---

*End of frozen contract. Confirmed implementation gaps remain 2.0 obligations
under PLAN.md F17–F26 unless a later governed decision explicitly changes
them.*
