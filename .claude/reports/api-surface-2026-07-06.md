# DTWC++ Public API Surface Inventory — 2026-07-06

Purpose: exact current API surface across C++/Python/MATLAB/CLI, to inform the CasADi-style
top-down redesign. Everything below is **[confirmed]** by reading the cited file (line numbers
from the working tree at commit 874edd5, branch `Claude`) unless tagged **[inferred]**.

---

## 1. C++ public surface

Umbrella header: `dtwc/dtwc.hpp` (includes settings, fileOperations, Problem, checkpoint,
DataLoader, distance, scores, utility, warping*, soft_dtw, algorithms/*, core/*; CUDA/Metal/MPI
behind `DTWC_HAS_*`) — dtwc/dtwc.hpp:11-58.

### 1.1 Types & settings

- `data_t = double` (storage/distance precision) — dtwc/settings.hpp:35.
- `settings::default_data_t = float` — **default template argument on all public distance
  helpers** — dtwc/settings.hpp:29. Known-open question (TODO.md:143): deliberate or accident.
- `settings::DEFAULT_BAND = -1`, `DEFAULT_MIP_SOLVER = Solver::HiGHS`,
  `DEFAULT_CLUSTERING_METHOD = Method::Kmedoids`, `DEFAULT_MAX_ITER = 100` — settings.hpp:93-107.
- Global mutable state: `inline std::mt19937 randGenerator(29)` (settings.hpp:43) and
  `settings::paths::data` / `::results` inline globals with `setDataPath`/`setResultsPath`
  (settings.hpp:62-83).

Enums:
- `Method { Kmedoids, MIP }` — enums/Method.hpp:13.
- `Solver { Gurobi, HiGHS }` — enums/Solver.hpp:13.
- `LowerBoundStrategy { Auto, None, Kim, Keogh, KimKeogh }` — enums/LowerBoundStrategy.hpp:22.
- `KernelOverride { Auto, Wavefront, WavefrontGlobal, BandedRow, RegTile }` — enums/KernelOverride.hpp:19.
- `core::ConstraintType { None, SakoeChibaBand }`, `core::MetricType { L1, L2, SquaredL2 }`,
  `core::DTWVariant { Standard, DDTW, WDTW, ADTW, SoftDTW }`,
  `core::MissingStrategy { Error, ZeroCost, AROW, Interpolate }` — core/dtw_options.hpp:17-48.
- `core::DTWVariantParams { variant=Standard, wdtw_g=0.05, adtw_penalty=1.0, sdtw_gamma=1.0 }`
  — core/dtw_options.hpp:51-57.
- `core::StoragePolicy { Auto, Heap, Mmap }`; `core::Precision { Float32, Float64 }` —
  core/storage.hpp:11-23. **Doc bug**: storage.hpp:21 comment says Float32 is "Default", but
  `Data::precision` defaults to `Float64` (Data.hpp:38). The float32 default exists only in the
  CLI (`--dtype` default "float32", dtwc_cl.cpp:148).
- `DistanceMatrixStrategy { Auto, BruteForce, Pruned, CUDA, Metal }` — Problem.hpp:69-75.
- `CUDASettings { device_id=0, precision=0 /*0=Auto,1=FP32,2=FP64, int for header independence*/ }`
  — Problem.hpp:47-54.
- `MIPSettings { mip_gap=1e-5, time_limit_sec=-1, warm_start=true, numeric_focus=1, mip_focus=2,
  verbose_solver=false, max_benders_iter=200, benders="auto" }` — Problem.hpp:57-66.

### 1.2 `Data` (dtwc/Data.hpp:32-152)

Struct with 4 storage modes (heap f64 / heap f32 / view f64 / view f32):
- Public fields: `p_vec : vector<vector<double>>`, `p_vec_f32 : vector<vector<float>>`,
  `p_names : vector<string>`, `ndim=1`, `precision=Float64` (Data.hpp:34-38).
- Ctors: default; `(vector<vector<double>>&&, vector<string>&&, ndim=1)`;
  `(vector<vector<float>>&&, ...)`; two view-mode ctors taking spans + string_views
  (Data.hpp:98-145).
- Accessors: `size()`, `series_length(i)` (= flat/ndim), `series(i)→span<const double>`,
  `series_f32(i)`, `series_flat_size(i)`, `name(i)→string_view`, `is_view()`, `is_f32()`,
  `validate_ndim()` (throws).
- **Wart**: mode-dependent accessor validity — `series(i)` only valid for f64/view,
  `p_vec` is wrong to touch in f32/view modes; nothing enforces it except comments.
- Multivariate layout: interleaved `[t0_f0, t0_f1, ..., t1_f0, ...]` (warping.hpp:137,388).

### 1.3 `DataLoader` (dtwc/DataLoader.hpp:26-161) — builder pattern

- State: `start_col=0, start_row=0, Ndata=-1, verbose=1, delim=',', data_path="."`.
- Chained setters returning `DataLoader&`: `startColumn(int)`, `startRow(int)`, `n_data(int)`,
  `delimiter(char)`, `path(fs::path)` (auto-sets delim from .csv/.tsv extension), `verbosity(int)`.
- Same-named no-arg getters (`startColumn()`, `path()`, ...) — getter/setter overloaded on arity.
- `load() → Data` (file vs folder dispatch via `load_batch_file`/`load_folder`); `count()`.
- **CSV/TSV only.** Parquet/Arrow/.dtws loading lives in the CLI, not here (dtwc_cl.cpp:471-555)
  — the loader story is forked.

### 1.4 `Problem` (dtwc/Problem.hpp:85-273) — god object

Public **fields** (direct-write configuration): `method`, `maxIter=100`, `N_repetition=1`,
`last_iterations`, `band=-1`, `variant_params`, `missing_strategy=Error`,
`distance_strategy=Auto`, `lb_strategy=Auto`, `storage_policy=Auto`, `cuda_settings`,
`mip_settings`, `verbose=false`, `init_fun = init::random` (a
`std::function<void(Problem&)>`), `output_folder = settings::paths::results`, `name`, `data`,
`clusters_ind`, `centroids_ind` (Problem.hpp:129-150).

Ctors: `Problem()`, `Problem(string_view name)`, `Problem(string_view, DataLoader&)`
(Problem.hpp:153-159).

Methods (exact names):
- size/access: `size()`, `cluster_size()`, `get_name(i)` (asserts !view), `p_vec(i)` (asserts
  !view), `series(i)→span`, `series_name(i)`, `centroid_of(i)`.
- config: `set_numberOfClusters(int)`, `set_clusters(vector<int>&)`, `set_solver(Solver)→bool`,
  `set_data(Data)`, `set_view_data(Data)`, `set_variant(DTWVariant)`,
  `set_variant(DTWVariantParams)`.
- distance matrix: `refreshDistanceMatrix()`, `resize()`, `readDistanceMatrix(path)`,
  `maxDistance()`, `distByInd(i,j)`, `dtw_function()`, `dtw_function_f32()`,
  `wdtw_weights_cache()`, `isDistanceMatrixFilled()`, `distance_matrix()`,
  `dense_distance_matrix()` (throws if mmap active), `use_mmap_distance_matrix(path)`,
  `fillDistanceMatrix()`, `printDistanceMatrix()`, `writeDistanceMatrix([name])`.
- clustering: `init()`, `cluster()`, `cluster_by_MIP()`, `cluster_by_kMedoidsLloyd()`,
  `cluster_and_process()`, `findTotalCost()`, `assignClusters()`, `calculateMedoids()`.
- output: `printClusters()`, `writeClusters()`, `writeMedoidMembers(iter, rep=0)`,
  `writeSilhouettes()`.
- Distance matrix storage is `std::variant<core::DenseDistanceMatrix, core::MmapDistanceMatrix>`
  (Problem.hpp:88).

**Naming style is mixed on one class**: `maxIter` (camel), `N_repetition` (snake+cap),
`set_numberOfClusters` (hybrid), `fillDistanceMatrix` (camel) — Problem.hpp:130-131,185,246.

### 1.5 Free-function distance API

Legacy surface (dtwc/warping.hpp): `dtwFull(x,y,metric=L1)`, `dtwFull_L(x,y,early_abandon=-1,
metric=L1)`, `dtwBanded(x,y,band=DEFAULT_BAND,early_abandon=-1,metric=L1)`,
`dtwFull_L_mv(...ndim...)`, `dtwBanded_mv(...)` — each with pointer+len, span, and vector
overloads (warping.hpp:263-456). Variants: `ddtwBanded`, `wdtwBanded(..., g)`,
`adtwBanded(..., penalty)`, `soft_dtw(x,y,gamma)` (no band), `soft_dtw_gradient`,
`dtwMissing_banded`, `dtwAROW{,_L,_banded}` in their warping_*.hpp headers.

New unified namespace (dtwc/distance.hpp:29-100): `distance::dtw(x,y,band=-1,metric=L1)`,
`distance::ddtw`, `distance::wdtw(...,g=0.05)`, `distance::adtw(...,penalty=1.0)`,
`distance::soft_dtw(...,gamma=1.0)`, `distance::missing`, `distance::arow`, plus a dispatcher
`distance::dtw(x, y, DTWVariantParams, band, metric, missing_strategy)`.
**All default `T = settings::default_data_t = float`** (distance.hpp:31 etc.).

### 1.6 Algorithms, scores, checkpointing

- `fast_pam(prob, k, max_iter)`, `algorithms::fast_clara(prob, CLARAOptions)`,
  `algorithms::clarans(prob, CLARANSOptions)`, `algorithms::build_dendrogram(prob,
  HierarchicalOptions)`, `algorithms::cut_dendrogram(dend, prob, k)` → `core::ClusteringResult
  {labels, medoid_indices, total_cost, iterations, converged}`.
  **C++ algorithms do NOT write results back into `Problem`** — the Python/MATLAB wrappers do
  that explicitly ("Store results back into Problem so silhouette(prob) works",
  python/src/_dtwcpp_core.cpp:573-576, 615-619, 744-747; bindings/matlab/dtwc_mex.cpp:647).
- `scores::` (dtwc/scores.hpp:22-32, camelCase): `silhouette(prob)`,
  `daviesBouldinIndex(prob)`, `dunnIndex(prob)`, `inertia(prob)`, `calinskiHarabaszIndex(prob)`,
  `adjustedRandIndex(l1,l2)`, `normalizedMutualInformation(l1,l2)`.
- Checkpointing (dtwc/checkpoint.hpp:31-90): `CheckpointOptions{directory="./checkpoints",
  save_interval, enabled=false}`; `save_checkpoint(prob, path)` (distances.csv + metadata.txt),
  `load_checkpoint(prob, path)→bool`; `save_binary_checkpoint(ClusteringResult, path)` /
  `load_binary_checkpoint`.

---

## 2. Python surface (`dtwcpp`, version 2.0.0)

Binding tech: **nanobind** in python/src/_dtwcpp_core.cpp (the task's `python/py_main.cpp` no
longer exists — replaced; NB_MODULE at _dtwcpp_core.cpp:52).

### 2.1 Bound C++ layer (snake_case throughout)

- Enums: Method, Solver, ConstraintType, MetricType, DTWVariant, MissingStrategy,
  DistanceMatrixStrategy, Linkage (_dtwcpp_core.cpp:59-103).
- Structs: DTWVariantParams, MIPSettings (**omits `max_benders_iter` and `benders`** —
  _dtwcpp_core.cpp:120-141), DendrogramStep, Dendrogram, HierarchicalOptions, CLARANSOptions,
  CLARAOptions, ClusteringResult, DenseDistanceMatrix (`resize/get/set/is_computed/size/max/
  to_numpy (copy)/write_csv/read_csv`), CheckpointOptions.
- `Data`: init `(series, names)`, fields `p_vec, p_names, ndim`, `size`, `series_length(i)`,
  `validate_ndim()` (_dtwcpp_core.cpp:396-410). **No f32, no view mode, no DataLoader binding.**
- `Problem` (_dtwcpp_core.cpp:416-510): rw props `method, max_iter, n_repetition, band,
  variant_params, missing_strategy, distance_strategy, mip_settings, verbose, name,
  clusters_ind, centroids_ind`; ro `size, cluster_size`; methods
  `is_distance_matrix_filled(), max_distance(), dist_by_ind(i,j),
  set_number_of_clusters(n_clusters), set_variant(variant /*enum only, no params*/),
  set_data(series, names), fill_distance_matrix(), distance_matrix_numpy() /*fills+copies*/,
  refresh_distance_matrix(), cluster(), find_total_cost(), assign_clusters(),
  calculate_medoids(), print_clusters(), write_clusters(), write_distance_matrix(),
  write_silhouettes(), set_distance_matrix_from_numpy(dm)`.
  **Not bound**: `set_solver`, `output_folder`, `storage_policy`, `lb_strategy`,
  `cuda_settings`, `init_fun`, `use_mmap_distance_matrix`, `read_distance_matrix`.
- Distance fns: `dtw_distance(x, y, band=-1, metric='l1')` (zero-copy ndarray),
  `ddtw_distance(x,y,band=-1)`, `wdtw_distance(x,y,band=-1,g=0.05)`,
  `adtw_distance(x,y,band=-1,penalty=1.0)`, `soft_dtw_distance(x,y,gamma=1.0)`,
  `soft_dtw_gradient(x,y,gamma=1.0)`, `dtw_distance_missing(...)`, `dtw_arow_distance(...)`
  (_dtwcpp_core.cpp:291-377). Mixed conventions: dtw/missing/arow take ndarray (zero-copy);
  ddtw/wdtw/adtw/soft take `std::vector` (copy).
- `compute_distance_matrix(series, band=-1, metric='l1', use_pruning=True)` (raw);
  `fast_pam(prob, n_clusters, max_iter=100)`;
  `fast_clara(prob, n_clusters, sample_size=-1, n_samples=5, max_iter=100, seed=42)`;
  `clarans(prob, opts)`; `build_dendrogram(prob, opts=HierarchicalOptions())`;
  `cut_dendrogram(dendrogram, prob, k)`; scores as snake_case free functions taking `prob`;
  `save_checkpoint/load_checkpoint`; CUDA/Metal: `cuda_available()`, `cuda_device_info(0)`,
  `compute_distance_matrix_cuda(series, band=-1, use_squared_l2=False, device_id=0,
  verbose=False, use_lb_keogh=False, lb_threshold=-1.0)`, `compute_lb_keogh_cuda`,
  `metal_available()`, `compute_distance_matrix_metal(..., lb_envelope_band=-1)`; capability
  attrs `CUDA_AVAILABLE / METAL_AVAILABLE / OPENMP_AVAILABLE / MPI_AVAILABLE`, `system_info()`.

### 2.2 Pure-Python layers (python/dtwcpp/)

- `__init__.py`: `device(name=None)` global getter/setter (PyTorch-style; cpu|gpu|cuda|cuda:N|hpc),
  `get_device()`, `compute_distance_matrix(series, band=-1, metric='l1', use_pruning=True, *,
  device=None)` (routes to CUDA or CPU; **rejects 'hpc'**), `check_system()` (__init__.py:123-262).
- `_api.py` (the new unified flow): `Dataset(source, *, skip_cols=0, delimiter=None, name=None)`
  lazy handle; `load(source, ...)`; `cluster(data, k, *, method="pam", band=-1, device=None,
  max_iter=100)` → `ClusterResult(labels, device, elapsed_s, k, n_series, medoid_indices,
  distance_matrix, cost, name)` with `.summary()` and `.plot(png=..., show=True)`; module-level
  `plot(result)` (_api.py:25-171). **`method` parameter is accepted but ignored on the local
  path — always FastPAM** (_api.py:154-166; only hpc forwards `method`).
- `_clustering.py`: sklearn-style `DTWClustering(n_clusters=3, variant="standard", band=-1,
  max_iter=100, n_init=1, wdtw_g=0.05, adtw_penalty=1.0, missing_strategy="error", device=None)`
  with `fit/predict/fit_predict/score`; attrs `labels_, medoid_indices_, cluster_centers_,
  inertia_, n_iter_` (_clustering.py:36-95). **No `metric` param** (MATLAB twin has one).
  Variant map excludes softdtw (_clustering.py:99-104).
- `distance.py`: `standard(x,y,band=-1,metric='l1')`, `ddtw`, `wdtw(...,g=0.05)`,
  `adtw(...,penalty=1.0)`, `soft_dtw(...,gamma=1.0)`, `missing`, `arow`, and dispatcher
  `dtw(x, y, *, variant="standard", band=-1, metric="l1", g=0.05, penalty=1.0, gamma=1.0,
  missing_strategy="error")` (distance.py:25-119). Note: `soft_dtw` silently ignores `band`.
- `io.py`: `save/load_dataset_csv`, `save/load_dataset_hdf5`, `save/load_dataset_parquet
  (path, columns=None|column, name_column)`.
- `preprocess.py` (`strip_idle, decimate_zoh, sg_smooth, derivative, z_normalize,
  power_signal`), `diagnose.py` (`diagnose_clusters, cluster_sizes, ...`), `features.py`
  (`summarise`), `convert.py` (dtwc-convert CLI).
- `_hpc.py`: `cluster_on_hpc(source, n_clusters, *, method="pam", device="cpu", band=-1, ...)`,
  `SlurmRemoteRunner(repo_root)` with `preflight/submit_cluster/wait/download_labels`;
  `write_series_tsv`, `parse_labels_csv`, `build_dtwc_command`, `find_dtwc_binary`
  (_hpc.py:31-166).

---

## 3. MATLAB surface (`+dtwc` package + dtwc_mex)

MEX dispatch (bindings/matlab/dtwc_mex.cpp:873-929): `Problem_new/delete/get_info/set_data/
set_band/get_band/set_verbose/set_max_iter/set_n_repetition/set_n_clusters/
set_missing_strategy/set_distance_strategy/set_variant/get_size/get_cluster_size/get_name/
get_centroids/get_clusters/is_distance_matrix_filled/fill_distance_matrix/dist_by_ind/cluster/
find_total_cost/get_distance_matrix/set_distance_matrix`; distance fns (dtw/ddtw/wdtw/adtw/
soft_dtw/soft_dtw_gradient/dtw_distance_missing/dtw_arow_distance); `compute_distance_matrix`;
`derivative_transform`, `z_normalize`; `fast_pam/fast_clara/clarans/build_dendrogram/
cut_dendrogram`; 7 scores; legacy `cluster`; `system_check`.

- `dtwc.Problem` handle class (Problem.m:4-216): props `Band=-1, Verbose=false, MaxIter=100,
  NRepetition=1` (PascalCase, live-synced to C++ on set); dependent ro `Size, ClusterSize, Name,
  CentroidsInd, ClustersInd`; methods `set_data(X /*N×L matrix only*/), fill_distance_matrix(),
  dist_by_ind(i,j) /*1-based*/, set_n_clusters(k), set_variant(name[, param]) /*one positional
  scalar*/, set_missing_strategy(str), set_distance_strategy(str), find_total_cost(),
  get_distance_matrix(), set_distance_matrix(D), is_distance_matrix_filled(), get_handle()`.
  **No mip_settings / method / solver / n-init / output paths** (MATLAB Phase 2, TODO.md:105).
- `dtwc.DTWClustering` value class (DTWClustering.m:50-94): name-value params `NClusters=3,
  Band=-1, Metric='l1', MaxIter=100, NInit=1, Variant='standard', WdtwG=0.05, AdtwPenalty=1.0,
  MissingStrategy='error'`; `fit/fit_predict`; ro `Labels, MedoidIndices, TotalCost` (int32,
  1-based). **No `device` param** (Python twin has one).
- Free functions mirror Python snake_case: `fast_pam(prob,k,varargin)`, `fast_clara`, `clarans`,
  `build_dendrogram`, `cut_dendrogram`, 7 scores, `z_normalize`, `derivative_transform`,
  `compute_distance_matrix(X, 'Band', b)` (**Band only — no Metric/UsePruning/device**;
  compute_distance_matrix.m:30-38; MEX routes through `Problem::fillDistanceMatrix`,
  dtwc_mex.cpp:585-601), `check_system()`, `+distance/` namespace: `standard/ddtw/wdtw/adtw/
  soft_dtw/missing/arow` + `dtw(x,y,'Variant',...,'Band',...,'Metric',...,'G',...,'Penalty',...,
  'Gamma',...,'MissingStrategy',...)` dispatcher (distance/dtw.m:12-19).
- Data layout: **rectangular N×L double matrix only** — `matrix_to_series` transposes
  column-major MATLAB to row series (dtwc_mex.cpp:97-110). No ragged input, no names (names are
  auto "0".."N-1", dtwc_mex.cpp:302-303), no ndim/multivariate. Indices converted 0↔1 based at
  the MEX boundary (`ivec_to_mx_1based`, dtwc_mex.cpp:112-118).

---

## 4. CLI (`dtwc_cl`, dtwc/dtwc_cl.cpp) — plus legacy `dtwc/main.cpp`

CLI11 + TOML (`--config`) + optional YAML (`--yaml-config`). Flags (dtwc_cl.cpp:132-282):
`-i/--input` (CSV/TSV file-or-folder, .parquet/.pq [+dir], .arrow/.ipc/.feather, .dtws),
`-o/--output` (default ./results), `--name` (default "dtwc"), `--column` (Parquet),
`--dtype|--data-precision|--data-type` (default **float32**), `--ram-limit`, `-k/--clusters`
(default 3), `-m/--method` auto|pam|clara|kmedoids|mip|hierarchical (auto: pam if N≤5000 else
clara, dtwc_cl.cpp:581-586), `-b/--band` (-1), `--metric` l1|squared_euclidean, `--variant`
standard|ddtw|wdtw|adtw|softdtw, `--max-iter` (100), `--n-init` (1), `--wdtw-g`,
`--adtw-penalty`, `--sdtw-gamma`, `--sample-size`, `--n-samples`, `--seed` (42), `--linkage`
single|complete|average (default average), `--skip-rows`, `--skip-cols`, `--dist-matrix`,
`--checkpoint`, `--resume|--restart`, `--mmap-threshold` (50000), `--solver` highs|gurobi,
`--mip-gap`, `--time-limit`, `--no-warm-start`, `--numeric-focus`, `--mip-focus`,
`--verbose-solver`, `--benders` auto|on|off, `-d/--device` cpu|cuda|cuda:N (no metal flag),
`--gpu-precision|--gpu-dtype`, `-v/--verbose`.

Outputs (dtwc_cl.cpp:79-125, 839-903): `<name>_labels.csv` ("name,cluster"),
`<name>_medoids.csv` ("cluster,medoid_index,medoid_name"), `<name>_distance_matrix.csv`,
`<name>_silhouettes.csv`, `<name>_checkpoint.bin` (always written), `<name>_distmat.cache`
(mmap, when N ≥ mmap-threshold). The Python HPC path parses `<name>_labels.csv`
(_hpc.py:45) and builds these flags (`build_dtwc_command`) — **the CLI flag set and output
filenames are a de-facto API for the SLURM/hpc integration**.

Known CLI warts (confirmed in TODO.md:70): `--metric` is silently ignored on the CPU path;
`--device` parse is case-sensitive with silent CPU fallback; `std::stoi` on bad `cuda:N`
terminates.

---

## 5. Inconsistency table (C++ vs Python vs MATLAB)

| # | Topic | C++ | Python | MATLAB |
|---|-------|-----|--------|--------|
| 1 | Set k | `set_numberOfClusters(int)` (Problem.hpp:185) | `set_number_of_clusters(n_clusters)` (_dtwcpp_core.cpp:445) | `set_n_clusters(k)` (Problem.m:113) — three different names |
| 2 | Iter/rep config | fields `maxIter`, `N_repetition`, `band` | props `max_iter`, `n_repetition`, `band` | props `MaxIter`, `NRepetition`, `Band` (PascalCase, cached+synced) |
| 3 | Variant + params | `set_variant(DTWVariant)` or `set_variant(DTWVariantParams)` | `set_variant(enum)` only; params via `variant_params` struct field | `set_variant('wdtw', 0.1)` — string + one positional scalar; no struct |
| 4 | Data model | ragged `vector<vector<double>>` + names + `ndim` + f32 + view modes | ragged list-of-lists + names; `Data.ndim` bound; no f32/view | rectangular N×L matrix only; no names (auto "0..N-1"); no ndim (dtwc_mex.cpp:97-110,302) |
| 5 | Distance matrix I/O | `dense_distance_matrix()`, `writeDistanceMatrix()`, `readDistanceMatrix(path)` | `distance_matrix_numpy()` (fills + copies), `set_distance_matrix_from_numpy(dm)`; `DenseDistanceMatrix.to_numpy()` copy | `get_distance_matrix()`, `set_distance_matrix(D)` — three naming schemes for the same 2 ops |
| 6 | Pairwise matrix helper | none at top level (Problem or `core::compute_distance_matrix_pruned`) | `compute_distance_matrix(series, band, metric, use_pruning, device=)` | `compute_distance_matrix(X, 'Band', b)` — no metric/pruning/device |
| 7 | Default scalar type | templates default `T=float` (settings.hpp:29) while storage `data_t=double` | always double | always double |
| 8 | Storage precision default | `Data.precision=Float64` (Data.hpp:38); storage.hpp:21 comment claims Float32 default | f64 only | f64 only; CLI defaults **float32** (dtwc_cl.cpp:148) — 4 answers to "what precision do I get" |
| 9 | MIP | `MIPSettings` incl. `max_benders_iter`, `benders`; `Problem::set_solver` | `MIPSettings` **without** benders fields; **no `set_solver` binding** (Solver enum bound but unusable on Problem) | no MIP surface at all (Phase 2 pending, TODO.md:105) |
| 10 | Result write-back | `fast_pam/fast_clara/clarans` do NOT mutate Problem | wrappers auto-wire labels/medoids/k back into prob (_dtwcpp_core.cpp:573-576) | same auto-wire (dtwc_mex.cpp:647) — C++ behaves differently from both bindings |
| 11 | Indexing | 0-based | 0-based | 1-based (converted at MEX boundary) |
| 12 | Scores naming | `scores::daviesBouldinIndex` (camelCase) | `davies_bouldin_index` | `davies_bouldin_index` — C++ is the odd one out |
| 13 | Metric selection | `core::MetricType` enum (L2 silently = L1, TODO.md:64) | strings 'l1'/'squared_euclidean' | strings via 'Metric' NV pair |
| 14 | High-level clustering | none (CLI plays the role) | `device()/load()/cluster()/plot()` + sklearn `DTWClustering(device=)` | `DTWClustering` (no device, has Metric which Python lacks) |
| 15 | GPU/backend control | `distance_strategy` + `cuda_settings` + `lb_strategy` + `KernelOverride` | `distance_strategy` prop only; direct `compute_distance_matrix_cuda/metal(...)` with kwargs | strings via `set_distance_strategy('cuda'/'metal')` only |
| 16 | Data loading | `DataLoader` builder (CSV/TSV); CLI adds Parquet/Arrow/.dtws separately | no DataLoader; numpy loadtxt (`Dataset`) + io.py CSV/HDF5/Parquet | none — user passes matrix |
| 17 | Checkpointing | `save/load_checkpoint`, `save/load_binary_checkpoint` | `save_checkpoint/load_checkpoint` + `CheckpointOptions` | none |
| 18 | Missing+variant combo | `distance::dtw(params,...)` throws unless Standard | dispatcher raises unless variant='standard' | dispatcher errors unless 'standard' — consistent, all three restrict |
| 19 | soft_dtw band | `soft_dtw(x,y,gamma)` — no band param | `soft_dtw(x,y,gamma)`; dispatcher silently drops `band` | `'Gamma'` NV pair — consistent absence, silent drop in dispatchers |
| 20 | Metric-fn arg types (Python only) | — | `dtw/missing/arow` take ndarray (zero-copy); `ddtw/wdtw/adtw/soft` take lists (copy) (_dtwcpp_core.cpp:291-334) | uniform double matrices |

---

## 6. Object model & warts

Shape: `DataLoader --load()--> Data --set_data()--> Problem <-- free algorithms(prob, opts) -->
ClusteringResult`, with `scores::*(prob)` reading `prob.clusters_ind/centroids_ind`.

Warts (all [confirmed] at the cited lines):
1. **Problem is a god object** — data storage, dtw-function binding + weights cache, distance
   matrix (variant Dense/Mmap), clustering method + its 3 algorithm entry points, GPU + MIP +
   storage + LB config, result state, and CSV output all on one class (Problem.hpp:96-272).
2. **`init_fun: std::function<void(Problem&)>`** mutable-callback initialisation
   (Problem.hpp:143) — not exposed in any binding; only `init::random` default used.
3. **Configuration via naked public fields** (band, maxIter, variant_params, ...) but some
   fields require a side-effectful setter to take effect (`set_variant` must run
   `rebind_dtw_fn`; writing `variant_params` directly de-syncs the bound `dtw_fn_`). Python
   binds `variant_params` as rw — a user can set it without rebinding. [inferred: de-sync risk
   from Problem.hpp:117 + def_rw at _dtwcpp_core.cpp:426; confirmed mechanism, not reproduced.]
4. **Builder-pattern DataLoader** with arity-overloaded getter/setters (DataLoader.hpp:46-110)
   — CSV-only, duplicated by the CLI's own multi-format loading and by Python's numpy path.
5. **Data's 4 modes** (heap/view × f64/f32) with mode-dependent accessor validity and public
   `p_vec` that is silently wrong in 3 of 4 modes; `Problem::get_name/p_vec` assert on views
   (Problem.hpp:165-170).
6. **Storage policy is declared but not honored end-to-end**: `Problem.storage_policy` exists
   (Problem.hpp:138) but the CLI copies .dtws/Arrow mmap data into heap vectors ("integration
   into Problem is a future step", dtwc_cl.cpp:495, 523).
7. Global mutable state (`randGenerator`, `settings::paths`) — hidden coupling for bindings.
8. Mixed naming on the same class + camelCase C++ scores vs snake_case everywhere else.

---

## 7. Transformation plan v3 status

**The plan file `.claude/reports/twinkly-scribbling-shamir.md` no longer exists.** The whole
`.claude/reports/` content (8 files) was deleted as "obsolete session artifacts" and gitignored
— CHANGELOG.md:285 [confirmed]. No copy found in the repo or the pouch-cell-spectral archive
(searched 2026-07-06). Phase status below is reconstructed from CHANGELOG (Unreleased) and
TODO.md:151-165 (itself reconciled 2026-07-06), so phase *names* are as recorded there, not from
the original plan text.

DONE (per CHANGELOG Unreleased + TODO "Completed"):
- Phase 0 — CPU throughput (-march=native, Lemire envelope) [TODO.md:164].
- Phase 1/2 — unified DTW kernel family; warping_missing fold; 1.54-2.83x banded speedup
  [CHANGELOG:72-85; TODO.md:161].
- Phase 2 (CPU perf) — adtwBanded rolling column + early abandon [TODO.md:163].
- Phase 3 (4 parts) — templated `resolve_dtw_fn`, AROW fold, Soft-DTW fold, MV AROW; f32
  silent-dispatch bug fixed [CHANGELOG:86-136; TODO.md:160].
- Phase 4 — standalone soft_dtw/AROW API folds [CHANGELOG:153-169]; data access + I/O + f32
  (span accessors, StoragePolicy, Arrow/Parquet readers, dtwc-convert) [TODO.md:162].
- GPU: CUDA + Metal backends, LB_Keogh pruning, kernel overrides, naming unification
  (breaking, pre-v2.0.0) [CHANGELOG:171-261].
- Device API: `device()/load()/cluster()/plot()`, `gpu`/`hpc` names, SLURM offload
  [CHANGELOG:11-42].
- Python wheel build unblocked (ninja propagation patch); wheel OpenMP regression fixed
  [CHANGELOG:60-62, 138-151].

NOT DONE (open, per TODO.md):
- PyPI 2.0 release (needs GitHub trusted publisher) [TODO.md:104].
- MATLAB Phase 2: MIPSettings, CUDA dispatch, checkpointing [TODO.md:105].
- `device="hpc"` end-to-end on real ARC cluster — BLOCKED, unverified [TODO.md:139].
- 2026-06-01 audit backlog: 7 critical + high findings all unfixed (CUDA >2048-length silent
  wrong results, Metal int32/FP32-sqrt overflows, mmap validation, MEX mxIsDouble guard, MIP
  assert-only status check, MetricType::L2==L1, fast_pam O(N²k) swap loop) [TODO.md:43-79].
- Config-file loading outside the CLI (shared settings representation) [design.md:62-64].
- UNIMODULAR/MIP solver research; docs website updates; `default_data_t=float` decision OPEN.

---

## 8. Redesign constraints — what any new API must preserve

1. **File formats read**: CSV/TSV (DataLoader semantics: start_row/start_col/delimiter/Ndata,
   folder-of-files mode), Parquet file+directory with `--column` override, Arrow IPC
   (.arrow/.ipc/.feather), `.dtws` mmap binary + `.names` sidecar (dtwc_cl.cpp:471-555);
   Python-side CSV/HDF5/Parquet incl. Polars `large_list<float>` ragged layout (CHANGELOG:68).
2. **Output contract**: `<name>_labels.csv` ("name,cluster"), `<name>_medoids.csv`,
   `<name>_distance_matrix.csv`, `<name>_silhouettes.csv`, `<name>_checkpoint.bin`,
   `<name>_distmat.cache` — the SLURM/hpc path machine-parses `<name>_labels.csv` and maps
   1-based lexically-sorted rows back to input order (_hpc.py:45; CHANGELOG:37).
3. **CLI flag set + TOML/YAML keys** (kebab-case) — `cluster_generic.slurm` and
   `_hpc.build_dtwc_command` compose `dtwc_cl` command lines; changing flags breaks the hpc
   device path.
4. **Checkpoint/resume**: directory checkpoint (distances.csv + metadata.txt), binary result
   checkpoint, mmap distance-matrix cache with `--resume` (dtwc_cl.cpp:588-603, 839-843) —
   long SLURM runs depend on all three.
5. **Precision contract**: distance matrix and returned distances are always double, even with
   f32 series storage (storage.hpp:19, Problem.hpp:93-94).
6. **Zero-copy / performance paths**: nanobind ndarray zero-copy + GIL release on every long
   call; Data view-mode spans for CLARA subsampling (48x, CHANGELOG:332); OpenMP flags must be
   linked into the extension (`project_options` — regression documented CHANGELOG:60-62);
   interleaved MV layout; lock-free row-partitioned matrix fill.
7. **Determinism**: fixed seeds (mt19937(29) global; CLARA/CLARANS seed=42 defaults);
   scoring functions read state from Problem, so the auto-wire of results into
   `clusters_ind/centroids_ind` after fast_pam/fast_clara/clarans must survive (bindings rely
   on it; documented in docstrings).
8. **MATLAB 1-based index conversion at the MEX boundary** and rectangular-matrix input
   (existing user scripts); optional deps stay optional (core builds without OpenMP/HiGHS/
   CUDA/Arrow — .claude/CLAUDE.md non-negotiable 3).
