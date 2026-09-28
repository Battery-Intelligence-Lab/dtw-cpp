# DTWC++ 2.0 user surface — interface design (read-only pass, 2026-09-27)

## 0. Verdict

The library already has the right skeleton: one `Config` keyed by the CLI long names, one `run(Config)` pipeline, one `Name<E>` table per enum, `Problem` as the Tier-2 session, and `device → load → cluster → Result` in three languages. What is wrong is everything built *beside* that skeleton: a second method enum, a second device vocabulary on `Problem` (`DistanceMatrixStrategy` + `CUDASettings`), a second dispatch and a second validator in Python, a second reader in MATLAB, four ways to persist one matrix, and deprecation shims for names nobody was ever given. The design below keeps the skeleton, deletes the second copies, and changes exactly two things users can see that are not deletions: `Method` becomes the one nine-value enum, and every count/index becomes `dtwc::index_t` (int64).

The five-line happy path is the same in every language, the run description is the same 44 keys everywhere, and Tier-2 shrinks to one `Problem` with private distance semantics plus seven algorithm functions.

## 1. What real users have (the v1.0.0 constraint, observed)

| Surface | v1.0.0 (observed via `git show v1.0.0:…`) | Consequence |
|---|---|---|
| C++ `Problem` | public fields `method, maxIter, N_repetition, band, init_fun, output_folder, name, data, clusters_ind, centroids_ind` (`std::vector<int>`); methods `size, cluster_size, get_name, p_vec, refreshDistanceMatrix, resize, centroid_of, readDistanceMatrix, set_numberOfClusters, set_clusters, set_solver, set_data, maxDistance, distByInd, isDistanceMatrixFilled, fillDistanceMatrix, printDistanceMatrix, writeDistanceMatrix, printClusters, writeClusters, writeMedoidMembers, writeSilhouettes, init, cluster, cluster_by_MIP, cluster_by_kMedoidsPAM, cluster_and_process, findTotalCost, assignClusters, calculateMedoids` | camelCase shims stay one release; `cluster_by_kMedoidsPAM` needs a shim (missing at HEAD, the only v1 method name without one); `clusters_ind`/`centroids_ind` type change is the one deliberate BREAK (§7) |
| C++ `DataLoader` | builder `startColumn, startRow, n_data, delimiter, path, verbosity, load()`; `Data{p_vec, p_names, size()->int}`; `Method{Kmedoids, MIP}`, `Solver{Gurobi, HiGHS}`; `scores::silhouette`; `init::random/Kmeanspp`; `dtwBanded/dtwFull/dtwFull_L`; `Range`, `Index` | all kept; `load()` reverts to the plain heap load |
| CLI `dtwc_cl` | `--Nc/--clusters/--number_of_clusters` (single or `i..j`), `--name/--probName`, `-i/--in/--input`, `-o/--out/--output`, `--skipRows`, `--skipCols/--skipColumns`, `--maxIter/--iter`, `--method kMedoids|MIP`, `--repeat/--Nrepeat/--Nrepetition/--Nrep`, `--solver/--mip_solver/--mipSolver`, `--bandwidth/--bandw/--bandlength`, `--distMat/--distance_matrix/--distances` | every spelling except `--clusters` is gone at HEAD and undocumented (records-docs silent bug 1). Restore all as hidden warn-once aliases, table-driven (~25 lines); `i..j` errors with a hint |
| Python | pybind11 `dtwcpp.Problem` with the camelCase names above (`cluster_size()` a method) + `DataLoader(path[, ndata])`; nothing else; never on PyPI (assumed) | owner decision D-A (§9) |
| MATLAB | none | the whole MATLAB surface is PRE-TAG |

## 2. The happy path (five lines, same shape everywhere)

**C++**
```cpp
#include <dtwc.hpp>
dtwc::device("gpu");                                   // cpu | gpu | gpu:N ; hpc is CLI/Python only (D-10)
auto data = dtwc::load("Crop.tsv", {.skip_cols = 1});  // lazy: nothing is read here
auto res  = dtwc::cluster(data, 3, {.band = 10});      // any Config key; method auto, seed 42
res.save("out/");                                      // <name>_labels/_medoids/_distance_matrix/_silhouettes.csv
std::cout << res.score("silhouette") << ' ' << res.cost() << '\n';
```
**Python**
```python
import dtwcpp as dtwc
dtwc.device("gpu")                                     # cpu | gpu | gpu:N | hpc | hpc:gpu
data = dtwc.load("Crop.tsv", skip_cols=1)
res  = dtwc.cluster(data, k=3, band=10)                # **kw are Config keys (snake_case)
res.save("out/"); res.score("silhouette"); res.plot()
```
**MATLAB**
```matlab
dtwc.device('gpu');
data = dtwc.load('Crop.tsv', 'skip_cols', 1);
res  = dtwc.cluster(data, 3, 'band', 10);              % name-value keys == Config keys (snake_case)
res.save('out'); res.score('silhouette'); res.plot();
```
**CLI and config file (the same 44 keys)**
```sh
dtwc_cl -i Crop.tsv --skip-cols 1 -k 3 --band 10 --device gpu -o out
dtwc_cl --config job.toml            # flags beat the file; --print-config writes one
```
```toml
input = "Crop.tsv"
skip-cols = 1
n-clusters = 3
band = 10
device = "gpu"
output = "out"
```

Three defaults are inconsistent today and become one: `method` defaults to `auto` everywhere (today `pam` in C++/Python/MATLAB `cluster()`, `auto` in the CLI); `k` is required everywhere (today the CLI silently uses 3); `name` defaults to the input stem everywhere (today `dtwc` in the CLI, stem in the API).

## 3. The one run description: `Config`

Keep `dtwc::Config` exactly as the mechanism it is (aggregate, member-initialiser defaults, `cli::bind` the only key table, `to_config_text`/`parse_config`, TOML/YAML through CLI11). Change its contents:

```cpp
struct Config {                       // dtwc/cli/config.hpp; key = CLI long name without "--"
  // input / output
  std::string input, output = "./results", name, column;
  index_t skip_rows = 0, skip_cols = 0;  char delimiter = 0;
  Precision dtype = Float64;  std::size_t ram_limit = 0;  std::size_t mmap_threshold = 50000;
  std::string dist_matrix, checkpoint;  int checkpoint_interval = 0;
  // what to compute
  Method method = Method::Auto;  index_t k = 0 /*required*/;  int max_iter = 100, n_init = 1;
  std::uint64_t seed = 42;  int sample_size = -1, n_samples = 5, batch_size = -1;
  Linkage linkage = Average;  double dc = -1;
  // what a distance means  (one struct, shared with distance::dtw and Problem::set_distance)
  DistanceConfig distance;            // variant + params, metric, missing_strategy, mv_mode, band
  // where
  Device device = env().device();     // the process device is the default; a Config never has an "unset" device
  int gpu_index = env().device_index();
  GpuPrecision gpu_precision = Auto;  // auto | float32 | float64
  // solver
  Solver solver = HiGHS;  MIPSettings mip;   // mip_gap, time_limit, warm_start, numeric_focus, mip_focus, verbose_solver, lr_max_nodes
  bool verbose = false;
};
```
Removed keys (all 2.0-only): `batch-weighting`, `benders`, `max-benders-iter`, `resume` (`--restart`), `gpu-dtype`/`data-precision`/`data-type` aliases, `use_pruning`. `checkpoint` + `checkpoint-interval` keep their names but mean the one binary matrix file (data-io-04): the matrix persists in `<checkpoint>/<name>.dtwm`, mapped when `N >= mmap_threshold`, else saved every `checkpoint_interval` rows, resumed on the next identical run; a wrong identity is `InvalidInput`, a corrupt file `IOError`, never a silent restart.

How each language spells it:

| Language | Spelling | Mechanism |
|---|---|---|
| CLI | `--band 10`, `--config job.toml`, `--print-config` | `cli::bind` |
| TOML / YAML | `band = 10` | CLI11 config reader (exists) |
| C++ | `dtwc::cluster(data, k, {.band = 10})`, or `Config c; c.distance.band = 10; run(c)` | designated initialisers on the aggregate |
| Python | `dtwc.cluster(data, k, band=10)`, `dtwc.run(input="x.csv", k=3, band=10)` | kwargs → `[(key, str(value))]` → `parse_config` in C++ (IF-2 S4). No nanobind `Config` class with 44 `def_rw`: the CLI's parser is the parser, so typos and errors are identical to the CLI's |
| MATLAB | `dtwc.cluster(data, 3, 'band', 10)` | the same pairs through the MEX `run` command |
| hpc | `job.toml` written by `to_config_text` and shipped | §4 |

Two rules make this hold: a value of a run that is not a `Config` field does not exist (a callable `init_fun` is Tier-2 only and not submittable), and every enum is spelled by its one `Name<E>` table (canonical name first; aliases trimmed to at most one widely used synonym per value: `sqeuclidean`, `f32`/`f64`; the `obp`/`lr`/`hclust`, `l2sq`, `float`/`double`/`fp32`… spellings go).

`Result` (every language): `labels`, `medoids`, `cost`, `method` (auto resolved), `iterations`, `converged`, `device`, `score(name)`, `save(dir)`, `distance_matrix` (N×N, filled on demand), `config` (the resolved run, what `--print-config` prints); Python/MATLAB add `plot()`. It holds labels/medoids/cost **by value** (IF-4) and keeps the session only for the lazy matrix. Python drops `summary()`, `elapsed_s`, `k`, `n_series`, `medoid_indices`.

## 4. Device: what the user says, what the library decides

Grammar (one function, `detail::parse_device`, used by C++/Python/MATLAB/CLI): `cpu | gpu | gpu:N | hpc | hpc:gpu` (+ `cuda`, `cuda:N` as aliases of `gpu`, contract §6.1, cheap). `dtwc::device(name)` sets the process default and returns the canonical name; `Config.device` is initialised from it; `Problem(name, device=…)` / `Problem::set_device` is the Tier-2 spelling. `device("hpc")` only records the selection — no `.env` read, no ssh, in any language (today C++/MATLAB `Env::select_hpc` runs ssh inside a setter; that copy goes).

| device | series storage | distance matrix | precision | `auto` method | refused loudly (typed error, before I/O) |
|---|---|---|---|---|---|
| `cpu` | heap, `dtype` float64/float32. Data larger than RAM enters as list-per-row Parquet with `ram_limit`: FastCLARA streams row groups (the one genuine large-N path). No auto-spill of a heap load to a temp `.dtws` (it copied after a full load and leaked the file) | dense packed doubles below `mmap_threshold`; above it the `.dtwm` mapped file (llfio build) or `IOError` naming `--mmap-threshold`, `onebatch`, `clara` | CPU kernels in `dtype` | `pam` for N ≤ 5000, else `clara` (one rule, in `run()`) | `ram_limit` on a format that cannot stream; `checkpoint` with streaming |
| `gpu`, `gpu:N` | heap float64 only | GPU fills the packed matrix (CUDA, else Metal, else `DeviceError` naming the CMake flag) | `gpu_precision auto` = float32 when the device's FP64 rate is < ½ FP32 (one attribute query), else float64; Metal float32 only | `pam` | non-standard variant, missing strategy, ndim > 1, float32 storage, matrix-free methods (`onebatch`, `tadpole`, `clara` with a partial sample), Metal with N ≠ 0 or float64. No CPU fallback; the GPU never returns an all-zero matrix (Metal throws) |
| `hpc`, `hpc:gpu` | never read locally: a path is forwarded; in-memory series are written to TSV and uploaded | remote | remote | forwarded (`auto` resolves on the cluster) | credentials checked **at `cluster()`**, in this order, each a `DeviceError` with the fix in the message: (1) no `.env` in the working directory or `$DTWC_REPO_ROOT` → shows the three required keys with an example; (2) a missing key → names it; (3) `ssh -o BatchMode=yes user@host` fails → "must succeed without a password prompt; add your key to ssh-agent / authorized_keys"; (4) no `dtwc_cl` on the cluster → "run `slurm_remote.sh build` once"; (5) the job failed → the tail of its stderr. C++ and MATLAB refuse `hpc` with the one message that names Python and the wrapper (D-10) |

The hpc transport is the config file (IF-2 S4): Python writes `job.toml` = `to_config_text(config)` with `device = cpu` (or `gpu` for `hpc:gpu`, which selects the GPU partition from `.env`), uploads it beside the input, and the job runs `dtwc_cl --config job.toml -i … -o …`. This deletes the 20-argument bash/sbatch/Python triple validation, and `skip_rows`, `delimiter`, `metric` etc. reach the remote run for free. The result is labels + medoids (`cost`, `distance_matrix` are `None`; `score()` raises `InvalidInput` saying so).

## 5. Tier-2 escape hatch (minimal)

**`Problem`** stays the v1 session. One rule decides what is a field and what is a setter: *a value that changes what a stored distance means is private and set through a validating setter that invalidates the matrix; everything else is a plain field checked when used.*

```cpp
class Problem {
public:
  Problem(std::string_view name = "");  Problem(std::string_view name, DataLoader&);   // v1
  // v1 plain fields (stay public, [[deprecated]] dropped from maxIter/N_repetition)
  Method method;  int maxIter, N_repetition;  int band;  std::function<void(Problem&)> init_fun;
  std::vector<index_t> clusters_ind, centroids_ind;  MIPSettings mip_settings;  CheckpointOptions checkpoint;
  // data
  void set_data(Data);  index_t size() const;  std::span<const data_t> series(index_t) const;  std::string_view series_name(index_t) const;
  // distance semantics: private, one struct, one validator, bound once
  void set_distance(const DistanceConfig&);  const DistanceConfig& distance() const;
  void set_band(int);  void set_metric(MetricType);  void set_variant(DTWVariant, /*params*/);  void set_missing_strategy(MissingStrategy);
  void set_device(Device, int index = 0);  void set_gpu_precision(GpuPrecision);  Device device() const;
  // matrix
  void fill_distance_matrix();  data_t dist_by_ind(index_t, index_t);  bool is_distance_matrix_filled() const;  data_t max_distance() const;
  const distMat_t& distance_matrix() const;  void set_distance_matrix(std::vector<double> nxn);
  void read_distance_matrix(path);  void write_distance_matrix(name) const;  void use_mmap_distance_matrix(path);  void refresh_distance_matrix();
  // clustering (v1 names) — cluster() dispatches every Method through set_result()
  void set_n_clusters(index_t);  index_t n_clusters() const;  bool set_solver(Solver);  void set_max_iter(int);  void set_n_repetitions(int);  void set_random_seed(std::uint64_t);
  void set_sample_size(int); void set_n_samples(int); void set_batch_size(int); void set_linkage(Linkage); void set_tadpole_dc(double);   // the four knobs Problem lacks today
  void init();  void cluster();  void cluster_by_mip();  void cluster_by_kmedoids_lloyd();  void cluster_and_process();
  const std::vector<index_t>& labels() const;  const std::vector<index_t>& medoids() const;  double find_total_cost();  void assign_clusters();  void calculate_medoids();
  void set_result(const core::ClusteringResult&);          // the one write-back (replaces 8 copies)
  // writers (v1 names and 1.x filenames unchanged)
  void print_distance_matrix() const;  void print_clusters() const;  void write_clusters();  void write_silhouettes();  void write_medoid_members(int, int = 0) const;
  // 21 v1 camelCase [[deprecated]] forwarders + cluster_by_kMedoidsPAM (new) ; cluster_by_kMedoidsLloyd deleted (2.0-born)
};
```
Gone from `Problem` (all 2.0-only): `variant_params`, `missing_strategy`, `distance_strategy`, `cuda_settings` as public fields; `DistanceMatrixStrategy`, `LowerBoundStrategy`, `StoragePolicy`, `CUDASettings`, `set_lb_strategy`, `set_storage_policy`, `set_ram_limit`, `set_distance_strategy`, `set_cuda_settings`, `set_view_data`, `dtw_function_f32()`, `wdtw_weights_cache()`, `distance_checkpoint_identity()`, and the whole drift-detection layer (snapshot compare, relocation repair, double preflight on every `dist_by_ind`). `is_distance_matrix_filled()` is a bool again (v1 had one).

**One `Method`** (nine values: `Auto, PAM, OneBatch, CLARA, Kmedoids, MIP, LRCore, TADPole, Hierarchical`), one table, used by `Config`, `Problem::cluster()`, the CLI and both bindings; `ClusterMethod` and `problem_route` go. `Problem::cluster()` dispatches all nine (resolving `Auto` from `size()` and `device()`), and `run()` becomes "apply Config to Problem, load, `prob.cluster()`, write". Streaming-Parquet CLARA stays `run()`'s one special route (it passes the Parquet path through `CLARAOptions`). This overturns the 2026-09-24 "new `ClusterMethod` type" note; its reason ("a runtime refusal where a type serves") disappears once `Problem` holds the four knobs above — nothing is refused.

**Algorithms** (C++ Tier-2, Python and MATLAB as the same names): `fast_pam(prob, k, max_iter = 100, seed = 42)` (merges `fast_pam_seeded`; 2.0-born, so free), `fast_clara(prob, CLARAOptions)`, `one_batch_pam(prob, OneBatchPAMOptions)`, `tadpole(prob, k, dc)`, `build_dendrogram(prob, linkage)` + `cut_dendrogram(dend, prob, k)` (fills the matrix itself), `dtw_barycenter(...)`, `barycenter_kmeans(...)`. One seed type (`uint64`), one spelling `max_iter`. `clarans`, `pdlp_lp_bound`, `fast_pam_swap`, `one_batch_pam_with_stats` leave the public surface.

**Pairwise distance**, one vocabulary = the `Config` distance block: C++ `distance::dtw(x, y, DistanceConfig)` (checked once, then `resolve_dtw_fn` — the same resolver the fill binds, so the two cannot disagree) plus the per-variant one-liners as they are; Python `dtwc.distance.dtw(x, y, variant="msm", msm_c=0.5, band=-1, metric="l1", missing_strategy="error", mv_mode="dependent")` bound once (MSM/TWE finally reachable); MATLAB `dtwc.distance.dtw(x, y, 'variant', 'msm', 'msm_c', 0.5)`; `soft_dtw_gradient` separate. The unchecked `warping*.hpp` kernels stay public C++ for speed, documented as unchecked.

**Distance matrix**: keep the name `compute_distance_matrix(X, band=, metric=, variant=…, device=)` in Python and MATLAB, reimplemented as "build a `Problem`, `set_distance`, `set_device`, fill, return" — one engine, no private OpenMP loop, no `use_pruning`, and the EAP kernel for unbanded standard DTW for free.

**Estimator**: one `DTWClustering(n_clusters, n_init, **distance keys, device)` in Python (folding `DTWCKMedoids`' `metric='precomputed'`, `transform`, tags) and MATLAB (`'n_clusters'`, snake_case). `fit` = one `run(config, data)` (n_init is a Config key; the matrix is computed once), `predict` = nearest medoid via the one distance binding, `score(X)` does not refit.

**Diagnostics**: one `system_check()` in C++/Python/MATLAB returning `{threads, threads_engaged (measured in a real parallel region), openmp, cuda, metal, highs, gurobi, arrow, llfio, gpu_name, gpu_validated, gpu_reason}`; replaces `check_system`, `system_info`, `dtwc::test::*`, `dtwcpp.test.*`, `+test/*`, `cuda_*`/`metal_*`/`*_AVAILABLE`. The CI wheel/MEX smoke asserts `threads_engaged >= 2` on it.

## 6. Old → new mapping (user-visible entry points)

Legend: v1 = shipped in v1.0.0; **shim** = `[[deprecated]]` forwarder / DeprecationWarning for one release; **delete** = pre-tag, never released.

**C++**
| Old | v1 | New / action |
|---|---|---|
| `Problem` camelCase methods (21) | yes | keep as shims (exist) |
| `Problem::cluster_by_kMedoidsPAM()` | yes | **add** shim → `cluster_by_kmedoids_lloyd()` |
| `Problem::cluster_by_kMedoidsLloyd()` | no | delete |
| `maxIter`, `N_repetition` `[[deprecated]]` | yes | plain fields; attribute and pragma blocks go |
| `clusters_ind`, `centroids_ind` `vector<int>` | yes | `vector<index_t>` (BREAK, §7); `set_clusters(vector<int>&)` kept as converting shim |
| `Data::size() -> int` (v1) / `size_t` (HEAD) | yes | `index_t` |
| `DataLoader::load/load_local/load_stored/load_metadata/count/storage_policy/ram_limit/mmap_cache_path` | `load` only | `load()` = heap load; the rest delete |
| `scores::daviesBouldinIndex, dunnIndex, calinskiHarabaszIndex, adjustedRandIndex, normalizedMutualInformation` | no | delete |
| `ClusterMethod`, `cluster_method_names` | no | `Method` (nine values), `method_names` |
| `DistanceMatrixStrategy`, `set_distance_strategy`, `CUDASettings`, `set_cuda_settings`, `LowerBoundStrategy`, `set_lb_strategy`, `StoragePolicy`, `set_storage_policy`, `set_ram_limit`, `set_view_data` | no | delete; `set_device(Device, index)`, `set_gpu_precision(GpuPrecision)` |
| `Env` class, `env()`, `AuthProbe`, `set_env_file_dir`, `Env::threads` | no | two free functions `device(name)` / `device()` over a static `{Device, index}` |
| `core::dtw_runtime`, `DTWOptions`, `ConstraintType`, `core::dtw_distance` | no | `distance::dtw(x, y, DistanceConfig)` |
| `MetricType::L2` | no | delete (no front end can name it) |
| `dtwc::load(path, skip_cols, skip_rows, delimiter, name)` + 2 `= delete` overloads | no | `load(path, LoadOptions{})` (designated init removes the char→int hazard) |
| `cluster(data, k, method, band, device, max_iter)` | no | `cluster(data, k, Config = {})`; `run(Config[, Data])` unchanged |
| `save_checkpoint/load_checkpoint(prob, dir)`, `CheckpointOptions` | no | keep, one binary format; `save/load_binary_checkpoint` delete |
| `dtwc::test::parallelisation/gpu` (`test_api.hpp`) | no | `system_check()` |
| `dtwc_main` executable, root forwarders `error/env/missing_utils/random_engine/system_memory.hpp` | no | delete (`parallelisation/settings/timing/utility.hpp` forwarders stay: v1 paths) |

**Python**
| Old | v1 | New / action |
|---|---|---|
| `cluster(data, k, *, method="pam", band, device, max_iter)` | no | `cluster(data, k, **config_keys)`; `_METHODS`, `_AUTO_PAM_SERIES_LIMIT`, `_run_local_method`, `_validate_common`, `Result.save` byte re-implementation → delete (one `run` binding) |
| `Result.medoid_indices`, `summary()`, `elapsed_s`, `k`, `n_series`, `ClusterResult`, `get_device` | no | delete |
| `Problem.variant_params`, `.missing_strategy`, `.distance_strategy`, `.cuda_settings`, `.lb_strategy`, `.storage_policy` (properties by reference) | no | `set_variant("wdtw", wdtw_g=0.1)`, `set_missing_strategy`, `set_device`, `set_gpu_precision`; read-only `variant`, `metric`, `device` strings |
| `Problem.n_clusters()` method / `cluster_size` property / `size` property / `labels()` method | mixed | properties for state (`size`, `n_clusters`, `labels`, `medoids`, `band`), methods for actions; `cluster_size()` stays a **method** shim (v1 shape) |
| `n_repetition`, `set_number_of_clusters`, `distance_matrix_numpy`, `set_distance_matrix_from_numpy`, `*_index`, `normalized_mutual_information`, `_F22_DEPRECATION_POLICY` | no | delete |
| `Env`, `env`, `device_to_string`, `Device` enum export, `ConstraintType`, `DistanceMatrixStrategy`, `StoragePolicy`, `LowerBoundStrategy`, `CUDASettings`, `DenseDistanceMatrix`, `OneBatchWeighting`, `OneBatchPAMStats`, `CLARANSOptions`, `clarans`, `pdlp_*`, `fast_pam_seeded`, `one_batch_pam_with_stats`, `compute_lb_keogh_cuda`, `save/load_binary_checkpoint`, `cuda_available/cuda_device_info/CUDA_AVAILABLE/metal_*/METAL_AVAILABLE/MPI_AVAILABLE/HIGHS_AVAILABLE/OPENMP_AVAILABLE/openmp_max_threads/system_info/check_system`, `test` | no | delete; `system_check()`; `GpuPrecision`, `Method` enums exported |
| `compute_distance_matrix(series, band, metric, use_pruning, *, device)` | no | `compute_distance_matrix(X, *, band=-1, metric="l1", variant="standard", …, device=None)` over `Problem` |
| `dtw_distance, ddtw_distance, …` (7 raw) + `distance.standard/ddtw/…` | no | one `distance.dtw(x, y, **keys)`; per-variant names as one-line aliases |
| `DTWCKMedoids`, `_variant_validation.py`, `_hpc.build_dtwc_command/find_dtwc_binary` + 20-arg transport | no | one `DTWClustering`; hpc by config file |
| `preprocess`, `diagnose`, `features` modules | no | delete (or an example) |
| `io.load_dataset_csv/parquet`, `save_dataset_csv/parquet`, `convert` `.dtws` writer | no | delete (`load(path)` reads CSV/TSV/folder/Parquet/Arrow via one C++ reader); HDF5 and `dtwc-convert` = owner (D-G) |
| v1 `Problem.set_numberOfClusters/fillDistanceMatrix/distByInd/refreshDistanceMatrix/isDistanceMatrixFilled/maxDistance/readDistanceMatrix/printDistanceMatrix/writeDistanceMatrix/writeClusters/writeMedoidMembers/writeSilhouettes/findTotalCost/assignClusters/calculateMedoids`, `DataLoader(path[, n])`, `Problem(name, DataLoader)` | **yes** | owner D-A: 15 one-line table-driven shims + a 5-line `DataLoader`, or a recorded break |

**MATLAB** (all PRE-TAG)
| Old | New |
|---|---|
| PascalCase keys (`'MaxIter'`, `'NClusters'`, `'Band'`, `'G'`, `'Penalty'`, `'Gamma'`) | snake_case Config keys everywhere (`'max_iter'`, `'n_clusters'`, `'band'`, `'wdtw_g'`, `'adtw_penalty'`, `'sdtw_gamma'`) |
| `Problem.Band/Verbose/MaxIter/NRepetition` properties + `*Value` caches, `Size/ClusterSize/Name/CentroidsInd/ClustersInd`, `get_distance_matrix` | canonical methods only; state read back from the MEX |
| `set_lb_strategy`, `set_storage_policy`, `set_distance_strategy('cuda'|'metal')`, `set/get_cuda_settings` | `set_device('gpu:1')`, `set_gpu_precision('float32')` |
| `*_index.m` (5), MEX commands named `*_index` | canonical names, MEX commands renamed |
| `dtwc.compute_distance_matrix(X, 'Band', b)` (no metric/device), `DTWClustering_compute_distance_matrix` MEX | `compute_distance_matrix(X, 'band', b, 'metric', m, 'device', d)` over `Problem` |
| `+distance/{standard,ddtw,wdtw,adtw,soft_dtw,missing,arow}.m` + `validate_metric.m` (rejects squared_euclidean) | `dtwc.distance.dtw(x, y, name-value)` = one MEX command; MSM/TWE included |
| `Dataset.materialize` (second CSV reader), `test_mex.m` on the path, MEX `cluster` legacy, `Problem_get_band/_get_cluster_size` | delete |
| `clarans.m`, `pdlp_*.m`, `save/load_binary_checkpoint.m`, `+test/*`, `check_system.m` | delete; `system_check` |
| `Result.plot` via a temp-dir save + `readmatrix` | MEX `Result_distance_matrix` |
| labels/medoids `int32` 1-based | `int64` 1-based |
| `fast_pam(prob, k, 'MaxIter', 'Seed')` | `fast_pam(prob, k, 'max_iter', 100, 'seed', 42)` |

**CLI**
| Old | New |
|---|---|
| `--batch-weighting`, `--benders`, `--max-benders-iter`, `--resume`, `--restart`, `--gpu-dtype`, `--data-precision`, `--data-type` | delete |
| `--gpu-precision auto|fp32|f32|float32|float|fp64|f64|float64|double` | `auto|float32|float64` (+ `f32`/`f64`), the `--dtype` spellings |
| `--method` aliases `obp`, `lr`, `hclust` | canonical only |
| `-k` default 3 | required |
| `--name` default `dtwc` | input stem |
| v1 `--Nc`, `--number_of_clusters`, `--probName`, `--in`, `--out`, `--skipRows`, `--skipCols`, `--skipColumns`, `--maxIter`, `--iter`, `--repeat`, `--Nrepeat`, `--Nrepetition`, `--Nrep`, `--mip_solver`, `--mipSolver`, `--bandwidth`, `--bandw`, `--bandlength`, `--distMat`, `--distance_matrix`, `--distances` | hidden warn-once aliases (table, the existing `--clusters` mechanism); `--Nc 3..5` → error "one k per run; loop in the shell" |

## 7. Integers under "one compile-time integer, default 64-bit"

```cpp
// dtwc/base/settings.hpp
#ifndef DTWC_INDEX_T
#define DTWC_INDEX_T std::int64_t
#endif
using index_t = DTWC_INDEX_T;   // every count of series or clusters, every label, every medoid index; signed
```
Rules: (1) counts and indices are `index_t` (`Data::size()`, `Problem::size()/n_clusters()/Nc`, `k`, `clusters_ind`, `centroids_ind`, `ClusteringResult::labels/medoid_indices`, `dist_by_ind(index_t, index_t)`, `centroid_of`, `skip_rows/skip_cols`, algorithm loops — MSVC's `/openmp:experimental` takes a signed 64-bit loop variable); (2) products stay 64-bit by type as today (pair index, packed offset, `N·L`, bytes: `int64_t`/`size_t`); (3) tuning parameters that are not indices stay `int` (`band`, `max_iter`, `n_init`, `time_limit`) — they are values, not sizes; (4) no count guard anywhere: `checked_parquet_series_count`, `saturating_add`, `run_openmp`'s INT_MAX throw, the Lloyd seed-overflow throw, FastCLARA's N > INT_MAX check, the DCKP int32 wire limit all go; (5) two third-party 32-bit boundaries remain, handled by the rule "chunk where we own the loop, check once where we do not": CUDA launches chunk on an `int64` pair offset (Metal already does), and the MIP model build keeps the single `index_guard` check (HiGHS `HighsInt` / Gurobi `int`; a plain cast builds a silently wrong model at N ≥ 26.8k nonzeros-wise). MPI goes.

Per language: C++ `index_t`; Python `np.int64` arrays (numpy ≥ 2's default int on every platform); MATLAB `int64`, 1-based (a MATLAB double is exact to 2^53, so `get_exact_int` on the way in is a type check, not a count guard).

The v1.0.0 `std::vector<int>` fields: they become `std::vector<index_t>`. Reads (`int c = prob.clusters_ind[i]`) still compile; only *assignment from* a `std::vector<int>` and `set_clusters(std::vector<int>&)` break. Justification: R3 — one index type across C++/Python/MATLAB has no additive route (two label types would reintroduce a narrowing at every `labels[i] = medoid`) and the owner's 2026-09-27 instruction names exactly this. Mitigation: keep `set_clusters(std::vector<int>)` as a `[[deprecated]]` converting overload for one release; the break register entry says "replace `std::vector<int>` with `std::vector<dtwc::index_t>` (int64)". PLAN §1.1 and design A19/§9 are rewritten to this rule.

## 8. Abstractions rejected (they do not pay for themselves)

- A second method enum (`ClusterMethod`) beside `Method`: one table serves.
- `DistanceMatrixStrategy`, `CUDASettings`/`GPUSettings`: `set_device(device, index)` + one `GpuPrecision` say the same thing; the cache fingerprint hashes the resolved backend and precision.
- `StoragePolicy`/`LoadedData`/`route_series_storage`/`available_ram_bytes`: it copies after a full heap load and cannot lower peak RAM; Parquet streaming plus the mapped matrix is the honest "too big" story.
- `Env` as a class with an injectable ssh probe: credentials belong to the submit step, where the message can name the fix.
- A nanobind `Config` class (44 `def_rw`) or a Python `Result`/`Config` dataclass mirror: kwargs → config text → the CLI's parser is one code path with identical errors.
- Python-side method dispatch, `auto` rule, semantic validation tables, CSV byte re-implementation; MATLAB `readmatrix` reader; MEX string→enum parsers: C++ owns all of it once.
- Four persistence formats: one binary matrix file (`checkpoint` dir; mapped or saved) plus CSV interchange; the `--resume` replay of a *finished* result and the `.dtws` series format go.
- `RunStats` as a public struct, a general progress-callback API: `Result.method/iterations/converged/device` plus `--verbose` cover users; Python keeps only `PyErr_CheckSignals` polling per block so Ctrl-C works during a GIL-released fill.
- Deprecation aliases for 2.0-born names (Python 12, MATLAB 15, C++ 6): a shim protects users, and these names had none.
- `Dataset` is kept: it is the `load` noun that makes "hpc never reads" true in three languages (30 lines); `materialize_local` stops copying (rvalue overload).

## 9. Decisions for the owner

- **D-A** v1.0.0 Python camelCase `Problem` names + `DataLoader`: bind ~20 one-line shims (recommended if any v1 wheel was ever handed to a user), else record the Python break and correct `design.md:41`.
- **D-B** `Method` becomes the nine-value enum and `Problem::cluster()` dispatches all nine (overturns the 2026-09-24 `ClusterMethod` note; recommended).
- **D-C** Delete `DistanceMatrixStrategy`, `CUDASettings`, `set_distance_strategy`, `set_cuda_settings`, `MetricType::L2` (contract §2.1/§6.4 name them → dated `DECISIONS.md` entry; recommended).
- **D-D** `hpc:gpu` in the device grammar for the remote GPU partition (recommended; the alternative is a `.env`-only choice).
- **D-E** `clusters_ind`/`centroids_ind` → `std::vector<index_t>` with the converting `set_clusters` shim (recommended; the one v1 BREAK).
- **D-F** Restore the v1 CLI spellings as hidden warn-once aliases (recommended, ~25 lines) rather than documenting them as removed.
- **D-G** HDF5 wrappers and `dtwc-convert` (`.dtws` writer): delete or keep in an example; project memory records "HDF5 + CSV" as the format preference, so this is yours.
- **D-H** `DEFAULT_RANDOM_SEED` 42 stays; `fast_pam`/`fast_pam_seeded` merge (contract §1.3 pins the unseeded engine → `DECISIONS.md` entry).

## 10. Evidence (files read; line numbers as at HEAD cd5d449)

- Tier-1 and Config: `C:\D\git\dtw-cpp\dtwc\api.hpp` (load :62-75 with the two `= delete` overloads; `cluster` :125 default `"pam"`), `C:\D\git\dtw-cpp\dtwc\cli\config.hpp` (`ClusterMethod` :41, `gpu_precision_names` 9 spellings :59-63, `Config` :65-108, `Config::method = Auto` :78, `k = 3` :79, `name = "dtwc"` :106), `C:\D\git\dtw-cpp\dtwc\cli\config.cpp` :237-380 (the key table; `--clusters`/`--restart` hidden-alias mechanism :253-261, :319-326).
- run pipeline: `C:\D\git\dtw-cpp\dtwc\cli\run.hpp` (method × device table :11-18), `C:\D\git\dtw-cpp\dtwc\cli\run.cpp` (`auto_pam_max_series = 5000` :60; `resolve_method` :66-70; hpc refusal :446-450; storage follows device :565; dispatch :690-745; n_init restart loop :677).
- Problem: `C:\D\git\dtw-cpp\dtwc\Problem.hpp` (`CUDASettings` :50-63, `DistanceMatrixStrategy` :95-114, drift machinery :154-263, public fields :347-366, `set_device` doc :546-554, shims throughout); `C:\D\git\dtw-cpp\dtwc\Problem.cpp` :462-497 (`set_device` maps to strategy), :1387-1410 (`cluster()` dispatches four).
- Device: `C:\D\git\dtw-cpp\dtwc\base\env.hpp` (`Env` class, `AuthProbe`, `.env` messages), `C:\D\git\dtw-cpp\dtwc\base\env.cpp` :214-233 (`parse_device` grammar), :308-325 (`select_hpc` runs ssh in the setter), :77-101 (the three `.env` messages).
- Python: `C:\D\git\dtw-cpp\python\dtwcpp\_api.py` (`_METHODS` :374, `_AUTO_PAM_SERIES_LIMIT` :376, `_run_local_method` :449-500, hand-written `Result.save` :255-328, `method="pam"` :503), `C:\D\git\dtw-cpp\python\dtwcpp\__init__.py` (exports; `use_pruning=True` :208; `_HPC_SELECTED` :162), `C:\D\git\dtw-cpp\python\dtwcpp\_clustering.py` (`_validate_semantics` :130-208, Problem-per-pair :249-261, restart loop :387-397, `score` refits :477-483), `C:\D\git\dtw-cpp\python\dtwcpp\sklearn.py`, `C:\D\git\dtw-cpp\python\dtwcpp\_hpc.py` :435-482, :549-640 (20-argument transport), `C:\D\git\dtw-cpp\python\src\_dtwcpp_core.cpp` (bound names; `n_clusters` method :1024 vs `cluster_size` property :1025; `variant_params` by reference :970).
- MATLAB: `C:\D\git\dtw-cpp\bindings\matlab\+dtwc\Problem.m` (PascalCase caches :21-42, aliases :102-144, :414-448), `DTWClustering.m` (`'NClusters'` :549; CPU matrix for non-L1 :609-613; Problem per restart :624-663), `Dataset.m` :776-832 (`readmatrix` reader), `Result.m` :959-975 (temp-dir save + `readmatrix`), `cluster.m` (snake_case keys, one gateway call), `fast_pam.m` (`'MaxIter'`, `'Seed'`), `+distance\dtw.m`, `dtwc_mex.cpp` :1497-1533 (`tier1_cluster`), :1656-1760 (command table), :235-240 (int32 1-based output).
- v1.0.0: `git show v1.0.0:dtwc/Problem.hpp` (public fields and camelCase methods, `cluster_by_kMedoidsPAM`), `git show v1.0.0:python/py_main.cpp` (camelCase pybind names, `cluster_size` a method, `DataLoader` ctors), `git show v1.0.0:dtwc/dtwc_cl.cpp` :43-54 (the flag spellings, `--Nc i..j`), `git show v1.0.0:dtwc/Data.hpp` (`size() -> int`), `git show v1.0.0:dtwc/DataLoader.hpp` (`load()` heap only), `git show v1.0.0:dtwc/scores.hpp` (`silhouette` only).
- Records: `C:\D\git\dtw-cpp\.claude\CHARTER.md` :90-101 (device=cpu|gpu|hpc, `.env` messages, lazy load), `C:\D\git\dtw-cpp\.claude\design.md` §2-§6, `C:\D\git\dtw-cpp\.claude\PLAN.md` §1.1 :45-58 (the `int` rule to rewrite), §3.3 IF-1…IF-8, `C:\D\git\dtw-cpp\.claude\plans\2026-09-24-if2-config-design.md` §5 (hpc by config file), `C:\D\git\dtw-cpp\.claude\DECISIONS.md` :258-298 (the 2026-09-24 `ClusterMethod` note this design overturns), `C:\D\git\dtw-cpp\tests\conformance\config_all_fields.toml` (the 49 keys today).