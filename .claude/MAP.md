# DTWC++ — MAP (as-is)

Snapshot: branch `design-2.0`, HEAD `9c08074`, 2026-09-21. `dtwc/` is byte-identical to `a31956e`, the
base of the 2026-09-07 deep-dive reports, so their `file:line` anchors still hold (one exception, §10).
**Read this instead of the tree.** Open a deep dive (§10) only for the section a task cites.

Regenerate the mechanical parts (§5):

```sh
uv run --no-project python scripts/repo_map.py layers            # include graph vs the target layers
doxygen <xml-only Doxyfile over dtwc/>; … repo_map.py symbols <xml-dir>   # classes, enums, functions
```

Layers below are the **target** layers of `design.md` §4; a file's layer is what it *is*, not where it sits.

## 1. One screen

```text
files ─▶ DataLoader / io readers ─▶ Data ─▶ Problem ─┬─ rebind_dtw_fn ─▶ dtw_fn_ (std::function, bound once)
                                                      ├─ fill_distance_matrix ─▶ BruteForce | Pruned | CUDA | Metal
                                                      ├─ dist_by_ind(i,j)  ◀── every algorithm, MIP, scores, init
                                                      └─ cluster() / free algorithm fns ─▶ clusters_ind, centroids_ind
Tier-1: device() → load() → Dataset → cluster() → Result(shared_ptr<Problem>)      CLI: dtwc_cl (one 1.9k-line TU)
Python: _dtwcpp_core.cpp (nanobind, 1.8k)   MATLAB: dtwc_mex.cpp (1.8k) + +dtwc package
```

| Layer | Files | LOC | Where it lives |
| --- | --- | --- | --- |
| base | 22 | 2.1k | `dtwc/{error,settings,missing_utils,parallelisation,timing,env}.*`, `system_memory.cpp`, `types/`, `enums/`, `core/{portable_random,crc32,sha256,llfio_include}.hpp` |
| core | 38 | 8.2k | `dtwc/core/`, `dtwc/{warping*,soft_dtw,distance,Data}.hpp`, `detail/decode_pair.hpp` |
| io | 9 | 2.7k | `dtwc/io/`, `DataLoader.hpp`, `fileOperations.hpp`, `core/matrix_io.hpp` |
| backends | 13 | 5.9k | `dtwc/{cuda,metal,mpi}/`, `core/gpu_dtw_common.hpp` |
| algorithms | 21 | 4.8k | `dtwc/algorithms/`, `initialisation.*`, `scores.*` |
| mip | 18 | 2.8k | `dtwc/mip/` |
| session | 5 | 3.6k | `Problem.{hpp,cpp}`, `Problem_IO.cpp`, `checkpoint.*` |
| surface | 9 | 3.1k | `api.*`, `dtwc_cl.cpp`, `cli/config_file.hpp`, `main.cpp`, `test_api.hpp`, `dtwc.hpp`, `utility.hpp` |
| vendored | 1 | 4.5k | `extern/nanoarrow` (0.8.0) |
| tests | 127 Catch2 files | 53k | `tests/` — 1 605 `TEST_CASE`, 131 CTest entries (134 with Arrow) |
| bindings | — | 10.4k | `python/` (17 files), `bindings/matlab/` (51) |

## 2. Layer cards

### base
Errors (`error.hpp`: base `dtwc::Error :46` — `DtwcError` is its Python name — with `InvalidInput`,
`UndefinedScore`, `SolverError`, `DeviceError`, `IOError`),
`settings.hpp` (defaults, `paths`, and the process-global `inline std::mt19937 randGenerator(29)` at `:55`),
`parallelisation.hpp` (`run_openmp` `:96`, region-local `num_threads` `:124`, the **unnamed** `omp critical`
`:148`, `get_max_threads`), `missing_utils.hpp`, `timing.hpp`, `types/{Index,Range,element_types}`,
`enums/{Method,Solver,LowerBoundStrategy,KernelOverride}`, portable RNG draws, CRC32, SHA-256.
`env.{hpp,cpp}`: process-wide `Env` — `Device{CPU,GPU,HPC}`, device index, thread policy, the `.env`
SLURM credential check with an injectable SSH probe; depends only on `error.hpp`.
Problems: three stdlib-only headers sit above `core/` (C-11); `dtwc::solver` types live in `types/` and reach
every TU (C-12); `available_ram_bytes()` is declared in `DataLoader.hpp` but defined in `system_memory.cpp`.

### core
**Purpose.** One templated DP kernel family under thin per-variant wrappers, a bind-once dispatcher, lower
bounds, and two matrix stores. Header-only except `dtw.cpp`, `dtw_dispatch.cpp`, `pruned_distance_matrix.cpp`.

| File | LOC | Responsibility |
| --- | --- | --- |
| `core/dtw_kernel.hpp` | 508 | **the** recurrences: `dtw_kernel_{full :214, linear :248, eap :323, banded :420}<T,Cost,Cell>`; Cells `Standard :55`, `ADTW :72`, `Soft :128`, `AROW :164` |
| `core/dtw_cost.hpp` | 278 | 13 index-based Cost functors (plain, NaN-aware, AROW; uni + MV) |
| `core/dtw_dispatch.{hpp,cpp}` | 52/380 | `resolve_dtw_fn<T>(const Problem&)` `:337`: variant × missing × ndim × T, bound once |
| `core/dtw.{hpp,cpp}` | 78/117 | `dtw_runtime` (per-call entry for bindings); two test-only `dtw_distance` |
| `core/{dtw_options,selector_validation,variant_validation,distance_semantics}.hpp` | 427 | enums + `DTWVariantParams`; enum / domain / cross-axis guards; `parse_metric_token` |
| `core/{msm,twe}.hpp` | 226 | own rolling-buffer DPs, not kernel-based |
| `core/{lower_bound_impl,lower_bounds,distance_metric}.hpp` | 1 029 | envelopes, LB_Kim / Keogh / Enhanced / Webb; a dead trait matrix |
| `core/pruned_distance_matrix.{hpp,cpp}` | 95/463 | LB-guided exact fill over `Problem&` `:53` + a drifted standalone copy `:322` |
| `core/distance_matrix.hpp` | 103 | `DenseDistanceMatrix`: packed triangular, NaN = uncomputed, lock-free |
| `core/mmap_distance_matrix.hpp` | 760 | llfio matrix: v3 header, SHA-256 identity, per-row XOR digest lanes, file lease |
| `core/mmap_data_store.hpp` | 314 | `.dtws` v1 mmap series store |
| `core/{scratch_matrix,public_distance,z_normalize,time_series,storage,clustering_result}.hpp` | 339 | Eigen scratch; f32→f64 sentinel normalisation; 3-pass z-norm; unused view types; storage enums; result POD |
| `core/{medoid_assignment_policy,distance_sampling_weights}.hpp` | 204 | `OrderedMedoidObjective` (volatile total), D-sampling weights |
| `warping.hpp` | 633 | Standard + MV wrappers; the single `dispatch_metric` `:263` / `dispatch_mv_metric` `:280` |
| `warping_{adtw,ddtw,wdtw,missing,missing_arow}.hpp`, `soft_dtw.hpp` | 1 568 | ADTW; derivative transform + DDTW; WDTW weights + cache; ZeroCost / AROW; Soft-DTW value + gradient |
| `distance.hpp` | 243 | public `dtwc::distance::*` facade |
| `Data.hpp` | 207 | series container, 4 modes in 3 booleans: heap f64, heap f32, view, metadata-only |

**Flows.** Pairwise: `distance::dtw` `distance.hpp:36` → `dtwBanded` `warping.hpp:389` → `dispatch_metric` →
`dtwBanded_impl :135` (empty / identity / orient) → `dtw_kernel_banded`. Bind: `Problem::rebind_dtw_fn` →
`resolve_dtw_fn<T>`: validate → MV-independent → missing switch (overrides variant) → variant switch → closure.
**Problems.** `core/` names `Problem` (C-01, C-04) and `Data`; **three resolvers** for variant × missing that
already disagree (`resolve_dtw_fn`, `distance::dtw`, `dtw_runtime`; C-02, C-14); the orientation preamble is
copied 12× (C-03); pruned fill ×2, drifted (C-04, C-25); `-fno-finite-math-only` is `PRIVATE` though the NaN
contract lives in headers (C-08); dead or test-only surface (C-09); `.dtws` open accepts `ndim = 0` and opens a
read-only input in write mode (S-06, S-07).

### io
| File | LOC | Responsibility |
| --- | --- | --- |
| `DataLoader.hpp` | 594 | builder; delimiter from extension `:365`; `load :420` (reads the global `Env`; `hpc` ⇒ metadata-only), `load_local`, `load_stored :464`; `detail::route_series_storage` is the only heap/mmap router |
| `fileOperations.hpp` | 491 | `from_chars` parsing, `load_folder :385`, `load_batch_file :434` |
| `io/parquet_{schema,reader,chunk_reader}.hpp` | 896 | column selection; eager load; `ParquetChunkReader` (metadata-only ctor, RAM estimates, row-group streaming) |
| `io/arrow_ipc_reader.hpp` | 205 | `ArrowIPCDataSource`: mmap Feather, Float64 spans |
| `io/arrow_c_data.{hpp,cpp}` | 273 | nanoarrow C-Data ingest; always compiled; bindings only |
| `core/matrix_io.hpp` | 243 | matrix CSV read/write with preflight (drags llfio into dense IO, C-21) |

Every include from io into the session is `→ Data.hpp`; io never names `Problem`.
**Problems.** Arrow IPC guards are `assert`-only, an unchecked `StringArray` cast, bare `stoul` (B-08); nanoarrow
does per-element null/type work and leaks a batch on failure (B-09); three artefact-writer families — CLI,
`api.cpp`, `Problem_IO` with its own 1.x filenames (O-09); writers never close-checked (B-05); test
instrumentation `s_bulk_read_invocations` compiled into the loader (O-17).

### backends
CUDA (`cuda/cuda_dtw.cu` 2 475: wavefront / warp / regtile kernels, `compute_distance_matrix_cuda :1436`,
`launch_dtw_kernel :1138`; host-testable `kernel_selection.hpp`, `launch_prep.hpp`), Metal (`metal/metal_dtw.mm`
2 237: MSL source string `:60-992`, `compute_distance_matrix_metal :1230`, 5e9-cell watchdog chunks), MPI
(`mpi/mpi_distance_matrix.cpp`: rank blocks + OpenMP + chunked `Allreduce`). All three decode the linear pair
index with the shared `detail::decode_pair`. None includes `Problem` or `Data`.
**Problems.** The GPU route silently ignores variant, missing strategy, `ndim > 1` and LB strategy
(`Problem.cpp:1041-1085`; G-04); Metal returns an all-zero matrix on allocation failure while claiming a CPU
fallback (`metal_dtw.mm:1417-1433`; G-01); MPI truncates counts to `int` and calls `dtwBanded` directly, so it
is Standard-only and **unreachable from `Problem`, CLI and bindings** (G-03, G-07); kernel bodies ×2–4 and a
double H2D transfer on the LB path (G-05); CUDA ≠ Metal semantics (G-06); device tests pass by skipping (G-08).

### algorithms
Free functions taking `Problem&` as oracle, configuration bag **and** result sink.
`fast_pam` (BUILD + `FastPAM1Naive | FastPAM1 | FasterPAM`; `fast_pam_seeded` is the workhorse behind Tier-1
`pam`, CLARA subsamples and the MIP warm start), `fast_clara` (view-mode sub-`Problem`s; Parquet streaming),
`one_batch_pam`, `clarans` (experimental, serial), `hierarchical` (`build_dendrogram` / `cut_dendrogram`, needs a
filled matrix), `tadpole` (+ `tadpole_auto_dc`), `barycenter` (own path-returning DP; DBA / SSG / Soft-DTW;
takes `const Problem&`), `initialisation` (`init::random`, `Kmeanspp` + seeded twins), `scores` (7 indices).
**Problems.** They include `Problem.hpp` and touch 33 of its members (A-03); six end with a non-transactional
triple write `set_n_clusters; centroids_ind=; clusters_ind=` (e.g. `fast_pam.cpp:537-539`; A-02); `Method`
names 4 of 10 algorithms and there are **three** dispatch tables — the enum, `api.cpp` strings, CLI strings
(A-01); unseeded `fast_pam` draws from the global engine (A-21); `hierarchical` / `tadpole` sum cost with a
plain `+=` (A-09); CLARANS can publish an empty clustering and has a `double → int` UB (A-05).

### mip
`mip_Highs.cpp` / `mip_Gurobi.cpp` (compact Balinski N² model, facility-major vs point-major), `benders.cpp`
(disaggregated cuts; nested Lloyd through a `friend`), `lagrangian_root.cpp` (subgradient + Kelley duals, B&B,
`prepare_dense_D`), `reduced_cost_fixing`, `pdlp_lp` (arbiter only), `solution_transaction`
(`ExactClusteringTransaction`: snapshot, validate, publish, roll back), `warm_start`, `nearest_medoid<Dist>`.
**Problems.** One model built three ways, equivalent only while D is symmetric (A-17); `prepare_dense_D`
doubles the N² matrix (A-12); Benders prints and returns on invalid input (A-07); `benders` and the PDLP variant
are strings compared with literals inside the library (A-08).

### session
`Problem` — **175 members: 105 public functions, 33 private, 12 public fields, 25 private** (the next largest
class has 58). Eight responsibility clusters: data, DTW binding, distance cache + identity, clustering state,
settings, method dispatch, output, checkpoint. Three `friend`s.

- **Lookup** `dist_by_ind(i,j)` `Problem.cpp:678`: preflight `:682` → `validate_mmap_cache_identity :683`
  (which preflights **again** at `:598`) → dense-cache snapshot compare `:684` → `i == j ⇒ 0` `:685` →
  unsynchronised `needs_init :694` → `critical(distByInd_init)` resize + rebind `:697-709` → hit `m.get`, or
  miss `dtw_fn_(series(i), series(j))` → `m.set :728`. ≈30–40 branches per O(1) lookup (O-01, O-02, O-08).
- **Fill** `fill_distance_matrix :856`: five validations → deferred allocation → rebind `:883` → serial NaN
  pre-scan `:891-918` → `Auto ⇒ Pruned | BruteForce :923-929` → downgrades → switch `:1024`: pruned, CUDA,
  Metal, or `fillDistanceMatrix_BruteForce :781-845` (row-disjoint `run_openmp`, two `visit_distmat` per pair,
  checkpoint after each joined block).
- **Dispatch** `cluster() :1118-1141` on the 4-value `Method`: Lloyd k-medoids, MIP (Benders when `"on"`, or
  `"auto"` and N > 200), LR-core, TADPole. Not reachable from `cluster()`: FastPAM, CLARA, OneBatchPAM,
  CLARANS, hierarchical, barycenter k-means.
- **Result out**: the algorithm's triple write, the MIP transaction, or Lloyd's best repetition →
  `labels() / medoids()` → Tier-1 `Result` reads them back through `shared_ptr<Problem>` (`api.cpp:213-214`).
- `checkpoint.cpp` (974): dense generation checkpoints (SHA-256 manifest, atomic `CURRENT`), binary result
  format, mid-fill interval saves.

**Problems.** God object (O-03); public configuration fields force revalidation on every lookup (O-02);
`is_distance_matrix_filled()` is an O(N²) NaN scan paid per Lloyd iteration and on every fill entry (O-13);
`distanceInClusters()` re-issues ≈N²/2k cached lookups per iteration (O-04); `set_data` does not resize the
label buffers, `set_view_data` does (O-05); `set_max_iter(0)` is accepted (O-06); **`d(i,i)` is hard-coded to
0, which is wrong for Soft-DTW** where self-distance is negative (S-01).

### surface
`api.{hpp,cpp}`: `load → Dataset → cluster → Result`, `device()`; nine method strings; `auto ⇒ pam` for N ≤ 5000,
else `clara`; per-call device override through a local `Env`; **`hpc` throws "beta" in C++**; GPU is refused for
matrix-free methods. `dtwc_cl.cpp`: 22 helpers, then `run_cli_main :764-1908` — 48 options, `--config` through
`cli/config_file.hpp` (TOML, or YAML via fkYAML; argv beats file, unknown key = error), lower-case and alias
remap `:1038-1065`, extension-based load chain spliced by `#ifdef` `:1243-1444`, dispatch `:1674-1822`, four CSV
writers + checkpoint. `test_api.hpp`: the production capability probes the bindings use, misnamed, pulling 69
project files. `main.cpp`: demo with a repo-relative path (non-negotiable #1).
**Problems.** B-01 (1 145-line function), B-02 / O-11 (twelve options re-parsed by hand), B-04 (`.h5` and any
unknown extension fall into the CSV parser while README advertises HDF5), B-06 (a failed `--dist-matrix` load
and a failed checkpoint save are warnings with exit 0), O-12 (in-memory `cluster()` copies the dataset).

## 3. Bindings

| | Python | MATLAB |
| --- | --- | --- |
| Glue | `python/src/_dtwcpp_core.cpp` 1 833 (nanobind, stable ABI ≥ 3.12) | `bindings/matlab/dtwc_mex.cpp` 1 797 (C MEX, handle + `mexLock`) |
| Package | `python/dtwcpp/`: `_api.py` (Tier-1), `_clustering.py` + `sklearn.py` (estimators), `_hpc.py` 601 (SSH + rsync + `sbatch` through `scripts/slurm/slurm_remote.sh`), `io.py`, `convert.py`, `distance.py`, `preprocess`, `diagnose`, `features`, `test` | `+dtwc/`: `Problem.m` 448, `DTWClustering.m`, `Dataset.m`, `Result.m`, `cluster.m`, `load.m`, `+distance/`, one file per algorithm and score |
| Surface | Tier-1, two sklearn estimators (`DTWClustering`, `DTWCKMedoids`), and Tier-2: `Problem`, every option struct and enum, all algorithms, scores, checkpoints, capability flags, `test.parallelisation()` / `test.gpu()` | same names (snake_case), 1-based indices at the MEX boundary |
| Tier-1 route | **`_api.py:461-500` re-implements `api.cpp`'s method resolution and dispatch in Python**; `dtwc::cluster` / `dtwc::load` are not bound | one call into `dtwc::load` + `dtwc::cluster` (`dtwc_mex.cpp:1482-1488`) |
| Release history | v1.0.0 shipped a thin pybind11 wrapper of `Problem` with 1.x names (`refreshDistanceMatrix`, `cluster_size`, `p_vec`), never on PyPI; everything else is 2.0-only | **no MATLAB file exists in v1.0.0** |
| Known gaps | raises its own `RuntimeError` text for `hpc` (F24); `Result.save` not through C++ (F37); `__all__` exports `IOError` (S-19); `distance.dtw(variant=)` knows five variants — no `msm` / `twe` (S-14); `set_data` / `Data` copy through Python floats; no `.pyi` stubs | 32 unchecked `static_cast<int>` although `get_exact_int` exists (S-05); no `msm` / `twe` in `+distance` (S-14); checkpoint findings F52 / F53 |

## 4. Vocabulary (what a setting can be)

| Axis | Enum(s) | Note |
| --- | --- | --- |
| where it runs | `Device{CPU,GPU,HPC}` (`env.hpp`) · `DistanceMatrixStrategy{Auto,BruteForce,Pruned,CUDA,Metal}` (`Problem.hpp:93`) · `detail::Tier1ExecutionTarget{CPU,GPU,HPC}` | three vocabularies; `Problem` never reads `Env` |
| precision | `core::Precision{Float32,Float64}` · `cuda::CUDAPrecision{Auto,FP32,FP64}` · `metal::MetalPrecision{…}` | three again |
| distance | `DTWVariant{Standard,DDTW,WDTW,ADTW,SoftDTW,MSM,TWE}` · `MetricType{L1,L2,SquaredL2}` (**default L1**; `L2` reachable from no token and identical to L1 for `ndim = 1`, C-19) · `ConstraintType{None,SakoeChibaBand}` (band in **integer cells**, `-1` = none; an infeasible band returns a finite `max()`, S-15) · `MVMode{Dependent,Independent}` · `MissingStrategy{Error,ZeroCost,AROW,Interpolate}`; **no final square root** anywhere | one closed enum guarded in five places (C-14); peers default to squared Euclidean (+ sqrt) and fractional windows |
| method | `Method{Kmedoids,MIP,LRCore,TADPole}` · `PAMVariant` · `Linkage` · `OneBatchWeighting` · `BarycenterMethod` · `Solver{Gurobi,HiGHS}` | |
| pruning, kernels | `LowerBoundStrategy{Auto,None,Kim,Keogh,KimKeogh,Enhanced,Webb}` · `KernelOverride` · `cuda::detail::KernelPath` | |
| storage | `StoragePolicy{Auto,Heap,Mmap}` | series storage is independent of matrix storage |

Randomness: one process-global `std::mt19937(29)` beside seeded, portable `_seeded` entry points.
Threads: OpenMP is required unless `DTWC_ALLOW_SEQUENTIAL=ON`; no silent sequential fallback.

## 5. Mechanical facts

**Upward includes against the target layers: 18. Seventeen are `→ Problem.hpp`** — from `core/` 2
(`dtw_dispatch.cpp`, `pruned_distance_matrix.hpp`), `algorithms` 9 (seven algorithm TUs, `initialisation.cpp`,
`scores.cpp`), `mip` 6 — and one is `system_memory.cpp → DataLoader.hpp`. No include cycles at file level.

```text
row includes column   base core   io back algo  mip sess surf   (vendored column omitted: io→nanoarrow 1)
base                    18    0    1    0    0    0    0    0
core                    35   90    0    0    0    0    2    0
io                       8    8    4    0    0    0    0    0
backends                 8    5    0   10    0    0    0    0
algorithms              24   16    1    0   15    0    9    0
mip                     20    5    0    0    1   25    6    0
session                 13   13    3    2    5    1    5    0
surface                 11   23    9    3   14    0    3    8
```

Fan-in: `error.hpp` 44, `settings.hpp` 36, **`Problem.hpp` 22**, `core/clustering_result.hpp` 13,
`core/dtw_options.hpp` 12, `warping.hpp` 11. Header weight (project files pulled in): `test_api.hpp` 69 files /
11.5k LOC, `dtwc.hpp` 68, `core/pruned_distance_matrix.hpp` 31, `distance.hpp` 24, `Problem.hpp` 21,
`DataLoader.hpp` 14; `core/dtw_kernel.hpp` pulls 1. Classes: 137; after `Problem` (175 members) come
`MmapDistanceMatrix` 58, `DataLoader` 45, `ParquetChunkReader` 35, `Data` 29. Enums: 25.

## 6. Build

Targets: `dtwc++` (STATIC, PIC; 19 `.cpp` + nanoarrow; MPI / CUDA / Metal TU when enabled) · `mip-solvers`
(OBJECT) · `dtwc_options` / `dtwc_warnings` (INTERFACE) · `dtwc_cl` (installed, CPack) · `dtwc_main` · 4
examples · 8 benchmarks · `_dtwcpp_core` · `dtwc_mex` · 127 test executables. **`dtwc++` and its headers have
no install or export rule**; consumption is `add_subdirectory` / CPM only (S-02).

Options: 19 `DTWC_*` user options and 13 `dtwc_*` developer options (B-14). `DTWC_BUILD_{TESTING,EXAMPLES,
BENCHMARK,PYTHON,MATLAB}` OFF; `DTWC_ENABLE_{HIGHS,GUROBI,LLFIO,YAML,METAL}` ON; `DTWC_ENABLE_{ARROW,CUDA,MPI}`
OFF; `DTWC_ALLOW_SEQUENTIAL` OFF (missing OpenMP is a `FATAL_ERROR`); `DTWC_ENABLE_NATIVE_ARCH` ON,
`DTWC_ARCH_LEVEL` empty; `DTWC_DEV_MODE` OFF, so **warnings are empty in every CI build**. Release FP flags are
applied at directory scope and therefore reach fetched HiGHS / llfio (S-03).

| Dependency | How | Pin | Gate | Used by |
| --- | --- | --- | --- | --- |
| CPM.cmake | downloaded at configure | 0.42.1 SHA | — | every fetch (vendoring planned, B-17) |
| Eigen | CPM, headers | 5.0.1 SHA | — (PUBLIC) | scratch matrix, matrix IO |
| OpenMP | `find_package` | system (`brew install libomp` on macOS) | required unless `ALLOW_SEQUENTIAL` | `dtwc++` |
| nanoarrow | vendored | 0.8.0 | always | Arrow C-Data ingest |
| CLI11 · fkYAML | CPM, headers | 2.6.2 · 0.4.4 SHA | always · `ENABLE_YAML` | `dtwc_cl` |
| HiGHS | CPM | 1.15.1 SHA | `ENABLE_HIGHS` | mip |
| Gurobi | `FindGUROBI` (`GUROBI_HOME`), probed on every configure | — | `ENABLE_GUROBI` | mip |
| llfio + quickcpplib | CPM git, quickcpplib cloned and patched | commit pins | `ENABLE_LLFIO` | mmap matrix, `.dtws` |
| Arrow + Parquet | `find_package`, else CPM static | 19.0.1 SHA | `ENABLE_ARROW` | io readers |
| CUDA · Metal · MPI | toolkit · system frameworks · `find_package` | system | `ENABLE_*` | backends |
| Catch2 · Google Benchmark · nanobind | CPM · CPM · pip then CPM | 3.13.0 SHA · 1.9.5 · ≥ 2.0 | testing · benchmark · python | |
| Docs | Hugo extended 0.147.8, Pagefind 1.3.0, Doxygen (`docs/Doxyfile`), lcov | CI | — | site |

Presets (all use `${sourceDir}/build`, only set testing ON): `clang-win`, `clang-win-debug`, `msvc`,
`gcc-linux`, `clang-macos`; only `clang-win` and `clang-macos` have build and test twins.

## 7. CI (`.github/workflows/`)

`ubuntu-unit` (gcc 11/12, clang 14–17, Debug; ASan+UBSan by raw flags; an Arrow leg) · `windows-unit` (MSVC
Debug) · `macos-unit` (preset `clang-macos`, Release, HiGHS, Metal) · `cuda-mpi-detect` (configure/build smoke,
one `mpiexec -n 2`) · `documentation` (coverage build, `check_docs_contract.py`, pin checks, Hugo + Doxygen,
Pages) · `matlab-mex` (3 OS, R2024b) · `python-tests` (3 OS × 3.9–3.14) · `python-wheels` (cibuildwheel) ·
`release-artifacts` · `draft-pdf` (JOSS).
**The unit workflows' push triggers name only `develop` and `Claude` (docs and cuda-mpi-detect add `main`,
wheels only `develop`); `pull_request` triggers name only `main`. A push to `design-2.0` runs nothing.** No leg builds with
every optional dependency OFF, none with `DTWC_DEV_MODE=ON`, none lints (B-15). CUDA never runs in CI.

## 8. Tests

| Directory | Files | Holds |
| --- | --- | --- |
| `tests/unit` (flat) | 71 | everything, unsorted by layer, several `*_phase0` names |
| `tests/unit/{core,algorithms,mip,io,types}` | 22, 11, 4, 1, 2 | by subject |
| `tests/unit/adversarial` | 15 | 334 cases; 158 assert structure only (T-18) |
| `tests/integration` | 13 | 6 real-CLI `cmake -P` scripts (the only reachability proof for the CLI), 3 fixture writers, 4 unregistered |
| `tests/conformance` | 7 | C++ / Python / MATLAB / CLI parity against one tracked reference |
| `tests/{support,fixtures,cmake}` | 1, 3, 1 | `deterministic_series.hpp` — used by no in-scope test (T-13) |
| `tests/{python,matlab,mutation}` | 32, 9, 1 | pytest, MATLAB suites |

Registration: every test goes through `dtwc_add_test` (`cmake/DtwcTest.cmake:46`): strict by default, the PASS
regex is Catch2's summary floored by `tests/floors.cmake` (generated by `scripts/measure_test_floors.py`), a
missing floor is a configure error, `MAY_SKIP` opts into exit code 4, `MARKER` proves the subject ran,
`REQUIRES` unregisters a test whose subject is absent. Known holes: explicit floors silently beat the measured
table (`DtwcTest.cmake:121-130`); a `MAY_SKIP` test carries no floor; seven device tests sit at floor `1;1`;
`unit_test_mpi` passes on its own skip line.
**Floors were measured on Windows only: on macOS 5 of 131 tests fail on floors alone** (case counts match,
assertion counts do not). Conformance rewrites its tracked reference when the file is absent or
`DTWC_CONFORMANCE_REGEN=1` (T-15). Serial suite: 129 s here (Apple clang, Release), 469 s MSVC Debug.

## 9. Outside `dtwc/` and `tests/`

`docs/` Hugo site, `api-contract-2.0.md` (the freeze artefact), `derivations/` (D1–D3 done), `Doxyfile`; nine
site pages are generated by `scripts/generate_docs.py` and freshness is enforced in CI · `scripts/` gate scripts
(`check_docs_contract`, `check_record_hygiene`, `check_repo_hygiene`, `check_supply_chain_pins`,
`check_site_links`), `measure_test_floors.py`, `repo_map.py`, `slurm/` (the live HPC transport) · `benchmarks/`
8 Google Benchmark targets plus UCR ledgers · `examples/` C++, Python, MATLAB · `data/dummy` (25 sample series,
used by tests) · `joss/`, `media/` (paper) · `develop/` (contributor-doc sources) · `cmake/` 16 modules.

**Gate scripts read process records by path.** `check_docs_contract.py` (in CI) and `check_record_hygiene.py`
pin exact strings in the archived plan, `LESSONS.md`, `CITATIONS.md`, `UNIMODULAR.md`, five baselines
(`2026-07-08-lb-cascade`, `2026-07-30-d2-lb-keogh`, `2026-07-30-d3-lb-enhanced-webb`,
`2026-07-08-faster-pam-bench`, `2026-07-23-r1-record-hygiene`), `summaries/handoff-2026-07-30-d3-lb-enhanced-webb.md`,
four `commands/`, `openmp-crashcourse.md`, and **`reports/test_kasper_analysis/` (REPORT.md + five scripts,
`check_docs_contract.py:2070-2077`)**; `generate_docs.py:201` reads `PLAN-archive-2026-07-20-phases0-9.md`.
Deleting or renaming any of those needs the script edited in the same commit. The F- and D-numbers of the old plan are public in the docs and pinned by the checker: keep the numbering.

## 10. Deep dives and their caveats

| Subject | File |
| --- | --- |
| core, session, algorithms + mip, io + cli + build, backends | `reports/2026-09-07-asis-{core,orchestration,algorithms-mip,io-cli-build,backends}.md` |
| test taxonomy, per-file verdicts | `reports/2026-09-07-tests-taxonomy-{unit-flat,unit-subdirs,adversarial-integration}.md` |
| row-level target and verdicts | `specs/2026-09-07-design-2.0-campaign.md`, `specs/2026-09-07-diff-ledger.md` |
| raw sweep, 39 findings | `reports/2026-09-21-multiagent-sweep.md` — verified digest: `PLAN.md` S-rows |
| first macOS baseline, codegen probe | `baselines/2026-09-21-macos-first-baseline.md` |

Caveats found while distilling: in the **backends** report the line anchors of six small files are cumulative
offsets, not file-local — subtract 140 (`cuda_memory.cuh`), 273 (`gpu_config.cuh`), 383 (`kernel_selection.hpp`),
455 (`launch_prep.hpp`), 86 (`mpi_distance_matrix.cpp`), 279 (`allreduce_chunking.hpp`). The orchestration
report understates `rebind_dtw_fn` call sites (also `Problem.cpp:457, 518, 709, 883`). Ledger row T-15 and the
2026-09-07 handoff overstate the conformance hole: the reference is git-tracked, so a normal checkout compares
for real. The 2026-09-21 handoff says the III.10 decisions are open; **the spec records them all as adopted
on 2026-09-07**.
