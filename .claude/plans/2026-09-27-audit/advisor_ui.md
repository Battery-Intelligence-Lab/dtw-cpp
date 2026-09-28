# Interface advisor — three users walk chair §3, then the final spec

Evidence: lines opened at HEAD cd5d449 are cited `file:line`. "Today" = what the code does now; "target" = chair §3 as amended in §B.7.

## A. Walkthroughs

### (a) Python data scientist, CSV of 2,000 series

```
pip install dtwcpp
```
**Stumble 1 — nothing to install.** `dtwcpp` was never on PyPI (main_session_checks:10). Wheels are built only as CI artifacts (`python-wheels.yml:97` `upload-artifact`; no release upload in any workflow). Today she must `pip install "git+https://github.com/Battery-Intelligence-Lab/dtw-cpp"` with CMake ≥ 3.26 and a C++20 compiler — a hard stop for most data scientists. Interface requirement, not code: RL-1..3 must put wheels on the GitHub release and `pip install dtwcpp` must resolve (PyPI upload is Volkan's action). `matplotlib` must be an extra (`pip install "dtwcpp[plot]"`): today `res.plot()` imports it lazily (`_api.py:342-343`) and there is no extra (`pyproject.toml:59-65`) → bare `ImportError`.

```python
import dtwcpp as dtwc
data = dtwc.load("sensors.csv", skip_cols=1)      # ok today: C++ DataLoader via _read_data (_dtwcpp_core.cpp:249-272)
res  = dtwc.cluster(data, k=3)                    # today method="pam" (_api.py:503); target auto→pam (N ≤ 5000, run.cpp:60,68)
res.labels, res.medoids, res.cost                 # ok; target np.int64
res.score("silhouette")                           # ok (Python re-implementation _api.py:231-253; target C++ Result::score)
res.save("out/")                                  # ok, four CSVs (_api.py:255-328)
res.plot()                                        # ok if matplotlib; writes ./clusters_2d.png and prints the path (_api.py:363-364)
```
Errors she meets: `dtwc.cluster(data, k=5000)` → `InvalidInput("cluster: k must not exceed the number of series.")` (run.cpp:602-603, mirrored `_api.py:571-572`); a typo `bnad=10` → today `TypeError` (no such kwarg); target: `parse_config` raises `InvalidInput` for an unknown key with CLI11's text (config.hpp:137) — identical to the CLI's, as intended.

**Stumble 2 — choosing k.** The first thing she does is a sweep:
```python
for k in range(2, 9): print(k, dtwc.cluster(data, k=k).score("silhouette"))
```
Today every call builds a fresh `Problem` and refills the 2,000² matrix (`_api.py:576-598`); `DTWClustering(n_init=5)` refills it per restart (`_clustering.py:376-397`). In the chair's target the same holds: `cluster()` = `run(Config, Data)` fills once per call. Fix (§B.7 #2): `dist_matrix` — an existing Config key (`config.cpp:313`) — accepts an N×N array in Python/MATLAB:
```python
res2 = dtwc.cluster(data, k=2)
for k in range(3, 9): print(k, dtwc.cluster(data, k=k, dist_matrix=res2.distance_matrix).score("silhouette"))
```
One key, no new name, and `DTWClustering(metric="precomputed").fit(D)` is the sklearn spelling of the same mechanism.

Feasible from today's code: yes for every line except the pip step and the array-valued `dist_matrix` (~10 lines in the `run` binding: `prob.set_distance_matrix(ndarray)` before dispatch).

### (b) MATLAB battery engineer, 5,000 variable-length cycles in a folder of CSVs, GPU workstation

**Stumble 1 — no GPU MEX exists.** `matlab-mex.yml:34-36,62-65` builds a CPU-only, `LLFIO=OFF` MEX and uploads it as a CI artifact; nothing is attached to a release. `docs/content/getting-started/matlab.md:16-26` says `-DDTWC_BUILD_MATLAB=ON` and "ensure it is on your MATLAB path", never `-DDTWC_ENABLE_CUDA=ON`. He must build: `cmake -S . -B build -DDTWC_BUILD_MATLAB=ON -DDTWC_ENABLE_CUDA=ON` (nvcc + MSVC host on Windows, main_session_checks:15), then `addpath build/bin; addpath bindings/matlab`. With the CPU MEX he sees, at `dtwc.device('gpu')`:

> `[dtwc] device='gpu' requested but this build has no GPU backend compiled in.` / `Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.` / `This build will not silently fall back to CPU.` (env.cpp:237-239, id `dtwc:deviceError`)

Correct and loud; it should name the artefact ("this dtwc_mex / this wheel") since "rebuild" means a different thing to a MEX user (§B.7 #9).

```matlab
dtwc.device('gpu');                                 % ok on a CUDA MEX (dtwc_mex.cpp:840-843 → dtwc::device)
data = dtwc.load('cycles/');                        % ok: a directory → load_folder, one file = one series, name = file stem (DataLoader.hpp:500-501)
res  = dtwc.cluster(data, 8, 'band', 100);          % ok today (cluster.m:33-36); ragged series ok on CUDA (per-series lengths, launch_prep.hpp:70-73)
```
**Stumble 2 — the band and variable lengths.** Cycles differ in length by hundreds of samples, so `band=100` is refused before any GPU call:

> `Problem::fill_distance_matrix: band = 100 is narrower than the length difference between series 'cycle_0007' (index 6, length 2811) and series 'cycle_4120' (index 4119, length 4261), so no warping path fits that pair. The smallest feasible band is 1450; pass band >= 1450, or band = -1 for full DTW.` (Problem.cpp:1006-1011, format)

That is the right message (D-12). Keep.

**Stumble 3 — any key beyond four.** `dtwc.cluster(data, 8, 'variant', 'msm')` → inputParser `'variant' is not a recognized parameter` (cluster.m:30-37 binds only `method/band/device/max_iter`; the MEX `tier1_cluster` takes exactly those, cluster.m:67-71). Target: name-value pairs → `parse_config` → `run` (config.hpp:133-138 exists; the MEX needs one `run` command). On the GPU, `msm` is then refused before I/O as `Problem::fill_distance_matrix: CUDA <request>; no backend call or CPU fallback was attempted. <fix>` (Problem.cpp:162-166).

```matlab
res.labels(1:5)                    % today int32 (Result.m:14); target int64, 1-based
res.medoids                        % 1-based indices into the sorted file order; res.save('out') writes cycle names into <name>_medoids.csv
res.score('silhouette'); res.plot();   % plot today saves 4 CSVs to a tempdir and reads one back (Result.m:112-118); target MEX Result_distance_matrix
```
What he does not see and should: GPU precision is `auto`; on an RTX (FP64 at 1/64 rate) the fill runs in float32 (chair §3 table). `res.config` must show `gpu-precision = "float32"` so he knows what he got. Threads: `maxNumCompThreads` governs the MEX (D-15).

Cost check [computed, ~3,000-sample cycles, band 1,450 forced by the length spread]: 12.5 M pairs × 3,000 × 2,900 ≈ 1.1e14 cell updates — tens of minutes on an RTX 4000 Ada at ~1e11 cells/s [assumed rate], impractical on 24 CPU cores. So the GPU MEX is the whole point for him; the release must ship one or the docs must give the one-line build.

### (c) HPC user, 10 M series in Parquet on a SLURM cluster with GPUs, from a laptop

```python
import dtwcpp as dtwc
dtwc.device("hpc:gpu")
```
Today: `DeviceError("[dtwc] unknown device 'hpc:gpu'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N), hpc.")` (env.cpp:71-75). `"hpc"` alone only records (`__init__.py:191-193`). And **today no path selects the GPU partition at all**: `dtwcpp.cluster` never passes `device` to `cluster_on_hpc` (`_api.py:552-555`), so every hpc run is CPU. Target grammar `hpc:gpu` is required, not optional.

```python
data = dtwc.load("/data/coml-battery/series.parquet")   # never read locally
res  = dtwc.cluster(data, k=50, band=400)
```
**Stumble 1 — which side is the path on?** A path source is by rule cluster-side: "pass through, no local read, no upload" (`_hpc.py:591-593`), an in-memory array is written to TSV and uploaded (`:594-602`). A laptop path is therefore forwarded verbatim and fails inside the job with the remote reader's IOError; nothing says so locally. Rule for the target (§B.7 #5): a path that exists locally is uploaded; one that does not is a cluster path, and the submit line prints which.

**First run, no `.env`.** Today `RuntimeError` (not `DeviceError`):
> `Missing .env in C:\Users\me\proj (the working directory, or DTWC_REPO_ROOT). Create it with SLURM_USER, SLURM_HOST and SLURM_REMOTE_BASE; scripts/slurm/env.example in a source checkout is a template.` (`_hpc.py:419-425`)

A pip user has no "source checkout". C++ has a second, different text mentioning "the repository root" and `arc-login.arc.ox.ac.uk` (env.cpp:77-85) that Python never reaches. One text, `DeviceError`, with the three lines inline (§B.5).

**Wrong key** (`SLURM_HOSTNAME=` instead of `SLURM_HOST=`): the wrapper exits 1 with `ERROR: SLURM_HOST is not set in .env` (slurm_remote.sh:73-77) and Python wraps it as `RuntimeError("submit-cluster failed (exit 1).\nWrapper output:\nERROR: SLURM_HOST is not set in .env")` (`_hpc.py:467-471`). Right content, wrong type, buried under "submit-cluster failed".

**ssh without a key**: `ssh` fails during upload/submit → same wrapper-output wrapping, plus the "no Job ID" checklist (`_hpc.py:473-479`) which is the best message on this path today. **No `dtwc_cl` on the cluster**: not checked before submission; the job itself dies with `ERROR: no build-*/bin/dtwc_cl found. Run 'slurm_remote.sh build' first.` (cluster_generic.slurm:73) after queueing.

**Stumble 2 — a failed job is invisible.** `wait()` returns when the id leaves `squeue`, whatever the state (`_hpc.py:513-517`); `download_labels` then raises `FileNotFoundError("exact labels for job 12345 not found after download: …/results/slurm/dtwc_series_12345/dtwc_series_labels.csv")` (`_hpc.py:542-545`). The stderr is in `logs/cluster_12345.err` on the cluster (cluster_generic.slurm:8) and is never fetched. Target: `sacct -j <id> --format=State,ExitCode` + the tail of that `.err` in the `DeviceError`.

**Stumble 3 — the laptop must stay up.** `cluster()` blocks up to 86,400 s (`_hpc.py:551`), the job id is captured but not printed (`_hpc.py:472,606`). If the laptop sleeps, the handle is gone. Target: print `[dtwc] submitted job 12345 → <remote>/results/<name>_12345; waiting (fetch later: slurm_remote.sh download <name> 12345)` at submit and repeat it in every timeout/interrupt error. No new API.

**Stumble 4 — the target itself.** 10 M series × 8 K samples = 640 GB resident float64 [computed], fine on a 2 TB node. But the remote `auto` on `gpu` resolves to `pam` (run.cpp:68), i.e. an N²/2 × 8 B = 4e14 B = 400 TB matrix [computed], and `clara` on `gpu` is refused unless its sample covers every series (run.cpp:71-78; run.hpp:16-17). So `hpc:gpu` + 10 M series fails after the job starts, for every method. On CPU, `clara` assignment alone is N·k = 5e8 DTWs; at band 400 that is ≈ 3.2e15 cell updates ≈ 9 h on 96 cores at 1e9 cells/s/core [assumed], ≈ 90 h unbanded. The rectangular (sample × all) GPU kernel the chair defers to a DECISIONS line (W4) is the only thing that makes `hpc:gpu` mean anything at the charter's scale. §B.5 makes `auto` one rule on every device and names the GPU-CLARA gap as a W13 deliverable; until it lands the remote refuses `gpu` + N > 5000 before I/O and the refusal is what the user reads (Stumble 2 fixed).

**Results**: today labels only (`_api.py:556-557`); `res.medoids is None`, `res.score()` → `InvalidInput("no local distance matrix available to score (an hpc run returns labels only; …)")` (`_api.py:207-211`). `<name>_medoids.csv` is written remotely and not fetched; fetching it costs nothing (§B.7 #6).

## B. Final spec

### B.1 C++ Tier-1 (`dtwc/api.hpp`)

```cpp
namespace dtwc {
using index_t = DTWC_INDEX_T;                       // std::int64_t by default (base/settings.hpp)

std::string device(std::string_view name);          // cpu | gpu | gpu:N | (cuda[:N] alias); returns canonical; "hpc[:gpu]" → DeviceError naming Python/CLI
std::string device();

struct LoadOptions { index_t skip_cols = 0, skip_rows = 0; char delimiter = 0; std::string column, name; };
Dataset load(const std::filesystem::path&, LoadOptions = {});     // lazy; CSV/TSV, folder, Parquet, Arrow by extension; typed IOError "not built" for a missing reader
Dataset load(std::vector<std::vector<data_t>>, LoadOptions = {});

Result  cluster(const Dataset&, index_t k, Config = {});           // = run(config with k, no output); designated initialisers for any key
Result  run(const Config&);                                        // the CLI's pipeline; writes `output`
Result  run(const Config&, Data);                                  // in-memory series (and, when config.dist_matrix is set, Data may carry names only)

class Result {                                                     // values; the session is kept only for the lazy matrix
  const std::vector<index_t>& labels() const;  const std::vector<index_t>& medoids() const;
  double cost() const;  Method method() const;  int iterations() const;  bool converged() const;
  std::string_view device() const;  const Config& config() const;   // the resolved run, what --print-config prints
  double score(std::string_view name) const;                       // silhouette (mean) | davies_bouldin | dunn | calinski_harabasz | inertia
  void   save(const std::filesystem::path& dir) const;             // <name>_labels/_medoids/_distance_matrix/_silhouettes.csv (v1 filenames)
  std::vector<double> distance_matrix() const;                     // N×N row-major, filled on demand
};

SystemCheck system_check();                                        // threads, threads_engaged, openmp, cuda, metal, highs, gurobi, arrow, llfio, gpu_name, gpu_validated, gpu_reason
namespace distance { double dtw(std::span<const double> x, std::span<const double> y, const DistanceConfig& = {}); }
}
```
`Config` stays **flat** (member per CLI long name, `cli::bind` the one table, `to_config_text`/`parse_config`, TOML/YAML through CLI11). `run()` builds the `DistanceConfig{variant+params, metric, missing_strategy, mv_mode, band}` from it once; that struct is what `distance::dtw` and `Problem::set_distance` take. Declaration order of the Tier-1 keys is the `--help` order below, so `{.method = …, .band = 10, .device = …}` compiles.

**Config keys — 45 (today's 49 minus `batch-weighting`, `benders`, `max-benders-iter`, `resume`); `--help` in four groups:**

| group | keys (CLI spelling; Python/MATLAB use `_`) |
|---|---|
| **Run** (what a user types) | `input`, `output`, `name` (= input stem), `n-clusters`/`-k` (**required**), `method` (`auto`), `band` (-1), `metric` (`l1`\|`squared_euclidean`), `variant` (`standard`\|`ddtw`\|`wdtw`\|`adtw`\|`softdtw`\|`msm`\|`twe`), `device` (process device), `seed` (42), `skip-rows`, `skip-cols`, `delimiter`, `dist-matrix`, `verbose` |
| **Variant** | `wdtw-g`, `adtw-penalty`, `sdtw-gamma`, `msm-c`, `twe-nu`, `twe-lambda`, `mv-mode`, `missing-strategy` |
| **Advanced** | `max-iter`, `n-init`, `sample-size`, `n-samples`, `batch-size`, `linkage`, `dc`, `dtype`, `gpu-precision` (`auto`\|`float32`\|`float64`), `ram-limit`, `mmap-threshold`, `checkpoint`, `checkpoint-interval`, `column` |
| **Solver** | `solver` (`highs`\|`gurobi`), `mip-gap`, `time-limit`, `no-warm-start`, `numeric-focus`, `mip-focus`, `verbose-solver`, `lr-max-nodes` |

Enum spellings: one `Name<E>` table each, canonical + at most one synonym (`sqeuclidean`, `f32`/`f64`); `obp`/`lr`/`hclust`/`fp32`/`float`/`double` go.

### B.2 Python

```python
dtwcpp.device(name=None) -> str                    # "hpc" / "hpc:gpu" record only; credentials checked at cluster()
dtwcpp.load(source, *, skip_cols=0, skip_rows=0, delimiter=None, column=None, name=None) -> Dataset
    # source: path (CSV/TSV/folder/Parquet/Arrow; .h5 via h5py if the owner keeps HDF5), ndarray, list of 1-D arrays (ragged),
    # or anything with __arrow_c_stream__/__arrow_c_array__ (polars, pyarrow, DuckDB) — the nanoarrow ingest, no separate name
dtwcpp.cluster(data, k, **config) -> Result        # kwargs == Config keys; dist_matrix may be a path or an N×N ndarray
dtwcpp.run(**config) -> Result                     # the CLI in Python: run(input="x.parquet", k=3, output="out")
dtwcpp.compute_distance_matrix(X, *, band=-1, metric="l1", variant="standard", device=None, **variant_keys) -> ndarray  # over Problem, device-aware
dtwcpp.distance.dtw(x, y, *, variant="standard", band=-1, metric="l1", missing_strategy="error", mv_mode="dependent", **variant_keys) -> float
dtwcpp.DTWClustering(n_clusters, *, n_init=1, metric="l1"|"precomputed", device=None, **config)   # fit / predict / fit_predict / transform / score (no refit); labels_, medoids_, inertia_
dtwcpp.system_check() -> dict;  dtwcpp.test.parallelisation(), dtwcpp.test.gpu()   # one-line wrappers (CHARTER:112-115)

class Result:  labels, medoids (np.int64), cost, method, iterations, converged, device, config (dict),
               distance_matrix (lazy ndarray; None on hpc), score(name), save(dir), plot(path=None, show=None)
               # hpc: labels + medoids; cost None; score()/distance_matrix raise InvalidInput naming the reason
```
Errors (unchanged hierarchy, `error.hpp:46-89`, bound with dual bases): `DtwcError(Exception)` → `InvalidInput(ValueError)`, `UndefinedScore(InvalidInput)`, `SolverError`, `DeviceError`, `IOError(OSError)`. Every hpc failure is `DeviceError` (today `RuntimeError`/`FileNotFoundError`/`TimeoutError`, `_hpc.py:411-545`); argument errors are `InvalidInput`.

`__all__` (≈48 names; 105 today, `__init__.py:367-409`):
`device load cluster run Dataset Result distance compute_distance_matrix DTWClustering system_check test` · Tier-2 `Problem Data ClusteringResult Method Solver` · `fast_pam fast_clara one_batch_pam tadpole build_dendrogram cut_dendrogram dtw_barycenter barycenter_kmeans` · `silhouette davies_bouldin dunn calinski_harabasz inertia adjusted_rand normalized_mutual_info` · `z_normalize derivative_transform soft_dtw_gradient` · `save_checkpoint load_checkpoint` · `DtwcError InvalidInput UndefinedScore SolverError DeviceError IOError DEFAULT_RANDOM_SEED __version__`. Strings spell every other enum (`variant="msm"`, `set_device("gpu:1")`); `Method`/`Solver` stay because `Problem.method`/`set_solver` are v1 fields. Gone: everything else in today's list (chair §3 old→new), plus `distance.standard/ddtw/…` aliases, `io.*`, `convert`, `preprocess/diagnose/features`, `check_system`, `get_device`, `ClusterResult`, `plot` (module-level).

`pyproject.toml`: `requires-python >= 3.10` (D-13), extras `plot = ["matplotlib"]`, `sklearn`, `hdf5` (owner Q2), `test`. `dtwc-convert` script removed.

### B.3 MATLAB (`+dtwc`, 1-based, `int64`, name-value pairs, snake_case Config keys)

```matlab
dtwc.device('gpu:1');  name = dtwc.device();                      % 'hpc' → dtwc:deviceError naming Python/CLI
data = dtwc.load('cycles/', 'skip_cols', 1);                       % path, N×L matrix, or cell of vectors (ragged)
res  = dtwc.cluster(data, 8, 'band', 1500, 'variant', 'msm');      % any Config key → one MEX 'run' command over parse_config pairs
res  = dtwc.run('input', 'x.parquet', 'k', 3, 'output', 'out');
res.labels; res.medoids; res.cost; res.method; res.iterations; res.converged; res.device; res.config   % struct
res.score('silhouette'); res.save('out'); res.plot(); D = res.distance_matrix;                        % MEX Result_distance_matrix
D = dtwc.compute_distance_matrix(X, 'band', 10, 'metric', 'squared_euclidean', 'device', 'gpu');
d = dtwc.distance.dtw(x, y, 'variant', 'msm', 'msm_c', 0.5);
est = dtwc.DTWClustering('n_clusters', 3, 'n_init', 3); est.fit(X); est.labels_;
info = dtwc.system_check(); dtwc.test.parallelisation(); dtwc.test.gpu();
```
Errors: `dtwc:invalidInput`, `dtwc:undefinedScore`, `dtwc:solverError`, `dtwc:deviceError`, `dtwc:ioError` (the MEX catch chain, kept). Tier-2 as in B.6 with the same names. Files deleted: `+distance/{standard,ddtw,wdtw,adtw,soft_dtw,missing,arow}.m`, `+distance/private/validate_metric.m`, the five `*_index.m`, `clarans.m`, `pdlp_*.m`, `save/load_binary_checkpoint.m`, `check_system.m`, `Dataset.materialize`, `test_mex.m`; `Problem.m` loses PascalCase properties and `*Value` caches. Release: the MEX (CPU) is attached to the GitHub release; the GPU MEX is a documented one-line build.

### B.4 CLI

```sh
dtwc_cl -i cycles/ -k 8 --band 1500 --device gpu -o out            # the whole Tier-1 in one line
dtwc_cl -i series.csv -k 3 --skip-cols 1 --seed 7 --print-config > job.toml
dtwc_cl --config job.toml --band 20                                # flags beat the file; unknown key = error (config.cpp:239)
```
`job.toml` (what `--print-config` writes, one `key = value` per Config key):
```toml
input = "series.csv"
output = "out"
n-clusters = 3
method = "auto"
band = 20
device = "gpu"
skip-cols = 1
seed = 7
```
v1 spellings (`--Nc`, `--probName`, `--in/--out`, `--skipRows`, `--skipCols`, `--maxIter/--iter`, `--repeat/--Nrep…`, `--mip_solver`, `--bandwidth/--bandw`, `--distMat/--distances`) return as hidden warn-once aliases through the existing `--clusters` mechanism (config.cpp:257-264); `--Nc 3..5` → `InvalidInput("one k per run; loop in the shell")`.

hpc from the CLI = the wrapper, which now takes the file, not 20 positionals:
```sh
bash slurm_remote.sh test                      # ssh, sinfo, and `test -x $REMOTE/build-*/bin/dtwc_cl` (else: "run slurm_remote.sh build")
bash slurm_remote.sh submit job.toml           # uploads the toml (+ the input when it exists locally); GPU partition when device = "gpu"; prints "Job ID: N" and the result dir
bash slurm_remote.sh status
bash slurm_remote.sh download <name> <id>      # <name>_labels.csv, <name>_medoids.csv, logs/cluster_<id>.{out,err}
```
The job runs `dtwc_cl --config job.toml -o results/<name>_<id>`; an unknown key from a newer laptop wheel is CLI11's loud error in the `.err` tail.

### B.5 Device semantics

| device | reads | matrix | precision | `auto` method | refused, typed, before I/O |
|---|---|---|---|---|---|
| `cpu` | heap, `dtype` f64/f32; > RAM only as list-per-row Parquet + `ram_limit` streamed by CLARA | packed dense below `mmap_threshold`, else the one `.dtwm` mapped file (llfio build) or `IOError` naming `mmap_threshold`/`onebatch`/`clara` | `dtype` | `pam` N ≤ 5000, else `clara` | `ram_limit` on a non-streamable input; `checkpoint` with streaming; band < length gap (Problem.cpp:1006) |
| `gpu`, `gpu:N` (`cuda[:N]` alias) | heap f64 | GPU writes the packed matrix, int64-chunked (CUDA, else Metal, else `DeviceError` naming the flag **and the artefact**) | `gpu_precision auto` = f32 when FP64 rate < ½ FP32 (one attribute read), else f64; Metal f32 only; `Result.config` shows the resolved value | **same rule as cpu**: `pam` N ≤ 5000, else `clara` | non-standard variant, missing strategy, ndim > 1, f32 storage, Metal with N ≠ 0 or f64 (Problem.cpp:1049-1066 wording); `onebatch`/`tadpole`; **`clara` until the rectangular sample×all GPU assignment lands (W13)** — the refusal names N and that limit; never a zero matrix, never a CPU fallback |
| `hpc`, `hpc:gpu` | never locally; a path that exists locally is uploaded, otherwise forwarded as a cluster path (the submit line says which); in-memory series → TSV upload | remote | remote | remote `auto` | at `cluster()`, in order, each a `DeviceError` whose message is the fix: (1) no `.env` in the working directory or `$DTWC_REPO_ROOT` → the three keys with a three-line example inline; (2) `SLURM_HOST is not set in .env` (wrapper text, re-raised typed); (3) `ssh -o BatchMode=yes user@host` fails → "must succeed without a password prompt; add your key to ssh-agent / authorized_keys"; (4) no `dtwc_cl` on the cluster → "run `slurm_remote.sh build`"; (5) `sacct` state ≠ COMPLETED → state + exit code + the last 20 lines of `logs/cluster_<id>.err`; (6) timeout/Ctrl-C → the job id and the `download` command. C++/MATLAB `device("hpc")`: one `DeviceError` naming Python and the wrapper |

What the library decides, and nothing else: the method for `auto` (one rule, in `run()`), dense vs mapped storage (`mmap_threshold`), GPU precision (`auto`), the CUDA/Metal backend, the SLURM partition (`device` in the toml), the reader (extension). What it never does: change the method, the device, the precision or the storage the user named.

### B.6 Tier-2 escape hatch (C++ names; Python/MATLAB identical, snake_case)

- `Problem(name)`, `set_data(Data)`, `size()`, `series(i)`, `series_name(i)`; v1 public fields `method, maxIter, N_repetition, band, init_fun, clusters_ind, centroids_ind` (`vector<index_t>`, Q3).
- Distance semantics, private behind setters that invalidate the matrix: `set_distance(DistanceConfig)`, `set_band`, `set_metric`, `set_variant`, `set_missing_strategy`, `set_device(Device, index=0)`, `set_gpu_precision`.
- Matrix: `fill_distance_matrix()`, `dist_by_ind(i, j)`, `is_distance_matrix_filled()` (bool), `distance_matrix()`, `set_distance_matrix(N×N)`, `read/write_distance_matrix`, `use_mmap_distance_matrix(path)`, `save_checkpoint(prob, dir)` / `load_checkpoint(prob, dir)` (one binary layout).
- Clustering: `set_n_clusters`, `set_solver`, `set_random_seed`, the five knob setters (`sample_size, n_samples, batch_size, linkage, tadpole_dc`), `cluster()` (dispatches all nine `Method` values), `set_result(ClusteringResult)`, `labels()`, `medoids()`, `find_total_cost()`; v1 camelCase shims one release (+ `cluster_by_kMedoidsPAM`).
- Algorithms (7): `fast_pam(prob, k, max_iter=100, seed=42)`, `fast_clara(prob, CLARAOptions)`, `one_batch_pam(prob, OneBatchPAMOptions)`, `tadpole(prob, k, dc)`, `build_dendrogram` + `cut_dendrogram`, `dtw_barycenter`, `barycenter_kmeans`.
- Scores (7): `silhouette, davies_bouldin, dunn, calinski_harabasz, inertia, adjusted_rand, normalized_mutual_info`.
- Pairwise: `distance::dtw(x, y, DistanceConfig)`; the unchecked `warping*.hpp` kernels stay public C++, documented as unchecked.
- `system_check()`.

### B.7 Changes relative to chair §3, one line of why each

1. **`device("hpc")` in C++/MATLAB refuses at the call, not "records then refuses at run"**: recording a device nothing honours is the silent state the charter forbids, and there is nothing to defer (credentials are Python's); one less state to carry.
2. **`dist_matrix` accepts an N×N array in Python/MATLAB (path in the CLI)**: the k sweep is the first thing every user does and today refills N² per call (`_api.py:576-598`); one existing key, ~10 lines, and `DTWClustering(metric="precomputed")` is the same mechanism.
3. **`auto` on `gpu` = the cpu rule (`pam` ≤ 5000, else `clara`)**: `pam` on `gpu` at 10 M series is a 400 TB matrix; one rule, and the GPU-CLARA rectangular assignment becomes a named W13 deliverable instead of a DECISIONS footnote — without it `hpc:gpu` is fiction at the charter's scale.
4. **No per-variant distance aliases in Python/MATLAB (`distance.dtw(x, y, variant=…)` is the one function)**: seven never-released names that restate a kwarg; C++ keeps its templates for zero overhead.
5. **hpc path rule: exists locally → upload, else cluster path, stated on the submit line**: today a laptop path is forwarded verbatim (`_hpc.py:591-593`) and fails invisibly inside the job.
6. **hpc failures typed and surfaced**: all transport errors `DeviceError` (today `RuntimeError`/`FileNotFoundError`, `_hpc.py:411-545`); `wait()` reads `sacct` state and fetches the `.err` tail (today a failed job is a `FileNotFoundError` for labels, `:542-545`); the job id + `download` command printed at submit and repeated on timeout/Ctrl-C; medoids fetched with labels (free); `dtwc_cl` presence checked in the ssh probe (today discovered after queueing, cluster_generic.slurm:73).
7. **One `.env` text, owned by Python/wrapper, no "repository root"/"source checkout" wording**: pip users have neither; the C++ copy (env.cpp:77-101) is unreachable from Python today and dies with `Env`.
8. **`Config` stays flat; `DistanceConfig` is built in `run()`**: flat keys are the CLI/TOML/kwargs spelling and designated initialisers of a nested aggregate are unusable (`{.distance = {.band = 10}}`); iface.md §3's nested field is dropped.
9. **`gpu_not_built_message` names the artefact** ("this dtwc_mex" / "this wheel" / "this dtwc_cl") beside the CMake flag: for a MEX or wheel user "rebuild" is a different action, and the CPU-only released MEX is what user (b) will hit first.
10. **`load()` absorbs `data_from_arrow_c_array` and (if kept, Q2) HDF5 by extension**: one loading noun; polars/pyarrow tables enter through `load(df)`, no second name.
11. **Python exports `Method`/`Solver` only; every other enum is a string**: `GpuPrecision`/`MetricType`/`DTWVariant`/… are spelled by the setters' strings and the Config keys; exporting them adds names with no caller.
12. **`plot(path=None, show=None)` + `[plot]` extra**: today `plot()` writes `./clusters_2d.png` unconditionally (`_api.py:363`) and `matplotlib` is an undeclared dependency.
13. **Python `run(**config)` is exported beside `cluster()`**: it is the bound function `cluster()` already calls; `dtwc.run(**res.config)` reproduces any run, including one written by `--print-config`.
14. **Release gate, not code: wheels + CPU MEX attached to the GitHub release, `pip install dtwcpp` resolving, GPU MEX build documented in one line**: users (a) and (b) cannot start otherwise (python-wheels.yml:97, matlab-mex.yml:62-65 upload CI artifacts only).

Kept from chair §3 unchanged: `device → load → cluster → Result` in four front ends over one `Config`; `hpc:gpu`; `method=auto`, `k` required, `name` = input stem; `Result` members and `save` filenames; `compute_distance_matrix` name; `test.parallelisation()/gpu()` wrappers; `cuda[:N]` alias; `Method` nine values; `set_device/set_gpu_precision`; `index_t`; v1 C++ and CLI shims; D-10.