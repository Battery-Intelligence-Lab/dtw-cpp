---
title: "Tier 1: device, load, cluster, Result"
weight: 10
description: "The high-level API: the same four steps in C++, Python and MATLAB."
---

# Tier 1: device, load, cluster, Result

Tier 1 is four steps, spelled the same way in every language: choose a device,
wrap the data, cluster it, read the result. `cluster()` runs `dtwc::run` in
every language, as `dtwc_cl` does. So one input with one set of settings gives
the same labels, medoids and cost in each. C++ and Python count from 0; MATLAB
counts from 1.

Each program below clusters the 27 series of
`tests/conformance/data/conformance_series.csv` into three groups with FastPAM
and a Sakoe-Chiba band of 3, prints the cost and the mean silhouette, and writes
the result files into `results/`. Run it from a repository checkout.

```cpp
#include <dtwc.hpp>

#include <iostream>

int main()
{
  dtwc::device("cpu");                                   // the process device
  auto data = dtwc::load("tests/conformance/data/conformance_series.csv");
  auto res = dtwc::cluster(data, 3, "pam", 3);           // k, method, band
  std::cout << "cost " << res.cost() << ", mean silhouette " << res.score("silhouette") << '\n';
  res.save("results");                                   // the files dtwc_cl writes
}
```

```python
import dtwcpp as dtwc

dtwc.device("cpu")
data = dtwc.load("tests/conformance/data/conformance_series.csv")
res = dtwc.cluster(data, k=3, method="pam", band=3)
print(f"cost {res.cost}, mean silhouette {res.score('silhouette')}")
res.save("results")
```

```matlab
dtwc.device('cpu');
data = dtwc.load('tests/conformance/data/conformance_series.csv');
res = dtwc.cluster(data, 3, 'Method', 'pam', 'Band', 3);
fprintf('cost %g, mean silhouette %g\n', res.cost, res.score('silhouette'));
res.save('results');
```

## Choose a device

| | C++ | Python | MATLAB |
|---|---|---|---|
| set | `dtwc::device("gpu")` | `dtwcpp.device("gpu")` | `dtwc.device('gpu')` |
| read | `dtwc::device()` | `dtwcpp.device()` | `dtwc.device()` |

The names are `cpu` (the default), `gpu`, `gpu:N`, `cuda`, `cuda:N`, `cpu+gpu`,
and `cpu+gpu:N`, in any case; `cuda` is another spelling of `gpu`, which means
CUDA where it is built and Metal on macOS, and `cpu+gpu` splits matrix fills and
FastCLARA assignment between the CPU and a CUDA GPU. Setting a device returns its
canonical name, so `cuda:0` comes back as `gpu`. An unknown name, `gpu` on a
build without a GPU backend, or `cpu+gpu` without CUDA raises `DeviceError`
(Python also checks that the machine has a GPU for `gpu`); no request falls back
to the CPU. Python also takes `hpc` and `hpc:gpu`, which send the whole run to a
SLURM cluster ([SLURM](../../getting-started/slurm/)); C++ and MATLAB refuse them
with `DeviceError`.

The process device is what `cluster()` uses when it is not given one. A Tier-2
`Problem` keeps its own device and ignores it ([Devices](../../guides/devices/)).

## Wrap the data: load

| | C++ | Python | MATLAB |
|---|---|---|---|
| call | `dtwc::load(source, skip_cols = 0, skip_rows = 0, delimiter = 0, name = "", dtype = core::Precision::Float64)` | `dtwcpp.load(source, *, skip_cols=0, skip_rows=0, delimiter=None, name=None, dtype="float64")` | `dtwc.load(source, 'SkipCols', 0, 'SkipRows', 0, 'Delimiter', '', 'Name', '', 'Dtype', 'float64')` |
| a path | `std::filesystem::path` | `str` or `os.PathLike` | char or string |
| series in memory | `std::vector<std::vector<double>>`, one series per row, any lengths | a 2-D array (one series per row), a list of 1-D arrays (any lengths), a pandas DataFrame (rows, named by its index) or an Arrow array | an N×L numeric matrix (one series per row) or a cell of numeric vectors (any lengths) |

`load` reads nothing: the file is read when `cluster()` runs. A path is a
CSV, TSV or TXT file, a folder (one series per file), a Parquet file or an Arrow
IPC file. Text files, folders and Arrow IPC files are read by the C++ reader
`dtwc_cl` uses, in every language and build (an Arrow IPC file is mapped, its series
views of the file; where it cannot be viewed in place, as a compressed file or in a build
without llfio, it is decoded or read into memory). Parquet is read by C++ in a build with Arrow
(`-DDTWC_ENABLE_ARROW=ON`), by Python through the installed `pyarrow`, and by
MATLAB through `parquetread`.

A `Dataset` of a path keeps no series: every `cluster()` call on it reads the
file again. The exception, in Python and MATLAB, is Parquet, which `pyarrow` or
`parquetread` reads once: the `Dataset` keeps its series. To read a file once for several
runs, read its series into memory first and cluster those: C++
`dtwc::run(config, dtwc::read_data(path))`, Python `dtwcpp.cluster(data.as_series(), k)`,
MATLAB `dtwc.cluster(data.as_series(), k)` (with `dtype` given again for float32). The
clustering is the path's, with two differences in Python and MATLAB. `cluster()` names
series in memory `0`, `1`, ... in the result files, not by the reader. And `cluster()`
takes no `ndim` for series in memory (MATLAB's `as_series()` returns it as a third output,
Python's returns the series only): an Arrow IPC file with `ndim` above 1 clusters as
multivariate from its path and as univariate series from `as_series()`. C++'s `read_data`
keeps both.

- `skip_cols` drops the leading fields of each line (an id column), the leading
  columns of a Parquet file, or the leading values of each series in memory.
- `skip_rows` drops leading lines of a file (a header), of each file in a folder,
  the leading rows of a Parquet file, or leading series in memory.
- The delimiter is inferred from the extension unless given: a tab for `.tsv`
  and `.txt`, else a comma.
- `name` names the run and prefixes its result files. Without one it is the
  file's or folder's name without extension, and `dataset` for series in memory.
- `dtype` is the precision the series are stored in: `float64` (the default) or
  `float32`, half the memory, with `dtwc_cl --dtype`'s spellings (`f32`, `fp32`,
  `float`; `f64`, `fp64`, `double`; any case; another is `InvalidInput`). Each
  reader rounds the double it reads, so a file gives the same float32 series in
  every language and no double copy is held. Series in memory are rounded when
  `cluster()` runs; float32 data in memory alone (NumPy `float32`, MATLAB `single`)
  does not choose it. The dataset says which: `Dataset::dtype()`, Python
  `Dataset.dtype`, MATLAB `Dataset.Dtype`.
- Each series has a name, which the result files carry: `1`, `2`, ... for the
  series of a text file, in the order read after `skip_rows`; the file's name
  without extension in a folder; `0`, `1`, ... for series in memory (in Python,
  a pandas DataFrame's index, or an Arrow array's own names, else `series_<i>`);
  Parquet and Arrow IPC files name theirs as
  [Supported data](../../getting-started/supported-data/) describes (a Parquet
  file's rows by its first string column, else `series_<i>`; a file whose one
  float column is one series, by the file).

A negative `skip_cols` or `skip_rows` is `InvalidInput`. In C++,
`load(path, 0, ',')` does not compile: give `skip_rows` before the delimiter,
`load(path, 0, 0, ',')`.

A `Dataset` is read with the reader options it was made with (`skip_cols`, `skip_rows`,
`delimiter` and `dtype`), so a call that gives another value for one is `InvalidInput` (MATLAB
`dtwc:invalidArgument`), naming the option and both values: pass it to `load` when the `Dataset`
is made, or leave it out. One equal to the `Dataset`'s own runs, and one not given is never
refused. In C++, `cluster(dataset, config)` and `cdist(X, Y, config)` read the options from the
`Config`, whose defaults stand for "not given": a field that is not at its default and differs is
refused, and so is a `column` beside series in memory (a C++ `Dataset` holds no `column`: a path's
read takes the `Config`'s). In Python and MATLAB, `cluster(dataset, k, dtype=...)` refuses a
`dtype` (`'Dtype'`) that was given and differs, its default value included, and so does MATLAB's
`load(dataset, ...)` for each of its options; Python's `load` cannot tell a keyword left at its
default from one not given, so it refuses a keyword only when it is not at its default. A `name`
given to `load` beside a `Dataset` is refused. `cdist` checks each `Dataset` by the same rule in
Python and MATLAB for `dtype` and `ndim` (Python's checks a `Data`'s `ndim` too); its `dtype`
(MATLAB `'Dtype'`) is not given unless passed (Python `None`, MATLAB `''`): a `Dataset`, or in
Python a `Data`, refuses one other than its own, and other series are stored as `float64` unless
it says otherwise. Once a `Dataset` has read an Arrow IPC file, the `ndim` `cdist` compares is the
file's; an `ndim` given for an Arrow IPC file whose metadata gives another `ndim` above 1 is
refused when the file is read, and given none, the series take the file's `ndim`.

## Cluster

| | C++ | Python | MATLAB |
|---|---|---|---|
| call | `dtwc::cluster(data, k, method = "auto", band = -1, device = "", max_iter = 100)`, or `dtwc::cluster(data, config)` | `dtwcpp.cluster(data, k, **keys)` | `dtwc.cluster(data, k, Name, Value, ...)` |
| `data` | a `Dataset` | a `Dataset`, a path or series in memory | a `dtwc.Dataset`, a path or series in memory |
| other settings | the fields of a `dtwc::Config`, below | keyword arguments | name-value pairs |

The settings are these `dtwc_cl` long options, as snake_case keywords in Python
and the same words in CamelCase in MATLAB: `method`, `band`, `metric`, `variant`
and its parameters (`wdtw_g`, `adtw_penalty`, `sdtw_gamma`, `msm_c`, `twe_nu`,
`twe_lambda`), `mv_mode`, `missing_strategy`, `max_iter`, `n_init`, `seed`,
`sample_size`, `n_samples`, `batch_size`, `linkage`, `dc`, `solver`, `mip_gap`,
`time_limit`, `no_warm_start`, `numeric_focus`, `mip_focus`, `verbose_solver`,
`lr_max_nodes`, `lr_root`, `device`, `gpu_precision`, `gpu_share`, `matrix_dtype`, `ram_limit`, `name` and `verbose`
(MATLAB: `'Method'`, `'Band'`, `'WdtwG'`, `'MaxIter'`, `'NInit'`, `'GpuShare'`, `'MatrixDtype'`, `'RamLimit'`, ...), and
`no_report` (MATLAB `'NoReport'`; `--no-report`: no run report). `dtype` (MATLAB
`'Dtype'`) is `load`'s: `cluster()` loads a path or series in memory in it, and
beside a `Dataset`, which keeps its own, refuses one that differs. `output`, `checkpoint` and
`checkpoint_interval` are the run's files, below. The others (`input`, `dist_matrix`, ...) are
`dtwc_cl`'s and `dtwc::run`'s, unknown keys here.
`ram_limit` is the run's soft memory budget, a byte count or a size such as `"8G"`, as
`--ram-limit` is: a reserve for the threads, then where the series and the matrix go (`0`, the
default, plans with 90 % of the memory available); a run that cannot fit under an explicit limit is
`InvalidInput`. The [CLI reference](../../getting-started/cli/) gives each one's meaning and default. C++ reads and checks the values: an unknown
key, or a name no table holds, is `InvalidInput` (MATLAB `dtwc:invalidArgument`)
naming the valid ones, an unknown device is `DeviceError`, and in Python a value
of the wrong type is `TypeError`.

`method` is one of `auto` (the default), `pam`, `onebatch`, `clara`, `kmedoids`,
`mip`, `lrcore`, `tadpole` and `hierarchical` (also `obp`, `lr` and `hclust`), in
any case. `auto` is `pam` for up to 5000 series on the CPU and `clara` above
that; on a GPU, and on `cpu+gpu`, it is `pam`. `auto` is also `clara` where `pam`'s distance matrix
does not fit the run's memory plan (`ram_limit`). A `device` given to `cluster()`
is for that run only.
On a GPU, `pam`, `kmedoids`, `mip`, `lrcore` and `hierarchical` run with the GPU
filling the distance matrix and `clara` with the GPU filling its samples, while
`onebatch` and `tadpole`, which compute on the CPU as they go, raise
`DeviceError`. CUDA reads the series of an Arrow IPC file, which are views, where they lie
(Metal's fill uploads an owned copy of them, which the memory plan counts). `device="cpu+gpu"`
(`'cpu+gpu'`, CUDA only) fills the matrix of those methods on the CPU's threads and the GPU
together and splits `clara`'s assignment pass between them; `onebatch`, `tadpole`, a run
that streams its series and a build without CUDA raise `DeviceError`. `gpu_share` is
the GPU's share of that split: `-1`, the default, measures it, a value in [0, 1] forces it so
that a run repeats, and anything else, or a forced value on another device, is
`InvalidInput` ([Devices](../../guides/devices/)). `k` below 1 or above the number of series, and an empty dataset,
are `InvalidInput`.

In Python, `device="hpc"` or `"hpc:gpu"` submits the run as one `job.toml` and
returns a `Result` holding the labels, medoids and run report; `gpu_device` (`"a100"`, `"a6000"`,
`"l40s"`, `"h100"`) picks the GPU of an `hpc:gpu` run. `upload_input=True` sends the local file or
folder a path names with the job, which otherwise reads an absolute path on the cluster
([SLURM](../../getting-started/slurm/)); `output` receives the downloaded files, and `checkpoint` and
`checkpoint_interval` are refused, since the cluster's run is its own.

### Result files and a checkpoint

`cluster()` writes no file unless it is asked to; `save` writes them on request. Three settings ask:

| | C++ (`Config` fields) | Python | MATLAB |
|---|---|---|---|
| the directory the run writes its result files into | `output` | `output` | `'Output'` |
| the directory it saves its distance matrix in, and resumes from | `checkpoint` | `checkpoint` | `'Checkpoint'` |
| the rows between saves of it | `checkpoint_interval` | `checkpoint_interval` | `'CheckpointInterval'` |

`output` receives the files `dtwc_cl` writes, as `--output` does, the run report last
(`<name>_report.txt`): the matrix and silhouette files only where the method filled the matrix,
where `save` fills it first. Leaving it out, `None` or `""` in Python and `''` in MATLAB writes
nothing. `checkpoint` saves the distance matrix there as `<name>.dtwm` and resumes from it where it
holds one for these data ([Checkpointing](../../getting-started/checkpointing/)); a method that computes
no distance matrix writes none. `checkpoint_interval` saves it every that many rows (0, the default:
once, at the end) and needs `checkpoint`: without one it is `InvalidInput`, and in Python and MATLAB so is
one below 0. In C++ they are
fields of the `Config` that `cluster(data, config)` takes, whose `output` starts at `dtwc_cl`'s
`./results`: clear it to cluster in memory, as the six-value `cluster()` does.

```cpp
dtwc::Config config;
config.k = 3;
config.output = "results";         // the run writes its files here ("" writes none)
config.checkpoint = "checkpoints"; // and saves its distance matrix here, resuming from it
const auto res = dtwc::cluster(data, config);
```

```python
res = dtwcpp.cluster(data, 3, output="results", checkpoint="checkpoints")
```

```matlab
res = dtwc.cluster(data, 3, 'Output', 'results', 'Checkpoint', 'checkpoints');
```

To run the same settings again, keep them: a `Config` in C++, a dict in Python
(`dtwcpp.cluster(data, 3, **opts)`; to override one, merge, `**{**opts, "band": 2}`, since Python
refuses a keyword given twice), a cell array in MATLAB (`dtwc.cluster(data, 3, opts{:}, 'Band', 2)`:
a name given twice takes its last value).

### Every setting in C++: Config

C++ `cluster(data, k, ...)` takes `k` and four settings. Any other goes through
`dtwc::Config`, the struct that `dtwc_cl`'s flags and its `--config` TOML file
fill: its fields follow the long options (`k` is `--n-clusters`, `tadpole_dc` is
`--dc`), and each starts at `dtwc_cl`'s default. `dtwc::cluster(data, config)` clusters a `Dataset` with
it; `dtwc::run(config)` reads `config.input`, and `dtwc::run(config, data)` clusters series already in
memory. All three write the result files into
`config.output` (`./results` unless cleared), and the run report last
(`<name>_report.txt`; `config.report = false`, `dtwc_cl --no-report`, makes none).
From `examples/cpp/tier1.cpp`:

```cpp
dtwc::Config config;   // every setting starts at dtwc_cl's default
config.k = 2;          // --n-clusters, the one setting a run needs
config.method = dtwc::Method::PAM;
config.band = 1;       // --band: a Sakoe-Chiba window of one step
config.output.clear(); // write no files; Result::save(dir) writes them on request

const dtwc::Result result = dtwc::run(config, dtwc::Data{ std::move(series), std::move(names) });
```

## Read the result

| | C++ `dtwc::Result` | Python `dtwcpp.Result` | MATLAB `dtwc.Result` |
|---|---|---|---|
| cluster of each series | `labels()` | `labels` | `labels` (1-based) |
| medoid of each cluster | `medoids()`, series indices | `medoids` | `medoids` (1-based) |
| total distance to the medoids | `cost()` | `cost` | `cost` |
| where it ran | `device()` | `device` | `device` |
| a quality score | `score(name)` | `score(name)` | `score(name)` |
| the N×N distances | `distance_matrix()`, row-major | `distance_matrix` | `distance_matrix()` |
| write the result files | `save(dir)` | `save(dir)` | `save(dir)` |
| the run report | `report()`: the text of `<name>_report.txt` | `report` (`None` for an `hpc` run that wrote none) | `Report` |
| plot | (plot the files `save` writes) | `plot(png="clusters_2d.png", show=True)` | `plot()` |
| run details | `method()` (`auto` resolved), `iterations()`, `converged()` | `summary()`, `elapsed_s`, `k`, `n_series`, `name` | |

`score(name)` takes `silhouette` (the mean over the series), `davies_bouldin`,
`dunn`, `calinski_harabasz` or `inertia`, in any case; another name is
`InvalidInput`. Where fewer than two clusters are non-empty, `silhouette`,
`dunn`, `davies_bouldin` and `calinski_harabasz` raise `UndefinedScore`. `onebatch`, `clara` and `tadpole` do not
build the N×N matrix; `score`, `distance_matrix` and `save` fill it the first
time they need it. Under `matrix_dtype` float32, `distance_matrix` is a NumPy
`float32` array in Python and `single` in MATLAB; C++ returns doubles.

`save(dir)` writes, with `dtwc_cl`'s writer and byte for byte as it writes them,
into `dir` (created if missing): `<name>_labels.csv` (`name,cluster`), `<name>_medoids.csv`
(`cluster,medoid_index,medoid_name`), `<name>_distance_matrix.csv` and
`<name>_silhouettes.csv` (`name,cluster,silhouette`), with the series' names
from the input. Under `softdtw` the matrix, here and in `distance_matrix`, holds the Soft-DTW
divergence the clustering uses ([DTW variants](../../method/dtw-variants/#soft-dtw)), not
`cdist`'s raw value. `save` then writes the run report, last, as `<name>_report.txt`
(the run's own: its times and machine; a run without an output directory, as
`cluster()` is unless it is given `output`, reports its silhouettes not computed though `save` computes them;
a `save` that fails before it writes none; none under `config.report = false`,
Python's `no_report=True`, MATLAB's `'NoReport', true`). `save` fills the matrix first, where `dtwc_cl` writes the
matrix and silhouette files only when the method filled it (not after
`onebatch`, `tadpole` or a sampled `clara`). With one cluster there is no
silhouette file; when fewer than two clusters are non-empty, `save` prints a
warning and skips it. `plot` draws a
two-dimensional classical-MDS map of the distance matrix, coloured by cluster.

A run that streamed its series (an Arrow IPC file, or with Arrow C++ a Parquet
file, beyond the memory plan's data cap: `ram_limit`) holds none: its `score` and
`distance_matrix` (so `plot`) raise `InvalidInput`, and `save` writes the labels and
medoids, with the reader's names, as `dtwc_cl` writes them, then the report.

A Python `hpc` result downloads the labels, medoids and `<name>_report.txt` from
the remote run: `score` raises `InvalidInput` because the series and distance
matrix remain on the cluster (where `dtwc_cl` already wrote the output files),
`save` copies the downloaded files into its directory, and `plot` prints the
cluster sizes.

## Cross-collection distances: cdist

`cdist` computes the rectangular `M × N` pairwise DTW distance matrix between
two collections `X` (`M` series) and `Y` (`N` series) without clustering:

| | C++ | Python | MATLAB |
|---|---|---|---|
| call | `dtwc::cdist(X, Y, opts)` | `dtwcpp.cdist(X, Y, *, band=-1, variant="standard", metric="l1", device=None, dtype=None, matrix_dtype="float64", ...)` | `dtwc.cdist(X, Y, 'Band', -1, 'Variant', 'standard', 'Device', 'cpu', 'Dtype', '', 'MatrixDtype', 'float64', ...)` |
| returns | `std::vector<double>` of `M × N` values, row-major: element `i * N + j` is the distance from `X[i]` to `Y[j]` | `np.ndarray` of shape `(M, N)` (`float64` or `float32`) | `M × N` numeric matrix (`double` or `single`) |

Standard univariate DTW dispatches across OpenMP + pre-packed SIMD lanes on
`cpu`, CUDA wavefront/regtile kernels on `gpu`, and row-partitioned concurrent
execution on `cpu+gpu` (`gpu_share`). All DTW variants (`ddtw`, `wdtw`, `adtw`,
`softdtw`, `msm`, `twe`), missing-value strategies, and multivariate modes
(`dependent`, `independent`) run in parallel on the CPU.

## The scikit-learn estimator (Python and MATLAB)

`dtwcpp.DTWClustering` and `dtwc.DTWClustering` wrap one `Problem` in the
scikit-learn pattern and cluster it in RAM (only `cluster()` runs `dtwc::run`); C++ has no
counterpart.

```python
import numpy as np
import dtwcpp

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 1.0, (10, 40)) for c in (0, 5, 10)])   # 30 series

est = dtwcpp.DTWClustering(n_clusters=3, band=10, n_init=3).fit(X)
print(est.labels_, est.medoid_indices_, est.inertia_)   # also cluster_centers_, n_iter_
print(est.predict(X[:2] + 0.1))                          # the nearest fitted medoid
```

```matlab
X = [randn(10, 40); randn(10, 40) + 5; randn(10, 40) + 10];   % 30 series
est = dtwc.DTWClustering('NClusters', 3, 'Band', 10, 'NInit', 3);
est = est.fit(X);
disp(est.Labels); disp(est.MedoidIndices); disp(est.Inertia);
disp(est.predict(X(1:2, :) + 0.1));
```

The parameters are `n_clusters` (3), `method` (`"pam"`), `variant`, `band`,
`max_iter`, `n_init`, `wdtw_g`, `adtw_penalty`, `msm_c`, `twe_nu`, `twe_lambda`,
`mv_mode`, `missing_strategy`, `metric`, `batch_size`, `random_state` and
`device`, with `cluster()`'s defaults; MATLAB's properties are the same words in
CamelCase (`NClusters`, `Method`, ..., `RandomState`, `Device`) and its fitted
values are `Labels`, `MedoidIndices`, `Inertia` and `ClusterCenters`. `fit` runs
`n_init` restarts on one distance matrix, restart `i` seeded `random_state + i`,
and keeps the lowest cost. `transform` gives the distances to the fitted
medoids, `predict` the nearest one, and `score(X)` the negative total distance
to them; none refits. Python also takes `metric="precomputed"`, where `X` is a
distance matrix: N×N to fit, M×N to predict.

## Seeds

A seeded method starts from 42 unless told otherwise
(`dtwc::settings::DEFAULT_RANDOM_SEED`, `dtwcpp.DEFAULT_RANDOM_SEED`,
`dtwc.default_random_seed()`); the setting is `seed` (`'Seed'`, `--seed`).
Restart `r` of `n_init` starts from `seed + r`. One seed gives one result on
every platform: the library turns `std::mt19937_64` output into choices with its
own maps rather than the standard library's distributions, whose output differs
between implementations.

## Errors

Every error the C++ library raises is one of five types, and each binding raises
its own form of it; Python also raises `TypeError` for an argument of the wrong
type.

| C++ | Python | MATLAB identifier | raised for |
|---|---|---|---|
| `dtwc::InvalidInput` | `dtwcpp.InvalidInput` (a `ValueError`) | `dtwc:invalidArgument` | an argument or input the call cannot take: an unknown name, a value out of range, empty data, a NaN or ±inf a distance does not accept, a matrix or cache made for other data; also a `dtwc::run` whose memory the system refuses, naming its stage, its memory plan and the process's peak |
| `dtwc::UndefinedScore` | `dtwcpp.UndefinedScore` (an `InvalidInput`) | `dtwc:invalidArgument` | a silhouette, Dunn, Davies–Bouldin or Calinski–Harabasz score of a labelling with fewer than two non-empty clusters |
| `dtwc::SolverError` | `dtwcpp.SolverError` (a `RuntimeError`) | `dtwc:solverError` | a solver this build lacks, or a MIP or LR-core run that ends without a proven optimum |
| `dtwc::DeviceError` | `dtwcpp.DeviceError` (a `RuntimeError`) | `dtwc:deviceError` | a device name, a GPU the build or machine lacks, or a setting the device cannot compute |
| `dtwc::IOError` | `dtwcpp.IOError` (an `OSError`) | `dtwc:ioError` | a file that cannot be read, parsed or written, or a format this build cannot read |

In C++ they derive from `dtwc::Error`, a `std::runtime_error`; in Python from
`dtwcpp.DtwcError` and the built-in the table names, so `except ValueError`
catches `InvalidInput` and `UndefinedScore` only; MATLAB's base identifier is
`dtwc:error`. Memory the system refuses is no fault of an argument or a file:
`load()` and the Tier-2 calls pass C++'s `std::bad_alloc` as it is (Python
`MemoryError`, as Python's `load()` raises for memory pyarrow is refused reading
Parquet; MATLAB `dtwc:internal`; Gurobi's MIP adapter reports it as a
`SolverError`), and `cluster()` reports it within the run as the `InvalidInput`
above (MATLAB `dtwc:invalidArgument`). Python reads Parquet, and copies series in
memory, before the run: `MemoryError` there; MATLAB's MEX copies series in memory
(Parquet's among them, which `parquetread` reads) before the run: `dtwc:internal`
there.
