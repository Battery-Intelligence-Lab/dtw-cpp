---
title: "Tier 1: device, load, cluster, Result"
weight: 10
description: "The high-level API: the same four steps in C++, Python and MATLAB."
---

# Tier 1: device, load, cluster, Result

Tier 1 is four steps, spelled the same way in every language: choose a device,
wrap the data, cluster it, read the result. C++ `cluster()` runs `dtwc::run`, as
`dtwc_cl` does; Python and MATLAB hand the same `dtwc::Config` to `dtwc::apply`
and run `Problem::cluster()`. So one input with one set of settings gives the
same labels, medoids and cost in each. C++ and Python count from 0; MATLAB
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

The names are `cpu` (the default), `gpu`, `gpu:N`, `cuda` and `cuda:N`, in any
case; `cuda` is another spelling of `gpu`, which means CUDA where it is built and
Metal on macOS. Setting a device returns its canonical name, so `cuda:0` comes
back as `gpu`. An unknown name, or `gpu` on a build without a GPU backend,
raises `DeviceError` (Python also checks that the machine has a GPU); no request
falls back to the CPU. Python also takes `hpc`
and `hpc:gpu`, which send the whole run to a SLURM cluster
([SLURM](../../getting-started/slurm/)); C++ and MATLAB refuse them with
`DeviceError`.

The process device is what `cluster()` uses when it is not given one. A Tier-2
`Problem` keeps its own device and ignores it ([Devices](../../guides/devices/)).

## Wrap the data: load

| | C++ | Python | MATLAB |
|---|---|---|---|
| call | `dtwc::load(source, skip_cols = 0, skip_rows = 0, delimiter = 0, name = "")` | `dtwcpp.load(source, *, skip_cols=0, skip_rows=0, delimiter=None, name=None)` | `dtwc.load(source, 'SkipCols', 0, 'SkipRows', 0, 'Delimiter', '', 'Name', '')` |
| a path | `std::filesystem::path` | `str` or `os.PathLike` | char or string |
| series in memory | `std::vector<std::vector<double>>`, one series per row, any lengths | a 2-D array (one series per row), a list of 1-D arrays (any lengths), a pandas DataFrame (rows, named by its index) or an Arrow array | an N×L numeric matrix (one series per row) or a cell of numeric vectors (any lengths) |

`load` reads nothing: the file is read when `cluster()` runs. A path is a
CSV, TSV or TXT file, a folder (one series per file), a Parquet file or an Arrow
IPC file. Text files and folders are read by the C++ reader `dtwc_cl` uses, in
every language. Parquet and Arrow IPC are read by C++ in a build with Arrow
(`-DDTWC_ENABLE_ARROW=ON`), by Python through the installed `pyarrow`, and by
MATLAB, Parquet only, through `parquetread`.

- `skip_cols` drops the leading fields of each line (an id column), the leading
  columns of a Parquet file, or the leading values of each series in memory.
- `skip_rows` drops leading lines of a file (a header), of each file in a folder,
  the leading rows of a Parquet file, or leading series in memory.
- The delimiter is inferred from the extension unless given: a tab for `.tsv`
  and `.txt`, else a comma.
- `name` names the run and prefixes its result files. Without one it is the
  file's or folder's name without extension, and `dataset` for series in memory.
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

## Cluster

| | C++ | Python | MATLAB |
|---|---|---|---|
| call | `dtwc::cluster(data, k, method = "auto", band = -1, device = "", max_iter = 100)` | `dtwcpp.cluster(data, k, **keys)` | `dtwc.cluster(data, k, Name, Value, ...)` |
| `data` | a `Dataset` | a `Dataset`, a path or series in memory | a `dtwc.Dataset`, a path or series in memory |
| other settings | through `dtwc::run(Config)`, below | keyword arguments | name-value pairs |

The settings are `dtwc_cl`'s long options that are not about files: snake_case
keywords in Python, the same words in CamelCase in MATLAB. They are `method`,
`band`, `metric`, `variant` and its parameters (`wdtw_g`, `adtw_penalty`,
`sdtw_gamma`, `msm_c`, `twe_nu`, `twe_lambda`), `mv_mode`, `missing_strategy`,
`max_iter`, `n_init`, `seed`, `sample_size`, `n_samples`, `batch_size`,
`linkage`, `dc`, `solver`, `mip_gap`, `time_limit`, `no_warm_start`,
`numeric_focus`, `mip_focus`, `verbose_solver`, `lr_max_nodes`, `device`,
`gpu_precision`, `name` and `verbose` (MATLAB: `'Method'`, `'Band'`, `'WdtwG'`,
`'MaxIter'`, `'NInit'`, ...). The [CLI reference](../../getting-started/cli/)
gives each one's meaning and default. C++ reads and checks the values: an unknown
key, or a name no table holds, is `InvalidInput` (MATLAB `dtwc:invalidArgument`)
naming the valid ones, an unknown device is `DeviceError`, and in Python a value
of the wrong type is `TypeError`.

`method` is one of `auto` (the default), `pam`, `onebatch`, `clara`, `kmedoids`,
`mip`, `lrcore`, `tadpole` and `hierarchical` (also `obp`, `lr` and `hclust`), in
any case. `auto` is `pam` for up to 5000 series on the CPU and `clara` above
that; on a GPU it is `pam`. A `device` given to `cluster()` is for that run only.
On a GPU, `pam`, `kmedoids`, `mip`, `lrcore` and `hierarchical` run with the GPU
filling the distance matrix and `clara` with the GPU filling its samples, while
`onebatch` and `tadpole`, which compute on the CPU as they go, raise
`DeviceError`. `k` below 1 or above the number of series, and an empty dataset,
are `InvalidInput`.

In Python, `device="hpc"` or `"hpc:gpu"` submits the run as one `job.toml` and
returns a `Result` holding the labels only; `gpu_device` (`"a100"`, `"a6000"`,
`"l40s"`, `"h100"`) picks the GPU of an `hpc:gpu` run.

### Every setting in C++: Config

C++ `cluster()` takes `k` and four settings. Any other goes through
`dtwc::Config`, the struct that `dtwc_cl`'s flags and its `--config` TOML file
fill: its fields follow the long options (`k` is `--n-clusters`, `tadpole_dc` is
`--dc`), and each starts at `dtwc_cl`'s default. `dtwc::run(config)` reads `config.input`; `dtwc::run(config, data)`
clusters series already in memory. Both write the result files into
`config.output` (`./results` unless cleared). From `examples/cpp/tier1.cpp`:

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
| plot | (plot the files `save` writes) | `plot(png="clusters_2d.png", show=True)` | `plot()` |
| run details | `method()` (`auto` resolved), `iterations()`, `converged()` | `summary()`, `elapsed_s`, `k`, `n_series`, `name` | |

`score(name)` takes `silhouette` (the mean over the series), `davies_bouldin`,
`dunn`, `calinski_harabasz` or `inertia`, in any case; another name is
`InvalidInput`. Where fewer than two clusters are non-empty, `silhouette`,
`dunn` and `davies_bouldin` raise `UndefinedScore` and `calinski_harabasz` raises
`InvalidInput`, as `davies_bouldin` does for one cluster. `onebatch`, `clara` and `tadpole` do not
build the N×N matrix; `score`, `distance_matrix` and `save` fill it the first
time they need it.

`save(dir)` writes, with `dtwc_cl`'s writer and byte for byte as it writes them,
into `dir` (created if missing): `<name>_labels.csv` (`name,cluster`), `<name>_medoids.csv`
(`cluster,medoid_index,medoid_name`), `<name>_distance_matrix.csv` and
`<name>_silhouettes.csv` (`name,cluster,silhouette`), with the series' names
from the input. `save` fills the matrix first, where `dtwc_cl` writes the
matrix and silhouette files only when the method filled it (not after
`onebatch`, `tadpole` or a sampled `clara`). With one cluster there is no
silhouette file; when fewer than two clusters are non-empty, `save` prints a
warning and skips it. `plot` draws a
two-dimensional classical-MDS map of the distance matrix, coloured by cluster.

A Python `hpc` result holds the labels only: `score` and `save` raise
`InvalidInput` (the cluster's `dtwc_cl` wrote the files there), and `plot`
prints the cluster sizes.

## The scikit-learn estimator (Python and MATLAB)

`dtwcpp.DTWClustering` and `dtwc.DTWClustering` wrap one `Problem` in the
scikit-learn pattern; C++ has no counterpart.

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
| `dtwc::InvalidInput` | `dtwcpp.InvalidInput` (a `ValueError`) | `dtwc:invalidArgument` | an argument or input the call cannot take: an unknown name, a value out of range, empty data, a NaN or ±inf a distance does not accept, a matrix or cache made for other data |
| `dtwc::UndefinedScore` | `dtwcpp.UndefinedScore` (an `InvalidInput`) | `dtwc:invalidArgument` | a silhouette, Dunn or Davies–Bouldin score of a labelling with fewer than two non-empty clusters |
| `dtwc::SolverError` | `dtwcpp.SolverError` (a `RuntimeError`) | `dtwc:solverError` | a solver this build lacks, or a MIP or LR-core run that ends without a proven optimum |
| `dtwc::DeviceError` | `dtwcpp.DeviceError` (a `RuntimeError`) | `dtwc:deviceError` | a device name, a GPU the build or machine lacks, or a setting the device cannot compute |
| `dtwc::IOError` | `dtwcpp.IOError` (an `OSError`) | `dtwc:ioError` | a file that cannot be read, parsed or written, or a format this build cannot read |

In C++ they derive from `dtwc::Error`, a `std::runtime_error`; in Python from
`dtwcpp.DtwcError` and the built-in the table names, so `except ValueError`
catches `InvalidInput` and `UndefinedScore` only; MATLAB's base identifier is
`dtwc:error`.
