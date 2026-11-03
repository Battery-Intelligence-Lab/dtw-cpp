---
title: "Tier 2: Problem and advanced APIs"
weight: 20
description: "Problem, the clustering algorithms, the scores, the distance functions and the .dtwm cache, in C++, Python and MATLAB."
---

# Tier 2: Problem and advanced APIs

Tier 2 is the object Tier 1 builds for you: a `Problem` holds the series, the
distance settings and the distance matrix, and the algorithms, the scores and the
writers work on it. Use it to run one algorithm with its own options, to reuse
one distance matrix across several clusterings, or to cache the matrix on disk.
C++ and Python count from 0; MATLAB counts from 1.

```cpp
#include <dtwc.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main()
{
  std::vector<std::vector<dtwc::data_t>> series{ { 0.0, 0.1, 0.2 }, { 0.1, 0.0, 0.2 }, { 0.2, 0.2, 0.0 },
                                                 { 9.0, 9.1, 9.2 }, { 9.1, 9.0, 9.2 }, { 9.2, 9.2, 9.0 } };
  std::vector<std::string> names{ "a1", "a2", "a3", "b1", "b2", "b3" };

  dtwc::Problem prob("demo");
  prob.set_data(dtwc::Data{ std::move(series), std::move(names) });
  prob.set_band(1);
  prob.fill_distance_matrix();

  const auto result = dtwc::fast_pam(prob, 2);           // k; max_iter = 100, seed = 42
  std::cout << "medoids " << result.medoid_indices[0] << ' ' << result.medoid_indices[1]
            << ", cost " << result.total_cost << ", Davies-Bouldin "
            << dtwc::scores::davies_bouldin(prob) << '\n';
}
```

```python
import dtwcpp

series = [[0.0, 0.1, 0.2], [0.1, 0.0, 0.2], [0.2, 0.2, 0.0],
          [9.0, 9.1, 9.2], [9.1, 9.0, 9.2], [9.2, 9.2, 9.0]]

prob = dtwcpp.Problem("demo")
prob.set_data(series, ["a1", "a2", "a3", "b1", "b2", "b3"])
prob.set_band(1)
prob.fill_distance_matrix()

result = dtwcpp.fast_pam(prob, 2)                      # max_iter=100, seed=42
print(list(result.medoid_indices), result.total_cost, dtwcpp.davies_bouldin(prob))
```

```matlab
prob = dtwc.Problem('demo');
prob.set_data([0 0.1 0.2; 0.1 0 0.2; 0.2 0.2 0; 9 9.1 9.2; 9.1 9 9.2; 9.2 9.2 9], ...
              {'a1', 'a2', 'a3', 'b1', 'b2', 'b3'});
prob.set_band(1);
prob.fill_distance_matrix();

result = dtwc.fast_pam(prob, 2);                      % 'MaxIter', 100, 'Seed', 42
disp(result.medoid_indices); disp(result.total_cost); disp(dtwc.davies_bouldin(prob));
```

## Problem

| | C++ `dtwc::Problem` | Python `dtwcpp.Problem` | MATLAB `dtwc.Problem` |
|---|---|---|---|
| create | `Problem("name")`, `Problem("name", loader)` (a `DataLoader`) | `Problem("name", device="cpu")` | `dtwc.Problem('name', 'Device', 'cpu')` |
| series | `set_data(Data)` | `set_data(series, names=None, ndim=1)` | `set_data(X, names, ndim, dtype)` |
| read | `size()`, `series(i)`, `series_name(i)`, `data()` | `size`, `series(i)`, `series_name(i)` | `size()` |
| k | `set_n_clusters(k)`, `n_clusters()` | the same | the same |
| method of `cluster()` | `set_method(Method)`, `method()` | `set_method(Method)`, `method` | `set_method(name)` |
| distance settings | `set_distance(core::DistanceConfig)`, `distance()` | `set_distance(variant=, band=, metric=, missing_strategy=, mv_mode=, wdtw_g=, ...)` | `set_distance('Variant', ..., 'Band', ..., ...)` |
| one distance setting | `set_band`, `set_variant`, `set_metric`, `set_missing_strategy` | `set_band`, `set_variant`, `set_variant_params`, `missing_strategy` | `set_band`, `set_variant(name, param)`, `set_missing_strategy` |
| device | `set_device(Device, index = 0)`, `device()`, `set_gpu_precision`, `gpu_precision()` | `set_device(name)`, `set_gpu_precision` | `set_device(name)`, `set_gpu_precision(name)` |
| iterations, restarts, seed | `set_max_iter`, `set_n_repetitions`, `set_random_seed` | the same, or `max_iter`, `n_repetitions`, `random_seed` | `set_max_iter`, `set_n_repetitions` |
| MIP solver | `set_solver(Solver)` (returns `bool`), `solver()`, `mip_settings` | `set_solver(Solver)`, `solver`, `mip_settings` | `set_solver(name)`, `set_mip_settings(s)` |
| CLARA, OneBatchPAM, hierarchical, TADPole | `set_sample_size`, `set_n_samples`, `set_batch_size`, `set_linkage`, `set_tadpole_dc` | | |
| output | `set_output_folder`, `set_verbose`, `set_name` | `output_folder`, `verbose`, `name` | `set_output_folder`, `set_verbose` |
| memory | `set_ram_limit(bytes)`, `ram_limit()` | `ram_limit` (bytes or `"8G"`) | `set_ram_limit(limit)`, `RamLimit` |
| matrix value type | `set_matrix_dtype(core::Precision)`, `matrix_dtype()` | `matrix_dtype` (`"float64"`, `"float32"`) | `set_matrix_dtype(name)` |

A distance setting is checked when it is set (`InvalidInput`, the setting left
as it was) and drops the distance matrix and the clustering, which describe the
old distances. The distance settings are the ones `cluster()` takes: `variant`
(`standard`, `ddtw`, `wdtw`, `adtw`, `softdtw`, `msm`, `twe`) and its parameters,
`band` (the Sakoe-Chiba half-width in steps, `-1` for none), `metric` (`l1` or
`squared_euclidean`, for Standard DTW and DDTW), `missing_strategy` (`error`,
`zero_cost`, `arow`, `interpolate`) and `mv_mode` (`dependent` or `independent`,
for multivariate series; MATLAB's `set_distance` and `dtwc.distance.dtw` do not take
`MvMode`, which `dtwc.cluster` and `DTWClustering` do). Multivariate series hold `ndim` values per time step,
interleaved. A `Problem` computes on the CPU until its device is set; it does
not follow the process device ([Devices](../../guides/devices/)).

The C++ fields `band`, `maxIter`, `N_repetition`, `init_fun`, `clusters_ind` and
`centroids_ind` stay public, as in v1.0.0, beside the new `mip_settings` and
`checkpoint`; a direct write is not checked, where the setters are.

### The distance matrix

| | C++ | Python | MATLAB |
|---|---|---|---|
| compute every pair | `fill_distance_matrix()` | `fill_distance_matrix()` | `fill_distance_matrix()` |
| one pair | `dist_by_ind(i, j)` | `dist_by_ind(i, j)` | `dist_by_ind(i, j)` |
| computed? | `is_distance_matrix_filled()` | `is_distance_matrix_filled()` | `is_distance_matrix_filled()` |
| the matrix | `distance_matrix()`: the packed `core::DistanceMatrix` | `distance_matrix()`: an N×N array, float64 or float32 | `distance_matrix()`: an N×N matrix, `double` or `single` |
| set it | `writable_distance_matrix()` | `set_distance_matrix(D)` | `set_distance_matrix(D)` |
| CSV in, out | `read_distance_matrix(path, not_computed)`, `write_distance_matrix()` | `read_distance_matrix(path, not_computed="nan")`, `write_distance_matrix()` | `read_distance_matrix(path, 'NotComputed', 'nan')` |
| forget it | `refresh_distance_matrix()` | `refresh_distance_matrix()` | `refresh_distance_matrix()` |
| largest distance | `max_distance()` | `max_distance()` | `max_distance()` |

The fill runs in parallel over pairs, on the `Problem`'s device. C++
`dist_by_ind` reads the matrix with no check and no computation, so a parallel
loop can call it: fill the matrix first, as every method that reads it does.
Python and MATLAB check the indices, and their `distance_matrix()` fills the
matrix first and returns a copy at the matrix's value type (`matrix_dtype`, below);
`set_distance_matrix` takes either type and stores it at that one. A matrix that is read or set must be the
`Problem`'s size and finite, and not negative under any variant but Soft-DTW;
NaN marks a pair still to compute. v1.0.0 wrote `-1` for such a pair: read its
files with `not_computed` set to `negative` (C++ `core::NotComputed::Negative`),
which reads every value below 0 as a pair to compute; the default, `nan`, reads
only an empty field or NaN so.

### Clustering a Problem

`cluster()` runs `method()` (`Auto` resolved as in Tier 1) with the `Problem`'s
settings and returns a `ClusteringResult` (MATLAB: a struct) of `labels`,
`medoid_indices`, `total_cost`, `iterations` and `converged`; the labels and
medoids are also kept in the `Problem` (`labels()`, `medoids()`). `set_result`
hands a `Problem` a clustering made elsewhere, so the scores and writers read it.
`write_clusters()`, `write_silhouettes()` and `write_medoid_members(iter, rep)`
write into `output_folder()` (`./results/`), `print_clusters()` prints, and
`find_total_cost()`, `assign_clusters()` and `calculate_medoids()` are the steps
of Lloyd's k-medoids. A method or score that needs a clustering raises
`InvalidInput` on a `Problem` that holds none.

## Algorithms

The clustering algorithms write the labels, medoids and `k` back into the
`Problem` and return their result; `dtw_barycenter` and `barycenter_kmeans`
leave the `Problem` as it was and return theirs.

| | C++ | Python | MATLAB |
|---|---|---|---|
| FastPAM | `dtwc::fast_pam(prob, k, max_iter = 100, seed = 42)` | `fast_pam(prob, n_clusters, max_iter=100, seed=42)` | `dtwc.fast_pam(prob, k, 'MaxIter', 100, 'Seed', 42)` |
| FastCLARA | `dtwc::algorithms::fast_clara(prob, CLARAOptions)` | `fast_clara(prob, n_clusters, sample_size=-1, n_samples=5, max_iter=100, seed=42)` | `dtwc.fast_clara(prob, k, 'SampleSize', -1, 'NSamples', 5, 'MaxIter', 100, 'Seed', 42)` |
| OneBatchPAM | `dtwc::algorithms::one_batch_pam(prob, OneBatchPAMOptions)` | `one_batch_pam(prob, n_clusters, batch_size=-1, max_iter=100, seed=42)` | `Method` `onebatch` |
| TADPole | `dtwc::algorithms::tadpole(prob, k, dc)` | `method` `tadpole` | `Method` `tadpole` |
| hierarchical | `dtwc::algorithms::build_dendrogram(prob, HierarchicalOptions)`, `cut_dendrogram(dend, prob, k)` | `build_dendrogram(prob, HierarchicalOptions())`, `cut_dendrogram(dend, prob, k)` | `dtwc.build_dendrogram(prob, 'Linkage', 'average', 'MaxPoints', 2000)`, `dtwc.cut_dendrogram(dend, prob, k)` |
| DTW barycenter | `dtwc::algorithms::dtw_barycenter(prob, indices, length, BarycenterOptions)` | `dtw_barycenter(prob, series_indices, target_length, options)` | |
| barycenter k-means | `dtwc::algorithms::barycenter_kmeans(prob, BarycenterClusteringOptions)` | `barycenter_kmeans(prob, options)` | |
| Lloyd k-medoids, MIP, LR-core | `set_method(...)`, then `cluster()` | the same | the same |

Where a cell names a method, that language reaches the algorithm through
`cluster()` or `set_method` with that name. FastPAM's `max_iter = 0` returns the
BUILD medoids without a SWAP. FastCLARA's `sample_size = -1` and OneBatchPAM's
`batch_size = -1` choose their own size. `CLARAOptions` holds `n_clusters`,
`sample_size`, `n_samples`, `max_iter` and `random_seed`;
`BarycenterClusteringOptions` holds `n_clusters`, `max_iter`, `target_length`
and `barycenter`, the `BarycenterOptions` each centre's update uses (`method`,
`max_iter`, `learning_rate`, `learning_rate_decay`, `gamma`, `tolerance`,
`random_seed`). The barycenter routines compute Standard DTW with squared
local costs and no band, and refuse a `Problem` set to another variant or a band.
`build_dendrogram` refuses more than `max_points` (2000) series. The solvers
behind `mip` and `lrcore` are on the [solvers page](../../guides/solvers/).

```python
import dtwcpp

prob = dtwcpp.Problem("demo")
prob.set_data([[0.0, 0.1, 0.2], [0.1, 0.0, 0.2], [0.2, 0.2, 0.0],
               [9.0, 9.1, 9.2], [9.1, 9.0, 9.2], [9.2, 9.2, 9.0]])

dend = dtwcpp.build_dendrogram(prob, dtwcpp.HierarchicalOptions())
print(list(dtwcpp.cut_dendrogram(dend, prob, 2).labels))

options = dtwcpp.BarycenterClusteringOptions()
options.n_clusters = 2
options.barycenter.max_iter = 10                       # the centre update's own options
print(list(dtwcpp.barycenter_kmeans(prob, options).labels))
```

## Scores

| | C++ `dtwc::scores::` | Python `dtwcpp.` | MATLAB `dtwc.` |
|---|---|---|---|
| silhouette of each series | `silhouette(prob)` | `silhouette(prob)` | `silhouette(prob)` |
| Davies–Bouldin | `davies_bouldin(prob)` | `davies_bouldin(prob)` | `davies_bouldin(prob)` |
| Dunn | `dunn(prob)` | `dunn(prob)` | `dunn(prob)` |
| inertia | `inertia(prob)` | `inertia(prob)` | `inertia(prob)` |
| Calinski–Harabasz | `calinski_harabasz(prob)` | `calinski_harabasz(prob)` | `calinski_harabasz(prob)` |
| a score by name | `score(prob, name)` | `Result.score(name)` | `Result.score(name)` |
| adjusted Rand index | `adjusted_rand(a, b)` | `adjusted_rand(a, b)` | `adjusted_rand(a, b)` |
| normalised mutual information | `normalized_mutual_info(a, b)` | `normalized_mutual_info(a, b)` | `normalized_mutual_info(a, b)` |

The first six read the clustering a `Problem` holds and fill its matrix if
needed; the last two compare two labellings.

## Distance functions

| | C++ `dtwc::distance::` | Python `dtwcpp.distance.dtw` | MATLAB `dtwc.distance.dtw` |
|---|---|---|---|
| Standard DTW | `dtw(x, y, band = -1, metric = L1)` | `dtw(x, y, band=-1, metric="l1")` | `dtw(x, y, 'Band', -1, 'Metric', 'l1')` |
| a variant | `dtw(x, y, core::DTWVariantParams, band, metric, missing_strategy)`, or `ddtw`, `wdtw`, `adtw`, `soft_dtw`, `msm`, `twe` | `dtw(x, y, variant="wdtw", wdtw_g=0.05, ...)` | `dtw(x, y, 'Variant', 'wdtw', 'WdtwG', 0.05, ...)` |
| missing values (NaN) | `missing(x, y, ...)`, `arow(x, y, ...)`, or `missing_strategy` | `missing_strategy="zero_cost"`, `"arow"`, `"interpolate"` | `'MissingStrategy', 'zero_cost'`, ... |

The keyword arguments and their defaults are `cluster()`'s distance settings.
Each call checks its settings and both series first and raises `InvalidInput`
for an empty series, a NaN or ±inf value (NaN is a missing value under a
missing-data strategy) or a parameter outside its domain. Python's
`compute_distance_matrix(series, band=-1, metric="l1", device=None, matrix_dtype="float64")` and MATLAB's
`dtwc.compute_distance_matrix(X, 'Band', b, 'MatrixDtype', 'float64')` return the N×N matrix of a set of
series.

```cpp
#include <dtwc.hpp>

#include <iostream>
#include <vector>

int main()
{
  const std::vector<double> x{ 1, 2, 3, 4, 5 }, y{ 2, 4, 6, 3, 1 };
  std::cout << dtwc::distance::dtw(x, y) << ' '          // Standard DTW, no band
            << dtwc::distance::dtw(x, y, 2) << ' '       // band 2
            << dtwc::distance::dtw(x, y, { .variant = dtwc::core::DTWVariant::MSM, .msm_c = 0.5 }) << '\n';
}
```

```python
import dtwcpp

x, y = [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 3.0, 1.0]
print(dtwcpp.distance.dtw(x, y), dtwcpp.distance.dtw(x, y, band=2),
      dtwcpp.distance.dtw(x, y, variant="msm", msm_c=0.5))
```

```matlab
x = [1 2 3 4 5]; y = [2 4 6 3 1];
disp([dtwc.distance.dtw(x, y), dtwc.distance.dtw(x, y, 'Band', 2), ...
      dtwc.distance.dtw(x, y, 'Variant', 'msm', 'MsmC', 0.5)]);
```

The per-pair kernels behind these functions (`dtwBanded`, `dtwFull`,
`dtwFull_L` and the variants' in `warping*.hpp`) stay public in C++. They check
nothing: they are what the fills call in their inner loop, so their caller checks
the input first.

## Precision

Series are stored and computed in `double` unless asked otherwise:
`dtwc::data_t` and the distance functions' default template argument are
`double`, and every `dtype` defaults to `float64`. Float32 is an opt-in that
halves the memory the series take, one word where data is loaded: Tier 1's
`load(..., dtype)` (MATLAB `'Dtype'`), `DataLoader::dtype(core::Precision::Float32)`
and `dtwc_cl --dtype float32`; or a C++ `Data` built from
`std::vector<std::vector<float>>`, Python's `Data.from_float32`, MATLAB's
`Problem.set_data(X, names, ndim, 'float32')`. Each reader rounds the double it
reads (a text value is its double parse; Parquet and Arrow values are cast), so a
float32 load holds no double copy and gives the same series in every language;
float32 data in memory alone stays `double`. A float32 series' recurrence runs in
float, so results can differ from the float64 ones. Python's `Data.series(i)`
reads a series of either precision; `p_vec` holds the float64 ones.
`dtwc::distance::dtw<float>(...)` returns a `float`.

The distance matrix has a value type of its own, `matrix_dtype`, `float64` by
default whatever the series' type: C++ `Problem::set_matrix_dtype` and
`Config::matrix_dtype`, Python `Problem.matrix_dtype` and `cluster(...,
matrix_dtype=)`, MATLAB `set_matrix_dtype` and `'MatrixDtype'`, `dtwc_cl
--matrix-dtype`. It sets every matrix the run makes: the `Problem`'s, in RAM or a
`.dtwm` file, CLARA's sample matrices and TADPole's cache. `matrix_dtype` float32
stores each distance rounded to float32: half the memory (N 100,000: 20 GB, not
40) and a relative error of at most 2^-24 (6e-8) a distance, none for float32
series. Costs and scores move by about as much; medoids and labels can change
where two choices differ by less. A distance beyond float32's range (3.4e38) is
refused. Every sum over the matrix still runs in `double`, and the exports return
the stored type: a NumPy `float32` array, a MATLAB `single` matrix (C++
`Result::distance_matrix()` stays `double`).

## The .dtwm cache and checkpoints

A distance matrix can live in a `.dtwm` file: a 64-byte header (`DTWM`, version 5,
N, a SHA-256 fingerprint of the series and the distance settings, the bytes of a
value and the rows written) followed by the packed lower triangle at the matrix's
value type, NaN for a pair not yet computed. A new file is not filled with NaN:
its rows from the rows written on hold no distance, and the fill records rows
once they are on the disk. A direct `get` or `dist_by_ind` of such a row reads
what the file holds there, 0, where every reader of the library takes it as not
computed. A file of version 4 (2.0 builds before `matrix_dtype`) is read and
mapped as float64. The same file is the memory-mapped cache and the checkpoint.

| | C++ | Python | MATLAB |
|---|---|---|---|
| map the matrix to a file | `prob.use_mmap_distance_matrix(path)` | `prob.use_mmap_distance_matrix(path)` | |
| save | `dtwc::save_checkpoint(prob, dir)` | `save_checkpoint(prob, dir)` | `dtwc.save_checkpoint(prob, dir)` |
| load | `dtwc::load_checkpoint(prob, dir)` | `load_checkpoint(prob, dir)` | `dtwc.load_checkpoint(prob, dir)` |
| save during a fill | `prob.checkpoint` (`CheckpointOptions`) | `prob.checkpoint` | `prob.set_checkpoint(dtwc.CheckpointOptions(...))` |

Mapping a file reopens the distances it holds, and the next CPU fill computes
only the pairs still NaN (a GPU fill computes them all), in slabs sized by the
`Problem`'s `ram_limit` (0: the memory available); an absent file is created. A checkpoint is
`<dir>/<name>.dtwm`. `load_checkpoint` returns false only when there is no file;
a file for other series, other distance settings or the other `matrix_dtype` is
`InvalidInput` (it names the `matrix_dtype` that reads it), and a file that is
not a whole `.dtwm` file is `IOError`, neither changing the `Problem`. With `checkpoint.enabled`, `fill_distance_matrix()` saves into
`checkpoint.directory` (`./checkpoints`) after every `save_interval` rows (100)
on the CPU, and once, at the end, on a GPU. On the
command line, `--checkpoint dir` saves the matrix as `dir/<name>.dtwm` and
resumes from it, and `--checkpoint-interval rows` also saves during the fill ([checkpointing](../../getting-started/checkpointing/)).

```python
import dtwcpp

series = [[0.0, 0.1, 0.2], [0.1, 0.0, 0.2], [0.2, 0.2, 0.0], [9.0, 9.1, 9.2]]

prob = dtwcpp.Problem("demo")
prob.set_data(series)
prob.fill_distance_matrix()
dtwcpp.save_checkpoint(prob, "ckpt")                   # ckpt/demo.dtwm

again = dtwcpp.Problem("demo")
again.set_data(series)
assert dtwcpp.load_checkpoint(again, "ckpt") and again.is_distance_matrix_filled()
```
