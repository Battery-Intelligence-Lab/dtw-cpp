---
title: "Migrating from v1.0.0 to 2.0"
weight: 50
description: "What v1.0.0's C++ library and dtwc_cl became in 2.0, and what to check."
---

# Migrating from v1.0.0 to 2.0

This page is for C++ code and `dtwc_cl` scripts written against v1.0.0. The
Python package `dtwcpp` and the MATLAB package `dtwc` are new in 2.0 (v1.0.0's
source held an unpublished Python binding, which 2.0 does not carry over): start
from [Tier 1](../../api/tier-1/).

Most v1.0.0 code builds and runs unchanged. Every `Problem` and `DataLoader`
method that 2.0 renamed keeps its v1.0.0 name as a `[[deprecated]]` alias that
forwards to the new one, and `dtwc_cl` still reads v1.0.0's flag spellings,
with a warning. What needs attention is the behaviour listed further down.

## Building

- CMake 3.26 and C++20 (v1.0.0: CMake 3.21, C++17). Armadillo is no longer used.
- OpenMP is required. `-DDTWC_ALLOW_SEQUENTIAL=ON` builds without it, and such a
  build says so once per process when it runs; so does a run capped at one
  thread on a machine with more.
- Gurobi is used only with `-DDTWC_ENABLE_GUROBI=ON`.
- `dtwc++` is now an INTERFACE target over static libraries: link it as before;
  a non-INTERFACE command on it (`target_compile_definitions(dtwc++ PRIVATE ...)`)
  fails at configure time.
- The `dtwc_main` demo executable is gone (`examples/cpp/MIP_single.cpp` runs a
  MIP clustering of the same sample data, taking the data folder as an
  argument), and `dtwc_cl` is built only when DTWC++ is the top-level project.
- A top-level build compiles for its machine's CPU (`-march=native`; MSVC
  `/arch:AVX2`), and an optimised build of the library uses the `fast`
  floating-point relaxations; v1.0.0 used neither (`DTWC_ARCH_LEVEL`,
  `DTWC_FP_MODEL=strict`).
- The maintainer CMake options are spelled `DTWC_*` (`DTWC_ENABLE_SANITIZER_ADDRESS`,
  `DTWC_WARNINGS_AS_ERRORS`, ...); v1.0.0's `dtwc_*` names still work for one
  release, with a warning.
- Including `dtwc.hpp` under `-ffast-math` stops the build: NaN marks missing
  values, which `-ffinite-math-only` would let the compiler ignore. Add
  `-fno-finite-math-only` after `-ffast-math`.
- `settings.hpp`, `timing.hpp` and `parallelisation.hpp` moved to `dtwc/base/`;
  the old include paths still work and print a message at compile time.

## C++ names

| v1.0.0 | 2.0 |
|---|---|
| `Problem::set_numberOfClusters(n)`, `cluster_size()` | `set_n_clusters(k)`, `n_clusters()` |
| `refreshDistanceMatrix`, `readDistanceMatrix`, `maxDistance`, `isDistanceMatrixFilled`, `fillDistanceMatrix`, `printDistanceMatrix`, `writeDistanceMatrix` | `refresh_distance_matrix`, `read_distance_matrix`, `max_distance`, `is_distance_matrix_filled`, `fill_distance_matrix`, `print_distance_matrix`, `write_distance_matrix` |
| `distByInd(i, j)` | `dist_by_ind(i, j)`, which reads the filled matrix and computes nothing; the old name fills the whole matrix on its first call |
| `printClusters`, `writeClusters`, `writeMedoidMembers`, `writeSilhouettes` | `print_clusters`, `write_clusters`, `write_medoid_members`, `write_silhouettes` |
| `findTotalCost`, `assignClusters`, `calculateMedoids` | `find_total_cost`, `assign_clusters`, `calculate_medoids` |
| `cluster_by_MIP()`, `cluster_by_kMedoidsPAM()` | `cluster_by_mip()`, `cluster_by_kmedoids_lloyd()` (Lloyd's k-medoids, as before) |
| `set_clusters(std::vector<int>&)` | `set_clusters(const std::vector<index_t>&)` |
| public fields `method`, `output_folder`, `name`, `data` | `method()` / `set_method`, `output_folder()` / `set_output_folder`, `name()` / `set_name`, `data()` / `set_data` |
| public fields `maxIter`, `N_repetition`, `band`, `init_fun`, `clusters_ind`, `centroids_ind` | still public (the last two now hold `index_t`); also `set_max_iter`, `set_n_repetitions`, `set_band`, which check their value |
| `Problem::cluster()` returning `void` | returns the `core::ClusteringResult` (labels, medoids, cost, iterations, converged) |
| `distMat`, `distMat_t` (Armadillo) | `distance_matrix()`, a `core::DistanceMatrix` |
| `resize()` | none: `set_n_clusters` no longer sizes `clusters_ind` and `centroids_ind`, which stay empty until a clustering writes them |
| copying a `Problem` | not allowed; a `Problem` moves |
| `DataLoader::startColumn(n)`, `startRow(n)` | `start_column(n)`, `start_row(n)` |
| `settings::DEFAULT_BAND_LENGTH` | `settings::DEFAULT_BAND` |
| the `dtwc::solver` MIP helpers that `dtwc.hpp` reached (`Element`, `Triplet`, `RowMajor`, `ColumnMajor`, `isAround`, `isFractional`, ...), `dtwc/types/element_types.hpp`, `types_util.hpp` | none |
| `settings::root_folder`, `dtwc_folder`, `resultsPath`, `dataPath`, `dtwc_dataPath`, `DTWC_ROOT_FOLDER`, `CURRENT_ROOT_FOLDER` | none: pass paths explicitly; a `Problem`'s output folder is `./results/`, relative to the working directory |
| `dtwc::randGenerator`, in `settings.hpp` | the same engine, in `base/random_engine.hpp`, which `dtwc.hpp` includes: one engine per program (v1.0.0 had a `static` copy in each translation unit) |
| `writeMatrix`, `readMatrix` | `io::write_csv`, `io::read_csv` |
| `dtwBanded`, `dtwFull`, `dtwFull_L` | the same names; the scalar type defaults to `double` (v1.0.0's `dtwBanded` defaulted to `float`), with new trailing defaulted parameters and span overloads |
| `run(task, n, numMaxParallelWorkers = 32)` | the same; the limit is now applied (v1.0.0 used every OpenMP thread) |

Counts of series and clusters, labels, medoids and indices are `index_t`
(`std::int64_t`) where v1.0.0 used `int`: `size()`, `n_clusters()`,
`clusters_ind`, `centroids_ind`, `centroid_of`, `Data::size()`,
`DataLoader::n_data`. `Method::Kmedoids`, `Method::MIP`, `Solver::Gurobi`,
`Solver::HiGHS`, `init::random`, `init::Kmeanspp`, `scores::silhouette`,
`MIP_clustering_byGurobi`, `MIP_clustering_byHiGHS`, `Range` and `Clock` keep
their names and signatures; `Index` keeps its name, and its `difference_type`
(what `operator-` returns) is `std::ptrdiff_t`, not `size_t`. The rest of 2.0's C++ surface is new;
the [Tier 1](../../api/tier-1/) and [Tier 2](../../api/tier-2/) pages describe it.

## dtwc_cl

v1.0.0's options are named here as the [CLI reference](../../getting-started/cli/)
names them, without their leading dashes.

| v1.0.0 option (default) | 2.0 (default) |
|---|---|
| Nc, clusters, number_of_clusters: a number or a range i..j (none) | `-k`, `--n-clusters`: one number, required |
| name, probName (dtwc) | `--name` (the input's file or folder name) |
| i, in, input (../data/dummy) | `-i`, `--input`, required |
| o, out, output (the working directory) | `-o`, `--output` (`./results`) |
| skipRows; skipCols, skipColumns (0) | `--skip-rows`; `--skip-cols` (0) |
| maxIter, iter (100) | `--max-iter` (100) |
| method (kMedoids) | `-m`, `--method` (`auto`) |
| repeat, Nrepeat, Nrepetition, Nrep (1) | `--n-init` (1) |
| solver, mip_solver, mipSolver (HiGHS) | `--solver` (`highs`) |
| bandwidth, bandw, bandlength (-1) | `-b`, `--band` (-1) |
| distMat, distance_matrix, distances | `--dist-matrix` |

The v1.0.0 spellings still work: they are hidden from `--help` and print
`[dtwc] warning: '<old>' is deprecated, use '<new>' instead`. The method values
`kMedoids` and `MIP` still read, in any case.

- `-k` and `-i` are required. v1.0.0 exited with success having clustered
  nothing when the number of clusters was missing, and ran one clustering per
  number of a range `i..j`; 2.0 refuses the range: run once per `k`.
- The default method is `auto`: FastPAM for up to 5000 series and FastCLARA
  above. `--method kmedoids` runs v1.0.0's Lloyd k-medoids.
- An unknown method or solver, a `--dist-matrix` that cannot be read, and
  `--solver gurobi` on a build without Gurobi stop the run with an error, where
  v1.0.0 printed a message and went on with a default; so do a `--band` below
  -1, which v1.0.0 ran as full DTW, a negative `--skip-rows` or `--skip-cols`
  and a failed write.
- A run prints a summary, not each iteration's cost and every cluster's members.

| v1.0.0 output | 2.0 `dtwc_cl` output |
|---|---|
| `<name>_Nc_<k>.csv` | `<name>_labels.csv` (`name,cluster`) and `<name>_medoids.csv` (`cluster,medoid_index,medoid_name`) |
| `<name>_silhouettes_Nc_<k>.csv` | `<name>_silhouettes.csv` (`name,cluster,silhouette`), when the method filled the matrix and k > 1 |
| `<name>_distanceMatrix.csv` | `<name>_distance_matrix.csv`, when the method filled the matrix |
| `<name>medoids_rep_<r>.csv`, `<name>_bestRepetition_Nc_<k>.csv` | not written |
| (none) | `<name>_report.txt`: the run report, written last; `--no-report` writes none |

The library's writers (`write_clusters`, `write_silhouettes`,
`write_distance_matrix`) keep v1.0.0's file names.

## Behaviour to check

- **A distance-matrix CSV written by v1.0.0 can hold `-1` for pairs it never
  computed** (v1.0.0 computed pairs on demand and wrote the matrix as it stood).
  Read it with `--not-computed negative` (`read_distance_matrix(path,
  core::NotComputed::Negative)`): every value below 0 is then a pair still to
  compute. By default 2.0 marks such a pair with an empty field or NaN, as it
  writes one, and refuses a negative entry, naming the pair and the option,
  under every variant but Soft-DTW, whose distances can be negative. v1.0.0
  read every negative entry as not computed.
- `cluster()` writes no files, and prints only after `set_verbose(true)`.
  `cluster_and_process()` clusters and writes the files, as before.
- Lloyd's k-medoids fills the whole distance matrix first. Its repetitions start
  from `random_seed() + r` (42 by default) instead of drawing from
  `randGenerator`, and the lowest-cost repetition is kept (v1.0.0 kept the last).
  A medoid at distance 0 from another medoid (a duplicate series) keeps its own
  cluster, where v1.0.0 gave it to the first tied medoid and left its cluster
  empty; a run that stops at `maxIter` assigns the series to its final medoids
  before its cost is taken.
- The band is the window |i − j| ≤ band for every pair. v1.0.0 centred it on
  the line joining the ends of two series of different lengths. In a `Problem`, a band
  narrower than the difference in length of two series is `InvalidInput`,
  naming the smallest band that works; `dtwBanded` and `distance::dtw` return
  the no-path value, `std::numeric_limits<T>::max()`, for such a pair.
- The setters check their values: `set_n_clusters`, `set_max_iter` and
  `set_n_repetitions` below 1 and `set_band` below -1 are `InvalidInput`. A
  direct write to a public field is checked where it is used: a `band` below -1
  at the next fill, a `maxIter` or `N_repetition` below 1 when Lloyd's
  k-medoids runs.
- A NaN or ±inf in a series is `InvalidInput` before any distance is computed,
  unless a missing-data strategy says what NaN means (`set_missing_strategy`).
  A series with no values is `InvalidInput` too, a blank line in a file of one
  series per row no longer being one.
- The scores, `find_total_cost` and the writers raise `InvalidInput` on a
  `Problem` that holds no clustering (`scores::silhouette` printed a line and
  returned `-1` per series). `scores::silhouette` raises `UndefinedScore` below
  two non-empty clusters, where v1.0.0 gave every series about 1.
- `read_distance_matrix` refuses a file of another size, a matrix that is not
  square and symmetric, ±inf, and a negative entry (above). The CSV it writes
  carries every digit of a double and LF line endings.
- Text input: a blank line is no longer an empty series; each file of a folder
  holds one value per line after the skipped columns (a pandas `index,value` file needs
  `start_row(1)` and `start_column(1)`, or `--skip-rows 1 --skip-cols 1`); a folder is read in sorted order; a
  `.txt` file is tab-separated unless a delimiter is given (v1.0.0 read it as
  comma-separated), and `DataLoader::path()` matches extensions in any case and
  keeps a delimiter set before it. `load_batch_file` with a `start_row` of 1 or
  more reads the series (v1.0.0 read none), and `Ndata` 0 reads no series from a
  folder as from a file (v1.0.0 read the whole folder).
- A series file that cannot be opened is an `IOError` (v1.0.0 read it as an empty
  series), and a field that is not a number is an error naming its row and
  column (v1.0.0 read it as 0 or as the series' end). A folder takes regular
  files only and skips dot-files. Text is read in binary: a Ctrl-Z byte no
  longer ends a file on Windows, and a file whose lines end in a bare CR is
  refused. Numbers are read with fast_float and the matrix CSV is written with
  `std::to_chars`, whatever the global C++ locale. Names taken from file names
  are UTF-8 on every platform, and on Windows `dtwc_cl` reads its arguments and
  paths as UTF-8 (v1.0.0 used the ANSI code page).
- Every error is a typed exception derived from `dtwc::Error`, itself a
  `std::runtime_error`; v1.0.0 threw integers in places
  ([the error types](../../api/tier-1/)).
- `set_solver(Solver::Gurobi)` on a build without Gurobi returns `false` and keeps
  HiGHS; it is `[[nodiscard]]`, so check it. `cluster_by_mip()` raises
  `SolverError` when its solver is not built or ends without a proven optimum.
- The MIP starts from FastPAM's clustering, and both solvers stop at a relative
  gap of 1e-5; v1.0.0 started cold, left HiGHS at its own gap, and set Gurobi's
  NumericFocus to 3 (now 1; MIPFocus is 2). `mip_settings` sets each of them.
- A direct write to the `band` field takes effect at the next
  `fill_distance_matrix()`; v1.0.0 kept the distances computed under the old
  band.
- Series are stored as `double`, as in v1.0.0. Storing them as `float` is new, and
  opt-in where they are loaded: `DataLoader::dtype`, `dtwc::load(..., dtype)`,
  Python's and MATLAB's `load`, `dtwc_cl --dtype float32`. A `float` read from text
  (`readFile<float>` too) is the double parse rounded, which can differ by one ulp
  from v1.0.0's direct rounding in a halfway case.

## Results that change

The behaviour above changes some results: distances between series of different
lengths under a band, and Lloyd k-medoids runs. These change too:

- **`dtwc_cl` with no `--method`** runs FastPAM from seed 42 where v1.0.0 ran
  Lloyd's k-medoids, so it can return other medoids. `--method kmedoids` runs
  Lloyd's, with the seeding and the tie rule above.
- **`init::random(prob)` and `init::Kmeanspp(prob)`** pick other medoids than
  v1.0.0 did for the same `randGenerator` state: each takes one draw from it as
  the seed of a sampler that gives the same picks on every platform, and neither
  picks a medoid twice.
- **`Method::MIP`** solves the full model with the selected solver at every N,
  as v1.0.0 did; with the start and the gap above, a run that stops at the gap
  can return another clustering within it.
