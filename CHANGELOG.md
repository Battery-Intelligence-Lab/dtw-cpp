# Changelog {#changelog}

[TOC]

This changelog contains a non-exhaustive list of new features and notable bug-fixes (not all bug-fixes will be listed).


<br/><br/>
# 2.0.0 (unreleased)

What changed since v1.0.0, for its users; the [migration guide](docs/content/guides/migration.md) maps each v1.0.0
name, flag and output file to 2.0. The development log stays in git: `git log v1.0.0..`, and entry by entry
`git show fa02acaf:CHANGELOG.md`.

## Breaking

- C++20 and CMake 3.26 (v1.0.0: C++17, CMake 3.21). OpenMP is required; `-DDTWC_ALLOW_SEQUENTIAL=ON` builds without
  it, and such a build says so when it runs.
- Counts, labels, medoids and indices are `dtwc::index_t` (`std::int64_t`): code that keeps `clusters_ind` or
  `centroids_ind` in a `std::vector<int>` stops compiling; declare it `std::vector<dtwc::index_t>`.
- A `Problem` moves but no longer copies. Its fields `method`, `output_folder`, `name` and `data` are accessors now;
  `maxIter`, `N_repetition`, `band`, `init_fun`, `clusters_ind` and `centroids_ind` stay public.
- `dtwc.hpp` under `-ffinite-math-only` (implied by `-ffast-math`) is a compile error, since NaN marks missing values
  and uncomputed distances: add `-fno-finite-math-only`.
- Errors are typed exceptions under `dtwc::Error` (a `std::runtime_error`): `InvalidInput`, `IOError`, `DeviceError`,
  `SolverError`, `UndefinedScore`; v1.0.0 printed a message and went on, or threw a bare `std::runtime_error` or an `int`.
- `dtwc_cl` needs `-k` and `-i` (without `--Nc`, v1.0.0 exited 0 having clustered nothing) and refuses v1.0.0's range
  `--Nc i..j`: run once per k.
- `dtwc_cl` stops with an error, a non-zero exit, on an unknown method or solver, an unreadable `--dist-matrix`,
  `--solver gurobi` on a build without Gurobi, a `--band` below -1, a negative `--skip-rows` or `--skip-cols` and a
  failed write; v1.0.0 went on.
- `dtwc_cl` writes `<name>_labels.csv` and `<name>_medoids.csv`, and `<name>_distance_matrix.csv` and
  `<name>_silhouettes.csv` when its method filled the matrix; v1.0.0 wrote `<name>_Nc_<k>.csv` and per-repetition files.
- `set_n_clusters` (v1.0.0's `set_numberOfClusters`) below 1 is `InvalidInput`, and so is a band below -1, set or
  written to `band` (v1.0.0 ran full DTW). A ±inf in a series is refused, a NaN unless a missing-data strategy applies.
- A series with no values is `InvalidInput`, and a blank line in a file of one series per row is no longer an empty
  series.
- A line of a folder's file with more than one value after the skipped columns is an error (v1.0.0 read the first):
  a pandas `index,value` file needs `--skip-rows 1 --skip-cols 1`.
- `read_distance_matrix` (`--dist-matrix`) refuses a matrix of another size, one not square and symmetric, and ±inf.
- A distance-matrix CSV written by v1.0.0 can hold `-1` for pairs it never computed, and 2.0 reads it as a distance:
  recompute such a matrix, or empty those fields; 2.0 reads and writes an uncomputed pair as an empty field.

## Added

- The Python package `dtwcpp` and the MATLAB package `+dtwc` (v1.0.0 published neither); `dtwc_cl` archives and wheels
  for Linux and Windows on x86-64-v3 (AVX2, FMA) and macOS 13.3+ on arm64, and wheels for Linux arm64.
- One four-step API in C++, Python and MATLAB, `device()`, `load()`, `cluster()` and a `Result` (labels, medoids, cost,
  `score`, `save`, `distance_matrix`; `plot` in Python and MATLAB), on the code `dtwc_cl` runs.
- `dtwc::Config` (a run's settings, keyed by `dtwc_cl`'s long options) and `dtwc::run`. `dtwc_cl --config` reads TOML or
  YAML (a flag beats the file, an unknown key is an error), `--print-config` writes one; `dtwc_cl --version`.
- Python: `dtwcpp.cluster(data, k, **keys)` takes `dtwc_cl`'s clustering keys (`method`, `band`, `variant`, `seed`, ...)
  and series in memory as NumPy arrays, lists of arrays of any length, pandas or Arrow; a scikit-learn `DTWClustering`.
- Python: `device="hpc"` and `"hpc:gpu"` (beta) send a run to a SLURM cluster as one `job.toml`; the extras `parquet`,
  `hdf5`, `io`, `sklearn` and `mip` (the wheel links no HiGHS, so `method="mip"` runs on highspy).
- MATLAB: `dtwc.cluster(data, k, Name, Value)` with the same keys in CamelCase, `dtwc.load` (Parquet through
  `parquetread`), `dtwc.DTWClustering`, `dtwc.Problem`, the algorithms and the scores, through one MEX file.
- DTW variants DDTW, WDTW, ADTW, Soft-DTW, MSM and TWE (`--variant`; Soft-DTW, MSM and TWE ignore the band), a squared
  Euclidean local cost for Standard DTW and DDTW (`--metric`), and `dtwc::distance::dtw` for one pair.
- Missing values as NaN (`--missing-strategy zero_cost`, `arow` or `interpolate`), multivariate series (`ndim` values
  per step, `--mv-mode dependent` or `independent`) and series stored as `float` on request (`--dtype float32`).
- Methods FastPAM (FasterPAM), FastCLARA, OneBatchPAM, hierarchical, TADPole and LR-core (an exact Lagrangian branch and
  bound) beside Lloyd's k-medoids and the MIP; `--method auto`: FastPAM to 5,000 series, then FastCLARA (GPU: FastPAM).
- `--seed` (42) for the seeded methods, whose draws are the same on every platform and standard library; `--n-init`
  (v1.0.0's `--repeat`) restarts FastPAM too.
- DTW barycenters (`dtw_barycenter`: SSG, DBA, Soft-DTW) and barycenter k-means (`barycenter_kmeans`).
- Scores: Davies–Bouldin, Dunn, inertia and Calinski–Harabasz beside the silhouette (`Result::score(name)`), and the
  adjusted Rand index and normalised mutual information of two labellings.
- GPUs: CUDA (compute capability 8.0 or newer, `-DDTWC_ENABLE_CUDA=ON`) and Metal (Apple silicon, built by default on
  macOS) fill the matrix of Standard DTW on univariate series and FastCLARA's samples (`--device gpu`).
- Input: Parquet and Arrow IPC (`-DDTWC_ENABLE_ARROW=ON`; a list column or several float columns hold a series per row,
  `--column` reads one) and `dtwc_cl --delimiter`.
- The distance matrix memory-mapped in a `.dtwm` file from 50,000 series (`--mmap-threshold`), also the checkpoint
  (`--checkpoint`, `--checkpoint-interval`: a fill resumes); with Arrow, `--ram-limit` streams Parquet through FastCLARA.
- `Problem::cluster()` returns a `ClusteringResult` (labels, medoids, cost, iterations, converged); `Problem` gains
  `set_method`, `set_device`, `set_distance`, `set_variant`, `set_metric`, `set_result`, `solver()`, `mip_settings`, ...
- Build options `DTWC_ENABLE_{HIGHS,GUROBI,CUDA,METAL,ARROW,LLFIO,YAML}` (each optional; an `ON` that cannot be honoured
  stops the configure), `DTWC_FP_MODEL` and `DTWC_ARCH_LEVEL`.
- On Windows, `dtwc_cl.exe` states its name, version and copyright in its file properties.

## Changed

- `dtwc_cl`'s default method is `auto` (above), where v1.0.0 ran Lloyd's k-medoids (`--method kmedoids`): a run
  without `--method` can return other medoids.
- The band is |i − j| ≤ band for every pair (v1.0.0 centred it on the line joining the ends of series of different
  lengths), so their banded distances change; a `Problem` refuses a band narrower than their length difference.
- Lloyd's k-medoids fills the matrix first, seeds repetition r with `random_seed() + r` (42 + r by default), keeps the
  lowest-cost one (v1.0.0: the last, seeded by `randGenerator`) and reassigns at `maxIter` before taking the cost.
- `init::random(prob)` and `init::Kmeanspp(prob)` take one draw of `randGenerator` as the seed of a sampler that picks
  the same medoids on every platform, never one twice: they pick other medoids than v1.0.0 did.
- The MIP starts from FastPAM's clustering, both solvers stop at a relative gap of 1e-5, and Gurobi runs NumericFocus 1
  and MIPFocus 2 (v1.0.0: a cold start, HiGHS's own gap, NumericFocus 3); `mip_settings` sets each.
- v1.0.0's flag spellings (`--Nc`, `--skipRows`, `--bandwidth`, ...) still work, hidden from `--help`, each printing a
  warning that names its 2.0 flag (`-k`, `--skip-rows`, `--band`, ...).
- `dtwc_cl -o` defaults to `./results` and `--name` to the input's file or folder name (v1.0.0: `.` and `dtwc`); a run
  prints a summary, not each iteration's cost and every cluster's members.
- `Problem` and `DataLoader` methods take snake_case names (`fill_distance_matrix`, `set_n_clusters`, `start_row`, ...);
  each renamed v1.0.0 name stays as a `[[deprecated]]` alias.
- `dist_by_ind(i, j)` reads the filled matrix in O(1) and computes nothing, so fill it first, as the library's methods
  do; the v1.0.0 name `distByInd` fills the whole matrix on its first call.
- `Problem::cluster()` writes no files and prints only after `set_verbose(true)`; `cluster_and_process()` clusters and
  writes as before.
- `set_solver` is `[[nodiscard]]`, and `set_solver(Solver::Gurobi)` on a build without Gurobi returns `false` silently;
  `cluster_by_mip()` raises `SolverError` for a solver not built or a run without a proven optimum.
- `dtwc::randGenerator` is one engine per program, in `dtwc/base/random_engine.hpp`, which `dtwc.hpp` includes and
  `settings.hpp` no longer does; v1.0.0 had a `static` copy in each translation unit.
- `run(task, n, numMaxParallelWorkers = 32)` applies its limit; v1.0.0 used every OpenMP thread for any value but 1.
- A `.txt` file is tab-separated unless a delimiter is given (v1.0.0: comma); `DataLoader::path()` matches extensions
  in any case and keeps a delimiter set before it.
- A top-level build compiles for its machine's CPU (`-march=native`; MSVC `/arch:AVX2`), and an optimised build of the
  library with the `fast` floating-point relaxations; v1.0.0 used neither (`DTWC_ARCH_LEVEL`, `DTWC_FP_MODEL=strict`).
- Gurobi is linked only with `-DDTWC_ENABLE_GUROBI=ON` (v1.0.0 linked any installation it found); HiGHS is the default.
- `dtwc++` is an INTERFACE target over static libraries: link it as before, but a non-INTERFACE command on it fails at
  configure. `dtwc_cl` is built only when DTWC++ is the top-level project.
- The maintainer CMake options are spelled `DTWC_*` (`DTWC_ENABLE_SANITIZER_ADDRESS`, `DTWC_WARNINGS_AS_ERRORS`, ...);
  v1.0.0's `dtwc_*` names still work for one release, with a warning.
- `settings.hpp`, `timing.hpp` and `parallelisation.hpp` moved to `dtwc/base/` (the old paths work, with a message at
  compile time); `dtwBanded`'s scalar type, where none is deduced, is `double` (v1.0.0: `float`).
- Numbers are read with fast_float and the matrix CSV written with `std::to_chars`, whatever the global C++ locale;
  names taken from file names are UTF-8 on every platform (v1.0.0 used the Windows code page).

## Performance

Each figure compares 2.0's code just before and after one change (v1.0.0 had none of FastPAM, OneBatchPAM, LR-core,
Soft-DTW, Python or CUDA), on the machine named.

- The CPU fill of Standard DTW on univariate series computes a series against 8 of its length at once in SIMD lanes
  (16 in `float32`): on an Intel Core Ultra 9 285 (24 threads, loaded) ECG5000's 4,500 series fill in 3.9 s, not 66 s.
- There a band-50 fill of 50 series of length 1,000 runs 5.1× faster; the lanes match the one-pair kernel bit for bit
  unless the compiler fuses a multiply-add in one and not the other (GCC does by default; squared L2 only).
- On 64-bit Arm the lanes hold 16 doubles (32 floats) and take each minimum with one `fminnm`: on an Apple M5 Pro
  1.41–2.00× faster on one thread, 1.61–1.69× in an 18-thread fill of equal-length series; bit for bit.
- The one-pair kernel without a band computes two columns per pass: on an Apple M5 Pro a pair runs 1.14–1.98× faster,
  the 18-thread fill of 1,000 series of lengths 90–110 1.42–1.50× faster; bit for bit with Apple clang.
- No DTW cell calls a library function (the MSVC STL made `std::min({…})` one): on an Intel Core Ultra 9 285, pinned,
  two 1,000-sample series take 1.4 ms, not 7.2 ms (band 100: 0.26 ms, not 1.4 ms); digit for digit.
- `dtwFull` and Soft-DTW keep one column per thread, not an n × m matrix (512 MB for two 8,000-sample series): an
  8-thread Soft-DTW fill of eight such series peaked at 3.4 GB, now 14 MB (Intel Core Ultra 9 285); digit for digit.
- FastPAM's swap reads the matrix directly: 5.6–5.9× faster on an Intel Core Ultra 9 285 (loaded), the same result.
- OneBatchPAM (N = 2,000, k = 10; Intel Core Ultra 9 285, 24 threads, loaded): its final assignment runs on every
  thread (the call 1.82 s → 0.96 s), then its batch table on the SIMD lanes (0.98 s → 0.25 s; the lanes' FMA caveat).
- LR-core's dual runs on OpenMP from 280 series: on an Apple M5 Pro (18 threads), at 800–3,200 series, its root runs
  3.5–7.3× and `dtwc_cl -m lrcore` 1.4–4.7× faster without HiGHS (the wheel), 1.2–1.5× and 1.1–1.5× with it; bit for bit.
- The Python extension's binding file compiles at `-O3`, not nanobind's `-Os`: on an Apple M5 Pro `dtwcpp.dtw` runs
  1.1–1.5× and an unequal-length distance matrix 1.5× faster.
- The CUDA fill writes the packed matrix: on an RTX 4000 Ada at N = 20,000 (FP32) host memory 6.0 → 1.6 GiB and GPU
  memory 1.6 → 1.1 GiB; 9,000 series of length 100 fill in 1.16 s, not 1.53 s.

## Fixed

- A medoid at distance 0 from another (a duplicate series) keeps its own cluster in every method; v1.0.0's Lloyd gave
  it to the first tied medoid and published its own cluster empty.
- `scores::silhouette` raises `InvalidInput` on an unclustered `Problem` (v1.0.0 printed a line and returned `-1` per
  series) and `UndefinedScore` below two non-empty clusters (v1.0.0 gave every series about 1).
- `find_total_cost()` and the writers raise `InvalidInput` on an unclustered `Problem` instead of reading empty labels.
- `load_batch_file` with a `start_row` of 1 or more reads the series (v1.0.0 read none), and `Ndata` 0 reads no series
  from a folder as from a file (v1.0.0 read the whole folder).
- A series file that cannot be opened is an `IOError` (v1.0.0 read it as an empty series), and a field that is not a
  number is an error naming its row and column (v1.0.0 read it silently, as 0 or as the series' end).
- A folder is read in sorted order, regular files only and dot-files skipped, so the series' order and names are the
  same on every machine; v1.0.0 took the directory's order and every entry.
- Text is read in binary: on Windows a Ctrl-Z byte no longer ends a file early (it is refused, as any non-numeric field
  is), and a file whose lines end in a bare CR is refused instead of read as one line.
- A direct write to the `band` field takes effect at the next `fill_distance_matrix()`; v1.0.0 kept the distances
  computed under the old band.

## Removed

- Armadillo, with `writeMatrix` / `readMatrix` (`io::write_csv` / `io::read_csv` write and read a distance matrix) and
  `Problem::distMat_t` (the matrix is a `core::DistanceMatrix`); code that used Armadillo now links it itself.
- `Problem::resize()`: `set_n_clusters` no longer sizes `clusters_ind` and `centroids_ind`, which stay empty until a
  clustering writes them.
- `settings::root_folder`, `dtwc_folder`, `resultsPath`, `dataPath`, `dtwc_dataPath` and the `DTWC_ROOT_FOLDER` and
  `CURRENT_ROOT_FOLDER` macros: pass paths explicitly; a `Problem` writes to `./results/` in the working directory.
- `settings::DEFAULT_BAND_LENGTH`: it is `settings::DEFAULT_BAND`.
- The `dtwc::solver` MIP helpers that `dtwc.hpp` reached (`Element`, `Triplet`, `RowMajor`, `ColumnMajor`, `isAround`,
  `isFractional`, ...), with `dtwc/types/element_types.hpp` and `types_util.hpp`.
- The `dtwc_main` demo executable: `examples/cpp/MIP_single.cpp` runs a MIP clustering of the same sample data, taking
  the data folder as an argument.
- v1.0.0's unpublished pybind11 binding (`python/py_main.cpp`, `setup.py`): the `dtwcpp` package replaces it.

<br/><br/>
# DTWC v1.0.0

## New features
* HiGHS solver is added for open-source alternative to Gurobi (which is now not necessary for compilation and can be enabled by necessary flags). 
* Command line interface is added. 
* Documentation is improved (Doxygen website).

## Notable Bug-fixes
* Sakoe-Chiba band implementation is now more accurate. 

## API changes
* Replaced `VecMatrix<data_t>` class with `arma::Mat<data_t>`. 

## Dependency updates:
* Required C++ standard is reduced from C++20 to C++17 as it was causing `call to consteval function 'std::chrono::hh_mm_ss::_S_fractional_width' is not a constant expression` error for clang versions older than clang-15.
* `OpenMP` for parallelisation is adopted as `Apple-clang` does not support `std::execution`. 

## Developer updates: 
* The software is now being tested via Catch2 library. 
* Dependabot is added. 
* `CURRENT_ROOT_FOLDER` and `DTWC_ROOT_FOLDER` are seperated as DTW-C++ library can be included by other libraries. 

<br/><br/>
# DTWC v0.3.0

## New features
* UCR_test_2018 data integration for benchmarking. 

## Notable Bug-fixes
* N/A

## API changes
* DataLoader class is added for data reading. 
* `settings::resultsPath` is changed with `out_folder` member variable to have more flexibility. 
* `get_name` function added to remove `settings::writeAsFileNames` repetition)
* `std::filesystem::path operator+` was unnecessary and removed. 

<br/><br/>
# DTWC v0.2.0

A user interface is created for other people's use. 

## New features / updates
- Scores file with silhouette score is added. 
- `dtwFull_L` (L = light) is added for reducing memory requirements substantially.  

## API changes
- Problem class for a better interface. 
- `mip.hpp` and `mip.cpp` files are created to contain MIP functions.

## Notable Bug-fixes
* Gurobi better path finding in macOS. 
* TBB could not be used in macOS so it is now option with alternative thread-based parallelisation. 
* Time was showing wrong on macOS with std::clock. Therefore, moved to chrono library.

## Formatting: 
- Include a clang-format file. 

## Dependency updates
  * Required C++ standard is upgraded from C++17 to C++20. 

<br/><br/>
# DTWC v0.1.0

This is the initial release of DTWC. 

## Features
- Iterative algortihms for K-means and K-medoids 
- Mixed-integer programming solution support via YALMIP/MATLAB. 
- Support for `*.csv` files generated by Pandas.  

## Dependencies
  * A compiler with C++17 support. 
  * We require at least CMake 3.16.
