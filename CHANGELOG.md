# Changelog {#changelog}

[TOC]

This changelog contains a non-exhaustive list of new features and notable bug-fixes (not all bug-fixes will be listed).


<br/><br/>
# Unreleased

## Documentation
* Fixed DTW formula in `docs/2_method/2_dtw.md`: changed from squared L2 `(x_i - y_j)^2` to L1 `|x_i - y_j|` to match actual code implementation.
* Fixed pairwise comparison count: corrected from `1/2 * C(p,2)` to `C(p,2) = p(p-1)/2`.
* Fixed warping window description: band=1 allows a shift of 1 (not equivalent to Euclidean distance); band=0 forces diagonal alignment.
* Fixed spelling errors: "seies" -> "series", "wapring" -> "warping", "assertain" -> "ascertain".
* Added note clarifying that the default pointwise metric is L1 (absolute difference).
* Added z-normalization section with note that population stddev (N, not N-1) is used.
* Added `docs/2_method/5_algorithms.md`: clustering algorithm documentation with corrected FastPAM citation (JMLR 22(1), 4653-4688) and accurate complexity descriptions.
* Added `docs/2_method/6_metrics.md`: distance metrics documentation with LB_Keogh compatibility table and corrected Huber LB_Keogh reasoning.

## New features

* Added **Google Highway SIMD infrastructure** (`DTWC_ENABLE_SIMD` CMake option, default OFF). Highway 1.2.0 fetched via CPM for future use on platforms where compiler auto-vectorization is insufficient (ARM NEON, older compilers). Includes prototype SIMD kernels for LB_Keogh, z_normalize, and multi-pair DTW.
* Added `#pragma omp simd` hints to LB_Keogh and z_normalize loops for portable guaranteed vectorization.
* Added LB_Keogh, z_normalize, and envelope computation benchmarks to `bench_dtw_baseline.cpp`.
* Added `DTWClustering` sklearn-compatible Python class with `fit()`, `predict()`, `fit_predict()`, `score()`, `get_params()`/`set_params()`. Supports all DTW variants (standard, DDTW, WDTW, ADTW), multi-restart via `n_init`, and works with or without sklearn installed.
* Added core type system (`dtwc::core` namespace): `ScratchMatrix<T>`, `DenseDistanceMatrix`, `TimeSeriesView<T>`/`TimeSeries<T>`, `ClusteringResult`, `DTWOptions`, distance metrics (L1, L2, SquaredL2).
* Added FastPAM1 k-medoids clustering algorithm (Schubert & Rousseeuw 2021, JMLR) — true PAM SWAP with O(N^2*k) per iteration.
* Added z-normalization (`z_normalize`, `z_normalized`) for preprocessing time series.
* Added lower bound implementations: `lb_keogh()` with envelope precomputation, `lb_kim()` with summary precomputation, `lb_keogh_symmetric()` for symmetric pruning.
* Added `DenseDistanceMatrix` with NaN sentinel, CSV I/O (`write_csv`/`read_csv`), and `operator<<`.
* Added pruned distance matrix builder with LB_Kim and LB_Keogh pair-skipping support.
* Added roofline-aware microbenchmark (`benchmarks/bench_dtw_baseline.cpp`) measuring cells/sec, FLOP/sec, and bytes/sec for DTW computation at various series lengths.
* Added Google Benchmark integration for structured performance measurement.

## Architecture

* Removed Armadillo dependency from all hot-path code (`warping.hpp`, `Problem.hpp/cpp`, `fileOperations.hpp`). Armadillo is no longer linked to the core library.
* Replaced `arma::Mat<data_t>` scratch buffer in DTW functions with `core::ScratchMatrix<data_t>` (column-major, same layout).
* Replaced `arma::Mat<double>` distance matrix in `Problem` with `core::DenseDistanceMatrix` (NaN sentinel instead of -1).
* Unified `FastPAMResult` with `core::ClusteringResult` (single result type for all clustering algorithms).
* Distance matrix I/O now uses `DenseDistanceMatrix::write_csv()`/`read_csv()` instead of Armadillo's `save`/`load`.

## Bug fixes

* Fixed `throw 1` bare integer throw in `Problem_IO.cpp` — now throws `std::runtime_error`.
* Fixed `static std::mt19937` ODR violation in `settings.hpp` — changed to `inline`.
* Fixed `.at()` bounds-checked access in hot DTW loop — replaced with `operator[]`.
* Fixed `dtwBanded` 512MB/thread allocation — replaced with rolling buffer (O(band) memory).
* Fixed `fillDistanceMatrix` integer overflow at N>46K — triangular iteration with `size_t`.
* Fixed `dtwBanded` template default from `float` to `double`.
* Fixed `static` vectors data race in `calculateMedoids`.
* Fixed DBI formula (was using `dist(medoid, medoid)` = 0 always).
* Fixed broken examples (`settings::band` → `prob.band`).
* Renamed `cluster_by_kMedoidsPAM` to `cluster_by_kMedoidsLloyd` (was mislabeled).
* Fixed z_normalize tests to use production code instead of local reference implementation.

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