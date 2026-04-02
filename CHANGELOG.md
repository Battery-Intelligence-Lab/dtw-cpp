# Changelog {#changelog}

[TOC]

This changelog contains a non-exhaustive list of new features and notable bug-fixes (not all bug-fixes will be listed).


<br/><br/>
# Unreleased

## Core

* **Parallel pruned distance matrix**: `fill_distance_matrix_pruned()` (Problem-based) is now parallelised with OpenMP. Uses lock-free atomic CAS on `nn_dist` for nearest-neighbor tracking across threads, with `schedule(dynamic, 16)` for load balancing. Precomputation of summaries and envelopes is also parallelised.
* **Distance matrix strategy selection**: new `DistanceMatrixStrategy` enum (`Auto`, `BruteForce`, `Pruned`, `GPU`) and `Problem::distance_strategy` member. `fillDistanceMatrix()` now dispatches based on the selected strategy. `Auto` (default) selects `Pruned` for Standard DTW variant, `BruteForce` for non-standard variants (DDTW, WDTW, ADTW, SoftDTW).

## CUDA

* **1-vs-N and K-vs-N GPU DTW kernels**: new `compute_dtw_one_vs_all()` and `compute_dtw_k_vs_all()` functions compute DTW distances from one or K query series against all N target series. Dedicated kernels (`dtw_one_vs_all_wavefront_kernel`, `dtw_one_vs_all_warp_kernel`, `dtw_one_vs_all_regtile_kernel`) use a 2D grid (targets x queries) with query loaded into shared memory per block. For k-medoids clustering with K=5 medoids and N=200 series, this avoids recomputing the full NxN matrix when medoids change (K*N = 1,000 pairs vs 19,900). Supports FP32/FP64 auto-precision, banded DTW, L1/squared-L2 metrics, and external query series. New result types: `CUDAOneVsNResult`, `CUDAKVsNResult`.
* **GPU-accelerated LB_Keogh pruning**: new `compute_lb_keogh_cuda()` function computes symmetric LB_Keogh lower bounds for all N*(N-1)/2 pairs on GPU. Two new CUDA kernels: `compute_envelopes_kernel` (sliding-window min/max per series) and `compute_lb_keogh_kernel` (embarrassingly parallel, one thread per pair). New `CUDADistMatOptions` fields: `use_lb_pruning` enables LB computation before DTW, `skip_threshold` prunes pairs with LB exceeding the threshold (set to INF without computing DTW). Standalone `CUDALBResult compute_lb_keogh_cuda(series, band)` API for use in clustering and nearest-neighbor search.
* **On-device pair index computation**: eliminated host-side pair index arrays (`h_pair_i`, `h_pair_j`) and their H2D transfers by computing `(i,j)` from the flat upper-triangle index directly on GPU via `decode_pair()`. For N=1000 this saves 4 MB of transfers and 2 device allocations.
* **GPU-side NxN result matrix**: kernels now write DTW distances directly into a symmetric NxN matrix on device, eliminating the per-pair distance array, its D2H transfer, and the host-side O(N^2) fill loop. A single contiguous `cudaMemcpyAsync` transfers the complete result.
* **Precomputed integer band boundaries**: banded DTW band checks (6 FP64 operations per cell) are now precomputed once per pair into shared memory (wavefront kernel) or registers (warp kernel), replacing per-cell FP64 math. On consumer GPUs with 1:64 FP64 rate, this eliminates ~192 FP32-equivalent cycles per cell in the banded path.
* **Stream-based async pipeline**: `launch_dtw_kernel` now uses a CUDA stream with `cudaMemcpyAsync` for H2D/D2H transfers and stream-ordered kernel launches. Pinned host memory (`cudaMallocHost`) is used for the two large buffers (flat series data and output distances) to enable true overlap of transfers with compute. Falls back gracefully to pageable memory if pinned allocation fails. GPU timing now uses `cudaEvent`-based measurement for accurate pipeline profiling. Added RAII wrappers (`PinnedPtr`, `CudaStream`, `CudaEvent`) to `cuda_memory.cuh`.
* **Persistent kernel mode**: the `dtw_wavefront_kernel` now supports persistent scheduling. When `num_pairs` significantly exceeds the GPU's resident block capacity (>4x), blocks loop over pairs using a global atomic work counter instead of the one-pair-per-block model. This eliminates block scheduling overhead for large-N workloads (e.g. N=1000: 499,500 pairs, but only ~80-160 blocks resident). Falls back to original behavior for small workloads. Fully transparent to the caller.
* Added optional CUDA GPU acceleration for batch DTW distance matrix computation (`DTWC_ENABLE_CUDA` CMake option, OFF by default).
* New `dtwc::cuda::compute_distance_matrix_cuda()` computes all N*(N-1)/2 DTW pairs on GPU.
* **Anti-diagonal wavefront kernel**: replaced single-threaded-per-block kernel with multi-threaded anti-diagonal wavefront parallelism. Achieved **24x kernel speedup** (910M -> 22 Gcells/sec), making GPU **5-7x faster than 10-core CPU** with OpenMP.
* **Warp-level DTW kernel for short series** (`dtw_warp_kernel`): for series with `max_L <= 32`, a new kernel packs 8 DTW pairs per block (one warp per pair) using register-based anti-diagonal propagation with `__shfl_sync()`. Eliminates shared-memory cost-matrix buffers and dramatically improves occupancy for short series workloads.
* **Register-tiled DTW kernel** (`dtw_regtile_kernel`): inspired by the cuDTW++ approach (Euro-Par 2020), a new kernel handles medium-length series (32 < max_L <= 256) using register tiling. Each thread processes a stripe of TILE_W columns entirely in registers, with inter-thread communication via `__shfl_sync`. TILE_W=4 covers up to 128 columns, TILE_W=8 covers up to 256. Eliminates shared-memory cost-matrix buffers for medium series, bridging the gap between the warp kernel (L<=32) and the shared-memory wavefront kernel (L>256).
* **`__ldg()` texture cache reads**: global memory reads for series data now use `__ldg()` intrinsic, forcing the read-only texture cache path for ~5-15% improvement on longer series.
* **Banded DTW on GPU**: the `CUDADistMatOptions::band` parameter is now honored by the kernel. Uses the same slope-adjusted Sakoe-Chiba window as the CPU `dtwBanded` implementation. When `band < 0` (default), full unconstrained DTW is computed with zero overhead.
* Python bindings expose `cuda_available()`, `cuda_device_info()`, `compute_distance_matrix_cuda()`, and `CUDA_AVAILABLE` flag.
* **GPU architecture detection** (`gpu_config.cuh`): runtime query and caching of GPU compute capability, SM count, shared memory limits, and FP64 throughput classification (Full vs Slow). Supports up to 16 devices.
* **FP32/FP64 templated kernel**: the DTW wavefront kernel is now templated on compute type (`float` or `double`). New `CUDAPrecision` enum (`Auto`, `FP32`, `FP64`) in `CUDADistMatOptions` controls precision. `Auto` selects FP32 on consumer GPUs (1:32 FP64 rate) and FP64 on HPC GPUs (1:2 FP64 rate), giving up to 32x throughput improvement on consumer hardware with ~1e-5 relative error.
* Requires CUDA Toolkit; all CUDA code is behind `#ifdef DTWC_HAS_CUDA` so the library builds without it.

## CI

* Added CUDA/MPI detection smoke test workflow (`.github/workflows/cuda-mpi-detect.yml`): Linux CUDA compile, Linux MPI build+test, macOS CUDA graceful rejection, macOS MPI build, and Windows MPI configure.

## Build system

* Fixed CUDA detection on Windows with multiple CUDA toolkit versions: auto-sets missing `CUDA_PATH_Vxx_y` env-vars and generates `Directory.Build.props` to persist `CudaToolkitCustomDir` for MSBuild at build time.
* Fixed MSVC flags (`/diagnostics:column`, `/fp:fast`, `/openmp:experimental`) leaking into nvcc by adding `$<$<COMPILE_LANGUAGE:C,CXX>:...>` generator expression guards.
* Fixed `find_package(OpenMP)` failure when CUDA language is enabled by requesting only the CXX component.
* Fixed MPI detection: `MPI_CXX_FOUND` variable didn't propagate from `dtwc_setup_dependencies()` function scope; now checks `TARGET MPI::MPI_CXX` instead.
* Improved MS-MPI SDK detection on Windows with fallback to default install path and actionable error messages.

## Benchmarks

* Added GPU benchmark suite (`benchmarks/bench_cuda_dtw.cpp`): GPU vs CPU comparison, N-scaling, L-scaling with throughput counters (pairs/sec, cells/sec).
* Added MPI benchmark suite (`benchmarks/bench_mpi_dtw.cpp`): distributed distance matrix scaling across ranks with speedup/efficiency reporting.
* Added `benchmarks/README.md` with baseline performance numbers and optimization history.

## Performance / API

* Added pointer+length overloads for all core DTW functions (`dtwFull`, `dtwFull_L`, `dtwBanded`, `dtwMissing_L`, `dtwMissing_banded`) enabling zero-copy calls from bindings. The `detail::*_impl` functions now operate on raw pointers; vector overloads forward to them.
* Python `dtw_distance` and `dtw_distance_missing` now accept numpy arrays via `nb::ndarray` (zero-copy, no vector allocation).
* Eliminated vector copies in `dtwc::core::dtw_distance` pointer overload and `dtw_runtime`.

## CLI

* Rewrote `dtwc_cl` CLI with full TOML configuration file support via CLI11 `--config` flag.
* CLI now supports all clustering methods (FastPAM, FastCLARA, kMedoids Lloyd, MIP), all DTW variants (standard, DDTW, WDTW, ADTW, Soft-DTW), checkpointing, and flexible CSV output (labels, medoids, silhouette scores, distance matrix).
* Added case-insensitive option validation for method, metric, variant, and solver flags.
* Added example TOML configuration file at `examples/config.toml`.

## Tests

* Added cross-language integration tests (`tests/integration/test_cross_language.py`): verifies C++ and Python interfaces produce identical results for DTW distances (L1/squared-euclidean, banded, missing-data), compute_distance_matrix, FastPAM/CLARA clustering, DTW variant consistency (DDTW/WDTW/ADTW/Soft-DTW), checkpoint save/load roundtrip, and end-to-end pipeline (data -> distance matrix -> clustering -> evaluation scores).

## Documentation

* Added MPI and CUDA installation guide (`docs/1_getting_started/3_mpi_cuda_setup.md`) covering all platforms (Windows, Linux, macOS), CMake detection, build flags, and troubleshooting.
* Updated installation guide: added Python installation section, GPU/MPI cross-references, macOS OpenMP note, and CUDA_PATH tip for Linux.

## Examples

* Added Python examples: `04_missing_data.py` (NaN-aware DTW), `05_fast_clara.py` (scalable clustering), `06_distance_matrix.py` (fast pairwise computation with timing), `07_checkpoint.py` (save/resume distance matrices).
* Added MATLAB quickstart example: `bindings/matlab/examples/example_quickstart.m` (DTW distance, distance matrix, clustering).
* Added C++ new-features example: `examples/example_new_features.cpp` (DTW variants, missing data, FastCLARA, checkpointing).

## Documentation
* Fixed DTW formula in `docs/2_method/2_dtw.md`: changed from squared L2 `(x_i - y_j)^2` to L1 `|x_i - y_j|` to match actual code implementation.
* Fixed pairwise comparison count: corrected from `1/2 * C(p,2)` to `C(p,2) = p(p-1)/2`.
* Fixed warping window description: band=1 allows a shift of 1 (not equivalent to Euclidean distance); band=0 forces diagonal alignment.
* Fixed spelling errors: "seies" -> "series", "wapring" -> "warping", "assertain" -> "ascertain".
* Added note clarifying that the default pointwise metric is L1 (absolute difference).
* Added z-normalization section with note that population stddev (N, not N-1) is used.
* Added `docs/2_method/5_algorithms.md`: clustering algorithm documentation with corrected FastPAM citation (JMLR 22(1), 4653-4688) and accurate complexity descriptions.
* Added `docs/2_method/6_metrics.md`: distance metrics documentation with LB_Keogh compatibility table and corrected Huber LB_Keogh reasoning.

## I/O

* Added **Python I/O utilities** (`dtwcpp.io`): `save_dataset_csv` / `load_dataset_csv` (always available), `save_dataset_hdf5` / `load_dataset_hdf5` (requires `h5py`), `save_dataset_parquet` / `load_dataset_parquet` (requires `pyarrow`).
* HDF5 files store series data, names, distance matrices, and metadata in a single compressed file.
* Added optional dependency groups in `pyproject.toml`: `hdf5`, `parquet`, `io` (both).

## New features

* Added **MATLAB MEX bindings** — new `bindings/matlab/` directory with C++ MEX API gateway and MATLAB `+dtwc` package: `dtwc.dtw_distance`, `dtwc.compute_distance_matrix`, `dtwc.DTWClustering`. Build with `cmake .. -DDTWC_BUILD_MATLAB=ON`.
* Added **checkpoint save/load** for distance matrix computation (`save_checkpoint`, `load_checkpoint`). Saves partial or complete distance matrices to disk as CSV + metadata text file, enabling resume after crashes.
* Added `count_computed()` and `all_computed()` methods to `DenseDistanceMatrix`.
* Added `distance_matrix()` accessors and `set_distance_matrix_filled()` to `Problem` class.
* Added Python bindings for `save_checkpoint`, `load_checkpoint`, and `CheckpointOptions`.
* Added **DTW with missing data** (`dtwMissing`, `dtwMissing_L`, `dtwMissing_banded`). NaN values contribute zero cost. Supports L1/SquaredL2, early abandon, banding. Reference: Yurtman et al. (2023), ECML-PKDD.
* Added `dtw_distance_missing` Python binding.
* Added **FastCLARA** scalable k-medoids clustering (`dtwc::algorithms::fast_clara`). Subsampling + FastPAM avoids O(N^2) memory. Reference: Kaufman & Rousseeuw (1990); Schubert & Rousseeuw (2021, JMLR).
* Added Python binding for `fast_clara()`.
* Added `MetricType` parameter (L1, SquaredL2) to all core DTW functions. Template lambda dispatch for zero inner-loop overhead.
* Refactored DTW into `detail::*_impl` helpers with distance callable template parameter.
* Added `metric` parameter to `dtw_distance` Python binding.
* Added `compute_distance_matrix` Python function with OpenMP parallelism.
* Added **LB-pruned distance matrix** (`compute_distance_matrix_pruned`). Precomputes envelopes and summaries once, then uses LB_Kim (O(1)) and LB_Keogh (O(n)) as early-abandon thresholds for each DTW computation. Reduces inner-loop work by 30-60% for correlated series. Enabled by default in the Python `compute_distance_matrix` binding via `use_pruning=True`.
* Added `distance_matrix_numpy()` method to `Problem` Python class.
* Added `-ffast-math` (GCC/Clang) and `/fp:fast` (MSVC) for Release builds.
* Added **Google Highway SIMD infrastructure** (`DTWC_ENABLE_SIMD` option, default OFF). Prototype kernels for future use.
* Added `#pragma omp simd` hints to LB_Keogh and z_normalize.
* Added LB_Keogh, z_normalize, envelope benchmarks.
* Added `DTWClustering` sklearn-compatible Python class.
* Added core type system (`dtwc::core` namespace).
* Added FastPAM1 k-medoids clustering algorithm.
* Added z-normalization, lower bounds, DenseDistanceMatrix, pruned distance matrix.
* Added Google Benchmark integration.

## Architecture

* Removed Armadillo dependency from all hot-path code.
* Replaced `arma::Mat` with `core::ScratchMatrix` and `core::DenseDistanceMatrix`.
* Unified `FastPAMResult` with `core::ClusteringResult`.

## Bug fixes

* Fixed `DenseDistanceMatrix` NaN sentinel broken under `-ffast-math` / `/fp:fast`. Replaced with boolean vector.
* Fixed `throw 1` bare integer throw — now throws `std::runtime_error`.
* Fixed `static std::mt19937` ODR violation — changed to `inline`.
* Fixed `.at()` in hot DTW loop — replaced with `operator[]`.
* Fixed `dtwBanded` 512MB/thread allocation — rolling buffer.
* Fixed `fillDistanceMatrix` integer overflow at N>46K.
* Fixed `dtwBanded` template default from `float` to `double`.
* Fixed `static` vectors data race in `calculateMedoids`.
* Fixed DBI formula.
* Fixed broken examples.
* Renamed `cluster_by_kMedoidsPAM` to `cluster_by_kMedoidsLloyd`.
* Fixed z_normalize tests.

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