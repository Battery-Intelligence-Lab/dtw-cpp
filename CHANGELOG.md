# Changelog {#changelog}

[TOC]

This changelog contains a non-exhaustive list of new features and notable bug-fixes (not all bug-fixes will be listed).


<br/><br/>
# Unreleased

- **Fixed (CLI, C++, Python, MATLAB):** a CSV/TSV series file or a distance-matrix CSV (`--dist-matrix`,
  `read_distance_matrix`) holding a Ctrl-Z (0x1A) byte is refused with `IOError`, as any non-numeric field is (the series
  reader names the row and column). v1.0.0 read series files in text mode, which on Windows ended the file at that byte
  and silently dropped the rows after it.
- **Fixed (CLI, C++, Python, MATLAB):** a series file whose lines end in a bare CR (no LF, as classic Mac OS wrote them) is
  refused with `IOError` naming the row and column whatever the delimiter: with `--delimiter ' '` it was read as one
  series holding every value, as v1.0.0 read it. A CRLF line end reads as LF, a CR anywhere else is part of its field, and
  the space delimiter splits on spaces and tabs only.
- **Fixed (CLI, C++, Python, MATLAB):** a text reader's error names its file in UTF-8 on Windows too. For a name outside
  the code page (a folder's `δ.csv`) the error was lost to "No mapping for the Unicode character exists in the target
  multi-byte code page" (from Python's `read_distance_matrix` an untyped `RuntimeError`), and a non-ASCII name inside it
  (`café.csv`) reached Python garbled.
- **Added (C++):** `dtwc::Config` is declared in `dtwc/config.hpp` with `apply(config, prob)`, which hands a `Problem`
  the clustering settings of a Config (distance, method, solver, device) as `dtwc::run` does before it reads a file, and
  `scores::score(prob, name)`, the score `Result::score(name)` returns.
- **Changed (C++, Python):** `Problem::cluster()` raises `InvalidInput` for a `Problem` without series ("cluster: dataset
  is empty.") or with more clusters than series ("cluster: k must not exceed the number of series."), as `dtwc_cl` and
  Tier-1 `cluster()` do, before any method runs.
- **Changed (Python):** `dtwcpp.cluster(data, k, **keys)` takes every `dtwc_cl` key that is not about files, by its long
  name in snake_case (`metric`, `variant`, `wdtw_g`, `n_init`, `seed`, `linkage`, `solver`, `gpu_precision`, ...): C++
  reads and checks them (an unknown key is `InvalidInput`) and `Problem::cluster()` runs the method, so `method`
  defaults to `auto` as in C++ and the CLI (the 2.0 previews defaulted to `pam` and took four keywords).
  `Problem.cluster()` returns its `ClusteringResult` in Python too.
- **Added (Python):** series already in memory go in as numpy, pandas, pyarrow or Python hold them: `cluster()`,
  `DTWClustering`, `load()`, `compute_distance_matrix` and `Problem.set_data` take a 2-D array, a list of 1-D arrays (any
  lengths), a pandas DataFrame (one series per row, named by its index) or an Arrow array, through one conversion
  (complex values are refused, never cast to their real part), and `Problem.set_data`'s names are optional. `dtwcpp.load()` reads
  an Arrow IPC file (`.arrow`, `.ipc`, `.feather`: the `data` column, named by `name`) through the installed pyarrow, as
  it reads Parquet; text is read by the C++ reader `dtwc_cl` uses. The extension module no longer links the CLI's
  pipeline or config code (CLI11, fkYAML).
- **Added (Windows):** `dtwc_cl.exe` states its name, version and copyright in its file properties (Details tab); v1.0.0's
  carried none.
- **Added (FastCLARA, CUDA):** on a GPU device FastCLARA's sample matrices fill on the GPU and, with CUDA, so does its
  assignment of every series to the k medoids (`cuda::compute_medoid_distances_cuda`: the fill's kernels, one launch per
  block of series, the GPU's and the host's memory bounded whatever N); FP64 gives the CPU's labels, medoids and cost. A
  Parquet file streamed under `--ram-limit` is assigned chunk by chunk the same way. `dtwc_cl --device gpu --method clara`
  with a sample smaller than N runs instead of raising `DeviceError`. On Metal the assignment runs on the CPU, which `-v`
  says.
- **Changed (build):** Gurobi is linked only when you configure with `-DDTWC_ENABLE_GUROBI=ON` (v1.0.0 linked it
  whenever it found an installation, and a MEX or binary built that way needed the Gurobi library to load); HiGHS
  solves the MIP by default. With the option ON and no installation found, the configure stops with an error that
  names the option.
- **Fixed (CLI):** v1.0.0's option names work again, hidden from `--help`, each printing one warning that names its 2.0
  flag: `--Nc`, `--clusters` and `--number_of_clusters` (`-k`), `--probName` (`--name`), `--in` (`--input`), `--out`
  (`--output`), `--skipRows` (`--skip-rows`), `--skipCols` and `--skipColumns` (`--skip-cols`), `--maxIter` and `--iter`
  (`--max-iter`), `--repeat`, `--Nrepeat`, `--Nrepetition` and `--Nrep` (`--n-init`), `--mip_solver` and `--mipSolver`
  (`--solver`), `--bandwidth`, `--bandw` and `--bandlength` (`--band`), `--distMat`, `--distance_matrix` and
  `--distances` (`--dist-matrix`). The 2.0 previews refused them as unknown options. v1.0.0's `--Nc i..j`, which
  clustered once per k in the range, is refused with `InvalidInput` naming the replacement: one run per k.
- **Changed (CLI, breaks a v1.0.0 command line without `--Nc`):** `-k/--n-clusters` is required. `dtwc_cl` without it
  exits 1 naming the flag, before any file is read or written. v1.0.0 printed an `Error processing input` line for a
  missing `--Nc` and exited 0 having clustered nothing; the 2.0 previews clustered with k = 3. `--print-config` writes
  `n-clusters = 0` for a k not given.
- **Changed (CLI):** a run's default name, the prefix of its output files and of its `.dtwm` cache, is its input's file
  name without the extension, or its folder's name (`-i data/cycles.csv` writes `cycles_labels.csv`), where v1.0.0 named
  every run `dtwc`; series passed in memory are `dataset`, as Tier-1 `load()` names them. `--name` still sets it.
- **Changed (CLI):** the default `--method` is `auto`: FastPAM for up to 5,000 series and FastCLARA above on the CPU,
  FastPAM on a GPU. v1.0.0 ran Lloyd k-medoids, which `--method kmedoids` (v1.0.0's `kMedoids`) still selects.
- **Added (C++, Python, MATLAB):** `Method` gains `Auto`, `PAM`, `OneBatch`, `CLARA` and `Hierarchical` (v1.0.0's
  `Kmedoids` and `MIP` keep their values), and `Problem::cluster()` runs any of the nine `--method` names: it publishes the
  labels and medoids as before and now returns them as a `ClusteringResult` with the cost, the iterations and whether the
  method converged. `Problem` gains `set_sample_size`, `set_n_samples` (CLARA), `set_batch_size` (OneBatchPAM) and
  `set_linkage` (hierarchical); `dtwc::run` and `dtwc_cl` cluster through `Problem::cluster()`, with unchanged results.
- **Changed:** a distance matrix that enters a `Problem` from outside its fill is checked once, where it enters, and a
  ±inf distance raises `InvalidInput` naming the first such pair: `read_distance_matrix`, `load_checkpoint`,
  `use_mmap_distance_matrix`, Python and MATLAB `set_distance_matrix`, and the next `fill_distance_matrix()` after a
  write through C++ `writable_distance_matrix()` (NaN still marks a pair to compute). v1.0.0 clustered such a matrix; the 2.0
  previews refused it only when FastPAM, Lloyd k-medoids, FastCLARA or OneBatchPAM read the value, with a check on
  every read.
- **Changed (C++):** `Problem::dist_by_ind` reads the distance matrix and computes nothing: an inlined O(1) load, where
  every lookup re-checked the distance settings (FastPAM's swap made N² such lookups per sweep). It needs a matrix that
  holds the pair: call `fill_distance_matrix()` first, as the library's methods that read the matrix now do
  (`assign_clusters`, `calculate_medoids` and `init::Kmeanspp` fill it before reading; `find_total_cost` computes the N
  point-to-medoid distances when no matrix is filled). The v1.0.0 `distByInd`, which computed one pair on demand, fills
  the matrix on its first call. `is_distance_matrix_filled()` is a flag again, as in v1.0.0, not a scan of the matrix.
  A direct write to the v1.0.0 `band` field takes effect at the next `fill_distance_matrix()`, where v1.0.0 kept the
  distances computed under the old band. The distance settings are one private `DistanceConfig`, changed through
  `set_distance`, `set_band`, `set_metric`, `set_variant` and `set_missing_strategy`; a change drops the matrix and the
  clustering.
- **Changed (C++, breaks source):** counts, labels and medoids are `dtwc::index_t` (`std::int64_t`): `Problem::clusters_ind`,
  `centroids_ind`, `labels()`, `medoids()`, `size()`, `n_clusters()`, `dist_by_ind`, `ClusteringResult`, `Result::labels()`
  and `medoids()`, every algorithm's `k` and the loaders' row, column and series counts; `band`, `max_iter`, `n_init` and
  `n_samples` stay `int`, and seeds are `std::uint64_t`. Code that keeps the outputs in a `std::vector<int>` stops
  compiling; declare it `std::vector<dtwc::index_t> medoids = prob.medoids();` (or `auto`). `set_clusters(std::vector<int>&)`
  still compiles, `[[deprecated]]`. `adjusted_rand` and `normalized_mutual_info` key their counts on whole labels, and
  `dtwc_cl` reads 64-bit `-k`, `--sample-size`, `--batch-size`, `--skip-rows`, `--skip-cols` and `--seed`.
- **Changed (Python, MATLAB):** labels and medoids are `np.int64` arrays in Python (`Problem.labels()`, `medoids()`,
  `clusters_ind`, `centroids_ind`, `ClusteringResult.labels` and `medoid_indices`, `Result.labels` and `medoids`, the
  estimators' `labels_` and `medoid_indices_`, `BarycenterClusteringResult.labels`; each read is an independent copy) and
  1-based exact doubles in MATLAB (they were `int32`, as were the `iterations` and dendrogram `n_points` fields). Python and
  MATLAB take `k`, `sample_size`, `batch_size`, `max_points` and the skip counts as 64-bit values, and `fast_clara` takes a
  64-bit seed, so the Python checks that refused `k` or a skip count above 2^31 - 1 are gone; `max_iter`, `n_init` and
  `band` stay `int`. No Python name v1.0.0 shipped (`cluster_size`, `centroid_of`) changes type.
  Code that kept `list`-typed labels (`labels == [0, 1]`, `labels.index(1)`) calls `.tolist()` first.
- **Changed (GPU):** the CUDA backend needs an NVIDIA GPU of compute capability 8.0 or newer (Ampere, 2021: A30/A100/RTX 30
  and later). An older GPU is refused with `DeviceError` naming its compute capability when a fill first reads the device,
  before anything is allocated. The default `DTWC_CUDA_ARCH_LIST` is `80-real;86-real;89-real;90` (was
  `60;70;75;80;86;89;90`): machine code for compute capability 8.0 to 9.0 and PTX that later GPUs compile at load.
- **Changed (GPU):** the CUDA distance-matrix fill accepts series of any length. A series whose wavefront buffers (three
  anti-diagonals) did not fit a block's shared memory was refused with `DeviceError`: on an RTX 4000 Ada any series longer
  than 8,446 samples in FP32 or 4,223 in FP64, so `data/dummy` (up to 9,405 samples) could not run with `--device gpu`.
  Those buffers now live in global memory, with the shared-memory kernel's arithmetic: the tests find every L1 distance,
  with and without a band, the host kernel's bit for bit (in FP32, the host FP32 kernel's), and `data/dummy` clusters on
  the GPU with the labels and medoids of `--device cpu`. Above 2,048 samples the fill keeps the anti-diagonals in shared
  memory only where three blocks fit an SM, which is where that is the faster route.
- **Changed (GPU):** the CUDA distance-matrix fill runs for any number of series; it refused more than 65,536. It
  computes at most 2^27 pairs per launch and copies each launch's share of the packed matrix straight into the
  `Problem`'s matrix, on the heap or memory-mapped, where it built an N×N matrix on the GPU and two more on the host
  and copied them element by element. On an RTX 4000 Ada at N = 20,000 (FP32) one fill's memory above its input fell
  from 6.0 to 1.6 GiB on the host and from 1.6 to 1.1 GiB on the GPU, and 9,000 series of length 100 fill in 1.17 s
  instead of 1.50 s. A fill its backend refuses (no device, a device index past the last) no longer leaves a matrix
  allocated.
- **Changed (performance, Windows):** a DTW cell no longer makes a library call. The MSVC STL compiles `std::min({…})` to an
  out-of-line helper (`__std_min_d` under clang, `__std_min_element_d` under cl), which full, banded, ADTW, AROW, MSM, TWE and
  the DBA alignment called once per cell; they now nest two-argument `std::min` and keep the value a cell stores in a register.
  On an Intel Core Ultra 9 285, pinned DTW of two 1000-sample series drops from 7.2 ms to 1.4 ms (band 100: 1.4 to 0.26 ms);
  results are unchanged digit for digit.
- **Changed (memory):** `dtwFull` and Soft-DTW (the distance-matrix fill, `soft_dtw`, `distance::soft_dtw`) keep one
  rolling column per thread, as `dtwFull_L` does, instead of a full n × m matrix that each thread kept for its lifetime:
  512 MB per thread for two 8,000-sample series. A Soft-DTW fill of eight such series on 8 threads peaked at 3.4 GB and
  now at 14 MB. The distances are unchanged digit for digit; `soft_dtw_gradient`, whose backward pass reads every cell,
  keeps its two matrices.
- **Changed (performance):** the CPU distance-matrix fill computes standard DTW (L1 or squared-L2 cost, univariate, no
  missing-data strategy) between a series and 8 others of its length at once (16 in `float32`), one pair per SIMD lane; every
  distance is what the one-pair kernel returns, bit for bit unless the compiler contracts a multiply-add into an FMA in one
  kernel and not the other (GCC does by default), in which case the two differ in the last bits. On an Intel Core Ultra 9 285
  (24 threads) the unbanded fill of ECG5000's 4,500 series drops from 66 s to 3.9 s, and a band-50 fill of 50 series of
  length 1,000 runs 5.1× faster.
- **Changed (performance):** OneBatchPAM's final exact assignment, each of the N series against the k medoids (N-1 DTW calls
  per medoid outside the batch), runs on all OpenMP threads instead of one. Labels, medoids, total cost and the distance count
  are bit for bit what the serial loop returned, at any thread count. On an Intel Core Ultra 9 285 (24 threads, shared
  machine) with N = 2,000, k = 10 and 200-sample random walks, the assignment takes 0.041 s instead of 0.90 s (21.9×) and the
  whole call 0.96 s instead of 1.82 s (1.9×); the table fill before it was already parallel and now bounds the gain.
- **Changed (performance):** OneBatchPAM fills its N x m batch table, m(N-1) DTW calls and nearly all of a call, with the same
  SIMD lanes as the distance-matrix fill: 8 batch series of the row's length per call (16 in `float32`), one pair per lane.
  Pairs of other lengths, every DTW variant, missing-data strategy and multivariate input keep the one-pair path. Labels,
  medoids and the distance count are what the one-pair fill returned; so are the table and the total cost, bit for bit unless
  the compiler contracts a multiply-add into an FMA in one kernel and not the other (GCC does by default; squared L2 only),
  in which case they differ in the last bits. On an Intel Core Ultra 9 285
  (24 threads, shared machine) with N = 2,000, k = 10 and 200-sample random walks the table fill takes 0.20 s instead of
  0.93 s (4.6×) and the whole call 0.25 s instead of 0.98 s (3.9×); on one thread, 3.0 s instead of 19.5 s (6.5×).
- **Changed (build):** llfio is header-only, from SHA-256-pinned GitHub archives, behind one `llfio_hl` target:
  `cmake/Dependencies.cmake` loses the quickcpplib bootstrap, its patched nested superbuild and `add_subdirectory(llfio)`
  (196 lines out, 69 in). The superbuild compiled quickcpplib from `master` and outcome from `develop`, whatever they
  were at configure time; both are now pinned, with wg14_signals, span-lite, byte-lite and (Windows) ntkernel-error-category
  at the commits llfio and quickcpplib record. The preprocessed `<llfio/v2.0/llfio.hpp>` is identical to the superbuild's.
  A fresh Windows configure took 44 s instead of 198 s (warm download cache). The Python wheels and the release CLI
  archives are built with `DTWC_ENABLE_LLFIO=ON`, so they map the distance matrix (`use_mmap_distance_matrix`,
  `--mmap-threshold`); they were built without it.
- **Changed (mmap, Windows):** a new mmap distance-matrix cache is no longer a sparse file (llfio's default on NTFS);
  random reads from a filled sparse cache measured 1.9x slower.
- **Changed (checkpoint, mmap):** a distance checkpoint is one file, `<dir>/<name>.dtwm`, the file a memory-mapped matrix
  lives in (a 48-byte header of magic, version 4, N and SHA-256 fingerprint, then the packed doubles); a checkpoint or cache
  written before this change is not readable (2.0-born, never released). `load_checkpoint` returns `false` only when the
  file is absent and raises `InvalidInput` for other data or settings and `IOError` for a damaged file, where it returned
  `false` and the CLI recomputed over the checkpoint.
- **Fixed (C++ compatibility):** `Problem::cluster_by_kMedoidsPAM()`, v1.0.0's name for the Lloyd k-medoids run, compiles
  again as a deprecated forwarder to `cluster_by_kmedoids_lloyd()` (the 2.0 spelling `cluster_by_kMedoidsLloyd()` is gone), and
  `Problem::maxIter` and `N_repetition` are plain public fields again, as in v1.0.0, with no deprecation warning.
- **Removed (build):** the `dtwc_main` demo executable (`dtwc/main.cpp`), which ran a MIP on `data/dummy` relative to the
  working directory; `examples/cpp/MIP_single.cpp` (`-DDTWC_BUILD_EXAMPLES=ON`) runs the same MIP and takes the data folder
  as an argument.
- **Fixed:** a medoid at distance 0 from another medoid (a duplicate series) is labelled with its own cluster by
  k-medoids (Lloyd), FastPAM, FastCLARA, OneBatchPAM and LR-core. v1.0.0's Lloyd gave it to the first tied medoid and
  published the other cluster empty; LR-core refused the valid optimum with `SolverError`.
- **Fixed (C++):** `scores::silhouette()` on a Problem that has not been clustered raises `InvalidInput`; v1.0.0 printed a
  line and returned one `-1` per series, a vector that reads as a (poor) score.
- **Fixed (C++, Python, MATLAB):** `find_total_cost()` and `write_clusters()` on a Problem that holds no clustering raise
  `InvalidInput` ("... cluster it first"); v1.0.0 read the empty label vector (an access violation in Python and in MATLAB
  R2024b), or, after `set_n_clusters`, its zeros. `print_clusters`, `write_medoid_members`, `calculate_medoids` and the scores
  refuse the same way. `set_n_clusters` no longer sizes `clusters_ind` and `centroids_ind`, which stay empty until a
  clustering writes them; a Problem is clustered when it holds one label per series and one medoid per cluster
  (`Problem::require_clustered`), so a clustering goes stale when `set_n_clusters` changes the count; `set_data` and
  `set_view_data` empty both vectors, whatever the new number of series (the labels describe the old series).
- **Changed (C++, Python, MATLAB):** `set_n_clusters(k)` with k < 1 raises `InvalidInput` naming the value; v1.0.0
  (`set_numberOfClusters`) accepted k = 0, which `cluster()` refused later, and failed on k = -1 with an untyped "vector too
  long" from a resize. `set_band(b)` with b < -1 raises the same, so `dtwc_cl --band -5` stops with that error where v1.0.0
  ran full DTW (a band below -1 has always run as full DTW, and a direct write to the `band` field still does). k above N is
  still refused by `cluster()`, since the data may change after the setter.
- **Changed (exact solvers):** `Method::MIP` and `Method::LRCore` publish through the new
  `Problem::set_result(ClusteringResult)`, which refuses a malformed clustering with `InvalidInput`. A solve that fails with
  `SolverError` leaves the Problem holding a valid clustering (the FastPAM warm start), not necessarily the one it held before
  (basic exception guarantee). LR-core seeds its upper bound with FastPAM instead of Lloyd k-medoids: the optimal cost is
  unchanged, the medoids may differ where optima tie.
- **Fixed (docs):** the `/cluster` and `/help` commands showed `--k` and `--output-dir`, and `/troubleshoot` showed
  `--repetitions` and `--prune`, none of which `dtwc_cl` has; they now show `-k`, `--output` and `--n-init`, and the pruning tip
  is gone. CI now fails when a page in `docs/content`, `README.md` or `.claude/commands` shows a `dtwc_cl` flag that the live
  `dtwc_cl --help` lacks (`scripts/check_docs.py`).
- **Changed (CLI):** `dtwc_cl` is `cli::bind` + `dtwc::run(Config)` (1,996 → 133 lines), with byte-identical outputs on every
  configuration compared. `--device` reads the one device grammar (`gpu`, `gpu:N`, Metal on macOS; `cuda` is `gpu`); `hpc` raises
  `DeviceError` (submission is `slurm_remote.sh submit-cluster` / Python's `device="hpc"`). On `gpu`, `auto` runs pam at any N;
  onebatch, tadpole, or a setting the GPU cannot honour raise `DeviceError` before the input is read. Squared L2 runs on the
  CPU; `--checkpoint-interval 0` saves once at the end; every check that needs no data, the MIP settings included, runs before
  any I/O; the loader prints no progress lines, and `-v` reports the series loaded.
- **Added (Python):** `dtwcpp.load()` reads a list-per-row Parquet file, or a folder of them, with the installed pyarrow (the
  `dtwcpp[parquet]` extra; the wheel links no Arrow C++): each row of the first list column is a series, named by the first
  string column. Without pyarrow it raises `ImportError` naming the extra.
- **Added (CLI):** `--print-config` writes the parsed settings as a TOML config file; the binary now reads `--delimiter`,
  and `--lr-max-nodes`.
- **Changed (C++ Tier-1):** `cluster()` wraps `run`, with results unchanged; path datasets read Parquet, Arrow and `.dtws`; the
  aliases `obp` and `lr` are accepted; `device="hpc"` raises without reading `.env`; `Result` reports `method()`, `iterations()`
  and `converged()`. `detail/tier1_method_resolution.hpp` is removed. An unreadable mmap-cache parent, an output directory that
  cannot be created, and a YAML config on a build without YAML each raise `IOError`.
- **Fixed (errors):** `save_checkpoint` to a path it cannot create, and an mmap matrix or series store that llfio cannot size, map or
  flush (a full disk, a file-size quota), raise `IOError` naming the path; Python saw `RuntimeError`, MATLAB `dtwc:runtime`. An mmap
  cache made for other data or configuration, and `skip_cols` wider than a file's rows, raise `InvalidInput` (was `IOError`), as a
  CSV matrix of the wrong size and in-memory `skip_cols` already did. `pdlp_lp_bound` with `use_gpu` on a build without
  `-DDTWC_HIGHS_GPU=ON` raises `DeviceError` instead of warning and solving on the CPU.
- **Fixed (distance, breaking):** `dtwc::distance::dtw(x, y, params, band, metric)` returned the L1 distance for another metric with
  WDTW, ADTW, Soft-DTW, MSM or TWE, whose kernels take none; it now raises `InvalidInput`. Standard DTW, DDTW and the missing-data
  strategies keep the metric. Python's `dtwcpp.distance.dtw` is one binding over this function, so it computes and
  refuses the same configurations (`InvalidInput`, a `ValueError`). `examples/cpp/example_new_features.cpp` compiles (a most-vexing parse) and builds under
  `-DDTWC_BUILD_EXAMPLES=ON`; `example_project/main.cpp` checks `set_solver`.
- **Fixed (MATLAB, macOS):** the MEX no longer aborts MATLAB ("OMP: Error #15") at its first parallel region: MATLAB loads its own
  libomp at start-up, and the MEX now runs on that copy (`@rpath/libomp.dylib`) instead of loading a second one, so MATLAB's
  `maxNumCompThreads` also limits the MEX's threads. The `dtwc.test.gpu` tests branch on the probe's result, as the C++ and
  Python ones do, so a Metal build checks its GPU against the CPU.
- **Added (metric):** `Problem::set_metric` / `metric()`. The metric is now part of a Problem's distances — the CPU fill and
  lookups, CUDA / Metal, the mmap and checkpoint identities, autosave and FastCLARA's samples. A metric other than L1 needs
  Standard DTW (with any missing-data strategy) or DDTW, else `InvalidInput`, the rule `dtwc::distance::dtw` applies too.
  L1 stays the default and computes as before.
  `use_mmap_distance_matrix(path, metric)` adopts the metric, so the CPU fills a squared-L2 cache instead of refusing it.
- **Fixed (multivariate, breaking):** the pruned CPU fill (by default at 64 series or more with a band) on multivariate
  series read the channels as one interleaved series, so every pair was wrong; it now fills with the multivariate kernel.
  A direct `fill_distance_matrix_pruned` call on multivariate or non-L1 data raises `InvalidInput`.
- **Removed (C++, pre-tag):** `settings::paths` and its four path setters (2.0-born; v1.0.0 had none). A `Problem` writes to
  `output_folder()`, `./results/` by default; pass data paths explicitly.
- **Fixed (FastCLARA, TADPole):** CLARA's sub-samples use the parent's metric, GPU index and precision (in-memory samples are views
  on the CPU and copies on a GPU); a missing Parquet build is `IOError`. TADPole under a non-L1 metric computes exactly
  instead of pruning with L1 bounds.
- **Added (config):** `dtwc::Config` (`dtwc/cli/config.hpp`) — every setting of one run, keyed by the `dtwc_cl` long names,
  with `dtwc_cl`'s defaults; `cli::bind(CLI::App&, Config&)`, the one key table, reading TOML or YAML `--config` files through
  `dtwc_cl`'s reader; `to_config_text` and `parse_config`. One string↔enum table beside each enum (`dtwc/base/names.hpp`:
  `parse_name`, `name_of`). `dtwc_cl` is unchanged for now. The library links CLI11 and, when YAML is enabled, fkYAML, and
  publishes `DTWC_HAS_YAML`.
- **Fixed (MATLAB):** `Problem.set_method('pam')` / `('auto')` run PAM and `auto`; they ran Lloyd k-medoids. MATLAB reads
  and reports device names only through the C++ grammar (its own ordinal parser and the MEX's second canonicaliser are
  gone).
- **Fixed (Python, hpc):** a file `Dataset` with a `delimiter` raises `InvalidInput`, because the SLURM transport does not carry
  it (it was dropped silently); an in-memory dataset's `skip_cols` is applied once, not twice.
- **Fixed (Python):** `Problem.dist_by_ind(i, j)` raises `InvalidInput` for an index outside `[0, N)`, naming the index and N;
  v1.0.0's `distByInd` read past the distance matrix (undefined behaviour). C++ `Problem::dist_by_ind` stays unchecked.
- **Fixed (Python):** `Problem.centroid_of(i)` raises `InvalidInput` for an index outside `[0, N)`, and for a `Problem`
  that holds no clustering yet; v1.0.0 read `centroids_ind[clusters_ind[i]]` unchecked and killed the interpreter
  (or returned a stale medoid). C++ `Problem::centroid_of` stays unchecked.
- **Fixed (CLI):** `--dtype float32 --device cuda` raises `DeviceError`; the CUDA fill received no series. Parquet / Arrow IPC
  input on a build without Arrow is `IOError`, like `.dtws` input without llfio. The documented `--method` default is `auto`.
- **Fixed (load):** Tier-1 `load` read errors again start `load: failed to read '<path>': ` and stay `IOError`; the partial
  byte-order-mark error names the file.
- **Breaking (errors):** the library's remaining bare `throw std::…` sites raise the type `docs/api-contract-2.0.md` §5 names —
  bad input or configuration → `InvalidInput`; a file, stream, filesystem or format failure → `IOError`; a Metal or CUDA failure
  → `DeviceError` (190 sites). Python sees `ValueError` / `OSError` subclasses where it saw `RuntimeError` — for example from
  `cut_dendrogram`, `read_distance_matrix` and the readers — and MATLAB `dtwc:invalidArgument`
  / `dtwc:ioError` where it saw `dtwc:runtime`; `adjusted_rand` and `normalized_mutual_info` raise `InvalidInput`, not
  `std::invalid_argument`. Messages are unchanged. A check only a programming error can reach throws `std::logic_error`.
  Building the tests no longer needs Python.
- **Fixed (Benders):** a direct `MIP_clustering_byBenders` call with the deprecated `N_repetition` below 1 raises `InvalidInput`
  instead of terminating the process.
- **Fixed (Python, devices):** `cuda` is a spelling of `gpu` in Python too, as in C++ and MATLAB (contract §6.1):
  `dtwcpp.device("cuda")` and the `device=` of `compute_distance_matrix`, `cluster` and `DTWClustering` select this build's
  GPU (CUDA, else Metal) and `device()` returns `"gpu"`; on a Metal build they raised `DeviceError`. With no GPU, `cuda` still
  raises. Python parses device names with the C++ grammar (`parse_device`); its own copy is gone.
- **Fixed (distances, breaking):** NaN or ±inf passed to a distance function no longer reaches the kernels, which
  returned NaN, the unreachable 1.8e308 or an ordinary-looking number. `dtwc::distance::*` and
  `soft_dtw_gradient` check x and y once and raise `InvalidInput` naming the series, the 0-based position and the fix;
  the Python distance functions and MATLAB `dtwc.distance.*` now call them, and Python's `compute_distance_matrix`
  (CPU, CUDA, Metal) and `compute_lb_keogh_cuda` check every series first. `missing`, `arow` and the ZeroCost / AROW /
  Interpolate strategies still read NaN as missing and reject only ±inf. The single-variant `distance::` functions
  now validate their parameters as the dispatcher did. The fill's machine code is unchanged. A `Problem` rejects the
  same values before computing, on the fill, the lazy lookups and the kernel accessor: ±inf always, NaN under
  `MissingStrategy::Error`, with `InvalidInput` naming the series and position (before, ±inf filled the matrix with
  inf, and NaN raised an untyped error on the fill only).
- **Breaking (`Problem`):** `set_max_iter` and `set_n_repetitions` reject values below 1 (zero iterations reported
  the initial medoids' cost as a result). `read_distance_matrix` rejects a matrix whose size is not the series
  count, or an empty file, with `InvalidInput` (before, it was discarded at the first lookup and every distance
  recomputed silently). `get_name` and `p_vec` raise `InvalidInput` on storage they do not own — a view, an mmap
  store, and for `p_vec` Float32 or metadata-only series — instead of reading out of bounds in a Release build.
  `set_solver` and `load_checkpoint` are `[[nodiscard]]`: a `false` means Gurobi was unavailable or the checkpoint
  was not loaded. The move assignment is no longer `noexcept`, which it could not honour. `Result::save` and
  `Problem`'s writers check each file after closing and raise `IOError` for a write that failed after opening.
- **Added (devices):** `Problem::set_device(Device, index = 0)`; in Python `Problem(name="", *, device="cpu")` and
  `Problem.set_device(name)`, in MATLAB `dtwc.Problem(name, 'Device', d)` and `set_device`, taking the names `dtwcpp.device()`
  takes. `gpu` selects CUDA, else Metal. A Problem does not follow the
  process-wide device; Tier-1 `cluster()` now calls `set_device`.
- **Breaking (GPU):** a CUDA or Metal fill with a DTW variant other than standard, a missing-data strategy, multivariate data,
  Float32 series or view / mmap series raises `DeviceError` naming the setting; before, the GPU silently computed standard
  univariate DTW. On Metal, precision FP64 (`set_gpu_precision(GpuPrecision::FP64)`) and a GPU index other than 0 raise
  too, instead of silently running FP32 on the default GPU. A squared-L2 mapped cache now
  fills on the GPU instead of being refused.
- **Breaking (band):** a band narrower than the length difference between the longest and shortest series raises
  `InvalidInput`, naming both series and the smallest feasible band, before any pair is computed; before, those pairs were
  stored as 1.8e308 and summed into the clustering cost. The check runs on every route that computes distances —
  the fill, the lazy lookups and the kernel accessor that OneBatchPAM and FastCLARA's assignment use (Tier-1 `auto`'s
  choice above N = 5000); a complete precomputed matrix (a loaded `--dist-matrix` or checkpoint) computes nothing and is
  served under any band. Soft-DTW, MSM and TWE, which ignore the band, are unaffected.
- **Breaking (CLI):** `dtwc_cl` exits 1, naming the option or file and the fix, when a `--checkpoint`
  save fails, a `--dist-matrix` cannot be loaded, or a labels / medoids / silhouettes write fails after
  the file was opened (full disk, quota). Before, it warned or left a truncated file, and exited 0. A
  `--checkpoint` path that cannot be a directory now stops the run before any data is read; the
  end-of-run checkpoint is saved before the results, and a save that fails is reported after the
  results are written, so neither failure loses the other's output. `--solver gurobi` on a build
  without Gurobi exits 1 (it silently solved with HiGHS), as does `--max-iter 0`. The `-k` error
  names `-k/--n-clusters`, not the deprecated `--clusters`.
- **Breaking (CLARANS, Benders):** CLARANS with `num_local < 1`, and Benders with no data or with k
  outside [1, N], raise `InvalidInput`; before, CLARANS published an empty clustering of cost `DBL_MAX`
  and Benders returned silently. CLARANS's size errors are `InvalidInput` too, and its automatic
  neighbour count is 64-bit.
- **Fixed (`.dtws`, Arrow IPC):** a `.dtws` header with `ndim = 0` raises `IOError` instead of
  dividing by zero, and the cache opens read-only, so a read-only cache file opens. An Arrow IPC
  `name` column may be Utf8 or LargeUtf8 (Polars' default); any other type, or an `ndim` metadata
  value that is not a positive integer, raises `IOError` naming the file.
- **Breaking (build):** including `dtwc.hpp` under `-ffinite-math-only` (implied by `-ffast-math`) is a
  compile error, because NaN marks missing values and pruned distances; add `-fno-finite-math-only`.
- **Fixed (build):** Clang builds no longer print "optimization flag '-fno-signaling-nans' is not supported" once
  per translation unit; the flag is passed to GCC only, and generated code is unchanged.
- **Fixed (packaging, macOS):** numbers are parsed by the vendored fast_float 8.3.0 instead of
  floating-point `std::from_chars`, which Apple's libc++ provides only from macOS 26 (2025), so the
  library no longer builds only for the newest macOS. The CLI archive and the wheels target macOS
  13.3, the first release with floating-point `std::to_chars`.
- **Fixed (packaging, macOS):** the CLI archive and the macOS wheel bundle LLVM's OpenMP runtime
  23.1.1 built for macOS 13.3 from pinned source (`scripts/build_libomp_macos.sh`) instead of
  Homebrew's `libomp`, which needs macOS 26, so both start on 13.3. The release smoke test fails if
  any bundled binary needs a newer macOS.
- **Fixed (packaging):** HiGHS is linked statically into the Python extension, so a wheel no longer needs a shared
  `libhighs` it never contained: the macOS wheel repair succeeds, and a source install imports without
  `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH`. The CLI archive is unchanged.
- **Fixed (Python, HPC):** `device="hpc"` works from an installed wheel: the SLURM wrapper and job
  script ship in the package (`dtwcpp/_slurm/`, found with `importlib.resources`) instead of being
  looked up in a source checkout. `.env` and `results/` live in `repo_root`, else `$DTWC_REPO_ROOT`,
  else the working directory; `bash scripts/slurm/slurm_remote.sh` still works in a checkout and always uses that
  checkout, whatever `DTWC_REPO_ROOT` says, and `upload` / `submit-*` refuse to run outside one. A job
  without a seed no longer fails under macOS's bash 3.2 ("SEED_ARGS[@]: unbound variable").
- **Breaking (text input):** a blank line is no longer an empty series. Trailing blank lines are
  ignored; a blank line followed by data is an error naming its row. `Problem::set_data`,
  `set_view_data` and `Problem(name, DataLoader&)` reject an empty series with `InvalidInput` naming
  its index.
- **Breaking (folder input):** each line of a one-series-per-file folder holds one value after
  `--skip-cols`; a pandas `index,value` file such as those in `data/dummy` needs `--skip-rows 1
  --skip-cols 1`, and without them the error says so (before, the index column was clustered).
- **Fixed (text input):** dot-files in a folder are skipped; a pipe without a BOM reads its rows;
  `+-1` is rejected; whitespace is ASCII in every locale; an empty first value after `--skip-rows`
  is an error instead of being dropped. Reading is also faster (491 → 775 MB/s on one 190 MB file,
  advisory).
- **Breaking (`--dist-matrix`):** a non-square, short-row or asymmetric matrix file raises
  `InvalidInput` instead of being truncated or last-write-wins; triangle files still load.
- **Fixed (Python):** `load_dataset_csv` handles a UTF-8 BOM instead of dropping the first series;
  `device="hpc"` sends values at full precision (`repr`) instead of rounding them to 10 digits.
- **Fixed (lower bounds, breaking):** `compute_distance_matrix_metal` / `_cuda` with `use_lb_keogh`
  could prune a pair whose DTW distance is within `lb_threshold`: excesses were not squared under
  squared L2, Metal's default envelope for full DTW was `max(1, max_L/10)`, and an `INT_MAX` band
  overflowed the envelope kernels. The bound now squares each excess, uses the DTW window as its
  default envelope and clamps the radius; a pruned pair is NaN, not `DBL_MAX`; a Metal
  `lb_envelope_band` narrower than the window raises `InvalidInput`. `compute_envelopes`,
  `compute_envelope` and `compute_envelopes_mv` with a negative band now build the full-DTW (global)
  envelope instead of a radius-0 one. The CUDA half is not yet compiled on a CUDA machine.
- **Changed (tests):** a test passes when Catch2 reports at least one assertion in at least one test
  case, no failure, and no skip unless it is registered `MAY_SKIP`. Per-test floors are removed, which
  fixes the five macOS failures; `unit_test_mpi` is registered only when MPI is built in;
  `tests/integration/stress_test_cli.sh` fails on a skip.
- **Changed (tests):** the F22 apparatus (3,804 script lines, three probe targets, two fixtures) is
  replaced by one test, `test_deprecated_shims_warn`: seven sampled 1.x shims must compile and warn.
  `unit_test_deterministic_series` no longer counts calls inside sibling test files.
- **Changed (tooling):** `scripts/check_supply_chain_pins.py` drops its count of CMake files and keeps
  every SHA and tag pin; `check_ipo_inlining` is a manual tool rather than a test.

- **Changed (gates):** `scripts/check_docs_contract.py` keeps only the checks that compare user docs with
  the code or stop a derivation oracle passing by skipping, and drops its pins on process records, CHANGELOG
  prose, source line numbers, test floors and bare finding numbers; `scripts/check_record_hygiene.py` is deleted.
- **Fixed (build, numerics):** the Release floating-point relaxations no longer reach third-party
  dependencies. They were applied with a directory-scope `add_compile_options()`, which every
  subdirectory added afterwards inherits — including the ones CPM creates for fetched projects — so
  146 dependency translation units were being compiled with `-fassociative-math` and friends. 31 of
  them were HiGHS, whose simplex and interior-point code is precisely where reassociating a
  floating-point sum can move a pivot or a tolerance comparison, and 107 were Catch2, meaning the
  test framework's own floating-point matchers were built relaxed. The flags now ride on the
  `dtwc_options` interface target, which is created after the dependencies are added and is linked
  only by our own targets. Flags on our own translation units are unchanged.
- **Added (build, numerics):** `DTWC_FP_MODEL` selects `fast` (the existing relaxations, and still
  the default) or `strict` (none of them, `/fp:strict` on MSVC). It is a cache variable, so it is
  recorded in machine and benchmark records rather than being invisible to them. An unrecognised
  value is a configure error.
- **Changed (packaging):** Release archives and wheels require an x86-64-v3 CPU (AVX2 and FMA: Intel 2013+,
  AMD 2015+). They are compiled for that level (`-march=x86-64-v3`, MSVC `/arch:AVX2`) and never for the CPU of
  the CI runner that built them. Apple Silicon and Linux arm64 wheels and archives are unchanged.
  `DTWC_ARCH_LEVEL` is the one build option for the level: `native` (the default for C++ builds), `v3` (the
  default for Python builds) or `v4` (AVX-512); `DTWC_ENABLE_NATIVE_ARCH` is gone.
- **Changed (headers):** the foundation headers moved into `dtwc/base/` — `error.hpp`,
  `settings.hpp`, `missing_utils.hpp`, `parallelisation.hpp`, `timing.hpp`, `env.hpp`,
  `system_memory.hpp` and `random_engine.hpp`. The old paths still work for one release and now emit
  a compile-time message naming the new one; they will be removed in the next release. Every include
  inside this repository was updated, so the message only reaches code outside it.
- **Changed (internal headers):** the MIP solvers' sparse-matrix helpers and tolerances —
  `dtwc::solver::{Element, Coordinate, Triplet, RowMajor, ColumnMajor, epsilon, isAround,
  isFractional}` and the two comparators — moved from `dtwc/types/{element_types,types_util}.hpp`
  into a single `dtwc/mip/solver_types.hpp`. They were in the base layer and reached by
  `utility.hpp`, so 97 of 297 translation units compiled them; their only consumer in the library is
  `mip_Highs.cpp`, which is now the only one that sees them. `dtwc/types/` keeps `Range` and
  `Index`. Because these names arrived through `<dtwc/dtwc.hpp>` transitively, code that used them
  without including a solver header must now include `<dtwc/mip/solver_types.hpp>`; the umbrella
  header has never included anything from `mip/`, and no forwarding header is left behind, because
  one at the old path would be a `base` → `mip` include — the coupling this removes. Also adds the
  `<cmath>` that `isAround` and `isFractional` always needed and had been getting by accident.
- **Changed (internal headers):** `dtwc::randGenerator` now lives in `dtwc/random_engine.hpp`
  rather than `settings.hpp`, and `settings.hpp` no longer includes `<random>` or `<iostream>`. It
  is reached by around forty translation units and was pulling both in for things almost none of
  them use. The public name is unchanged — `dtwc/dtwc.hpp` includes the new header — so only code
  that included `settings.hpp` directly and relied on it transitively needs the new include. The
  engine, its seed and its behaviour are untouched.
- **Removed (dependencies):** Eigen. It was the project's only copyleft dependency (MPL-2.0) and the
  only MPL obligation in the Python wheel, and it was carried for two uses: `ScratchMatrix`'s base
  class, now a grow-only uninitialised buffer with the same semantics, and a `to_full_matrix` return
  type that every caller immediately copied into a `std::vector<double>` — so that function now
  returns one directly, removing an N×N copy. A third link, from `mip/`, included nothing at all.
  The MPL-2.0 §3.2 source offer is gone from `THIRD_PARTY_LICENSES.md` with it. The ~5 %
  `BM_dtwFull` slowdown first recorded for this change does not reproduce: interleaved A/B runs of
  fresh builds agree within noise, and the hot loop compiles to the same 17 instructions per cell
  either way. What does move it is code placement: a 4-byte shift that splits an `fcmp`/`fcsel` pair
  across a 64-byte boundary costs 28–46 % on an Apple M5 Pro, so a timing gap between two builds is
  not on its own evidence about the code. Under `DTWC_FP_MODEL=fast`, `soft_dtw_gradient` can
  differ from the Eigen build in the last bit because the compiler re-associates its final sum
  differently; under `strict` every output is bit-identical.
- **Changed (internal headers):** `dtwc::detail::available_ram_bytes()` is declared in a new
  `dtwc/system_memory.hpp` instead of `DataLoader.hpp`. `DataLoader.hpp` includes the new header and
  re-exports the name, so no consumer changes. This removes the last include edge from a foundation
  file to a higher layer that did not point at `Problem.hpp`.
- **Added (tooling, evidence):** two report-only codegen tools. `scripts/check_ipo_inlining.py`
  disassembles the built CLI and counts surviving out-of-line calls to `Problem::dist_by_ind` inside
  the FastPAM SWAP kernels; it is a manual tool, not a test (as a test its pass pattern accepted any
  count). `scripts/codegen_report.py`, with `scripts/codegen_probe.cpp`, replays a real compile
  command with clang's vectorisation remarks enabled and reports which hot-path loops vectorise.
  Findings: ThinLTO does not inline `dist_by_ind` (18 of 20 call sites
  survive), and none of the six DTW kernel loops vectorise, because the recurrence carries a genuine
  dependency and the early-abandon variants exit early while writing memory.
- **Changed (build options, deprecation):** the thirteen maintainer CMake options are now spelled
  `DTWC_*` like every other cache variable in the project — `DTWC_ENABLE_SANITIZER_ADDRESS`,
  `DTWC_WARNINGS_AS_ERRORS`, `DTWC_ENABLE_PCH` and so on. The old lowercase `dtwc_*` names are still
  accepted for one release: the value is honoured and a warning names the replacement. Passing a
  legacy name whose value matches the new default is accepted silently, so an existing build
  directory does not report options its owner never chose.
- **Fixed (gates):** `scripts/check_supply_chain_pins.py` was failing and unnoticed, because the
  runbook's gate command lists three check scripts and there are four. Its tracked-CMake-manifest
  ratchet still expected 30 files against 33, stale since the commit that moved test registration
  into `dtwc_add_test` — the same commit that had also left the documentation-contract gate stale.
  The constant is reconciled and now names the four added and one removed manifest. Separately, the
  scanner correctly refused to audit a `CPMAddPackage` argument containing a variable expansion;
  that option is now set outside the call so the pins remain statically readable.
- **Fixed (tests, conformance):** the cross-language conformance test could certify itself. It
  regenerated its pinned reference whenever the reference file was absent — not only when asked —
  and then compared the freshly written values against themselves, so every assertion passed
  trivially and the only signal was a `WARN`, which does not fail a test. Regeneration is now
  explicit (`DTWC_CONFORMANCE_REGEN=1`) and a missing reference fails, naming how to restore or
  deliberately re-record it.
- **Added (benchmarks):** `DTWC_BENCHMARK_PMU` builds the benchmarks against libpfm4 through Google
  Benchmark's `BENCHMARK_ENABLE_LIBPFM`, so hardware counters can be read instead of wall-clock.
  libpfm4 is MIT, benchmark-only and never redistributed. It needs `perf_event_open`, so it is
  bare-metal Linux only, and every configuration that cannot deliver counters — a non-Linux host,
  `DTWC_BUILD_BENCHMARK=OFF`, or a `benchmark::benchmark` supplied by an enclosing project — is a
  `FATAL_ERROR` naming the reason rather than a quiet downgrade.
- **Fixed (benchmarks):** `scripts/run_bench.sh` no longer returns a counters record with no
  counters in it. A binary built without libpfm4 accepts `--benchmark_perf_counters`, prints one
  line to stderr, and then writes a complete, ordinary-looking JSON and exits 0 — the saved file
  carries no trace that the request was dropped. Google Benchmark's own guard does not catch this:
  in v1.9.5 the `BM_CHECK` at `benchmark_runner.cc:323` tests the inverse of its message, and it
  only runs for benchmarks that set an aggregation report mode, which none of ours do. The driver
  now verifies each requested counter is present in the JSON, and on failure renames the file to
  `*.no-counters.json` and exits 65, so a wall-clock record cannot be filed as a PMU one.
- **Fixed (build, llfio):** a failed patch of quickcpplib's `QuickCppLibUtils.cmake` no longer
  passes silently. CMake's `string(REPLACE)` succeeds and changes nothing when its pattern is
  absent, so an upstream text change downgraded to one line of build noise and then a confusing
  failure deep inside llfio's nested superbuild. Configure now fails outright, checks both patch
  sites on every run rather than only the one that clones, names the pinned quickcpplib commit and
  points at `-DDTWC_ENABLE_LLFIO=OFF` as the way past it.
- **Changed (build, dependency pins):** the two llfio-related pins marked "needs maintainer
  blessing" are resolved and documented in place. The llfio pin stays where it is: it is five
  commits past release tag `20260506`, and those commits include the upstream fix guarding a
  `char8_t`→`wchar_t` locale codecvt for libc++, so moving back to the tag would regress macOS.
  quickcpplib publishes no tags at all, so pinning a reviewed commit is the only mechanism
  available rather than a placeholder for a future version number.
- **Fixed (gates):** `scripts/check_docs_contract.py` was failing on `design-2.0` for a reason
  unrelated to the documentation it checks. The D2 and D3 lower-bound derivation gates read
  `tests/CMakeLists.txt` for hand-written `if(TARGET …)` CTest policy blocks, which the move to
  `dtwc_add_test(...)` had replaced; because the script aborts on its first failure, the D3 gate was
  invisible behind the D2 one. Both now read the `dtwc_add_test` registration for their own floors,
  environment, serial and timeout settings, and reject a `MAY_SKIP` registration. The skip hardening
  those blocks used to assert inline moved into `cmake/DtwcTest.cmake`, which no gate referenced at
  all — it is now checked directly, so "this test proved it ran" is pinned where it is implemented.
- **Fixed (packaging, macOS and Linux):** the native CLI release archive could not start on any
  machine but the one that built it. `cmake --install` never set an `INSTALL_RPATH`, so the packed
  `dtwc_cl` carried **no `LC_RPATH` at all** and `dyld` aborted on `@rpath/libhighs.1.dylib`
  (HiGHS builds shared by default). The rpath is now set unconditionally — `@loader_path/../lib` on
  macOS, `$ORIGIN/../lib` on Linux, which previously had no rule at all. On macOS the LLVM OpenMP
  runtime is also bundled under `lib/` and the CLI's load command repointed at `@rpath`, so the
  archive no longer depends on a Homebrew prefix existing. The block that was supposed to do this
  had been unreachable: it guarded on `OpenMP_omp_LIBRARY`, which is only ever written by the
  Windows-Clang branch, while AppleClang gives `OpenMP_libomp_LIBRARY`.
- **Fixed (CI):** the release-archive and Python-wheel workflows install keg-only `libomp` on macOS
  but never told CMake where it is, unlike `macos-unit.yml`, which `brew link --force`s it. The
  release workflow's exact configure line fails locally for that reason — `find_package(OpenMP)` is
  a configure `FATAL_ERROR` since the OpenMP requirement was made hard. Both jobs now export
  `OpenMP_ROOT`.
- **Changed (release gate):** `scripts/smoke_release_archive.py` now also asserts that the archive
  is self-contained (no dependency resolving to an absolute path outside it) and carries the
  notices redistribution requires. Running the CLI proves nothing about portability on the machine
  that linked it, which is why the previous gate passed over a broken archive.
- **Fixed (licences):** third-party notices now match what is actually shipped. nanoarrow's
  `LICENSE.txt` and `NOTICE.txt` are vendored beside the amalgamation, installed to
  `share/doc/dtwc/nanoarrow/` and included in the wheel's `license-files`, satisfying Apache-2.0
  §4(a) and §4(d) for a dependency that is compiled into every artefact. `THIRD_PARTY_LICENSES.md`
  is now a per-artefact inventory covering Eigen (with the MPL-2.0 §3.2 source offer), nanoarrow,
  HiGHS, CLI11, fkYAML, nanobind, the bundled macOS libomp and optional llfio, instead of naming
  HiGHS alone.
- **Changed (tooling):** `scripts/machine_facts.py` now prints the compiler flags CMake recorded
  in its markdown record, not only in `--json`, so a benchmark record says what was compiled.
  The row is labelled as the cache-level flags alone: `-march=native` and the floating-point
  settings are attached to the `project_options`/`dtwc++` targets and never reach
  `CMakeCache.txt`, so it must be read together with the `DTWC_*` option list.
- **Added (tooling):** `scripts/machine_facts.py` describes the machine and the build a measurement
  ran on — CPU model, physical/logical cores, memory, GPU (CUDA name and compute capability, or the
  Metal device), OS, compiler, generator and every resolved `DTWC_*` option read from the CMake
  cache — as a markdown table or JSON, on macOS, Linux and Windows. Stdlib only. Optional
  `--with-dtwcpp` adds the `test.parallelisation()` and `test.gpu()` engagement probes. Three
  skills use it: `dtwc-run-benchmarks` (benchmark with its hardware recorded), `dtwc-verify` (build,
  serial ctest and the gate scripts) and `dtwcpp` (a user-facing entry point to the library that
  routes to the `.claude/commands/` procedures).
- **Change (repository, records):** the development plan moved into `.claude/`.
  The root `PLAN.md` is archived as
  `.claude/PLAN-archive-2026-09-21-research-release-campaign.md`; the live
  documents are `.claude/{CHARTER,MAP,design,PLAN,DECISIONS}.md`. 105 superseded
  session records, twelve one-off evidence scripts of closed findings, generated
  benchmark plots, unused figures and a boilerplate `.cmake-format.yaml` were
  removed (all recoverable at `9c08074`). `scripts/check_docs_contract.py` and
  `scripts/check_record_hygiene.py` read the archived plan at its new path; their
  assertions are unchanged. New `scripts/repo_map.py` reports the include graph
  against the target layer model (18 upward includes today, 17 of them into
  `Problem.hpp`). No library code changed.
- **Change (tests, build):** every CTest entry is now gated by an execution
  floor or an explicit skip. `cmake/Coverage.cmake` (whose
  `add_executable_with_coverage_and_test` registered every test with
  `SKIP_RETURN_CODE 4` and no pass/fail regex, so a test that skipped or
  asserted nothing scored green) is replaced by `cmake/DtwcTest.cmake`:
  `dtwc_add_test` fails a test that prints a skip line unless it opts into
  `MAY_SKIP`, and requires Catch2's summary to report at least one assertion
  in at least one test case (per-test floors were tried and removed: assertion
  counts differ across standard libraries).
  `test_io_readers` is no longer registered at all where Arrow is absent, and
  `tests/unit/mip/test_pdlp_lp.cpp` now `SKIP`s instead of `WARN`ing when
  HiGHS is missing, so its three cases can no longer report "3 passed" with
  no assertions. `DTWC_ENABLE_COVERAGE` is unchanged.
- **Fix (tests, Windows/MSVC):** the Windows unit job failed three tests, each
  for its own reason. (1) The barycenter k-means fingerprint compared centres
  and inertia bitwise; MSVC 19.50 lands 1 ULP from GCC/Clang on one coordinate,
  so the comparison is now 1e-12 relative — RNG or schedule drift, which is
  what the fingerprint guards, moves those numbers by O(1). (2) The F22
  deprecation probes reported `legacy=0/33` on `cl`: MSVC diagnoses a
  deprecated entity only where it is *used*, never on `&Class::member`, and its
  `C4996` text does not contain the word "deprecated". The fixture now calls
  every retained 1.x entity (one use per line — `cl` reports only the first
  diagnostic on a line) alongside the pointer-to-member signature pins, and the
  probe accounting accepts the `C4996` spelling. The probes also define
  `_SILENCE_ALL_CXX20_DEPRECATION_WARNINGS`, because llfio instantiates
  `std::codecvt<char16_t, char8_t>` (deprecated by LWG-3767) and `/we4996`
  turned that third-party deprecation into a hard error before any dtwc
  diagnostic was emitted. (3) `.gitattributes` pins the conformance fixtures
  and every tracked `.csv` to LF: Git for Windows' system default
  `core.autocrlf=true` — what the `windows-latest` runner uses — checked them
  out as CRLF, and `test_cli_resume_state` pins their SHA-256, so it failed
  with `F17 conformance input hash drift`. The tracked blobs are already LF, so
  no content changes.
- **Fix (build, GCC + LTO):** `run_openmp`'s exception-capture region is an
  unnamed `#pragma omp critical` again. The named form made GCC emit a COMMON
  `.gomp_critical_user_dtwc_run_openmp_exception` symbol into every TU that
  includes `parallelisation.hpp`; with IPO/LTO on and a static `libdtwc++.a`,
  ld.bfd's search for a real definition of that COMMON symbol pulled unused
  archive members into the link and gave their COMDAT symbols a second
  `PREVAILING_DEF_IRONLY` resolution, so `lto1` aborted with
  `multiple prevailing defs for 'resize'` (binutils PR ld/32083, GCC PR
  lto/116361). Seen on GCC 13.3 / binutils 2.42 in the Release Linux MPI CI
  job, linking `unit_test_run_thread_scope`. Behaviour is unchanged: the
  region is on the exception path only and still rethrows the lowest failing
  index.
- **Fix (silent wrong numbers):** the distance-matrix CSV reader parsed with
  `std::stod`, which honours the C locale, so under a comma-decimal locale
  `1.5` read as `1`. It now uses `std::from_chars` like the series loader
  (locale-independent), with a de-DE regression test.
- **Fix (YAML):** a YAML `null`/`~` value is an error naming the key
  (`omit the key to use the default`); it used to reach CLI11 as an empty
  value that set `band` to 0 or switched flags on.
- **Changed:** Tier-1 `cluster()` on a GPU device pins `StoragePolicy::Heap`
  before loading, so large datasets never land on the mmap store that CUDA/Metal
  reject; `Problem::set_ram_limit`/`ram_limit` are honoured by `set_data` (the
  limit was hardcoded to 0 on that path). macOS free RAM is now
  `host_statistics64` free+inactive (was total RAM). `<windows.h>` no longer
  leaks through the public headers (`available_ram_bytes` moved to
  `system_memory.cpp`).
- **Fix (Windows):** `StoragePolicy::Auto` never spilled series to the mmap
  store on Windows because the free-RAM query was unimplemented and returned
  0, which the threshold treated as unlimited. It now uses
  `GlobalMemoryStatusEx`; an unknown free-RAM value is documented as
  heap-only. The decision is the pure `choose_storage(estimated, available,
  limit)`. Consequence: large datasets on Windows now behave as on Linux,
  including the explicit `DeviceError` when a mapped store meets a CUDA/Metal
  strategy.
- **Fix (C++ Tier-1):** `Problem::cluster()` with `Method::Kmedoids` wrote
  per-repetition medoid and best-repetition CSVs into the CWD-relative
  `./results/` on every call and threw when the folder was missing, and it
  printed Lloyd progress unconditionally. Clustering routes no longer perform
  file I/O and print only when `verbose()` is set; `cluster_and_process()`
  still writes exactly what it wrote before, and every writer now creates its
  output directory. Benders progress lines are gated the same way.
- **Added:** `Result::distance_matrix()` (dense row-major N×N, filling on
  demand like `score`) in C++, mirrored by Python and used by MATLAB
  `Result.plot`.
- **Changed:** `MIPSettings::lr_max_nodes` and `LagrangianParams::max_nodes`
  are `std::int64_t` (were `long`, 32-bit on Windows).
- **Fix:** series and dataset names derived from file paths are UTF-8 on every
  platform (were the native code page on Windows, which Python could not
  decode and which the CLI wrote as mojibake).
- **Changed (build):** `dtwc::load()` rejects the pre-2.0
  `load(source, skip_cols, delimiter)` argument shape at compile time instead
  of binding the delimiter character to `skip_rows`.
- **Changed (MATLAB Tier-1):** `dtwc.cluster` delegates to C++ `dtwc::cluster`
  through one MEX call instead of re-implementing routing in `.m`. `kmedoids`
  now runs Lloyd k-medoids rather than FastPAM; `onebatch`, `lrcore` and
  `tadpole` are added; `auto` follows the C++ rule; `k <= N` is enforced with
  the C++ message; `max_iter` reaches CLARA; `skip_cols`/`skip_rows` apply to
  in-memory sources; `device=` is a per-call override that no longer mutates
  the process device. `Result.score`/`save` are the C++ members, so the output
  CSVs carry the dataset's series names and match the CLI byte for byte.
- **Changed (MATLAB):** `dtwc.DTWClustering` executes `Metric` and `Device`
  instead of only storing them (closes F18/F40); `save_checkpoint`/`load_checkpoint`
  take the optional `metric` token C++ and Python already had, so a SquaredL2
  matrix is no longer stamped and reloaded as L1. `dtwc.cluster`/`dtwc.load`
  accept a cell array of numeric vectors as a ragged in-memory source;
  `DTWClustering` with `Device='gpu:N'` forwards the ordinal to the
  Problem's device.
- **Changed (Python):** `dtwcpp.device()` returns the canonical name from
  `dtwc::device` and `Env` is the only device store; path sources are parsed by
  the C++ `DataLoader` (non-numeric id columns and ragged rows now load; no
  numpy text parser remains); in-memory `skip_cols` erases columns as C++ does;
  `cluster()` enforces `k <= N` and rejects an empty dataset with the C++
  messages; `Result.score`/`save` work after matrix-free methods by filling the
  matrix lazily as C++ does; `Problem.checkpoint` is bound (in-place mutation
  works); `dtwcpp.UndefinedScore` is bound (subclass of `InvalidInput`) and
  `Result.save` warns and skips the silhouettes file on it as C++ does; ragged
  in-memory sources load; series names come from the C++ loader, so all four
  `Result.save` CSVs are byte-identical to the CLI's (UTF-8 names included).
  Tier-1 `Dataset` materialises straight into a bound `dtwc::Data` with the
  GIL released (2000×500: 0.40 s and 32 MB of Python objects → 0.19 s and
  none); `Result.distance_matrix` fills on demand; NaN/inf in `save` follow
  the C++ writer (empty field / `InvalidInput`). A single-column CSV now
  yields N series of length 1, as the CLI does.
- **Added:** automatic mid-fill checkpointing. `Problem::checkpoint`
  (`CheckpointOptions{directory, save_interval, enabled}`) is now consumed by
  `fill_distance_matrix()`: `save_interval` is the number of completed matrix
  rows between saves; the brute-force fill runs disjoint row blocks and
  publishes one generation after each block, including the last, so a crash
  loses at most one block and a completed fill leaves a complete checkpoint.
  `enabled` with `save_interval < 1`, an empty directory, or mmap storage is an
  `InvalidInput` before any work; `Pruned` + `enabled` runs the exact
  brute-force path (verbose note); CUDA/Metal save once after the fill. Each
  save writes the full N×N matrix (O(N²) per save), documented with sizing
  guidance. CLI: `--checkpoint-interval <rows>` (requires `--checkpoint`).
  Exposed as `Problem.checkpoint` in Python (in-place mutation works) and
  MATLAB (`set_checkpoint` / `get_checkpoint`).
- **Fix:** resuming from a checkpoint through `fill_distance_matrix()` never
  worked: the brute-force fill called `resize(N)` unconditionally, and
  `DenseDistanceMatrix::resize` NaN-fills every slot, so a restored matrix was
  discarded and recomputed. The resize is now conditional on a size change;
  a real resume on a 25-series set went from 15 s to 0.15 s.
- **Fix:** a checkpoint directory now holds exactly one generation; the
  previous one is removed only after `CURRENT` points at the new one. Saves
  used to accumulate one full N×N CSV each.
- **Breaking (CLI/build):** the `--yaml-config` option, the
  `DTWC_ENABLE_YAML` build option and the yaml-cpp dependency are removed.
  `--config` is the only configuration mechanism and now accepts **TOML or
  YAML**, detected from content and parsed into CLI11's own config items
  (`dtwc/cli/config_file.hpp`), so both formats share keys, validators and
  deprecation warnings, and a command-line value always beats the file (the
  old YAML loader silently overrode explicit flags). YAML parsing uses the
  optional header-only fkYAML (MIT; `DTWC_ENABLE_YAML`, default ON; OFF refuses
  YAML with `built without YAML support; use TOML`). A config key matching no
  CLI option is now an error in both formats (was silently ignored). Example
  `examples/cpp/config.yaml`. The required-input error reads
  `Error: --input is required via CLI or config file (TOML or YAML)`.
- **Fix (MATLAB):** a MEX built with `-DDTWC_ENABLE_HIGHS=ON` no longer crashes
  MATLAB R2024b (`0xc0000005` on a HiGHS worker thread) on `Method::MIP`
  solves. A MEX runs against MATLAB's private `msvcp140.dll`; STL 14.40+ emits
  a constexpr `std::mutex` that older runtimes fault on. MSVC-ABI MATLAB builds
  now compile every TU, including fetched HiGHS and llfio, with
  `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`. Verified: the same binary crashed
  R2024b and passed R2025b before the fix, passes both after.
- **Added:** `tests/matlab` runs under CTest as `matlab_suite` when
  `DTWC_BUILD_MATLAB` is ON and MATLAB is found; a MATLAB "Incomplete"
  (filtered-by-assumption) fails the gate instead of reading as a pass, and the
  passed count is floored.
- **Added (Tier-1 parity):** `load()` gains `skip_rows` after `skip_cols` in
  C++, Python and MATLAB: leading lines for a path source, leading series for an
  in-memory source; negative values are rejected. Python `device="hpc"` rejects
  a non-zero value because the SLURM transport cannot carry it.
- **Added:** `MIPSettings.lr_max_nodes` is readable and writable from Python
  (shown in `repr`) and MATLAB (`set_mip_settings` / `get_mip_settings`).
- **Changed (tests):** the shared deterministic generators in
  `tests/support/deterministic_series.hpp` produce byte-identical doubles on
  every conforming C++20 implementation: an exact 53-bit `genrand_res53`
  conversion replaces `std::uniform_real_distribution`, whose engine-to-real
  mapping is implementation-defined. F15 registers one fingerprint per
  schedule instead of four per-toolchain profiles. Nominal ranges are
  `[-1, 1)` and `[-10, 10)`.
- Records: stale agent-facing guidance corrected (pybind11 wrapper skill marked
  historical, dead `develop/TODO.md` links, closed TODO rows B03/D03,
  `medoid_utils.hpp` tombstone); `.mailmap` maps Kasper Westman's PR #32
  commits to his GitHub address.
- **Breaking:** Benders decomposition (`Method::MIP` with
  `mip_settings.benders = "on"/"auto"`) now throws `dtwc::SolverError` when its
  cut loop reaches `max_benders_iter`, or its master stops non-optimally,
  without closing the bound gap. It used to print "Benders decomposition
  complete" and publish the PAM-quality incumbent through the EXACT entry
  point. The message names the knobs that exist
  (`mip_settings.max_benders_iter`, `mip_settings.mip_gap`,
  `mip_settings.benders = "off"`, or `Method::Kmedoids`). The shipped default
  `max_benders_iter = 200` is unchanged and was verified sufficient on the
  default route: a 250-point, 5-group separable instance on stock settings
  (`benders = "auto"`, so Benders engages at N > 200) converges at Benders
  iteration 4 (5 master solves), pinned by
  `tests/unit/mip/test_mip_backend_guards.cpp`.
- **Breaking:** the Benders lower bound is HiGHS's `mip_dual_bound` for the
  master, not the master's incumbent objective. The old bound could exceed the
  true master optimum and declare convergence while a strictly better medoid
  set existed.
- Benders publishes through `mip::ExactClusteringTransaction`, so a failed
  solve leaves `centroids_ind` / `clusters_ind` untouched.
- **Fix:** the Benders tolerances are now separated by what they compare. The
  two COST comparisons (the UB/LB convergence test and the per-point cut-skip
  test) use `mip::benders_abs_eps(max_distance)` =
  `1e-6 * max(max_distance()/2, 1)`, the same conditioning factor the compact
  HiGHS backend divides its objective by, so both backends accept the same
  relative violation; the compact backend has no `1e-6` tolerance of its own.
  That scaled tolerance was also filtering the coefficients of each
  disaggregated Benders cut, which it must not: every coefficient
  `c_i = max(0, d_nearest - d_ji)` is non-negative, so dropping a positive one
  shrinks the cut's left-hand side and makes it STRICTER than the valid Benders
  cut — the master's dual bound was inflated by up to
  `N * 1e-6 * max_distance/2` and could cut off the true optimum while the loop
  reported a proved optimum. The cut filter is now the unscaled
  `mip::benders_cut_coefficient_threshold(d_nearest)` =
  `1e-12 * max(1, d_nearest)`, relative to the cut's own right-hand side.
- **Breaking:** `Method::LRCore` throws `dtwc::SolverError` when the
  branch-and-bound node cap stops the tree before optimality is proven, and
  publishes through the same validated transaction as the HiGHS/Gurobi backends
  (exactly k unique medoids, in-range labels, every medoid in its own cluster).
  It used to write an uncertified incumbent straight into `Problem` with no
  validation.
- New `MIPSettings::lr_max_nodes` (default 2,000,000, i.e. the previous
  hard-wired `mip::LagrangianParams::max_nodes`) is forwarded to
  `mip::lagrangian_root_exact` by `Method::LRCore`, and the SolverError above
  names it. The message previously told callers to "raise the node cap" through
  a knob that did not exist. C++ only for now: the Python and MATLAB
  `MIPSettings` bindings still expose the pre-existing fields.
- `mip::lagrangian_root(Problem&)` (bound-only) no longer overwrites the
  caller's `centroids_ind` / `clusters_ind` with its internal k-medoids seed,
  and `mip::lagrangian_root_exact` throws instead of reading out of bounds when
  reduced-cost fixing proves more than k facilities open. The Lagrangian primal
  repair (`pmedian_local_search`) now sorts its medoids BEFORE the final
  assignment sweep, so the cost, the point-index `labels` AND the internal
  `cluster_of` positions all describe the sorted medoid set it returns;
  previously the sweep ran first and the sort then left `cluster_of` holding
  pre-sort positions, contradicting the in-file invariant. It also fixes the
  older case where exhausting `max_sweeps` left cost/labels describing the
  previous sweep's medoids.
- The exact MIP backends validate the FastPAM warm start before indexing the
  solver's start vector with it (Benders additionally requires its own
  nested-Lloyd warm start to hold exactly `k` medoids, instead of failing later
  inside `publish`), and the `N*N` model-dimension guard is now backend-neutral
  (`mip::require_index_range`, `dtwc/mip/index_guard.hpp`, no solver header)
  and applied on all three exact routes. The compact HiGHS backend bounds by
  `min(HighsInt, int)` because its triplets are plain `int`, so a `HIGHSINT64`
  build no longer lets `N > 46340` through to truncate; Gurobi's
  `addVars(N*N)` (an `int` count) is guarded too, where it previously had no
  check at all.
- HiGHS options: an option this repo relies on for solver IDENTITY still throws
  when HiGHS rejects it (a mistyped `PdlpParams::variant` used to run dual
  simplex while the result was still reported as a PDLP bound), but
  `kkt_tolerance` — pure tuning, and absent from older HiGHS builds — is now
  best-effort with a stderr note, so a version skew no longer means no PDLP
  bound at all.
- `MIPSettings` is validated where it is consumed (`Problem::cluster_by_mip`,
  `LR_core_clustering`): `mip_gap >= 0`, `max_benders_iter >= 1`,
  `lr_max_nodes >= 1` and an unrecognised `benders` selector all raise
  `dtwc::InvalidInput` instead of being treated as off. A negative `mip_gap`
  used to reach HiGHS as an out-of-domain `mip_rel_gap` and surface as a
  solver-worded error. The CLI's `--benders` validates its value the same way
  (`auto|on|off`, plus `true/false`, `yes/no`, `1/0`); `--benders ON` used to
  mean *off*.
- `Benders` and `LR-core` no longer write `core::ClusteringResult::total_cost`
  before publishing: `ExactClusteringTransaction::publish` swaps only medoids
  and labels (as it does for the HiGHS and Gurobi backends), so those stores
  were dead. Recompute with `Problem::find_total_cost()`.
- MATLAB: `dtwc.cluster(..., 'method','mip')` passes `k` to the solver. It
  ignored `k` entirely, so the MIP route returned one cluster for every
  request. **[BLOCKED-ENV] Not executable in this environment** (every recorded
  MEX build here is HiGHS-OFF and a HiGHS-enabled MEX crashes MATLAB on this
  machine — see `.claude/LESSONS.md`), so `tests/matlab/test_cluster_mip.m`
  takes its `assumeTrue` skip and this fix has never been run.
- MATLAB: `dtwc_mex` validates the `merges` column count before reading a
  dendrogram, and rejects NaN/Inf/fractional entries in label and dendrogram
  index vectors instead of casting them (undefined behaviour). `INT_MIN` is
  exactly representable as a double and passed that check, so the 1-based `- 1`
  shift was itself signed overflow; the shift now goes through a helper that
  rejects it, on both the double and the `int32` element paths. Verified by
  `tests/matlab/test_mex_input_validation.m::test_int_min_label_rejected`
  against a rebuilt HiGHS-OFF MEX under MATLAB R2024b (42 passed / 0 failed /
  0 skipped; it fails on the pre-fix MEX). These `.m` files are not registered
  in CTest — they run only by hand under a MATLAB installation.
- The PDLP LP-relaxation arbiter is reachable from the bindings, not only from
  C++: new `dtwc.pdlp_lp_bound` / `dtwc.pdlp_gpu_available` in MATLAB, and
  `pdlp_lp_bound`, `PdlpParams`, `PdlpResult`, `pdlp_gpu_available()` and
  `PDLP_GPU_AVAILABLE` in `dtwcpp._dtwcpp_core`, mirroring the C++ names,
  arguments and result fields.
- Build: the MATLAB MEX exports `mexFunction` explicitly on Windows for
  non-MSVC compilers. CMake 4.2's `FindMatlab` exports it only under
  `if(MSVC)`, so a Clang MEX linked with no entry point and MATLAB refused it
  ("Gateway function is missing").
- **Breaking:** `scores::silhouette` now throws `dtwc::InvalidInput` when the
  labels realise fewer than 2 non-empty clusters. It used to return ~ +1.0 for
  every point (a "perfect" score) because b(i) was left at `DBL_MAX`; sklearn
  raises for the same input, and `davies_bouldin`/`dunn` already did.
- **Breaking:** `silhouette`, `davies_bouldin`, `dunn` and `calinski_harabasz`
  are computed over the REALISED label set rather than the declared
  `n_clusters`. Empty declared clusters are skipped and no longer contribute to
  the 1/k normaliser; `dunn` with `n_clusters = 3` but only label 0 in use now
  throws instead of returning ~1.8e308 as a finite number. All four also reject
  a `clusters_ind` whose size or label range disagrees with the Problem.
- **Breaking:** the score guards use the project taxonomy throughout.
  `silhouette`, `davies_bouldin`, `dunn` and `calinski_harabasz` throw
  `dtwc::InvalidInput` (which derives from `std::runtime_error`) for a
  mismatched `clusters_ind` and for fewer than 2 clusters, where they
  previously threw a mixture of `std::runtime_error` and
  `std::invalid_argument` from the same function; so do the "cluster first"
  guards of `davies_bouldin`, `dunn`, `inertia` and `calinski_harabasz` and the
  Calinski-Harabasz "at least 2 clusters" / "more points than clusters" guards.
  `catch (const std::invalid_argument &)` no longer matches these;
  `catch (const std::exception &)` and `catch (const std::runtime_error &)` do.
- New `dtwc::UndefinedScore : dtwc::InvalidInput`, thrown only when a score is
  mathematically undefined for the labelling (fewer than two non-empty
  clusters), and the output paths no longer abort on one:
  `Problem::write_silhouettes()` (hence `Problem::cluster_and_process()`) and
  `Result::save()` catch exactly that, warn on stderr and skip the silhouettes
  file when fewer than 2 clusters are realised — `k = 1`, or a `k >= 2` request
  that collapses on duplicate series (`Result::save()` and `dtwc_cl` write no
  silhouettes file for `k = 1` and say nothing). They used to write labels, medoids and
  the distance matrix and only then throw, leaving a partially populated output
  directory; the CLI already behaved this way. A corrupt `clusters_ind`, an
  out-of-range label or bad data raise the plain `InvalidInput` of the other
  guards and propagate as before instead of being swallowed as a warning.
  `Result::score("silhouette")` and `scores::silhouette()` still throw: asking
  for the number is a different contract from asking for the files.
- `scores::silhouette` returns 0 instead of NaN when a(i) = b(i) = 0 (all
  distances in and around the point's cluster are zero) — Rousseeuw's
  convention.
- **Breaking:** `scores::davies_bouldin` treats a zero medoid distance M_ij as
  R_ij = +infinity (the worst pair) instead of skipping the pair, so two
  clusters with coincident medoids and real internal spread no longer report
  DBI = 0.0. M_ij = 0 with zero scatter on both sides stays 0 (Davies & Bouldin
  axiom 3).
- `algorithms::tadpole` no longer reads float64 storage on a Float32 Problem.
  LB/UB pruning is disabled there (as `dtw_barycenter` already did) because it
  would not be admissible: the bound path reads the float64 series while the
  exact side routes through `Problem::dist_by_ind`, which branches on
  `is_f32()`, so the two sides of the bound would come from different data.
  (`Data::series()` now throws on Float32, so the unguarded read would be a
  loud error rather than the historical out-of-range read.) `prune = true` on
  Float32 therefore costs the same as `prune = false`; the new
  `TADPoleStats::pruning_enabled` reports it so the fallback is not silent.
- `algorithms::cut_dendrogram` validates the supplied `Dendrogram`
  (`n_points == prob.size()`, exactly `n_points - 1` merge steps, every cluster
  id in range, and a merge list that really reduces N points to k components)
  instead of trusting it. A hand-built `Dendrogram` — reachable from Python —
  previously read `merges` and the union-find parent array out of bounds.
- **Breaking:** `algorithms::one_batch_pam`'s final assignment rejects a
  non-finite distance with `dtwc::InvalidInput` (it used to leave every
  affected point silently in cluster 0 with a non-finite `total_cost`), and its
  `total_cost` is now accumulated in point order by `OrderedMedoidObjective`
  instead of `std::accumulate`, which the build's `-fassociative-math` was free
  to reassociate and vectorise; expect last-ulp differences from 2.0.x. An
  objective that overflows to infinity while every individual point cost is
  finite is now rejected rather than returned as `inf`.
- **Breaking:** every CUDA entry point (`compute_distance_matrix_cuda`,
  `compute_lb_keogh_cuda`, `compute_dtw_one_vs_all`, `compute_dtw_k_vs_all`)
  now throws `dtwc::DeviceError` when no CUDA device is present instead of
  returning an all-zero NxN matrix with `kernel_used == "none"`. Metal's
  entry points do the same for an uninitialised backend.
- **Breaking:** the same CUDA entry points reject a pair count above INT_MAX
  with `dtwc::InvalidInput` before allocating anything. The LB_Keogh pre-pass
  used to truncate `N*(N-1)/2` to `int` first (N >= 65537 gave an illegal
  memory access on device), and `compute_lb_keogh_cuda` had no guard at all.
- CUDA honours `max_length_hint` for kernel selection, as Metal already did;
  it was declared, advertised and silently ignored.
- CUDA raises a typed `dtwc::DeviceError` naming the shared-memory shortfall
  instead of a bare CUDA "invalid argument" when a wavefront launch exceeds the
  device's opt-in per-block shared memory.
- `query_gpu_config` no longer takes a process-global mutex on cache hits (it
  ran once per kernel launch, per host thread); the two `static bool logged`
  warning latches in the CUDA launchers are atomic.
- Python: the GIL policy is now *consistent release, no per-object lock*.
  `Problem.fill_distance_matrix`, `distance_matrix`, `cluster`,
  `assign_clusters`, `calculate_medoids`, `write_distance_matrix`,
  `print_distance_matrix`, `DenseDistanceMatrix.to_numpy`, `save_checkpoint`,
  `dist_by_ind`, `find_total_cost`, `write_clusters`, `write_silhouettes`,
  `read_distance_matrix` and `load_checkpoint` all release it, matching their
  `*_binary_*` siblings. A `Problem` instance must not be used concurrently
  from multiple Python threads (the same contract as C++): the GIL is released
  during C++ work so that other threads can run, but two threads calling
  methods on the same `Problem` race on its lazily-filled distance cache. Use
  one `Problem` per thread, or call `fill_distance_matrix()` first and only
  read afterwards. (Holding the GIL in some bindings gave no mutual exclusion
  against the many that released it, so it documented a safety property the
  module did not have.) Relatedly, `Problem::mmap_cache_data_validated_`,
  written from the `const` `validate_mmap_cache_identity()`, is a relaxed
  `std::atomic<bool>`, so two threads holding one `const Problem &` no longer
  race on it by the memory model.
- Python: numpy results are handed over through an owning buffer instead of a
  raw `new double[n*n]`, so a throw between the allocation and the capsule no
  longer leaks the whole N^2 matrix; a GPU backend returning the wrong number
  of distances is now a `DeviceError` rather than a zero-padded matrix.
- Dense checkpoints fingerprint the pointwise metric, so a matrix computed with
  `--metric squared_euclidean` is no longer accepted by a later `--metric l1`
  run. `save_checkpoint` / `load_checkpoint` /
  `Problem::distance_checkpoint_identity` take an optional `core::MetricType`
  (defaulting to `L1`, so existing calls compile unchanged), and the Python
  `save_checkpoint`/`load_checkpoint` take the matching optional `metric`
  argument (`dtwcpp.MetricType`, default `L1`) mirroring the CLI's `--metric`.
- **Breaking:** `Problem::read_distance_matrix` now throws when the CSV cannot
  be opened or parsed instead of printing "Distance matrix could not be read!"
  and returning normally. `dtwc_cl --dist-matrix <bad path>` no longer prints
  "Loaded distance matrix from ..." straight after the failure message; it
  warns on stderr and continues without a precomputed matrix.
- **Breaking:** `dtwc_cl` rejects an unknown `--method`, `--solver` or
  `--linkage` instead of running with an empty result / the default solver /
  Average linkage. Only a config file could reach these (CLI11 already checked
  the command line); an unknown method used to write a checkpoint and label
  files for a default-constructed result and still exit 0. YAML `method:` also
  accepts the same `obp` and `lr` aliases as the command line; its hand-written
  normalisation only mapped `hclust`.
- **Breaking:** `dtwc_cl` rejects `.parquet`/`.pq` and
  `.arrow`/`.ipc`/`.feather` input on a build without Arrow/Parquet, naming the
  missing build option. Those inputs previously fell through to the CSV reader
  and failed with a numeric-parse error (or, worse, parsed).
- **Breaking:** `dtwc_cl` rejects `--column` on a non-Parquet input and
  `--skip-rows`/`--skip-cols` on a non-text input. Both were accepted and
  silently ignored off their own format. (HPC job scripts that pass
  `--skip-cols 1` alongside a `.parquet` or `.dtws` input must drop the flag.)
- Parquet readers reject null values instead of reading uninitialised buffer
  bytes as data: a null list cell, a null list element, and a null in a scalar
  column are now errors ("drop or fill nulls before clustering"), matching the
  Arrow C-Data ingest path.
- Folder loads now iterate a sorted, regular-files-only directory listing, so
  series order — and every name, label, medoid and distance-matrix index — is
  the same on every machine. `DataLoader::count()` counts the same entries the
  loader reads.
- Made mapped-series temp-path collisions improbable (they are not impossible:
  the file is still not created with exclusive `O_EXCL` semantics). Its
  sequence counter was a non-atomic static (concurrent loads could collide on
  one `.dtws` file) and its "unique" component was a static address, identical
  in every process of the same image, so two processes generated byte-identical
  paths; the name is now a `random_device` + clock process tag plus one atomic
  counter per load.
- **Breaking:** removed `readCSV`, `readTimeSeriesCSV` and `readCSVColumn` from
  `dtwc/fileOperations.hpp` (no callers repo-wide) and, with them, the RapidCSV
  dependency: the CPM package, the `rapidcsv::rapidcsv` links in
  `dtwc++`/`mip-solvers` and the `--rapidcsv-include` plumbing in
  `scripts/test_f19_problem_encapsulation.py` are gone. Those readers used
  locale-dependent `std::stod` inside a silent `catch`, so under a
  comma-decimal locale `"1.5"` parsed as `1` and a malformed field silently
  shortened the series. One fewer fetched dependency at configure time; no
  behaviour change on any live path.
- Removed unused `ParquetChunkReader::estimated_total_bytes()`,
  `estimated_bytes_per_series()` and `read_row_group(int)` (the per-series
  estimate is no longer computed on every open) and the dead core cost functors
  `core::L1Dist`, `core::SquaredL2Dist`, `core::MVL1Dist`,
  `core::MVSquaredL2Dist`, `core::SpanSquaredL2Cost` and
  `core::SpanMVSquaredL2Cost`, plus several unused includes.
- `Data::series()` and `Data::series_f32()` now reject a precision mismatch
  instead of indexing the empty vector for the other precision, and `ndim == 0`
  is rejected at construction (`series_length()` divides by `ndim`).
- One `Ndata` contract across every loader: a negative value means "read all",
  otherwise exactly `Ndata` series. `Ndata == 0` used to give 1 from
  `DataLoader::count()`, ALL series from the folder load and 0 from the batch
  load; `Ndata < -1` is now rejected instead of silently meaning "all" or
  "none" depending on the route.
- `DataLoader::path()` matches extensions case-insensitively (`data.TSV` now
  selects the tab delimiter) and no longer overwrites a delimiter the caller
  set explicitly.
- `load_folder` / `load_batch_file` honour `verbosity(0)`; "Reading data:" and
  "N time-series data are read." were printed unconditionally, so the Tier-1
  `dtwc::load` path could not be silenced.
- `Problem::write_clusters`, `write_silhouettes`, `write_medoid_members` and
  `writeBestRep` now report an unopenable or unwritable output file instead of
  silently producing nothing. The mmap distance-matrix CSV is written through
  the shared `core/matrix_io.hpp` formatter rather than a third copy of the
  same loop (bytes unchanged; the non-finite preflight still runs before the
  destination is truncated, and it runs exactly once -- the emitter is called
  through the already-preflighted entry point, not through `operator<<`, so no
  second O(N^2) scan is paid on precisely the matrices large enough to map).
- Known limitation (documented, not changed): nothing checkpoints from inside
  `fill_distance_matrix()`. `CheckpointOptions::save_interval` is inert and a
  crash during one uninterrupted fill loses that fill. `checkpoint.hpp` used to
  imply an automatic mid-fill checkpoint that has never existed.
- Fixed multivariate dispatch silently flattening channels: `ndim > 1` with
  `MissingStrategy::Interpolate` or `DTWVariant::SoftDTW` ran a *univariate*
  recurrence over the interleaved channel stream (and counted `band` in flat
  elements, not timesteps). Both are now rejected at bind time with
  `InvalidInput`, mirroring MSM/TWE.
- Missing-data handling is enforced at every entry point.
  `MissingStrategy::Error` now applies to the pairwise API
  (`dtwc::distance::dtw`), where NaN input
  previously returned NaN — which is also the distance matrix's "uncomputed"
  sentinel, so a computed result was indistinguishable from an unfilled entry.
  An all-NaN series under `MissingStrategy::Interpolate` is rejected by
  `Problem::fill_distance_matrix` in the serial pre-scan, naming the offending
  series and index, instead of surfacing as a bare `interpolate_linear` failure
  thrown from inside the parallel per-pair fill.
- Fixed `lb_enhanced` / `lb_webb` returning an inadmissible bound for
  `band < 0`. A negative band means unbanded DTW, not radius 0; clamping it
  pinned the elastic arms to the diagonal (counterexample `A=[0,5,0,0]`,
  `B=[0,0,5,0]`: bound 10 against a true DTW of 0). They now return 0.
- The `lb_keogh` / `lb_enhanced` / `lb_webb` envelope entry points now validate
  every envelope array they index, not just `upper`; a ragged `Envelope` or
  `WebbEnvelope` used to read `lower` / `ul` / `lu` out of bounds. `lb_keogh`
  keeps its D2-derived semantics unchanged: it still includes the first
  `n = min(query.size(), env.upper.size())` rows (the unequal-length prefix
  theorem), and returns 0 only when an array it would index is shorter than
  that prefix. `lb_enhanced` / `lb_webb` have no prefix theorem and keep exact
  length equality.
- The four weights-taking WDTW overloads now reject a weight array shorter than
  `max(|x|, |y|)` instead of reading past its end.
- The pruned distance-matrix fill no longer discards entries that are already
  present, so a restored checkpoint survives. `Auto` resolves to `Pruned` only
  for Standard/ADTW with `MissingStrategy::Error`, dense Float64 storage,
  `band >= 0` and `N >= 64`; that exact configuration is now what the
  regression test drives through `Problem::fill_distance_matrix()`, alongside
  the direct `fill_distance_matrix_pruned()` contract test. The fill also
  routes pair decoding through the shared `dtwc::detail::decode_pair` instead
  of a local copy of the retired single-`if` correction.
- Float32 and `Pruned` no longer interact silently: `Auto` never resolves to
  `Pruned` for Float32 data, and an explicitly requested
  pruned fill on a dense Float32 `Problem` is a typed
  `dtwc::InvalidInput` raised at strategy resolution. The pruned summaries,
  envelopes and kernels are f64-only and read through `Problem::series()`, so
  the combination previously surfaced as a `Data::series` precision error
  thrown from inside the parallel fill (and, before that guard existed, as
  undefined behaviour indexing the empty Float64 vector). That rejection now
  sits BELOW the `Pruned + mmap -> BruteForce` downgrade, so Float32 data with
  an mmap-backed distance matrix — which the downgrade routes to the exact
  generic row fill, which handles Float32 — fills again instead of being
  rejected.
- `dtwc::detail::decode_pair` hoists the row start so the two-way row
  correction costs one comparison per pair instead of a fresh 64-bit multiply
  and divide -- it is the on-device decoder, paid per pair per launch -- and
  applies the low clamp last, so `N < 2` can no longer hand back a negative
  row.
- `dtwc::run` no longer mutates process-wide OpenMP state: the worker limit is
  applied through a `num_threads(...)` clause, so one constrained call can no
  longer pin every later distance fill (and pruning statistics stop depending
  on call order).
- Added the reproducible D3 derivation and fail-closed executable oracles for
  LB_Enhanced and the local LB_Webb_NoLR-plus-tail-cap implementation. The
  finite exact-arithmetic campaign covers 2,004 envelope cases, 35,982 path
  and Webb cases, 68,787 Enhanced configurations, all four Webb correction
  branches, strict ordering/tail witnesses, both live pruning routes, and
  saturated `INT_MAX` window arithmetic. The post-D3 native floors are
  125/125 (llfio ON), 125/125 (llfio OFF), and 127/127 (Arrow ON), with the
  Arrow reader and all four real-CLI integration routes executing.
- Corrected the lower-bound provenance contract: the public `Webb` API is the
  paper's all-index `LB_Webb_NoLR` formula plus a conservative trailing-flag
  cap, not full Algorithm 2 with `MinLRPaths`. Only that separate tail cap is
  proved to loosen exact-predicate NoLR; no universal ordering with full Webb
  is claimed. Enhanced dominates matching-direction Keogh at effective
  `V=1`, while exact D3 witnesses establish both order directions at `V>=2`.
- Fixed CPU LB_Webb and LB_Enhanced window geometry for valid extreme radii:
  radii above `n-1` now use the equivalent global window, and LB_Webb's
  doubled radius, free-run counters, and shifted indices cannot overflow.
- Fixed the CPU `Enhanced` pruning strategy to evaluate
  `max(LB_Keogh, LB_Enhanced)` after LB_Kim. Effective `V>=2` has no
  pointwise ordering with Keogh, so selecting Enhanced alone could silently
  weaken the documented cascade.
- Added public Python `save_binary_checkpoint` and
  `load_binary_checkpoint` bindings for binary-v1 `ClusteringResult`
  checkpoints. Both accept path-like objects and release the GIL during native
  filesystem work; failed loads raise the public typed `dtwcpp.IOError`.
- Made binary-v1 clustering-result checkpoints host-independent and
  fail-closed: all scalars use explicit little-endian encoding, readers reject
  noncanonical structural bytes and wrong lengths before count-derived
  allocation, failed reads leave the destination unchanged, and writer I/O
  failures use the public error taxonomy. The format version and its existing
  semantic/provenance limits are unchanged.
- Added the D2 envelope/LB_Keogh derivation and executable oracle. It confirms
  scalar L1 and unrooted squared-L2 admissibility, including the feasible
  unequal-length prefix theorem and additive dependent/independent
  multivariate extensions, and records the current envelope/Kim and GPU
  metric/window/prefix-execution/overflow limitations as F46-F50 and
  F27-F29. F48 records the empty-series TADPole inconsistency; F49 records
  direct-call band/cache provenance. GPU threshold-result documentation now
  names the actual finite public double-max sentinel rather than IEEE
  infinity. The former unconditional TADPole prune/brute identity wording is
  now scoped to exact arithmetic and the exactly representable regression;
  floating threshold identity remains open under D17.
- Tightened lower-bound interface contracts: TADPole LB/UB counters describe
  density decisions rather than guaranteed avoided DTWs, no prune rate is
  inferred from admissibility alone, additive multivariate unit claims require
  commensurate/scaled channels, and CUDA's Python pruning documentation now
  states the required nonnegative band.
- Corrected the legacy CPU exact-matrix lower-bound documentation: a cutoff
  result is recomputed without a cutoff because every entry is required, and
  `band=-1` disables LB_Keogh. This removes the unsupported speed implication
  without changing runtime behavior.
- Enforced the frozen C++ deprecation policy for all 33 retained 1.x
  compatibility entities. `Problem::maxIter`/`N_repetition` and the seven
  legacy Problem I/O overloads now emit their registered replacement
  diagnostics while remaining behavior-identical; canonical I/O names own the
  implementations and canonical Problem moves remain warning-silent.
- Fixed LLFIO-enabled public headers so third-party pragmas no longer suppress
  downstream Clang deprecation diagnostics.
- Added the frozen C++ `DataLoader::start_column`/`start_row` and
  `settings::paths::set_data_path`/`set_results_path` canonical names, including
  exact filesystem-path and C-string overloads. The four camelCase 1.x names
  remain behavior-identical deprecated forwarders for the 2.x transition.
- Fixed `Problem::set_storage_policy` so the next owning `set_data` call now
  selects Heap or mmap series backing across C++, Python, and MATLAB.
  Mmap-backed Problems retain their store/name lifetime after moves; unsupported
  Float32, llfio-OFF, CUDA, and Metal combinations now fail loudly instead of
  silently using Heap or empty owning vectors.
- **Breaking:** encapsulated `Problem`'s `method`, `random_seed`,
  `last_iterations`, `tadpole_dc`, `lb_strategy`, `storage_policy`, `verbose`,
  `output_folder`, `name`, and `data` fields behind canonical accessors and
  nine canonical setters; `last_iterations()` and `data()` are read-only.
  Existing Python properties and MATLAB mutators remain writable and delegate
  to those C++ setters, and redundant MATLAB-side clustering-result writeback
  was removed.
- Corrected the frozen MATLAB Tier-1 documentation to flag that functional
  `dtwc.cluster(...,'device',...)` currently reports Env selection without
  routing its local `Problem` computation (F40); real Metal reachability for the
  separate MATLAB estimator is explicitly environment-blocked under F41.
- Added executable red-first gates for MATLAB estimator metric/device routing
  and recorded the blocking Windows CUDA-MEX Auto-precision access violation
  under F42; no failed F18 product change is retained.
- Fixed CLI `--resume`: it now validates and exactly replays the completed
  binary clustering result, skips every clustering method, preserves the source
  checkpoint, and fails loudly when requested state is missing or incompatible.
  Directory distance checkpoints and mmap caches remain independent.
- Made the tracked CMake presets portable and truthful: they now declare the
  actual CMake 3.26 floor, discover Windows Clang through `PATH`, and hide
  host-specific configure/build/test choices on other operating systems. The
  installation guides now state the active CMake 3.26 and C++20 requirements.
- **Breaking:** distance-matrix CSV output now uses locale-independent
  round-trippable binary64 tokens, preserves signed zero, writes LF-only bytes
  on every host, and rejects computed infinity before writing matrix bytes.
- Corrected nearest-medoid assignment in FastPAM, CLARANS, resident/streamed
  FastCLARA, and Lloyd: first-slot ties and finite `DBL_MAX` objectives now
  publish consistently, objectives fold in point order, and non-finite
  distances or overflowing objectives raise typed diagnostics.
- Corrected CPU Float32 DTW no-path results to the public double `DBL_MAX`
  sentinel instead of exposing widened `FLT_MAX`; CPU and GPU result
  boundaries now share the same exact normalization policy.
- **Breaking:** corrected CUDA banded DTW to the public canonical
  `|i-j| <= band` contract for unequal lengths. All CUDA pairwise and
  one/K-vs-N kernel families now use fixed geometry without signed
  `band+1` arithmetic, and FP32 GPU no-path results are translated to the exact
  public double `DBL_MAX` sentinel instead of exposing widened `FLT_MAX`.
  Metal now has the matching source-level sentinel and overflow-safe no-LB
  DTW-kernel bound repair; real-device validation remains pending.
- Pinned the standalone C++ example's DTWC++ archive to the 2.0.0rc1 commit
  and its SHA-256. The supply-chain gate now scans every tracked
  `CPMAddPackage(URL ...)` declaration, including inline CMake syntax, and
  requires the exact registered path/name/URL/SHA-256 inventory, rejecting
  missing or comment-only hashes, dynamic or semicolon-expanded argument
  injection, ten direct alternate-source selectors, decoy package identities,
  and branch archives. HiGHS' conditional cuPDLP setting is now explicit while
  its archive declaration remains fully literal. Quoted CMake line
  continuations and CPM custom-cache-key overrides remain tracked parser
  limitations rather than silently claimed coverage.
- **Breaking:** corrected CPU banded-DTW routes to the canonical Sakoe–Chiba
  window `|i-j| <= band`. Unequal-length inputs now return the documented
  no-path sentinel when `band < |n-m|`; the previous endpoint-scaled slanted
  corridor could return a finite distance, and could bypass feasibility for
  singleton scalar, dependent-multivariate, and DTW-AROW calls. The shared
  Standard/ADTW/WDTW/AROW/missing-data kernel now uses one fixed-width window;
  independent multivariate no-path calls preserve the finite sentinel instead
  of summing it to infinity; and maximum-width integer bands no longer evaluate
  an overflowing `band+1`.
- Reconciled the remaining API examples and method documentation with the live
  canonical names, multivariate route limits, cluster-score edge behavior, and
  selected floating-point/sentinel contracts.
- Corrected the GPU backend guide to the live option/dispatch surface, scoped
  lower-bound pruning to its currently defensible equal-length L1 regime,
  identified correctness/loudness gaps F27–F31, and removed untraceable
  pruning/regtile speed claims and exact-matrix pruning advice.
- Re-audited the frozen cross-language API contract against the current tree,
  resolved all eight stale reviewer questions, and named the nine unfulfilled
  2.0 implementation promises as R3 findings F18–F26 instead of presenting
  them as shipped behavior or silently deferring them to 2.1.
- Documented the mmap distance-cache version-3 format delivered by M53:
  semantic SHA-256 identity, per-row payload digests, exclusive session lease,
  rejection of legacy v1/v2 caches, and accidental-corruption (not keyed
  tamper-proof) scope.
- Reconciled the configuration reference with all 52 live CLI flags and
  separated TOML from the smaller YAML key map/precedence bug.
- Corrected the website method catalog: removed unsupported Huber, documented
  MSM/TWE and all CLI clustering methods, fixed FasterPAM provenance/complexity,
  and stopped presenting the exact-matrix recomputation route as an accelerator.
- Corrected the README's live variant/method counts, CUDA architecture defaults,
  complete CMake option inventory, Float32/streaming scope, and removed
  performance numbers that lacked a tracked originating result artifact.
- Corrected the frozen API precision record: public templates and the CLI
  default to Float64, while explicit Float32 storage also uses Float32 DTW
  recurrence arithmetic before its result is stored as a double.
- Fixed system-package Arrow builds silently omitting Parquet support even
  after `find_package(Parquet)` succeeded. The detected package now reaches the
  parent build scope, so `libparquet-dev` enables the production reader and its
  tests as configured.
- Fixed `--ram-limit` so Parquet is planned from schema and row-group metadata
  before the selected payload is materialised. The fail-closed binary-size
  parser is exact through the platform `size_t` boundary; conservative
  Float64/Float32 decode peaks select eager loading only when it fits. An
  over-budget request streams only non-full FastCLARA over one list-per-row
  Float32/Float64 file; scalar columns, directories, other methods, CUDA, full
  samples, indivisible row groups that do not fit, and incompatible legacy
  matrix/checkpoint inputs now fail with remediation before a hidden full load.
  **Breaking:** `--ram-limit` is now rejected outright for non-Parquet input
  (CSV/TSV, HDF5, Arrow IPC, `.dtws`, and CSV directories). It previously warned
  and continued, which loaded the whole file anyway while reporting a cap that
  was never applied.
  Streamed samples, medoids, and assignments preserve requested Float32 storage
  and produce byte-identical labels, medoids, and binary result checkpoints to
  resident execution without constructing dense matrix or silhouette output.
  Eager and streaming Parquet readers now share exact scalar/List/LargeList
  schema selection, checked list offsets, and Arrow-field-to-Parquet-leaf
  mapping even when earlier nested fields own multiple physical leaves.
- **Breaking:** the CLI now rejects `--device cuda` for every non-full-sample
  FastCLARA run, including `--method auto` above 5,000 series where `auto`
  resolves to CLARA. This fires on default flags and independently of
  `--ram-limit`. Non-full FastCLARA became a matrix-free CPU schedule, so the
  GPU distance matrix it previously built was computed and then discarded;
  rejecting is truthful where silently paying for an unused matrix was not. Use
  `--device cpu`, or `--method pam`/full-sample CLARA to keep the GPU route.
- Fixed distance-proportional initialization for Soft-DTW, whose finite
  raw dissimilarities may be negative. Sampling now translates all unselected
  distances by one common offset while keeping selected medoids at zero;
  nonnegative DTW/MSM/TWE schedules and their portable seeded fingerprints are
  unchanged. Degenerate all-zero weights (for example, identical series) now
  complete the medoid set with the first unselected index instead of constructing
  an invalid standard-library discrete distribution.
  That fix is now pinned directly: a new unit suite asserts the sampling-weight
  contract (nonnegative input unchanged, selected entries exactly zero and
  excluded from the total, a common shift computed from unselected entries only,
  exactly zero totals for degenerate input, and typed rejection of non-finite
  distances and out-of-range selected indices), and the previously untested
  degenerate and signed branches of `init::Kmeanspp_seeded` and
  `fast_pam_seeded` now have behavioural cases.
- Made every FastPAM entry enforce the library's signed-32-bit point-index
  boundary before distance-matrix materialisation or result mutation. Oversized
  datasets now raise a typed error instead of narrowing `Problem::size()`;
  supported inputs use one checked conversion and retain int-indexed kernels.
- Removed FastCLARA's hidden packed O(N²) parent-cache allocation. Non-full
  assignment now evaluates the configured float64/float32 DTW dispatcher
  directly with O(N) result scratch, while subsample PAM alone owns O(s²)
  storage. Existing dense or mapped parent caches are ignored and preserved,
  so injected cache contents can no longer change FastCLARA's bound-DTW result.
- Hardened FastCLARA's execution plan before any allocation or Parquet I/O.
  Invalid repetition, iteration, and sample-size controls now raise typed
  errors; datasets beyond the signed-32-bit medoid-index ABI fail loudly; and
  the Schubert--Rousseeuw auto sample formula uses checked 64-bit arithmetic.
  Resident full samples still run one seeded PAM, while a full sample that
  contradicts a streaming RAM limit is rejected with remediation. The CLI no
  longer substitutes a divergent `sqrt(N)*k` policy for the documented formula.
- Made every maintained invocation-local seeded clustering path independent of
  the host C++ standard library's unspecified random-distribution mappings.
  Barycenter SSG/k-means++, seeded FastPAM and Lloyd initializers, OneBatchPAM,
  CLARANS, and both FastCLARA paths now share one versioned portable
  `mt19937_64` map: bounded draws are unbiased, while floating and weighted
  draws have a fixed 53-bit mapping. Exact seeds therefore agree across MSVC
  STL and libstdc++.
  FastCLARA also drops its avoidable 8*N-byte sampling index pool. The mutable
  unseeded Tier-2 `std::mt19937` contract remains unchanged.
  **Breaking compatibility (portable-v1):** rc1 seeded calls used
  vendor-defined mappings around `mt19937_64`; after this supersession, the
  same explicit seed can therefore produce different literal medoids, labels,
  and barycenters than rc1. This boundary does not promise historical
  fingerprint reproduction. The mutable unseeded Tier-2 contract is unchanged.
- Made HiGHS-dependent MIP, Benders, and Lagrangian comparison tests skip
  explicitly when the optional solver is absent. No-solver builds still test
  the typed unavailable-backend contract, while their full CTest gate no
  longer reports capability absence as a product failure.
- Made every public non-distance enum selector fail closed. Invalid clustering
  method, exact solver, hierarchical linkage, PAM variant, OneBatch weighting,
  barycenter method, environment device, and assignment-matrix layout values
  now raise typed errors before shortcuts, allocation, backend selection,
  distance/table work, stats, or output mutation; every declared value remains
  operational. Python's typed enum casters and MATLAB's string parsers retain
  their existing surfaces while rejecting arbitrary numeric selector values.
- Replaced unauthenticated dense CSV checkpoints with versioned immutable
  generations selected by an atomically replaced `CURRENT`. The strict v2
  manifest binds exact bit-round-trippable N-by-N CSV bytes to the full
  dataset/distance-configuration identity and a payload SHA-256; malformed,
  asymmetric, non-finite, torn, stale, or legacy payloads fail without changing
  the existing `Problem` or cache. Save validates all source semantics and
  values before filesystem effects, failed overwrites preserve the active
  generation, and load publishes only by one non-throwing move. Payload streams
  are closed but not explicitly fsync'd, so power-loss tears fail closed and may
  sacrifice resume availability rather than expose unauthenticated distances.
- Made CUDA `KernelOverride` effective and observable across pairwise,
  one-vs-N, and K-vs-N APIs. Implemented Wavefront and RegTile requests now
  force those kernel families within their supported ranges; CUDA-only missing
  families (`WavefrontGlobal` and `BandedRow`) and oversized RegTile requests
  use Auto with an explicit fallback flag. Results report the actual kernel,
  while no-work and fully-pruned calls report `none` without claiming fallback.
  Invalid precision or override enums fail before device capability checks.
- Made `Problem` distance-semantic rejection effect-free across C++, Python,
  and MATLAB. One ordered preflight now validates complete variant parameters,
  variant/missing compatibility, active float32 narrowing, and multivariate
  capability before selector/data publication, cache clear or resize,
  dispatcher rebinding, mmap detachment, or file creation. Rejected variant,
  missing-strategy, owning-data, and view-data candidates preserve exact prior
  state and dense/mapped cache values; valid and no-op routes are unchanged.
  Invalid enum membership remains the separate M47 hardening scope. This
  guarantee covers deterministic semantic validation, not later allocation or
  filesystem failures after a successful preflight.
- Made float32 `Problem` variant binding reject active double parameters that
  overflow float or collapse from nonzero to zero before any data, cache,
  dispatcher, or mmap mutation. Float64 retains its full parameter domain;
  inactive float32 fields, exact-zero WDTW/ADTW limits, and minimum-positive
  float controls remain valid. Direct and algorithmic f32 callable access now
  crosses one validated boundary, so an unavailable f32 dispatcher cannot
  surface `std::bad_function_call`.
- Made public distance dispatch fail closed across C++, Python, and MATLAB.
  Unknown raw metric tokens now raise typed invalid-input errors instead of
  silently selecting L1, and non-Standard/non-Error cross-products fail before
  kernel selection. Accepted Standard missing-data routes now execute the
  requested recurrence consistently in the runtime API, while registered
  metric aliases and non-Standard/Error outputs remain unchanged. Transactional
  rollback after rejected `Problem` setter calls is tracked separately.
- Made public `softmin_gamma` enforce Soft-DTW's exact finite-positive gamma
  contract with typed `InvalidInput` failures in Release builds. Validated
  gradient/DP loops use a non-throwing unchecked cell primitive with scaling
  state precomputed once; denormal float/double gamma remains finite through a
  `frexp`/`scalbn` fallback resistant to reciprocal-math optimization, while
  ordinary arithmetic order and warmed allocation behavior are unchanged.
- Guarded matrix-free `Problem::dtw_function()` access with the same fixed-size
  semantic snapshot as cached distances. Mutable f64/f32 access now rebinds
  after legacy raw configuration edits, const access rejects stale dispatch,
  and mmap replacement reconciles the callable before publishing its identity;
  unchanged getters remain allocation-free. OneBatchPAM resolves both callables
  serially before its OpenMP table build so first-use repair cannot race.
- Contained every pruned distance-matrix worker failure inside the shared
  deterministic OpenMP boundary. Summary, envelope, pair-block, and standalone
  row failures now rethrow their original type on the caller thread; exact
  quotient/remainder blocks cover each pair once and partial matrices remain
  incomplete.
- Replaced pruned nearest-neighbor thresholds' mixed ordinary/compiler-atomic
  access with portable `atomic<double>` storage and explicit relaxed loads/CAS.
  This removes the C++ data race, strict-aliasing violation, and MSVC volatile
  pseudo-atomic load without changing finite distance matrices.
- Restricted lower-bound pruning to the missing-data semantics it implements.
  Auto and explicit Pruned requests now preserve ZeroCost, AROW, and Interpolate
  behavior through the exact generic fill instead of feeding NaNs to raw DTW;
  explicit routing is explained in verbose output.
- Made mapped distance storage part of pruning selection. Auto/explicit Pruned
  now fill mmap caches through the exact generic path, while a direct low-level
  dense-only pruned call reports actionable `InvalidInput` instead of leaking
  `std::bad_variant_access`.
- Replaced formatted-stream batch parsing with exact-delimiter, full-token
  numeric parsing. Textual NaNs and later fields are preserved; malformed,
  empty, out-of-range, or unapproved non-finite values fail with file/row/column
  context, and metadata-only sizing follows the identical rules.
- Made OpenMP row execution capture and deterministically rethrow the lowest-
  index worker failure after the join, so a failed distance pair cannot publish
  a complete cache. The CLI now catches operational exceptions at its outer
  boundary and reports an actionable error with a normal nonzero exit.
- Enforced one finite mathematical domain for every DTW-variant parameter at
  C++ free-function/runtime/`Problem`, CLI/YAML, Python, HPC, and MATLAB
  boundaries. WDTW `g` and ADTW penalty retain their valid zero limits;
  Soft-DTW gamma, MSM cost, and TWE stiffness/edit penalties must be positive.
  Invalid state now raises the typed public input error before side effects,
  while ordinary outputs are unchanged.
- Hardened every non-arbitrary SLURM wrapper entrypoint: build profiles,
  benchmark GPU types, SSH endpoints, remote bases, and optional Slurm settings
  now have explicit grammars before network effects. Remote build, preflight,
  status, submission, and transfer commands use quoted argv construction; the
  documented ARC profiles/configuration remain valid and `ssh` stays the
  explicitly arbitrary escape hatch.
- Aligned Python Tier-1 `cluster()` validation with C++ `validate_common`:
  `k` and `max_iter` must be positive signed C++ integers and `Dataset.skip_cols`
  must be nonnegative. Python and NumPy integers are normalized consistently;
  booleans, non-integral values, and overflow now fail before loading data,
  resolving a device, submitting HPC work, or constructing local compute state.
  Direct HPC calls also preserve the validated `max_iter` instead of reverting
  to the remote default.
- Made Python Tier-1 `kmedoids`, `mip`, `lrcore`, and `tadpole` honor the
  public `max_iter` argument before `Problem.cluster()` dispatch. The M29
  one-iteration Lloyd result is now reachable without changing the default-100
  result or methods that already receive their limit directly.
- Made direct HiGHS and Gurobi clustering transactional: FastPAM incumbents no
  longer overwrite caller state, exact assignments are decoded and validated
  privately, and only a complete k-medoid/N-label result is published. Solver
  or extraction failures restore the caller's prior medoids and labels.
- Made Python HPC clustering submissions transactionally job-specific: local
  inputs, remote uploads, and submitted scripts no longer collide; polling has
  a real wall-clock bound and propagates configured-cluster failures; and label
  retrieval requires a successful exact job-ID transfer instead of choosing a
  newest same-name file.
- Made Python Tier-1 Lloyd k-medoids explicitly select the shared
  invocation-local seed 42 before clustering. Consecutive and legacy-Tier-2-
  interleaved calls are reproducible; explicit advanced seeds, custom
  initializers, the unseeded Tier-2 RNG contract, and other methods are unchanged.
- Made iteration-capped Lloyd k-medoids return one coherent final state by
  assigning labels to the final medoids before calculating restart costs.
  Fully converged runs retain their existing assignment and iteration path.
- Bound dense and injected precomputed distance matrices to the exact band,
  variant parameters, multivariate mode, missing-data policy, backend, and CUDA
  settings that produced them. Semantic setters now invalidate stale work;
  legacy raw C++ and nested Python mutations are detected before reuse, while
  identical assignments preserve the existing matrix and mmap remains loud.
- Documented the live `--missing-strategy` values, aliases, default, and
  TOML/YAML key in the canonical CLI and configuration references; the exact
  live-binary flag drift gate now covers the option.
- Hardened Python-to-SLURM clustering submissions with pre-side-effect job/path
  grammars, bounded positional validation, transfer option protection, and
  per-argument remote-shell quoting. Unsafe export bytes now fail locally,
  while accepted paths reach the job unchanged and inherited seed/dtype values
  cannot silently alter the request.
- Made `DTWClustering` honor one distance contract across fit, inertia, and
  predict. Standard squared DTW now drives CPU/GPU medoid selection; MSM, TWE,
  and missing-data predictions use their configured production recurrence;
  unsupported cross-products and all-non-finite restart sets fail loudly. The
  default Standard-L1 CPU path remains lazy and output-compatible.
- Prevented inherited Slurm environments from supplying an unintended HPC
  clustering seed when Python callers omit `seed`; omission now explicitly
  preserves the remote CLI default while explicit seeds remain unchanged.
- Fixed barycenter k-means falsely converging before its first center update
  under the default positive tolerance. Computed non-finite squared/soft-DTW
  costs, gradients, updates, initialization weights, and assignment costs now
  fail with rescaling guidance instead of returning invalid results.
- Preserved the complete `DTWClustering(device="hpc")` distance configuration
  through SLURM submission to the final CLI command. Unsupported CPU/CUDA,
  variant, missing-data, metric, multivariate, ordinal, and numeric combinations
  now fail before remote side effects; the CLI also exposes
  `--missing-strategy` in command-line, TOML, and YAML configuration.
- Stopped Benders warm starts from writing nested Lloyd medoid and
  best-repetition artifacts. The private no-persistence route retains identical
  initialization, incumbent, progress output, and exact-solver trajectory;
  direct public Lloyd clustering keeps its documented files and stdout.
- Honored deterministic PAM restart counts in the CLI and Python HPC path.
  Restart `i` uses checked `seed+i` and the strict best objective is retained;
  count/seed now reach the final SLURM command without overriding the CLI seed
  default when callers omit it.
- Made Benders warm starts exception-safe: temporary Lloyd method, restart,
  iteration, medoid, and label state is restored on both success and failure,
  while the final exact Benders clustering result remains exposed to callers.
- Enforced LF checkouts for every shell and SLURM entrypoint so documented Git
  Bash workflows remain parseable on Windows with `core.autocrlf=true`; all 11
  tracked scripts pass a repository-wide `bash -n` gate.
- Made Lloyd k-medoids and direct HiGHS/Gurobi MIP warm starts invocation-local
  at the shared seed-42 default. Lloyd repetitions now use checked `seed+i` and
  return the actual lowest-cost state instead of the final run; custom
  initializers and the unseeded Tier-2 RNG contract remain unchanged.
- Made mmap distance-cache creation crash-consistent and single-writer. A
  CRC-covered initializing/ready state now publishes only after every NaN
  sentinel is durably flushed; incomplete caches fail with recompute guidance,
  concurrent creators cannot truncate or alias one path, and explicit sync is
  blocking. The 64-byte v2 layout is retained by consuming a reserved byte;
  unreleased pre-fix v2 caches are deliberately rejected as incomplete.
- Unified seed-aware PAM, OneBatchPAM, and CLARA defaults on invocation-local
  seed 42 across C++, Python, MATLAB, sklearn, and CLI. Estimator restarts now
  use distinct deterministic seeds and FastCLARA no longer consumes the legacy
  global seed-29 engine; the unseeded Tier-2 FastPAM overload remains compatible.
  **Breaking compatibility (seed-42 default):** ambiguous default outputs can
  change from rc1 when a caller omitted a seed. Lloyd local optima and
  time-limited solver trajectories can consequently change. The exact optimum
  value of a fully certified exact solve does not depend on the initialization
  seed, although tied representative solutions may differ.
- Reused preallocated hard-DTW barycenter workspaces and parallelized independent
  assignment and cluster-update work while preserving digit-identical results,
  deterministic RNG streams, serial reductions, and empty-cluster repair.
- Replaced N-only mmap warm-start validation with a version-2 SHA-256 identity
  over the exact data and every distance-affecting setting. Stale, corrupted,
  legacy-v1, or ambiguous-backend caches now fail before exposing distances;
  legacy dense CSV checkpoint/matrix combinations are rejected at mmap scale.
- Restored the frozen API contract's decision-log governance and recorded the
  approved 2.0 scope for MATLAB's Tier-1 method set and the C++ HPC
  throwing-beta transport boundary.
- Fixed scikit-learn 1.6+ pairwise tags for
  `DTWCKMedoids(metric="precomputed")`, so cross-validation and `GridSearchCV`
  slice both rows and columns of a precomputed distance matrix.
- Fixed Python Tier-1 clustering silently ignoring Sakoe–Chiba bands in the
  matrix-free OneBatchPAM, CLARA, and TADPole paths.
- Added non-trivial finite-difference validation of the production soft-DTW
  barycenter adjoint at gamma 0.1 and 1.0; the existing arithmetic passed the
  registered `1e-5` relative band unchanged.
- Documented that matrix-free OneBatchPAM, CLARA, and TADPole results set
  `Result.distance_matrix` to `None`; callers needing N×N distances must compute
  them explicitly or use a matrix-based method.
- Completed the rc1 migration notes for hard GPU/solver errors and separated
  the absorbed development history from the release-candidate summary.
- Fixed both barycenter entry points to reject non-Standard DTW variants and
  finite bands with actionable `InvalidInput` errors instead of silently
  computing an unbanded Standard squared-cost objective.
- Fixed Python device parsing to accept `gpu:N` with the same grammar as C++,
  preserve ordinals through CUDA and Metal resolution, and use the frozen typed
  errors without falling back to another backend.
- Promoted the LR-core derivation into tracked documentation sources and made
  generated-doc checks fail loudly if that canonical source is missing.
- Replaced the mirrored Metal chunk-offset assertion with a host-testable
  production `int64_t` conversion seam used by both dispatch paths and pinned
  beyond `INT32_MAX` by a real pair-decode round trip.
- Corrected OneBatchPAM finite-maximum debiasing for distance tables with
  `0 < Dmax < 1`; normalization now uses the actual table maximum, with a
  finite fallback only for all-zero tables, and its hybrid provenance is explicit.
- Fixed OneBatchPAM's relative stopping tolerance below unit objective values;
  it now scales by the current estimate instead of an absolute floor of one.
- Pinned every external GitHub Action to a reviewed full commit SHA, corrected
  the nonexistent Doxygen action tag, and SHA-256 pinned the optional Arrow
  19.0.1 source archive with an executable repository-wide drift gate.
- Fixed SSG barycenters to retain Schultz–Jain warping-path valence in the true
  squared-DTW stochastic gradient, with a scalar inverse-Lipschitz step cap that
  prevents extreme unequal-length alignments from exploding.
- Added a cross-implementation soft-DTW guard: public L1 and barycenter squared
  recurrences agree when their local cost matrices coincide, while a sensitivity
  assertion preserves their intentional public semantic difference.
- Restored separate MATLAB assertions for honest sequential MEX reporting while
  retaining the OpenMP engagement gate; both flavors are now tested explicitly.
- Fixed CLI TADPole to honor `--mmap-threshold`: exact/fallback distances now
  use the file-backed packed cache, while LLFIO-off builds fail before an O(N²)
  heap allocation with actionable alternatives. OneBatchPAM remains O(Nm)-exempt.
- Made Tier-1 `method="auto"` device-compatible: CUDA and Metal now select PAM
  at every N instead of resolving large datasets to unsupported CPU-only CLARA;
  explicit incompatible methods remain loud and unchanged.
- OneBatchPAM now warns on every explicit `batch_size < n_clusters` correction,
  including requested/effective values and remediation; automatic sizing and
  already-valid explicit sizes stay silent.
- FastPAM's seeded BUILD now documents and tests objective-matched D sampling
  for its sum-of-DTW-distances PAM objective, and no longer uses deprecated
  `Problem::distByInd` access internally.
- Strengthened the OneBatchPAM 50k release validation with warped length-64–128
  signals, an exact exhaustive profile oracle, a discriminating mutation check,
  and tight work/memory bands instead of the former scalar length-1 fixture.
- Ported the cross-platform fixes from PR #32 (Kasper Westman, Apple Clang +
  libc++ and GCC 14 HPC): `Problem` construction no longer diagnoses the
  deprecated `maxIter`/`N_repetition` fields under GCC's constructor-NSDMI
  check while caller access still warns; the F13 assignment oracle stores
  exact-zero nearest/second distances as `+0.0` (GCC flushes `-0.0` under
  `-fno-signed-zeros`); F8 Soft-DTW parity accepts a registered 2-ULP GCC
  encoding of `total_cost` while still requiring byte-identical
  resident/stream checkpoints and exact labels/medoids.
- Registered the F15 `libcxx` deterministic-series fingerprint for Apple Clang
  + libc++ (scalar/row hashes match libstdc++; the continuous accelerator
  stream differs by a few ULPs and is now an accepted coherent profile).
- Fixed the F22 C++ deprecation probe for Apple Clang: paired
  `-Xpreprocessor -fopenmp` / `-Xclang -fopenmp` are copied from
  `compile_commands.json` instead of a bare `-fopenmp`, and Unix absolute
  paths are no longer parsed as MSVC `/U` flags. LLFIO's libc++
  `char_traits<std::byte>` deprecation is ignored only inside
  `llfio_include.hpp`.
- Fixed the F14 distance-matrix CSV contract so its exact-path markers are
  platform-neutral: the escaped Windows separator from `std::quoted` is
  collapsed before comparison, so the same single-slash marker holds on
  Windows and POSIX.

# 2.0.0rc1 - 2026-07-10

- Release-candidate packaging is reproducible from the root `VERSION` file:
  Python wheel/sdist, MATLAB MEX, and CPack executable archives report the same
  version. The [2.0 migration guide](docs/content/guides/migration.md) covers
  renamed APIs and behavioural changes.
- Added the owning Tier-1 C++ `device` / `load` / `cluster` / `Result` API and
  executable C++, Python, and MATLAB quickstarts backed by one conformance
  fixture.
- Added OneBatchPAM, DBA/SSG/soft-DTW barycenters, barycenter k-means, and the
  sklearn-compatible `DTWCKMedoids` estimator. The profiler gate retained the
  adaptive OpenMP schedule and rejected a speculative SIMD layer for lack of
  evidence.
- Wheels bundle HiGHS and its attribution, exercise a real MIP solve after
  installation, and exclude build headers/libraries. Source distributions now
  exclude local build and generated-site trees.
- Added Hugo contract/guides/math/benchmark documentation with generated-SSOT,
  live CLI/error-string drift checks, and internal-link validation.
- Explicit CUDA/Metal requests now raise `DeviceError` when the requested
  backend is unavailable instead of warning and silently running on CPU.
- Requesting an uncompiled Gurobi, HiGHS, or Benders backend now raises
  `SolverError` instead of printing a message and returning without a result.
- Primed the lazy distance dispatch before parallel k-means++ initialization,
  closing the `dist_by_ind` rebind race that produced Windows `0xc0000409`
  crashes.
- See the [2.0.0rc1 release notes](docs/content/releases/2.0.0rc1.md) for
  platform status and the deliberately gated production-release steps.

# Development history absorbed into 2.0.0rc1

The task-level development ledger that fed the release candidate is in git
history: `git show e784e5c:CHANGELOG.md`.

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
