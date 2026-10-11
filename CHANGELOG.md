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
  series. Every `Problem` fill and `dtw_function()` refuses one on the CPU too, for a series emptied through
  `Data::p_vec` after `set_data` (it was refused on a GPU only, and a CPU stored `1.8e308`).
- A line of a folder's file with more than one value after the skipped columns is an error (v1.0.0 read the first):
  a pandas `index,value` file needs `--skip-rows 1 --skip-cols 1`.
- `read_distance_matrix` (`--dist-matrix`) refuses a matrix of another size, one not square and symmetric, and ±inf.
- A negative distance-matrix entry (v1.0.0's `-1`, a pair it never computed) is read as not computed only with
  `--not-computed negative` (C++ `NotComputed::Negative`, Python `not_computed="negative"`, MATLAB
  `'NotComputed','negative'`), and refused otherwise under every variant but Soft-DTW; v1.0.0 read every negative entry
  as not computed. 2.0 reads and writes an uncomputed pair as an empty field.

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
- Missing values as NaN (`--missing-strategy zero_cost`, `arow` or `interpolate`) and multivariate series (`ndim` values
  per step, `--mv-mode dependent` or `independent`).
- Series stored as `float`, half the memory, chosen where they are loaded: `dtwc_cl --dtype float32`, C++
  `DataLoader::dtype` and `dtwc::load(..., dtype)`, Python `dtwcpp.load(..., dtype="float32")`, MATLAB
  `dtwc.load(..., 'Dtype', 'float32')`. Each reader rounds the double it reads and holds no double copy; so does
  `readFile<float>` (v1.0.0 rounded the text to `float` directly: one ulp apart in a rare halfway case).
- The distance matrix stored as `float`, half the memory, whatever the series' type: `dtwc_cl --matrix-dtype float32`,
  C++ `Problem::set_matrix_dtype` and `Config::matrix_dtype`, Python `cluster(..., matrix_dtype="float32")`,
  `Problem.matrix_dtype` and `compute_distance_matrix(..., matrix_dtype=)`, MATLAB `'MatrixDtype'` and
  `set_matrix_dtype`; CLARA's sample matrices and TADPole's cache follow it. Each distance is rounded once to the
  nearest float (at most 2^-24 of it), so costs and scores move by about as much and medoids and labels can change
  where two choices differ by less; a distance beyond float32's range (3.4e38) is refused. Every sum over the matrix
  stays in double, and its exports keep its type (NumPy `float32`, MATLAB `single`).
- Methods FastPAM (FasterPAM), FastCLARA, OneBatchPAM, hierarchical, TADPole and LR-core (an exact Lagrangian branch and
  bound) beside Lloyd's k-medoids and the MIP; `--method auto`: FastPAM to 5,000 series, then FastCLARA (GPU: FastPAM).
- LR-core's root bound is the subgradient root in every language and build; `--lr-root kelley` (Python `lr_root`,
  MATLAB `LrRoot`) runs Kelley's cutting planes instead, whose master LP solves on linked HiGHS or, in Python, on
  highspy (the `mip` extra). Both reach the same optimum; `-v` names the root and its branch-and-bound nodes.
- `--seed` (42) for the seeded methods, whose draws are the same on every platform and standard library; `--n-init`
  (v1.0.0's `--repeat`) restarts FastPAM too.
- DTW barycenters (`dtw_barycenter`: SSG, DBA, Soft-DTW) and barycenter k-means (`barycenter_kmeans`).
- Scores: Davies–Bouldin, Dunn, inertia and Calinski–Harabasz beside the silhouette (`Result::score(name)`), and the
  adjusted Rand index and normalised mutual information of two labellings.
- GPUs: CUDA (compute capability 8.0 or newer, `-DDTWC_ENABLE_CUDA=ON`) and Metal (Apple silicon, built by default on
  macOS) fill the matrix of Standard DTW on univariate series and FastCLARA's samples (`--device gpu`).
- Input: Parquet and Arrow IPC (`-DDTWC_ENABLE_ARROW=ON`; a list column or several float columns hold a series per row,
  `--column` reads one) and `dtwc_cl --delimiter`.
- The distance matrix memory-mapped in a `.dtwm` file once it outgrows the memory limit (`--ram-limit`), also the
  checkpoint (`--checkpoint`, `--checkpoint-interval`: a fill resumes); with Arrow, `--ram-limit` streams Parquet
  through FastCLARA.
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
- `--ram-limit` (Python `ram_limit`, MATLAB `'RamLimit'`, `Problem::set_ram_limit`) is the memory a run may use, by
  default the memory available (or a Linux cgroup's limit, which a SLURM job's memory request sets): the distance
  matrix is mapped once its bytes exceed half of what the series leave of it, and `--mmap-threshold` (mapped from
  50,000 series) is gone; a Parquet file over it streams through `clara`; other inputs no longer refuse it.
- A checkpoint save during a fill writes only the rows filled since the last one, in place, the first the whole
  matrix: a fill writes at most twice the matrix's bytes, where each save wrote it whole. Saves come every
  `--checkpoint-interval` rows rounded up to the lane width (16 rows of doubles where the lanes are 128 bytes).
- FastCLARA runs every sample's PAM before one assignment pass over all of them: `-v` prints one assignment line a
  pass, not one a sample; a non-finite distance is reported at the lowest failing series over all samples; and series
  it cannot take (NaN, ±inf, a band narrower than two series' length difference) are refused before any sample, by
  their index in the data. Streamed (`--ram-limit`), the limit holds the sampled rows beside one decode and a
  sample's PAM, then the medoids, 8 bytes of labels and of nearest distance per row and sample, one decode and a batch
  of row groups, and after the stream every series' name and label: a smaller limit is refused, naming the bytes,
  where it streamed before.
- `.dtwm` files are version 5: a 64-byte header that adds the bytes of a value (8, or 4 for `matrix_dtype` float32)
  and the rows written. Version-4 files, from earlier 2.0 builds, are still read and mapped as float64; a file of the
  other `matrix_dtype` is refused (`InvalidInput`, naming the one that reads it), never converted.

## Performance

Each figure compares 2.0's code just before and after one change (v1.0.0 had none of FastPAM, OneBatchPAM, LR-core,
Soft-DTW, Python or CUDA), on the machine named.

- The CPU fill of Standard DTW on univariate series computes a series against 8 of its length at once in SIMD lanes
  (16 in `float32`): on an Intel Core Ultra 9 285 (24 threads, loaded) ECG5000's 4,500 series fill in 3.9 s, not 66 s.
- There a band-50 fill of 50 series of length 1,000 runs 5.1× faster; the lanes match the one-pair kernel bit for bit
  unless the compiler fuses a multiply-add in one and not the other (GCC does by default; squared L2 only).
- On 64-bit Arm the lanes hold 16 doubles (32 floats) and take each minimum with one `fminnm`: on an Apple M5 Pro
  1.41–2.00× faster on one thread, 1.61–1.69× in an 18-thread fill of equal-length series; bit for bit.
- On x86-64 the lanes of a clang or GCC build hold 16 doubles (32 floats) too: on an Intel Core Ultra 9 285 clang is up
  to 1.97× faster on one thread and 1.17–1.19× in a 24-thread fill of equal-length series; GCC 13.3 (in WSL) 1.36–3.87×
  on one thread and 2.39× and 2.54× in the fills, a ragged fill unchanged; bit for bit, but for GCC's squared L2 (the
  multiply-add caveat above), whose clusterings were unchanged. MSVC builds are unchanged.
- Without a band the lanes of a clang or GCC build take series of different lengths too, each lane stopping at its
  own, each row of the fill taking the series after it longest first: on an Intel Core Ultra 9 285 clang's ragged
  fills run 6.8–6.9× faster on one thread and 3.3–3.6× at 24 threads, GCC 13.3's 8.1–8.7× and 4.9–5.5×, equal-length
  fills unchanged; OneBatchPAM's batch table too; bit for bit, but for GCC's squared L2. MSVC builds keep the
  one-pair route for series of different lengths.
- FastCLARA's CPU assignment computes each medoid against 16 series at once (32 in `float32`) in the same SIMD lanes,
  in RAM and streamed from Parquet: on an Intel Core Ultra 9 285 3.4–3.8× faster at 24 threads and 7.8× on one thread
  under clang, 4.8–5.7× and 10.0× under GCC 13.3 (`float32` 7.4× and 7.9×; under a band 6.3× and 6.6×, MSVC's 1.29×);
  the same labels, medoids and costs, but for GCC's squared L2. MSVC builds take the lanes there only under a band.
- The CPU fill goes block by block, as matrix multiplication does: rows of the packed matrix whose series stay in a
  core's L2, each thread taking consecutive columns of them, so a cache line of the matrix comes from memory once for
  its eight entries; the NaN prefill runs on every thread. On an Intel Core Ultra 9 285 at 24 threads, for 8,000
  series of lengths 40–60, 1.08× faster (GCC 13.3: 1.11×) with 0.41–0.45 last-level-cache misses per matrix entry
  against 0.87–1.18; GCC's fills of 1,000 series run 5–11 % slower. A mapped matrix fills slab by slab, the next slab
  read ahead and the finished one written back: at 45,000 series 22.6× faster in a 4 GiB cgroup and at least 37× in a
  2 GiB one. Bit for bit, but for GCC's squared L2 (the multiply-add caveat above), whose clusterings were unchanged.
- A new mapped `.dtwm` file is not filled with NaN before its fill: its header records the rows written, which the fill
  advances once they are on the disk, so creating the file and reading its NaN back go. On an Intel Core Ultra 9 285
  (GCC 13.3, WSL) the fill of 45,000 series in a 4 GiB cgroup runs «S2_F64»× faster, «S2_F32»× more with
  `--matrix-dtype float32`; the same distances.
- FastCLARA reads its series once: all samples are drawn first and one pass assigns every series to the medoids of
  every sample, where each sample took a pass of its own. Streamed from Parquet, the sampled rows are read once and
  the file once more, its next row groups decoding while the current ones are assigned (before: per sample, a pass of
  the file and a read of its sampled rows and of its medoids' rows): on an Intel Core Ultra 9 285 3.4–3.6× faster
  streaming 4.2 million series of 128 samples (1.25× where the assignment outweighs the decoding), in RAM 1.00–1.86×;
  the same labels, medoids and costs.
- From 2,000 series on, FasterPAM's swap evaluates candidates in parallel blocks: eight consecutive candidates a
  thread, each matrix line read once for the eight, runs of such blocks on every thread against the state at the run's
  start; the accepted swaps, medoids, labels and costs are the sequential sweep's. On an Intel Core Ultra 9 285 at 24
  threads 13.6× faster at 50,000 series (k = 5) under clang and 16.7× under GCC 13.3, 3.6× and 3.7× at k = 200, 3.8×
  at 2,000 series (MSVC 6.3×); 1.1–1.6× on one thread. The silhouette score reads eight points a pass and the Dunn
  index runs row by row on every thread (1.3–1.4× and 30–49× faster at 24 threads); the same values.
- The Windows wheels are built by clang-cl and carry LLVM's OpenMP runtime (`libomp.dll`), so their fill of
  equal-length series runs on the SIMD lanes, which MSVC's cl leaves scalar: on an Intel Core Ultra 9 285
  3.96–4.31× faster than the MSVC wheel with 24 threads, 8.1–8.2× on one thread; results identical.
- The Windows archive's `dtwc_cl.exe` is built by clang-cl too, with MSVC's build beside it as `dtwc_cl_msvc.exe`,
  the fallback: on an Intel Core Ultra 9 285 `dtwc_cl.exe` filled the distance matrices of equal-length series
  4.04–4.18× faster with 24 threads, 8.05–8.14× on one thread.
- The one-pair kernel without a band computes two columns per pass: on an Apple M5 Pro a pair runs 1.14–1.98× faster,
  the 18-thread fill of 1,000 series of lengths 90–110 1.42–1.50× faster; bit for bit with Apple clang.
- No DTW cell calls a library function (the MSVC STL made `std::min({…})` one): on an Intel Core Ultra 9 285, pinned,
  two 1,000-sample series take 1.4 ms, not 7.2 ms (band 100: 0.26 ms, not 1.4 ms); digit for digit.
- `dtwFull` and Soft-DTW keep one column per thread, not an n × m matrix (512 MB for two 8,000-sample series): an
  8-thread Soft-DTW fill of eight such series peaked at 3.4 GB, now 14 MB (Intel Core Ultra 9 285); digit for digit.
- FastPAM's swap reads the matrix directly: 5.6–5.9× faster on an Intel Core Ultra 9 285 (loaded), the same result.
- OneBatchPAM (N = 2,000, k = 10; Intel Core Ultra 9 285, 24 threads, loaded): its final assignment runs on every
  thread (the call 1.82 s → 0.96 s), then its batch table on the SIMD lanes (0.98 s → 0.25 s; the lanes' FMA caveat).
- LR-core's dual runs on OpenMP from 280 series: on an Apple M5 Pro (18 threads), at 800–3,200 series, its subgradient
  root runs 3.5–7.3× and `dtwc_cl -m lrcore` 1.4–4.7× faster, its Kelley root 1.2–1.5× and 1.1–1.5×; bit for bit.
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
- On Windows `dtwc_cl` reads its arguments and paths as UTF-8, and `DataLoader::path` accepts a non-ANSI extension;
  v1.0.0 read arguments in the ANSI code page, so `δ.csv` could name `d.csv`, and threw for such an extension.
- A direct write to the `band` field takes effect at the next `fill_distance_matrix()`; v1.0.0 kept the distances
  computed under the old band.
- The Windows wheels and `dtwc_cl` archive carry the runtime DLLs they import (MSVC's C++ runtime, `msvcp140`, and
  each build's OpenMP runtime: LLVM's `libomp` beside clang-cl's builds, MSVC's `vcomp140` beside
  `dtwc_cl_msvc.exe`), so `import dtwcpp` and `dtwc_cl` run on a machine without the Visual C++ Redistributable.
- FastCLARA streamed from Parquet (`--ram-limit`) checks each row it reads as the in-RAM run checks its series: NaN,
  ±inf, an empty list cell or a row longer than the band allows, outside the samples, is an `InvalidInput` naming the
  row, where the stream published a cost of 1.8e308 (NaN: an error naming a point and a medoid).
- FastCLARA streamed from Parquet held `--ram-limit` only approximately: it decoded a batch of row groups in one
  read, whose buffers Arrow rounds up to a power of two over the batch, beyond what the batch was priced. On Windows
  4.2 million series of 128 doubles streamed under `--ram-limit 1G` peaked at 1.47 GB of private bytes (4G: 5.2 GB).
  Every Parquet read now decodes about 1 MiB of values at a time and the stream is priced by what it holds: the same
  runs peak at 0.70 GB and 2.4 GB. On Linux the stream returns its freed batches to the system before the results
  are written.

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
