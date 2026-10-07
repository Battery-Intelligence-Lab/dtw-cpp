# DTWC++ — MAP

Where things are today (2026-10-07, phase G: W14a and W14b merged, W14c; the "After G" kernel units merged), the
target the phases A–G build (`PLAN.md`), and the invariants a refactor must keep. **Read this instead of the tree.**
No line numbers: grep for the symbol. Sizes are `git ls-files | xargs cat | wc -l`, rounded.

## 1. Pipeline today

```text
text file / folder ──▶ io/read_data (dtwc_core) ───┐
Parquet / Arrow IPC ──▶ io/read_arrow (dtwc_io) ───┴▶ Data ─▶ Problem ─┬─ bind the DTW function once (dtw_dispatch)
                                                                       ├─ fill_distance_matrix: CPU lanes | per pair | CUDA | Metal
                                                                       ├─ dist_by_ind(i, j): an O(1) read ◀── algorithms, MIP, scores
                                                                       └─ cluster() ─▶ ClusteringResult, or a free algorithm function
Tier-1  device() → load() → Dataset → cluster() → run(Config) → Result       CLI  dtwc_cl → cli::bind → Config → run()
Python  _dtwcpp_core + dtwcpp: apply(Config) → Problem::cluster()          MATLAB  dtwc_mex + +dtwc: the same
```

## 2. Where the code is (`dtwc/`)

| Layer | Where | Lines | What |
| --- | --- | --- | --- |
| base | `base/` (+ forwarders at `dtwc/` root), `types/`, `enums/` | 1.1k | errors (`Error` → `InvalidInput`, `DeviceError`, `IOError`, `SolverError`, `UndefinedScore`), settings (`index_t`), OpenMP helpers (`run_openmp`: per-thread failure slots, the lowest-index failure rethrown), `device()` (`env`), names tables, `Index` / `Range` |
| core | `core/`, `warping*.hpp`, `soft_dtw.hpp`, `distance.hpp`, `Data.hpp`, `detail/decode_pair.hpp` | 5.4k | `dtw_kernel.hpp` (linear: two columns per pass; lanes: 128 bytes on AArch64, 64 elsewhere; banded; × Cost × Cell), `dtw_dispatch` (bind once), MSM, TWE, envelopes + LB_Keogh, `DistanceMatrix` (packed, heap or mapped `.dtwm`; llfio only in `distance_matrix.cpp`), SHA-256, portable RNG, `kmedoids_pp` |
| io | `io/`, `DataLoader.hpp`, `fileOperations.hpp`, `core/matrix_io.hpp` | 2.5k | text reader `read_data` (fast_float via `parse_number`), nanoarrow C-Data ingest; in `dtwc_io`: `read_arrow` (Parquet, Arrow IPC), `parquet_schema` (one layout rule), `parquet_chunk_reader` (row groups) |
| backends | `cuda/`, `metal/` | 2.9k | GPU fills; CUDA also FastCLARA's assignment |
| algorithms | `algorithms/`, `initialisation.*`, `scores.*` | 4k | FastPAM, FastCLARA (`fast_clara_parquet.cpp`: the stream, in `dtwc_io`), OneBatchPAM, hierarchical, TADPole, barycenter; seeding; seven scores |
| mip | `mip/` | 1.5k | `mip-solvers`: HiGHS and Gurobi p-median (`build_p_median_model`), LR-core (`lagrangian_root`: dual, reduced-cost fixing, branch and bound) |
| session | `Problem.{hpp,cpp}`, `Problem_IO.cpp`, `checkpoint.*` | 2.3k | the `Problem` session: data, distance binding, matrix cache, clustering state, checkpoints; `apply(Config, Problem&)` |
| surface | `api.*`, `cli/`, `config.hpp`, `dtwc_cl.cpp`, `test_api.hpp`, `dtwc.hpp` | 2k | Tier-1 API; `Config` (`config.hpp`, in `dtwc_core`); `cli::bind` (the one key table), `run()`; diagnostics |
| vendored | `extern/nanoarrow`, `extern/fast_float` | — | not ours; never edited |

Bindings: `python/` (4.9k; `src/_dtwcpp_core.cpp`, `dtwcpp/_api.py`, `_clustering.py`, `_hpc.py` + `_slurm/`, `_mip.py`:
`method="mip"` through highspy), `bindings/matlab/` (3.2k; `dtwc_mex.cpp`, `+dtwc/`). Tests: `tests/` (42k lines).

## 3. Build

- Targets (L2b): static `dtwc_core` (all of the above but `cli/` and `api.cpp`; what the Python module and the MEX
  link), `dtwc_cli` (`cli/`, `api.cpp`; CLI11, fkYAML) and, with Arrow, `dtwc_io` (the Arrow readers and FastCLARA's
  Parquet stream), behind `dtwc++` (INTERFACE, the consumers' link name; headers in `FILE_SET`s based at `dtwc/`);
  `mip-solvers` (object, OpenMP), `dtwc_options` / `dtwc_warnings`, `dtwc_cl` (top-level project only, not under
  scikit-build), examples, benchmarks, `_dtwcpp_core` (nanobind, `NOMINSIZE`), `dtwc_mex`.
- Presets `clang-macos`, `clang-win`, `clang-win-debug`, `msvc`, `gcc-linux`, each into `build/`, tests ON. Mac:
  `build/` (Ninja Release; HiGHS, llfio, Metal, YAML), `build-matlab/` (+ the MEX, `matlab_suite`), `build-asan/`
  (ASan + UBSan). Windows box: `build/` (clang Release, benchmarks; Arrow OFF), `build/arrow-pyarrow-23` (Arrow via its
  `pyarrow-config/` shim; counts only if `ctest -N` lists `test_io_readers`), `build/cuda-verify-0928` (CUDA).
- Options: `DTWC_BUILD_{TESTING,EXAMPLES,BENCHMARK,PYTHON,MATLAB}`, `DTWC_ENABLE_{HIGHS,LLFIO,YAML}` (ON),
  `DTWC_ENABLE_METAL` (ON on Apple only), `DTWC_ENABLE_{GUROBI,ARROW,CUDA}` (OFF); an `ON` that cannot be honoured
  stops the configure. `DTWC_ALLOW_SEQUENTIAL` (OFF: no OpenMP is a configure error), `DTWC_FP_MODEL` (`fast` | `strict`),
  `DTWC_ARCH_LEVEL` (`native` | `v3` | `v4`; `v3` for Python), `DTWC_CUDA_ARCH_LIST` (8.0–9.0), `DTWC_DEV_MODE`.
- Dependencies (`cmake/Dependencies.cmake`, pinned, checked by `check_pins.py`): CLI11, fkYAML, HiGHS, llfio
  (header-only, with pinned quickcpplib and outcome), Arrow, Catch2, Google Benchmark, nanobind (PyPI first); OpenMP,
  Gurobi, CUDA, Metal from the system. FP flags: `-fassociative-math` without `-ffinite-math-only`, on `dtwc_options`.
- macOS needs `brew install libomp` and `-DOpenMP_ROOT=/opt/homebrew/opt/libomp`; Windows CUDA, nvcc 13.0 with a
  supported MSVC host. The wheel links no HiGHS (`mip` extra: highspy); the MEX links it statically; archives and
  wheels are x86-64-v3 or arm64, macOS ≥ 13.3.

## 4. Tests and gates

- Every test is registered through `dtwc_add_test` (`cmake/DtwcTest.cmake`): it passes on Catch2's summary with ≥ 1
  assertion in ≥ 1 case, no failure, and no skip unless `MAY_SKIP`; `REQUIRES` unregisters one whose subject is not built.
- `tests/unit` (flat, plus `core/`, `algorithms/`, `mip/`, `io/`, `types/`), `tests/integration` (real-binary CLI
  scripts driven by `cmake -P`, the deprecated-shim probe), `tests/conformance` (one tracked reference for C++,
  Python, MATLAB and the CLI; `DTWC_CONFORMANCE_REGEN=1` rewrites it), `tests/python` (pytest; not run by ctest),
  `tests/matlab`, `tests/data/reader`, `tests/support` (oracles, `scratch_directory.hpp`), `tests/fixtures`.
- Full run: `ctest --test-dir build -C Release -j1` — 94 on the Mac's `build/` (CUDA's a `MAY_SKIP`); `build-matlab/`
  adds `matlab_suite` (138/138; an Incomplete fails it); pytest from a fresh venv 895 / 11 skipped / 0 (2026-10-07).
- Gates: `scripts/check_docs.py --cli <dtwc_cl>` (both directions: every flag the docs, README and `.claude/commands`
  show is in the live `--help`; every flag the help prints has a row in `getting-started/cli.md`), `check_pins.py`,
  `generate_docs.py --check`; in CI also `check_site_links.py` and gitleaks. Manual: `codegen_report.py`,
  `machine_facts.py`, `smoke_release_archive.py`, `check_ipo_inlining.py`, `repo_map.py`, `run_bench.sh`.
- CI (`.github/workflows/`): ubuntu, windows and macOS unit jobs, documentation (docs gates, Hugo, Doxygen,
  coverage), MATLAB MEX, Python tests and wheels, release artefacts, CUDA configure smoke, JOSS draft.

## 5. Elsewhere

`docs/` Hugo site (`content/`, `derivations/`, `examples/`, `Doxyfile`) · `benchmarks/` · `examples/` (`cpp/`,
`python/`, `matlab/`) · `data/dummy` (series the tests use) · `scripts/slurm/` (HPC transport; the package ships its
own in `dtwcpp/_slurm/`) · `develop/` (contributor-doc sources) · `.claude/commands/` (user slash commands) ·
`.claude/skills/` (`dtwc-verify`, `dtwc-run-benchmarks`, `dtwcpp`, `session-handoff`).

## 6. Target interface (design review §3)

```cpp
dtwc::device("gpu");                                    // cpu | gpu | gpu:N (cuda[:N] alias); "hpc" throws, naming Python and slurm_remote.sh
auto data = dtwc::load("cycles/", 1);                   // skip_cols, skip_rows, delimiter, name; CSV/TSV, folder, Parquet, Arrow
auto res  = dtwc::cluster(data, 8, "auto", 100);        // method, band, device, max_iter; any other key: dtwc::run(Config)
res.labels(); res.medoids(); res.cost(); res.score("silhouette"); res.save("out/");
double d  = dtwc::distance::dtw(x, y, {.variant = DTWVariant::MSM, .msm_c = 0.5});
```

```python
dtwc.device("hpc:gpu")                                   # cpu | gpu | gpu:N | hpc | hpc:gpu
res = dtwc.cluster(dtwc.load("series.parquet"), k=50, band=400)          # kwargs == Config keys
for k in range(3, 9): prob.set_n_clusters(k); prob.cluster()             # Tier 2: a k-sweep fills once
est = dtwc.DTWClustering(n_clusters=3, n_init=3).fit(X)  # the one estimator
```

```sh
dtwc_cl -i cycles/ -k 8 --band 1500 --device gpu -o out        # or --config job.toml; flags beat the file
```

- `Config` is flat, keyed by the CLI long names; `cli::bind` is the one key table; TOML / YAML through CLI11.
  `k` is required, `method = auto` (cpu: `pam` for N ≤ 5000, else `clara`; gpu: `pam`).
- `Result` in every language: `labels, medoids, cost, device, score(name), save(dir), distance_matrix`; C++ adds
  `method, iterations, converged`, Python and MATLAB `plot()`.
- Tier-2 (C++, Python, MATLAB alike): `Problem` with the v1 fields; `set_distance / set_band / set_metric /
  set_variant / set_missing_strategy / set_device / set_gpu_precision` each invalidate the matrix;
  `fill_distance_matrix`, O(1) `dist_by_ind`, `cluster()` over nine `Method` values, `set_result`; algorithms
  `fast_pam, fast_clara, one_batch_pam, tadpole, build_dendrogram / cut_dendrogram, dtw_barycenter,
  barycenter_kmeans` (each language's set on the Tier 2 page); the seven scores. The `warping*.hpp` kernels stay
  public C++ and unchecked.
- Devices: `cpu` (packed matrix in RAM, else the mapped `.dtwm`), `gpu[:N]` (CUDA, else Metal, else
  `DeviceError` naming the build flag and the artefact; never a zero matrix, never a CPU fallback),
  `hpc[:gpu]` in Python and `slurm_remote.sh` only, submitting `job.toml`.
- Persistence: one binary matrix file `.dtwm` (magic, version, N, SHA-256 fingerprint, packed doubles, NaN = not
  computed); the mapped cache is the checkpoint. CSV stays for interchange.

## 7. Invariants — deliberate; do not "clean up"

- **Types, not checks.** Counts of series, clusters and rows, labels, medoids and `dist_by_ind` indices are
  `index_t` (`std::int64_t`); tuning values are `int`; products of counts are `size_t` / `int64_t` by type. No
  runtime guard on a count. The only checks are where HiGHS and Gurobi take `int`. Where we own a loop we chunk
  (CUDA: int64 pair offset) instead of refusing. (`index_t` exists since Y3; public counts move to it in phase D.)
- Cost and Cell are **template parameters**, never a `std::function` or a virtual per cell.
- Orient so `n_short ≤ n_long` before every kernel call; buffers are sized on that.
- `thread_local` scratch grows and never shrinks; a Cost must not re-enter its kernel.
- The dense matrix has **no locks or atomics**; fills partition pairs. `resize()` wipes to NaN, so it stays
  conditional.
- Bind the distance function **once**; every guard lives in the builder, outside the closure (a throw inside
  an OpenMP region is UB). Series travel as spans.
- No `omp critical`, mutex or atomic on a data path (DECISIONS §2 rule 6); region-local `num_threads`, never
  `omp_set_num_threads`.
- No `std::min({…})`, `std::max({…})` or `std::min_element` in a hot loop: the MSVC STL makes them library calls.
  `test_codegen_no_calls` fails a clang build in which any loop of a probe kernel calls anything (Apple clang's
  `memset_pattern16` idiom sat one loop out from the innermost).
- `decode_pair` is the single pair decoder on host and device, with its integer corrections.
- FP model: `-fassociative-math` **without** `-ffinite-math-only`, and the `#error` under finite-math-only. NaN
  means missing or not computed.
- The kernels' `numeric_limits::max()` is the DP's unreachable value and must never leave a kernel as a
  distance; NaN is the only "not a distance" outside kernels.
- `find_best_swap` and the FasterPAM sweep are sequential on purpose; the nearest-medoid scans stay separate at
  run time (a compile-time template collapse with identical code is allowed).
- Parquet planning is metadata-first; argv beats the config file; an unknown key is an error.
- `portable_random`: the same seed gives the same result on every standard library.
- Conventions: local cost L1, band in integer cells, no final square root (`DECISIONS.md` §2).
