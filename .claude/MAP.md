# DTWC++ — MAP

Where things are today (2026-09-29, phase B: X1, Z1, Y1, K1, Y2, Y3 merged), the target the phases A–G build
(`PLAN.md`), and the invariants a refactor must keep. **Read this instead of the tree.** No line numbers:
grep for the symbol. Sizes are `git ls-files | xargs cat | wc -l`, rounded.

## 1. Pipeline today

```text
file / folder / Parquet / Arrow ─▶ DataLoader, io/ readers ─▶ Data ─▶ Problem ─┬─ bind the DTW function once
                                                                               ├─ fill_distance_matrix: CPU brute force | CUDA | Metal
                                                                               ├─ dist_by_ind(i, j)  ◀── algorithms, MIP, scores, init
                                                                               └─ cluster() or a free algorithm function ─▶ labels, medoids
Tier-1  device() → load() → Dataset → cluster() → Result          CLI  dtwc_cl → cli::run(Config)
Python  nanobind _dtwcpp_core + dtwcpp package                     MATLAB  dtwc_mex + +dtwc package
```

## 2. Where the code is (`dtwc/`)

| Layer | Where | Lines | What |
| --- | --- | --- | --- |
| base | `base/` (+ forwarders at `dtwc/` root), `types/`, `enums/` | 1.7k | errors (`Error` → `InvalidInput`, `DeviceError`, `IOError`, `SolverError`, `UndefinedScore`), settings (`index_t`), OpenMP helpers (`run_openmp`: per-thread failure slots), `device()`, names tables, `Index` / `Range` |
| core | `core/`, `warping*.hpp`, `soft_dtw.hpp`, `distance.hpp`, `Data.hpp`, `detail/decode_pair.hpp` | 9k | `dtw_kernel.hpp` (full, linear, lanes, banded recurrences × Cost × Cell), `dtw_dispatch` (bind once), MSM, TWE, envelopes + LB_Keogh, `DistanceMatrix` (packed, heap or mapped `.dtwm`; llfio only in `distance_matrix.cpp`), SHA-256, portable RNG |
| io | `io/`, `DataLoader.hpp`, `fileOperations.hpp`, `core/matrix_io.hpp` | 2.6k | CSV/TSV/folder text readers (fast_float via `io/parse_number`), Parquet eager + chunked, Arrow IPC, nanoarrow C-Data ingest |
| backends | `cuda/`, `metal/` | 4k | GPU fills (MPI deleted, Z1) |
| algorithms | `algorithms/`, `initialisation.*`, `scores.*` | 4.9k | FastPAM, FastCLARA, OneBatchPAM, CLARANS, hierarchical, TADPole, barycenter; seeding; seven scores |
| mip | `mip/` | 3k | HiGHS and Gurobi p-median, LR-core (`lagrangian_root`, `reduced_cost_fixing`), Benders, PDLP |
| session | `Problem.{hpp,cpp}`, `Problem_IO.cpp`, `checkpoint.*` | 4k | the `Problem` session: data, distance binding, matrix cache, clustering state, checkpoints |
| surface | `api.*`, `cli/`, `dtwc_cl.cpp`, `test_api.hpp`, `dtwc.hpp` | 2.5k | Tier-1 API; `cli::bind` (the one key table), `Config`, `run()`; diagnostics |
| vendored | `extern/nanoarrow`, `extern/fast_float` | — | not ours; never edited |

Bindings: `python/` (6.7k; `src/_dtwcpp_core.cpp`, `dtwcpp/_api.py`, `_clustering.py`, `_hpc.py` + `_slurm/`),
`bindings/matlab/` (4.7k; `dtwc_mex.cpp`, `+dtwc/`). Tests: `tests/` (75k lines).

## 3. Build

- Targets: `dtwc++` (static library), `mip-solvers` (object), `dtwc_options` / `dtwc_warnings` (interface),
  `dtwc_cl` (installed CLI), examples, benchmarks, `_dtwcpp_core` (Python), `dtwc_mex`.
- Presets: `clang-win`, `clang-win-debug`, `msvc`, `gcc-linux`, `clang-macos`. `build/` on the Windows box is
  clang + Ninja Release with HiGHS, Gurobi, llfio and benchmarks; Arrow is ON but not found there, so
  `test_io_readers` is not registered (`build/arrow-pyarrow-23` has it, through the shim in its `pyarrow-config/`).
  `build/cuda-verify-0928` is the CUDA dir.
- Options: `DTWC_BUILD_{TESTING,EXAMPLES,BENCHMARK,PYTHON,MATLAB}`, `DTWC_ENABLE_{HIGHS,GUROBI,LLFIO,YAML,METAL}`
  (ON), `DTWC_ENABLE_{ARROW,CUDA}` (OFF), `DTWC_ALLOW_SEQUENTIAL` (OFF: no OpenMP is a configure error),
  `DTWC_FP_MODEL` (`fast` | `strict`), `DTWC_ENABLE_NATIVE_ARCH`, `DTWC_DEV_MODE`.
- Dependencies (`cmake/Dependencies.cmake`, all pinned to a commit or SHA, checked by `check_pins.py`): CLI11,
  fkYAML, HiGHS, llfio + quickcpplib, Arrow, Catch2, Google Benchmark, nanobind (PyPI first); OpenMP, Gurobi,
  CUDA and Metal from the system. The FP flags are `-fassociative-math` without `-ffinite-math-only`, on
  `dtwc_options` only.
- Per platform: macOS needs `brew install libomp` and `-DOpenMP_ROOT=/opt/homebrew/opt/libomp`
  (`baselines/2026-09-21-macos-first-baseline.md`); Windows CUDA needs nvcc 13.0 with a supported MSVC host;
  MATLAB and Python are built only when asked.

## 4. Tests and gates

- Every test is registered through `dtwc_add_test` (`cmake/DtwcTest.cmake`): it passes on Catch2's summary
  with ≥ 1 assertion in ≥ 1 case, no failure, and no skip unless `MAY_SKIP`; `REQUIRES` unregisters a test
  whose subject is not built.
- `tests/unit` (flat, plus `core/`, `algorithms/`, `mip/`, `io/`, `types/`, `adversarial/`), `tests/integration`
  (real-binary CLI scripts driven by `cmake -P`, the deprecated-shim compile probe), `tests/conformance`
  (one tracked reference for C++, Python, MATLAB and the CLI; `DTWC_CONFORMANCE_REGEN=1` rewrites it),
  `tests/python` (pytest; not run by ctest), `tests/matlab` (`matlab_suite`), `tests/data/reader` (reader inputs).
- Full run: `ctest --test-dir build -C Release -j1 --output-on-failure` — 125 tests, 3 `MAY_SKIP` (CUDA,
  Metal) on the Windows box (2026-09-29, after Y3).
- Gates: `scripts/check_docs.py --cli <dtwc_cl>` (every flag the docs, README and `.claude/commands` show is
  in the live `--help`; the harness still fails an unregistered skip), `scripts/check_pins.py`,
  `scripts/generate_docs.py --check`, gitleaks in CI. Manual tools: `codegen_report.py`, `machine_facts.py`,
  `smoke_release_archive.py`, `check_ipo_inlining.py`, `repo_map.py`, `run_bench.sh`.
- CI (`.github/workflows/`): ubuntu, windows and macOS unit jobs, documentation (docs gates, Hugo, Doxygen),
  MATLAB MEX, Python tests and wheels, release artefacts, CUDA/MPI configure smoke, JOSS draft.

## 5. Elsewhere

`docs/` Hugo site (`content/`, `derivations/`, `api-contract-2.0.md` until phase G, `Doxyfile`) · `benchmarks/`
· `examples/` (C++, Python, MATLAB) · `data/dummy` (sample series the tests use) · `scripts/slurm/` (HPC
transport) · `develop/` (contributor-doc sources) · `.claude/commands/` (user slash commands) ·
`.claude/skills/` (`dtwc-verify`, `dtwc-run-benchmarks`, `dtwcpp`, `session-handoff`).

## 6. Target interface (design review §3)

```cpp
dtwc::device("gpu");                                    // cpu | gpu | gpu:N (cuda[:N] alias); "hpc" throws, naming Python/CLI
auto data = dtwc::load("cycles/", {.skip_cols = 1});    // CSV/TSV, folder, Parquet, Arrow by extension
auto res  = dtwc::cluster(data, 8, {.band = 100});      // any Config key; method auto, seed 42
res.labels(); res.medoids(); res.cost(); res.score("silhouette"); res.save("out/");
double d  = dtwc::distance::dtw(x, y, {.variant = DTWVariant::MSM, .msm_c = 0.5});
```

```python
dtwc.device("hpc:gpu")                                   # cpu | gpu | gpu:N | hpc | hpc:gpu
res = dtwc.cluster(dtwc.load("series.parquet"), k=50, band=400)          # kwargs == Config keys
for k in range(3, 9): dtwc.cluster(data, k=k, dist_matrix=res.distance_matrix)   # a k-sweep fills once
est = dtwc.DTWClustering(n_clusters=3, n_init=3).fit(X)  # the one estimator
```

```sh
dtwc_cl -i cycles/ -k 8 --band 1500 --device gpu -o out        # or --config job.toml; flags beat the file
```

- `Config` is flat, keyed by the CLI long names; `cli::bind` is the one key table; TOML / YAML through CLI11.
  `k` is required, `method = auto` (cpu: `pam` for N ≤ 5000, else `clara`; gpu: `pam`).
- `Result` in every language: `labels, medoids, cost, method, iterations, converged, device, config,
  score(name), save(dir), distance_matrix`; Python and MATLAB add `plot()`.
- Tier-2 (C++, Python, MATLAB alike): `Problem` with the v1 fields; `set_distance / set_band / set_metric /
  set_variant / set_missing_strategy / set_device / set_gpu_precision` each invalidate the matrix;
  `fill_distance_matrix`, O(1) `dist_by_ind`, `cluster()` over nine `Method` values, `set_result`; algorithms
  `fast_pam, fast_clara, one_batch_pam, tadpole, build_dendrogram / cut_dendrogram, dtw_barycenter,
  barycenter_kmeans`; the seven scores. The `warping*.hpp` kernels stay public C++ and unchecked.
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
  `test_codegen_no_calls` fails a clang build whose DP inner loop calls anything.
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
