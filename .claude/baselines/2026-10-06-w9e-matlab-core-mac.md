# 2026-10-06 — W9e: MATLAB on the C++ core (Mac)

Unit W9e, worktree branch from design-2.0 33302535. Apple M5 Pro, macOS 26.6.2, Apple clang 21.0.0, CMake 4.4.3,
MATLAB R2026a Update 5 (26.1.0.3346908). `build/` = `--preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp`;
`build-matlab/` = the same flags plus `-DDTWC_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/Applications/MATLAB_R2026a.app`
(HiGHS ON as a shared libhighs, Gurobi OFF, Metal ON, testing ON; the DTWC_* cache equals the main checkout's
build-matlab but for three options CMake no longer defines). Counts, sizes, maps and bytes are [confirmed];
wall-clock is [inferred].

## matlab_suite (`ctest --test-dir build-matlab -C Release -R matlab_suite -V`) [confirmed]

| tree | run | passed | failed | incomplete |
|---|---:|---:|---:|---:|
| base 33302535 | 140 | 139 | 0 | 1 |
| 9230bd5e, 1000f8d0, 8dccdc68, bebcb149 | 142 | 141 | 0 | 1 |

The incomplete case is the allowed `test_test_api/test_parallelisation_serial_is_honest` (an OpenMP MEX filters it)
before and after. By name (runtests on tests/matlab, one row per test, base vs after):
- gone: `test_contract_parity/test_dtwclustering_forwards_the_gpu_ordinal` (tested the deleted MATLAB helper
  `apply_device_strategy`; cover: `test_run_resolution.cpp` index_1 on Metal, `test_cli_device_matrix` gpu_index_1,
  `test_hpc_is_a_device_error_as_in_cpp` for DTWClustering's Device reaching C++'s grammar);
  `test_dtwc/test_clustering_constructor_defaults` (MATLAB's copies of the C++ defaults, which went; cover:
  `tests/conformance/config_defaults.toml`, Python's `test_cluster_signature_defaults`);
  `test_tier1_route_parity/test_result_distance_matrix_is_returned_directly` (drove `tier1_cluster`/`Result_*`).
- new: `test_tier1_route_parity/test_result_distance_matrix_is_the_saved_matrix` (the last one, on the public
  surface, plus the onebatch fill), `test_parquet_reads_as_cpp_reads`, `test_arrow_ipc_is_refused_naming_the_format`;
  `test_contract_parity/test_keys_are_python_words_in_camel_case`,
  `test_dtwclustering_predict_and_score_read_the_fitted_medoids`; renamed (bebcb149):
  `test_k_above_n_is_rejected_with_the_cpp_message` -> `test_k_above_n_and_no_series_raise_the_cpp_messages`.
- 140 - 3 + 5 = 142; every other name passed before and after (per-name listing rerun at bebcb149).
- Bite: the four tests bebcb149 extends fail against 8dccdc68's `+dtwc` and MEX (54 run, 4 failed) and pass at
  bebcb149.

By hand (not in matlab_suite; the CI MATLAB job runs them): `tests/conformance/test_conformance.m` 1 run / 1 passed
before and after; `docs/examples/quickstart.m` prints `labels: 0x9 1x9 2x9`, `medoids: 4 13 22`, `mean silhouette:
0.968950` before and after; `examples/matlab/example_quickstart.m` runs before and after, and from bebcb149
prints `Total cost: 1031.54` and medoids `7 12 8` (it printed `NaN` and no medoids: fit_predict on a value class).

## The MEX (`build-matlab/bindings/matlab/dtwc_mex.mexmaca64`) [confirmed]

| | base | 9230bd5e | bebcb149 |
|---|---:|---:|---:|
| bytes | 1,165,776 | 1,125,456 | 1,125,296 |
| libdtwc++.a members loaded (link map) | 22 | 20 | 20 |
| symbols / bytes in the map's symbol table | 3,822 / 790,170 | 3,706 / 761,709 | 3,701 / 762,077 |
| `nm -U` defined symbols | 2,301 | 2,229 | 2,227 |

Members: `run.cpp.o` and `api.cpp.o` go; `read_data.cpp.o` (the C++ text reader), `parse_number.cpp.o`
(fast_float) and `Problem_IO.cpp.o` (the writer) stay; `cli/config.cpp.o` (CLI11, fkYAML), `arrow_c_data.cpp.o`,
`nanoarrow.c.o` and `barycenter.cpp.o` are loaded in neither (this Mac builds no Arrow). Map needles: `CLI` 1 -> 0
symbols (a string literal of run.cpp), `fkyaml` 0 -> 0, `dtwc3run` 2 -> 0. Demangled `nm -U` counts, base -> after:
`dtwc::run(` 2 -> 0, `dtwc::load(` 1 -> 0, `dtwc::cluster(` 2 -> 0, `dtwc::Result::` 1 -> 0, `dtwc::Dataset::` 1 -> 0,
`HandleManager<dtwc::Result>` 1 -> 0, `dtwc::read_data(` 1 -> 1, `dtwc::detail::write_result_files(` 3 -> 3,
`dtwc::detail::default_name(` 1 -> 1, `dtwc::apply(` 1 -> 1, `CLI::` / `fkyaml` / `ArrowArray` / `nanoarrow` 0 -> 0.
The map is a manual re-link of the build's link line with `-Wl,-map,<file>`; the re-linked MEX is byte-identical
to the built one (named `dtwc_mex.mexmaca64`: ld64's ad-hoc signature carries the file name).

## Behaviour [confirmed; the CLI byte check, the refusals and the GPU case rerun at bebcb149]

- `dtwc.cluster(dtwc.load('data/dummy', 'SkipRows', 1, 'SkipCols', 1), 3, 'Method', 'pam')` then `save`: the four
  files are byte-identical to `dtwc_cl -i data/dummy --skip-rows 1 --skip-cols 1 -k 3 --method pam`'s (cost
  148361.91988495924 in both).
- Parquet, `tests/fixtures/fast_clara_streaming_8x4.parquet` (one `list<double>` column): MATLAB's
  `as_series()` equals pyarrow 25.0.1's rows exactly (8 series of 4, `series_0` .. `series_7`); C++ with Arrow was
  not run (this Mac has none). R2026a: `featherread` is undefined (`MATLAB:UndefinedFunction`) and there is no
  `arrow.*` reader, so `.arrow`/`.ipc`/`.feather` are refused (`dtwc:invalidArgument`). Nulls (files written by
  pyarrow): a null list cell reaches MATLAB as `missing` and is refused with the C++ reader's words (1000f8d0); a
  null value reads as NaN (`parquetread` cannot tell it from NaN), where the C++ reader refuses it.
- Advisor's `Metric='squared_euclidean'` + `Device='gpu'`: on Metal (Apple M5 Pro), `DTWClustering('NClusters', 2,
  'Metric', 'squared_euclidean', 'Device', 'gpu').fit(X)` gives Inertia 0.00062188584706746042 against the CPU's
  0.00062187500000003299 (FP32 against FP64), same labels; `dtwc.cluster` reports device `gpu`. The GPU computes;
  fixed by construction (apply sets the metric and the device on one Problem). No case added.
- Refusals: `'max_iter'` is an unknown key naming the 32 valid ones; k 0, `MaxIter` 0, `NInit` 0, `Seed` -1,
  `'Verbose', 1`, an odd pair count, a cell element that is a matrix: `dtwc:invalidArgument`; `Solver gurobi`:
  `dtwc:solverError`; onebatch on `gpu`: `dtwc:deviceError`. `prob.Band = 3`: `MATLAB:class:SetProhibited`.
- Every MATLAB block of `docs/content/getting-started/matlab.md` (6) and the new one of `supported-data.md` ran on
  real files (a CSV with a header line and an id column, a folder of CSVs, the Parquet fixture).

## Gates [confirmed]

- `ctest --test-dir build -C Release -j1 --output-on-failure` at d4c43410 (no C++ changes after it): 95 tests, 100 % passed, one MAY_SKIP
  (`test_cuda_correctness`), 48.08 s; the core is unchanged by this unit (`build/` rebuilt nothing; `dtwc_cl` is
  byte-identical to base).
- `DTWC_CONFORMANCE_REGEN=1 ./build/bin/cpp_conformance` then `git diff tests/conformance/conformance_reference.txt`:
  only `silhouette,0.96894972764334841` -> `0.9689497276433483` (D-19), before and after; the file was restored.
- CLI, the brief's 25 runs on data/dummy with `build/bin/dtwc_cl` (byte-identical to base's): `diff -r` of the
  base and d4c43410 run folders is empty: 156 files (81 output files, and per run stdout without timing, stderr and the
  exit code); 24 runs exit 0, `--band 10` exits 1 (an infeasible band) in both.
- `check_docs.py --cli build/bin/dtwc_cl` PASS (392 flags, 59 pages); `check_pins.py` 0 failures;
  `generate_docs.py --check` current.
- Lint: MATLAB `checkcode` on the changed `.m` files 14 -> 12 messages, none new; `dtwc_mex.cpp` under
  `-Wall -Wextra -Wshadow -Wconversion` (syntax only) 93 -> 91 warnings, no new kind (unused command parameters,
  one pre-existing sign comparison).
- pytest: not run, no Python file changed.
- matlab_suite wall time 98.46 s base, 15.8 to 34.6 s after [inferred: the base run was the session's first
  MATLAB start; not a measured speed-up].

## Review (adversarial agent over 33302535..8dccdc68; each cited line opened)

Fixed in bebcb149: series names and keys pass to C++ as local code page on Windows (now UTF-8); no series gave
"data must not be empty." instead of "cluster: dataset is empty."; a matrix cell under SkipRows/SkipCols was
flattened; transform/predict/score took input fit refuses; parquetread read every column and raised MATLAB ids;
the example's NaN; two doc statements on keys. Not changed: a logical is read as 1 for a numeric key (`'Band',
true`; Python refuses a bool), as MATLAB functions generally read it; the null-series refusal (1000f8d0) has no test
(no tracked Parquet file holds a null).

**Not proven here:** Windows `build/mex` with R2024b (the brief's second matlab_suite), a CUDA MEX, Linux, CI.

## Commands

```sh
cmake -S . -B build-matlab -G Ninja -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_TESTING=ON -DOpenMP_ROOT=/opt/homebrew/opt/libomp \
  -DDTWC_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/Applications/MATLAB_R2026a.app
cmake --build build-matlab && ctest --test-dir build-matlab -C Release -R matlab_suite -V  # map: the link line of `ninja -C build-matlab -t commands bindings/matlab/dtwc_mex.mexmaca64`, output renamed,
# plus -Wl,-map,<dir>/dtwc_mex.map; `nm -U <mex> | c++filt` for the symbol counts
# CLI: the 25 runs on data/dummy (--skip-rows 1 --skip-cols 1, -k 3 unless -k 1), one directory each with out/,
# stdout (the "[m:s min:sec]" tokens and the Time: value stripped), stderr and the exit code; diff -r base new
```

## Merged on the main tree (orchestrator, `bf82dc1b` = `b9839200` (M1 merged) + `pb/W9e`), all `[confirmed]`

`build/`: zero warnings. `ctest -j1`: 100 % of 95, `test_cuda_correctness` skipped, 19.34 s (quiet machine).
Conformance regenerated: the one silhouette ulp only (D-19). `check_docs` PASS, `check_pins` 0 failures,
`generate_docs --check` current. `build-matlab/` (HiGHS static since M1): zero warnings; `dtwc_mex.mexmaca64`
4,516,640 bytes (4,557,072 before W9e on the same static link); `matlab_suite: 142 run, 141 passed, 0 failed,
1 incomplete` (the registered filter). Fresh venv (`.[test,dev,io,mip]`, matplotlib, pandas): **974 passed,
11 skipped, 0 failed, 94.17 s**, as after M1 (W9e changes no Python file).
