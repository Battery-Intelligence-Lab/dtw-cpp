# 2026-10-06 — W9f: Python test and example trims (Mac)

Unit W9f, branch pb/W9f from b36fad43 (design-2.0). Apple M5 Pro, macOS 26.6.2, Apple clang, MATLAB R2026a.
`build/` = `--preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp -DDTWC_BUILD_EXAMPLES=ON`;
`build-matlab/` = the W9e recipe (`-DDTWC_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/Applications/MATLAB_R2026a.app`).
Wheels from fresh venvs (`uv venv --python 3.12`, extras `.[test,dev,io,mip]` + matplotlib, `DTWC_REQUIRE_HIGHSPY=1`,
`DTWC_CL_PATH` = the tree's `dtwc_cl`); base = a `git archive b36fad43` snapshot with the stable base `dtwc_cl`.
Other agents built on the machine: no timing claim. Counts and names are [confirmed] (run here).

Rule (runbook PR checklist; Volkan 09-30, "delete the trivial tests"): each kept test pins a named contract against
an independent oracle; each removed case is listed below with the contract it pinned and what still covers it.

## pytest, reconciled by id [confirmed]

| tree | ids | passed | skipped | failed |
|---|---:|---:|---:|---:|
| base b36fad43 | 934 | 922 | 12 | 0 |
| pb/W9f | 901 | 889 | 12 | 0 |

934 − 46 removed + 13 added = 901; 24 ids moved file (`test_clustering_semantics.py::X` →
`test_sklearn_estimator.py::X`, same names); no kept id changed outcome. The 12 skips are the base's, by id (9 CUDA,
the GPU-absent device case, the scipy-absent case, the pandas form: pandas is not in the recipe's extras). The
first after-run (before the contract rows came back) was 898 / 886 / 12 / 0; the table's row is the run at
b5ee6515, after the review's fixes, identical by id and outcome to the run before them.

Added: `test_examples.py::test_example_runs[01_quickstart.py … 09_device_clustering.py]` (9, all passed, none
skipped: pyarrow and matplotlib installed); `test_contract_parity.py::test_the_diagnostics_return_the_cpp_report_fields`;
and, restored as before W9b (handoff 10-06: W9b dropped them with the methods and bound the methods again without
them), `test_contract_name_exists[Problem.read_distance_matrix]`, `[Problem.print_distance_matrix]` and
`test_cpp_file_failure_raises_io_error`. Bites: `01_quickstart.py` with a trailing `raise SystemExit(3)` fails its
case, restored passes.

### Removed ids, the contract each pinned, and its cover

`tests/python/test_api.py` (38):

| id | pinned | cover |
|---|---|---|
| TestLoad::test_load_array_returns_dataset | `load(array)` is a `Dataset`, not a path | nothing contractual (type check); the round trip: TestRaggedInMemorySource::test_as_series_preserves_variable_lengths |
| TestLoad::test_as_series_materializes_array | in-memory round trip | TestRaggedInMemorySource::test_as_series_preserves_variable_lengths (the same, ragged) |
| TestLoad::test_skip_rows_drops_leading_series_in_memory | in-memory `skip_rows` (Python's `Dataset`) | TestRaggedInMemorySource::test_skip_rows_and_skip_cols_apply_to_ragged_rows |
| TestLoad::test_reader_errors_name_the_load_and_keep_their_type[None], [1,2,x\n4,5,6\n] | `IOError` "load: failed to read '<path>': " | test_io_error_contract.cpp:68-82 (the same two files, prefix and type) and :111 (`read_data`, Python's route); test_io.py corpus (type is exactly `dtwcpp.IOError`) |
| TestLoad::test_path_source_parses_a_non_numeric_id_column | the C++ reader drops text fields before parsing | unit_test_fileOperations.cpp:359; forwarding of `skip_cols`: TestLoad::test_skip_rows_drops_leading_file_lines, test_io.py corpus `folder_two_column` |
| TestLoad::test_path_source_supports_ragged_rows | the C++ reader keeps variable-length rows | unit_test_fileOperations.cpp:262 (ragged random data); test_io.py corpus (Python hands the file to that reader) |
| TestLoad::test_in_memory_source_honours_skip_cols | in-memory `skip_cols` | TestRaggedInMemorySource::test_skip_rows_and_skip_cols_apply_to_ragged_rows |
| TestLoad::test_in_memory_skip_cols_beyond_series_length_is_rejected | its message | TestRaggedInMemorySource::test_skip_cols_beyond_a_ragged_series_is_rejected (same message); test_typed_errors.py::TestSkipColsWiderThanARow |
| TestClusterKeywords::test_valid_minimum_numpy_integers_run_locally | NumPy integers for `k`, `max_iter`, `skip_cols` | test_hpc.py::TestJobToml::test_a_path_names_a_file_on_the_cluster (np.int64 through the same `_config`/`_set_key` and `Dataset` checks, values exact) |
| TestClusterLocal::test_k_above_series_count_is_rejected | k > N message, local | test_index_types.py::test_counts_that_only_size_things_take_64_bits; test_run_resolution.cpp:303 |
| TestClusterLocal::test_recovers_two_groups | two groups recovered | TestResultWriteBackInCpp::test_cluster_tier1_end_to_end_results_visible (the same data, the same split, plus a score) |
| TestClusterLocal::test_result_fields_populated | fields not `None`; `device == "cpu"` | `device`: that assertion moved into TestResultWriteBackInCpp::test_cluster_tier1_end_to_end_results_visible (review: no other test pinned a local `Result.device`); the rest existence, values in the seed tests (cost, medoids) and TestMatrixFreeScoring (matrix) |
| TestClusterLocal::test_summary_contains_device_and_timing | `summary()` text | nothing contractual (not a §1.4 member); `examples/python/09` prints it under test_examples |
| TestClusterLocal::test_uses_global_device | `device == "cpu"` after `device("cpu")` | the assertion above; test_device.py::TestGlobalDevice |
| TestClusterLocal::test_accepts_raw_array_without_explicit_load | `cluster(ndarray)` | test_already_read_data_goes_in_as_it_is[2-D array] |
| TestMatrixFreeScoring::test_unknown_score_still_rejected_after_matrix_free_run | unknown score name | the type: test_tier1_cpp_api.cpp:137 (`scores::score`, which Python's `Result.score` calls), test_contract_parity.m:388; the text "unknown score" (scores.cpp:498, undocumented) is matched nowhere now |
| TestSaveUndefinedSilhouette::test_undefined_score_is_a_bound_leaf_under_invalid_input | `issubclass` | test_contract_parity.py::test_error_hierarchy |
| TestSaveUndefinedSilhouette::test_silhouette_of_one_cluster_raises_undefined_score | one cluster raises `UndefinedScore`, "at least 2 non-empty" | TestSaveUndefinedSilhouette::test_score_silhouette_still_raises_undefined_score (the same `scores::silhouette` and translator; it now matches the message, scores.cpp:77); unit_test_Problem.cpp:94 |
| TestRaggedInMemorySource::test_cluster_runs_on_a_ragged_list | `len(labels)` | TestRaggedInMemorySource::test_labels_match_the_cpp_path_on_the_same_ragged_data (the same call, labels equal to `fast_pam`'s) |
| TestSeriesNames::test_batch_file_names_are_the_loader_row_numbers | names 1..N | unit_test_fileOperations.cpp:286; TestLoad::test_parquet_path_reads_like_the_same_csv (`["1", "2", "3"]`) |
| TestSeriesNames::test_folder_names_are_file_stems | file stems | unit_test_fileOperations.cpp:601 (two stems in order, through the `DataLoader` `read_data` uses); TestNonAsciiSeriesNames::test_load_decodes_a_non_ascii_file_stem |
| TestSeriesNames::test_in_memory_names_are_the_zero_based_ordinals | "0", "1", ... | test_already_read_data_goes_in_as_it_is (`load(data).series_names()`) |
| TestSeriesNames::test_saved_labels_carry_the_file_names | names in the labels CSV | TestSeriesNames::test_save_is_byte_identical_to_the_cli (CI sets `DTWC_CL_PATH`, so it cannot skip there) |
| TestPlot::test_plot_writes_png | `plot()` writes a PNG | TestMatrixFreeScoring::test_plot_works_after_a_matrix_free_cpu_run (the same call and assertion) |
| TestClusterMethodDispatch::test_unknown_method_raises | unknown method refused | TestClusterKeywords::test_unknown_method_still_fails_before_load_or_device; test_names.cpp:149 |
| TestClusterMethodDispatch::test_unknown_method_rejected_before_hpc_offload | a bad method never reaches SLURM (mock) | the same test: the method is refused before the device is resolved, so before the hpc branch; test_hpc.py::TestJobToml::test_a_value_cpp_refuses_fails_before_anything_is_written |
| TestClusterMethodDispatch::test_local_clara_runs_end_to_end | clara dispatch | TestMatrixFreeBand[clara] (clara's cost oracle), TestMatrixFreeScoring[clara]; the partition: unit_test_fast_clara.cpp. Its `_distance_matrix is None` could not fail: the cache is `None` after every `cluster()` until the property is read (`_api.py:198`) |
| TestClusterMethodDispatch::test_documented_methods_accepted_and_forwarded_to_hpc[auto, pam, clara, kmedoids, mip, hierarchical] | each name reaches the transport (mock) | test_names.cpp:93 (the nine names and the aliases); test_hpc.py::TestJobToml (the canonical name crosses into job.toml) |
| TestClusterMethodDispatch::test_hclust_alias_normalizes_to_hierarchical | `hclust` is `hierarchical` (mock) | test_names.cpp:98; test_hpc.py::TestJobToml::test_a_path_names_a_file_on_the_cluster |
| TestResultWriteBackInCpp::test_fast_pam_writes_back_without_wrapper, …fast_clara…, …cut_dendrogram… | the algorithms write labels/medoids (and k) to the Problem | test_index_types.py::test_clustering_result_and_problem_arrays_are_int64[fast_pam, fast_clara, cut_dendrogram]; test_problem_api_2_0.cpp:485; k: the same `Problem::set_result` call (Problem.cpp:362; fast_pam.cpp:274, fast_clara.cpp:315, hierarchical.cpp:269) |

`tests/python/test_hpc.py` (3): TestWriteSeriesTSV::test_writes_one_row_per_series and ::test_roundtrips_through_loadtxt
(cover: ::test_values_keep_full_precision, the exact round trip; TestJobToml::test_series_in_memory_travel_as_input_tsv,
the exact bytes); TestParseLabelsCSV::test_maps_one_based_names_to_input_order (cover: ::test_robust_to_lexical_row_order,
the same mapping with scrambled rows). The bash-driven transport cases all stay.

`tests/python/test_sklearn_estimator.py` (2): test_raw_fit_predict_transform_and_score (cover:
test_score_reads_the_fitted_medoids_and_never_refits, score = −Σ min transform; test_training_predict_matches_configured_problem_nearest,
predict = labels_ = nearest medoid; test_fit_computes_every_metric_cpp_computes, inertia = Σ distance; check_estimator);
test_onebatch_raw_mode_and_sklearn_clone_contract (cover: check_estimator's clone and get_params checks,
test_precomputed_native_pairwise_tag; onebatch runs the same `Problem::cluster()` as test_api.py's TestMatrixFreeBand/Scoring[onebatch]).

`tests/python/test_clustering_semantics.py` (1, the file merged into test_sklearn_estimator.py):
test_default_cpu_l1_keeps_lazy_matrix_path — a tripwire on `dtwcpp.compute_distance_matrix`, which `_clustering.py`
never looks up (pins nothing); its L1 result: test_contract_parity.py::test_dtwclustering_constructor_param_set (metric
default `l1`) and test_default_seed_matches_tier1_seed_contract.

`tests/python/test_test_api.py` (2): test_parallelisation_schema_and_engagement, test_gpu_schema_and_validation_or_reason
(cover: tests/unit/test_test_api.cpp, the probes' owner; `_wheel_smoke.run` under test_wheel_smoke.py, available and
threads_engaged ≥ 2; the binding's own part, the dict keys: test_contract_parity.py::test_the_diagnostics_return_the_cpp_report_fields).

## MATLAB, tests/matlab by name (runtests, the worktree's MEX) [confirmed]

| tree | run | passed | failed | incomplete |
|---|---:|---:|---:|---:|
| base tests | 142 | 141 | 0 | 1 (`test_test_api/test_parallelisation_serial_is_honest`, filtered on an OpenMP MEX) |
| pb/W9f | 136 | 136 | 0 | 0 |

Gone: the eight `test_test_api/*` (the probes' behaviour is test_test_api.cpp's; the CI MEX job keeps its
`dtwc_mex('test_parallelisation')` assertion). Moved: `test_version_matches_ssot` → `test_contract_parity`. Added
(after the review, below): `test_contract_parity/test_diagnostics_return_the_cpp_report_fields`, MATLAB's twin of
the Python field line, since the MEX also copies both reports by hand (dtwc_mex.cpp:896, 911). 142 − 8 + 2 = 136, all
passed by name. `ctest --test-dir build-matlab -R matlab_suite -V`, the gate now without the allowed-incomplete pair
(it named only the deleted file's cases): "136 run, 136 passed, 0 failed, 0 incomplete", passed. Bite: the gate's
expression over a folder of one passed and one filtered test prints "2 run, 1 passed, 0 failed, 1 incomplete" and
fails ("matlab_suite gate failed", exit 1).

MATLAB's `verifyEqual(r.available, dtwc.gpu_available())` was the one check, in any language, that the probe does
not call a present GPU unavailable; it moved to the owner, test_test_api.cpp
(`REQUIRE(r.available == dtwc::gpu_available())`, "true == true" on this Mac's Metal).

## Review [confirmed where opened]

An adversarial reviewer (read-only) opened each removed test and its cited cover. Acted on: a local
`Result.device` was no longer pinned (the assertion moved into the kept Tier-1 test); the MEX field names and the
GPU cross-check (above); the `UndefinedScore` message (now matched by the kept test); three citations in the table.
Left: the "unknown score" text (undocumented; the type stays pinned in C++ and MATLAB); `score(X) == −inertia_` on
the training set (it follows from the kept score, nearest-medoid and cost pins); the Python dict values beyond the
keys (the brief allows one Python line; the wheel smoke checks available and threads_engaged).

## ctest, `build/` serial (`-j1`) [confirmed]

Base names (b36fad43, the same configuration): 95. After: 96 = the 95 + `example_tier1`; 100 % passed,
`test_cuda_correctness` skipped as at base, before and again after the review's line in test_test_api.cpp (zero
build warnings). Bite: `example_tier1` with its check inverted fails ("Required regular
expression not found"), restored passes.

## Examples [confirmed]

All nine `examples/python/*.py` exit 0 against the base wheel and the new one (09 with `cpu`: its default is the GPU).
`03_clustering_evaluation.py` printed `Medoid names: N/A` (`Problem.get_name` is not in the 2.0 binding); it now
prints `['s26', 's1', 's11']`. `example_tier1` prints `medoids: 4 1`, `cost: 1.2`, `mean silhouette: 0.98642`.
The other C++ examples build at base and after.

## Lines (`git diff --numstat b36fad43`) [confirmed]

test_api.py 836 → 551; test_hpc.py 1,243 → 1,222; test_sklearn_estimator.py + test_clustering_semantics.py
139 + 266 → 360; test_test_api.py 66 → 0; test_test_api.m 146 → 0 (test_contract_parity.m +16: the version
check and the field line); test_contract_parity.py 237 → 251; test_examples.py 0 → 31; test_test_api.cpp +2;
example_new_features.cpp 213 → tier1.cpp 41. The unit without `.claude/`: +379 / −1,043 in 17 files.

## Gates [confirmed]

`check_docs.py --cli build/bin/dtwc_cl` PASS (397 flags, undocumented 0); `check_pins.py` 0 failures;
`generate_docs.py --check` current; `ruff check --select F,E9` on the touched Python files clean.
