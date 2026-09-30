# 2026-09-30 — F1: every C++ test writes to its own directory

Question: which C++ tests write a path that another test, or another process running the same test,
also writes, and does one shared `tests/support/scratch_directory.hpp` remove every such path?

Branch pb/F1, base a931bd7. Windows 11, clang 21.1.8, Release, HiGHS/Gurobi/llfio ON, no Arrow
(`build/`); the Arrow build (`build-arrow`, Arrow 23.0.1 through the pyarrow shim) is a second tree.
The machine was shared: counts of failures are evidence, seconds are not.

## Inventory at the base [confirmed]

Fixed names under `%TEMP%` (`git grep` at a931bd7): test_dense_distance_matrix_adversarial:262,
test_config_spellings:191/411/461, test_fill_request:136, test_io_readers:74, test_metal_mmap:99/135,
test_run_resolution:236 (a path only read), unit_test_DataLoader:228, unit_test_checkpoint:52,
unit_test_cli_args:44, unit_test_cli_checkpoint:162/193/216/241, unit_test_fileOperations:215/472/604/
831/865/896, unit_test_variant_distmat:259/289 (`%TEMP%/dtwc_test`, whose cleanup removed the whole
directory), unit_test_mmap_distance_matrix:55, unit_test_wave1a/2a/2b (a shared output directory,
never written), test_cuda_correctness:1452 (W4d's file, left).

Names in the working directory (the repo root under ctest): unit_test_fileOperations:149 `test_distmat.csv`,
:267 `test_matrix.csv/.tsv`, :442 folder `CSV`.

The temp root itself as `set_output_folder`: test_fast_pam_adversarial:46, unit_test_clustering_algorithms:57
(nothing is written there: cluster() writes no files).

Suffixed by an object address or a clock (unique in practice, not per process, not removed if an assertion
throws first): test_error_taxonomy, test_io_error_contract, test_problem_loud_contracts, test_problem_metric,
unit_test_invalid_distance_enums, unit_test_problem_semantic_transactions, unit_test_dtw_function_semantics,
unit_test_variant_precision, unit_test_variant_distmat (ScratchCache), unit_test_fileOperations (batch file,
UTF-8 folder), test_tier1_cpp_api, unit_test_Problem, unit_test_Problem_phase0,
unit_test_clustering_algorithms (capped Lloyd).

Not touched: the three `FIXTURE_ROOT` tests (test_problem_api_2_0, unit_test_distance_matrix_csv,
unit_test_problem_encapsulation) and the `cmake -P` CLI tests use one directory each inside the build tree
(`WORK_ROOT`, the fixture root, which also redirects TEMP for the process). Two ctest runs of different
build trees do not meet there; two runs of the same build tree still would.

## Base measurements [confirmed]

- `ctest -j1`: 122 tests, 119 passed, 3 skipped (test_cuda_correctness, test_metal_correctness,
  test_metal_mmap), 0 failed (`F1-base-ctest.log`). The run left five directories in TEMP
  (dtwc_distmat_roundtrip_test, dtwc_read_distmat_failure_test, dtwc_wave1a/2a/2b_*).
- `ctest -j 8`, three times, one TEMP: 122 tests each, 0 failed each. Distinct tests never chose the
  same name, so `-j 8` alone does not expose the collision.
- The collision needs the same test twice. Harness below: 4 copies of one test binary at once from the repo
  root, one TEMP, 5 rounds. Failed processes of 20 at the base: unit_test_checkpoint 20,
  unit_test_cli_checkpoint 20, unit_test_fileOperations 20 (11 of them segfaults), unit_test_variant_distmat 18,
  test_config_spellings 15, unit_test_cli_args 15, unit_test_mmap_distance_matrix 14,
  unit_test_DataLoader 11, test_fill_request 6, test_dense_distance_matrix_adversarial 4; 0 of 20 for the
  other 17 binaries (address- or clock-suffixed, or nothing written).
- Control: one copy at a time, 3 rounds: 0 failed for those ten binaries. (unit_test_mmap_distance_matrix
  failed 3 of 3 first: the test asserts `dtwc_mmap_test_write_self.dtwm.tmp` is absent and the file was in
  TEMP after the collision runs [inferred: left by a copy that died in the collision]; after deleting it
  the test passed 3 of 3. A collision can leave state that fails later runs.)

## Head [confirmed]

- `ctest -j1`: 122 tests, 119 passed, 3 skipped (the same three), 0 failed; the list of test names is
  identical to the base; nothing left in TEMP or in the source tree.
- `ctest -j 8`, three times, one TEMP: 0 failed each, the same 3 skipped.
- The same harness: 0 failed of 20 for all 27 binaries.
- `git grep` for `temp_directory_path`, `dtwc_test`, `nonce`, `uintptr_t>(this` in tests/ (C++): the helper
  and test_cuda_correctness:1452 only.
- No library file changed; `cpp_conformance` passes against the unchanged reference.
- Arrow tree (`build-arrow`, configured with both `-DArrow_DIR` and `-DParquet_DIR` at the pyarrow shim;
  with `Arrow_DIR` alone `DTWC_HAS_PARQUET` is not defined and test_io_readers skips its Parquet case;
  the DLLs need `pyarrow` and `pyarrow.libs` on PATH): `ctest -j1` 125 tests, 122 passed, 3 skipped (the
  same three), 0 failed; test_io_readers, test_error_taxonomy, test_io_error_contract and the two
  fast-clara Parquet script tests passed. No base run of this tree.

## Harness

```sh
# collide.sh <build_dir> <copies> <rounds> <test>...   (Git Bash; TEMP is one shared directory)
export TMP='C:\D\git\wt\tmp\F1' TEMP='C:\D\git\wt\tmp\F1'
cd "$(dirname "$1")"
for t in "${@:4}"; do fail=0; total=0
  for r in $(seq $3); do pids=()
    for c in $(seq $2); do "$1/bin/$t.exe" > /dev/null 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait $p || fail=$((fail+1)); total=$((total+1)); done
  done; echo "$t: $fail failed of $total"; done
```
