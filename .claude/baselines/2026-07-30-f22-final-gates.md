# F22 final serial full-gate adjudication

Date: 2026-07-30 (Europe/London)

## Scope and immutable prior verdict

Current clean HEAD before execution:

```text
e0df96ea09d5f7c7a6dad620c399ec12edbcbb32
```

This run completes only the already-registered F22 full-gate adjudication.
It does not rerun or reinterpret the exhausted C++ mutation campaign.
The F22 C++ mutation band remains **FALSIFIED at 33/46** after both permitted
attempts, exactly as recorded in
`.claude/baselines/2026-07-29-f22-deprecation-policy.md`.

## Registered bands

These bands are copied unchanged from the 2026-07-29 F22 record and
`AGENTS.md` before any decisive run:

- canonical `build/highs-1151`: 122/122, zero failed, exactly six capability
  skips (`test_cuda_correctness`, `test_cuda_lb_keogh`, `test_io_readers`,
  `test_metal_correctness`, `test_metal_lb_keogh`, `test_metal_mmap`);
- llfio-OFF `build/nollfio`: 122/122, zero failed, exactly nine capability
  skips (the six above plus `unit_test_mmap_data_store`,
  `unit_test_mmap_distance_matrix`, and `unit_test_benders`);
- Arrow-ON `build/arrow-pyarrow-23`: 124/124, zero failed, exactly eight
  capability skips, with `test_io_readers` executing 390 assertions in
  11 cases and no skip marker;
- Python focused F22: 18/18, the exact
  `F22_PYTHON_GATE ... verdict=PASS` marker, a forced-fresh built/imported
  extension identity, and no skip;
- Python full: 1,041 collected and either
  `1028 passed, 12 skipped, 1 failed` with the sole F39 inventory failure, or
  the arithmetically equivalent `1029 passed, 12 skipped` if F39 has closed;
- MATLAB focused F22: 1/1 on both R2024b and R2025b through the same
  forced-fresh OpenMP MEX, exact F22 marker, no skip;
- MATLAB full on each release: 85 total, 82 passed, two failed, three
  incomplete; the failed names are exactly the two retained F18 cases and the
  only additional incomplete name is exactly
  `test_test_api/test_parallelisation_serial_is_honest`;
- every native matrix and MATLAB release runs serially. No configured matrices
  run concurrently because their tests share source-root artifacts.

## Adversarial pre-run checks

The following checks were completed before any build or test:

1. `git status --short` printed no entries.
2. `git diff --name-status 43e1c44..HEAD` contains only campaign records,
   `AGENTS.md`, and `PLAN.md`; no F22 product/test source changed after the
   documented policy commit.
3. `ctest -N` reports exactly 122, 122, and 124 tests in the canonical,
   llfio-OFF, and Arrow-ON build directories respectively.
4. Cache inspection confirms Release/Ninja and the intended feature matrix:
   canonical = Arrow OFF / HiGHS ON / llfio ON; llfio-OFF = Arrow OFF /
   HiGHS OFF / llfio OFF; Arrow-ON = Arrow ON / HiGHS OFF / llfio OFF.
5. A dry build reports that all three directories must first re-run CMake, so
   no stale “ninja: no work to do” claim is assumed.
6. In all three generated CTest inventories, `test_problem_api_2_0` launches
   `scripts/test_f22_cpp_deprecations.py`. Its `PASS_REGULAR_EXPRESSION`
   requires the exact 33/33 diagnostic marker, the exact 33/33 behavior
   marker, and the Catch2 assertion/case floor; its failure regex rejects skip
   text.
7. Python collection reports 18 focused cases and 1,041 full cases. The
   current package/core provenance is:

   ```text
   PKG=C:\D\git\dtw-cpp\python\dtwcpp\__init__.py
   CORE=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
   CORE_SHA256=9FE527ADC2C813FF59618D34000D6C9707536131DD7033DDD255F1C88AD54165
   F22_NATIVE=True
   HIGHS=False
   ```

   The decisive gate will nevertheless force a native rebuild, copy the built
   extension and `libomp.dll`, and re-check built/imported hashes.
8. The pre-existing MEX and Python extension timestamps follow their owning
   source timestamps, but neither timestamp is accepted as final provenance;
   both decisive binding gates force compilation.
9. A combined Python discovery/provenance wrapper printed the correct
   inventories but exceeded its 60-second outer timeout. No DTWC++ child
   process remained. This is classified as a harness failure and contributes
   no pass/fail evidence; subsequent checks use one bounded process per
   subject.
10. An independent read-only adversarial review found that the native
    `LastTestsFailed.log` files still name earlier focused failures
    (`test_problem_api_2_0` in canonical/llfio-OFF and
    `unit_test_distance_matrix_csv` in Arrow-ON). More importantly, the C++
    mutation campaign restored source bytes but did not leave a separately
    evidenced final clean native rebuild after the timed-out runtime mutant.
    Source identity therefore cannot certify object-file identity. All three
    native matrices and both binding artifacts must be rebuilt clean-first
    before receiving evidence credit.
11. The F22 record's R6 bullets are registered expectations, not completed
    final-run evidence; this log is the owner of the missing verbatim outputs.
12. `AGENTS.md` still states the pre-F22 Python and MATLAB floors. The
    preregistered F22 ledgers above govern this run; the working-rule floors
    will be updated only after the decisive inventories reproduce them.

Verdict before execution: **CONDITIONAL GO only after clean-first rebuilds;
no test credit yet.**

## Decisive outputs

Execution commit:

```text
aff26272b56370d29557900c6005e01fde9407c5
```

### Canonical LLFIO-ON / Arrow-OFF

The clean-first build was initially launched with `--parallel 1`. At
115/622 objects this unnecessary compile-only restriction was stopped by
verifying and terminating exactly its `ninja.exe` PID 48188 and owning
`cmake.exe` PID 54972. The clean step had already removed the old graph.
The same Ninja graph resumed with normal parallel compilation and exited 0
after 261.1 seconds; the settling build printed:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

The regenerated inventory remained exactly 122 with one
`test_problem_api_2_0` entry. The serial full command was:

```text
ctest --test-dir build/highs-1151 -C Release --output-on-failure --no-tests=error -j 1
```

Its decisive summary was:

```text
100% tests passed, 0 tests failed out of 122

Total Test time (real) = 175.39 sec

The following tests did not run:
	 51 - test_cuda_correctness (Skipped)
	 53 - test_cuda_lb_keogh (Skipped)
	 57 - test_io_readers (Skipped)
	 58 - test_metal_correctness (Skipped)
	 59 - test_metal_lb_keogh (Skipped)
	 60 - test_metal_mmap (Skipped)
```

An exact set comparison reported:

```text
SKIPS=6 NAMES=test_cuda_correctness,test_cuda_lb_keogh,test_io_readers,test_metal_correctness,test_metal_lb_keogh,test_metal_mmap
SKIP_SET_MATCH=True
```

The full transcript recorded the F22 entry itself:

```text
Start  61: test_problem_api_2_0
61/122 Test  #61: test_problem_api_2_0 ......................   Passed   52.51 sec
```

The same fresh entry was then run verbosely to expose its subject markers:

```text
61: F22_CPP_SILENT count=0/33 entities=none canonical_deprecation_lines=0
61: F22_CPP_DIAGNOSTICS inventory=33/33 legacy=33/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=PASS
61: F22_CPP_COMPAT inventory=33/33 behavior=33/33 field_routes=4/4 io_routes=7/7 file_identity=6/6 stdout_identity=2/2 skips=0 verdict=PASS
61: ===============================================================================
61: All tests passed (229 assertions in 5 test cases)
1/1 Test #61: test_problem_api_2_0 .............   Passed   15.81 sec
```

`Testing/Temporary/LastTestsFailed.log` still contained the historical line
`61:test_problem_api_2_0` after both green executions. It is mutable stale
state, not a current verdict; the complete current transcripts above are the
evidence.

Verdict: **PASS [confirmed]** — 122/122, zero failed, the exact six registered
capability skips, and both F22 subjects executed above their registered
assertion/case floors.

### LLFIO-OFF / Arrow-OFF

The independent preflight found no competing DTWC++ build/test process and
reconfigured the established Ninja/Release matrix. Cache assertions printed:

```text
--   llfio:    OFF (DTWC_ENABLE_LLFIO=OFF) — mmap disabled.
-- ║  CUDA:      OFF
-- ║  Metal:     OFF
-- ║  HiGHS:     OFF
NOLLFIO_CACHE_ASSERT=PASS
```

The clean-first build removed 385 files, rebuilt its complete 386-edge graph,
exited 0, and the settling build printed:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
INVENTORY_COUNT=122
F22_ENTRIES=1
PRE_CTEST_EXCLUSIVITY=PASS
```

The registered serial command was:

```text
ctest --test-dir build/nollfio -C Release --output-on-failure --no-tests=error -j 1
```

Its decisive summary and subject entry were:

```text
Start  61: test_problem_api_2_0
61/122 Test  #61: test_problem_api_2_0 ......................   Passed    7.40 sec
100% tests passed, 0 tests failed out of 122
Total Test time (real) =  88.07 sec
```

The exact skip-set check printed:

```text
SKIPS=9 NAMES=unit_test_mmap_data_store,unit_test_mmap_distance_matrix,test_cuda_correctness,test_cuda_lb_keogh,test_io_readers,test_metal_correctness,test_metal_lb_keogh,test_metal_mmap,unit_test_benders
SKIP_SET_MATCH=True
```

The fresh verbose subject execution printed:

```text
61: F22_CPP_SILENT count=0/33 entities=none canonical_deprecation_lines=0
61: F22_CPP_DIAGNOSTICS inventory=33/33 legacy=33/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=PASS
61: F22_CPP_COMPAT inventory=33/33 behavior=33/33 field_routes=4/4 io_routes=7/7 file_identity=6/6 stdout_identity=2/2 skips=0 verdict=PASS
61: All tests passed (229 assertions in 5 test cases)
1/1 Test #61: test_problem_api_2_0 .............   Passed   23.75 sec
```

Verdict: **PASS [confirmed]** — 122/122, zero failed, the exact nine registered
capability skips, and both F22 subjects executed above their registered
assertion/case floors.

### Arrow-ON / LLFIO-OFF

The independent preflight found PyArrow 23.0.1, its runtime directories, and
the pinned package descriptors. Configuration printed:

```text
PYARROW_RUNTIME=C:\D\git\dtw-cpp\.venv\Lib\site-packages\pyarrow
PYARROW_LIBS_RUNTIME=C:\D\git\dtw-cpp\.venv\Lib\site-packages\pyarrow.libs
PYARROW_LIBS_DLL_COUNT=1
--   llfio:    OFF (DTWC_ENABLE_LLFIO=OFF) — mmap disabled.
--   Arrow:    YES (v23.0.1) — system install
--   Parquet:  YES (v23.0.1) — system install
-- Arrow + Parquet linked — IPC and Parquet reading enabled
ARROW_CACHE_ASSERT=PASS
```

The clean-first build removed 387 files, rebuilt its complete 388-edge graph,
exited 0, and its settling inventory printed:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
INVENTORY_COUNT=124
F22_ENTRIES=1
READER_ENTRIES=1
```

The first runtime-metadata assertion rejected the preflight:

```text
READER_METADATA_PYARROW=False
READER_METADATA_PYARROW_LIBS=False
```

No test had run. The strings were present in `ctest -N -V`; the assertion had
compared Windows backslashes against CTest's forward slashes. The independent
CTest JSON model then exposed the exact three `ENVIRONMENT_MODIFICATION`
entries, and an exact-string assertion printed:

```text
READER_ENV_ENTRIES=3
READER_RUNTIME_METADATA=PASS
PRE_CTEST_EXCLUSIVITY=PASS
```

The registered serial command was:

```text
ctest --test-dir build/arrow-pyarrow-23 -C Release --output-on-failure --no-tests=error -j 1
```

Its decisive summary and F22 entry were:

```text
Start  61: test_problem_api_2_0
61/124 Test  #61: test_problem_api_2_0 ......................   Passed    7.73 sec
100% tests passed, 0 tests failed out of 124
Total Test time (real) =  92.37 sec
```

The exact skip-set check printed:

```text
SKIPS=8 NAMES=unit_test_mmap_data_store,unit_test_mmap_distance_matrix,test_cuda_correctness,test_cuda_lb_keogh,test_metal_correctness,test_metal_lb_keogh,test_metal_mmap,unit_test_benders
SKIP_SET_MATCH=True
```

The fresh verbose F22 execution printed:

```text
61: F22_CPP_SILENT count=0/33 entities=none canonical_deprecation_lines=0
61: F22_CPP_DIAGNOSTICS inventory=33/33 legacy=33/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=PASS
61: F22_CPP_COMPAT inventory=33/33 behavior=33/33 field_routes=4/4 io_routes=7/7 file_identity=6/6 stdout_identity=2/2 skips=0 verdict=PASS
61: All tests passed (229 assertions in 5 test cases)
1/1 Test #61: test_problem_api_2_0 .............   Passed    9.20 sec
```

The real reader then ran verbosely under the CTest-provided runtime paths:

```text
57: All tests passed (390 assertions in 11 test cases)
1/1 Test #57: test_io_readers ..................   Passed    0.58 sec
READER_ASSERTION_FLOOR=True
READER_SKIP_ABSENT=True
```

Verdict: **PASS [confirmed]** — 124/124, zero failed, the exact eight
registered capability skips, both F22 subjects above their floors, and the
reader executed 390 assertions / 11 cases rather than skipping.

### Python forced-fresh binding — attempt 1

The preflight confirmed a clean tree, no competing process, Ninja/Release,
Python and llfio ON, Arrow and HiGHS OFF, one built and one installed ABI
artifact, and LLVM `libomp.dll`. The clean-first `_dtwcpp_core` target removed
168 files, rebuilt a 42-edge dependency graph, and linked the extension.
The settling build printed `ninja: no work to do`. Installation and import
provenance printed:

```text
BUILT_PYD_COUNT=1
INSTALLED_PYD_COUNT=1
BUILT=C:\D\git\dtw-cpp\build\cfg-gate-normal\python\_dtwcpp_core.cp313-win_amd64.pyd
INSTALLED=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
BUILT_SHA256=0E6FCE5C3C312B916E845F1A1D30F2E86F37B663D30809EAE131160562932BDF
INSTALLED_SHA256=0E6FCE5C3C312B916E845F1A1D30F2E86F37B663D30809EAE131160562932BDF
BUILT_IMPORTED_HASH_MATCH=True
LIBOMP_SHA256=5E6AC41ED81DFF9B41642A2F62CFD4784AA1C7CA1D348BEBD08FC54492C94466
PACKAGE=C:\D\git\dtw-cpp\python\dtwcpp\__init__.py
CORE=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
F22_NATIVE=True
HIGHS=False
PYTHON_PROVENANCE=PASS
```

This is a genuinely new binary: its pre-build hash was
`9FE527ADC2C813FF59618D34000D6C9707536131DD7033DDD255F1C88AD54165`.
Fresh collection printed 18 focused and 1,041 full tests. The decisive
focused run printed:

```text
..................F22_PYTHON_GATE alias_symbols=12 operations=13 primary_warn_once=13 canonical_silent=13 equivalent=13 class_routes=3 identity=2 ordinary_legacy=0 verdict=PASS

18 passed in 7.97s
PYTHON_FOCUSED_SUMMARY=True
PYTHON_FOCUSED_MARKER=True
PYTHON_FOCUSED_SKIP_ABSENT=True
```

The first full attempt was outside the registered band:

```text
=========== 7 failed, 1022 passed, 12 skipped in 101.62s (0:01:41) ============
PYTEST_EXIT=1
PYTHON_FULL_EXPECTED_SUMMARY=False
PYTHON_FULL_FAILED_COUNT=0
PYTHON_FULL_SOLE_F39=False
PYTHON_FULL_LEDGER_28_VS_27=True
```

The seven pytest failure lines were:

```text
FAILED tests/python/test_hpc.py::TestLocalRoundTrip::test_required_input_message_names_toml_first
FAILED tests/python/test_hpc.py::TestLocalRoundTrip::test_two_groups_recovered
FAILED tests/python/test_hpc.py::TestLocalRoundTrip::test_seeded_restart_improves_registered_fixture
FAILED tests/python/test_hpc.py::TestLocalRoundTrip::test_cli_rejects_zero_restart_count
FAILED tests/python/test_hpc.py::TestLocalRoundTrip::test_cli_missing_strategy_is_honored
FAILED tests/python/test_supply_chain_pins.py::test_live_tracked_cmake_inventory_is_complete
FAILED tests/python/test_version_ssot.py::test_version_file_python_metadata_and_cli_match
```

The parser's zero failed-count is itself rejected: it disagrees with pytest's
verbatim seven-line ledger and cannot adjudicate this run. The trace identifies
the setup fault [confirmed]: clean-first building only `_dtwcpp_core` deleted
`build/cfg-gate-normal/bin/dtwc_cl.exe`; the HPC helper then selected
`build/arrow-pyarrow-23/bin/dtwc_cl.exe`, whose direct subprocesses returned
`3221225781`, while the version test reported:

```text
AssertionError: dtwc_cl executable not found; set DTWC_CL_PATH
```

The expected F39 failure remained exactly `assert 28 == 27`. No source change
is justified. Attempt 2 may run only after rebuilding `dtwc_cl` from the same
clean graph, pinning `DTWC_CL_PATH` to it, and independently executing the CLI
version/error/runtime probes.

Attempt-1 verdict: **FALSIFIED [confirmed]** — 7/1,041 failures rather than the
registered sole F39 red.

### Python forced-fresh binding — attempt 2

The sibling CLI was rebuilt from the same clean `cfg-gate-normal` graph:

```text
[1/3] Building CXX object CMakeFiles/dtwc_cl.dir/dtwc/dtwc_cl.cpp.obj
[2/3] Linking CXX executable bin\dtwc_cl.exe
CFG_CLI_SHA256=F88AD896455CF708710B980E30FD17FA2FFF74F474C7EEF1BD892B7F53A8745E
```

`test_hpc` does not honor `DTWC_CL_PATH`: its helper chooses the newest
build-tree CLI. Both the helper route and the environment route were therefore
asserted separately:

```text
EXPECTED=C:\D\git\dtw-cpp\build\cfg-gate-normal\bin\dtwc_cl.exe
SELECTED=C:\D\git\dtw-cpp\build\cfg-gate-normal\bin\dtwc_cl.exe
HPC_ROUTE_MATCH=True
```

Independent subprocess probes then printed:

```text
CLI_VERSION_STDOUT='2.0.0rc1\n'
CLI_VERSION_STDERR=''
CLI_VERSION_EXIT=0
CLI_REQUIRED_STDOUT=''
CLI_REQUIRED_STDERR='Error: --input is required via CLI or config file (TOML; YAML if built with DTWC_ENABLE_YAML)\n'
CLI_REQUIRED_EXIT=1
CLI_DIAGNOSTIC_PREFLIGHT=PASS
```

A real CLI conformance run against the recorded non-degenerate dataset exited
zero and reproduced the canonical partition:

```text
CLI_REAL_EXIT=0
CLI_REAL_STDERR=''
CLI_REAL_LABELS=0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2
CLI_REAL_MEDOIDS=4,13,22
CLI_REAL_SILHOUETTE=0.96894972666666657
CLI_REAL_CONFORMANCE=PASS
```

The first cleanup assertion expected only three result files and rejected the
probe because the CLI also emitted its checkpoint and distance matrix. All
five exact files were then validated under the probe directory and removed:

```text
PROBE_FILES=conformance_checkpoint.bin,conformance_distance_matrix.csv,conformance_labels.csv,conformance_medoids.csv,conformance_silhouettes.csv
PROBE_CLEANED=True
POST_CLI_EXTENSION_MATCH=True
```

The six formerly unexpected pytest nodes passed before the final full run:

```text
......                                                                   [100%]
6 passed in 8.62s
CLI_PREFLIGHT_SUMMARY_MATCH=True
CLI_PREFLIGHT_SKIP_LINES=0
```

An independent adversarial execution reproduced the same route, exact
diagnostics, a separate four-series real clustering, and 6/6 focused result,
then issued GO. The first attempt to launch the final suite was stopped before
pytest because its exclusivity guard observed two finishing provenance-only
Python processes; both were absent on the resolving probe. No full-run credit
or attempt was assigned to that aborted launch.

The final full command used the exact intended CLI and fresh installed
extension. Its complete ledger was:

```text
ATTEMPT2_EXCLUSIVITY=PASS
ATTEMPT2_ROUTE=C:\D\git\dtw-cpp\build\cfg-gate-normal\bin\dtwc_cl.exe
collected 1041 items
FAILED tests/python/test_supply_chain_pins.py::test_live_tracked_cmake_inventory_is_complete
=========== 1 failed, 1028 passed, 12 skipped in 132.10s (0:02:12) ============
PYTEST_EXIT=1
PYTHON_FULL_COLLECTION_MATCH=True
PYTHON_FULL_SUMMARY_MATCH=True
PYTHON_FULL_FAILED_COUNT=1
PYTHON_FULL_SOLE_F39=True
PYTHON_FULL_LEDGER_28_VS_27=True
PYTHON_FULL_EXPECTED_F39_RED=PASS
PYTHON_FULL_ERROR_LINES=0
```

The sole failure retained the exact frozen ledger:

```text
>       assert manifest_total == 27
E       assert 28 == 27
```

Post-run built/installed extension hashes remained identical at
`0E6FCE5C3C312B916E845F1A1D30F2E86F37B663D30809EAE131160562932BDF`,
and repository hygiene passed.

Verdict: **PASS WITH REGISTERED F39 RED [confirmed]** — focused 18/18 with the
exact F22 marker; full 1,041 = 1,028 passed + 12 registered skips + only the
known F39 inventory failure; zero error nodes; fresh native identity proven.

### MATLAB fresh MEX and R2024b

The preflight found both release executables, a clean tree, one old MEX, and
the source hash
`BDB6086BE23E3DB71C0B2FE3FF5536A8BCBA624331CD3CB3A47160E9B78C1782`.
The current Visual Studio 18/R2024b-SDK configuration regenerated with
`DTWC_ALLOW_SEQUENTIAL=OFF`, llfio/Arrow/CUDA/Metal/HiGHS OFF in the effective
summary, and OpenMP enabled through `-openmp:experimental`.

The clean-first Release target compiled `dtwc_mex.cpp`, linked the MEX, and a
settling build changed no source. Provenance printed:

```text
MEX_SOURCE_COMPILED=True
MEX_LINKED=True
MEX_SOURCE_SHA256=BDB6086BE23E3DB71C0B2FE3FF5536A8BCBA624331CD3CB3A47160E9B78C1782
MEX_ARTIFACT_COUNT=1
MEX_ARTIFACT=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
MEX_BYTES=685056
MEX_SHA256=8C39AD28001B9D0C30824CCDE15CAC84AC3F1ADCAB1079FA69E8671C0A1D9064
MEX_HASH_CHANGED=True
PRE_MATLAB_EXCLUSIVITY=PASS
```

The old pre-build MEX hash was
`8872FF10EA1F05FFD4A7E4D7657DE39BD64EE0974DB99282F1125F98071B26EE`.
An independent static audit counted the five suites as 14 + 34 + 28 + 8 + 1 =
85 tests and reconciled the registered failed/incomplete sets before any
MATLAB execution.

#### R2024b focused

The focused process used unique repository-local preferences and temp
directories, `OMP_NUM_THREADS=2`, repository sources first, and the fresh MEX
directory last. Its decisive output was:

```text
MATLAB_RELEASE=R2024b
MEX_SELECTED=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
MEX_COUNT=1
OMP_AVAILABLE=1 OMP_MAX=2 OMP_ENGAGED=2 OMP_PASS=1 OMP_REASON=
F22_MATLAB_DEPRECATION aliases=15/15 warning_profiles=15/15 messages=15/15 canonical_silent=15/15 equivalence=15/15 constructor_silent=1/1 tier1_silent=1/1 skips=0 verdict=PASS
F22_MATLAB_FOCUSED release=R2024b total=1 passed=1 failed=0 incomplete=0
MATLAB_EXIT=0
FOCUSED_MARKER_COUNT=1
FOCUSED_SUMMARY_COUNT=1
POST_FOCUSED_MEX_SHA256=8C39AD28001B9D0C30824CCDE15CAC84AC3F1ADCAB1079FA69E8671C0A1D9064
```

The isolated runtime contained no reparse points, every entry resolved below
its exact directory, and cleanup reported `RUNTIME_CLEANED=True`.

#### R2024b full attempt 1

An overbroad process guard first observed unrelated MATLAB sessions in other
repositories and exited before creating a runtime or starting MATLAB. It
contributed no attempt or test credit. The corrected repository-scoped guard
then launched the 85-test suite.

The suite itself ran all five files, printed the exact F22 marker, and printed
the registered names:

```text
FAILED_NAME=test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
FAILED_NAME=test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
INCOMPLETE_NAME=test_test_api/test_parallelisation_serial_is_honest
```

The post-run assertion was nevertheless invalid: embedded MATLAB double-quote
string delimiters were stripped at the PowerShell-to-`-batch` boundary. MATLAB
therefore exited 1 with:

```text
Unrecognized function or variable 'test_dtwclustering_metric_routes_match_exhaustive_oracle'.
```

The custom 85/82/2/3 line never executed, so the otherwise matching result
names do not receive gate credit. The MEX remained byte-identical, the log
contained one F22 marker, two failed-name lines, and three incomplete-name
lines, and the isolated runtime was safely removed.

Attempt-1 verdict: **FALSIFIED [confirmed]** — the post-run name-set oracle did
not execute. The sole remaining attempt must use single-quoted MATLAB cell
arrays and reproduce the complete custom ledger.

#### R2024b full attempt 2

Before the final attempt, an isolated PowerShell-to-MATLAB boundary probe used
the intended single-quoted cell-array representation and printed:

```text
CELL_ORACLE=PASS
MATLAB_EXIT=0
CELL_ORACLE_MARKERS=1
```

Its runtime was validated and removed. The final full process then used the
same fresh MEX and source paths as the focused gate. Its decisive output was:

```text
MATLAB_RELEASE=R2024b
MEX_SELECTED=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
MEX_COUNT=1
OMP_AVAILABLE=1 OMP_MAX=2 OMP_ENGAGED=2 OMP_PASS=1 OMP_REASON=
F22_MATLAB_DEPRECATION aliases=15/15 warning_profiles=15/15 messages=15/15 canonical_silent=15/15 equivalence=15/15 constructor_silent=1/1 tier1_silent=1/1 skips=0 verdict=PASS
FAILED_NAME=test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
FAILED_NAME=test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
INCOMPLETE_NAME=test_test_api/test_parallelisation_serial_is_honest
F22_MATLAB_FULL release=R2024b total=85 passed=82 failed=2 incomplete=3
MATLAB_EXIT=0
FULL_MARKER_COUNT=1
FULL_SUMMARY_COUNT=1
FAILED_SET_MATCH=True
INCOMPLETE_SET_MATCH=True
POST_FULL_MEX_SHA256=8C39AD28001B9D0C30824CCDE15CAC84AC3F1ADCAB1079FA69E8671C0A1D9064
```

The retained F18 details remained exact: one route reports unknown MEX command
`DTWClustering_compute_distance_matrix`, and the validation-order case reports
actual `MATLAB:fit:expectedNonempty` versus expected `dtwc:invalidArgument`.
No new failure appeared. The 30-entry isolated runtime contained no reparse
points, every entry resolved below its exact directory, and cleanup reported
`RUNTIME_CLEANED=True`.

Verdict: **PASS WITH REGISTERED F18 RED [confirmed]** — focused 1/1 and full
85/82/2/3, exact failed/incomplete sets, exact F22 marker, OpenMP engaged,
unambiguous source/MEX path, and unchanged fresh binary hash.

### MATLAB R2025b

The second release reused the exact same source tree and immutable fresh MEX
as R2024b. A repository-scoped process guard found no competing MATLAB
process before either launch.

#### R2025b focused

The focused process again used unique repository-local preferences and temp
directories, `OMP_NUM_THREADS=2`, repository sources first, and the fresh MEX
directory last. Its decisive output was:

```text
MATLAB_RELEASE=R2025b
MEX_SELECTED=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
MEX_COUNT=1
OMP_AVAILABLE=1 OMP_MAX=2 OMP_ENGAGED=2 OMP_PASS=1 OMP_REASON=
F22_MATLAB_DEPRECATION aliases=15/15 warning_profiles=15/15 messages=15/15 canonical_silent=15/15 equivalence=15/15 constructor_silent=1/1 tier1_silent=1/1 skips=0 verdict=PASS
F22_MATLAB_FOCUSED release=R2025b total=1 passed=1 failed=0 incomplete=0
MATLAB_EXIT=0
FOCUSED_MARKER_COUNT=1
FOCUSED_SUMMARY_COUNT=1
POST_FOCUSED_MEX_SHA256=8C39AD28001B9D0C30824CCDE15CAC84AC3F1ADCAB1079FA69E8671C0A1D9064
```

The isolated runtime contained no reparse point, every entry resolved below
its exact directory, and cleanup reported `RUNTIME_CLEANED=True`.

#### R2025b full

The full process ran the same five suites and emitted:

```text
MATLAB_RELEASE=R2025b
MEX_SELECTED=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
MEX_COUNT=1
OMP_AVAILABLE=1 OMP_MAX=2 OMP_ENGAGED=2 OMP_PASS=1 OMP_REASON=
F22_MATLAB_DEPRECATION aliases=15/15 warning_profiles=15/15 messages=15/15 canonical_silent=15/15 equivalence=15/15 constructor_silent=1/1 tier1_silent=1/1 skips=0 verdict=PASS
FAILED_NAME=test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
FAILED_NAME=test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
INCOMPLETE_NAME=test_test_api/test_parallelisation_serial_is_honest
F22_MATLAB_FULL release=R2025b total=85 passed=82 failed=2 incomplete=3
MATLAB_EXIT=0
FULL_MARKER_COUNT=1
FULL_SUMMARY_COUNT=1
FAILED_SET_MATCH=True
INCOMPLETE_SET_MATCH=True
POST_FULL_MEX_SHA256=8C39AD28001B9D0C30824CCDE15CAC84AC3F1ADCAB1079FA69E8671C0A1D9064
```

The two failures and three incompletes are byte-for-byte the registered F18
set; no additional failure appeared. The 233-entry isolated runtime contained
no reparse point, every entry resolved below its exact directory, and cleanup
reported `RUNTIME_CLEANED=True`. A post-cleanup process probe found no
repository-scoped MATLAB process. The source and MEX hashes remained:

```text
MEX_SOURCE_SHA256=BDB6086BE23E3DB71C0B2FE3FF5536A8BCBA624331CD3CB3A47160E9B78C1782
MEX_SHA256=8C39AD28001B9D0C30824CCDE15CAC84AC3F1ADCAB1079FA69E8671C0A1D9064
```

Verdict: **PASS WITH REGISTERED F18 RED [confirmed]** — focused 1/1 and full
85/82/2/3, exact failed/incomplete sets, exact F22 marker, OpenMP engaged,
unambiguous source/MEX path, and unchanged fresh binary hash.

## Final adjudication

The final execution campaign met every preregistered full-gate band:

- canonical: 122/122, zero failed, exact six capability skips;
- llfio-OFF: 122/122, zero failed, exact nine capability skips;
- Arrow-ON: 124/124, zero failed, exact eight capability skips, with the
  Arrow reader executing 390 assertions in 11 cases;
- Python: focused 18/18; full 1,041 = 1,028 passed + 12 skipped + only the
  registered F39 `28 == 27` failure;
- R2024b and R2025b: focused 1/1 and full 85/82/2/3 on each, with only the
  registered F18 names.

Final-full-gate verdict: **PASS WITH REGISTERED F18/F39 RED [confirmed]** by
the verbatim ledgers above.

F22 closure verdict: **FALSIFIED [confirmed]**. These green full gates do not
alter the independently registered C++ mutation result: only 33 of 46
required mutants were killed before both permitted attempts exhausted their
300-second runtime-mutant cap. F22 therefore remains unchecked. Its product,
tests, and documentation stay retained; no third mutation run or
reinterpretation is permitted.

The claim most likely to be wrong is that the full Python inventory always
selects the intended sibling CLI: `test_hpc` discovers build-tree binaries by
mtime rather than honoring `DTWC_CL_PATH`. This run confirms the claim only
for the recorded attempt-2 route, whose selected executable, byte-level
diagnostics, real conformance output, and formerly failing nodes were checked
before the full suite. Future fresh-extension gates must rebuild and assert
both native siblings again.
