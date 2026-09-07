# Test taxonomy — `tests/unit/*` (flat), `tests/support/*`, and their CMake registration

Scope: the 72 files directly in `tests/unit/` (subdirectories `core/`, `algorithms/`,
`adversarial/`, `mip/`, `io/`, `types/` belong to other agents), plus
`tests/support/deterministic_series.hpp`, `tests/test_util.hpp`, and the parts of
`tests/CMakeLists.txt` that register these targets.

Repository `C:\D\git\dtw-cpp`, branch `Claude`, HEAD `a31956e`. Read-only pass; every
file below was read in full.

**This is a taxonomy with verdict *candidates*. Nothing here authorises a deletion.**

Timings quoted are from `build/msvc-debug/Testing/Temporary/LastTest.log` (MSVC Debug,
2026-09-02 run, 131 targets). That build has `DTWC_HAS_MMAP` **on**, and `DTWC_HAS_CUDA`,
`DTWC_HAS_METAL`, `DTWC_HAS_ARROW`, `DTWC_HAS_MPI` **off**.

---

## 0. Registration model (read this before the ledger)

`tests/CMakeLists.txt:1` globs **every** `*.cpp` under `tests/` recursively and hands each
to `add_executable_with_coverage_and_test` (`cmake/Coverage.cmake:3`). Consequences that
bind the whole campaign:

* One target per file, named after the basename. **Deleting a file deletes a CTest entry**;
  merging two files removes one entry. There is no explicit inventory to update, but there
  *are* named guards (below) that break if a target disappears.
* `set_tests_properties(... SKIP_RETURN_CODE 4)` (`cmake/Coverage.cmake:32`) is applied to
  every target, so **a Catch2 `SKIP` scores green** unless a per-target guard clears it.
* `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` — tests run from the repo root, so any test that
  writes a relative path pollutes the source tree (see §3, `unit_test_fileOperations.cpp`).
* `DTWC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"` is defined for every target; several
  tests use `fs::path{DTWC_TEST_DATA_DIR}.parent_path()` as "the repo root" and then read
  *tracked source files* (`test_supply_chain_pinning.cpp:48`,
  `unit_test_variant_precision.cpp:178`, `unit_test_deterministic_series.cpp:114`).

Per-target special cases in `tests/CMakeLists.txt` **inside this scope**:

| target | lines | what the guard does |
|---|---|---|
| `test_problem_api_2_0` (F22) | 125-140, 528-560 | prepends `scripts/test_f22_cpp_deprecations.py --launch` to the CTest command (this is why it takes 18.76 s); clears `SKIP_RETURN_CODE`; `FAIL_REGULAR_EXPRESSION` on skip; two-marker `PASS_REGULAR_EXPRESSION` + Catch2 floor; `RUN_SERIAL` |
| `unit_test_checkpoint_binary` (F51) | 145-175 | absolute build-local `DTWC_F51_TEST_ROOT`, `TMP/TEMP/TMPDIR` redirected there, clears skip code, exact marker + `>=270 assertions in >=2 cases`, `RUN_SERIAL` |
| `unit_test_DataLoader` (F21) | 232-247 | clears skip code, `F21_CPP_NAMES ...` marker + `>=79 assertions in >=2 cases` |
| `unit_test_problem_encapsulation` (F19) | 594-612 | `DTWC_F19_TEST_ROOT`, clears skip code, `F19_PROBLEM_API ...` marker + floor |
| `unit_test_problem_storage_policy` (F20) | 614-654 | `DTWC_F20_TEST_ROOT`, `TMP/TEMP/TMPDIR` redirect, **two different markers** for llfio-on/off, `RUN_SERIAL` |
| `unit_test_deterministic_series` (F15) | 656-673 | `DTWC_F15_SOURCE_ROOT`, clears skip code, `F15_TEST_SUPPORT ...` marker + `>=177 assertions in >=6 cases` |
| `test_supply_chain_pinning` (F16) | 400-620 | a **large configure-time CMake guard** parses `CMakePresets.json` with `string(JSON)` and `FATAL_ERROR`s on drift, *plus* runtime marker `F16_CMAKE_PRESETS ...` + floor |
| `test_io_readers` (F6) | 622-646 | attached **only** when `DTWC_HAS_ARROW`; in the canonical Arrow-OFF gate no guard is attached and the file's single `SKIP` is the expected outcome |

Registration quirks worth flagging (§5): the F20 test *requires* the CMake `ENVIRONMENT`
redirect to run at all (`unit_test_problem_storage_policy.cpp:119` asserts
`fs::equivalent(fs::temp_directory_path(), root)`), so the executable cannot be run
standalone; and `unit_test_mpi` is registered as an ordinary single-process CTest target
although its documented invocation is `mpiexec -n 4`.

---

## 1. Per-file ledger

Columns: `lines` = wc -l; `TC` = `TEST_CASE(` count; `asrt` = Catch2 assertion-macro
occurrences (`REQUIRE*`/`CHECK*`/`STATIC_REQUIRE`/`SUCCEED`/`FAIL`; `unit_test_mpi` counted
via its own `MPI_CHECK`).

| path | lines | TC | asrt | subject | category | oracle | overlap candidates | verdict candidate | evidence |
|---|---|---|---|---|---|---|---|---|---|
| `tests/unit/test_build_parallelism.cpp` | 81 | 2 | 2 | build-mode macros `DTWC_HAS_OPENMP` / `DTWC_SEQUENTIAL_BUILD` / `_OPENMP` propagation | capability-guard | none (the compile-time `#error` is the real gate) | none | keep — the load-bearing part is the `#error` block at :33-46; the two runtime `REQUIRE`s are ceremony but cost 0.07 s | :33-46 `#error`; :65 `REQUIRE(has_openmp != sequential)` |
| `tests/unit/test_cuda_correctness.cpp` | 1860 | 62 | 213 | CUDA distance-matrix / warp / regtile / wavefront / 1-vs-N / k-vs-N kernels vs CPU | mixed(contract/oracle, capability-guard) | independent DP (`gpu_fixed_band_oracle.hpp`) + CPU `dtwFull_L`/`dtwBanded` cross-route | `tests/unit/test_metal_correctness.cpp` (same F12 ledger, same `cpu_distance_matrix` shape) | keep, but **move the F12 oracle case outside the `#ifdef`** — `TEST_CASE` at :129 (`F12 independent fixed-band arbiters ...`) is pure host C++ yet sits inside `#else // DTWC_HAS_CUDA` at :40, so it never compiles in the canonical gate | :33-40 guard; :129-193 host-only oracle case; canonical run = `test cases: 1 \| 1 skipped` / `assertions: - none -` (LastTest.log:911-913) |
| `tests/unit/test_cuda_kernel_override.cpp` | 390 | 6 | 42 | `cuda::detail::select_kernel` / `kernel_path_name` + device dispatch discriminator | mixed(contract/oracle, capability-guard) | table-driven expected values (`auto_cases`/`override_cases`) + CPU oracle on device | `test_metal_correctness.cpp:662` (`KernelOverride forces requested pipeline`) | keep — the host seam at :34-101 runs in every build (`__has_include` at :18) and is the only kernel-selection contract | :18-23 seam probe; :52-80 registered tables; canonical run = 2 cases, 1 passed 1 skipped |
| `tests/unit/test_cuda_launch_guards.cpp` | 175 | 6 | 31 | `require_pair_count_fits`, `upper_triangle_pairs`, `require_cuda_device` (A15/A16) | contract/oracle | independent arithmetic (`upper_triangle_pairs(65537) == 2147516416`) | none | keep — best-in-class host-testable seam; runs fully in the CUDA-OFF gate | :55-63 exact 64-bit counts; :69-75 typed rejection |
| `tests/unit/test_cuda_lb_keogh.cpp` | 357 | 9 | 27 | `compute_lb_keogh_cuda`, LB pruning in the CUDA matrix | contract/oracle (device-only) | CPU `core::lb_keogh_symmetric` reference | `test_metal_lb_keogh.cpp` (identical structure) | keep — but it is 100 % dark in the canonical gate (`assertions: - none -`) | :36-43 guard; :52-74 CPU reference |
| `tests/unit/test_decode_pair.cpp` | 415 | 8 | 41 | `dtwc::detail::decode_pair` SSOT + `metal::detail::pair_chunk_offset` | contract/oracle + adversarial | three **independent** oracles: brute-force enumeration, the MPI reference formula, and a verbatim MSL transliteration | none | keep — exemplary; pins Criticals #2/#3 with a *negative control* (`old_metal_fp32_decode`) | :44-56 MPI oracle; :219-243 negative control; :387-414 degenerate N |
| `tests/unit/test_env_device.cpp` | 257 | 8 | 24 | `Env::set_device`, `.env` HPC gate, `dtwc::env()` singleton | contract/oracle | recorded byte-exact contract strings transcribed from `docs/api-contract-2.0.md` | `test_storage_policy.cpp:98` reuses the same `.env` + injected-probe fixture | keep | :43-68 registered messages; :112 `== kMsgUnknownFoo` |
| `tests/unit/test_error_taxonomy.cpp` | 126 | 7 | 26 | `dtwc::Error` hierarchy + one live throw site | mixed(trivial, contract/oracle) | none for :41-99 (asserts what `class X : public Y` obviously does); live path for :107-126 | none | **merge** — :41-99 (5 cases, ~20 assertions) restate a class hierarchy the compiler already enforces; :107-126 (`soft_dtw_gradient` empty input) is the only behavioural content and belongs with `unit_test_soft_dtw.cpp` | :43-47 `what()` echo; :56-59 `REQUIRE_THROWS_AS(throw X, base)` |
| `tests/unit/test_io_readers.cpp` | 552 | 14 | 104 | Arrow IPC + Parquet reader hardening (Float64 check, list-offset bounds, `ndim=0`, nulls) | adversarial | crafted fixtures + positive controls | none in scope | keep — high value; but see §3: **the skip path is the common path** in the canonical gate | :34-41 guard; F6 note `tests/CMakeLists.txt:622-635`; canonical run = 1 skipped, `assertions: - none -` |
| `tests/unit/test_metal_correctness.cpp` | 684 | 22 | 70 | Metal wavefront / regtile / banded_row / KvN kernels vs CPU | mixed(contract/oracle, capability-guard) | F12 oracle + CPU `dtwFull_L`/`dtwBanded` | `test_cuda_correctness.cpp` (mirror) | keep — but the file has **no** host-only case at all: with `DTWC_HAS_METAL` off it is 1 skip, 0 assertions, and there is no macOS CI runner | :32-39 guard; :43-66 CPU refs |
| `tests/unit/test_metal_lb_keogh.cpp` | 248 | 7 | 24 | `compute_lb_keogh_metal` + LB pruning + banded_row fallback | contract/oracle (device-only) | CPU `lb_keogh_symmetric` | `test_cuda_lb_keogh.cpp` | keep | :32-39 guard; :189-194 CPU ref |
| `tests/unit/test_metal_mmap.cpp` | 115 | 3 | 2 | Metal → `visit_distmat` dense/mmap dispatch | capability-guard | CPU BruteForce cross-route | `unit_test_variant_distmat.cpp` (dense/mmap dispatch on CPU) | keep — smallest Metal file, 0 assertions off-device; the *dispatch* contract it names is already covered on CPU | :25-30 guard; :76-78 tolerant compare |
| `tests/unit/test_problem_api_2_0.cpp` | 1609 | 5 | 41 | every retained 1.x C++ alias vs its 2.0 canonical spelling (F22) | mixed(contract/oracle, fingerprint) | recorded bit-exact matrix / stdout / file literals + canonical-vs-legacy cross-route | `unit_test_DataLoader.cpp:235` (F21, overlapping `startColumn`/`setDataPath` aliases) | keep+tolerance — see §3: the 33 sub-checks are aggregated into `bool`s and reported through `f22_keyed_ledger::record`, so a failure names the ledger slot, not the sub-condition | :439-462 ledger; :1577-1608 counts; 18.76 s |
| `tests/unit/test_runtime_loudness_compute.cpp` | 151 | 1 | 7 | `get_max_threads()` chokepoint emits the single-thread warning once, without `Env` | contract/oracle | byte-exact message transcribed from `dtwc/env.cpp` | `test_runtime_loudness_env.cpp` (same strings, `Env` half) | keep — a separate binary is *required* (process-once `call_once` guard); documented at :19-23 | :56-64 registered strings; :136 `count_occurrences(...) == 1` |
| `tests/unit/test_runtime_loudness_env.cpp` | 132 | 2 | 14 | `detail::sequential_cause` / `sequential_warning_text` + live `Env` ctor | contract/oracle | byte-exact recorded strings + full predicate truth table | `test_runtime_loudness_compute.cpp` (duplicated string constants) | keep; **hoist the two message constants and `capture_cerr` into shared support** | :52-60 constants (byte-identical to compute file :56-64); :94-99 truth table |
| `tests/unit/test_runtime_loudness_gpu.cpp` | 83 | 2 | 8 | explicit GPU strategy never silently degrades to CPU | contract/oracle | byte-exact recorded `DeviceError` messages | `unit_test_problem_storage_policy.cpp:477-488` (different messages, same no-fallback rule) | keep | :16-21 messages; :61-62 no-fallback |
| `tests/unit/test_storage_policy.cpp` | 385 | 8 | 42 | `StoragePolicy::Auto` routing, hpc metadata-only load, mmap-vs-heap digit identity, Pruned→mapped routing, `choose_storage` | contract/oracle | derived footprint arithmetic (288 B) + digit-identical cross-route | `unit_test_problem_storage_policy.cpp` (F20 — the Problem side of the same rule) | keep | :86-88 Auto→mmap; :180 `d_heap == d_mmap`; :318-327 `choose_storage` truth table |
| `tests/unit/test_supply_chain_pinning.cpp` | 372 | 4 | 73 | on-disk text of `cmake/Dependencies.cmake`, `dtwc/mip/CMakeLists.txt`, `documentation.yml`, `CMakePresets.json`, `pyproject.toml` | implementation-detail (text sentinel, self-declared at :12-16) | none — regex over build files | the F16 **configure-time** guard in `tests/CMakeLists.txt:400-620` re-checks the same preset facts with a real JSON parser | keep+tolerance — the `TEST_CASE` at :184 duplicates the CMake guard almost line-for-line and additionally pins *counts* (`== 3`, `== 10`, `== 6`, `== 1`) that break on any legitimate preset addition | :191-193 self-declared "not a replacement parser"; :306-318 exact host-condition counts |
| `tests/unit/test_test_api.cpp` | 90 | 2 | 18 | `dtwc::test::parallelisation()` / `gpu()` introspection API | contract/oracle | self-consistency of the reported struct + branch-on-`available` | `tests/matlab/test_test_api.m` (same API through MEX; allow-listed as Incomplete in `tests/CMakeLists.txt:914`) | keep | :53 `threads_engaged >= 2` (the one load-bearing assertion) |
| `tests/unit/test_tier1_cpp_api.cpp` | 664 | 16 | 92 | Tier-1 `load`/`cluster`/`Result`, seed locality, CWD purity, UTF-8 end-to-end | contract/oracle | recorded labels/medoids/costs + cross-route (`Result::distance_matrix` vs `Problem::dist_by_ind`) + deleted-overload `static_assert`s | `tests/conformance/cpp_conformance.cpp`; `unit_test_cli_args.cpp:387` (same seed fixture, different entry point) | keep — but **belongs in a `cli`/`session` subdirectory** by layer | :37-50 overload poison; :127-131 recorded labels/medoids; :523 `fs::is_empty(sandbox)` |
| `tests/unit/unit_test_Clock.cpp` | 65 | 1 | 6 | `dtwc::Clock` | trivial | self-consistency only (`clk.start() == start_time`) | none | keep — but 1.27 s of the suite is `sleep_for(1s)` at :61 for one `find("min:sec")`; **reduce the sleep** | :55 the one real assertion (`"2:30 min:sec\n"`); :61 the 1 s sleep |
| `tests/unit/unit_test_Data.cpp` | 308 | 6 | 80 | `Data` heap/view/f32 constructors, `series()`/`series_f32()` precision guard, `ndim==0` | contract/oracle | independent expected values + zero-copy pointer identity | `unit_test_multivariate_data.cpp` (overlapping `ndim`/`series_length`/`validate_ndim`); `tests/unit/core/unit_test_time_series.cpp` (`TimeSeriesView`) | keep; **absorb `unit_test_multivariate_data.cpp`** | :134 `view.series(0).data() == parent.series(1).data()`; :279-307 A12 precision guard |
| `tests/unit/unit_test_DataLoader.cpp` | 498 | 6 | 77 | `DataLoader` builder state, F21 legacy aliases, `Ndata` contract, delimiter/case, verbosity, cache-path uniqueness | mixed(contract/oracle, capability-guard) | canonical-vs-legacy cross-route + poison/restore of globals | `test_problem_api_2_0.cpp:1370-1410` (entities 26/27 re-test `startColumn`/`startRow`); F21 gate `tests/CMakeLists.txt:232-247` | keep | :235-358 F21 case + marker; :443-455 `Ndata` truth table |
| `tests/unit/unit_test_Problem.cpp` | 192 | 5 | 30 | `Problem` construction from loader; silhouette write-back error taxonomy | mixed(trivial, contract/oracle) | recorded ground truth 13 (duplicated) / typed-exception discrimination | **`unit_test_warping.cpp:40-58,187-210` is byte-identical** to `unit_test_Problem.cpp:63-107` | merge — delete the duplicated `dtwFull_L_test`/`dtwBanded_test` here; keep :109-192 (silhouette guard, audit D2) | `diff` of `unit_test_Problem.cpp:63-107` vs `unit_test_warping.cpp:40-58;187-210` differs only by one blank line and the `TEST_CASE` header placement |
| `tests/unit/unit_test_Problem_phase0.cpp` | 172 | 3 | 12 | `writeMedoids` throws `runtime_error`; distance-matrix properties; `Data::size()` | mixed(adversarial, trivial) | properties (symmetry/diagonal) + `Data::size()` tautologies | `unit_test_distance_matrix_properties.cpp` (same 3 properties, larger N); `unit_test_Data.cpp` (`size()`) | merge into `unit_test_Problem.cpp` — only :58-84 (the `throw 1` regression) is unique; :89-126 and :131-171 are strict subsets | :80 `REQUIRE_THROWS_AS(cluster_and_process(), std::runtime_error)`; :156-171 `size()` "usable as int and size_t" |
| `tests/unit/unit_test_accuracy.cpp` | 587 | 39 | 64 | cross-variant / cross-metric / early-abandon / stability / lower-bound / MV / fuzz | mixed(contract/oracle, trivial) | hand-computed values in §2; **self-consistency** in §1, §3, §7, §8 | §5 ⊂ `adversarial/test_lower_bounds_adversarial.cpp` + `core/unit_test_lower_bounds.cpp`; §7-8 ⊂ `adversarial/test_dtw_mathematical_properties.cpp`; §6 ⊂ `unit_test_multivariate_dtw.cpp` | **merge** — keep §2 (:123-191 hand-computed SquaredL2) and §3 (early abandon, unique); §5/§7/§8 (14 cases) are covered | :348-424 vs `test_lower_bounds_adversarial.cpp:115-479`; :551-587 vs `test_dtw_mathematical_properties.cpp:50-368` |
| `tests/unit/unit_test_adtw.cpp` | 983 | 47 | 72 | `adtwFull_L`, `adtwBanded`, ADTW+Pruned integration | contract/oracle + adversarial | **independent full-matrix DP reference** at :325-375, plus hand-computed values | `unit_test_mv_variants.cpp:112-193` (MV ADTW ndim=1 parity) | keep — the reference DP makes this a genuine oracle test, not self-consistency | :325-367 reference; :382-397 cross-check sweep; :877-905 LB≤DTW≤ADTW chain |
| `tests/unit/unit_test_arow_dtw.cpp` | 800 | 57 | 79 | `dtwAROW`, `dtwAROW_L`, `dtwAROW_banded`, unified-kernel parity, MV AROW | mixed(contract/oracle, self-consistency) | hand-computed DP traces (:384-518) for ~7 cases; **self-consistency** for ~20 `dtwAROW == dtwAROW_L` / symmetry / non-negativity cases | `adversarial/test_arow_adversarial.cpp` (20 cases: `Linear-space == full-matrix` for every NaN pattern, plus 200 random pairs) | **merge** — :332-378 (6 cases) is a strict subset of `test_arow_adversarial.cpp:141-268`; keep the hand-computed block and the MV/kernel-policy blocks | :332-378 vs `test_arow_adversarial.cpp:141-268` |
| `tests/unit/unit_test_benders.cpp` | 289 | 7 | 24 | Benders decomposition through `Problem::cluster()` | integration | hand-computed optimal cost (4.0) + Benders-vs-compact cross-route | `unit_test_mip.cpp:553-610` (`MIP Benders: forced on`, `cost matches direct HiGHS`) — same contract, same solver | **merge into `unit_test_mip.cpp`** — 5 of 7 cases duplicate it; see §3 for the dead second escape hatch | :102-106 `require_highs_solver` SKIP; :117-120 unreachable `has_solution` WARN-and-return; :269-288 "auto mode selects correctly" asserts nothing about selection |
| `tests/unit/unit_test_checkpoint.cpp` | 609 | 16 | 89 | `save_checkpoint`/`load_checkpoint`, automatic mid-fill checkpointing, crash resume | mixed(contract/oracle, integration) | poisoned-sentinel discrimination (900+10i+j) + bit-exact matrix compare | :129-328 (round-trip, partial, metadata fields, mismatch, missing, overwrite) ⊂ `unit_test_checkpoint_robustness.cpp:529-599,684-762` | **merge** — the unique content is :368-609 (auto-checkpoint schedule, resume, interval/mmap/empty-dir rejection, failing save). The first seven cases are the weak form of the robustness file's contract and carry the whole 51.98 s (16 `DataLoader` reads of `data/dummy`) | :129-328 vs `unit_test_checkpoint_robustness.cpp:529-762`; 51.98 s / 265 assertions |
| `tests/unit/unit_test_checkpoint_binary.cpp` | 586 | 2 | 32 | binary-v1 `ClusteringResult` wire format (F51) | mixed(fingerprint, adversarial, perf-fence) | **72 recorded wire bytes** + an 85-case corruption corpus + an allocation probe | none | keep — best file in scope. One marker-overclaim note in §3 | :93-95 wire bytes; :283-351 corpus; :522-550 size-preflight allocation probe |
| `tests/unit/unit_test_checkpoint_robustness.cpp` | 1040 | 9 | 51 | dense CSV checkpoint identity/integrity/transaction (M49) | adversarial | independent SHA-256 recomputation + full-state snapshot/restore comparison | `unit_test_checkpoint.cpp:129-328` (weaker subset) | keep — 16 identity axes, 25 metadata mutations, 7 CURRENT mutations, 14 CSV mutations, each with a state-unchanged post-condition; 1.62 s | :612-652 identity axes; :697-747 metadata mutations; :229-251 `state_matches` |
| `tests/unit/unit_test_cli_args.cpp` | 754 | 27 | 149 | `dtwc_cl` parse/validate helpers, Parquet RAM planner, storage routing, seed, deprecation registry, resume validation | contract/oracle | exact expected strings + recorded medoids/labels/costs | `tests/integration/test_cli_*.cmake` (real binary); `test_tier1_cpp_api.cpp:204` (same seed fixture) | keep — but **belongs in a `cli/` subdirectory**; it `#include`s `dtwc/dtwc_cl.cpp` with `DTWC_CL_NO_MAIN` (:33-34) | :128-161 exact validator messages; :394-403 recorded seed results |
| `tests/unit/unit_test_cli_checkpoint.cpp` | 214 | 2 | 15 | real `dtwc_cl` binary: `--checkpoint` / `--checkpoint-interval` | integration | real-binary stdout + exit code | `tests/integration/test_cli_resume_state.cmake` (F17, binary checkpoint resume) | keep — the only *unit*-dir test that spawns the built CLI; **move to `tests/integration/`** for consistency with F14/F17/CLI-config | :50-68 `executable_directory()`; :174 `"No valid checkpoint found"`; :181 `"Resumed from checkpoint"` |
| `tests/unit/unit_test_clustering_algorithms.cpp` | 413 | 14 | 49 | Lloyd k-medoids convergence/labels/medoids/cost, `init::random`/`Kmeanspp`/`Kmeanspp_seeded`, capped Lloyd | mixed(contract/oracle, trivial) | recorded medoids `{1,5}` + labels + cost 96.0 (:370-413); the rest are range/typing checks | `unit_test_problem_encapsulation.cpp:138-155` runs the **same** capped-Lloyd fixture and asserts the **same** `{1,5}` / labels / 96.0 | **merge** — :108-239 (7 cases) are property checks any correct or badly-wrong implementation passes ("labels in [0,k)", "cost >= 0"); they cost 57.53 s because each rebuilds a `Problem` from `data/dummy`. Keep :267-342 (degenerate k-means++ / signed Soft-DTW weights) and :370-413 | :179-198 `Multiple repetitions pick the best` asserts only `cost >= 0` for both runs; 57.53 s for 82 assertions |
| `tests/unit/unit_test_ddtw.cpp` | 223 | 16 | 31 | `derivative_transform`, `ddtwBanded`, `ddtwFull_L` | mixed(contract/oracle, self-consistency) | hand-derived derivative values (:32-46) for 6 cases; **self-consistency** for the 5 "equivalence to manual derivative + dtwBanded" cases (they recompute the implementation's own composition) | `unit_test_multivariate_dtw.cpp:330-364` (`derivative_transform_mv` ndim=1 parity) | keep — the derivative oracle is genuinely independent; note that :137-207 only proves `ddtw == dtw ∘ derivative`, which is the definition | :32-46 hand-derived; :137-154 composition identity |
| `tests/unit/unit_test_deferred_allocation.cpp` | 152 | 7 | 18 | `Problem::set_data` does not allocate O(N²); lazy `dist_by_ind` | contract/oracle | direct state observation (`dense_distance_matrix().size() == 0`) | `unit_test_wave2a_integration.cpp:149-205,557-637` (same contract, N=5000 and N=500) | **merge into `unit_test_wave2a_integration.cpp`** — every assertion here appears there at larger N | :35 vs `wave2a:579`; :127-152 vs `wave2a:607-637` |
| `tests/unit/unit_test_deterministic_series.cpp` | 390 | 6 | 59 | `tests/support/deterministic_series.hpp` byte contract + consumer source audit (F15) | mixed(fingerprint, implementation-detail) | recorded SHA-256 + recorded IEEE bit patterns; **source-text audit** at :292-390 | none | keep+tolerance — see §3 and §5: :318-349 asserts **exact call counts inside four sibling test files** (`46`, `7`, `18`, `2`, `2`, `1`, `1`, `2`), so any edit to those files breaks this gate | :47-69 recorded bits; :319-321 `Consumer{"tests/unit/test_cuda_correctness.cpp", 46}` |
| `tests/unit/unit_test_distance_matrix_properties.cpp` | 244 | 9 | 16 | distance-matrix diagonal / symmetry / non-negativity / idempotence + LB-strategy invariance | mixed(contract/oracle, trivial) | independent `dtwFull` recomputation (:176) + LB-strategy cross-route (:186-221) | :52-119 ⊂ `unit_test_Problem_phase0.cpp:89-126`; ⊂ `adversarial/test_dense_distance_matrix_adversarial.cpp` | **merge** — the unique, load-bearing case is :186-221 (`LowerBoundStrategy variants yield identical results`, 5 strategies × 25×25). Cases 1-4, 6, 9 re-derive `dtwFull`'s own symmetry. **115.98 s is the worst in scope** | 115.98 s / 3369 assertions; 12 `DataLoader` reads of `data/dummy` (LastTest.log:2600-2626) |
| `tests/unit/unit_test_dtw_function_semantics.cpp` | 230 | 5 | 22 | `Problem::dtw_function()` / `dtw_function_f32()` rebind and stale detection (M37) | contract/oracle | registered oracle 3.0→4.0 + storage-alternative identity | `unit_test_variant_distmat.cpp:88-243` (same stale-cache rule via `dist_by_ind`); `test_problem_api_2_0.cpp:644-666` (same 3.0→4.0 oracle) | keep | :74 registered band; :143-158 stable callable-address (allocation-freedom proxy) |
| `tests/unit/unit_test_dtw_variants.cpp` | 309 | 13 | 43 | identity / symmetry / band monotonicity / empty / single element / hand-computed | contract/oracle | one hand-computed matrix (:216-226), one derived diagonal cost (:116-129); the rest is self-consistency across the library's own three entry points | **`adversarial/test_dtw_mathematical_properties.cpp:50-464` covers the first 12 cases** with more inputs | **delete** — identity/symmetry/non-negativity/empty/single/known-value → `test_dtw_mathematical_properties.cpp:50-368`; wide band == full → :399; band=-1 == full → :417; band=0 diagonal → :429; `dtwFull` vs `dtwFull_L` → :380. The **one** case with no home is :279-308 (`WDTW rejects a weight array shorter than max(|x|,|y|)`, A6) — move it to `unit_test_wdtw.cpp` | side-by-side of the two TEST_CASE lists |
| `tests/unit/unit_test_fileOperations.cpp` | 487 | 14 | 48 | `ignoreBOM`, batch/folder loaders, strict NaN parser, distance-matrix CSV round-trip, UTF-8 names | mixed(contract/oracle, adversarial) | independent re-parse of the written CSV + exact error positions ("row 1, column 3") | `unit_test_DataLoader.cpp` (loader state/verbosity); `core/unit_test_distance_matrix_csv.cpp` (F14 byte contract) | keep+tolerance — see §3 (writes into the source root) and §4 (11.28 s from `GENERATE(1,2,10,1000)²`) | :95 / :229-230 / :378-379 generators; :108 `fs::path tempFilePath = "test_distmat.csv"` (relative → repo root) |
| `tests/unit/unit_test_mip.cpp` | 874 | 17 | 109 | MIP settings, warm-start locality, exact-publication transaction (M33), HiGHS/Benders end-to-end | mixed(contract/oracle, integration, trivial) | recorded warm-start medoids/cost + full-configuration snapshot/restore + Benders-vs-direct cross-route | `unit_test_benders.cpp` (5/7 cases duplicate); `tests/unit/mip/test_mip_backend_guards.cpp` | keep (absorb `unit_test_benders.cpp`) — but see §3: :512-524 and :854-874 are `REQUIRE_NOTHROW` smoke tests whose names promise dispatch verification | :283-303 default echo (trivial); :344-433 M33 transaction (strong); :765-769 exact stdout pins |
| `tests/unit/unit_test_missing_dtw.cpp` | 517 | 35 | 51 | `dtwMissing*` family + `MissingStrategy::Error` on the pairwise entry points | mixed(contract/oracle, self-consistency) | hand-computed DP traces for ~8 cases; self-consistency for the `dtwMissing == dtwMissing_L` and symmetry families | `unit_test_arow_dtw.cpp:154-190` (AROW ≥ ZeroCost); `unit_test_mv_missing.cpp` (MV form) | keep — :433-517 (A3 pre-scan diagnostic, A4 `Error` on the free functions) is unique and high value | :346-388 hand-computed; :477-517 A4 regression |
| `tests/unit/unit_test_missing_utils.cpp` | 200 | 21 | 43 | `is_missing`, `has_missing`, `missing_rate`, `interpolate_linear` | contract/oracle | independent expected values | `adversarial/test_missing_utils_adversarial.cpp` | keep — small, fast (0.86 s), fully independent oracles | :145-186 LOCF/NOCB expected values |
| `tests/unit/unit_test_mpi.cpp` | 219 | 0 | 19 (`MPI_CHECK`) | `mpi::compute_distance_matrix_mpi` | integration | serial `dtwFull_L`/`dtwBanded` cross-route via shared support | `unit_test_mpi_allreduce_chunking.cpp` (host-side chunking math) | keep+tolerance — see §3: with `DTWC_ENABLE_MPI=OFF` `main()` prints one line and returns 0, so **CTest records a pass with zero assertions and no Catch2 summary**; and the registered command is single-process although the file documents `mpiexec -n 4` | :213-218 the no-op `main()`; LastTest.log:1888 `"MPI not enabled ... Skipping MPI tests." / Test Passed` |
| `tests/unit/unit_test_mpi_allreduce_chunking.cpp` | 140 | 6 | 23 | `allreduce_chunk_count` / `max_allreduce_chunk` INT_MAX chunking | contract/oracle | independent arithmetic + loop replay | none | keep — MPI-free by design, so it actually runs; pins the `N=46341` overflow | :115-139 `N*N` for N=46341 |
| `tests/unit/unit_test_multivariate_data.cpp` | 109 | 11 | 21 | `Data::ndim`/`series_length`/`validate_ndim`, `TimeSeriesView` | contract/oracle | independent expected values | `unit_test_Data.cpp:90-98,271-307`; `core/unit_test_time_series.cpp` | **merge into `unit_test_Data.cpp`** (Data half) and `core/unit_test_time_series.cpp` (view half) | :19-34 vs `unit_test_Data.cpp:90-98`; :73-108 view cases |
| `tests/unit/unit_test_multivariate_dtw.cpp` | 364 | 26 | 36 | MV metric functors, `dtwFull_L_mv`, `dtwBanded_mv`, `derivative_transform_mv`, Problem MV | mixed(contract/oracle, perf-fence) | hand-computed 2D/3D DP traces | `unit_test_wave2b_integration.cpp:98-222` (ndim=1 parity for every MV variant) | keep, **delete :244-281** — `MV DTW: D=1 performance parity` runs 2×1100 DTWs on 200-length series and states "We don't assert on timing"; it is 14.24 s of a 26-case, 47-assertion target | :279-280 "just print for observation"; 14.24 s |
| `tests/unit/unit_test_mv_lower_bounds.cpp` | 222 | 16 | 31 | `compute_envelopes_mv`, `lb_keogh_mv`, `lb_keogh_squared`, `lb_keogh_mv_squared` | contract/oracle | hand-computed per-channel envelopes (:44-58) + LB≤DTW | `unit_test_wave2b_integration.cpp:227-336` (LB≤DTW for ndim=2/3) | keep | :44-58 hand-computed windows; :166-173 known LB value |
| `tests/unit/unit_test_mv_missing.cpp` | 542 | 30 | 48 | `MissingMVL1Dist`/`MissingMVSquaredL2Dist`, `dtwMissing_*_mv`, MV+Interpolate rejection | mixed(contract/oracle, self-consistency) | hand-computed L1/SquaredL2 (:195-244) and the 3-4-5 L2 triangle (:441-508); self-consistency for symmetry/non-negativity | `unit_test_mv_variants.cpp:499-519` (same bind-time rejection pattern for SoftDTW) | keep — :503-542 (explicit-L2 contract, MV+Interpolate rejection) is unique and adversarial | :479-485 L1=28 / L2=20 / SqL2=100 controls |
| `tests/unit/unit_test_mv_variants.cpp` | 519 | 32 | 37 | MV WDTW/ADTW/DDTW, `dtw_runtime` variant dispatch, `dtw_function_f32` dispatch | mixed(contract/oracle, adversarial) | ndim=1 parity with the scalar kernels + shifted-peak discriminators | `unit_test_wave2b_integration.cpp:98-222` (overlapping ndim=1 parity) | keep — :362-519 (dispatch regressions: `dtw_runtime` dropped the variant; `dtw_function_f32` silently ran Standard) is exactly the silent-wrong-answer class | :325-354 banded discriminators; :406-488 f32 dispatch |
| `tests/unit/unit_test_parallelisation.cpp` | 165 | 7 | 13 | `run_openmp`, `run`, `omp_chunk_size`, lowest-index typed rethrow (M40) | mixed(trivial, adversarial) | expected exception identity by canonical loop index | `core/unit_test_run_thread_scope.cpp`; `unit_test_problem_missing.cpp:72-78` names this file as the owner of the rethrow contract | keep — :105-165 is the whole value; :26-103 (5 cases) only check that a `results[i]=1` loop ran | :155-156 `REQUIRE_THROWS_WITH(..., "failure at row 2")` with a forced wall-clock inversion |
| `tests/unit/unit_test_problem_encapsulation.cpp` | 156 | 3 | 42 | F19 `Problem` accessor/setter round-trip, transactional rejection, capped Lloyd | contract/oracle | recorded defaults + recorded Lloyd result `{1,5}` / cost 96.0 | `unit_test_clustering_algorithms.cpp:370-413` (same fixture, same recorded values) | keep — F19-gated; the duplicate is on the `unit_test_clustering_algorithms.cpp` side | :146-151 vs `unit_test_clustering_algorithms.cpp:382-387` |
| `tests/unit/unit_test_problem_missing.cpp` | 215 | 6 | 19 | `Problem::missing_strategy` wiring, failed-fill cache invariant (M40) | contract/oracle | hand-computed distances (1.0/4.0/5.0) + partial-cache post-condition | `unit_test_missing_dtw.cpp:433-468` (pre-scan diagnostic); `unit_test_variant_distmat.cpp:118-130` | keep — the "a rejected fill does not publish a full cache" invariant at :79-114 is unique | :110-111 `count_computed() < packed_count()`; :140-142 recorded fingerprints |
| `tests/unit/unit_test_problem_storage_policy.cpp` | 745 | 5 | 133 | F20 `Problem::set_storage_policy` + `set_data` routing, move semantics, transactionality, `.dtws` artifact bytes | mixed(contract/oracle, fingerprint) | **independent DP oracle** (`independent_dependent_l1`, :207-238) + recorded bit patterns + recorded `.dtws` header bytes | `test_storage_policy.cpp` (DataLoader side of the same rule) | keep — but see §3 (`ProblemStoragePolicyTestAccess` reads the private `series_storage_owner_`) and §5 (needs the CMake `TMP` redirect to run) | :47-53 private-state access; :66-82 recorded pair bits; :289-322 `.dtws` byte contract |
| `tests/unit/unit_test_scores_new.cpp` | 297 | 20 | 22 | Dunn, inertia, Calinski-Harabasz, ARI, NMI | contract/oracle | hand-computed values from a 4-point fixture (Dunn 4.0, inertia 2.0, CH 34.0) | `adversarial/test_scores_adversarial.cpp:692-1067` covers NMI (9 cases), Dunn (6), inertia (6) including hand-computed values and degenerate cases | **merge** — the unique content is the CH block (:146-202; no CH case exists elsewhere) and the ARI block (:207-246; also absent there). The Dunn/inertia/NMI blocks (11 cases) are covered | :69-141 vs `test_scores_adversarial.cpp:803-1067`; :251-297 vs :692-801 |
| `tests/unit/unit_test_scores_phase0.cpp` | 91 | 3 | 4 | Davies-Bouldin positive / throws / ordering | contract/oracle | ordering comparison (good < bad), no closed-form value | `adversarial/test_scores_adversarial.cpp:457-486`; `unit_test_scores_single_cluster.cpp` | **delete** — `dbi > 0` → `test_scores_adversarial.cpp:457`; `throws when not clustered` → `unit_test_scores_single_cluster.cpp:77-88` (which additionally pins the *type*); `bad > good` → `test_scores_adversarial.cpp:471`. No assertion is unique | :62-66, :73, :90 vs the three cited cases |
| `tests/unit/unit_test_scores_single_cluster.cpp` | 89 | 3 | 6 | R4(a) `Nc<2` guard for DBI and Dunn | adversarial | typed-exception discrimination (`InvalidInput`, not a silent 0/inf) | `test_scores_adversarial.cpp:486` (`DBI: single cluster throws`), :838 (`Dunn: unclustered throws`) | keep — the only place that separates the `Nc<2` guard from the "cluster first" guard (:77-88) | :66, :74, :87-88 |
| `tests/unit/unit_test_simd.cpp` | 389 | 16 | 20 | `core::lb_keogh`, `core::z_normalize` vs scalar references | contract/oracle | **independent scalar reimplementations** in-file (:37-77, :120-134) | `unit_test_z_normalize.cpp` (same function, closed-form values); `core/unit_test_lower_bounds.cpp` | keep — but `dtw_scalar` at :80-102 is **dead code** (never called; the multi-pair SIMD tests were removed, see :698) | :80-102 unused; :138-140 tail-handling length table |
| `tests/unit/unit_test_soft_dtw.cpp` | 446 | 24 | 42 | `softmin_gamma`, `soft_dtw`, `soft_dtw_gradient`, unified-kernel parity | contract/oracle | closed-form softmin (`a - γ·log 3`), **finite differences** for the gradient, step-by-step DP recomputation | `test_error_taxonomy.cpp:107-126` (empty-input throw on the same function) | keep — finite-difference gradient checks are a real independent oracle | :101-104 closed form; :303-321 finite differences |
| `tests/unit/unit_test_soft_dtw_hotpath.cpp` | 110 | 2 | 6 | `detail::softmin_gamma_unchecked` noexcept + allocation-freedom; warmed gradient reuses buffers (M46) | perf-fence | global `operator new` counting | `unit_test_checkpoint_binary.cpp:38-85` (a second, differently parameterised `operator new` probe) | keep; **hoist the allocation probe into shared support** | :50-63 global `operator new`; :84 `allocations == 0` |
| `tests/unit/unit_test_variant_distmat.cpp` | 636 | 13 | 66 | dense/mmap distance-matrix variant, stale-cache invalidation, mmap fingerprint axes | mixed(adversarial, contract/oracle, perf-fence) | fingerprint-mismatch discrimination + recorded distances | `unit_test_dtw_function_semantics.cpp` (dispatcher half of the same rule); `unit_test_variant_precision.cpp:321-370` | keep — but see §4: :553-596 (`Warm mmap cached lookups remain O(1)`) is 2×100 000 lookups and is the bulk of 17.52 s | :88-201 stale-cache axes; :332-429 fingerprint axes; :594 the timing band |
| `tests/unit/unit_test_variant_precision.cpp` | 463 | 7 | 49 | float32 representability rejection, transactional `set_variant`/`set_data`, mmap preflight (M45) | mixed(contract/oracle, implementation-detail) | exact message + full dense-cache snapshot comparison | `core/unit_test_variant_domains.cpp` (M34 domain validation) | keep, **drop :401-408** — `Problem float32 callable uses stay behind validated access` greps `dtwc/Problem.cpp` for `dtw_fn_f32_(` and `validated_dtw_function_f32()`; that is a source-text assertion on a private member name | :102-139 the 12 narrowing cases; :404-408 the source grep |
| `tests/unit/unit_test_warping.cpp` | 307 | 18 | 61 | `dtwFull`, `dtwFull_L`, `dtwBanded`, early abandon, SquaredL2, pointer overloads | mixed(contract/oracle, trivial) | recorded ground truths 13 / 35 + pointer-vs-vector cross-route | `adversarial/test_dtw_mathematical_properties.cpp`; `core/unit_test_dtw_api.cpp:43-206`; `unit_test_Problem.cpp:63-107` (byte-identical) | keep the early-abandon (:60-110) and pointer-API (:244-308) blocks; the identity/known-value blocks are triple-covered | duplicate proof under `unit_test_Problem.cpp` above |
| `tests/unit/unit_test_warping_phase0.cpp` | 126 | 2 | 15 | `dtwBanded` default-template-parameter deduction; DTW metric properties | mixed(adversarial, trivial) | `dtwFull` as ground truth | `test_dtw_mathematical_properties.cpp:50-368` covers :66-125 entirely | **merge into `unit_test_warping.cpp`** — only :32-61 (deduced vs explicit `dtwBanded<double>` template argument) is unique, and it is 3 assertions | :49-60 the unique case; :66-125 the covered one |
| `tests/unit/unit_test_wave1a_integration.cpp` | 498 | 8 | 53 | full missing-data pipeline → clustering → all 5 metrics | integration | ground-truth ARI/NMI on synthetic separated clusters | `unit_test_wave2a_integration.cpp`, `unit_test_wave2b_integration.cpp` (same shape) | keep, **delete :438-498** — the perf case runs 3 000 DTWs on length-100 series for a `ms_arow < ms_zero*20 + 100` band; it is essentially all of the target's 4.55 s | :494-497 the band; :479-483 the print |
| `tests/unit/unit_test_wave2a_integration.cpp` | 763 | 15 | 101 | deferred allocation, FastCLARA fixes, hierarchical, CLARANS | integration | independent recomputation of labels/medoids/costs + state observation | `unit_test_deferred_allocation.cpp` (subset); `algorithms/unit_test_fast_clara.cpp`, `unit_test_hierarchical.cpp`, `unit_test_clarans.cpp` | keep (absorb `unit_test_deferred_allocation.cpp`); overlap with `tests/unit/algorithms/` is out of my scope and needs that agent's ledger | :557-602 parent-matrix non-allocation |
| `tests/unit/unit_test_wave2b_integration.cpp` | 762 | 20 | 75 | MV cross-variant ndim=1 parity, MV LB validity, MV pipelines ndim=3 | mixed(integration, contract/oracle) | scalar-kernel parity | `unit_test_mv_variants.cpp`, `unit_test_mv_lower_bounds.cpp`, `unit_test_multivariate_dtw.cpp` (the ndim=1 parity block :98-222 is duplicated across all four) | keep, **delete :656-762** — two perf cases; :707-708 asserts `ms1 >= 0.0` and `ms3 >= 0.0`, which cannot fail | :707-708 vacuous; :98-222 vs `unit_test_mv_variants.cpp:27-193` |
| `tests/unit/unit_test_wdtw.cpp` | 218 | 14 | 21 | `wdtwFull`, `wdtwBanded` | mixed(contract/oracle, self-consistency) | one full hand-computed DP (:61-83) + the derived `g=0 ⇒ WDTW = DTW/2` identity | `unit_test_accuracy.cpp:46-74` (uniform weights == DTW); `unit_test_mv_variants.cpp:27-110` | keep (receive the A6 weight-length case from `unit_test_dtw_variants.cpp:279-308`) — but :97-125 (`large g penalizes off-diagonal`) asserts only finiteness and "differs from g=0", not the property in its name | :61-83 hand-computed; :118-124 the weak assertions |
| `tests/unit/unit_test_z_normalize.cpp` | 177 | 9 | 15 | `core::z_normalize` / `z_normalized` | contract/oracle | closed-form values ({2,4,4,4,5,5,7,9} → mean 5, σ 2) | `unit_test_simd.cpp:253-387` (same function against an in-file scalar reference, more lengths) | **merge into `unit_test_simd.cpp`** — the closed-form cases (:53-64, :154-162) are the unique part and belong beside the scalar-reference sweep; the mean/σ property cases duplicate `unit_test_simd.cpp:270-293` | :69-88 vs `unit_test_simd.cpp:270-293` |
| `tests/unit/gpu_fixed_band_oracle.hpp` | 169 | 0 | 0 | test-only fixed-band DTW oracle (full-matrix DP + exhaustive path enumeration + recorded ledger) | (support) | independent by construction (shares no production helper) | used by `test_cuda_correctness.cpp`, `test_metal_correctness.cpp`, `unit_test_deterministic_series.cpp` | keep; **move to `tests/support/`** — it is shared support living in a test directory | :1-13 header docstring; :70-86 recorded ledger |
| `tests/support/deterministic_series.hpp` | 122 | — | — | portable deterministic generators + dense symmetric assembly | (support) | portability argued from `std::mt19937` + exact power-of-two scaling | consumers pinned by `unit_test_deterministic_series.cpp:300-349` | keep — genuinely shared (5 benchmarks + 6 tests) | :28-58 the exactness argument |
| `tests/test_util.hpp` | 102 | — | — | `get_random_data`, `get_random_names`, `write_data_to_folder`, `write_data_to_file` | (support) | none | used by `unit_test_fileOperations.cpp`, `unit_test_checkpoint.cpp`, `unit_test_clustering_algorithms.cpp`, `unit_test_distance_matrix_properties.cpp` | keep+tolerance — it draws from the **global mutable** `dtwc::randGenerator` (:28, :45), so its output depends on whatever else in the binary touched that engine; `tests/support/deterministic_series.hpp` exists precisely to avoid that | :28 `dis(randGenerator)`; contrast `deterministic_series.hpp:64` (fresh engine per call) |

---

## 2. Cross-file duplication of test scaffolding

Seven distinct groups. All are candidates for one shared `tests/support/` library.

**(a) Temp/scratch directory RAII — 15 independent implementations**
`test_env_device.cpp:74` `make_scratch_dir()` · `test_storage_policy.cpp:51` `make_scratch_dir(tag)` ·
`test_tier1_cpp_api.cpp:489` `make_sandbox(tag)` · `unit_test_checkpoint.cpp:55` `make_temp_dir(suffix)`
plus :65 `cleanup_dir` · `unit_test_checkpoint_binary.cpp:105` `struct ScratchDirectory` ·
`unit_test_checkpoint_robustness.cpp:63` `struct ScratchDirectory` · `unit_test_cli_args.cpp:190`
`struct ScratchDirectory` · `unit_test_variant_distmat.cpp:34` `struct ScratchCache` ·
`unit_test_variant_precision.cpp:141` `struct ScratchCache` (byte-identical to the previous) ·
`unit_test_dtw_function_semantics.cpp:40` `struct ScratchCaches` ·
`unit_test_clustering_algorithms.cpp:61` `struct TemporaryOutputDirectory` ·
`unit_test_problem_encapsulation.cpp:47` `struct OutputDirectory` ·
`unit_test_fileOperations.cpp:37` `struct TemporaryBatchFile` · `unit_test_DataLoader.cpp:402`
`struct SeriesFolder` · `unit_test_wave1a_integration.cpp:77` / `wave2a:49` / `wave2b:82` `g_tmp_dir()`.

Three different uniqueness strategies coexist: a `this`-pointer cast
(`unit_test_variant_distmat.cpp:42`, `unit_test_checkpoint_robustness.cpp:70`,
`unit_test_fileOperations.cpp:44`), a steady-clock nonce (`unit_test_mip.cpp:616`,
`test_tier1_cpp_api.cpp:491`), and a static counter (`test_env_device.cpp:76`).
`.claude/LESSONS.md:395-401` records that the *production* version of the address-as-entropy trick was
a real bug (`default_series_cache_path`); the tests still use it.

**(b) stdout/stderr capture — 9 independent implementations**
`test_runtime_loudness_env.cpp:63` and `test_runtime_loudness_compute.cpp:67` (`capture_cerr`,
byte-identical) · `test_problem_api_2_0.cpp:188` `f22_cout_capture` · `test_tier1_cpp_api.cpp:477`
`ScopedCoutRedirect` · `unit_test_DataLoader.cpp:425` `CoutCapture` ·
`unit_test_checkpoint_robustness.cpp:84` `SilenceCout` · `unit_test_mip.cpp:91` `ScopedCoutCapture` ·
`unit_test_problem_storage_policy.cpp:374` `CerrCapture` · `test_storage_policy.cpp:239` an inline
`RestoreCout` struct.

**(c) Global-`operator new` allocation counters — 2, with different parameterisations**
`unit_test_soft_dtw_hotpath.cpp:15-63` (counts *all* allocations plus a ≥500 KiB class) and
`unit_test_checkpoint_binary.cpp:38-85` (counts allocations of exactly 1028 bytes). Both replace the
six global `operator new`/`delete` overloads. These are performance fences under the project rules; a
shared, parameterised probe would preserve both contracts.

**(d) CPU DTW reference matrices — 4 near-identical wrappers**
`test_cuda_correctness.cpp:48,59` · `test_metal_correctness.cpp:48,58` · `test_metal_lb_keogh.cpp:56` ·
`test_cuda_kernel_override.cpp:122`. All four call
`test_support::symmetric_zero_diagonal_matrix` with a `dtwFull_L`/`dtwBanded` lambda — only the
wrapper is duplicated, and `unit_test_deterministic_series.cpp:336-341` **pins how many times each
file calls it**.

**(e) Independent DP oracles — 5, all genuinely independent, all one-off**
`unit_test_adtw.cpp:325` `adtwBanded_reference` · `unit_test_problem_storage_policy.cpp:207`
`independent_dependent_l1` · `unit_test_simd.cpp:80` `dtw_scalar` (**dead — never called**) ·
`gpu_fixed_band_oracle.hpp:96` `full_matrix_oracle` · `test_decode_pair.cpp:44` `mpi_reference_decode`.
`gpu_fixed_band_oracle.hpp` is already shared by three files and is the natural home for a shared
`reference_dtw(x, y, band, metric, penalty)`; that would delete `unit_test_simd.cpp:80-102` and let
`unit_test_problem_storage_policy.cpp:207-238` reuse it.

**(f) `Problem` builders — 21 `make_*_problem` helpers**
`test_metal_mmap.cpp:45` · `test_problem_api_2_0.cpp:76,143` · `test_runtime_loudness_gpu.cpp:34` ·
`unit_test_Problem_phase0.cpp:36` · `unit_test_adtw.cpp:779` · `unit_test_benders.cpp:42` ·
`unit_test_checkpoint.cpp:46` · `unit_test_clustering_algorithms.cpp:42,79` ·
`unit_test_distance_matrix_properties.cpp:38` · `unit_test_dtw_function_semantics.cpp:24` ·
`unit_test_mip.cpp:33,57` · `unit_test_scores_new.cpp:40` · `unit_test_scores_phase0.cpp:26` ·
`unit_test_scores_single_cluster.cpp:41` · `unit_test_wave1a_integration.cpp:90` ·
`unit_test_wave2a_integration.cpp:65,84,119` · `test_storage_policy.cpp:197`.

Four of them (`unit_test_checkpoint.cpp:46`, `unit_test_distance_matrix_properties.cpp:38`,
`unit_test_clustering_algorithms.cpp:42`, `unit_test_variant_distmat.cpp:76`) are the same
`DataLoader{data/dummy, N}.start_column(1).start_row(1)` recipe and are the direct cause of the four
slowest targets in scope. Three more (`unit_test_mip.cpp:57`, `test_tier1_cpp_api.cpp:75`,
`unit_test_cli_args.cpp:369`) are the same eight-waveform *seed-sensitive* fixture with the same
recorded medoids `{6,2,5}` and cost 24.0.

**(g) Random-series generators — 6 local ones beside the mandated shared one**
`unit_test_simd.cpp:109` · `unit_test_accuracy.cpp:33` · `unit_test_wave1a_integration.cpp:55` ·
`unit_test_wave2b_integration.cpp:55` · `test_metal_lb_keogh.cpp:43` ·
`test_cuda_correctness.cpp:440` (`generate_random_walks`), while
`tests/support/deterministic_series.hpp` exists and is *required* of other files.
`unit_test_deterministic_series.cpp:356-363` explicitly **whitelists** `test_metal_lb_keogh.cpp`'s
local `uniform_real_distribution<double> dist(-5.0, 5.0)`, freezing the inconsistency into a gate.

---

## 3. Tests that test the wrong thing

### 3.1 The skip path is the common path

Canonical MSVC-Debug gate, from LastTest.log:

| target | canonical outcome | lines of test code that never compile |
|---|---|---|
| `test_cuda_correctness` | `test cases: 1 \| 1 skipped` / `assertions: - none -` | 1820 (:40-:1860) |
| `test_metal_correctness` | 1 skipped, no assertions | 645 (:39-:684) |
| `test_cuda_lb_keogh` | 1 skipped, no assertions | 314 |
| `test_metal_lb_keogh` | 1 skipped, no assertions | 209 |
| `test_metal_mmap` | 1 skipped, no assertions | 85 |
| `test_io_readers` | 1 skipped, no assertions | 511 |
| `test_cuda_kernel_override` | 2 cases, 1 passed / 1 skipped | 279 |
| `test_cuda_launch_guards` | 3 cases, 2 passed / 1 skipped | 68 |

That is ≈3 900 lines — 13 % of the scope — dark in the canonical gate, all scoring green because
`SKIP_RETURN_CODE 4` (`cmake/Coverage.cmake:32`) is left in place for these targets. F6 attaches its
anti-skip guard **only** under `DTWC_HAS_ARROW` (`tests/CMakeLists.txt:633`).

Two files fix this properly and are the pattern to copy: `test_cuda_launch_guards.cpp` and
`test_cuda_kernel_override.cpp` both expose a CUDA-free host seam via `__has_include` and `FAIL(...)`
if the seam disappears (`test_cuda_launch_guards.cpp:42-44`). `test_cuda_correctness.cpp:129` is the
clearest missed opportunity: a pure-host oracle case trapped inside `#else // DTWC_HAS_CUDA`.

### 3.2 A pass with literally zero assertions

`unit_test_mpi.cpp:213-218`: with `DTWC_ENABLE_MPI=OFF` the whole file reduces to
`int main() { std::cout << "MPI not enabled..."; return 0; }`. It is not a Catch2 binary at all, so
there is no summary line for any `PASS_REGULAR_EXPRESSION` to check. CTest recorded `Test Passed` in
0.71 s (LastTest.log:1883-1895).

### 3.3 Vacuously-true assertions

* `unit_test_wave2b_integration.cpp:707-708` — `REQUIRE(ms1 >= 0.0); REQUIRE(ms3 >= 0.0);` on
  `chrono::duration` counts. Cannot fail.
* `test_build_parallelism.cpp:73-79` — reads `_OPENMP` into a `constexpr bool` and asserts it, three
  lines after an `#error` (:44-46) that already made the alternative uncompilable.

### 3.4 Assertions on source text rather than behaviour

* `unit_test_variant_precision.cpp:404-408` greps `dtwc/Problem.cpp` for the private member spelling
  `dtw_fn_f32_(`.
* `test_supply_chain_pinning.cpp` (whole file) greps five build files; self-declared as a sentinel at
  :12-16, and duplicated by a real parser at `tests/CMakeLists.txt:400-620`.
* `unit_test_deterministic_series.cpp:300-379` greps five benchmarks and five test files and asserts
  **exact occurrence counts** (`46`, `7`, `18`, `2`, `2`, `1`, `1`, `2`). **This is the single largest
  obstacle to the campaign**: merging, splitting or editing `test_cuda_correctness.cpp`,
  `test_cuda_lb_keogh.cpp`, `test_metal_correctness.cpp`, `test_metal_lb_keogh.cpp`,
  `test_cuda_kernel_override.cpp` or `unit_test_mpi.cpp` breaks F15, and F15's
  `PASS_REGULAR_EXPRESSION` (`tests/CMakeLists.txt:664-673`) turns the break into a hard CTest failure.

### 3.5 Assertion on private state

`unit_test_problem_storage_policy.cpp:47-53` defines `dtwc::ProblemStoragePolicyTestAccess` to read
`Problem::series_storage_owner_`, which requires a `friend` declaration in production. The lifetime
invariant it checks is real (:85-99 proves the mapped names are not dangling after a move), but the
mechanism couples the test to a private member name.

### 3.6 The oracle is the code under test

* `unit_test_ddtw.cpp:137-207` — five cases assert
  `ddtwBanded(x,y,b) == dtwBanded(derivative(x), derivative(y), b)`, i.e. the definition of DDTW as
  implemented.
* `unit_test_arow_dtw.cpp:332-378` — six cases assert `dtwAROW == dtwAROW_L`: two implementations of
  the same recurrence in the same library.
* `unit_test_accuracy.cpp:476-545` — four "fuzz" cases assert `dtwFull == dtwFull_L == dtwBanded(100)`.

None is worthless (they catch divergence between siblings), but none can detect a shared error in the
recurrence, and each is already covered by `adversarial/test_dtw_mathematical_properties.cpp:380-417`
and `adversarial/test_arow_adversarial.cpp:141-268`.

### 3.7 Names that promise more than the body checks

* `unit_test_benders.cpp:269` `Benders auto mode selects correctly by size` — asserts only
  `centroids_ind.size() == 2` and `0 <= cost < 1e10`; nothing observes which route ran.
* `unit_test_mip.cpp:854` `MIP Benders: auto dispatches based on N threshold` — the comment at
  :857-859 admits "not tested here".
* `unit_test_mip.cpp:512` `settings propagate without crash` — one `REQUIRE_NOTHROW`.
* `unit_test_wdtw.cpp:97` `large g penalizes off-diagonal, wdtw >= dtw for shifted series` — asserts
  finiteness and `|wdtw(g=100) - wdtw(g=0)| > 1e-6`; the DTW comparison named in the title is computed
  at :113 and never used.
* `unit_test_clustering_algorithms.cpp:179` `Multiple repetitions pick the best (lowest) cost` — the
  body comment (:194-195) says the multi-rep run "may or may not beat the single run", and both
  assertions are `cost >= 0.0`.

### 3.8 A redundant escape hatch that can silently skip real assertions

`unit_test_benders.cpp` calls `require_highs_solver()` (SKIP if absent, :102-106) and *then also*
guards every body with `if (!has_solution(prob)) { WARN(...); return; }` (:117, :135, :157, :193,
:218, :256, :280). `has_solution` (:89-100) is a heuristic — "all centroids zero means the solver did
not run". After the SKIP the second guard is unreachable in intent but still live in code: if a future
defect makes the solver return all-zero medoids, the test warns and passes. Same pattern at
`unit_test_mip.cpp:478, :495, :537, :570, :595, :836`.

### 3.9 A gate marker that asserts counts its own test case never established

`unit_test_checkpoint_binary.cpp:580-585` prints
`F51_BINARY_CHECKPOINT ... valid_bytes=72/72 fields=5/5 resave=72/72 semantic_compat=7/7 ...` from the
*corruption* case, gated only on `corpus_green && preflight_green`. The
`valid_bytes`/`fields`/`resave`/`semantic_compat` numbers are established by the sibling case
(:373-463) and are not observable at the print site — the comment at :577-579 acknowledges this. The
run is still sound because the sibling's `CHECK`s fail the binary, but the marker text that
`tests/CMakeLists.txt:168` matches is not evidence for four of its own fields.

### 3.10 Aggregated booleans destroy failure diagnosis

`test_problem_api_2_0.cpp:723-1568` evaluates 33 compatibility entities as `bool` expressions
(e.g. :746-753 ANDs six sub-conditions) and feeds each to `f22_keyed_ledger::record` (:443-453), where
the only assertion is `CHECK(passed)`. A regression in any one of ~150 sub-conditions reports as
"`CHECK(passed)` failed" at :452.

---

## 4. Slow tests (> 10 s, MSVC Debug)

| target | t (s) | assertions | what dominates | smaller fixture? |
|---|---|---|---|---|
| `unit_test_distance_matrix_properties` | **115.98** | 3369 | 12 `DataLoader` reads of `data/dummy` (LastTest.log:2600-2626) plus the N=25 × 5-strategy × 25×25 comparison loop at :186-221 (3125 of the 3369 assertions) | Yes. LB-strategy invariance is a property of the pruning cascade, not of N: N=8 with band=2 exercises the same branches. The other 8 cases (N=5..10, :52-244) each rebuild the loader; one shared in-memory fixture removes almost all of the cost |
| `unit_test_clustering_algorithms` | **57.53** | 82 | 11 separate `make_dummy_problem(10, ·)` calls, each re-reading 10 CSVs and re-running Lloyd (:42-59) | Yes. Only :370-413 needs a real Lloyd run; the seven property cases at :108-239 can share one clustered `Problem` |
| `unit_test_checkpoint` | **51.98** | 265 | 16 `make_problem(N)` calls, each a `DataLoader` read of `data/dummy` (:46-52), plus SHA-256 generation publishing per save | Yes — `unit_test_checkpoint_robustness.cpp` covers the same format in 1.62 s using in-memory `Data` (:43-51). Switching `make_problem` to in-memory series keeps every assertion |
| `test_problem_api_2_0` | **18.76** | 229 | Not the C++ at all: `tests/CMakeLists.txt:125-140` prepends `scripts/test_f22_cpp_deprecations.py --launch`, which drives `cmake --build` on three `EXCLUDE_FROM_ALL` object probes (`tests/CMakeLists.txt:45-66`) before the test binary runs | No — the compile probe *is* the contract. It could be split into its own CTest entry so the C++ half is not serialised behind it |
| `unit_test_variant_distmat` | **17.52** | 97 | :553-596 runs 2 × 100 000 `dist_by_ind` lookups; plus 4 `DataLoader` reads of the 25-series folder | Partly. The O(1) fence needs repetition to discriminate, but 10 000 iterations gives the same 8× band; the four loader reads are replaceable by in-memory data |
| `unit_test_multivariate_dtw` | **14.24** | 47 | :244-281 alone: 2 × 1100 DTWs on 200-length series (≈88 M DP cells in Debug) for one `REQUIRE_THAT(sum_mv, WithinAbs(sum_std, 1e-6))` and a `std::cout` timing line the file itself says is "not a hard timing assertion" (:279-280) | Yes — delete the timing loop; keep the equality with 1 iteration |
| `unit_test_fileOperations` | **11.28** | 9664 | `GENERATE(1,2,10,1000)` × `GENERATE(1,2,10,1000)` at :229-230 (16 combinations, up to 1000×1000 random values written to and re-read from disk) and `GENERATE(1,2,10,100)` × `GENERATE(1,2,10,1000)` at :378-379 (up to 100 files) | Yes. The parser contract is per-field; `1000` adds ~9 000 of the 9 664 assertions and no new branch. `{1, 2, 10}` × `{1, 2, 257}` covers the same paths |

Just under the line: `unit_test_mip` 9.35 s (HiGHS solves), `unit_test_wave1a_integration` 4.55 s
(effectively all of it in the perf case at :438-498), `unit_test_wave2a_integration` 2.42 s.

Note also `unit_test_Clock.cpp` at 1.27 s, of which 1.0 s is `sleep_for(1s)` at :61 supporting one
`str.find("min:sec")` assertion.

---

## 5. Naming and layout

### 5.1 Names that do not match the subject

* `unit_test_accuracy.cpp` — "accuracy" is not a module. Six unrelated groups (cross-variant,
  cross-metric, early abandon, numerical stability, lower bounds, multivariate), each belonging beside
  its subject.
* `unit_test_variant_distmat.cpp` — 11 of its 13 cases are about **cache invalidation and mmap
  fingerprints**, not about the `std::variant` the name refers to.
* `test_test_api.cpp` — a target literally named `test_test_api`; the subject is `dtwc/test_api.hpp`.
* `unit_test_Problem_phase0.cpp`, `unit_test_warping_phase0.cpp`, `unit_test_scores_phase0.cpp`,
  `unit_test_wave1a/2a/2b_integration.cpp` — named for a **campaign phase**, not a subject. Two of the
  three "phase0" files are now covered elsewhere (§1).
* `unit_test_scores_new.cpp` — "new" relative to 2026-04-02.
* `test_storage_policy.cpp` vs `unit_test_problem_storage_policy.cpp` — the same subject with two
  prefixes, split by *entry point* (DataLoader vs Problem), which the names do not say.
* The `test_*` / `unit_test_*` prefix split carries no meaning anywhere in the directory: compare
  `test_decode_pair.cpp` (pure unit) with `unit_test_cli_checkpoint.cpp` (spawns a subprocess).

### 5.2 Files that belong in a subdirectory (by the layer they exercise)

* **core** — `unit_test_warping.cpp`, `unit_test_warping_phase0.cpp`, `unit_test_dtw_variants.cpp`,
  `unit_test_adtw.cpp`, `unit_test_wdtw.cpp`, `unit_test_ddtw.cpp`, `unit_test_soft_dtw.cpp`,
  `unit_test_soft_dtw_hotpath.cpp`, `unit_test_arow_dtw.cpp`, `unit_test_missing_dtw.cpp`,
  `unit_test_missing_utils.cpp`, `unit_test_multivariate_dtw.cpp`, `unit_test_mv_variants.cpp`,
  `unit_test_mv_missing.cpp`, `unit_test_mv_lower_bounds.cpp`, `unit_test_multivariate_data.cpp`,
  `unit_test_z_normalize.cpp`, `unit_test_simd.cpp`, `unit_test_accuracy.cpp`,
  `test_decode_pair.cpp`, `test_error_taxonomy.cpp`, `unit_test_parallelisation.cpp`,
  `test_build_parallelism.cpp`.
* **algorithms** — `unit_test_clustering_algorithms.cpp`, `unit_test_wave2a_integration.cpp`.
* **mip** — `unit_test_mip.cpp`, `unit_test_benders.cpp`.
* **session** (Problem / Data / DataLoader / checkpoint / env / api) — `unit_test_Problem.cpp`,
  `unit_test_Problem_phase0.cpp`, `unit_test_Data.cpp`, `unit_test_DataLoader.cpp`,
  `unit_test_problem_encapsulation.cpp`, `unit_test_problem_missing.cpp`,
  `unit_test_problem_storage_policy.cpp`, `test_storage_policy.cpp`,
  `unit_test_deferred_allocation.cpp`, `unit_test_variant_distmat.cpp`,
  `unit_test_variant_precision.cpp`, `unit_test_dtw_function_semantics.cpp`,
  `unit_test_checkpoint.cpp`, `unit_test_checkpoint_binary.cpp`,
  `unit_test_checkpoint_robustness.cpp`, `test_env_device.cpp`, `test_test_api.cpp`,
  `test_tier1_cpp_api.cpp`, `test_problem_api_2_0.cpp`, `test_runtime_loudness_*.cpp` (3),
  `unit_test_Clock.cpp`, `unit_test_scores_*.cpp` (3), `unit_test_distance_matrix_properties.cpp`.
* **io** — `test_io_readers.cpp`, `unit_test_fileOperations.cpp`.
* **cli** (does not exist yet) — `unit_test_cli_args.cpp`, `unit_test_cli_checkpoint.cpp`.
* **gpu** (does not exist yet) — `test_cuda_*.cpp` (4), `test_metal_*.cpp` (3),
  `gpu_fixed_band_oracle.hpp`.
* **mpi** (does not exist yet) — `unit_test_mpi.cpp`, `unit_test_mpi_allreduce_chunking.cpp`.
* **build/meta** — `test_supply_chain_pinning.cpp`, `unit_test_deterministic_series.cpp`.
* **support** — `gpu_fixed_band_oracle.hpp` is shared by three files and belongs in `tests/support/`.

Because the CMake glob is recursive (`tests/CMakeLists.txt:1`) and target names are basenames, moving
a file changes **nothing** in CTest — except that eight F-gates key off `if(TARGET <name>)` and F15
keys off the *path strings* `tests/unit/test_cuda_correctness.cpp` etc.
(`unit_test_deterministic_series.cpp:319-341`). Any move must update F15.

### 5.3 Registration quirks

1. `unit_test_problem_storage_policy.cpp:119` — `REQUIRE(fs::equivalent(fs::temp_directory_path(), root))`
   only holds because `tests/CMakeLists.txt:645-647` redirects `TMP/TEMP/TMPDIR` to the F20 root, so
   the executable cannot be run outside CTest. Same coupling for `unit_test_checkpoint_binary.cpp`
   (`tests/CMakeLists.txt:170-172`).
2. `test_problem_api_2_0` is the only target whose CTest command is not the test binary; the variadic
   arm of `add_executable_with_coverage_and_test` (`cmake/Coverage.cmake:11-30`) exists solely for it.
3. `test_io_readers` gets its anti-skip guard only under `DTWC_HAS_ARROW`
   (`tests/CMakeLists.txt:633`). The comment at :622-631 documents this as deliberate, but it means
   the canonical gate has no defence against the file degrading further.
4. `unit_test_mpi` is registered as a plain one-process CTest entry despite documenting
   `mpiexec -n 4` at `unit_test_mpi.cpp:6`. There is no MPI CTest configuration anywhere in
   `tests/CMakeLists.txt`.
5. The F16 preset facts are asserted twice — once at configure time by CMake's own JSON parser
   (`tests/CMakeLists.txt:400-598`) and once at run time by regex in
   `test_supply_chain_pinning.cpp:184-371`. The runtime copy additionally pins *counts* that the
   configure-time copy does not.
6. `test_multivariate_adversarial` (out of scope) gets `RUN_SERIAL` explicitly because it contains a
   wall-clock parity assertion (`tests/CMakeLists.txt:656-665`). Five in-scope files contain
   wall-clock assertions (`unit_test_variant_distmat.cpp:594`,
   `unit_test_wave1a_integration.cpp:497`, `unit_test_multivariate_dtw.cpp:244`,
   `unit_test_wave2b_integration.cpp:656` and `:717`) and **none** is `RUN_SERIAL`.

---

## 6. Totals

**Corpus.** 73 files in scope (72 `.cpp` + `gpu_fixed_band_oracle.hpp`), **28 819 lines**,
**894 `TEST_CASE`s**, **3 148 assertion-macro sites**. (Catch2 reports far more at run time because of
loops, `GENERATE` and `SECTION`s — e.g. `unit_test_fileOperations` reports 9 664.) Plus
`tests/support/deterministic_series.hpp` (122 lines) and `tests/test_util.hpp` (102 lines).

**By primary category** — files are counted once, under their heaviest component:

| category | files | lines | TEST_CASEs | assertion sites |
|---|---|---|---|---|
| contract/oracle | 31 | 10 366 | 425 | 1 224 |
| capability-guard (GPU / Arrow / MPI-gated) | 9 | 4 520 | 126 | 512 |
| adversarial | 9 | 4 246 | 106 | 442 |
| fingerprint | 4 | 3 330 | 22 | 265 |
| integration | 6 | 2 928 | 55 | 316 |
| mixed (no dominant component) | 4 | 1 700 | 100 | 93 |
| trivial | 6 | 1 052 | 35 | 92 |
| implementation-detail (source/text scans) | 2 | 762 | 10 | 132 |
| perf-fence | 2 | 746 | 15 | 72 |

**Verdict candidates**

| verdict | count | files |
|---|---|---|
| **delete** | 2 | `unit_test_dtw_variants.cpp` (one case relocates to `unit_test_wdtw.cpp`); `unit_test_scores_phase0.cpp` (nothing to relocate) |
| **merge into `<file>`** | 11 | `unit_test_deferred_allocation.cpp` → `unit_test_wave2a_integration.cpp`; `unit_test_multivariate_data.cpp` → `unit_test_Data.cpp` + `core/unit_test_time_series.cpp`; `unit_test_benders.cpp` → `unit_test_mip.cpp`; `unit_test_warping_phase0.cpp` → `unit_test_warping.cpp`; `unit_test_Problem_phase0.cpp` → `unit_test_Problem.cpp`; `unit_test_z_normalize.cpp` → `unit_test_simd.cpp`; `unit_test_scores_new.cpp` → `adversarial/test_scores_adversarial.cpp` (CH + ARI blocks only); `unit_test_accuracy.cpp` (§5/§7/§8 → adversarial + core LB files); `unit_test_checkpoint.cpp` (:129-328 → `unit_test_checkpoint_robustness.cpp`); `unit_test_distance_matrix_properties.cpp` (all but :186-221); `test_error_taxonomy.cpp` (:107-126 → `unit_test_soft_dtw.cpp`) |
| **keep+tolerance** | 6 | `test_problem_api_2_0.cpp`, `test_supply_chain_pinning.cpp`, `unit_test_deterministic_series.cpp`, `unit_test_fileOperations.cpp`, `unit_test_mpi.cpp`, `tests/test_util.hpp` |
| **keep** | 52 | the remainder (including all 7 GPU files and `gpu_fixed_band_oracle.hpp`) |
| **rewrite as contract** | 0 | — |

**Sub-file deletions proposed** (TEST_CASE-level, not file-level):
`unit_test_multivariate_dtw.cpp:244-281`, `unit_test_wave1a_integration.cpp:438-498`,
`unit_test_wave2b_integration.cpp:656-762`, `unit_test_variant_precision.cpp:401-408`,
`unit_test_simd.cpp:80-102` (dead helper). Together ≈290 lines and ≈20 s of wall clock, with 4
non-vacuous assertions to relocate.

**Hard constraint on all of the above.** `unit_test_deterministic_series.cpp:300-379` pins the source
text *and exact call counts* of five sibling test files and five benchmarks, and F15's
`PASS_REGULAR_EXPRESSION` (`tests/CMakeLists.txt:664-673`) turns any mismatch into a CTest failure.
That coupling must be loosened before any GPU or MPI test file is moved, merged or edited.
