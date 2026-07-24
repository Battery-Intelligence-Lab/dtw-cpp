# R3-F11 tracked CMake archive integrity — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `658a9cd` (`docs: route supply-chain audit findings`)
- Subject: every tracked `CPMAddPackage(URL ...)` remote archive declaration,
  including `examples/cpp/example_project/CMakeLists.txt`.
- Existing GitHub Action SHA validation and the exact Arrow URL/digest ledger
  remain load-bearing and may not be weakened.
- The adjacent workflow-download/container policy and the malformed
  `cmake>=3.26` shell command are separate findings F34 and F35. They are not
  evidence that F11 passed or failed.

The PLAN archive, live PLAN killed-ideas section, and `.claude/LESSONS.md` were
searched before this branch. No killed idea is reopened. The relevant prior
rule is Task 0.12: remote build inputs must be immutable or content-verified,
and optional dependencies remain optional.

## Registered inherited inventory

The pre-implementation tracked-CMake audit found 25 tracked `CMakeLists.txt` /
`*.cmake` files and seven `CPMAddPackage(URL ...)` archive declarations:

| Manifest | Archive | Same-block SHA-256 |
|---|---|---|
| `cmake/Dependencies.cmake` | Catch2 v3.13.0 | `650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef` |
| `cmake/Dependencies.cmake` | HiGHS v1.15.1 | `a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16` |
| `cmake/Dependencies.cmake` | CLI11 v2.6.2 | `c6ea6b2e5608b3ea8617999bd5f47420c71b2ebdb8dc4767c1034d1da5785711` |
| `cmake/Dependencies.cmake` | Eigen 5.0.1 | `e4de6b08f33fd8b8985d2f204381408c660bffa6170ac65b68ae1bd3cd575c0a` |
| `cmake/Dependencies.cmake` | yaml-cpp 0.9.0 | `25cb043240f828a8c51beb830569634bc7ac603978e0f69d6b63558dadefd49a` |
| `cmake/Dependencies.cmake` | Arrow 19.0.1 | `4c898504958841cc86b6f8710ecb2919f96b5e10fa8989ac10ac4fca8362d86a` |
| `examples/cpp/example_project/CMakeLists.txt` | `refs/heads/documentation_update.zip` | **missing** |

Thus the registered inherited archive result is exactly 6/7 verified, one
mutable, one unhashed. The current checker is expected to false-green because
it hard-codes only Arrow and does not enumerate the example manifest.

The two tracked CPM bootstrap scripts use a different, already content-checked
form, `file(DOWNLOAD ... EXPECTED_HASH SHA256=${CPM_HASH_SUM})`; they are part
of F34's wider acquisition inventory, not silently counted as
`CPMAddPackage(URL ...)` declarations here.

## Registered replacement identity

The example will consume the committed 2.0.0rc1 boundary:

```text
commit=eda1b92bc89ee51568b052a6af86f615d336de3c
version=2.0.0rc1
url=https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/eda1b92bc89ee51568b052a6af86f615d336de3c.zip
size=4928286
sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696
```

This commit is the PLAN's named rc1 boundary and an ancestor of the published
`origin/Claude` snapshot. Its `examples/cpp/example_project/CMakeLists.txt`
and `main.cpp` Git blobs are byte-identical to the current files:

```text
current_example_cmake_blob=3e024a0fb598ad46c4a966155732cbfc1c250226
rc1_example_cmake_blob=3e024a0fb598ad46c4a966155732cbfc1c250226
current_example_main_blob=f123274e2ccd692154bcafb3eb10ba4b38cfa02b
rc1_example_main_blob=f123274e2ccd692154bcafb3eb10ba4b38cfa02b
rc1_version=2.0.0rc1
```

Two independent HTTPS routes returned identical bytes before registration:

```text
url=https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/eda1b92bc89ee51568b052a6af86f615d336de3c.zip
size=4928286
sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696
url=https://codeload.github.com/Battery-Intelligence-Lab/dtw-cpp/zip/eda1b92bc89ee51568b052a6af86f615d336de3c
size=4928286
sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696
```

Those downloads select the candidate identity; they are exploration, not the
F11 verdict. The fresh example configure below is the live CPM/hash proof.

## Acceptance band

F11 passes only if all of the following hold:

1. On the registered base, `python scripts/check_supply_chain_pins.py` exits
   zero and prints `supply-chain pins verified` despite the exact mutable,
   unhashed example declaration. Record that result as **FALSIFIED** for
   repo-wide tracked-CMake coverage, not as a passing supply-chain verdict.
2. Implement the generic checker and its tests *before* editing the example.
   The modified production CLI, run against the real inherited tracked tree,
   must exit 1, name
   `examples/cpp/example_project/CMakeLists.txt:12`, and print counters
   `verified=6 total=7 mutable=1 unhashed=1 verdict=FAIL`. This is the PLAN's
   first real gate and proves `main()` reaches the generic scanner; a helper
   fixture alone cannot satisfy it.
3. The permanent checker enumerates paths from the Git index rather than
   recursively entering untracked `build*/_deps`. It finds exactly 25 tracked
   CMake manifests and all seven `CPMAddPackage(URL ...)` declarations.
4. URL and `URL_HASH SHA256=<64 hex>` must occur as active directives in the
   same CPM call. A hash in another package, a comment, or prose does not
   satisfy the declaration. `refs/heads/` and other registered mutable-branch
   archive forms fail even if an attacker supplies a digest.
5. The existing 39/39 full-SHA workflow-action check and the exact Arrow
   URL/digest check remain active.
6. The example URL equals the registered full-commit archive and its
   `URL_HASH` equals the registered SHA-256. No branch, tag alias, shortened
   commit, or post-run digest substitution is accepted.
7. A permanent focused Python suite runs at least nine tests with zero failure
   or skip. It must kill, by name: the exact inherited example fixture; a
   missing, shortened, and comment-only URL hash; a branch URL retaining a
   valid hash; removal of a currently pinned main archive hash; Arrow digest
   drift; and a mutable workflow action. At least one valid hashed archive must
   pass the same parser.
8. The real checker exits zero on the repaired tree and prints counters derived
   from successful checks:

   ```text
   WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
   CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
   ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
   supply-chain pins verified
   ```

9. Before the live download, `build/f11-example-pin` must be absent; do not
   reuse or clean another build tree. Unset `CPM_SOURCE_CACHE` so neither an
   environment cache nor a prior package source can satisfy the request. Run
   these exact PowerShell commands from the repository root:

   ```powershell
   if (Test-Path 'build/f11-example-pin') { throw 'F11 fresh build directory already exists' }
   $env:CPM_SOURCE_CACHE = $null
   cmake -S examples/cpp/example_project -B build/f11-example-pin -G Ninja -DCMAKE_BUILD_TYPE=Release -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_HIGHS_GPU=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_ARROW=OFF -DDTWC_ENABLE_CUDA=OFF -DDTWC_ENABLE_METAL=OFF -DDTWC_ENABLE_MPI=OFF -DDTWC_ENABLE_YAML=OFF -DDTWC_BUILD_TESTING=OFF -DBUILD_TESTING=OFF -DDTWC_BUILD_BENCHMARK=OFF -DDTWC_BUILD_EXAMPLES=OFF -DDTWC_BUILD_MATLAB=OFF -DDTWC_BUILD_PYTHON=OFF
   ```

   It must perform the registered download/hash check and exit zero. Because
   the project's configuration summary is top-level-only, the reachable
   nested-consumer version proof is instead exact content `2.0.0rc1` in
   `build/f11-example-pin/_deps/dtw-cpp-src/VERSION` plus at least one active
   `DTWC_VERSION_STRING=\"2.0.0rc1\"` definition in the generated
   `build.ninja`. Then
   `cmake --build build/f11-example-pin --target dtwc++ --parallel` must exit
   zero. A cached or historical hand-linked artifact is not acceptable.
10. The existing C++ supply-chain test executes rather than skipping. The full
   canonical gate remains 114/114, zero failed, with exactly the six registered
   capability skips.
11. `git diff --check`, generated documentation, documentation contract,
    record hygiene, repository hygiene, supply-chain checker, and an
    independent adversarial review all pass. The F11 PLAN/handoff closure is a
    separate documentation commit.

Any missed tracked CPM URL, wrong count, branch archive, absent/malformed/
comment-only hash, weakened Arrow/action check, failed fresh configure/build,
skipped subject, or canonical regression is **FALSIFIED**. There are at most
two implementation repair attempts. The commit, URL, digest, seven-declaration
inventory, same-block rule, and mutation requirements do not move after the
first decisive execution.

## Rollback and expected-risk claim

The implementation rollback will be the single F11 checker/example/test
commit. The claim most expected to be wrong is that a fresh nested CPM
configure can consume the rc1 archive with every optional dependency disabled;
the archive bytes and API compatibility are confirmed, but that exact
standalone configure route has not yet run.

## Inherited red

After registration and before implementation, the production checker
false-greened exactly as registered:

```text
supply-chain pins verified
checker_exit=0
```

The separate tracked-CMake inventory printed:

```text
tracked_cmake=25 cpm_url=7 verified=6 mutable=1 unhashed=1
unsafe=examples/cpp/example_project/CMakeLists.txt:11: url=https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/refs/heads/documentation_update.zip mutable=true hashed=false
```

The inventory prototype's diagnostic line calculation was one low because it
added a body-relative match offset to the opening-command offset instead of the
body offset. The inventory counts and URL are unaffected. Two independent
line readers arbitrate the location:

```text
12:    URL "https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/refs/heads/documentation_update.zip"
powershell_line=12 text=URL "https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/refs/heads/documentation_update.zip"
```

Verdict: **FALSIFIED** for the inherited production checker. It exits zero
while exactly one tracked CPM archive is both mutable and unhashed.

## Implemented checker red

Before changing the example manifest, the generic production checker reached
the real inherited declaration and failed with the registered line, inventory,
and counters:

```text
mutable or unhashed tracked CMake archives:
  examples/cpp/example_project/CMakeLists.txt:12: mutable remote archive URL: https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/refs/heads/documentation_update.zip
  examples/cpp/example_project/CMakeLists.txt:12: missing or invalid URL_HASH SHA256=<64 hex>: https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/refs/heads/documentation_update.zip
Example archive URL is missing or changed: https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/eda1b92bc89ee51568b052a6af86f615d336de3c.zip
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=6 total=7 mutable=1 unhashed=1 verdict=FAIL
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=25
checker_exit=1
```

Verdict: **PASS** for the registered F11 first gate. The production entry point,
not only a parser fixture, finds all 25 tracked CMake manifests and all seven
archive declarations, preserves the existing action and Arrow checks, and
rejects the sole inherited unsafe declaration.

## Mutation gate and adversarial review

The permanent suite grew as independent read-only reviews found executable
false-greens in the first parser revisions: bracket comments, inline and
semicolon/variable-expanded arguments, branch archive aliases and API routes,
multiple URL values, `cmake_language(CALL|DEFER|EVAL ...)`, and a
`URL_HASH` token captured by CPM's `OPTIONS` multi-value argument. Each was
reproduced before repair. The final suite covers those classes plus every
registered mutation:

```text
..........................................                               [100%]
42 passed in 0.23s
```

The final real-tree production run is:

```text
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=25
supply-chain pins verified
checker_exit=0
```

The last independent re-probe returned CLEAN within the registered direct and
`cmake_language` CPM URL/hash scope. The production checker also enforces the
registered 25-manifest and seven-archive floors, so hiding a current
declaration cannot turn a lower inventory into a green run.

## Fresh nested configure

The first fresh attempt satisfied the preflight and reached CMake generation,
but the execution wrapper terminated before capturing CMake's process exit:

```text
command timed out after 61837 milliseconds
F11_PREFLIGHT build_dir=ABSENT fetch_overrides=NONE
-- Configuring done (61.0s)
-- Generating done (0.3s)
-- Build files have been written to: C:/D/git/dtw-cpp/build/f11-example-pin
```

Wrapper result: exit 124. Verdict: **FALSIFIED** for the required
configure-exit sub-band. Generated files are not substituted for the missing
exit code. No matching CMake/Git/Ninja process remained. The complete tree was
preserved, not deleted:

```text
preserved_attempt1=C:\D\git\dtw-cpp\build\f11-example-pin-timeout-attempt1
registered_build_dir_absent=True
```

The second and final allowed attempt used the unchanged configure flags from a
newly absent registered directory, with every probed CPM/FetchContent override
still absent. It completed:

```text
F11_ATTEMPT2_PREFLIGHT build_dir=ABSENT fetch_overrides=NONE
--   llfio:    OFF (DTWC_ENABLE_LLFIO=OFF) — mmap disabled.
-- Found OpenMP_CXX: -fopenmp (found version "5.1")
-- Found OpenMP: TRUE (found version "5.1") found components: CXX
-- OpenMP 5.1 found — parallel execution enabled
-- Configuring done (43.1s)
-- Generating done (0.2s)
-- Build files have been written to: C:/D/git/dtw-cpp/build/f11-example-pin
configure_exit=0
```

The downloaded archive, generated verification script, source version, and
active compile definitions then passed the preregistered exact checks:

```text
F11_ARCHIVE_GATE size=4928286 expected_size=4928286 sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696 expected_sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696 generated_hash_script_matches=3 verdict=PASS
F11_VERSION_GATE source_version=2.0.0rc1 expected=2.0.0rc1 active_definitions=20 verdict=PASS
```

Both the registered library target and the actual standalone consumer compiled
and linked:

```text
[28/29] Linking CXX static library _deps\dtw-cpp-build\bin\dtwc++.lib
dtwc_target_build_exit=0
[2/3] Linking CXX executable MY_AWESOME_PROJECT.exe
example_consumer_build_exit=0
```

The consumer is deliberately not run: its example data path is absent and the
registered configuration disables MIP solver backends. Compile/link is the
integration claim.

## Canonical regression gate

The canonical rebuild and focused C++ subject gate ran:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
canonical_build_exit=0
RNG seed: 1532408775
All tests passed (20 assertions in 3 test cases)
supply_chain_cpp_exit=0
```

The full canonical inventory and exact capability-skip list are:

```text
100% tests passed, 0 tests failed out of 114

Total Test time (real) = 100.44 sec

The following tests did not run:
	 48 - test_cuda_correctness (Skipped)
	 50 - test_cuda_lb_keogh (Skipped)
	 54 - test_io_readers (Skipped)
	 55 - test_metal_correctness (Skipped)
	 56 - test_metal_lb_keogh (Skipped)
	 57 - test_metal_mmap (Skipped)
canonical_ctest_exit=0
```

Verdict: **PASS** for the F11 behavioral and build gates, with the first wrapper
attempt retained as a named falsification rather than hidden. Final hygiene and
record gates follow below before commit.

## Final hygiene and record gates

All final pre-commit gates exited zero:

```text
py_compile_exit=0
generated documentation is current
documentation contract checks passed
docs_static_exit=0
generated documentation is current
documentation contract checks passed
docs_cli_exit=0
record hygiene checks passed
record_hygiene_exit=0
banned_tracked_paths=0
unexpected_zero_byte_files=0
targeted_duplicate_groups=0
asset_routes=4/4
required_ignore_targets=23/23
high_confidence_secret_hits=0
codecov_badge_query_hits=0
changelog_structure=PASS
seed_compatibility_markers=2/2
VERDICT=PASS
repo_hygiene_exit=0
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=25
supply-chain pins verified
supply_chain_exit=0
diff_check_exit=0
```

Verdict: **PASS**. The generated documentation, live CLI contract, record
hygiene, repository hygiene, production supply-chain checker, Python syntax,
and whitespace gates all passed before the implementation commit.

The exact final mutation-suite state was then re-run:

```text
..........................................                               [100%]
42 passed in 0.26s
focused_pytest_exit=0
```

## Final-audit falsification and repair band

The final independent diff audit rejected the pending commit after reproducing
three in-scope false-greens:

1. `${ARCHIVE_ARGS}` beside an otherwise valid literal URL/hash was accepted.
2. `cmake_language(DEFER ID call CALL ...)` and the `ID_VAR` equivalent treated
   the option operand `call` as the scheduling operator and skipped the real
   call.
3. An exact registered Example or Arrow URL/hash in a differently named decoy
   package masked a drifted real package declaration.

No commit was made. Before the repair run, the registered extension is five
executed cases (the DEFER spelling is parameterized twice), raising the focused
floor from 42 to **47/47**. The inherited implementation must fail all three
reproduction classes. The repaired production source must then pass 47/47,
retain the exact 25-manifest/seven-archive counters, and pass a fresh
independent audit. A dynamic argument anywhere in an archive-bearing CPM call
is fail-closed; the one live HiGHS declaration using dynamic `OPTIONS` must be
made literal without changing its two option sets. Maximum repair attempts:
two.

The inherited pending implementation failed exactly the five registered
executions:

```text
.......................F..FF..........F.......F                          [100%]
=========================== short test summary info ===========================
FAILED tests/python/test_supply_chain_pins.py::test_dynamic_arguments_beside_a_literal_url_are_rejected
FAILED tests/python/test_supply_chain_pins.py::test_cmake_language_defer_option_operand_named_call_is_scanned[ID]
FAILED tests/python/test_supply_chain_pins.py::test_cmake_language_defer_option_operand_named_call_is_scanned[ID_VAR]
FAILED tests/python/test_supply_chain_pins.py::test_arrow_exact_pin_cannot_be_satisfied_by_a_decoy_package
FAILED tests/python/test_supply_chain_pins.py::test_example_exact_pin_cannot_be_satisfied_by_a_decoy_package
5 failed, 42 passed in 0.20s
post_audit_inherited_concise_exit=1
```

Verdict: **FALSIFIED**, matching the independent reviewer rather than the
earlier CLEAN claim.

Repair attempt 1 implemented package-name binding, DEFER option grammar, and
fail-closed expansion handling. It repaired the five new reproductions but
FALSIFIED the full 47-case band because two older tests still expected their
weaker pre-repair failure categories:

```text
..................F................F...........                          [100%]
=========================== short test summary info ===========================
FAILED tests/python/test_supply_chain_pins.py::test_dynamic_mirror_url_is_rejected
FAILED tests/python/test_supply_chain_pins.py::test_dynamic_url_is_rejected_fail_closed
2 failed, 45 passed in 0.19s
repair_attempt1_concise_exit=1
```

Both fixtures now fail earlier and more strictly as unclassifiable dynamic
CPM arguments. Their assertions are updated to require that fail-closed
diagnostic; no production rule or acceptance number changed for attempt 2.

## Repair attempt 2 and integration gates

The second and final repair attempt passed the registered focused band:

```text
...............................................                          [100%]
47 passed in 0.32s
repair_attempt2_exit=0
```

Python compilation and the production checker retained every exact inventory
counter:

```text
py_compile_exit=0
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=25
supply-chain pins verified
supply_chain_exit=0
```

Making the live HiGHS call fully literal moved `CUPDLP_GPU` to an explicit
cache setting bound to `DTWC_HIGHS_GPU`. Both real configurations regenerated
and built. Their cache values and HiGHS' own configure reports agree:

```text
build\highs-1151\CMakeCache.txt:570:CUPDLP_GPU:BOOL=OFF
build\highs-1151\CMakeCache.txt:653:DTWC_ENABLE_HIGHS:BOOL=ON
build\highs-1151\CMakeCache.txt:671:DTWC_HIGHS_GPU:BOOL=OFF
-- Build pdlp with GPU: OFF
canonical_rebuild_exit=0

build\highs-gpu\CMakeCache.txt:774:CUPDLP_GPU:BOOL=ON
build\highs-gpu\CMakeCache.txt:861:DTWC_ENABLE_HIGHS:BOOL=ON
build\highs-gpu\CMakeCache.txt:879:DTWC_HIGHS_GPU:BOOL=ON
-- Build pdlp with GPU: ON
===BUILD_EXIT=0
highs_gpu_build_wrapper_exit=0
```

The first PowerShell invocation copied the Git-Bash slash spelling and never
started the batch file:

```text
'build' is not recognized as an internal or external command,
operable program or batch file.
highs_gpu_build_wrapper_exit=1
```

The corrected absolute Windows path drove the same preserved build directory;
this was a wrapper spelling failure, not a configure or compilation failure.
The rebuilt GPU/HiGHS subject then ran:

```text
Test project C:/D/git/dtw-cpp/build/highs-gpu
    Start 45: test_pdlp_lp
1/1 Test #45: test_pdlp_lp .....................   Passed   82.40 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =  82.43 sec
highs_gpu_pdlp_run_exit=0
```

Finally, the regenerated canonical tree passed its full exact inventory:

```text
100% tests passed, 0 tests failed out of 114

Total Test time (real) =  92.13 sec

The following tests did not run:
	 48 - test_cuda_correctness (Skipped)
	 50 - test_cuda_lb_keogh (Skipped)
	 54 - test_io_readers (Skipped)
	 55 - test_metal_correctness (Skipped)
	 56 - test_metal_lb_keogh (Skipped)
	 57 - test_metal_mmap (Skipped)
canonical_ctest_final_exit=0
```

Verdict: **PASS** for the second/final repair, both conditional HiGHS option
sets, the real GPU/HiGHS test, and the canonical 114-test regression band.
The fresh independent parser audit remains the final pre-commit gate.

## Second-audit falsification and inventory pivot

The fresh audit confirmed the prior three repairs, then rejected the pending
commit on two additional in-scope false-greens:

1. Unquoted semicolon lists in `cmake_language(CALL|DEFER|EVAL ...)` were split
   by CMake before invocation but skipped by the scanner.
2. Renaming the real package and transferring the expected package name plus
   registered URL/hash to an active `DOWNLOAD_ONLY` decoy satisfied the
   package-name heuristic for both Example and Arrow.

The earlier hand-written exact-package heuristic is therefore **FALSIFIED**
despite the 47/47 repair band. The replacement is a different, simpler
invariant: the complete registered identity multiset of all seven tracked URL
archive declarations (path, package name, URL, SHA-256) must match exactly.
An added decoy, removed declaration, renamed package, URL drift, hash drift, or
duplicate is then an inventory difference rather than a control-flow guess.

Before implementation, the extension registers three semicolon spellings and
two name-transfer decoys, raising the focused floor to **52/52**. The inherited
checker must fail all five. The repaired checker must reject semicolon-expanded
`cmake_language` arguments before dispatch, report an exact seven-entry
identity inventory, retain 25 tracked manifests, and pass a fresh independent
audit. This inventory pivot gets at most two attempts.

The inherited checker failed exactly those five registered executions:

```text
..................................FFF.....F........F                     [100%]
=========================== short test summary info ===========================
FAILED tests/python/test_supply_chain_pins.py::test_semicolon_expanded_cmake_language_arguments_are_rejected[CALL;CPMAddPackage;NAME;hidden;URL;https://github.com/example/project/archive/refs/heads/main.zip]
FAILED tests/python/test_supply_chain_pins.py::test_semicolon_expanded_cmake_language_arguments_are_rejected[DEFER;CALL;CPMAddPackage;NAME;hidden;URL;https://github.com/example/project/archive/refs/heads/main.zip]
FAILED tests/python/test_supply_chain_pins.py::test_semicolon_expanded_cmake_language_arguments_are_rejected[EVAL;CODE;message]
FAILED tests/python/test_supply_chain_pins.py::test_arrow_exact_name_cannot_be_transferred_to_a_download_only_decoy
FAILED tests/python/test_supply_chain_pins.py::test_example_exact_name_cannot_be_transferred_to_a_download_only_decoy
5 failed, 47 passed in 0.20s
second_audit_inherited_exit=1
```

Verdict: **FALSIFIED** for the inherited name-only/unguarded-wrapper design.

The exact-inventory pivot passed on its first attempt:

```text
....................................................                     [100%]
52 passed in 0.23s
inventory_pivot_attempt1_exit=0
```

The production scan then reproduced the complete registered inventory:

```text
inventory_pivot_py_compile_exit=0
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=25
supply-chain pins verified
inventory_pivot_supply_chain_exit=0
```

Verdict: **PASS** for pivot attempt 1. Two fresh independent read-only audits
remain binding before commit.

Both auditors independently FALSIFIED pivot attempt 1 with the same
production-equivalent override:

```text
PRODUCTION_EQUIVALENT manifests=25 pins=7 verified=7
archive_failures= []
registered_inventory_failures= []
arrow_error= None
example_error= None
```

The registered Example identity remained present, but
`DOWNLOAD_COMMAND git clone --branch main ...` made its URL/hash inert.
Installed CMake documentation states:

```text
DOWNLOAD_COMMAND <cmd>...
Overrides the command used for the download step
all other download options will be ignored.
```

The same audit found a diagnostic-only defect: sorting unexpected identities
with both a missing and a string package name raised:

```text
TypeError: '<' not supported between instances of 'str' and 'NoneType'
```

Before the second and final pivot attempt, the registered band expands to ten
alternate-source directives (`DOWNLOAD_COMMAND`, `SOURCE_DIR`, the four CPM
repository shorthands, the three other ExternalProject repository methods,
and `FIND_PACKAGE_ARGUMENTS`) plus one mixed-name diagnostic case: **63/63**.
Every alternate source selector in a URL declaration must fail before identity
comparison, and inventory drift must print deterministic diagnostics rather
than a traceback. No third pivot attempt is permitted.

Attempt 1 failed all 11 newly registered executions:

```text
........................FFFFFFFFFF.....................F.......          [100%]
=========================== short test summary info ===========================
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[DOWNLOAD_COMMAND-git clone --branch main https://example.invalid/project.git]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[SOURCE_DIR-vendor/project]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[GITHUB_REPOSITORY-example/project]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[GITLAB_REPOSITORY-example/project]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[BITBUCKET_REPOSITORY-example/project]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[GIT_REPOSITORY-https://example.invalid/project.git]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[SVN_REPOSITORY-https://example.invalid/project/svn]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[HG_REPOSITORY-https://example.invalid/project/hg]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[CVS_REPOSITORY-https://example.invalid/project/cvs]
FAILED tests/python/test_supply_chain_pins.py::test_url_archive_rejects_alternate_source_directives[FIND_PACKAGE_ARGUMENTS-CONFIG]
FAILED tests/python/test_supply_chain_pins.py::test_inventory_diagnostics_sort_missing_and_string_names
11 failed, 52 passed in 0.22s
final_pivot_inherited_exit=1
```

Verdict: **FALSIFIED**, exactly matching the registered final extension.

The second and final inventory-pivot attempt passed:

```text
...............................................................          [100%]
63 passed in 0.19s
final_pivot_attempt2_exit=0
```

The exact production inventory remained green:

```text
final_pivot_py_compile_exit=0
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=25
supply-chain pins verified
final_pivot_supply_chain_exit=0
```

Verdict: **PASS** for the second/final pivot attempt. Two bounded independent
audits are reprobing the previously falsifying cases before commit.

## Final bounded-audit verdict

Both final auditors rejected closure. The main audit found
`CUSTOM_CACHE_KEY`; the independent postfix audit found a quoted CMake line
continuation. Local in-memory production-equivalent probes reproduced both
false-greens:

```text
CUSTOM_CACHE_KEY_PROBE manifests=25 pins=7 verified=7
archive_failures=[]
registered_inventory_failures=[]
arrow_error=None
example_error=None
QUOTED_CONTINUATION_PROBE manifests=25 pins=7 verified=7
archive_failures=[]
registered_inventory_failures=[]
arrow_error=None
example_error=None
final_audit_probe_exit=0
```

The installed CPM 0.42.1 source confirms that, when `CPM_SOURCE_CACHE` is set,
`CUSTOM_CACHE_KEY` selects a cache directory; an existing directory becomes
`SOURCE_DIR` and sets `CPM_SKIP_FETCH`. Thus URL/hash verification is skipped
under that configuration. The original decisive example configure remains
valid because its preflight proved all CPM/FetchContent overrides absent; the
generic future-syntax claim does not.

The local CMake 4.2.3 interpreter independently confirmed the lexical bypass:

```text
-- F11_CMAKE_ARGUMENT=[DOWNLOAD_COMMAND]
-- F11_CMAKE_ARGUMENT=[marker]
cmake_continuation_probe_exit=0
```

The checker instead retains a newline in `"DOWNLOAD_\<newline>COMMAND"` and
does not classify the resulting CMake `DOWNLOAD_COMMAND`.

Final verdict:

- **[confirmed] PASS** — the tracked example now names the registered commit
  archive and SHA-256; a fresh no-override configure downloaded the exact
  4,928,286-byte artifact, the real consumer linked, the exact seven-entry live
  inventory passes, both HiGHS option sets built, GPU/HiGHS ran, and canonical
  CTest is 114/114 with six capability skips.
- **[confirmed] FALSIFIED** — the hand-written scanner is not fail-closed for
  every CMake/CPM spelling in the registered scope. Evidence is the two probe
  blocks above.

No third inventory-pivot attempt is permitted. The current hardening is a
deliverable, but F11 cannot be marked closed. The replacement parser/canonical
manifest design must be a new finding rather than another rescue patch.

The exact partial-result commit state passed every non-adversarial gate:

```text
...............................................................          [100%]
63 passed in 0.19s
partial_commit_pytest_exit=0
generated documentation is current
documentation contract checks passed
partial_commit_docs_static_exit=0
generated documentation is current
documentation contract checks passed
partial_commit_docs_cli_exit=0
record hygiene checks passed
partial_commit_record_exit=0
banned_tracked_paths=0
unexpected_zero_byte_files=0
targeted_duplicate_groups=0
asset_routes=4/4
required_ignore_targets=23/23
high_confidence_secret_hits=0
codecov_badge_query_hits=0
changelog_structure=PASS
seed_compatibility_markers=2/2
VERDICT=PASS
partial_commit_repo_exit=0
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=25
supply-chain pins verified
partial_commit_supply_exit=0
partial_commit_diff_exit=0
```

This green block validates the committed hardening and current tracked tree; it
does not overturn the named final-audit falsifications.
