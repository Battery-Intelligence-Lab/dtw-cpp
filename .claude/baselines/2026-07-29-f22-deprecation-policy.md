# F22 — cross-language deprecation policy

Date: 2026-07-29 (Europe/London)

## Subject and inherited state

Base commit:

```text
5352bc0b0e459011ae1dbe8ef38b38262306459d
```

F22 owns the retained compatibility names promised by
`docs/api-contract-2.0.md` §§2–4 and the missing diagnostic policy across C++,
Python, and MATLAB. It does not remove an alias, change its transition window,
privatize F19's retained raw C++ fields, reopen F18, or change an alias's
valid-domain behavior.

The clean canonical baseline was rebuilt before registration:

```text
ninja: no work to do.
100% tests passed, 0 tests failed out of 122
Total Test time (real) =  28.50 sec
```

The exact six capability skips were:

```text
test_cuda_correctness
test_cuda_lb_keogh
test_io_readers
test_metal_correctness
test_metal_lb_keogh
test_metal_mmap
```

Environment:

```text
Microsoft Windows NT 10.0.26200.0
clang version 21.1.8
cmake version 4.2.3
CMAKE_BUILD_TYPE:STRING=Release
DTWC_ENABLE_ARROW:BOOL=OFF
DTWC_ENABLE_HIGHS:BOOL=ON
DTWC_ENABLE_LLFIO:BOOL=ON
```

Verdict: **CLEAN BASELINE PASS [confirmed]** at the base commit above.

## Frozen inventory and adjudications

### C++

The frozen C++ surface contains exactly **30 names / 33 diagnostic entities**:

- 20 `Problem` function overloads;
- two `Problem` public `int` fields;
- five score functions;
- six F21 loader/path function overloads.

The name/entity difference is the second `writeDistanceMatrix` overload and
the second overload of each path setter. The no-argument
`DataLoader::startColumn()`/`startRow()` reads were never renamed and are not
aliases.

An exploratory public-header Clang probe used every entity once and printed:

```text
EXIT=0 DEPRECATION_WARNING_LINES=24
```

The otherwise complete canonical probe printed:

```text
EXIT=0 OUTPUT_LINES=0
```

Thus the inherited ledger is 24/33 loud and 9/33 silent. The exact missing
entities are:

1. `Problem::maxIter`;
2. `Problem::N_repetition`;
3. `Problem::readDistanceMatrix(const fs::path&)`;
4. `Problem::writeDistanceMatrix(const std::string&) const`;
5. `Problem::writeDistanceMatrix() const`;
6. `Problem::printClusters() const`;
7. `Problem::writeClusters()`;
8. `Problem::writeMedoidMembers(int,int) const`;
9. `Problem::writeSilhouettes()`.

The seven I/O functions are directionally inverted: the camelCase functions
own the out-of-line implementations and canonical functions forward to them.
F22 inverts that ownership. The old names become inline deprecated shims.

F19 requires `maxIter` and `N_repetition` to remain actual public `int` data
members with `int&`/`const int&` access. F22 therefore annotates the members
directly. The canonical setter/read bodies move out of line and use the
narrowest compiler-specific diagnostic suppression around their unavoidable
access to the retained deprecated storage. A proxy, reference member, layout
change, or privatization is outside the registered band.

### Python

The retained Python surface is **12 symbols / 13 base operations**:

1. `Problem.set_number_of_clusters`;
2. `Problem.n_repetition` read;
3. `Problem.n_repetition` write;
4. `Problem.cluster_size`;
5. `Problem.distance_matrix_numpy`;
6. `Problem.set_distance_matrix_from_numpy`;
7. `davies_bouldin_index`;
8. `dunn_index`;
9. `calinski_harabasz_index`;
10. `adjusted_rand_index`;
11. `normalized_mutual_information`;
12. `dtwcpp.ClusterResult`;
13. `Result.medoid_indices`.

`Problem.cluster_size` is included because the live frozen parity inventory
retains it as a deprecated binding even though §4's abbreviated Python list
omits it. Rows 29a/29b and §2.2 explicitly retain both distance aliases.
`ClusteringResult.medoid_indices`, sklearn's `medoid_indices_`, and the
`Result` constructor parameter are different surfaces and remain canonical.

Only `Result.medoid_indices` currently warns. Its exact existing message is:

```text
Result.medoid_indices is deprecated; use Result.medoids
```

The root `ClusterResult` alias must preserve exact class identity. It is
implemented through module `__getattr__`, remains in `__all__`, emits one
caller-attributed warning on resolution, and returns `Result`; a subclass,
proxy, or factory is forbidden.

All Python messages use:

```text
<qualified-old> is deprecated; use <qualified-new>
```

with `DeprecationWarning`, exactly once per operation, before side effects, and
with the warning filename attributed to the caller.

### MATLAB

The frozen MATLAB surface contains **15 aliases / 15 diagnostic operations**:

- assignment through `Problem.Band`, `Verbose`, `MaxIter`, and
  `NRepetition`;
- `Problem.get_distance_matrix`;
- dependent reads `Size`, `ClusterSize`, `Name`, `CentroidsInd`, and
  `ClustersInd`;
- the five legacy score wrappers.

The four configuration properties have only canonical setters in the frozen
MATLAB table. Their assignment warns and forwards; their retained reads remain
functional and are not newly treated as four undocumented getter aliases.
Canonical setters update private backing state and the MEX route directly so
construction and canonical calls remain silent.

Frozen row 29 writes the same `set_distance_matrix` spelling on both sides.
Sections 2.2 and 3 identify it as the live canonical writer. It is therefore
not a distinguishable alias and must remain silent. Only
`get_distance_matrix` is deprecated in MATLAB. This resolves the contradictory
§4 shorthand `get_/set_distance_matrix` without inventing an impossible
warning/silence split on one symbol.

All MATLAB aliases emit exactly one warning per operation with:

```text
identifier: dtwc:deprecatedAlias
message:    '<old>' is deprecated; use '<new>' instead.
```

The old/new table uses fully qualified public spellings. The warning occurs in
the `.m` compatibility shim, before its side effect, never in a shared MEX
command. Canonical construction, setters, reads, scores,
`set_distance_matrix`, and the public `DTWClustering.fit` route are silent.

The two F18 cases retained red in `test_contract_parity.m` remain owned by F18:

```text
test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
```

The latest executed five-suite evidence after those tests were committed is
84 total / 81 passed / 2 failed / 3 incomplete, where the third incomplete is
the expected opposite-flavor capability case
`test_test_api/test_parallelisation_serial_is_honest`. That source-backed
ledger supersedes the older 82-test pre-F18 recipe for F22 acceptance. F22 may
not hide or claim either retained F18 failure.

## Pre-run adversarial review

1. A deprecated field referenced by an inline canonical accessor makes a
   canonical consumer warn. Keep the actual fields, move only the canonical
   accessor bodies out of line, and suppress only those definitions.
2. A canonical I/O wrapper that calls a newly deprecated old function makes
   every consumer noisy. Rename the seven definitions to canonical names;
   old functions forward in the header.
3. Ordinary repository calls must migrate before deprecations become errors.
   Intentional compatibility fixtures are allowlisted; tests are not exempt
   merely because they are tests.
4. Native Python aliases must warn before releasing the GIL or mutating state.
   Pure-Python aliases must set `stacklevel=2`. A newly built extension carries
   a private F22 discriminator so a stale `.pyd` cannot pass.
5. Eagerly binding `ClusterResult = Result` cannot warn on later access.
   Module `__getattr__` preserves identity and permits a per-use warning.
6. MATLAB warnings belong in old wrappers. Warning in a MEX command would also
   contaminate canonical wrappers that intentionally share that command.
7. MATLAB's current canonical setters call PascalCase property setters.
   Introduce private backing values so construction and canonical setters
   bypass the warning while old assignments still forward exactly once.
8. MATLAB ARI/NMI legacy wrappers have broader validation/normalization than
   their canonical wrappers. Preserve it and compare behavior only on
   non-degenerate contract-valid labels; invalid-input parity is not an F22
   claim.
9. Full MATLAB acceptance includes the two named F18 failures. Any additional
   failed/incomplete name is an F22 regression.
10. Reuse existing CTest and MATLAB suite entries. F39 alone owns the tracked
    CMake inventory mismatch; F22 must not add a `.cmake` manifest or a new
    CTest target.

## Registered bands

All bands below were written before the decisive expected-red fixtures or any
product edit. Product work is capped at two attempts.

### R1 — inherited expected red

After the permanent fixtures are committed but before product work:

- the unmasked canonical LLFIO-ON C++ diagnostic marker first exposes F45's
  inherited dependency-state leak at legacy diagnostics 0/33 and all 33
  entities silent; after F45 is independently repaired, both canonical
  LLFIO-ON and llfio-OFF report inventory 33/33, legacy diagnostics 24/33, the
  exact nine silent entities above, and zero canonical deprecation lines;
- Python focused collection is exactly 18 cases: 2 pass and 16 fail, where the
  failures are only the twelve previously silent symbols/operations and the
  three additional `ClusterResult` resolution routes plus the ordinary-call
  allowlist;
- MATLAB focused F22 case fails solely because all 15 alias operations emit
  zero warnings, while its canonical silence/equivalence observations are
  retained.

Expected-red evidence consumes no product attempt. A harness error is not an
F22 verdict.

## Decisive red-first evidence

The permanent fixtures were committed in `b70a259`. No product file changed.

### C++ public-header diagnostics

The canonical LLFIO-ON driver exited 1 and printed:

```text
F22_CPP_SILENT count=33/33 entities=Problem::set_numberOfClusters(int),Problem::refreshDistanceMatrix(),Problem::readDistanceMatrix(const fs::path&),Problem::maxDistance() const,Problem::distByInd(int,int),Problem::isDistanceMatrixFilled() const,Problem::fillDistanceMatrix(),Problem::printDistanceMatrix() const,Problem::writeDistanceMatrix(const std::string&) const,Problem::writeDistanceMatrix() const,Problem::printClusters() const,Problem::writeClusters(),Problem::writeMedoidMembers(int,int) const,Problem::writeSilhouettes(),Problem::findTotalCost(),Problem::assignClusters(),Problem::calculateMedoids(),Problem::cluster_by_MIP(),Problem::cluster_by_kMedoidsLloyd(),Problem::cluster_size() const,scores::daviesBouldinIndex(Problem&),scores::dunnIndex(Problem&),scores::calinskiHarabaszIndex(Problem&),scores::adjustedRandIndex(labels,labels),scores::normalizedMutualInformation(labels,labels),DataLoader::startColumn(int),DataLoader::startRow(int),settings::paths::setDataPath(const fs::path&),settings::paths::setDataPath(const char*),settings::paths::setResultsPath(const fs::path&),settings::paths::setResultsPath(const char*),Problem::maxIter,Problem::N_repetition canonical_deprecation_lines=0
F22_CPP_DIAGNOSTICS inventory=33/33 legacy=0/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=FAIL
```

The llfio-OFF arbiter exited 1 at the registered alias gap:

```text
F22_CPP_SILENT count=9/33 entities=Problem::readDistanceMatrix(const fs::path&),Problem::writeDistanceMatrix(const std::string&) const,Problem::writeDistanceMatrix() const,Problem::printClusters() const,Problem::writeClusters(),Problem::writeMedoidMembers(int,int) const,Problem::writeSilhouettes(),Problem::maxIter,Problem::N_repetition canonical_deprecation_lines=0
F22_CPP_DIAGNOSTICS inventory=33/33 legacy=24/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=EXPECTED_RED
```

The exact upstream leak and its separately registered repair are F45; see
`.claude/baselines/2026-07-29-f45-llfio-diagnostic-state.md`. F45 closed in
`392d3ed`; canonical LLFIO-ON and actual llfio-OFF now print the identical
24/33 expected-red ledger above. F22 product attempt 1 remains unused.

### Python runtime policy

The focused collection printed:

```text
18 tests collected in 8.26s
```

The decisive run printed:

```text
FFFFFFFFFFFF.FFF.F                                                       [100%]
16 failed, 2 passed in 7.08s
```

The 16 failures are exactly twelve silent primary operations, three silent
`ClusterResult` resolution routes, and the ordinary-use gate's exact nine
calls. The two passes are the existing `Result.medoid_indices` warning and a
silent plain package import.

### MATLAB runtime policy

Both R2024b Update 1 and R2025b Update 5 selected exactly one F22 case and
reported zero passed / one failed / zero incomplete with:

```text
F22_MATLAB_DEPRECATION aliases=15/15 warning_profiles=0/15 messages=0/15 canonical_silent=15/15 equivalence=15/15 constructor_silent=1/1 tier1_silent=1/1 skips=0 verdict=FAIL
```

Both runs used:

```text
C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
bytes=1501696
sha256=45CFBDFD2C8A4795D23790406F656FE1E5A789A8607490B8ED8832D53B78390F
```

This older MEX is valid inherited interface evidence, not final acceptance;
R4 still requires a fresh build and two-release hash/provenance gate.

### Harness notes

During registration, two guessed checker names were invalid:

```text
C:\D\git\dtw-cpp\.venv\Scripts\python.exe: can't open file 'C:\\D\\git\\dtw-cpp\\scripts\\check_generated_docs.py': [Errno 2] No such file or directory
C:\D\git\dtw-cpp\.venv\Scripts\python.exe: can't open file 'C:\\D\\git\\dtw-cpp\\scripts\\check_documentation_contract.py': [Errno 2] No such file or directory
```

They are harness errors, not F22 results. The exact tracked commands
`scripts/generate_docs.py --check` and `scripts/check_docs_contract.py`
subsequently passed.

## Pre-product C++ behavior-fixture evidence

Commit `c210504` adds the exhaustive behavior fixture and makes the existing
`test_problem_api_2_0` CTest entry run the real compiler-diagnostic driver
before the Catch2 executable. It adds no CTest entry. The launcher requires
Python 3.9 only when tests are enabled; test-off/core configurations remain
Python-free. A generator-independent, `EXCLUDE_FROM_ALL` object-probe fallback
preserves the diagnostic gate on Visual Studio, where
`compile_commands.json` is unavailable.

The fallback itself was forced in the canonical build before relying on it. It
compiled all three marked subjects and printed:

```text
F22_CPP_SILENT count=9/33 entities=Problem::readDistanceMatrix(const fs::path&),Problem::writeDistanceMatrix(const std::string&) const,Problem::writeDistanceMatrix() const,Problem::printClusters() const,Problem::writeClusters(),Problem::writeMedoidMembers(int,int) const,Problem::writeSilhouettes(),Problem::maxIter,Problem::N_repetition canonical_deprecation_lines=0
F22_CPP_DIAGNOSTICS inventory=33/33 legacy=24/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=EXPECTED_RED
F22_FORCE_PROBES_EXIT=1
```

The preferred compile-database path printed the same two diagnostic lines and
`F22_DIRECT_DRIVER_EXIT=1`. The real canonical executable then printed:

```text
F22_CPP_COMPAT inventory=33/33 behavior=33/33 field_routes=4/4 io_routes=7/7 file_identity=6/6 stdout_identity=2/2 skips=0 verdict=PASS
===============================================================================
All tests passed (229 assertions in 5 test cases)

F22_CPP_BEHAVIOR_CANONICAL_EXIT=0
```

The llfio-OFF/HiGHS-OFF executable independently exercised the exact
`SolverError` and unchanged-state route and printed:

```text
F22_CPP_COMPAT inventory=33/33 behavior=33/33 field_routes=4/4 io_routes=7/7 file_identity=6/6 stdout_identity=2/2 skips=0 verdict=PASS
===============================================================================
All tests passed (229 assertions in 5 test cases)

F22_CPP_BEHAVIOR_NOLLFIO_EXIT=0
```

Its preferred diagnostic driver again reported exactly 24/33 and the same nine
silent entities. The combined CTest entry failed in both builds with exit 8
because the required 33/33 diagnostic PASS marker was absent; each output
contained only the exact 24/33 `EXPECTED_RED` marker above and did not launch
the behavior child. Both CTest inventories remain 122.

Verdict: **BEHAVIOR FIXTURE PASS / PRODUCT DIAGNOSTICS EXPECTED RED
[confirmed]** by commit `c210504` and the verbatim outputs above. F22 product
attempt 1 remains unused.

### R2 — C++ diagnostics, ownership, and behavior

The permanent public-header gate table-drives all 33 entities:

- exact overload types for 31 functions and `int Problem::*` for two fields;
- each unsuppressed legacy use emits the registered replacement diagnostic;
- the suppression control compiles, proving that the intended deprecation is
  the only failure;
- all canonical counterparts compile together under
  `-Werror=deprecated-declarations` with zero diagnostic output;
- zero skips.

It prints exactly:

```text
F22_CPP_DIAGNOSTICS inventory=33/33 legacy=33/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=PASS
```

`test_problem_api_2_0` then proves behavior identity for all 33 entities,
combining its live routes with F21's already-permanent six-overload state/path
fixture. The F22 additions use asymmetric, non-degenerate data and require:

- canonical-write/legacy-read and legacy-write/canonical-read for both fields;
- digit-identical numeric/state, labels, medoids, and exception outcomes;
- byte-identical output files from both distance-writer overloads, clusters,
  medoid members, and silhouettes;
- exact stdout for print routes and failed matrix reads;
- separate RAII output roots and exact source-file cleanup.

The focused target prints:

```text
F22_CPP_COMPAT inventory=33/33 behavior=33/33 field_routes=4/4 io_routes=7/7 file_identity=6/6 stdout_identity=2/2 skips=0 verdict=PASS
```

and Catch2 reports at least 65 assertions in at least five cases. CTest rejects
skip text and requires both lines.

Source hygiene requires all 31 old functions to be inline canonical
forwarders, the seven I/O definitions to use canonical names, no canonical
implementation to call an old alias, exact public `int` field shape, and zero
ordinary legacy C++ uses outside named compatibility fixtures/history.

### R3 — Python diagnostics and behavior

`tests/python/test_deprecation_policy.py` has 18 tests:

- 13 primary operations, each with exact category/message/count, caller
  attribution, canonical silence, and behavior identity;
- direct lookup, construction, `from ... import`, and star-import resolution
  collectively prove three supplemental `ClusterResult` routes;
- plain `import dtwcpp` is silent;
- an ordinary-call allowlist is empty.

It additionally proves two consecutive uses each warn under an `always`
filter, warnings precede side effects, distance arrays are independent copies,
and `ClusterResult is Result`.

The final marker is:

```text
F22_PYTHON_GATE alias_symbols=12 operations=13 primary_warn_once=13 canonical_silent=13 equivalent=13 class_routes=3 identity=2 ordinary_legacy=0 verdict=PASS
```

Focused acceptance is 18 passed / zero skipped. The extension must be rebuilt
through `build/cfg-gate-normal`, copied with `libomp.dll`, and verified by
matching built/imported SHA-256 plus a new private native F22 discriminator.

The current Python inventory is 1,023; the new 18-case file registers a final
inventory of 1,041. With F39 still open, the full expected ledger is:

```text
1028 passed, 12 skipped, 1 known F39 failure
```

The sole failure must be
`test_supply_chain_pins.py::test_live_tracked_cmake_inventory_is_complete`
with observed 28 versus frozen 27. If F39 closes before this gate, the
arithmetically equivalent accepted ledger is 1029 passed / 12 skipped.

### R4 — MATLAB diagnostics and behavior

Add one table-driven case to `test_contract_parity.m`, keeping the five-suite
file inventory fixed. It executes the 15 alias operations with fresh fixtures:

- exactly one `dtwc:deprecatedAlias` warning and exact message;
- matching canonical operation with no warning;
- exact behavior identity on non-degenerate values;
- no warnings from construction, `set_distance_matrix`, or a CPU/L1
  `DTWClustering.fit`;
- zero subject skips.

It prints:

```text
F22_MATLAB_DEPRECATION aliases=15/15 warning_profiles=15/15 messages=15/15 canonical_silent=15/15 equivalence=15/15 constructor_silent=1/1 tier1_silent=1/1 skips=0 verdict=PASS
```

The focused case passes on both R2024b and R2025b through one freshly rebuilt
OpenMP MEX. The source/MEX hashes, observed MATLAB release, and exact
`which(...,'-all')` paths are captured and rehashed after both runs.

Adding one case to the inherited 84-test inventory registers:

- inherited red: 85 total / 81 passed / 3 failed / 4 incomplete;
- final: 85 total / 82 passed / 2 failed / 3 incomplete.

The two failed names remain exactly the F18 cases above; the third incomplete
remains exactly the opposite-flavor capability case. Any other name or count
is a regression.

### R5 — adversarial mutations

Permanent mutation runners must restore exact source bytes, verify pre/post
SHA-256 identity, run clean controls before and after, and kill:

- C++: **46/46** — 33 per-entity diagnostic removals, seven repaired I/O
  forwarding/ownership corruptions, four field read/write cross-wires, one
  canonical-warning leak, and one wrong-message family;
- Python: **31/31** — 13 warning removals, 13 behavior/identity corruptions,
  and five category/count/attribution/canonical-silence policy corruptions;
- MATLAB: **33/33** — 15 warning removals, 15 behavior target corruptions,
  duplicate warning, canonical-warning leak, and hidden internal-alias use.

No survivor is rescue-tuned. A falsified mutation band is recorded as such.

#### Executed C++ mutation verdict — FALSIFIED

Commit `26330fd` adds the permanent exact-byte runner
`scripts/test_f22_cpp_deprecation_mutations.py`. Its preflight prints:

```text
F22_CPP_MUTATION_PREFLIGHT inventory=46/46 diagnostic_removals=33/33 io_forwarding=7/7 field_crosswires=4/4 canonical_warning=1/1 wrong_message=1/1 source_files=5/5 verdict=PASS
```

Both permitted decisive attempts passed the initial real-binary control:

```text
F22_CPP_MUTATION_CONTROL label=initial diagnostic=33/33 behavior=33/33 assertions=229 cases=5 verdict=PASS
```

Both then killed and restored all 33 diagnostic-removal mutants. The last
successful per-mutant line in each attempt was:

```text
F22_CPP_MUTATION name=diag-set-results-path-cstring category=diagnostic_removal gate=diagnostic killed=1 restore=pass
```

Attempt 1 used a missing-path corruption for the first I/O forwarder. Attempt
2 replaced that with the bounded corruption
`read_distance_matrix(p); refresh_distance_matrix();`; both attempts reached
the same verbatim harness failure:

```text
F22_CPP_MUTATION_HARNESS_ERROR command failed to run: ['C:\\Program Files\\CMake\\bin\\ctest.EXE', '--test-dir', 'C:\\D\\git\\dtw-cpp\\build\\highs-1151', '-C', 'Release', '-R', '^test_problem_api_2_0$', '--output-on-failure', '-V']: Command '['C:\\Program Files\\CMake\\bin\\ctest.EXE', '--test-dir', 'C:\\D\\git\\dtw-cpp\\build\\highs-1151', '-C', 'Release', '-R', '^test_problem_api_2_0$', '--output-on-failure', '-V']' timed out after 300 seconds
```

No I/O, field, canonical-warning, or wrong-message runtime mutant is credited
as killed. The registered 46/46 band is therefore **FALSIFIED at 33/46
[confirmed]**, not partially passed. Both outer `finally` paths restored all
five targets; `git diff --quiet HEAD -- <five targets>` exited 0 after each
attempt. Post-attempt SHA-256 values were:

```text
dtwc/Problem.hpp F725479D5BF3CABA4A8246DEC7E76BDD60A3CBAD743B7881BB8C87F2F692F99C
dtwc/Problem.cpp 1D25162CEADC4E7009BC4BAC404F6A7399A982E73DF10467B76B4206C2B1C97A
dtwc/scores.hpp 6B02592A51DEAC76C90BA569C70F50F04D69608425A4807738C17EF09830B55F
dtwc/DataLoader.hpp FB267CB9BCB8259A08961D574817E5A6657785DDB54AAAB479F4AABA200A236F
dtwc/settings.hpp 8DB709C7A67C0D3E2D4093B6E0780EB9F54A4E50A98CE45C74001C81A8C82063
```

The two-attempt cap is exhausted. Do not rescue-tune a third C++ mutation run;
continue the independently registered Python and MATLAB mutation gates and
the remaining F22 evidence.

#### Executed Python mutation verdict — PASS

Commit `fd6a664` adds
`scripts/test_f22_python_deprecation_mutations.py`. Preflight resolved one
built and one installed CPython 3.13 extension and printed:

```text
F22_PYTHON_MUTATION_PREFLIGHT inventory=31/31 warning_removals=13/13 behavior_identity=13/13 policy=5/5 native=25/25 pure=6/6 source_files=3/3 built=C:\D\git\dtw-cpp\build\cfg-gate-normal\python\_dtwcpp_core.cp313-win_amd64.pyd installed=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd verdict=PASS
```

Attempt 1 killed all 31 mutants and passed its final 18/18 control, but the
runner rejected the evidence because it had imposed an unregistered
initial-hash-equals-final-hash requirement on separately linked clean
extensions. The two clean hashes were
`28C5AC8D8F15438E1E62AD45698F16B33B2CC23A4201852A0472E72A02063080`
and
`2E5A5FCE59D8276FFCF0EB11DD63F2F0C5B0C6322DA6AA994FC518791D6EC60D`.
Both artifacts independently matched their installed copy and passed 18/18;
byte-reproducible linking was never part of the registered band.

Attempt 2 retained exact source restoration and fresh built/imported identity
but removed only that false deterministic-link assumption. Its clean controls
were:

```text
F22_PYTHON_MUTATION_CONTROL label=initial tests=18/18 sha256=CD5F4B7A9B230F842731686D2AF169596E7F7E44AB15BBF9EF390B7DE8A571E8 verdict=PASS
F22_PYTHON_MUTATION_CONTROL label=final tests=18/18 sha256=B572C24045572E842B24369384E69981596C354AD3F53C2C2BD4C280D12C1B25 verdict=PASS
F22_PYTHON_NATIVE_RESTORE clean_rebuild=1 initial_sha256=CD5F4B7A9B230F842731686D2AF169596E7F7E44AB15BBF9EF390B7DE8A571E8 final_sha256=B572C24045572E842B24369384E69981596C354AD3F53C2C2BD4C280D12C1B25 built_imported_match=1 verdict=PASS
```

Every native mutant compiled `_dtwcpp_core.cpp`, produced a non-clean binary,
was copied and hash-verified through a fresh interpreter, and failed its exact
registered pytest dimension. The decisive final marker was:

```text
F22_PYTHON_MUTATIONS controls=2/2 inventory=31/31 warning_removals=13/13 behavior_identity=13/13 policy=5/5 native_kills=25/25 pure_kills=6/6 killed=31/31 survived=0 source_restores=31/31 source_files=3/3 native_restore=pass skips=0 verdict=PASS
```

Post-run `git diff --quiet HEAD --` over the three sources exited 0. The final
built and installed extension SHA-256 were both
`B572C24045572E842B24369384E69981596C354AD3F53C2C2BD4C280D12C1B25`.

#### Executed MATLAB mutation verdict — PASS

Commit `8d66125` adds the permanent dual-release exact-byte runner
`scripts/test_f22_matlab_deprecation_mutations.py`. Its preflight prints:

```text
F22_MATLAB_MUTATION_PREFLIGHT inventory=33/33 warning_removals=15/15 behavior_corruptions=15/15 policy_mutations=3/3 versions=2/2 source_files=7/7 mex=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64 verdict=PASS
```

The runner touched `dtwc_mex.cpp`'s timestamp without changing its bytes,
forced a real Release rebuild, required the compile/link lines, checked
`DTWC_ALLOW_SEQUENTIAL:BOOL=OFF`, and then printed:

```text
F22_MATLAB_MEX_BUILD source_compiled=1 linked=1 openmp_cache=1 sha256=B81FA4EF34E7845DE104DE677CC04857BAB0CE6497DC07B890396928872A91D2 bytes=685056 verdict=PASS
```

Every MATLAB process used build-local unique preferences and temporary
directories, resolved the exact repository `.m` source and fresh MEX, observed
its requested installed release, validated the exact five-field OpenMP probe,
and engaged at least two real threads. The clean controls were:

```text
F22_MATLAB_CONTROL_RESULT label=initial release=R2024b passed=1 failed=0 incomplete=0
F22_MATLAB_CONTROL_RESULT label=initial release=R2025b passed=1 failed=0 incomplete=0
F22_MATLAB_CONTROL_RESULT label=final release=R2024b passed=1 failed=0 incomplete=0
F22_MATLAB_CONTROL_RESULT label=final release=R2025b passed=1 failed=0 incomplete=0
```

MATLAB marks a test with a fatal `assert*` as both failed and incomplete. The
runner therefore registered, before mutation execution, exact
`passed=0 failed=1 incomplete=1` kills for MW01–MW04, MB01–MB08, and
MB11–MB15, and exact `passed=0 failed=1 incomplete=0` kills for the other
sixteen mutants. All 66 release results matched those fixed ledgers and their
mutation-specific diagnostic; syntax, undefined-symbol, stale-MEX, wrong-path,
wrong-release, missing-marker, and unexpected-result failures remained harness
errors rather than kills.

The decisive final marker was:

```text
F22_MATLAB_MUTATIONS controls=4/4 versions=2/2 mutations=33/33 release_kills=66/66 warning_removals=15/15 behavior_corruptions=15/15 policy_mutations=3/3 source_restores=33/33 source_files=7/7 mex_hash_checks=140/140 skips=0 survived=0 verdict=PASS
```

The complete raw ledger is
`build/mex-verify-msvc/tests/f22-matlab-mutations-attempt1.log`, SHA-256
`21407D915115F529F94C390BB08A884B9188EEBE1574EA5A1C990080D1D55748`.
It contains exactly 66 `F22_MATLAB_MUTANT_RESULT` lines, 33 per-mutant verdict
lines, and four control lines. The completed recovery manifest has eight
records and independently rehashes to:

```text
bindings/matlab/+dtwc/Problem.m CDBE501BA6013FAEC3CE4B80B116BF1D40241131F40833EC296529399FF7FEDF
bindings/matlab/+dtwc/DTWClustering.m 756BA4D335D85844C70DB3DDA78ED2505488962CF99CCA4B46DECC503986F6D1
bindings/matlab/+dtwc/davies_bouldin_index.m 8994F006ED0758406B031CF2AD4BA0AF1A87AF13B615FA3B2A6E705EABF62F6D
bindings/matlab/+dtwc/dunn_index.m 6B21D9F94900BC2B7CDE487544F59B04483C5130DE06C2ED084CE28F561E21E9
bindings/matlab/+dtwc/calinski_harabasz_index.m F8CECE1FEEDA33D5861E9CCD66F2309259BCDDE65C4C20DF83E68082CC7CD6C3
bindings/matlab/+dtwc/adjusted_rand_index.m 4FF76BDABE944E69EEE7EC32499655CD69595F8E458426377D631EE30D7B0E1C
bindings/matlab/+dtwc/normalized_mutual_information.m 5888161C79501B4D3639DAF8EA786C68A5FE45962505BD687DBC4B82B6AAE378
build/mex-verify-msvc/bin/dtwc_mex.mexw64 B81FA4EF34E7845DE104DE677CC04857BAB0CE6497DC07B890396928872A91D2
```

`git diff --quiet HEAD -- <seven targets>` exited 0 after the run, and a fresh
runner preflight passed. Earlier launcher/audit interruptions were rejected
before any mutant result: one terminal wrapper exited 124 while its clean MEX
build child remained, and two source reviews found the marker-reachability and
fatal-incomplete ledger assumptions during clean-control startup. The exact
process trees were stopped, the seven targets were checked clean, both
assumptions were fixed before the decisive run, and none is credited as a
mutation attempt.

#### Executed documentation and ordinary-call verdict — PASS

Commit `43e1c44` synchronises the authoritative contract and generated Tier-1,
Tier-2, and migration pages with the implemented 33/13/15 deprecation
inventories. It also migrates every ordinary call found by the independent
whole-tree sweep and adds a permanent, path-attributed checker over the
affected command, skill, guide, report, and executable-script surfaces.

The checker pins 30 unique C++ names / 33 diagnostic entities, 12 Python
symbols / 13 warning operations, and 15 MATLAB warning operations. It executes
a positive control for every registered name/operation family and negative
controls for the private `fillDistanceMatrix_BruteForce`, the canonical
no-argument `startColumn()`/`startRow()` getters, MATLAB's `'Band'` option name,
and native `ClusteringResult.medoid_indices`.

The decisive documentation gate printed:

```text
generated documentation is current
generated documentation is current
documentation contract checks passed
record hygiene checks passed
F22_DOC_CANONICAL_IMPORTS count=5/5 verdict=PASS
```

`py_compile` over `scripts/check_docs_contract.py` and all five migrated Kasper
executables exited 0. `git diff --check` exited 0 with only the repository's
line-ending conversion warnings. A second adversarial source-anchor and
inventory review reported no blocker after every enumerated stale local line
reference was corrected. The ordinary-call verdict is **0 remaining
[confirmed]** by `scripts/check_docs_contract.py` plus the independent
whole-tree inventory sweep; compatibility declarations, fixtures, generated
rename inventories, historical records, native result fields, and the private
brute-force implementation are deliberately excluded.

### R6 — documentation, immutable hygiene, and full native gates

- `CHANGELOG.md` Unreleased names the warning policy and retained aliases.
- The authoritative and generated contract pages resolve the MATLAB writer
  ambiguity and no longer claim F22 is open.
- Both documentation checkers and record hygiene pass.
- Ordinary legacy calls are zero outside the named compatibility fixtures.
- Canonical: 122/122, zero failed, exactly six capability skips.
- llfio-OFF: 122/122, zero failed, exactly nine capability skips.
- Arrow-ON: 124/124, zero failed, exactly eight capability skips, with
  `test_io_readers` executing.
- CTest and tracked CMake inventories do not move.

## Rollback and most-likely-wrong claim

Every change is local and reversible by its eventual conventional commit; no
remote or irreversible action is authorized.

The claim most likely to be wrong is the adjudication that the four readable
PascalCase MATLAB configuration properties warn only on assignment. The
frozen table gives each only a canonical setter replacement and separately
enumerates every deprecated read-only property, which supports that decision;
an explicit canonical getter registry entry would be required to change it.
