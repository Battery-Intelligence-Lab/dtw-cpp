# F21 — frozen C++ snake_case entry points

Date: 2026-07-29 (Europe/London)

## Subject and inherited state

Base commit:

```text
ab41930b35e1cce25af2b785c154e92628822fb1
```

The inherited public headers expose only the four legacy names:

- `DataLoader::startColumn(int)`
- `DataLoader::startRow(int)`
- `settings::paths::setDataPath(...)`
- `settings::paths::setResultsPath(...)`

F21 owns the four absent canonical names frozen by
`docs/api-contract-2.0.md` rows 35–38:

- `DataLoader::start_column(int)`
- `DataLoader::start_row(int)`
- `settings::paths::set_data_path(...)`
- `settings::paths::set_results_path(...)`

No killed idea or contrary decision names this surface. F22 retains ownership
of the exhaustive cross-language deprecation audit; F21 must nevertheless
honour the normative C++ rule for its own four rows: canonical implementations
own behaviour and the old spellings are inline
`[[deprecated("use <canonical>")]]` forwarders.

## Pre-run adversarial review

1. `DataLoader` already has a private data member named `start_row`.
   A member function with the same name is ill-formed. Rename the two coupled
   private fields to `start_col_` and `start_row_`, update every internal use,
   and make no layout, validation, or zero-based-skip change.
2. The frozen table specifies the two `(int)` builder setters, not canonical
   no-argument getters. Do not invent an unregistered getter surface.
3. Both path functions currently have exact `const fs::path&` and
   `const char*` overloads. Mirror both overloads canonically; implicit
   conversion of a string literal to `fs::path` is not proof that the exact
   C-string overload survived.
4. Each canonical function owns the state mutation. Legacy names forward to
   it, never the reverse.
5. Repository-owned production, examples, benchmarks, conformance, and
   ordinary tests migrate to canonical setters so warnings identify only
   deliberate compatibility probes.
6. Path state is process-global. The focused test snapshots and restores both
   globals through RAII and poisons the non-target path before every call, so
   a no-op or wrong-global mutation cannot inherit a false green.
7. Reuse `unit_test_DataLoader`: it is already a public consumer translation
   unit. This keeps CTest inventories and the tracked CMake-manifest inventory
   unchanged; adding a new `.cmake` driver is forbidden by scope and would
   worsen F39's separately owned 28-versus-27 mismatch.

## Registered bands (written before the decisive runs)

### R1 — inherited expected-red and legacy control

Compile one syntax-only public-header fixture against the inherited tree.

- canonical fixture exit: nonzero;
- diagnostics name all four canonical spellings;
- the `start_row` diagnostic must identify the private-member/called-object
  collision, not be relabelled as a simple missing method;
- otherwise-identical legacy fixture exit: zero.

This is expected-red evidence. It consumes no product attempt.

### R2 — exact public signatures

The permanent public-header fixture must compile these exact types:

- canonical and legacy loader setters:
  `DataLoader &(DataLoader::*)(int)` (4/4);
- canonical and legacy data/results path functions, each as both
  `void (*)(const fs::path&)` and `void (*)(const char*)` (8/8).

Canonical no-argument getters are outside this band.

### R3 — focused runtime contract

`unit_test_DataLoader` must:

- return the original receiver from every canonical and legacy builder;
- preserve asymmetric column/row values and change only the targeted field;
- produce identical canonical and legacy loader state;
- exercise all eight path overload calls with target/non-target checks;
- retain four C-string path values after their source strings die;
- restore both process-global paths even when a Catch2 assertion throws;
- print exactly:

```text
F21_CPP_NAMES canonical=4/4 legacy=4/4 overloads=12/12 loader_state=22/22 path_state=16/16 cstring_copy=4/4 skips=0 verdict=PASS
```

Catch2 floor: at least 79 assertions in at least 2 test cases. The CTest
metadata must reject skip output and require both the marker and Catch2's own
summary. Run in both canonical llfio-ON and llfio-OFF builds.

### R4 — deprecation and repository-use hygiene

- each of the four legacy names is an inline deprecated forwarder with the
  exact `use <canonical>` replacement text;
- canonical-only public-header compilation emits no deprecation diagnostic;
- a legacy-only diagnostic probe emits all four replacement names;
- `rg` finds no ordinary repository call to the four legacy setters outside
  the focused compatibility region and historical/evidence records.

F22 still owns the complete table-driven diagnostic policy across C++, Python,
and MATLAB.

### R5 — adversarial mutations

At least 12/12 registered mutations must fail the focused gate, classified as:

- remove each canonical name (4 compile kills);
- remove each legacy name while preserving the canonical name (4 compile
  kills; remove both overloads for each path name);
- swap column/row canonical assignments (2 runtime kills);
- redirect canonical data/results path assignment to the other global
  (2 runtime kills).

The mutation harness must restore exact source bytes and verify pre/post
SHA-256 identity. No mutation may touch an untracked `build*/` directory
except for generated compiler output.

### R6 — documentation and full gates

- `CHANGELOG.md` Unreleased names the four canonical functions and retained
  deprecated aliases.
- The authoritative and rendered contract pages no longer claim F21 is
  unimplemented; the drift checker passes.
- Canonical: 122/122, zero failed, exactly 6 capability skips.
- llfio-OFF: 122/122, zero failed, exactly 9 capability skips.
- Arrow-ON: 124/124, zero failed, exactly 8 capability skips, with
  `test_io_readers` executing rather than skipping.
- Tracked CMake manifests remain 28 observed versus 27 frozen, with F39 the
  sole owner of that already-recorded mismatch.

Two product attempts maximum. A falsified band is retained and not rescue-tuned
past that cap.

## Clean inherited baseline

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

The clean-tree rebuild at the base commit exited 0. The full canonical CTest
baseline then printed:

```text
100% tests passed, 0 tests failed out of 122

Total Test time (real) =  34.73 sec

The following tests did not run:
	 51 - test_cuda_correctness (Skipped)
	 53 - test_cuda_lb_keogh (Skipped)
	 57 - test_io_readers (Skipped)
	 58 - test_metal_correctness (Skipped)
	 59 - test_metal_lb_keogh (Skipped)
	 60 - test_metal_mmap (Skipped)
```

Verdict: **BASELINE PASS [confirmed]** at
`ab41930b35e1cce25af2b785c154e92628822fb1`.

## Decisive evidence

### R1 — inherited expected-red and legacy control

The first ad-hoc compiler command was invalid because it omitted the configured
RapidCSV include directory. Both fixtures stopped before the subject:

```text
dtwc\fileOperations.hpp:40:10: fatal error: 'rapidcsv.h' file not found
canonical_exit=1
legacy_exit=1
```

This is harness evidence, not an F21 verdict and not a product attempt. The
corrected command used the include directories from
`build/highs-1151/compile_commands.json`. It printed:

```text
<stdin>:5:10: error: no member named 'start_column' in 'dtwc::DataLoader'
    5 |   loader.start_column(3);
      |   ~~~~~~ ^
<stdin>:6:10: error: 'start_row' is a private member of 'dtwc::DataLoader'
    6 |   loader.start_row(7);
      |          ^
dtwc\DataLoader.hpp:242:7: note: implicitly declared private here
  242 |   int start_row{ 0 };                     //!< Starting row for data extraction
      |       ^
<stdin>:6:19: error: called object type 'int' is not a function or function pointer
    6 |   loader.start_row(7);
      |   ~~~~~~~~~~~~~~~~^
<stdin>:7:26: error: no member named 'set_data_path' in namespace 'dtwc::settings::paths'; did you mean 'setDataPath'?
    7 |   dtwc::settings::paths::set_data_path("data");
      |                          ^~~~~~~~~~~~~
      |                          setDataPath
dtwc\settings.hpp:74:13: note: 'setDataPath' declared here
   74 | inline void setDataPath(const fs::path &path) { data = path; }
      |             ^
<stdin>:8:26: error: no member named 'set_results_path' in namespace 'dtwc::settings::paths'; did you mean 'setResultsPath'?
    8 |   dtwc::settings::paths::set_results_path("results");
      |                          ^~~~~~~~~~~~~~~~
      |                          setResultsPath
dtwc\settings.hpp:82:13: note: 'setResultsPath' declared here
   82 | inline void setResultsPath(const fs::path &path) { results = path; }
      |             ^
5 errors generated.
canonical_exit=1
legacy_exit=0
```

Verdict: **R1 PASS [confirmed]**. All four canonical names are unreachable on
the inherited public-header surface, the `start_row` collision is explicit,
and the otherwise-identical legacy control compiles.

### R2/R3 — exact signatures and focused runtime contract

The permanent fixture was committed red-first in `a48635b`. After product
attempt 1 (`5e4a7b6`), both registered builds printed the exact marker and
exceeded the registered Catch2 floor:

```text
F21_CPP_NAMES canonical=4/4 legacy=4/4 overloads=12/12 loader_state=22/22 path_state=16/16 cstring_copy=4/4 skips=0 verdict=PASS
All tests passed (81 assertions in 2 test cases)
```

The canonical llfio-ON and llfio-OFF CTest invocations each reported 1/1
passed. Their CTest metadata rejects skip text and requires both lines above.

Verdict: **R2 PASS and R3 PASS [confirmed]** by the public function-pointer
assertions and runtime ledger in `tests/unit/unit_test_DataLoader.cpp`.

### R4 — deprecation diagnostics and generated documentation

The canonical-only public-header probe compiled with
`-Werror=deprecated-declarations` and exited 0. The legacy-only probe exited 1
and emitted all four registered replacement diagnostics:

```text
startColumn is deprecated: use start_column
startRow is deprecated: use start_row
setDataPath is deprecated: use set_data_path
setResultsPath is deprecated: use set_results_path
legacy_messages=True,True,True,True
```

The documentation gates printed:

```text
generated documentation is current
generated documentation is current
documentation contract checks passed
```

The repository-use search and final immutable-commit documentation rerun remain
part of R6.

Verdict: **R4 diagnostic sub-band PASS [confirmed]**. Final call-site hygiene
is pending the R6 immutable-product check.

### R5 — permanent adversarial mutation gate

The permanent harness is `scripts/test_f21_cpp_rename_mutations.ps1`, committed
in `36b9c99`. Its decisive stdout was:

```text
F21_CONTROL label=initial build=pass test=pass
F21_MUTATION name=remove-canonical-start-column class=compile result=killed
F21_MUTATION name=remove-canonical-start-row class=compile result=killed
F21_MUTATION name=remove-canonical-data-path class=compile result=killed
F21_MUTATION name=remove-canonical-results-path class=compile result=killed
F21_MUTATION name=remove-legacy-start-column class=compile result=killed
F21_MUTATION name=remove-legacy-start-row class=compile result=killed
F21_MUTATION name=remove-legacy-data-path class=compile result=killed
F21_MUTATION name=remove-legacy-results-path class=compile result=killed
F21_MUTATION name=swap-start-column-assignment class=runtime result=killed
F21_MUTATION name=swap-start-row-assignment class=runtime result=killed
F21_MUTATION name=redirect-data-path class=runtime result=killed
F21_MUTATION name=redirect-results-path class=runtime result=killed
F21_CONTROL label=final build=pass test=pass
F21_MUTATIONS controls=2/2 mutations=12 killed=12 compile_killed=8 runtime_killed=4 survived=0 source_restore=pass verdict=PASS
```

Stderr was empty. `git status --short` was empty immediately afterward. The
restored source hashes were:

```text
fb267cb9bcb8259a08961d574817e5a6657785ddb54aaab479f4aaba200a236f  dtwc/DataLoader.hpp
8db709c7a67c0d3e2d4093b6e0780eb9f54a4e50a98ce45c74001c81a8c82063  dtwc/settings.hpp
```

Verdict: **R5 PASS [confirmed]**. All 12 registered mutants were killed, both
controls passed, and the exact source bytes were restored.

R6 full-matrix and immutable-product checks pending.
