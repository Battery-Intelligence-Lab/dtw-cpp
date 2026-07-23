# F9 Arrow + Parquet execution gate — 2026-07-23

## Scope and baseline

Finding F9 is that `test_io_readers` is registered in the canonical
Arrow-OFF build but exits through Catch2 `SKIP`, which CTest counts as a
passing target.  The current source has four `[io][arrow]` cases and seven
`[io][parquet]` cases (`tests/unit/test_io_readers.cpp:122-452`).  The previous
manual PyArrow-linked run recorded only the seven Parquet cases:

```text
test_io_readers [io][parquet]: 348 assertions, 7 cases
```

Source: `.claude/baselines/2026-07-12-fast-clara-streaming.md`.

Base commit for the decisive local rebuild:

```text
741d51a docs: record F7 routing coverage
```

The official Apache Arrow installation page was re-read on 2026-07-23.  It
lists the same Ubuntu repository bootstrap used by the proposed workflow and
separate `libarrow-dev` and `libparquet-dev` packages:
<https://arrow.apache.org/install/>.

## Registered bands (written before the decisive runs)

### Primary local closure

Build a fresh CMake directory against the Arrow and Parquet libraries bundled
with the already-installed local PyArrow 23 wheel.  The build must:

1. configure with `DTWC_ENABLE_ARROW=ON`, `DTWC_ENABLE_LLFIO=OFF`, and the
   unrelated optional solvers/accelerators OFF;
2. compile the current `test_io_readers` target with both
   `DTWC_HAS_ARROW` and `DTWC_HAS_PARQUET`;
3. link the current production `dtwc++` library and import both Arrow and
   Parquet runtime libraries;
4. run the test binary itself with the skip text absent; and
5. print a Catch2 success summary with **at least 348 assertions in at least
   11 test cases**.

The assertion floor comes from the prior seven-case Parquet run.  The case
floor is the count in the current non-skip source: four Arrow IPC plus seven
Parquet.  A missing or unparseable summary is failure.

The direct `[io][parquet]` slice must independently print at least 348
assertions in at least seven cases.  The CTest wrapper must also run the
subject without a skip.

### Secondary CI closure

The proposed Ubuntu job must:

- use Apache's documented repository bootstrap and install both development
  packages;
- configure with `DTWC_ENABLE_ARROW=ON`;
- reject the Arrow-disabled skip text and any generic skip report;
- parse both the assertion and case counts from the test binary's own Catch2
  summary;
- require at least 348 assertions and 11 cases; and
- fail closed when the summary is absent or malformed.

Local workflow validation band:

1. YAML parses successfully;
2. `actionlint` passes if installed, otherwise its exact missing-command probe
   is recorded `[BLOCKED-ENV]` and a named YAML-parser fallback is used;
3. the shell evidence parser accepts exactly `348 assertions in 11 test
   cases`, and rejects each of: skip text, 347 assertions, 10 cases, and a
   missing summary.

The hosted GitHub Actions execution is operator-owned and is not claimed by
this local record.

### Regression gate

After the workflow and local proof are repaired, the canonical build must
rebuild and pass **114/114 targets, zero failed**, with exactly the six known
capability skips:

```text
test_cuda_correctness
test_cuda_lb_keogh
test_io_readers
test_metal_correctness
test_metal_lb_keogh
test_metal_mmap
```

The canonical `test_io_readers` skip remains intentional because that build
proves the optional-dependency-OFF configuration; the new Arrow-ON directory
is the complementary execution gate.

## Results

### Deliberate red: system Parquet scope leak

The fresh configure used CMake 4.2.3, Ninja 1.13.2, Clang 21.1.8, PyArrow
23.0.1, Release mode, and a build-local pair of CMake package shims under
`build/f9-arrow-config/`.  The shims only describe the already-installed
PyArrow headers and import libraries; they do not alter the source tree or the
test binary.

Configure found both packages:

```text
--   Arrow:    YES (v23.0.1) — system install
--   Parquet:  YES (v23.0.1) — system install
-- Arrow + Parquet linked — IPC and Parquet reading enabled
```

But the generated compile command proved the announcement false:

```text
-DDTWC_HAS_ARROW -DDTWC_HAS_OPENMP
```

`DTWC_HAS_PARQUET` was absent.  The direct binary therefore ran only four
Arrow cases and also exposed the independent Windows fixture-lifetime defect:

```text
C:/D/git/dtw-cpp/tests/unit/test_io_readers.cpp(139): FAILED:
  {Unknown expression after the reported line}
due to unexpected exception with message:
  remove: The process cannot access the file because it is being used by
  another process.: "C:\Users\engs2321\AppData\Local\Temp\dtwc_io_reader_test\
  valid_ndim2.arrow"

===============================================================================
test cases:  4 |  3 passed | 1 failed
assertions: 43 | 42 passed | 1 failed

exit=42
```

Root cause: `find_package(Parquet)` ran inside
`dtwc_setup_dependencies()`, but only an internal capability variable—not
`Parquet_FOUND`—was exported.  The parent `dtwc/CMakeLists.txt` tested the
unexported package variable.  Repair `833f570` exports the result alongside the
existing capability signal.

The regenerated compile command then contained:

```text
-DDTWC_HAS_ARROW -DDTWC_HAS_OPENMP -DDTWC_HAS_PARQUET
```

The newly reachable Parquet slice passed:

```text
Filters: [io] [parquet]
Randomness seeded to: 2703213395
===============================================================================
All tests passed (348 assertions in 7 test cases)

exit=0
```

### Primary local gate: PASS

Repair `0c91c9b` scopes the mmap-owning `ArrowIPCDataSource` so it is destroyed
before the fixture is unlinked.  The complete current binary then passed above
both registered floors:

```text
Randomness seeded to: 2956850487
===============================================================================
All tests passed (390 assertions in 11 test cases)

exit=0
```

CTest also ran—not skipped—the same subject:

```text
54: All tests passed (390 assertions in 11 test cases)
54:
1/1 Test #54: test_io_readers ..................   Passed    0.29 sec

The following tests passed:
	test_io_readers

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.31 sec
exit=0
```

`llvm-readobj --coff-imports` independently confirmed the runtime boundary:

```text
Name: parquet.dll
Name: arrow.dll
Name: libomp.dll
```

### Secondary CI gate: local validation PASS, hosted run not claimed

The workflow now pins Ubuntu 24.04, installs only the relevant optional
capability, and delegates its evidence decision to
`.github/scripts/assert-arrow-suite.sh`.  The Apache repository bootstrap
downloaded on 2026-07-23 was 80,002 bytes with SHA-256:

```text
5B6BEE96EBBDA57BC4DBB2DD262B1ECD7AEBB63DBC47733E3F1A32AD8148ED81
```

That digest is registered in the workflow before the package is installed.
The exact production parser passed the current binary:

```text
F9 gate PASS: test_io_readers executed 390 assertions in 11 cases, no skips.
pipeline-exit=0
```

Synthetic mutation probes exercised the same script:

```text
exact-floor: exit=0, expected=0
  F9 gate PASS: test_io_readers executed 348 assertions in 11 cases, no skips.
skip: exit=1, expected=1
  ::error::F9: test_io_readers SKIPPED - Arrow did not compile into dtwc++.
low-assertions: exit=1, expected=1
  ::error::F9: only 347 assertions ran, expected >= 348.
low-cases: exit=1, expected=1
  ::error::F9: only 10 test cases ran, expected >= 11 (4 Arrow + 7 Parquet).
missing-summary: exit=1, expected=1
  ::error::F9: expected exactly one parseable Catch2 success summary.
ambiguous-summary: exit=1, expected=1
  ::error::F9: expected exactly one parseable Catch2 success summary.
```

The named YAML fallback passed:

```text
PyYAML 6.0.3: PASS .github\workflows\ubuntu-unit.yml
```

The preferred workflow and shell linters are environment-impossible:

```text
[BLOCKED-ENV] actionlint: The term 'actionlint' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
```

```text
/bin/bash: line 1: shellcheck: command not found
```

`bash -n .github/scripts/assert-arrow-suite.sh` and `git diff --check` both
returned exit zero.  The hosted GitHub Actions job remains an operator-owned
verification and is not represented as executed here.

### Canonical regression gate

PASS.  After commits `833f570`, `0c91c9b`, and `e323197`, the canonical build
reconfigured and rebuilt successfully.  CTest met the registered band exactly:

```text
100% tests passed, 0 tests failed out of 114

Total Test time (real) = 103.24 sec

The following tests did not run:
	 48 - test_cuda_correctness (Skipped)
	 50 - test_cuda_lb_keogh (Skipped)
	 54 - test_io_readers (Skipped)
	 55 - test_metal_correctness (Skipped)
	 56 - test_metal_lb_keogh (Skipped)
	 57 - test_metal_mmap (Skipped)
```

A separate verbose run of those six targets printed their own capability
messages.  In particular, the canonical reader binary said:

```text
54: C:/D/git/dtw-cpp/tests/unit/test_io_readers.cpp(37): SKIPPED:
54: explicitly with message:
54:   DTWC_HAS_ARROW not defined — Arrow/Parquet readers not built
54:
54: ================================================================================
54: test cases: 1 | 1 skipped
54: assertions: - none -
```

That is the intended optional-dependency-OFF half of the two-build gate; the
fresh Arrow-ON build above is the execution half.

## Verdict

**PASS / KEEP.**  Primary local closure exceeded both preregistered floors:
390 assertions (floor 348) in 11 cases (floor 11), with no skip.  The canonical
regression gate passed 114/114 with exactly the six registered capability
skips.  The secondary Ubuntu workflow is locally syntax- and
mutation-validated, while its hosted execution remains explicitly
operator-owned and unclaimed.
