# Phase 6/7 floor reproduction — 2026-07-10

This is the durable evidence that was missing from the rc1 omnibus commit
`8debf1d`. The run was orchestrated at `b160269` (the only intervening tracked
change opened Phase 8 in `PLAN.md`); production C++/Python/MATLAB sources were
still byte-for-byte those of `8debf1d`.

## Registered bands (written before the reruns)

- Native Clang/HiGHS/Gurobi gates: exactly 99 registered tests and zero
  failures in each configuration. LLFIO ON must have six named capability
  skips; LLFIO OFF must have those six plus the two mmap suites.
- Python gate: exactly 418 collected tests, zero failures. The release-wheel
  configuration must report 407 passed / 11 capability skips; a developer
  extension without HiGHS is recorded separately rather than blended into the
  release floor.
- MATLAB gate: 61 passed, zero failed/incomplete, and the OpenMP runtime probe
  must engage at least two threads.
- Wheel smoke: exit 0, OpenMP engaged, bundled HiGHS available, and a real MIP
  solve completed.
- Sdist: zero repository-build, generated-site, virtual-environment,
  site-packages, or `.git` entries; required version/build/license inputs
  present.
- TestPyPI dry-run: both the wheel and sdist accepted for checking, with no
  upload attempted.
- C++/Python/MATLAB quickstarts: labels `0x9 1x9 2x9`, medoids `4 13 22`, and
  mean silhouette `0.968950` (C++ prints the equivalent rounded `0.96895`).
- Documentation: generated-source/contract drift gate passes; pinned Hugo
  build exits 0; the complete Hugo + Doxygen site has zero broken internal
  links.
- Native archive: CPack exits 0 and the archive smoke test executes the packed
  CLI successfully.

The hidden `[.][50k]` OneBatchPAM case is not selected by normal CTest and is
not part of this floor reproduction. Its degenerate fixture is repaired and
the expensive simulation rerun separately under Task 8.1-M6.

## Native C++ gate

Preflight: `build/highs-1151` points at this checkout and uses Clang 21.1.8,
Ninja, Release, LLFIO ON, HiGHS ON, Gurobi ON, CUDA OFF, and MPI OFF. `ctest -N`
registered exactly 99 tests. The current build reported `ninja: no work to do`.

Command:

```powershell
ctest --test-dir build/highs-1151 --output-on-failure --no-tests=error -j 4
```

Decisive output (verbatim):

```text
100% tests passed, 0 tests failed out of 99

Total Test time (real) =  39.05 sec

The following tests did not run:
	 38 - test_cuda_correctness (Skipped)
	 39 - test_cuda_lb_keogh (Skipped)
	 43 - test_io_readers (Skipped)
	 44 - test_metal_correctness (Skipped)
	 45 - test_metal_lb_keogh (Skipped)
	 46 - test_metal_mmap (Skipped)
```

Verdict: **PASS — 99 registered, 93 executed, zero failures, six named
capability skips.** The old
PLAN wording of "8 capability skips" described a different LLFIO-OFF
configuration and is disambiguated alongside this log.

A separate fresh Clang 21.1.8/Ninja/Release build was then configured in
`build/phase8-protocol-nollfio` with LLFIO OFF, HiGHS ON, Gurobi ON, OpenMP
5.1, and exactly 99 registered tests.

Command:

```powershell
ctest --test-dir build/phase8-protocol-nollfio --output-on-failure --no-tests=error -j 4
```

Decisive output (verbatim):

```text
100% tests passed, 0 tests failed out of 99

Total Test time (real) =  29.27 sec

The following tests did not run:
	 27 - unit_test_mmap_data_store (Skipped)
	 28 - unit_test_mmap_distance_matrix (Skipped)
	 38 - test_cuda_correctness (Skipped)
	 39 - test_cuda_lb_keogh (Skipped)
	 43 - test_io_readers (Skipped)
	 44 - test_metal_correctness (Skipped)
	 45 - test_metal_lb_keogh (Skipped)
	 46 - test_metal_mmap (Skipped)
```

Verdict: **PASS — 99 registered, 91 executed, zero failures, eight named
capability skips.**

## Python floors and compiled-extension provenance

The release wheel was freshly built from the current source with:

```powershell
uv build --wheel --out-dir build/phase8-protocol-dist
```

It has SHA-256
`93BEC7D1888094572213E9EE14C14D21D114F0D4E5EF472AF58D9BFB82DB604D`,
21 archive entries, and zero forbidden top-level `include/`, `lib/`, `bin/`, or
`share/` entries. It was installed with the test dependencies into a new
environment under `build/phase8-wheel-floor-20260710`; no editable install or
backup extension exists there. Import provenance was:

```text
PREFIX=C:\D\git\dtw-cpp\build\phase8-wheel-floor-20260710
PKG=C:\D\git\dtw-cpp\build\phase8-wheel-floor-20260710\Lib\site-packages\dtwcpp\__init__.py
CORE=C:\D\git\dtw-cpp\build\phase8-wheel-floor-20260710\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
PKG_UNDER_PREFIX=True
CORE_UNDER_PREFIX=True
VERSION=2.0.0rc1
HIGHS=True
PARALLEL={'available': True, 'max_threads': 24, 'threads_engaged': 24, 'pass': True, 'reason': ''}
CORE_SHA256=BE864226FAD6FAA9EABAB3EC5C02A4C37CEF64E7AEF8356083103E62C83E863D
```

Collection preflight in that environment:

```text
418 tests collected in 2.62s
```

Command:

```powershell
build/phase8-wheel-floor-20260710/Scripts/python.exe -m pytest tests/python -q -rs
```

Decisive output (verbatim):

```text
SKIPPED [1] tests\python\test_cuda.py:77: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:82: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:86: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:90: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:95: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:99: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:119: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:125: CUDA not available
SKIPPED [1] tests\python\test_cuda.py:133: CUDA not available
SKIPPED [1] tests\python\test_device.py:30: no GPU available
SKIPPED [1] tests\python\test_preprocess.py:111: scipy IS installed; test only when missing
407 passed, 11 skipped in 15.24s
```

Verdict: **PASS — the complete release floor is 407 passed / 11 skipped, not
409 passed.** The two-test discrepancy is a stale historical claim, not two
missing tests: the collection contains 418 total cases.

For configuration clarity only, a separately rebuilt developer extension with
HiGHS OFF reported 406 passed / 12 skipped; the sole extra skip was the
HiGHS-specific wheel smoke. That editable/mixed environment is **not** used as
the authoritative release floor.

## MATLAB + OpenMP gate

The MEX was rebuilt from this checkout in `build/mex-verify-msvc` (Visual Studio
18 2026, `/openmp:experimental`, LLFIO/HiGHS/Gurobi OFF). The wrapper path was
added first and the fresh binary directory last, which prepends it. MATLAB
confirmed the selected binary before running:

```text
MEX=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
```

The five suites were `test_dtwc.m`, `test_mex_input_validation.m`,
`test_contract_parity.m`, `test_test_api.m`, and `test_conformance.m`.

Decisive output (verbatim):

```text
MATLAB_TOTAL=61 PASSED=61 FAILED=0 INCOMPLETE=0
labels: 0x9 1x9 2x9
medoids: 4 13 22
mean silhouette: 0.968950
OMP_AVAILABLE=1 OMP_MAX=24 OMP_ENGAGED=24 OMP_PASS=1 OMP_REASON=
```

Verdict: **PASS — 61/61 and 24 OpenMP threads engaged.**

## Wheel, sdist, and TestPyPI dry-run

Fresh release wheel SHA-256:

```text
93BEC7D1888094572213E9EE14C14D21D114F0D4E5EF472AF58D9BFB82DB604D
```

Isolated wheel smoke (verbatim):

```text
dtwcpp wheel smoke OK: 2.0.0rc1 OpenMP threads = 24 HiGHS MIP cost = 48.0
```

A new sdist was built with `uv build --sdist --out-dir
build/phase8-protocol-dist`. Its SHA-256 is
`F7309668583B45AE71E337B7CAFF95339C20A47CD6573614D363AAC951BB5158`.
The archive inspection printed:

```text
SDIST_ENTRIES=614
FORBIDDEN_ENTRIES=0
REQUIRED_ENTRIES=PASS
```

The non-uploading TestPyPI check used explicit placeholder credentials and
`--trusted-publishing never` so a local lack of GitHub OIDC could not masquerade
as an artifact failure. Decisive output (verbatim):

```text
Checking 2 files against https://test.pypi.org/legacy/
Checking dtwcpp-2.0.0rc1.tar.gz (4.1MiB)
Checking dtwcpp-2.0.0rc1-cp313-cp313-win_amd64.whl (2.1MiB)
```

Verdict: **PASS — both artifacts checked; no upload attempted.**
This validates local artifact parsing and the publish command only; it does not
claim that GitHub's tag-triggered OIDC/TestPyPI upload path ran locally.

## Quickstarts and documentation

C++ quickstart (verbatim):

```text
Reading data:
27 time-series data are read.
labels: 0x9 1x9 2x9
medoids: 4 13 22
mean silhouette: 0.96895
```

Python quickstart (verbatim):

```text
labels: 0x9 1x9 2x9
medoids: 4 13 22
mean silhouette: 0.968950
```

The MATLAB quickstart output is recorded in the MATLAB section above.

Generated docs/contract drift (verbatim):

```text
generated documentation is current
documentation contract checks passed
```

The first Hugo invocation was **FALSIFIED as a gate run** before building: the
pinned Hugo executable could not find the bundled `go` binary on `PATH`. With
the pinned Go 1.22.12 directory explicitly prepended, the real build printed:

```text
Pages            │ 66
Static files     │ 12
Total in 1343 ms
```

A Hugo-only link check initially and correctly reported 11 missing Doxygen
targets; it was not accepted as the workflow result. After generating and
staging the Doxygen subtree in the same order as CI, the decisive output was:

```text
all internal site links resolve
```

Local Doxygen generation exited 0 but emitted Graphviz warnings because
`dot.exe` is not installed. Those optional graph-render warnings are recorded,
not relabelled as a clean Doxygen graph gate; they did not remove the HTML
targets used by the internal-link check.

Verdict: **PASS for generated-doc drift, all three executable quickstarts, Hugo
generation, and the complete-site internal-link gate.**

## Native release archive

`cmake --build build/final-verify-20260710 --target package` regenerated the
Windows archive. Its SHA-256 is
`284558310BAFEC4B3EDF00DB6336DE605FEFECFF9DC72A9477E3CE6B594A9591`.

Decisive output (verbatim):

```text
CPack: - package: C:/D/git/dtw-cpp/build/final-verify-20260710/dtwc-2.0.0rc1-Windows-AMD64.zip generated.
CPack: - checksum file: C:/D/git/dtw-cpp/build/final-verify-20260710/dtwc-2.0.0rc1-Windows-AMD64.zip.sha256 generated.
release archive smoke OK: dtwc-2.0.0rc1-Windows-AMD64.zip -> C:\Users\engs2321\AppData\Local\Temp\dtwc-release-smoke-iepj1i0l\result\archive_smoke_labels.csv
```

Verdict: **PASS.** The temporary extraction path is shown because the smoke
test intentionally verifies the archive outside the source/build tree.

## Reconciled floor

- C++ LLFIO-ON gate: **99 registered / 93 executed, zero failures, 6 capability
  skips**.
- C++ LLFIO-OFF gate: **99 registered / 91 executed, zero failures, 8
  capability skips**.
- Python release-wheel gate: **407 passed, 11 skipped** (418 collected).
- Python developer/no-HiGHS diagnostic: **406 passed, 12 skipped**.
- MATLAB: **61/61**, OpenMP **24/24** threads engaged.
- Wheel smoke, clean sdist, TestPyPI dry-run, three quickstarts, docs drift,
  Hugo/internal links, and native archive smoke: **all PASS** subject to the
  explicitly recorded local Doxygen Graphviz limitation.

No production tag, TestPyPI upload, or PyPI upload was attempted.
