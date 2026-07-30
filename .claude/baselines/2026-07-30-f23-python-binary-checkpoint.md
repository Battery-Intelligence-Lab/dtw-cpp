# F23 Python binary-result checkpoint bindings - 2026-07-30

## Scope and clean base

- Base: `ab08ac1a5b514ed07f45b50e9f8f6168d621feb6`
  (`docs: close D2 campaign task`).
- `git status --short` produced no output.
- Subject: expose the frozen binary `ClusteringResult` checkpoint writer and
  reader through the live public Python package as
  `save_binary_checkpoint(result, path) -> None` and
  `load_binary_checkpoint(path) -> ClusteringResult`.
- Product attempts: at most two. The committed red-first tests and the inherited
  baseline are not product attempts. A failed product compile or decisive
  runtime gate consumes an attempt; no registered band may be weakened after a
  run.
- Precondition: F51's wire-canonicality repair must be green before either
  Python function is exposed. This preflight finding was registered before any
  F23 product or test edit.
- Rollback is the eventual local F23 product commit plus its test/documentation
  commits in reverse order. No remote or operator state may change.

The live killed-ideas section, archived checkpoint decisions, `LESSONS.md`, the
F17 baseline/handoff, the frozen API contract, the C++ serializer/deserializer,
the MATLAB binding, and the current Python extension/package/tests were inspected
before registration. No F23-specific route was killed. F17 deliberately froze
binary v1 as completed-result replay rather than method-specific continuation.

## Confirmed inherited state

At the clean base, both the built and installed extension are the same artifact:

```text
C:\D\git\dtw-cpp\build\cfg-gate-normal\python\_dtwcpp_core.cp313-win_amd64.pyd
0E6FCE5C3C312B916E845F1A1D30F2E86F37B663D30809EAE131160562932BDF
C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
0E6FCE5C3C312B916E845F1A1D30F2E86F37B663D30809EAE131160562932BDF
```

The import route and missing-symbol observations are:

```text
PUBLIC_FILE=C:\D\git\dtw-cpp\python\dtwcpp\__init__.py
CORE_FILE=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
SAVE_BINARY=False
LOAD_BINARY=False
CORE_SAVE_BINARY=False
CORE_LOAD_BINARY=False
```

Source SHA-256 values at registration are:

```text
python/src/_dtwcpp_core.cpp
6C247C4554BA57BBC3A71B21179CCA81F67F0E84EF8246A73BAE4DB4A19B81B5
python/dtwcpp/__init__.py
7B92FA33A154E6393B4EB55D6566D39BA33BAE57B9937D8055EE6E888EE4FF42
dtwc/checkpoint.cpp
00F09F3BFEC55BCE039547A4E83E6BF54517541003D204955312DFF66EFCC130
tests/python/test_contract_parity.py
F1FA984228A121CDF528E312F6F9F135BFF3EEE5A7C216070A1DC7C19B4F06C7
```

The configured Python gate is Release, Clang/Ninja, Python 3.13, llfio ON,
Arrow OFF, HiGHS OFF. The inherited collections are:

```text
155 tests collected in 1.74s
1041 tests collected in 1.88s
2 tests collected in 3.57s
1043 tests collected in 2.55s
```

Those are respectively contract parity, `tests/python`, conformance, and their
combined public-Python inventory. The last recorded full result for the
1,041-item Python inventory was 1,028 passed, 12 skipped, and the sole expected
F39 supply-chain failure. D2 added two conformance nodes, so the pre-F23
arithmetic ledger is 1,030 passed + 12 skipped + 1 expected F39 red = 1,043.

The native binary checkpoint test contains one case and nine assertions:
round-trip of all five fields, missing-file false, and bad-magic false. Its
actual focused execution is recorded only after this registration.

## Frozen Python boundary

The two public functions are direct Tier-2 bindings, not a new file format:

```text
save_binary_checkpoint(result: ClusteringResult, path) -> None
load_binary_checkpoint(path) -> ClusteringResult
```

Both names must exist on `dtwcpp._dtwcpp_core`, be imported by `dtwcpp`, and
appear in `dtwcpp.__all__`. A stale native extension must make the public
package import fail rather than allowing the editable Python layer to mask the
missing core symbols.

`path` accepts `str | os.PathLike[str]` through nanobind's filesystem caster.
The functions release the GIL around filesystem work. Existing native
`std::runtime_error` and `std::filesystem::filesystem_error` writer failures are
translated at this public boundary to `dtwcpp.IOError`; an unrelated exception
such as `std::bad_alloc` is not misclassified as I/O. A failed native read
raises `dtwcpp.IOError` with the exact message:

```text
load_binary_checkpoint: cannot read a valid binary result checkpoint from '<path>'.
```

This intentionally differs from directory `load_checkpoint`, whose frozen
return type remains `bool`. A direct binary-result load has no destination
`Problem` and therefore returns a new `ClusteringResult`; silently returning an
empty result or `None` would erase the distinction between a legitimate empty
object and malformed state.

F23 does not change binary-v1 bytes, add provenance, reinterpret
`converged=false`, or validate a result against an external N/k. The preflight
found that a thin binding would expose unchecked signed count allocation,
native-endian decoding despite the documented little-endian format, and
noncanonical reserved/padding/convergence/trailing bytes. F51 therefore owns
that narrow deterministic wire-canonicality prerequisite. The later
checkpoint/config robustness lens retains randomized corruptions, fuzz
crash/hang/leak work, semantic/provenance/authentication policy, and CLI/TOML
combinations; F51's fixed corpus seeds rather than duplicates it.

## Independent wire oracle

The non-degenerate fixture is deliberately non-uniform and coherent: each
medoid's label equals its medoid-vector slot.

| Field | Registered value |
|---|---|
| labels | `[2, 1, 0, 2, 2, 0, 0]` |
| medoid indices | `[6, 1, 4]` |
| total cost | `-13.25` |
| iterations | `0x01020304` = `16909060` |
| converged | `true` |

An independent Python `struct.pack` oracle using
`<4sHHiiiB3xd3i7i` produced:

```text
WIRE_LENGTH=72
WIRE_SHA256=DC832EDBD214FD847B7EC8BC57884881F1CEA196139FAC7DD5B939D5E7CD1A98
WIRE_HEX=44434b5001000000030000000700000004030201010000000000000000802ac006000000010000000400000002000000010000000000000002000000020000000000000000000000
DOUBLE_HEX=0000000000802ac0
```

The byte count is derived independently:

```text
4 magic + 2 version + 2 reserved + 3*4 signed header
+ 1 converged + 3 padding + 8 binary64
+ 3*4 medoids + 7*4 labels = 72 bytes
```

This fixture catches field-order, count, padding, integer-width, byte order,
convergence, signed-binary64, and sign errors. The finite negative cost is valid
for Soft-DTW results and is exactly representable, so no rounded decimal claim
substitutes for the registered eight raw bytes. Two independent read-only
computations (.NET `BinaryWriter`/SHA-256 and Node literal decoding/hash) agree
with the Python oracle. This host reports little endian; current production
native-object writes match the registered bytes here but are not evidence for a
big-endian implementation.

## Registered red-first tests

Add `tests/python/test_binary_checkpoint.py` before product work. Its module
imports both names from the live public package at collection time, so the
inherited extension must produce a collection error naming a missing F23
symbol. Also add the two frozen names to the hard-coded `_CHECKPOINT` contract
inventory. Before implementation:

1. the focused file fails during import/collection and runs zero subject tests;
2. the two new contract-parity parameters fail because both public symbols are
   absent;
3. no passing F23 marker exists.

Any inherited green is a false subject and stops product work until explained.

After implementation the focused file has exactly three tests:

1. exact bytes plus two production C++ reader calls, all ten loaded field
   comparisons, `None` writer return, missing/malformed typed read errors, and
   a typed writer-open error;
2. public/core export identity for both names;
3. exact membership of both names in `dtwcpp.__all__`.

The sole green marker is:

```text
F23_PYTHON_CHECKPOINT exports=2/2 fields=10/10 bytes=72/72 cpp_reader=2/2 io_errors=3/3 skips=0 verdict=PASS
```

The focused acceptance band is exactly 3 passed, zero failed/error/skipped,
one marker, the 72-byte SHA-256 above, and all counters exact. Contract parity
must become 157/157. No test may use `skip` or `xfail`.

## Fresh-extension and regression gates

Before a decisive green:

1. clean-first rebuild `_dtwcpp_core` in `build/cfg-gate-normal`;
2. rebuild its sibling `dtwc_cl` after the clean so full Python route selection
   cannot fall through to a stale external CLI;
3. copy the fresh `.pyd` and `libomp.dll` into the venv package;
4. print package/core paths plus built/installed SHA-256 equality;
5. import both newly added symbols before pytest.

All pytest temporary state is rooted under `build/`. The combined post-F23
Python/conformance inventory is registered at exactly 1,048 nodes:

```text
1035 passed, 12 skipped, 1 expected F39 failure
```

The sole allowed failure is
`tests/python/test_supply_chain_pins.py::test_live_tracked_cmake_inventory_is_complete`
with its inherited 28-versus-27 F39 discrepancy. If F39 closes before this gate,
the arithmetically equivalent band is 1,036 passed and 12 skipped. Any other
failure or error is F23 red.

The existing native binary reader/writer test must still execute at its
inherited assertion/case floor. The serial native matrices remain:

- canonical: 123/123, zero failed, exact six capability skips;
- llfio OFF: 123/123, zero failed, exact nine capability skips;
- Arrow ON: 125/125, zero failed, exact eight capability skips, reader
  390 assertions / 11 cases.

Documentation acceptance requires the frozen contract and rendered Tier-2
mirror to say both Python binary functions are live, the checkpointing/Python
guides to demonstrate the returned `ClusteringResult` and typed failure, one
Unreleased changelog line, current generated docs, documentation-contract
checks, record hygiene, repository hygiene, and zero `git diff --check`
diagnostics.

## Adversarial checks before execution

- A Python-only serializer is forbidden: both operations must call the existing
  production C++ functions.
- Comparing two Python objects alone is insufficient: bytes must match the
  independent wire oracle and each saved file must be parsed twice through the
  bound C++ reader.
- A helper import is insufficient: tests drive `import dtwcpp`, core symbols,
  package re-exports, and `__all__`.
- A stale `.pyd` cannot pass because the new symbol import precedes pytest and
  built/installed hashes must match.
- A generic `RuntimeError`/`OSError` is insufficient: both malformed and missing
  paths must be exact `dtwcpp.IOError` instances and also satisfy the frozen
  dual inheritance.
- F51 must make binary-v1 deterministic wire parsing green before F23 exposure;
  wider fuzz/semantic/config robustness remains an explicit future owner, not
  an inferred F23 green.

The claim most likely to be wrong is the exact 1,048-node full inventory because
collection can change independently while this campaign advances. It is
decisive here: any mismatch must be explained from named collected nodes before
a verdict, never silently re-registered after execution.

## Expected-red execution

Commit `90fa58d` added the permanent tests before any F23 product edit. The
built and installed native extensions remained byte-identical to the registered
stale SHA-256
`0E6FCE5C3C312B916E845F1A1D30F2E86F37B663D30809EAE131160562932BDF`.

The focused command was:

```text
.venv\Scripts\python.exe -m pytest tests/python/test_binary_checkpoint.py -q -s --basetemp=build/f23-red/tmp-focused -o cache_dir=build/f23-red/cache-focused
```

It ran zero subject tests, printed no PASS marker, and failed at the public
import exactly as registered:

```text
=================================== ERRORS ====================================
___________ ERROR collecting tests/python/test_binary_checkpoint.py ___________
ImportError while importing test module 'C:\D\git\dtw-cpp\tests\python\test_binary_checkpoint.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\engs2321\AppData\Roaming\uv\python\cpython-3.13.7-windows-x86_64-none\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\python\test_binary_checkpoint.py:12: in <module>
    from dtwcpp import load_binary_checkpoint, save_binary_checkpoint
E   ImportError: cannot import name 'load_binary_checkpoint' from 'dtwcpp' (C:\D\git\dtw-cpp\python\dtwcpp\__init__.py)
=========================== short test summary info ===========================
ERROR tests/python/test_binary_checkpoint.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 2.73s
```

The complete parity command was:

```text
$env:PYTHONDONTWRITEBYTECODE='1'; .venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/python/test_contract_parity.py -q --basetemp=build/f23-red/tmp-parity
```

Its verdict was the registered two missing symbols with every inherited node
green:

```text
.................................................FF..................... [ 45%]
........................................................................ [ 91%]
.............                                                            [100%]
================================== FAILURES ===================================
______________ test_module_symbol_exists[save_binary_checkpoint] ______________
E       AssertionError: dtwcpp.save_binary_checkpoint missing (contract §1/§2/§5/§6)
E       assert False
E        +  where False = hasattr(dtwcpp, 'save_binary_checkpoint')
______________ test_module_symbol_exists[load_binary_checkpoint] ______________
E       AssertionError: dtwcpp.load_binary_checkpoint missing (contract §1/§2/§5/§6)
E       assert False
E        +  where False = hasattr(dtwcpp, 'load_binary_checkpoint')
=========================== short test summary info ===========================
FAILED tests/python/test_contract_parity.py::test_module_symbol_exists[save_binary_checkpoint]
FAILED tests/python/test_contract_parity.py::test_module_symbol_exists[load_binary_checkpoint]
2 failed, 155 passed in 2.53s
```

**[confirmed] Expected-red verdict: PASS.** The focused public import failed
before collection, exactly two new parity nodes failed, all 155 inherited
parity nodes passed, and no F23 success marker existed. Product attempts remain
`0 / 2`.

## Product attempt 1 - invalid harness

The source-only adversarial review was GO: the save lambda snapshots all native
state before releasing the GIL; the load lambda prepares UTF-8 path text and
native result state before a scoped release, restores the GIL before throwing
the exact typed error, and returns by value; no generic binding catch can
misclassify `std::bad_alloc`. The clean-first Clang 21.1.8 build succeeded, a
separate extension build printed `ninja: no work to do.`, the sibling CLI was
rebuilt, and the combined settling build printed `ninja: no work to do.`.

The installed extension provenance was:

```text
BUILT_COUNT=1
INSTALLED_COUNT=1
BUILT_PYD=C:\D\git\dtw-cpp\build\cfg-gate-normal\python\_dtwcpp_core.cp313-win_amd64.pyd
INSTALLED_PYD=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
PYD_SHA256=9523C92343748374D70012CA51B91250E0D888982C4145B46B9582BF9D25499D
PYD_DIFFERS_FROM_STALE=True
LIBOMP_SHA256=5E6AC41ED81DFF9B41642A2F62CFD4784AA1C7CA1D348BEBD08FC54492C94466
LIBOMP_MATCH=True
PUBLIC_FILE=C:\D\git\dtw-cpp\python\dtwcpp\__init__.py
CORE_FILE=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
SYMBOLS=4/4
IDENTITIES=2/2
ALL_UNIQUE=2/2
HPC_SELECTED=C:\D\git\dtw-cpp\build\cfg-gate-normal\bin\dtwc_cl.exe
FRESH_IMPORT=PASS
2.0.0rc1
```

The first decisive command was:

```text
$env:PYTHONDONTWRITEBYTECODE='1'; $env:DTWC_CL_PATH=(Resolve-Path 'build/cfg-gate-normal/bin/dtwc_cl.exe').Path; $env:DTWC_CL_BIN=$env:DTWC_CL_PATH; .venv\Scripts\python.exe -m pytest -p no:cacheprovider -q -s --basetemp=build/f23-green/focused tests/python/test_binary_checkpoint.py
```

Its output was:

```text
E..
=================================== ERRORS ====================================
_ ERROR at setup of test_binary_checkpoint_exact_wire_roundtrip_and_io_errors _

self = WindowsPath('C:/D/git/dtw-cpp/build/f23-green/focused'), mode = 448
parents = False, exist_ok = False

>           os.mkdir(self, mode)
E           FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\D\git\dtw-cpp\build\f23-green\focused'

=========================== short test summary info ===========================
ERROR tests/python/test_binary_checkpoint.py::test_binary_checkpoint_exact_wire_roundtrip_and_io_errors
2 passed, 1 error in 2.50s
```

**[confirmed] Attempt-1 verdict: INVALID-HARNESS, registered band not met.**
`build/f23-green` did not exist, so pytest failed before the subject fixture
could run. The product source is unchanged for attempt 2, and the 3/3 band is
unchanged. Conservatively, the preregistered attempt counter advances to
`1 / 2`.

## Product attempt 2 - focused acceptance

Before retrying, `build/f23-green` was created and verified as a directory.
The unchanged product target and sibling CLI printed
`ninja: no work to do.`, and the built/installed extension pair retained
SHA-256
`9523C92343748374D70012CA51B91250E0D888982C4145B46B9582BF9D25499D`.

The same focused command then printed:

```text
F23_PYTHON_CHECKPOINT exports=2/2 fields=10/10 bytes=72/72 cpp_reader=2/2 io_errors=3/3 skips=0 verdict=PASS
...
3 passed in 2.10s
```

**[confirmed] Attempt-2 focused verdict: PASS.** Exactly three tests passed,
the sole marker printed once with every counter exact, no skip/xfail/error
occurred, and the emitted bytes matched the independently registered
72-byte/SHA-256 oracle. The product was committed immediately as `5bf517f`.
Both registered product attempts have now been executed; no later gate may
weaken or rescue-tune this implementation.

## Python regression inventories

The complete contract-parity command was:

```text
$env:PYTHONDONTWRITEBYTECODE='1'; $env:DTWC_CL_PATH=(Resolve-Path 'build/cfg-gate-normal/bin/dtwc_cl.exe').Path; $env:DTWC_CL_BIN=$env:DTWC_CL_PATH; .venv\Scripts\python.exe -m pytest -p no:cacheprovider -q --basetemp=build/f23-green/parity tests/python/test_contract_parity.py
```

It passed exactly:

```text
........................................................................ [ 45%]
........................................................................ [ 91%]
.............                                                            [100%]
157 passed in 1.91s
```

Before the combined gate, both `DTWC_CL_PATH` and `DTWC_CL_BIN` named the fresh
`build/cfg-gate-normal/bin/dtwc_cl.exe`, and the independent `_hpc` helper
selected that same path:

```text
FULL_GATE_CLI=C:\D\git\dtw-cpp\build\cfg-gate-normal\bin\dtwc_cl.exe
```

The combined registered inventory was exact:

```text
$env:PYTHONDONTWRITEBYTECODE='1'; $env:DTWC_CL_PATH=(Resolve-Path 'build/cfg-gate-normal/bin/dtwc_cl.exe').Path; $env:DTWC_CL_BIN=$env:DTWC_CL_PATH; .venv\Scripts\python.exe -m pytest -p no:cacheprovider -q --basetemp=build/f23-green/full tests/python tests/conformance/test_conformance.py
```

```text
================================== FAILURES ===================================
________________ test_live_tracked_cmake_inventory_is_complete ________________

    def test_live_tracked_cmake_inventory_is_complete():
        archive_pins, manifest_total = pins.tracked_cmake_archive_pins(ROOT)
>       assert manifest_total == 27
E       assert 28 == 27

tests\python\test_supply_chain_pins.py:493: AssertionError
=========================== short test summary info ===========================
FAILED tests/python/test_supply_chain_pins.py::test_live_tracked_cmake_inventory_is_complete
1 failed, 1035 passed, 12 skipped in 74.96s (0:01:14)
```

The Python-only floor was also executed rather than inferred:

```text
$env:PYTHONDONTWRITEBYTECODE='1'; $env:DTWC_CL_PATH=(Resolve-Path 'build/cfg-gate-normal/bin/dtwc_cl.exe').Path; $env:DTWC_CL_BIN=$env:DTWC_CL_PATH; .venv\Scripts\python.exe -m pytest -p no:cacheprovider -q --basetemp=build/f23-green/python-only tests/python
```

```text
================================== FAILURES ===================================
________________ test_live_tracked_cmake_inventory_is_complete ________________
E       assert 28 == 27
tests\python\test_supply_chain_pins.py:493: AssertionError
=========================== short test summary info ===========================
FAILED tests/python/test_supply_chain_pins.py::test_live_tracked_cmake_inventory_is_complete
1 failed, 1033 passed, 12 skipped in 76.73s (0:01:16)
```

**[confirmed] Python regression verdict: PASS against the registered expected
red.** The only failure in each inventory is the exact inherited F39
28-versus-27 supply-chain assertion. The totals are exactly 1,046 Python nodes
and 1,048 combined nodes; no F23, CLI-routing, conformance, or unrelated
failure occurred.

## Native regression matrices

All three native build commands printed `ninja: no work to do.` before
execution. The
focused native command was:

```text
ctest --test-dir build/highs-1151 -R '^unit_test_checkpoint_binary$' -j 1 -V --output-on-failure
```

Its subject evidence was exact:

```text
F51_RED_OBSERVATION corpus=85 false=85 accepted=0 threw=0 unchanged=85/85
F51_SIZE_PREFLIGHT_OBSERVATION false=1 threw=0 unchanged=1/1 allocations_1028=0
F51_BINARY_CHECKPOINT corpus=85 rejected=85 throws=0 unchanged=85/85 size_preflight=1/1 valid_bytes=72/72 fields=5/5 resave=72/72 semantic_compat=7/7 skips=0 verdict=PASS
All tests passed (298 assertions in 2 test cases)
100% tests passed, 0 tests failed out of 1
```

The three full matrices ran serially:

```text
ctest --test-dir build/highs-1151 -j 1 -V --output-on-failure
ctest --test-dir build/nollfio -j 1 -V --output-on-failure
ctest --test-dir build/arrow-pyarrow-23 -j 1 -V --output-on-failure
ctest --test-dir build/arrow-pyarrow-23 -R '^test_io_readers$' -j 1 -V --output-on-failure
```

Their verbatim summaries were:

```text
100% tests passed, 0 tests failed out of 123
Total Test time (real) =  91.60 sec
The following tests did not run:
         52 - test_cuda_correctness (Skipped)
         54 - test_cuda_lb_keogh (Skipped)
         58 - test_io_readers (Skipped)
         59 - test_metal_correctness (Skipped)
         60 - test_metal_lb_keogh (Skipped)
         61 - test_metal_mmap (Skipped)

100% tests passed, 0 tests failed out of 123
Total Test time (real) =  78.29 sec
The following tests did not run:
         37 - unit_test_mmap_data_store (Skipped)
         38 - unit_test_mmap_distance_matrix (Skipped)
         52 - test_cuda_correctness (Skipped)
         54 - test_cuda_lb_keogh (Skipped)
         58 - test_io_readers (Skipped)
         59 - test_metal_correctness (Skipped)
         60 - test_metal_lb_keogh (Skipped)
         61 - test_metal_mmap (Skipped)
         80 - unit_test_benders (Skipped)

100% tests passed, 0 tests failed out of 125
Total Test time (real) =  79.07 sec
The following tests did not run:
         37 - unit_test_mmap_data_store (Skipped)
         38 - unit_test_mmap_distance_matrix (Skipped)
         52 - test_cuda_correctness (Skipped)
         54 - test_cuda_lb_keogh (Skipped)
         59 - test_metal_correctness (Skipped)
         60 - test_metal_lb_keogh (Skipped)
         61 - test_metal_mmap (Skipped)
         80 - unit_test_benders (Skipped)

All tests passed (390 assertions in 11 test cases)
100% tests passed, 0 tests failed out of 1
```

**[confirmed] Native regression verdict: PASS.** The three matrix inventories
are exactly 123/123, 123/123, and 125/125 with exactly 6/9/8 registered
capability skips; the Arrow reader executed rather than skipped at 390
assertions / 11 cases. The stronger post-F51 binary subject executed at 298/2
with its exact marker.

## Documentation, checker controls, and hygiene

Commit `282cbb9` updates the frozen contract, generated Tier-2 mirror, both
Python/checkpointing guides, and the permanent documentation checker. The
contract limits the live path statement to valid Unicode and discloses F56
rather than claiming exhaustive native-filename coverage.

The positive gates printed:

```text
generated documentation is current

generated documentation is current
documentation contract checks passed

record hygiene checks passed

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
```

Three in-memory negative controls proved that the new checker rejects its
load-bearing drift classes without modifying the worktree:

```text
NEGATIVE_CONTROL=missing_doc_signature verdict=REJECTED detail=Python binary-checkpoint documentation drift: {'contract': ['`load_binary_checkpoint(path) -> ClusteringResult`']}
NEGATIVE_CONTROL=reordered_save_gil verdict=REJECTED detail=Python binary-checkpoint writer omits or reorders marker after offset 237: nb::gil_scoped_release release;
NEGATIVE_CONTROL=missing_native_export verdict=REJECTED detail=Python binary-checkpoint exports must each appear once in the native import and once in __all__: {'save_binary_checkpoint': {'native_import': 0, '__all__': 1}}
F23_DOC_CHECKER_NEGATIVE_CONTROLS=3/3
```

**[confirmed] F23 verdict: PASS.** Product attempt 2 met every frozen focused
counter; the public, parity, full Python, native, generated-documentation,
contract, and hygiene gates all meet their preregistered bands. F56 is an
explicit later finding, not a hidden generalisation of the valid-path evidence.
Rollback remains the F23 commits in reverse order; no remote state changed.
The claim most likely to be wrong is that the source-text ordering checker
captures every semantically equivalent GIL lifetime, because it intentionally
pins the current implementation shape rather than parsing C++ control flow.
