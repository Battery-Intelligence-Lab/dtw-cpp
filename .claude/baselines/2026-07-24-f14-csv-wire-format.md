# F14 distance-matrix CSV wire-format contract - 2026-07-24

## Scope and base

- Branch: `Claude`
- Base commit: `95ffd63e389b008335851cef3e54f7e867bbbff7`
  (`docs: record F13 assignment verdict`)
- Base status: clean; `git status --porcelain=v1` and `git diff --check`
  produced no output.
- Canonical build: `build/highs-1151` (clang, Ninja, Release, HiGHS ON,
  llfio ON, Arrow OFF).
- llfio-OFF build: `build/nollfio`.
- Arrow build: `build/arrow-pyarrow-23`.
- Subject: the four C++ distance-matrix CSV formatter bodies and their native
  public consumers. Python and MATLAB own additional emitters; the separately
  confirmed cross-language contradiction is assigned to F37 below rather than
  silently added to F14.

This registration precedes every new F14 test, production edit, decisive
execution, and mutation run. Existing-artifact inspection and source inventory
below are preflight evidence only.

The killed-ideas section, archived plan, `.claude/LESSONS.md`, and the R4
duplication boundary were searched. No killed idea conflicts with fixing the
wire format. R4 still owns formatter consolidation: F14 may change each
existing body enough to meet one behavioral contract. A shared scalar-token
formatter and a shared finite-preflight primitive are permitted so one policy
cannot drift; the four row/delimiter/output loops must remain independent.
F14 must not remove a body, redirect one whole body through another, or claim
the later duplication task.

## Confirmed inherited inventory

Four formatter bodies currently exist:

1. dense file writer, `io::write_csv`,
   `dtwc/core/matrix_io.hpp:41-59`;
2. dense stream operator, `matrix_io.hpp:121-136`;
3. mmap stream operator, `matrix_io.hpp:139-153`;
4. mmap `Problem::writeDistanceMatrix` visitor body,
   `dtwc/Problem_IO.cpp:156-180`.

The dense `Problem` visitor delegates to body 1. Native public consumers are:

- `Result::save`, which visits dense/mmap and writes through the stream
  operators (`dtwc/api.cpp:253-260`);
- `Problem::print_distance_matrix`, which visits both stream operators and
  currently appends another newline (`dtwc/Problem.cpp:183-188`);
- the real CLI matrix-output route
  (`dtwc/dtwc_cl.cpp:1725-1730`).

All four inherited bodies use formatted stream insertion at precision 15.
The stream bodies restore only precision. Locale, floatfield, `showpos`,
`showpoint`, `uppercase`, fill, and width can alter bytes or leak state. The
three file-opening consumers use Windows text mode, so literal `'\n'` becomes
CRLF on disk. Computed positive and negative infinity are emitted; NaN is the
uncomputed sentinel and becomes an empty field.

The existing real-CLI artifact proves that file bytes already disagree with
stream bytes on this host:

```text
path=build/highs-1151/f7-cli-smoke-20260723/f7_uncapped_distance_matrix.csv bytes=3177 lf=27 cr=27 crlf=27 sha256=A8B9AC33349C16AE00DCAE44F44048B258E7073798007F7CA7F30A94411EBFC8
```

Its first row terminator contains bytes `0D 0A`. Removing only the 27 carriage
returns produces 3,150 bytes and SHA-256
`2E90CE455A7F3C77998D7C821224E3E90034F098E0E437D44B795A2998384E61`;
that transform is diagnostic, not a final CLI oracle.

Existing dense adversarial tests cover ordinary round trips, empty fields,
large finite values through `1e300`, and dense file-versus-stream text
equality. They read the file in text mode and have no independent literal,
hostile locale/state, signed zero, `DBL_MAX`, infinity, mmap, visitor, print,
or real-CLI route oracle. Text-mode reading hides CRLF.

Checkpoint v2 is separate architecture and stays unchanged. It is relevant
precedent only: it preflights computed infinity, opens binary, and formats with
locale-free `to_chars(general, max_digits10)` at
`dtwc/checkpoint.cpp:471-550`.

## Independent 83-byte oracle

The permanent oracle is a hand-written ASCII literal, not a production
formatter. A 3x3 packed symmetric matrix contains:

```text
(0,0) = uncomputed NaN sentinel
(0,1) = raw bits 0x8000000000000000
(0,2) = nextafter(1.0, 2.0)
(1,1) = DBL_MAX
(1,2) = -1.25
(2,2) = +0
```

The negative zero is injected with
`std::bit_cast<double>(UINT64_C(0x8000000000000000))`, and its stored raw bits
must be asserted before formatting. The Release build uses
`-fno-signed-zeros`; this fixture pins formatter behavior only and makes no
kernel-level signed-zero claim.

Registered value bits and token lengths:

| value | binary64 bits | token | token bytes |
|---|---:|---|---:|
| `nextafter(1,2)` | `3FF0000000000001` | `1.0000000000000002` | 18 |
| `DBL_MAX` | `7FEFFFFFFFFFFFFF` | `1.7976931348623157e+308` | 23 |
| negative zero | `8000000000000000` | `-0` | 2 |
| `-1.25` | `BFF4000000000000` | `-1.25` | 5 |
| positive zero | `0000000000000000` | `0` | 1 |

Exact payload, with one byte `0A` after every displayed row:

```text
,-0,1.0000000000000002
-0,1.7976931348623157e+308,-1.25
1.0000000000000002,-1.25,0
```

Independent ASCII accounting produced:

```text
bytes=83 lf=3 cr=0 sha256=7754CFF0231360D60A69B034CA5136E88EEFE8B53A6509706D99071C436F3813
line_bytes=23,33,27 commas=6
hex=2C2D302C312E303030303030303030303030303030320A2D302C312E37393736393331333438363233313537652B3330382C2D312E32350A312E303030303030303030303030303030322C2D312E32352C300A
```

The inherited ordinary stream spelling is only 47 bytes:

```text
,-0,1
-0,1.79769313486232e+308,-1.25
1,-1.25,0
```

Its SHA-256 is
`3938EE683B38246834F256FD581DBA254FF5B89C70557525E6DA462D2E4C1478`.
It loses the adjacent-to-one bit and does not provide a safe `DBL_MAX`
round trip.

The registered native format is therefore:

- ASCII full NxN CSV, comma delimiter, no BOM;
- locale-independent general binary64 formatting at `max_digits10`;
- general-format spellings under that contract, including `0` and `-0`;
- uncomputed NaN sentinel as an empty field;
- exactly one LF after every nonempty row, including the last;
- zero bytes when N is zero;
- no CR byte.

## Hostile-state and non-finite contracts

A custom `numpunct<char>` uses decimal comma and grouping. Before insertion,
the stream is configured with scientific, `showpos`, `showpoint`, uppercase,
precision 3, fill `#`, and nonzero width. The matrix bytes must still equal the
83-byte literal. Flags, precision, locale, fill, and width must be unchanged
after insertion. File routes must also ignore a hostile global locale.

Separate fixtures place computed `+Inf` at `(0,1)` and computed `-Inf` at
`(1,2)`. Every applicable route must throw `dtwc::InvalidInput` with the exact
row-major diagnostic:

```text
distance-matrix CSV: computed non-finite value at row 0, column 1.
distance-matrix CSV: computed non-finite value at row 1, column 2.
```

Validation completes before the first matrix byte:

- stream content and stream state remain unchanged;
- a missing direct matrix path is not created;
- an existing direct matrix file retains its exact bytes;
- print capture remains empty.

NaN remains the uncomputed sentinel and is not rejected. The two infinity
signs and coordinates are independent discriminators.

## Permanent route gates

One focused Catch2 executable must run in every build, with
`SKIP_RETURN_CODE` removed:

- llfio ON: at least 80 assertions in at least 12 cases, marker
  `F14_CSV_CONTRACT dense=ran mmap=ran skips=0`;
- llfio OFF: at least 50 assertions in at least 8 cases, marker
  `F14_CSV_CONTRACT dense=ran mmap=unavailable skips=0`.

On llfio ON, the exact literal is required from the dense stream, dense file,
dense `Problem` visitor, dense print route, mmap stream, mmap `Problem`
visitor, and mmap print route. The publicly reachable dense `Result::save`
matrix file must equal the corresponding direct native stream bytes on a fully
computed non-degenerate fixture. The public `Result` factory exposes no mmap
selection seam; mmap public reachability is therefore proved through
`Problem` and the real CLI, not a private-access test hack. Empty and
non-finite cases cover both storage types.
On llfio OFF, only mmap construction is excluded; every dense assertion still
runs.

A permanent native-public-route CMake gate runs in every build against the
tracked 27-series conformance CSV and TOML. A test-owned `.cc` helper drives
the real C++ `Result::save`; it is not auto-registered as another CTest:

- resident run: explicit high mmap threshold, verbose real `dtwc_cl`, no mmap
  marker;
- llfio-ON mmap run: `--mmap-threshold 0`, exactly one
  `Using memory-mapped distance matrix:` marker;
- native C++ `Result::save` run: the same public load/cluster configuration;
- all output matrices contain 27 LF, zero CR, one final LF, and no blank
  trailing row;
- llfio ON proves all three matrix pairs byte-identical and reports 3/3 runs;
- llfio OFF proves resident CLI versus native Result byte-identical, reports
  2/2 runs and `mmap=unavailable`, and never treats a skip as execution;
- the script prints `subject=real_dtwc_cl+native_result`, exact
  run/route/pair counters, and `skips=0`.

The exact successful summary lines are:

```text
F14_CSV_PUBLIC subject=real_dtwc_cl+native_result runs=3/3 route_markers=2/2 matrix_pairs=3/3 rows=27/27/27 lf=27/27/27 cr=0/0/0 final_lf=3/3 blank_tail=0 mmap=ran skips=0
F14_CSV_PUBLIC subject=real_dtwc_cl+native_result runs=2/2 route_markers=1/1 matrix_pairs=1/1 rows=27/27 lf=27/27 cr=0/0 final_lf=2/2 blank_tail=0 mmap=unavailable skips=0
```

The gate owns only an exact child of the configured build/tests directory and
validates that path before any recursive cleanup. Tracked data are read-only.

The new tracked integration `.cmake` file changes the Git-index-owned manifest
total from 26 to 27. The same F14 test commit must update only the production
and direct-test constants, run the live checker, and prove that restoring
either constant to 26 fails.

## Registered mutations

After the repaired focused gates are green, replay all of these against the
retained test suite and restore exact committed sources after each:

1. `max_digits10` to 15/default precision: 1 execution, focused literal and
   round-trip target fails;
2. remove binary mode independently from dense file, mmap visitor file, and
   `Result::save` matrix file: 3 executions, the corresponding binary-byte
   route fails with CR on Windows;
3. replace locale-free numeric emission by formatted stream insertion in each
   dense/mmap stream body: 2 executions, the corresponding hostile-state
   route fails;
4. canonicalize negative zero to positive zero: 1 execution, raw-bit control
   stays green while both `-0` fields fail;
5. emit a token for the uncomputed NaN sentinel: 1 execution, the empty-field
   literal fails;
6. bypass the `+Inf` preflight: 1 execution, positive-infinity side-effect and
   exact-message assertions fail;
7. bypass the `-Inf` preflight: 1 execution, negative-infinity side-effect and
   exact-message assertions fail;
8. append a suffix to the non-finite diagnostic: 1 execution, whole-message
   equality fails;
9. remove the final LF, then add a second final LF: 2 executions, exact length,
   hash, and final-row assertions fail;
10. change the comma delimiter independently in each of the four formatter
    bodies: 4 executions, the corresponding exact route fails and proves every
    body reachable;
11. restore the extra newline in `Problem::print_distance_matrix`: 1
    execution, print and N=0 routes fail;
12. ignore `--mmap-threshold 0` in the real CLI route: 1 execution, mmap
    marker/cache and three-way parity gate fails;
13. restore either tracked-manifest constant from 27 to 26: 2 executions, the
    production checker or its direct Python test fails.

This is exactly 13 mutation classes and 21 executions.

Mutation class 2 is a decisive Windows gate; the permanent source audit additionally
requires binary mode in all three file-opening consumers so Unix text-mode
equivalence cannot hide its removal.

## Acceptance band

F14 passes locally only if:

1. the independent fixture reproduces the registered raw bits, 83 bytes,
   counts, hex, and SHA before judging production;
2. the inherited focused gate fails on precision, CRLF, hostile state,
   infinity, and the extra print newline before repair;
3. the final focused executable meets its build-specific assertion/case floor,
   exact marker, literal, state-restoration, round-trip, side-effect, native
   `Result::save`, dense/mmap, and empty/non-finite bands without skip;
4. the real binary plus native `Result::save` helper meet their registered
   resident/result/mmap route counters and byte comparisons; a unit helper
   alone cannot close the CLI subject;
5. every registered mutation is killed;
6. source audit finds four retained formatter bodies, no precision-15 numeric
   emission, binary mode at all three matrix-file openings, finite preflight
   before output, no extra print newline, and no F14 formatter consolidation;
7. the live supply-chain checker reports 39/39 actions, 7/7 archives, one
   Arrow pin, and 27 tracked CMake manifests;
8. fresh full gates pass at the post-F14 inventories:
   canonical 118/118 with six capability skips, llfio-OFF 118/118 with nine,
   and Arrow 120/120 with eight. Both F14 CTests execute; Arrow metadata gives
   all 120 tests the registered runtime paths while Arrow-OFF builds give none.

There are at most two implementation attempts. A missed band is recorded
**FALSIFIED** with verbatim output and is not relaxed. Rollback is a local
revert of the dedicated F14 implementation commit and its separate
bookkeeping commit.

The claim most likely to be wrong is that the inherited four-body inventory
captures every native distance-matrix CSV emitter. The post-repair call-site
audit, public `Result::save`/Problem/CLI routes, and later R4 duplication census
are the checks that can overturn it.

## Inherited decisive baseline — FALSIFIED

The decisive inherited run was made at
`509510cd401defa1da7db07e59b0a23e5384ab49`
(`docs: register F14 CSV wire contract`). A path-scoped diff over
`dtwc/core/matrix_io.hpp`, `dtwc/Problem_IO.cpp`, `dtwc/Problem.cpp`, and
`dtwc/api.cpp` produced no output before the run: production remained
digit-identical to the registered base while only the new gate was present.

Before judging production, an independent PowerShell ASCII calculation and
the Catch2 oracle-only case produced, verbatim:

```text
F14_ORACLE bytes=83 lf=3 cr=0 commas=6 line_bytes=22,32,26 sha256=7754CFF0231360D60A69B034CA5136E88EEFE8B53A6509706D99071C436F3813
hex=2C2D302C312E303030303030303030303030303030320A2D302C312E37393736393331333438363233313537652B3330382C2D312E32350A312E303030303030303030303030303030322C2D312E32352C300A
Filters: [oracle]
Randomness seeded to: 781207429
===============================================================================
All tests passed (10 assertions in 1 test case)
```

The `line_bytes` values above exclude each LF; including the registered
terminator they are 23, 33, and 27 bytes. The first attempted hash probe used
two unavailable .NET static helpers and emitted empty hash/hex fields; it is
invalid evidence. The quoted rerun used `SHA256.Create().ComputeHash()` and
explicit byte formatting.

The inherited focused command was:

```text
build/highs-1151/bin/unit_test_distance_matrix_csv.exe [f14] --reporter console --rng-seed 424242
```

Its final verdict was, verbatim:

```text
F14_CSV_CONTRACT dense=ran mmap=ran skips=0
===============================================================================
test cases:  13 |  2 passed | 11 failed
assertions: 137 | 76 passed | 61 failed
```

The eleven failing case names were:

```text
F14 dense stream emits the exact independent literal
F14 mmap Problem visitor and print route are literal-identical
F14 dense stream ignores and preserves hostile caller state
F14 mmap empty and nonfinite routes execute without partial output
F14 dense file is binary locale-free and bit-roundtrippable
F14 native Result save matches the registered dense stream bytes
F14 negative infinity rejects before dense Problem output
F14 zero-size dense routes emit zero bytes
F14 positive infinity rejects before any dense output
F14 dense Problem visitor and print route are literal-identical
F14 mmap stream is literal and hostile-state independent
```

Load-bearing failure expansions from that run were:

```text
CHECK( bytes.size() == 83 )
with expansion:
  47 == 83

CHECK( bytes.size() == 83 )
with expansion:
  50 == 83

CHECK( std::count(bytes.begin(), bytes.end(), '\r') == 0 )
with expansion:
  3 == 0

CHECK( std::count(bytes.begin(), bytes.end(), ',') == 6 )
with expansion:
  14 == 6

CHECK( output.width() == width )
with expansion:
  0 == 9

CHECK( stream_rejection.typed )
with expansion:
  false

CHECK( stream_rejection.message == kPositiveInfinityMessage )
with expansion:
  "<no exception>"
  ==
  "distance-matrix CSV: computed non-finite value at row 0, column 1."

CHECK( read_binary(existing) == "seed" )
with expansion:
  ",inf,1
  inf,1.79769313486232e+308,-1.25
  1,-1.25,0
  "
  ==
  "seed"

CHECK_FALSE( fs::exists(missing) )
with expansion:
  !true

CHECK( capture.str().empty() )
with expansion:
  false
```

Thus precision, locale/flags/fill/width, binary newlines, non-finite
preflight/side effects, and the extra print newline are independently red.
The fixture-bit/oracle case and focused route marker passed, so this is not a
skip or a degenerate-oracle failure.

The corrected native-public inherited CTest then reached the real resident
CLI artifact and failed, verbatim:

```text
CMake Error at tests/integration/test_distance_matrix_csv_contract.cmake:264 (message):
  F14 resident raw terminators lf=27 cr=27; expected lf=27 cr=0
Call Stack (most recent call first):
  tests/integration/test_distance_matrix_csv_contract.cmake:332 (inspect_matrix)

0% tests passed, 1 tests failed out of 1

The following tests FAILED:
    118 - test_distance_matrix_csv_contract (Failed)        f14 integration
```

The first public probe stopped earlier because its test-owned marker parser
expected a one-line path while the real CLI prints a multiline marker with its
inherited doubled separator. Two independent gate audits also found that the
original broad skip regex self-matched the required `skips=0` success marker,
the focused PASS regex assumed a randomized Catch2 case would run last, and
the cleanup REAL_PATH comparison resolved both sides through the same possible
reparse point. Those are harness falsifications, not production evidence.
They were corrected without changing the registered wire band or production,
and only the quoted reruns are decisive.

Verdict: **FALSIFIED [confirmed]**. The inherited native implementation misses
every registered behavioral category; implementation attempt 1 may begin.

## Separate confirmed finding: cross-language save bytes

The frozen contract says corresponding `Result::save` and CLI files are
byte-identical across C++, Python, and MATLAB
(`docs/api-contract-2.0.md:194-200,728-734`). PLAN F14 names only the four
native C++ formatter bodies.

Python instead calls `np.savetxt(..., delimiter=",")`
(`python/dtwcpp/_api.py:164-198`). Its in-memory formatting stage on the F14
3x3 values is 209 bytes with three LF, no CR, and SHA-256
`719F01F8B8E72B1D9C012F33F8D3F3C23BF435D7A8494DF7CD8A2911EEDDBE1A`;
the first row is:

```text
first_row=nan,-0.000000000000000000e+00,1.000000000000000222e+00
```

The real public `Result.save` was then run against the fresh loaded extension,
wrote all four files successfully under
`build/f37-python-save-preflight-95ffd63`, and produced:

```text
path=C:\D\git\dtw-cpp\build\f37-python-save-preflight-95ffd63\f37_distance_matrix.csv bytes=212 lf=3 cr=3 sha256=E894009679F58362E6CCBBB821C5DCB8BF2EAAC4A21E576AD77350BBA261A23F
first_line=nan,-0.000000000000000000e+00,1.000000000000000222e+00
```

Package provenance was
`C:\D\git\dtw-cpp\python\dtwcpp\__init__.py`; the native extension was
`C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd`,
SHA-256
`5139A6745C325C8614A1191605E85227564910E49D70DBF41CE2FDDDFC58BF50`.

This differs from the registered native 83-byte format in sentinel spelling,
numeric notation, line endings, and byte count. MATLAB separately calls `writematrix`
(`bindings/matlab/+dtwc/Result.m:66-100`); its exact bytes are
**[inferred]** until a fresh MEX/MATLAB execution records them.

Verdict: **F37 CONFIRMED** for the Python contradiction. F37, not F14, owns
all four cross-language result files and the fresh Python/MATLAB/native/CLI
parity matrix after F14 freezes the native format.
