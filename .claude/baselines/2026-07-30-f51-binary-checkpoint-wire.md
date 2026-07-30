# F51 binary-v1 checkpoint wire canonicality - 2026-07-30

## Scope and clean base

- Base: `147585e` (`docs: register F23 Python checkpoint bindings`).
- F51 was found by two independent read-only audits during F23 preflight,
  before any F23 test or product edit.
- Subject: make the existing binary-v1 `ClusteringResult` reader/writer obey its
  documented little-endian wire contract and reject structurally noncanonical
  input without exceptions, count-derived allocation, or destination mutation.
- Product attempts: at most two. A failed product compile or decisive runtime
  gate consumes an attempt. Tests and the inherited expected red do not.
- Rollback is the eventual local F51 product commit plus its tests and
  bookkeeping in reverse order. No format version, remote, published artifact,
  data file, or operator state changes.

This is a safety prerequisite to F23, not a second owner for the later
checkpoint/config robustness lens. These 85 deterministic corruptions become
that lens's fixed seed corpus. The later lens retains randomized corruption,
fuzz crash/hang/leak execution, semantic/provenance/authentication policy,
atomicity/durability, and CLI/TOML option combinations.

## Confirmed source defect

At registration, `dtwc/checkpoint.cpp:630-678` writes the in-memory
representations of `uint16_t`, `int32_t`, and `double` directly. The public
format in `dtwc/checkpoint.hpp:71-82` says little endian, but the implementation
contains no endian conversion. The current Windows x86-64 host is little endian
and IEC-559 binary64, so its valid bytes happen to agree; that is not a
cross-host implementation.

The reader at `dtwc/checkpoint.cpp:682-753`:

- accepts signed `k` and `N` from the file and constructs
  `std::vector<int>(k)` / `std::vector<int>(N)` before proving payload length;
- ignores both reserved bytes and all three padding bytes;
- treats every nonzero convergence byte as `true`;
- accepts trailing bytes;
- decodes native integer/double representations rather than explicit LE;
- can throw `std::length_error`/allocation exceptions for malformed counts even
  though the API promises `false`;
- publishes only after payload reads, so ordinary short reads already leave the
  destination unchanged.

F17 previously recorded the same residual class and routed it to the
checkpoint/config robustness lens. Pulling this narrow canonical-wire subset
forward is justified because F23 would otherwise add a new public parser entry
point. No killed idea is reopened.

The untouched canonical target inventory and focused execution at the
registered source base were:

```text
Test  #82: unit_test_checkpoint_binary
Test #123: test_cli_resume_state
Total Tests: 2

All tests passed (9 assertions in 1 test case)

1/1 Test #82: unit_test_checkpoint_binary ......   Passed    4.58 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) =   4.70 sec
```

This is the inherited baseline only: it covers a clustering-derived round trip,
missing path, and bad magic, but none of F51's wire-canonicality classes.

The untouched real-CLI compatibility baseline also executed:

```text
F17_CLI_RESUME subject=real_dtwc_cl writer=production_serializer runs=12/12 replay_fields=10/10 markers=2/2 algorithm_skipped=1/1 fresh_discriminator=4/4 checkpoint_preserved=10/10 rejection_cases=9/9 sources_preserved=2/2 skips=0
1/1 Test #123: test_cli_resume_state ............   Passed    2.13 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) =   2.16 sec
```

Because its fixture writer round-trips every non-malformed mode through the
production reader before invoking the CLI, this executes the seven registered
structurally valid/semantically invalid parser cases as well as the valid file.

## Exact preserved valid oracle

The fixture is coherent (`labels[medoids[slot]] == slot`) and discriminates
field order, sign, integer/double byte order, padding, and convergence:

| Field | Registered value |
|---|---|
| labels | `[2, 1, 0, 2, 2, 0, 0]` |
| medoid indices | `[6, 1, 4]` |
| total cost | `-13.25` |
| iterations | `0x01020304` = `16909060` |
| converged | `true` |

Two independent read-only computations (.NET `BinaryWriter`/SHA-256 and Node
literal decode/hash) and a third Python `struct.pack` oracle agree:

```text
bytes=72
SHA256=DC832EDBD214FD847B7EC8BC57884881F1CEA196139FAC7DD5B939D5E7CD1A98
hex=44434b5001000000030000000700000004030201010000000000000000802ac006000000010000000400000002000000010000000000000002000000020000000000000000000000
```

The length is independently `32 + 4*(3+7) = 72`. Binary64 bytes at offsets
24-31 are `00 00 00 00 00 80 2a c0`. Save -> load -> save must reproduce all
72 bytes exactly and all five result fields exactly.

Version remains 1. No provenance, checksum, input identity, or producing-method
identity is added.

## Registered 85-input corruption corpus

Every case starts from the 72-byte oracle and changes exactly the named
structural property:

| Class | Cases | Exact construction |
|---|---:|---|
| truncation | 72 | every prefix length 0 through 71 |
| negative count | 2 | `k=-1`; `N=-1` |
| reserved | 2 | set byte 6 or byte 7 to 1 |
| padding | 3 | set byte 21, 22, or 23 to 1 |
| convergence | 2 | set byte 20 to 2 or 255 |
| trailing payload | 1 | append one zero byte |
| bad magic | 1 | change byte 0 from `D` to `X` |
| bad version | 1 | encode LE version 2 |
| wrong-endian count | 1 | encode intended BE `k=128` as `00 00 00 80`; inherited LE decoding observes `INT32_MIN` and synchronously fails vector length validation before reaching the allocator |
| **total** | **85** | |

For each input:

1. `load_binary_checkpoint` returns `false`;
2. it throws no exception;
3. a sentinel destination remains identical in all five fields.

The strict size formula is:

```text
expected_bytes = 32 + 4 * (k + N)
```

Both counts must first be nonnegative. Addition and multiplication are checked
in an unsigned size domain, and actual file length must equal the expected
length before either result vector allocates from `k` or `N`. Exact EOF is
required after the registered payload.

The inherited code is predicted to report:

```text
F51_RED_OBSERVATION corpus=85 false=74 accepted=8 threw=3 unchanged=77/85
```

The 72 truncations plus bad magic/version return false. Nonzero
reserved/padding, noncanonical convergence, and trailing data produce eight
false acceptances. Negative `k`, negative `N`, and the safely wrong-endian
count throw before publication. The committed test must print observed counts
before failing; disagreement changes the diagnosis, never the registered green
band.

### Independent size-before-allocation discriminator

The fixed 85-case corpus proves rejection/canonicality but does not by itself
execute-prove ordering of the size check: its positive counts are only 3/7,
while its negative counts fail `vector::max_size` validation before heap
allocation. Add one separate safe adversary, not counted among the 85:

- retain the 72-byte file but encode little-endian `k=257`;
- arm the repository's existing global `operator new` probe pattern for exactly
  `257 * sizeof(int) = 1028` bytes only around the production load;
- require false return, zero throws, unchanged destination, and zero matching
  allocations after repair.

The inherited reader is predicted to make exactly one 1028-byte allocation
before discovering the short payload. This is a bounded 1 KiB discriminator,
not a large-allocation experiment. The repaired marker adds
`size_preflight=1/1`.

## Registered implementation boundary

Use explicit helpers for LE `uint16_t`, signed `int32_t` through its bit pattern,
and IEC-559 binary64 through `std::bit_cast<uint64_t>`. The implementation must:

1. validate writer counts and every serialized `int` as int32-representable
   before any filesystem effect;
2. write the exact v1 fields in documented order with explicit zero
   reserved/padding bytes;
3. read and validate the complete 32-byte header;
4. require version 1, zero reserved/padding, and convergence in `{0,1}`;
5. reject negative counts;
6. prove exact stream/file length using checked arithmetic before allocating
   result vectors;
7. decode payload values explicitly as LE int32;
8. publish the fully local candidate only after exact payload and EOF checks;
9. return false without throwing for every malformed corpus member;
10. throw `dtwc::InvalidInput` for unrepresentable writer state before
    filesystem effects and `dtwc::IOError` for native writer filesystem
    failures.

The code must not use packed structs, depend on host struct layout, reinterpret
unaligned bytes, or duplicate a Python serializer.

## Compatibility boundary

F51 validates wire structure, not clustering meaning. F17's seven production
serializer fixtures remain structurally valid and must still parse 7/7:

```text
wrong-n
wrong-k
bad-label
bad-medoid
duplicate-medoid
negative-iterations
nonfinite-cost
```

The real CLI remains the contextual validator. Its exact 12/12 F17 marker and
diagnostics must remain unchanged. F51 must not reject those files in the
generic reader or move expected-N/k policy into the file format.

Out of scope:

- label range, medoid range/uniqueness, iteration sign, and finite cost;
- expected N/k, data/configuration/method identity, or authentication;
- version 2 or migration;
- atomic replacement, fsync, concurrent-writer semantics, symlink policy;
- dense directory checkpoint v2 and mmap cache v3;
- randomized fuzzing and CLI/TOML flag combinations.

## Permanent test and exact green marker

Rewrite the existing auto-globbed
`tests/unit/unit_test_checkpoint_binary.cpp`; add no new target, source file, or
CMake manifest. Attach the required metadata to that existing target in the
already-tracked `tests/CMakeLists.txt`. Use deterministic build-root-local
scratch state rather than running clustering to obtain a result.

The target has at least two Catch2 cases and at least 270 assertions:

- valid exact bytes, five fields, resave identity, missing path, and typed
  blocked-parent writer failure;
- all 85 corruptions, three assertions per input, sentinel transaction checks,
  and the exact observed ledger.

The sole green marker is:

```text
F51_BINARY_CHECKPOINT corpus=85 rejected=85 throws=0 unchanged=85/85 size_preflight=1/1 valid_bytes=72/72 fields=5/5 resave=72/72 semantic_compat=7/7 skips=0 verdict=PASS
```

No case may skip or xfail. CTest metadata must require the marker, reject actual
skip diagnostics, clear `SKIP_RETURN_CODE`, run serially, and root temporary
state under the configured build directory.

The expected-red execution occurs after the test commit and before product
work. All corruptions were selected adversarially before that execution.
Negative counts request impossible vector lengths but fail synchronously before
the allocator; the separate positive-count discriminator requests exactly one
bounded 1028-byte allocation from the inherited code.

## Expected-red execution

The permanent fixture was committed as `6f30665` before product work. The
canonical target rebuilt successfully, then CTest executed target #82 with the
registered build-local environment and failed closed because the exact green
marker was absent.

The observed ledgers matched the preregistration exactly:

```text
F51_RED_OBSERVATION corpus=85 false=74 accepted=8 threw=3 unchanged=77/85
F51_SIZE_PREFLIGHT_OBSERVATION false=1 threw=0 unchanged=1/1 allocations_1028=1
```

The three throws were verbatim `vector too long` for negative k, negative N,
and the wrong-endian count. The eight false acceptances were both reserved
bytes, all three padding bytes, convergence 2/255, and one trailing byte; each
published over the sentinel destination. The allocation discriminator returned
false without throwing and left the destination unchanged, but observed the
single registered 1028-byte allocation.

The independent typed-writer branch also failed as predicted:

```text
CHECK( writer_threw_io_error )
with expansion:
  false

CHECK_FALSE( writer_threw_other )
with expansion:
  !true
```

Catch2 and CTest summaries were:

```text
test cases:   2 |   0 passed |  2 failed
assertions: 298 | 276 passed | 22 failed

0% tests passed, 1 tests failed out of 1
Total Test time (real) =   0.54 sec
The following tests FAILED:
         82 - unit_test_checkpoint_binary (Failed)
```

Verdict against the registered inherited band: **FALSIFIED [confirmed]**.
Every predicted defect class executed, the allocation-order discriminator was
exact, no unregistered large allocation was attempted, and no product attempt
had begun.

## Decisive gates

F51 passes only if:

- the inherited red prints the observed ledger and fails;
- product attempt 1 or 2 prints the exact green marker at the assertion/case
  floor with zero skip;
- a second independent valid-file decoder confirms the exact hex/SHA;
- the production F17 writer parses all seven semantic-invalid fixtures and the
  real CLI gate retains its exact 12/12 marker/diagnostics;
- the canonical binary test and F17 real-CLI test execute, not merely enumerate;
- serial full matrices retain 123/123 canonical, 123/123 llfio-OFF, and 125/125
  Arrow-ON with exact 6/9/8 capability skips and the Arrow reader at 390/11;
- `checkpoint.hpp`, the rendered checkpoint guide, and Unreleased changelog say
  v1 is strict little endian while preserving its semantic/provenance limits;
- the new bug class is recorded in `LESSONS.md`;
- documentation contract, record hygiene, repository hygiene, and
  `git diff --check` all pass.

No pass band may be revised after execution. The claim most likely to be wrong
is the predicted inherited `74/8/3` split; the exact 85/85 repaired verdict,
unchanged-state requirement, and valid bytes are immutable even if the red
implementation fails in a different safe way.

## Product attempt 1 and adversarial pre-run review

The implementation was held from execution until an independent source-only
review re-read every changed line. It confirmed size-before-allocation,
transactionality, exact EOF, and semantic compatibility, then found two
pre-run gaps:

1. numeric-limits assertions alone did not reject mixed-word-order binary64;
2. accepted round trips used only `converged=true` and nonnegative payloads.

Before the decisive run, the product added the registered `-13.25` binary64
bit-pattern static assertion. The existing seven-case compatibility inventory
was strengthened without changing its marker: `wrong-n` now carries canonical
false and `bad-label` carries `-1`. That test-only strengthening is commit
`68773a7`. A second review returned `GO` and confirmed that narrow
filesystem/stream exception translation cannot remap `InvalidInput`.

Targeted format checks and the focused compile exited 0. Product attempt 1 is
commit `59e5ebc` (`fix: canonicalize binary checkpoint wire format`).

## Product attempt 1 focused execution

The canonical real executable printed:

```text
F51_RED_OBSERVATION corpus=85 false=85 accepted=0 threw=0 unchanged=85/85
F51_SIZE_PREFLIGHT_OBSERVATION false=1 threw=0 unchanged=1/1 allocations_1028=0
F51_BINARY_CHECKPOINT corpus=85 rejected=85 throws=0 unchanged=85/85 size_preflight=1/1 valid_bytes=72/72 fields=5/5 resave=72/72 semantic_compat=7/7 skips=0 verdict=PASS
===============================================================================
All tests passed (298 assertions in 2 test cases)

1/1 Test #82: unit_test_checkpoint_binary ......   Passed    0.64 sec
100% tests passed, 0 tests failed out of 1
```

Verdict: **PASS [confirmed]** against the immutable marker on product attempt
1. Attempt 2 was not consumed. Exact byte equality against all 72 registered
bytes entails the production artifact's registered SHA-256
`DC832EDBD214FD847B7EC8BC57884881F1CEA196139FAC7DD5B939D5E7CD1A98`;
the production reader independently recovered all five fields.

## Real-CLI compatibility

After rebuilding both the production fixture writer and `dtwc_cl`, test #123
printed:

```text
F17_CLI_RESUME subject=real_dtwc_cl writer=production_serializer runs=12/12 replay_fields=10/10 markers=2/2 algorithm_skipped=1/1 fresh_discriminator=4/4 checkpoint_preserved=10/10 rejection_cases=9/9 sources_preserved=2/2 skips=0
1/1 Test #123: test_cli_resume_state ............   Passed    1.80 sec
100% tests passed, 0 tests failed out of 1
```

Verdict: **PASS [confirmed]**. All seven structurally valid but semantically
invalid fixtures still reached the CLI's contextual validator.

## Serial native matrices

Every build was rebuilt and then independently settled to:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

The first canonical all-target wrapper exceeded its 120-second process
ceiling. Its result was discarded. The exact build tree had no live Ninja
process afterward; `ninja -t recompact` repaired the truncated build log, the
header-dependent rebuild completed, and only the subsequent no-work settling
run authorized CTest.

Completed serial CTest summaries were:

```text
canonical build/highs-1151:
100% tests passed, 0 tests failed out of 123
The following tests did not run:
 52 - test_cuda_correctness (Skipped)
 54 - test_cuda_lb_keogh (Skipped)
 58 - test_io_readers (Skipped)
 59 - test_metal_correctness (Skipped)
 60 - test_metal_lb_keogh (Skipped)
 61 - test_metal_mmap (Skipped)

llfio-OFF build/nollfio:
100% tests passed, 0 tests failed out of 123
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

Arrow-ON build/arrow-pyarrow-23:
100% tests passed, 0 tests failed out of 125
The following tests did not run:
 37 - unit_test_mmap_data_store (Skipped)
 38 - unit_test_mmap_distance_matrix (Skipped)
 52 - test_cuda_correctness (Skipped)
 54 - test_cuda_lb_keogh (Skipped)
 59 - test_metal_correctness (Skipped)
 60 - test_metal_lb_keogh (Skipped)
 61 - test_metal_mmap (Skipped)
 80 - unit_test_benders (Skipped)
```

These are the exact registered `123/123`, `123/123`, `125/125` inventories and
`6/9/8` capability-skip counts. The Arrow reader separately executed:

```text
All tests passed (390 assertions in 11 test cases)
1/1 Test #58: test_io_readers ..................   Passed    0.14 sec
100% tests passed, 0 tests failed out of 1
```

## Documentation, hygiene, and final verdict

The commands printed:

```text
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

Targeted formatting, `git diff --check`, and the clean-worktree check also
passed. `checkpoint.hpp`, the rendered guide, Unreleased changelog, and
`LESSONS.md` now describe strict little endian and preserve the
semantic/provenance boundary.

Final F51 verdict: **PASS [confirmed]**. Every registered decisive band is
green, the invalid corpus becomes fixed seed material for the later randomized
robustness lens, and F23 may now expose the native codec.
