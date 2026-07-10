# Mmap distance-cache creation durability (Task 8.1-M15)

Date: 2026-07-10

Platform: Windows, clang 21.1.8, Ninja, Release

Scope: `dtwc::core::MmapDistanceMatrix` creation/publication and same-path races

> Current-format note (Task 8.1-M53): this file records the historical M15
> decision when v2 was current. Format v3 now preserves the same 64-byte header,
> byte-52 publication state, and byte-64 packed-data offset, adds initial row
> digests to the first durability barrier, and explicitly rejects every v2 file.
> See `2026-07-10-mmap-payload-integrity.md` for current evidence.

## Registered failure before implementation

The finding and tests were registered before editing production code. Both
exercised the M14 mmap implementation from commit `9511efd`.

```text
build/highs-1151/bin/unit_test_mmap_distance_matrix.exe "[durability]" -s
test cases: 1 | 1 failed
assertions: 1 | 1 failed
REQUIRE_THROWS_WITH(... "initialization incomplete")
because no exception was thrown where one was expected
exit 42

build/highs-1151/bin/unit_test_mmap_distance_matrix.exe "[race]" -s
test cases: 1 | 1 failed
assertions: 1 | 1 failed
REQUIRE(successes == 1)
with expansion: 2 == 1
exit 42
```

The durability fixture is a complete, exact-length, CRC-valid v2 header with
state byte zero and a packed tail containing only zero bits. The old reader
accepted all zeros as computed distances. The race fixture releases two
threads onto the same absent path and keeps both handles alive until both
constructors return; the old `creation::if_needed` path let both win.

## Protocol and format decision

The version remains 2. M14's 64-byte header reserved bytes 52--59 and included
them in the CRC, and v2 has not been released. M15 assigns byte 52 as:

- `0`: initializing
- `1`: ready
- any other value: unsupported and rejected

Bytes 53--59 remain zero-reserved. A pre-M15 v2 file therefore decodes as
incomplete and fails with delete/rename/recompute guidance. Version 1 remains
unsupported. Header size, fingerprint offsets, packed-data offset, and file
length do not change.

Creation is now an ordered two-phase publication:

1. Atomically claim an absent path with LLFIO `creation::only_if_not_exist`.
2. Extend/map it, write a CRC-valid `initializing` header, and fill the complete
   packed region with quiet-NaN sentinels.
3. Execute a blocking `wait_all` barrier over the mapping, making data and file
   metadata durable while the only valid header still says `initializing`.
4. Change the state to `ready`, recompute the CRC, and execute a blocking
   `wait_all` barrier for the 64-byte header range only. This avoids a second
   O(N^2) data flush.

Crash-point audit:

| Crash point | Reopen result |
|---|---|
| Before/while extending or writing the first header | length, magic, or CRC rejection |
| After an initializing header appears, before its barrier | initializing rejection (ready was never written) |
| After the first barrier, before ready publication | initializing rejection; all NaNs are already durable |
| During ready/CRC publication | initializing rejection, CRC rejection, or a valid ready header whose data was durably ordered first |
| After the ready barrier | valid ready cache |

Explicit `sync()` now uses blocking `wait_data_only`; its previous `nowait`
behavior did not match the public "flush" contract.

## Green validation

LLFIO enabled (`build/highs-1151`):

```text
unit_test_mmap_distance_matrix
All tests passed (1189 assertions in 24 test cases)

ctest -R "^(unit_test_mmap_data_store|unit_test_mmap_distance_matrix|unit_test_sha256|test_metal_mmap|unit_test_cli_args|unit_test_variant_distmat)$"
100% tests passed, 0 tests failed out of 6
5 executed passes; test_metal_mmap capability-skipped on Windows
```

The repaired directed cases contribute one durability assertion and four race
assertions. The concurrent loser reports that exclusive creation failed because
the file already exists/another creator won; the winning cache reopens at
N=128 with zero computed entries.

LLFIO disabled (`build/phase8-protocol-nollfio`):

```text
focused build: exit 0
100% tests passed, 0 tests failed out of 6
3 executed passes; mmap data store, mmap distance matrix, and Metal skipped by capability
```

The host was concurrently busy with unrelated builds, so elapsed times are not
used as performance evidence. The structural range-limited second barrier is
the performance safeguard: only the first durability phase flushes O(N^2)
data; ready publication flushes the fixed 64-byte header.
