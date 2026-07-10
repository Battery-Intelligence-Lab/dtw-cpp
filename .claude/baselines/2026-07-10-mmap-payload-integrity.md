# Mmap mutable-payload integrity (Task 8.1-M53)

Date: 2026-07-10

Platform: Windows 10, clang 21.1.8, Ninja, Release

Scope: `dtwc::core::MmapDistanceMatrix` persisted values, computed-state bits,
row digests, session isolation, and format compatibility

## Registration and red evidence

The defect was registered in PLAN before production work (`719d2cc`). The
finite-bit, NaN/status, lazy/full `Problem`, durability, parallelism, header,
and no-exposure tests were then committed independently (`940396e`). The live
session lease was also preregistered independently (`db00be9`, with portable
file-snapshot correction `2650657`) and move coverage followed in `3450b91`.

The exact live-session red against the pre-lease implementation was:

```text
build/highs-1151/bin/unit_test_mmap_distance_matrix.exe "[m53][lease]" \
  --order lex --rng-seed 1 --reporter console

test cases: 1 | 1 failed
assertions: 7 | 4 passed | 3 failed
```

The competing live open returned successfully; the failed assertions required
non-return, a typed `std::runtime_error`, and the actionable phrase
`exclusive session lease`. The payload tests were source-preregistered before
the implementation; no aggregate pre-fix payload count is claimed from the
shared M47 build boundary.

## Format and algorithm decision

Format v3 keeps the M14/M15 64-byte header and byte-64 packed-double offset:

- byte 52 remains the CRC-covered `initializing`/`ready` publication state;
- bytes 53, 54, and 55 declare payload-integrity algorithm 1, two digest lanes,
  and eight-byte digest words;
- bytes 56--59 remain zero-reserved and header CRC32 still covers bytes 0--59;
- the packed triangular doubles remain byte-for-byte at offset 64;
- an aligned footer stores two `uint64_t` digest words per logical row.

Every cell contributes two domain-separated, avalanche-mixed 64-bit values
keyed by logical row, packed index, and exact IEEE-754 bits. A supported
`set(old -> new)` XORs out the old contributions and XORs in the new ones.
The two row words use lock-free `std::atomic_ref<uint64_t>::fetch_xor`, so a
disjoint write performs constant work, has no global mutex, and remains safe
when different cells share one row. Same-cell concurrent writes remain outside
the class contract. All region counts, offsets, and lengths are overflow
checked; compile-time and run-time checks require eight-byte alignment and
lock-free 64-bit atomics.

This is accidental-corruption detection, not keyed cryptographic
authentication. Once data and both lanes are stable, an undetected accidental
change must collide in both domain-separated lanes. During a crash window in
which only one lane delta persists, the supported claim is at least 64-bit
detection, not unconditional 128-bit strength. A crash may also preserve the
entire old data/digest state, which is a lost unsynchronized update rather than
a falsely authenticated new update; `sync()` is the durability boundary.

Reopen takes a nonblocking exclusive LLFIO whole-file lease before inspecting
the mapping. Validation recomputes every row into local words and returns no
matrix until every stored word agrees. Failure is read-only and leaves the file
byte-for-byte unchanged. The lease is owned by the native mapped-file handle,
not by an RAII guard containing a member address: move construction and move
assignment transfer the locked handle, and mapped-file teardown unmaps before
closing the handle releases the OS lock. This prevents cooperating
`MmapDistanceMatrix` objects/processes from validating while a live writer is
updating. Filesystems that cannot provide LLFIO whole-file locking fail loudly;
external writers that ignore advisory locks are outside the session contract.

Mutable `raw()` is retained for API compatibility and inspection. Writing
through it bypasses `set()` and therefore deliberately causes payload-integrity
rejection on the next reopen.

## Publication and crash audit

Creation now initializes canonical NaNs and both row-digest lanes while the
header says `initializing`. The first `wait_all` barrier covers header, payload,
footer, and metadata. Only after that barrier does creation publish and flush
the 64-byte `ready` header.

| Persisted state at reopen | Result |
|---|---|
| short/bad/CRC-invalid header | existing typed header rejection |
| valid `initializing` header | incomplete-initialization rejection |
| v1 or v2 header | explicit legacy/unauthenticated-version rejection |
| ready header, payload/footer disagreement | payload-integrity rejection before exposure |
| ready header, matching payload/footer | valid warm start |

The footer makes v2 length-incompatible by design, but version dispatch rejects
v2 first with specific guidance: it authenticates header and data/config
identity but not mutable packed distances, so it must be recomputed as v3.

## Green validation

LLFIO enabled (`build/highs-1151`, Clang Release):

```powershell
cmake --build build/highs-1151 --target unit_test_mmap_distance_matrix --parallel 1
build/highs-1151/bin/unit_test_mmap_distance_matrix.exe "[m53]" --order lex --rng-seed 1 --reporter console
build/highs-1151/bin/unit_test_mmap_distance_matrix.exe --order lex --rng-seed 1 --reporter console
```

```text
M53 focused:                 13 cases, 154/154 assertions
full mmap matrix suite:      37 cases, 1,343/1,343 assertions
```

The final focused set includes finite mantissa corruption, both one-bit
computed-status directions, footer-only corruption, repaired-CRC fingerprint
corruption, v2 rejection, partial lazy and full parallel `Problem` warm starts,
explicit sync, normal destructor persistence, four-thread disjoint fill,
constant-time source guard, live-session rejection, and move construction plus
move assignment.

Problem identity and semantic regressions against the same implementation:

```powershell
cmake --build build/highs-1151 --target unit_test_variant_distmat unit_test_dtw_function_semantics unit_test_problem_semantic_transactions --parallel 1
build/highs-1151/bin/unit_test_variant_distmat.exe --order lex --rng-seed 1 --reporter console
build/highs-1151/bin/unit_test_dtw_function_semantics.exe --order lex --rng-seed 1 --reporter console
build/highs-1151/bin/unit_test_problem_semantic_transactions.exe --order lex --rng-seed 1 --reporter console
```

```text
variant/distmat identity:       13 cases, 97/97 assertions
dispatcher semantics:           5 cases, 2,068/2,068 assertions
semantic transactions:          9 cases, 242/242 assertions
```

LLFIO disabled (`build/cfg-normal`, a separate Clang Release configure whose
summary printed `llfio: OFF (DTWC_ENABLE_LLFIO=OFF)`):

```powershell
cmake --build build/cfg-normal --target unit_test_mmap_distance_matrix unit_test_variant_distmat unit_test_dtw_function_semantics unit_test_problem_semantic_transactions --parallel 1
build/cfg-normal/bin/unit_test_mmap_distance_matrix.exe --order lex --rng-seed 1 --reporter console
build/cfg-normal/bin/unit_test_variant_distmat.exe --order lex --rng-seed 1 --reporter console
build/cfg-normal/bin/unit_test_dtw_function_semantics.exe --order lex --rng-seed 1 --reporter console
build/cfg-normal/bin/unit_test_problem_semantic_transactions.exe --order lex --rng-seed 1 --reporter console
cmake --build build/cfg-normal --target test_storage_policy --parallel 1
build/cfg-normal/bin/test_storage_policy.exe --order lex --rng-seed 1 --reporter console
```

```text
mmap matrix suite:            1 expected capability skip (Catch exit 4)
variant/distmat identity:     61/61 assertions; 3 passed, 10 mmap skips
dispatcher semantics:        2,058/2,058 assertions; 4 passed, 1 mmap skip
semantic transactions:       198/198 assertions; 7 passed, 2 mmap skips
storage policy:              14/14 assertions; 2 passed, 2 mmap skips
```

The storage-policy gate includes the explicit LLFIO-off path: requesting mapped
distance storage throws the documented rebuild/capability error instead of
silently allocating a dense heap matrix.

Two independent compile-only checks also passed: inclusion and move traits with
LLFIO absent, and LLFIO-enabled inclusion with nothrow move construction and
assignment. The latter emitted only LLFIO's existing header-only ntkernel error
category warning.

## Commits

- `719d2cc` -- register M53 in PLAN
- `940396e` -- preregister mutable-payload tests
- `db00be9`, `2650657`, `3450b91` -- preregister lease and move behavior
- `9becd53` -- implement v3 payload integrity and native-handle lease
- `5658a2f` -- add direct digest/v2 checks and repair the v3 overflow fixture
