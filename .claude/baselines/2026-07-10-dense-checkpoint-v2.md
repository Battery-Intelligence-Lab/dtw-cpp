# Dense checkpoint identity and transactional publication (Task 8.1-M49)

Date: 2026-07-10

Platform: Windows 10, Clang 21.1.8 with the MSVC runtime, Ninja, Release;
LLFIO-enabled and LLFIO-disabled configurations

Scope: dense CSV checkpoint data/configuration identity, byte integrity, strict
parsing, strong load transaction, generation publication, and legacy handling

## Registration and red evidence

The defect was registered in PLAN before production work. The committed
`unit_test_checkpoint_robustness.cpp` contract covered bit-exact full, partial,
and zero-work round trips; every M14 identity axis; strict metadata and CSV;
cross-generation tears; unverifiable legacy directories; invalid-save
filesystem effects; and failed overwrite preservation.

Against the legacy direct-file implementation, all eight cases failed:

```text
test cases:   8 | 0 passed | 8 failed
assertions: 412 | 294 passed | 118 failed
```

The failures included successful publication of wrong-data/wrong-configuration
distances, mutation on malformed input, `std::stoull` exceptions escaping the
boolean load API, acceptance of malformed/asymmetric/non-finite CSV, absent
identity and integrity fields, filesystem effects from invalid saves, and no
recoverable generation boundary.

## Format and identity decision

The checkpoint root contains a strict selector and immutable generations:

```text
CURRENT                         # exactly 64 lowercase hex characters plus LF
generations/<id>/metadata.txt   # exact seven-key v2 manifest
generations/<id>/distances.csv  # exact full N-by-N payload
```

The manifest keys are exactly `format`, `version`, `n`, `pairs_computed`,
`timestamp`, `identity_sha256`, and `payload_sha256`. Unknown, duplicate,
missing, empty, partial, overflowing, or noncanonical numeric fields fail.
Timestamps use `YYYY-MM-DDTHH:MM:SSZ` and validate the Gregorian month/day and
leap-year calendar, not only string shape.

`identity_sha256` is M14's full L1 distance-cache identity. It covers series
order, flat lengths, every IEEE value bit, dimensions, stored precision, band,
variant selector, every active and inactive variant parameter, multivariate
mode, missing strategy, distance backend, CUDA device, and CUDA precision.
Names and clustering outputs are excluded because they cannot affect a stored
distance.

`payload_sha256` hashes the exact CSV bytes after the domain
`dtwc-dense-checkpoint-payload-v2`. Save uses locale-independent `to_chars`
with `max_digits10`; load uses full-token `from_chars`. Every payload has
exactly N rows and N fields. Empty is the only uncomputed representation;
numeric tokens must be finite; both symmetric cells must have identical
computed status and IEEE-754 bits; and the reconstructed packed count must
equal `pairs_computed`. File, row, token, square, and packed sizes are checked
before allocation or parsing.

## Transaction and publication decision

Load computes the expected Problem identity and captures the already-proven
Dense alternative before reading checkpoint metadata. It parses into a local
`DenseDistanceMatrix`; every error is caught and returns `false` with the
matrix allocation address, bits, Problem configuration, data, labels, and
medoids unchanged. The final operation is a move assignment guarded by
`is_nothrow_move_assignable_v<DenseDistanceMatrix>`; no logging, accessor, or
other potentially throwing work follows publication.

Save completes semantic/snapshot, logical dimension, packed-size, finite-value,
identity, and row-bound preflight before its first filesystem effect. It then
streams one bounded row at a time into a newly and exclusively created 64-hex
generation while updating SHA-256, writes metadata after the payload closes,
and publishes only by replacing same-directory `CURRENT`. Windows uses
`MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`; POSIX uses
same-filesystem `rename`. RAII removes an unpublished generation and temporary
selector after write/allocation/publication failure. A new identifier is used
for every save, so an active generation is never overwritten and older
generations remain available. Direct legacy files are ignored by load; a valid
save into that directory creates v2 generations and `CURRENT` without trusting
the old payload.

## Durability boundary

Payload, manifest, and temporary selector streams are explicitly closed before
publication, which flushes C++ stream buffers to the operating system. The
implementation does **not** call `FlushFileBuffers` for payload/manifest files,
`fsync` for POSIX files, or fsync either generation/root directory. Windows
requests write-through for the `CURRENT` replacement; the POSIX rename is not
followed by directory fsync.

Therefore the supported claim is atomic selection during normal OS operation
and fail-closed safety, not fully durable power-loss recovery. If a crash or
power loss exposes a missing, short, or torn selected generation, exact
shape/hash/identity validation rejects it before publication. Previous
generations are not overwritten, but load does not scan them automatically, so
resume availability can be lost even though wrong distances cannot be trusted.

## Green validation

Production is `9c2b1c8`; legacy-suite migration and strict `CURRENT`
corruption/traversal tests are `742e770`. The original eight-case focused gate
passes 541/541 assertions after production. With empty/missing-newline,
extra-newline, traversal, uppercase, non-hex, nonexistent-generation, and
non-regular `CURRENT` cases added, the final focused gates are:

```text
LLFIO OFF: 585/585 assertions in 9 cases
LLFIO ON:  587/587 assertions in 9 cases
```

The two extra LLFIO-on assertions exercise rejection of dense-format save from
a mapped Problem.

LLFIO disabled (`build/cfg-normal`, configuration summary reported
`DTWC_ENABLE_LLFIO=OFF`):

```powershell
cmake --build build/cfg-normal --target unit_test_checkpoint unit_test_checkpoint_robustness unit_test_checkpoint_binary unit_test_mmap_distance_matrix unit_test_problem_semantic_transactions unit_test_invalid_distance_enums unit_test_variant_distmat unit_test_dtw_function_semantics --config Release -j 2
build/cfg-normal/bin/<target>.exe --order lex --rng-seed 1
```

```text
dense checkpoint compatibility: 115/115 assertions, 8 cases
dense checkpoint robustness:    585/585 assertions, 9 cases
binary checkpoint isolation:       9/9 assertions, 1 case
mmap matrix:                       1 expected capability skip
semantic transactions:          198/198 assertions, 7 pass + 2 mmap skips
invalid distance selectors:   1,533/1,533 assertions, 9 cases
variant/cache identity:           61/61 assertions, 3 pass + 10 mmap skips
dispatcher semantics:        2,058/2,058 assertions, 4 pass + 1 mmap skip
```

LLFIO enabled (`build/cfg-gate-normal`, configuration cache reported
`DTWC_ENABLE_LLFIO=ON`):

```powershell
cmake --build build/cfg-gate-normal --target unit_test_checkpoint unit_test_checkpoint_robustness unit_test_checkpoint_binary unit_test_mmap_distance_matrix unit_test_problem_semantic_transactions unit_test_invalid_distance_enums unit_test_variant_distmat unit_test_dtw_function_semantics --config Release -j 2
build/cfg-gate-normal/bin/<target>.exe --order lex --rng-seed 1
```

```text
dense checkpoint compatibility: 115/115 assertions, 8 cases
dense checkpoint robustness:    587/587 assertions, 9 cases
binary checkpoint isolation:       9/9 assertions, 1 case
mmap payload/lease/integrity: 1,343/1,343 assertions, 37 cases
semantic transactions:          242/242 assertions, 9 cases
invalid distance selectors:   1,533/1,533 assertions, 9 cases
variant/cache identity:           97/97 assertions, 13 cases
dispatcher semantics:        2,068/2,068 assertions, 5 cases
```

## Commits

- `554b19e` -- preregister dense checkpoint identity/integrity transaction
- `9c2b1c8` -- implement authenticated v2 generations and no-throw publication
- `742e770` -- migrate compatibility coverage and pin strict `CURRENT`
