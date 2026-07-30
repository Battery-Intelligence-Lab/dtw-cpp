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
| wrong-endian count | 1 | encode intended BE `k=128` as `00 00 00 80`; inherited LE decoding observes `INT32_MIN`, so the red is safe and cannot request a huge allocation |
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
`tests/unit/unit_test_checkpoint_binary.cpp`; add no CTest target or tracked
CMake manifest. Use deterministic build-root-local scratch state rather than
running clustering to obtain a result.

The target has at least two Catch2 cases and at least 270 assertions:

- valid exact bytes, five fields, resave identity, missing path, and typed
  blocked-parent writer failure;
- all 85 corruptions, three assertions per input, sentinel transaction checks,
  and the exact observed ledger.

The sole green marker is:

```text
F51_BINARY_CHECKPOINT corpus=85 rejected=85 throws=0 unchanged=85/85 valid_bytes=72/72 fields=5/5 resave=72/72 semantic_compat=7/7 skips=0 verdict=PASS
```

No case may skip or xfail. CTest metadata must require the marker, reject actual
skip diagnostics, clear `SKIP_RETURN_CODE`, run serially, and root temporary
state under the configured build directory.

The expected-red execution occurs after the test commit and before product
work. All corruptions were selected adversarially before that execution; none
requests a large allocation from the inherited code.

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
