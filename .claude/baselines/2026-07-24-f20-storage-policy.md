# F20 Problem series-storage routing - 2026-07-24

## Scope and base

- Base: `253dfda8e5d150492261582673b47f7859c7d78e`
  (`docs: close F19 Problem encapsulation`).
- `git status --short` produced no output.
- Subject: make the existing cross-language
  `Problem::set_storage_policy(core::StoragePolicy)` setting govern the next
  owning `Problem::set_data(Data)` operation.
- Out of scope: mmap distance-matrix selection, CLI `--mmap-threshold`, CLI
  `--ram-limit`, direct `.dtws`/Arrow zero-copy ingestion, F26's Python
  `set_view_data` aliasing gap, and a new series-store backend or file format.
- Product attempts: at most two. The red-first inherited run and deliberate
  mutations are tests, not product attempts. A failed product compile or
  runtime gate consumes an attempt; there is no rescue-tuning after attempt 2.

The killed-ideas section, plan archive, and `LESSONS.md` were searched before
choosing the route. No F20-specific idea was killed. The still-binding
rejections forbid conflating series storage with distance-matrix storage,
replacing llfio without new backend gates, or treating old mmap timings as
evidence for a default policy.

## Historical and live-path audit

Task 1.4 was closed by `4778377` after testing
`DataLoader::load_stored()`. The permanent cases at
`tests/unit/test_storage_policy.cpp` configure the loader directly; they never
call `Problem::set_storage_policy`. The later setter only validates and stores
the enum. This is the already-recorded dead-sibling/live-path bug class.

The storage namespaces remain separate:

| Control | Governed storage |
|---|---|
| `Problem::set_storage_policy` | local time-series backing on owning `set_data` |
| `DataLoader::storage_policy` | `DataLoader::load_stored()` result backing |
| CLI `--mmap-threshold` | pairwise distance-matrix backing |
| CLI `--ram-limit` | Parquet decode/materialisation cap |

`LoadedData` is load-bearing: on the mmap route its `Data` contains non-owning
series and name views, while the bundle owns the `MmapDataStore` and name
strings. Moving only `loaded.data` into `Problem` would dangle. The repair must
retain the whole backing lifetime and preserve normal `Problem` copyability.

## Registered semantic decision

The governed operation is the existing owning
`Problem::set_data(Data)` call. This is the only candidate already shared by
C++, Python, and MATLAB after their exposed Problem policy setter/property.
Adding a loader-only overload would leave both bindings advisory and therefore
cannot close F20.

The exact contract is:

1. `set_storage_policy` validates and records policy; it does not retroactively
   migrate existing data or touch a distance matrix.
2. The next owning `set_data(Data)` applies that recorded policy before
   publishing the candidate:
   - `Heap` retains owning heap storage;
   - `Mmap` stores a resident Float64 payload in `MmapDataStore`, and Problem
     owns the store and names for as long as any copied Problem needs its views;
   - `Auto` uses the existing series-storage threshold rule. The Windows
     default-RAM query is unavailable and therefore is not claimed as a
     validated automatic spill; deterministic threshold behavior is covered
     through the existing `DataLoader::ram_limit(1)` case and the loader
     constructor integration below.
3. `set_view_data(Data)` is an explicit caller-owned zero-copy override and is
   outside policy routing. It must retain pointer identity and must not create a
   second mmap store.
4. `Problem(name, loader)` honors the loader's existing policy and adopts the
   complete `LoadedData` bundle. An injected one-byte loader threshold supplies
   the deterministic Auto integration; no new public overload is introduced.
5. An explicit `Mmap` request without llfio fails loudly before data
   publication. It never reports success with heap backing. Auto selecting
   Mmap without llfio retains the existing loud-warning Heap behavior.
6. `Mmap` for Float32 is unsupported by the existing Float64 `.dtws` v1 store:
   explicit Mmap fails loudly, while Auto warns and retains Float32 Heap.
   Extending the store format is not F20.
7. Policy routing occurs only after the existing precision, shape, and
   distance-semantics preflight, so invalid candidates create no store and
   leave prior Problem data/cache state unchanged.
8. CUDA and Metal must not consume empty owning vectors from mapped Data.
   Until their upload APIs accept spans, the mapped-series cross-product fails
   loudly before backend compute. The CLI explicitly keeps series on Heap
   because it exposes no series-policy option; its two RAM flags retain their
   already-bound meanings.

Direct `.dtws`/Arrow CLI zero-copy is an adjacent gap and is not claimed fixed.

## Confirmed inherited baseline

Canonical rebuild:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

Canonical focused storage test:

```text
All tests passed (8320 assertions in 4 test cases)

1/1 Test #65: test_storage_policy ..............   Passed    0.21 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) =   0.23 sec
```

llfio-OFF rebuild:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

llfio-OFF focused storage test:

```text
test cases:  4 |  2 passed | 2 skipped
assertions: 14 | 14 passed

1/1 Test #65: test_storage_policy ..............   Passed    0.07 sec
100% tests passed, 0 tests failed out of 1
Total Test time (real) =   0.08 sec
```

Those four cases prove DataLoader storage and mapped distance storage only.
They are not evidence that the Problem setter ran.

## Registered non-degenerate fixture and bands

Use the existing CSV fixture
`data/test/nonUnimodular_1_Nc_2.csv`: six Float64 series of six values.

| Quantity | Registered value/band |
|---|---:|
| payload values | exactly 36 |
| payload footprint | exactly 288 bytes |
| deterministic threshold | 1 byte; `288 > 1` |
| series and names compared | 6/6 |
| payload equality | all 36 doubles byte-identical |
| downstream pairwise distances | all 36 ordered pairs exactly equal |
| policy transitions | Heap remains heap; Mmap becomes view-backed only on the subsequent `set_data` |
| loader constructor | Heap is owning; Auto above 1 byte is mapped with llfio |
| copied-Problem lifetime | mapped copy remains readable after the source adopts different data |
| view override | all six series retain caller pointer identity |
| subject skips | exactly zero in both llfio-ON and llfio-OFF builds |

This real fixture is non-uniform and non-zero. Exact byte comparison catches
storage corruption; the full ordered distance matrix catches a view that is
alive but semantically wrong.

## Registered gates

### S1 - red-first live Problem route

Extend `test_storage_policy` with one non-skipping F20 case that drives the real
Problem setter followed by real owning `set_data`. On the inherited tree both
forced policies retain heap backing, so the registered Mmap discriminator must
fail. The failure is evidence only after the fixture prints its exact footprint
and both observed backing modes.

The repaired llfio-ON marker must report:

```text
F20_PROBLEM_STORAGE_POLICY footprint=288 threshold=1 heap=owning mmap=view values=36 names=6 ordered_pairs=36 copy_lifetime=pass view_override=pass subject_skips=0 verdict=PASS
```

The repaired llfio-OFF branch must execute Heap plus the explicit-Mmap error
path, preserve the prior series bytes, print `subject_skips=0`, and never call
Catch2 `SKIP` for the F20 case.

### S2 - transaction and unsupported-precision checks

- invalid selector behavior already owned by M47 remains unchanged;
- invalid candidate preflight precedes any mapped-file creation;
- explicit Float32+Mmap is rejected before replacing prior data;
- Auto Float32 above the injected threshold warns and remains exact on Heap;
- explicit Mmap without llfio is rejected before replacing prior data;
- Heap Float32 remains resident and exact.

### S3 - compatibility and real bindings

- existing DataLoader storage tests retain digit-identical results;
- `set_view_data` pointer identity remains exact under both Problem policies;
- a copied mapped Problem retains readable names/series after the source
  releases its ownership;
- loader construction exercises its real injected-threshold route;
- mapped CUDA/Metal selection rejects loudly before an empty-vector backend
  call, while existing Heap backend behavior remains unchanged;
- fresh Python and MEX artifacts drive their real
  `set_storage_policy` -> `set_data` sequence. On an llfio-OFF artifact the
  explicit Mmap sequence must raise; on llfio-ON it must complete and its
  downstream distance/result oracle must remain exact.

### S4 - build matrix

- focused canonical and llfio-OFF binaries must contain the F20 marker and
  execute the subject with zero subject skips;
- canonical full gate: at least the inherited 121 tests, zero failures, exact
  six documented capability skips;
- llfio-OFF full gate: at least 121 tests, zero failures, exact documented
  capability skips;
- fresh Python extension and both installed MATLAB releases must pass their
  existing full floors, subject to already-open independently named failures
  being recorded rather than hidden;
- generated documentation, live-CLI contract, and record hygiene must pass.

## Mutation set

The retained implementation must reject at least these eleven changes:

1. restore the advisory-only setter;
2. ignore `Heap`;
3. ignore `Mmap`;
4. publish only `LoadedData::data` and drop its owner;
5. drop mapped name ownership;
6. make mapped ownership unique to the source Problem so a copy dangles;
7. route `set_view_data` through a copy/store;
8. silently heap-fallback for explicit Mmap without llfio;
9. silently heap-fallback or reinterpret Float32 for explicit Mmap;
10. leave `Problem(name, loader)` on heap-only `loader.load()`;
11. pass mapped Data's empty `p_vec` into a GPU backend.

## Numbers ledger

| Quantity | Independent source | Current value |
|---|---|---:|
| fixture values | CSV dimensions and existing test comment | 36 |
| fixture bytes | `36 * sizeof(double)` | 288 |
| inherited canonical assertions/cases | Catch2 focused run | 8320 / 4 |
| inherited llfio-OFF assertions/cases/skips | Catch2 focused run | 14 / 4 / 2 |
| F20 product attempts consumed | none | 0 / 2 |

## Current verdict

`REGISTERED`; product code is untouched. The claim most likely to be wrong is
that direct owning `set_data` is the intended boundary rather than a future
loader-mediated Problem API. The frozen cross-language surface supports the
former; an explicit superseding contract decision would be required to choose
the latter.
