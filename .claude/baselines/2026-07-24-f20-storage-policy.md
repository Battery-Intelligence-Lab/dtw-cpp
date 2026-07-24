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

## Registration audit correction

Commit `0c6fe79` preregistered the governed operation before the independent
test audit completed. That audit then falsified three gate assumptions before
any product or test edit:

1. Problem is already non-copyable through `MmapDistanceMatrix`, so the lifetime
   gate is move-and-source-destruction, not a new copy contract.
2. Direct `set_data(Data)` has no public threshold injection. Its discriminator
   uses explicit Heap/Mmap; the one-byte threshold belongs only to the real
   DataLoader/Problem-constructor integration.
3. Native `ndim=1` cannot catch shape loss. The direct candidate reuses the
   exact 288 payload bytes with `ndim=2` and adds an independent full-matrix DP
   oracle.

This correction consumes no product attempt. The dedicated target and
build-root temporary environment also replace the first draft's weaker plan to
append a case to a suite containing unrelated capability skips.

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
retain the whole backing lifetime across normal Problem moves. Problem is
already non-copyable because `distMat_t` contains the explicitly non-copyable
`MmapDistanceMatrix`; F20 does not add a new copy contract.

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
     owns the store and names after moves and source destruction;
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
   explicit Mmap fails loudly. Auto's Float32 spill behavior is specified as a
   loud warning plus Float32 Heap, but the Windows default threshold cannot
   drive that branch and it is not a decisive local claim. Extending the store
   format is not F20.
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
`data/test/nonUnimodular_1_Nc_2.csv`: six Float64 series of six values. The
direct Problem discriminator reconstructs those same 288 payload bytes with
`ndim=2`, so a route that loses multivariate shape cannot pass. The loader
constructor integration separately reads the fixture at its native `ndim=1`.

| Quantity | Registered value/band |
|---|---:|
| payload values | exactly 36 |
| payload footprint | exactly 288 bytes |
| direct Problem policies | explicit Heap and explicit Mmap; no threshold claim |
| loader constructor threshold | 1 byte; `288 > 1` |
| series and names compared | 6/6 |
| direct Problem ndim | exactly 2 on both routes |
| payload equality | all 36 doubles byte-identical |
| payload little-endian-double SHA-256 | `1c1e8815b6e7a4a8b4ccd3d9892132a65e110ee0ecddcb9828fa817faa78f840` |
| independent DP oracle | 15 upper-triangle values, little-endian-double SHA-256 `d8331a04d296db5bf2c791359df8d87e6d841e9564636e245b36a90979ce7db9` |
| downstream pairwise distances | all 36 ordered pairs exact; 15 nontrivial pairs equal the independent DP |
| mapped file | exactly one `.dtws`, 408 bytes, `DTWS` v1, elem-size 8, N=6, ndim=2, header CRC `0xab81a0e4`, offsets `0,48,96,144,192,240,288` |
| policy transitions | Heap remains heap; Mmap becomes view-backed only on the subsequent `set_data` |
| loader constructor | Heap is owning; Auto above 1 byte is mapped with llfio |
| moved-Problem lifetime | mapped destination remains readable after source/candidate/loader destruction |
| view override | all six series retain caller pointer identity |
| subject skips | exactly zero in both llfio-ON and llfio-OFF builds |

This real fixture is non-uniform and non-zero. Exact byte comparison catches
storage corruption; the full ordered distance matrix catches a view that is
alive but semantically wrong.

## Registered gates

### S1 - red-first live Problem route

Add a dedicated auto-globbed
`tests/unit/core/unit_test_problem_storage_policy.cpp` target. It drives the
real Problem setter followed by real owning `set_data`; it never calls Catch2
`SKIP`. On the inherited tree both forced policies retain heap backing, so the
registered Mmap discriminator must fail. The failure is evidence only after the
fixture prints its exact footprint and both observed backing modes.

CTest clears `SKIP_RETURN_CODE`, rejects skip text, runs the target serially,
and sets `TMP`, `TEMP`, and `TMPDIR` to a unique directory under the configured
build root. Thus the decisive run writes nowhere outside the project and can
inspect the exact generated mapping. Require at least 180 assertions / 4 cases
with llfio and at least 90 assertions / 4 cases without it.

The repaired llfio-ON marker must report:

```text
F20_PROBLEM_STORAGE_POLICY build=llfio-on footprint=288 heap=owning mmap=view values=72/72 names=12/12 ndim_routes=2/2 ordered_pairs=72/72 artifact=pass lifetime=pass loader_auto=mmap view_override=pass subject_skips=0 verdict=PASS
```

The repaired llfio-OFF branch must execute Heap plus the explicit-Mmap error
path, preserve the prior series bytes, print `subject_skips=0`, and never call
Catch2 `SKIP` for the F20 case.

```text
F20_PROBLEM_STORAGE_POLICY build=llfio-off footprint=288 heap=owning mmap=rejected values=36/36 names=6/6 ndim_routes=1/1 ordered_pairs=36/36 transaction=pass loader_auto=heap-warning view_override=pass subject_skips=0 verdict=PASS
```

### S2 - transaction and unsupported-precision checks

- invalid selector behavior already owned by M47 remains unchanged;
- invalid candidate preflight precedes any mapped-file creation;
- explicit Float32+Mmap raises `InvalidInput` before replacing prior data:
  `Problem::set_data: StoragePolicy::Mmap supports Float64 series only; Float32 mmap requires a new .dtws format version.`;
- explicit Mmap without llfio raises `IOError` before replacing prior data:
  `Problem::set_data: StoragePolicy::Mmap requested but mmap support (llfio) is not compiled in. Rebuild with -DDTWC_ENABLE_LLFIO=ON.`;
- Heap Float32 remains resident and exact.

### S3 - compatibility and real bindings

- existing DataLoader storage tests retain digit-identical results;
- `set_view_data` pointer identity remains exact under both Problem policies;
- a moved mapped Problem retains readable names/series after the source,
  candidate, and loader lifetimes end;
- loader construction exercises its real injected-threshold route;
- mapped CUDA/Metal selection rejects loudly before an empty-vector backend
  call, while existing Heap backend behavior remains unchanged;
- six fresh zero-skip binding profiles drive their real
  `set_storage_policy` -> `set_data` sequence: Python llfio-ON/OFF plus MATLAB
  llfio-ON/OFF under both R2024b and R2025b. Their exact build routes are
  `build/cfg-gate-normal`, `build/phase8-m13`, `build/mex-verify`, and
  `build/f19-mex-profile-build`;
- each profile sets `TMP`/`TEMP`/`TMPDIR` (and MATLAB `MATLAB_PREFDIR`) to a
  fresh build-root directory. No new public capability/backing getter is
  needed: ON observes zero Heap artifacts then exactly one independently
  parsed Mmap artifact; OFF observes typed rejection, zero artifacts, and
  unchanged sentinel Problem state;
- ON profiles compare all 36 ordered distances and FastPAM Heap/Mmap outputs;
  OFF profiles require Python `IOError` / MATLAB `dtwc:ioError` plus the exact
  registered message.

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
6. fail to transfer mapped ownership when Problem moves;
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
| direct fixture ndim | six scalars / two features | 2 |
| mapped file bytes | `64 + 7*8 + 288` | 408 |
| mapped header CRC | independent header parser | `0xab81a0e4` |
| payload hash | independent byte reader | `1c1e8815b6e7a4a8b4ccd3d9892132a65e110ee0ecddcb9828fa817faa78f840` |
| independent DP pairs | `6*5/2` | 15 |
| independent DP hash | inline full-matrix recurrence | `d8331a04d296db5bf2c791359df8d87e6d841e9564636e245b36a90979ce7db9` |
| inherited canonical assertions/cases | Catch2 focused run | 8320 / 4 |
| inherited llfio-OFF assertions/cases/skips | Catch2 focused run | 14 / 4 / 2 |
| F20 product attempts consumed | none | 0 / 2 |

## Inherited expected-red gate

The dedicated gate was committed as `e4688a7` while product code remained
unchanged. Its first execution exposed a gate-only trailing-separator path
comparison; changing that assertion to `filesystem::equivalent` was not a
product attempt. Both corrected inherited runs then executed 404 assertions in
four cases with 401 passing and exactly the three registered product failures.

Canonical llfio-ON evidence:

```text
F20_RED_OBSERVATION footprint=288 heap=owning mmap=owning
C:/D/git/dtw-cpp/tests/unit/unit_test_problem_storage_policy.cpp(391): FAILED:
  REQUIRE( source->data().is_view() )
with expansion:
  false
C:/D/git/dtw-cpp/tests/unit/unit_test_problem_storage_policy.cpp(551): FAILED:
  REQUIRE( heap.storage_policy() == dtwc::core::StoragePolicy::Heap )
with expansion:
  0 == 1
C:/D/git/dtw-cpp/tests/unit/unit_test_problem_storage_policy.cpp(461): FAILED:
explicitly with message:
  explicit Float32 Mmap set_data succeeded
test cases:   4 |   1 passed | 3 failed
assertions: 404 | 401 passed | 3 failed
```

llfio-OFF evidence:

```text
F20_RED_OBSERVATION footprint=288 heap=owning mmap=owning
C:/D/git/dtw-cpp/tests/unit/unit_test_problem_storage_policy.cpp(551): FAILED:
  REQUIRE( heap.storage_policy() == dtwc::core::StoragePolicy::Heap )
with expansion:
  0 == 1
C:/D/git/dtw-cpp/tests/unit/unit_test_problem_storage_policy.cpp(421): FAILED:
explicitly with message:
  explicit Problem Mmap set_data succeeded without llfio
C:/D/git/dtw-cpp/tests/unit/unit_test_problem_storage_policy.cpp(461): FAILED:
explicitly with message:
  explicit Float32 Mmap set_data succeeded
test cases:   4 |   1 passed | 3 failed
assertions: 404 | 401 passed | 3 failed
```

The subject marker is absent in both runs, as required for a red inherited
product. Product attempts consumed remain `0 / 2`.

## Retained product attempts

The inherited expected-red state above was followed by exactly two product
attempts:

1. `aa9781c` (`fix: honor Problem series storage policy`) made owning
   `Problem::set_data(Data)` use the shared series-storage router, retained the
   returned backing/name owner, routed `Problem(name, loader)` through the
   loader's stored result, and added the registered unsupported-route errors.
   Uncached work after moving a mapped Problem exposed that the derived
   `std::function` closures still captured the source `this`.
2. `f1ef5e0` (`fix: preserve Problem storage across moves`) added an explicit
   move repair for the derived dispatch state and made the backing/name bundle
   relocation-aware. Tests poison the moved-from object and force uncached
   const and mutable distance work after both move construction and move
   assignment.

The frozen/generated public contract was published separately in `52b9786`.
The real Python binding gate is `5e67207`; the deterministic owner/name
identity seam and tests are `4834418`; the permanent mutation runner is
`b1aae56`; and the permanent MATLAB oracle is `ec5f218`.

Product attempts consumed: **2 / 2 [confirmed]**. No product rescue-tuning is
permitted inside F20.

## Final focused native subjects

After the mutation campaign, the three source identities were:

```text
76103AABCC5B2492E629AF7C23E9061AA98E0F4C5E6C3B5EFB78DDEC3410848E dtwc/Problem.hpp
BBCEB2B4B61D1A0EE0460718944C6468AC86DDFA99040A3C10A492CDB16E40E7 dtwc/DataLoader.hpp
6B7AF0D832020191473E55984ADC336C2EA940A1A9C78EAB93EAB9EC81652CE4 dtwc/Problem.cpp
```

Both target rebuilds reported `ninja: no work to do.`. The canonical
llfio-ON executable then printed:

```text
F20_PROBLEM_STORAGE_POLICY build=llfio-on footprint=288 heap=owning mmap=view values=72/72 names=12/12 ndim_routes=2/2 ordered_pairs=72/72 artifact=pass lifetime=pass loader_auto=mmap view_override=pass subject_skips=0 verdict=PASS
===============================================================================
All tests passed (963 assertions in 5 test cases)
```

The llfio-OFF executable printed:

```text
F20_PROBLEM_STORAGE_POLICY build=llfio-off footprint=288 heap=owning mmap=rejected values=36/36 names=6/6 ndim_routes=1/1 ordered_pairs=36/36 transaction=pass loader_auto=heap-warning view_override=pass subject_skips=0 verdict=PASS
===============================================================================
All tests passed (606 assertions in 5 test cases)
```

Verdict: **PASS [confirmed]**. Both real executables ran the subject with zero
subject skips after mutation restoration.

## Eleven-mutation execution

The committed runner
`tests/mutation/f20_problem_storage_policy_mutations.ps1` first required
exactly one source match for every operator, recorded the control and mutant
SHA-256 identities, then materialized each mutant, rebuilt and ran its named
profile, and restored/rehashed the source before continuing. The durable
summary is `build/highs-1151/tests/f20-mutation-execution.log`.

```text
F20_MUTATION_SUMMARY controls=4/4 mutations=11 killed=11 survived=0 source_restore=pass verdict=PASS
```

The materialized mutant hashes were:

| ID | SHA-256 |
|---|---|
| M01 | `c0767ab755a928a98e1389f156e078ac6c5cb776d20889f470b45b30c73189ed` |
| M02 | `2fabfd08b7675927654839aed9161433100e058b213978886bc257c3ac1b42fa` |
| M03 | `4da317497c97514edd1e6c1052052f00131138e13f9291ed12ca3a0d498361f2` |
| M04 | `4feb495d739e75f19dbfee02e4b195d06bb461e81382da18604d5167f4b25156` |
| M05 | `63832a8f37f12cbaafa1618e680494fd79dcbc9ce265a0196025ed25c0555d62` |
| M06 | `55776f7d4c9ae143adc11c8e519f4b3fd153a3cfcd25ad3f655193946f00bd36` |
| M07 | `e961ea6439223fa48d7ba3d200ed65cc560367ad5af6e8f39245456fee5acc98` |
| M08 | `b055b58476c9e0f11454b4cccc226decce50349d024d12433e919e12872461bc` |
| M09 | `00ebf4887dac1fe7d1a89a45c7b39d998440d3ff7ddc9bb5b313b45b2bdbb56b` |
| M10 | `8324ffad695084cd0d32f35c1115e453006aac33ff982e3d487a90d5014cbcc1` |
| M11 | `9ff0d181bf926561f532809ba6f2ca69651acf7ac0c3c3b2968aab55aebf1ca7` |

The log contains the runner's 36 machine-readable preflight,
materialization, restoration, and summary lines. PowerShell's `Tee-Object`
did not capture the nested build/test `Write-Host` stream in that file; those
commands were observed in the completed exit-zero runner output and the final
four controls were rerun. The log is therefore cited for the exact mutation
summary, not falsely described as a complete compiler/test transcript.

Verdict: **PASS [confirmed]** - 11/11 registered mutations were killed, all
four controls passed, and every source was restored to its registered hash.

## Final native build matrix

The final canonical gate discovered 122 tests and reported:

```text
100% tests passed, 0 tests failed out of 122

Total Test time (real) = 75.12 sec
```

Its exact six binary-confirmed capability skips were CUDA correctness, CUDA
LB, Arrow I/O reader, Metal correctness, Metal LB, and Metal mmap.

The final llfio-OFF gate discovered 122 tests and reported:

```text
100% tests passed, 0 tests failed out of 122

Total Test time (real) = 67.15 sec
```

Its exact nine binary-confirmed capability skips were mmap data store, mmap
distance matrix, CUDA x2, Arrow I/O reader, Metal x3, and Benders/HiGHS.

The system-Arrow build `build/arrow-pyarrow-23` discovered 124 tests and
reported:

```text
100% tests passed, 0 tests failed out of 124

Total Test time (real) = 70.41 sec
```

Its exact eight binary-confirmed capability skips were mmap data store, mmap
distance matrix, CUDA x2, Metal x3, and Benders/HiGHS. The Arrow reader subject
ran 390 assertions / 11 cases.

Verdict: **PASS [confirmed]** - all three native matrices rebuilt, ran every
non-capability subject, and failed zero tests. Skip reasons were confirmed from
the test binaries rather than inferred from CTest's skip count.

## Final Python binding and suite

The final fresh llfio-ON extension had SHA-256:

```text
94E720F4E6CCBB0ACAC1FDDA748A0F37000EA3F3480836E7A497562E17F34864
```

The focused fresh-extension subjects printed:

```text
F20_PYTHON_STORAGE build=llfio-on route=mmap artifact=pass distances=72/72 fastpam=exact subject_skips=0 verdict=PASS
1 passed in 6.89s
```

and, from the independently rebuilt llfio-OFF extension:

```text
F20_PYTHON_STORAGE build=llfio-off route=rejected transaction=pass artifacts=0 distances=36/36 subject_skips=0 verdict=PASS
1 passed in 0.10s
```

The first full run after rebuilding against system Arrow did not put all
runtime directories on subprocess `PATH`. It produced five HPC CLI child
failures with Windows exit `0xC0000135`, plus the known F39 failure:

```text
6 failed, 1005 passed, 12 skipped
```

That run is retained as a non-decisive environment failure. With all three
required runtime directories present -
`.venv/Lib/site-packages/dtwcpp`,
`.venv/Lib/site-packages/pyarrow`, and
`.venv/Lib/site-packages/pyarrow.libs` - the final inventory was:

```text
1 failed, 1010 passed, 12 skipped in 69.52s
```

The sole failure was exactly the already-open F39 subject:

```text
tests/python/test_supply_chain_pins.py::test_live_tracked_cmake_inventory_is_complete
assert 28 == 27
```

Verdict: **F20 PASS / full-suite known-red [confirmed]**. Both fresh binding
profiles ran with zero F20 skips. The final 1,023-outcome Python inventory has
no F20 failure; its only red remains owned by F39.

## Final MATLAB binding matrix

Artifact identities:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| MEX source | - | `09767547047D72FAAD6A3DB9E8D83A42AEF26B856DC196EE71C9B5398088ED9A` |
| permanent F20 oracle | 12,196 | `DFD3B796D895FE273FC74B015FA862642A0706B253BEF87DAA173EB8575A42F8` |
| llfio-ON MEX | 1,846,784 | `F067ACEB064F795D50CFB0E1294AAD71BAA5522E43C55FD01D056746F590B98A` |
| llfio-OFF MEX | 684,544 | `773517E3AD53881DAB931B9A265BEC63DC4C3229E1B06D319B3748F5027CD883` |

The four registered MATLAB profiles produced:

1. R2024b / llfio-ON: **FALSIFIED [confirmed]**. The last oracle marker was
   `F20_MATLAB_STAGE build=llfio-on stage=mmap-set-data-enter`; MATLAB exited
   `-1073741819` (`0xc0000005`), created zero stores, and wrote
   `build/f20-matlab-storage/llfio-on/R2024b-isolate/temp/matlab_crash_dump.110828-1`.
2. R2024b / llfio-OFF: **PASS [confirmed]**:

   ```text
   F20_MATLAB_STORAGE requested_release=R2024b observed_release=R2024b build=llfio-off route=rejected error_id=dtwc:ioError transaction=pass artifacts=0 distances=36/36 subject_skips=0 verdict=PASS
   ```

3. R2025b / llfio-ON: **PASS [confirmed]**:

   ```text
   F20_MATLAB_STORAGE requested_release=R2025b observed_release=R2025b build=llfio-on route=mmap artifact=408/408 distances=72/72 fastpam=exact transaction=pass subject_skips=0 verdict=PASS
   ```

   The independently parsed 408-byte store had header CRC `0xab81a0e4` and
   SHA-256
   `E284DAFFB9C91C0CA47E80CFC79229A3B855BDC3055DAA135D7D358B7BC5B49B`.
4. R2025b / llfio-OFF: **PASS [confirmed]**:

   ```text
   F20_MATLAB_STORAGE requested_release=R2025b observed_release=R2025b build=llfio-off route=rejected error_id=dtwc:ioError transaction=pass artifacts=0 distances=36/36 subject_skips=0 verdict=PASS
   ```

The same clean RelWithDebInfo llfio-ON MEX passed R2025b and reproduced the
R2024b crash. Its identity was:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| clean RelWithDebInfo MEX | 1,393,664 | `4AE0FE630BE2F5F83A54B4BF34C81ABDE12A0C473BFE15E4325968C51F493846` |
| matching PDB | 9,924,608 | `CB941B2C398981B728DD3DEF91BB0CC90B0B020E4DED3BD33FBAC46607D26830` |

Its symbolized optimized stack was:

```text
[0] MSVCP140.dll Thrd_yield+184
[1] llfio_v2::windows_nt_kernel::doinit+48
[2] llfio_v2::file_handle::file+68
[3] llfio_v2::mapped_file_handle::mapped_file+227
[4] dtwc::core::MmapDataStore::create+3204
[5] dtwc::detail::route_series_storage+680
[6] dtwc::Problem::set_data+440
[7] cmd_Problem_set_data+3688
[8] mexFunction+681
```

PDB line data, imports, and disassembly refine the nearest-export label:
LLFIO commit `b17613fb2149a93b0cc7022c8e649dbf5a015b90` enters its first
Windows initialization `static std::mutex`; the VS 14.50 header emitted
constexpr mutex bytes and no `_Mtx_init_in_situ` import. R2024b's private
MSVCP140 14.36 runtime dereferences the missing legacy vptr at mutex offset
`+8` inside `_Mtx_lock`. R2025b's private 14.40 runtime uses the compatible
SRW representation. The runtime identities were:

```text
R2024b MSVCP140 SHA256=7B0D0D624AC04411646C75555A285D4B33CB5976D80A35B8249D11B33EB631D4
R2025b MSVCP140 SHA256=0CD75546FC6DA6467729F0A60A6705A9391D27526325BD6BBB80E5ED427F6285
```

A matching Debug `-O0` MEX completed and created the 408-byte store. This
optimization/build-representation differential does not overrule the optimized
registered profile.

The full current MATLAB inventory was identical under both releases:

```text
MATLAB_TOTAL=84 PASSED=81 FAILED=2 INCOMPLETE=3
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_routes_match_exhaustive_oracle
INCOMPLETE_NAME=test_contract_parity/test_dtwclustering_metric_validation_precedes_effects
INCOMPLETE_NAME=test_test_api/test_parallelisation_serial_is_honest
OMP_AVAILABLE=1 OMP_MAX=24 OMP_ENGAGED=24 OMP_PASS=1 OMP_REASON=
```

The first two incomplete entries are the two committed F18 expected-red
failures:

```text
Unknown command: 'DTWClustering_compute_distance_matrix'.
Actual: MATLAB:fit:expectedNonempty
Expected: dtwc:invalidArgument
```

The third incomplete entry is the intentional opposite-flavor assumption
filter. Neither release had a second MEX on its path before or after the suite.

Verdict: **MATLAB MATRIX FALSIFIED [confirmed]** - three of four registered
MATLAB profiles pass, but the R2024b llfio-ON profile crashes. Across Python
and MATLAB, the six-profile binding band is therefore **5 / 6**, not 6 / 6.

## Final F20 verdict

**REPAIR RETAINED / CLOSURE FALSIFIED [confirmed].** The retained semantic
repair passes both focused native routes, 11/11 mutations, all three full
native matrices, both fresh Python routes, R2025b MATLAB ON/OFF, and R2024b
MATLAB OFF. The preregistered decisive band nevertheless required all six
binding profiles. R2024b MATLAB llfio-ON exits `0xc0000005`, so F20 remains
unchecked after exhausting both product attempts. The failure occurs before
file creation or publication into Problem and is isolated as F43; it does not
falsify the native/Python/R2025b storage semantics.

Microsoft's STL changelog independently warns that mixing a newer-toolset
constexpr mutex with an older redistributable can null-dereference and names
`_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` as an escape hatch. **[inferred]** That
macro is the narrowest next differential, but it was not runtime-confirmed:
the diagnostic configure omitted `DTWC_ALLOW_SEQUENTIAL=ON` and stopped before
compilation. It is not an F20 rescue attempt.

The final review also found a separate taxonomy gap: default series-cache path
discovery calls `std::filesystem::temp_directory_path()` outside the router's
`IOError` translation. That unexecuted environment-failure route is isolated
as F44 rather than hidden in a green F20 subject.

Rollback is local revert of `f1ef5e0` followed by `aa9781c` (and, for complete
campaign removal, their associated documentation/tests in reverse commit
order). No remote or operator state changed. The retained repair is not rolled
back because every route unaffected by the host-runtime incompatibility is
green.

The claim most likely to be wrong is that
`_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` alone repairs every R2024b MEX route.
Microsoft documents it as the compatibility escape hatch and the disassembly
matches that failure class, but only a registered optimized R2024b/R2025b
differential can confirm the remedy.
