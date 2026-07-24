# Handoff - 2026-07-24 - F20 Problem storage policy

## Accomplishments

- Read the live F20 plan, frozen API contract, killed ideas, plan archive,
  `LESSONS.md`, F19 handoff, Task 1.4 history, current Data/DataLoader/Problem
  code, existing storage tests, GPU/CLI direct-vector consumers, and
  Python/MATLAB setter/data bindings.
- Confirmed Task 1.4 tested `DataLoader::load_stored()` only. It did not drive
  `Problem::set_storage_policy`.
- Registered the existing owning `Problem::set_data(Data)` operation as the
  governed cross-language boundary.
- Registered the ownership, transaction, Float32, llfio-OFF, view-preservation,
  move-lifetime, loader-constructor, GPU-loudness, exact-byte, exact-distance,
  binding, mutation, and full-gate bands in
  `.claude/baselines/2026-07-24-f20-storage-policy.md`.
- Captured clean inherited focused baselines at
  `253dfda8e5d150492261582673b47f7859c7d78e`: canonical 8320 assertions /
  4 cases; llfio-OFF 14 assertions / 4 cases / 2 capability skips.
- Corrected the initial `0c6fe79` registration before product/test work:
  Problem is move-only rather than copyable, direct policies are explicit
  rather than threshold-injected, and the direct oracle is `ndim=2` with an
  independent 15-pair DP. No product attempt was consumed.
- Committed the dedicated red-first target as `e4688a7`. Canonical and
  llfio-OFF each execute 404 assertions / 4 cases with 401 passing and exactly
  three F20 failures. Both print
  `F20_RED_OBSERVATION footprint=288 heap=owning mmap=owning`; neither prints a
  passing subject marker.
- Registered six real binding profiles without adding a public capability
  getter: Python ON/OFF plus MATLAB ON/OFF under R2024b/R2025b, discriminated by
  repo-local `.dtws` artifacts or typed transactional rejection.

## Decisions

- Problem series policy governs the next owning `set_data`, not existing data
  retroactively.
- `set_view_data` remains an explicit non-owning bypass.
- Series backing remains separate from distance-matrix mmap and both CLI RAM
  controls.
- Problem must retain a move-stable owner for mapped series and names; storing
  only the view Data is forbidden because it dangles. Problem is already
  non-copyable through `MmapDistanceMatrix`, so F20 adds no copy contract.
- `Problem(name, loader)` honors and owns the loader's complete stored result.
- Explicit unsupported Mmap requests fail before publication; no silent heap
  substitution. Auto retains the existing loud best-effort behavior.
- Mapped Data must be rejected before CUDA/Metal's current owning-vector upload
  paths; the CLI keeps series Heap-backed until it has a governed series-policy
  option.
- Product attempts are capped at two.

## Historical resume point (superseded)

Implement the shared series router and move-stable Problem ownership, then make
the already-committed dedicated target green in canonical and llfio-OFF.
Product attempts consumed: `0 / 2`. Do not touch F21, F26, F39,
distance-matrix selection, or CLI RAM semantics.

## Historical open risks (resolved or superseded below)

- `LoadedData` is move-only with llfio, matching Problem's existing move-only
  effective contract. Its backing and name views must remain valid after move.
- MmapDataStore v1 stores Float64 only. F20 must reject explicit Float32 Mmap
  rather than reinterpret it.
- The default Auto threshold is not observable on Windows because the free-RAM
  query returns unavailable; do not claim that branch validated.
- Fresh Python/MATLAB artifacts may be llfio-OFF. Their decisive real-binding
  proof is then the explicit-Mmap error, not a fake mapped success.

## Final F20 update

- Product attempt 1, `aa9781c`, routes owning `Problem::set_data(Data)` and
  loader construction through the shared series-storage router, retains mmap
  data/name ownership, and rejects unsupported Float32, llfio-OFF, CUDA, and
  Metal routes loudly.
- Product attempt 2, `f1ef5e0`, repairs derived DTW closures after Problem move
  construction/assignment. The tests poison the source and force uncached work.
- `4834418` adds a test-only friend seam and proves exact owner/name pointer
  identity across both moves; `b1aae56` adds the permanent 11-mutation runner.
- `5e67207` and `ec5f218` add the fresh Python and MATLAB real-binding subjects.
- The final llfio-ON focused binary passes 963 assertions / 5 cases and prints
  `values=72/72 names=12/12 ndim_routes=2/2 ordered_pairs=72/72
  artifact=pass lifetime=pass subject_skips=0 verdict=PASS`.
- The final llfio-OFF focused binary passes 606 assertions / 5 cases and prints
  `values=36/36 names=6/6 ndim_routes=1/1 ordered_pairs=36/36
  transaction=pass subject_skips=0 verdict=PASS`.
- Mutation execution: `controls=4/4 mutations=11 killed=11 survived=0
  source_restore=pass verdict=PASS`. The restored source SHA-256 values are
  `76103AAB...0848E` (`Problem.hpp`), `BBCEB2B4...40E7`
  (`DataLoader.hpp`), and `6B7AF0D8...2CE4` (`Problem.cpp`); full hashes are in
  the baseline.
- Full native gates: canonical 122/122 with the exact six capability skips;
  llfio-OFF 122/122 with the exact nine; system-Arrow 124/124 with the exact
  eight and its Arrow reader running 390 assertions / 11 cases.
- Fresh Python ON/OFF focused profiles both pass with zero subject skips. The
  final full inventory is `1 failed, 1010 passed, 12 skipped in 69.52s`; its
  sole failure is the existing F39 `28 == 27` CMake inventory mismatch.
- MATLAB focused profiles: R2024b OFF PASS, R2025b ON PASS, R2025b OFF PASS,
  R2024b ON FALSIFIED with exit `-1073741819` / `0xc0000005` during mapped
  `set_data`, before any `.dtws` exists.
- Both MATLAB full suites report `MATLAB_TOTAL=84 PASSED=81 FAILED=2
  INCOMPLETE=3`; the two failures are the committed F18 expected-red subjects,
  and the remaining incomplete is the intentional flavor filter.

## Crash isolation and new owners

The clean optimized llfio-ON MEX
(`4AE0FE630BE2F5F83A54B4BF34C81ABDE12A0C473BFE15E4325968C51F493846`)
crashes under R2024b and passes under R2025b. PDB/import/disassembly evidence
localizes the fault to LLFIO's first Windows initialization `std::mutex` lock:
VS 14.50 headers emitted the new constexpr representation; R2024b's private
MSVCP140 14.36 dereferences the absent legacy vptr, while R2025b's private
14.40 runtime accepts the SRW representation. This occurs before file I/O
returns or Problem publishes state.

F43 owns that confirmed MATLAB/llfio toolset-runtime incompatibility.
`_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` is the registered next differential but
remains **[inferred]** until an optimized R2024b/R2025b run confirms it. F44
separately owns the review finding that default cache-path discovery can throw
`std::filesystem::filesystem_error` before the router's `IOError` translation.
F42 remains the distinct CUDA Auto-precision crash; common mutex-looking
symptoms do not prove a common source statement.

## Final verdict and exact resume point

F20 is **REPAIR RETAINED / CLOSURE FALSIFIED [confirmed]**. The six-binding
band is 5/6 because R2024b MATLAB llfio-ON crashes, and both permitted product
attempts are consumed. Keep F20 unchecked; do not rescue-tune it. The complete
evidence is in
`.claude/baselines/2026-07-24-f20-storage-policy.md`.

Resume at **F21**. Before any edit, search F21 in the killed-ideas section,
archive, `LESSONS.md`, and prior summaries; inspect the four live legacy
spellings and register the public-header compile/state-equivalence band. Do not
work F42, F43, or F44 out of order.

Rollback, if explicitly required, is local revert of `f1ef5e0` followed by
`aa9781c`, plus associated documentation/tests in reverse commit order. No
remote or operator state changed. The claim most likely to be wrong is that
`_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` alone repairs every R2024b MEX route.
