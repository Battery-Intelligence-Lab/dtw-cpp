# `available_ram_bytes()` returned 0 on Windows — StoragePolicy::Auto never spilled

Date: 2026-09-02 · Owner files: `dtwc/DataLoader.hpp`, `tests/unit/test_storage_policy.cpp`

## Observed before (confirmed)

`dtwc/DataLoader.hpp` had no `_WIN32` branch, so `detail::available_ram_bytes()`
hit the `#else return 0`. Probe (scratchpad `probe_ram.cpp`, clang++ 21 with the
canonical gate's include flags, run on this host, verbatim):

```
available_ram_bytes = 0
threshold(override=0) = 18446744073709551615
SIZE_MAX = 18446744073709551615
```

Consumer trace: `series_storage_threshold(0)` mapped `available == 0` to
`SIZE_MAX`, and `route_series_storage` spilled only when
`footprint > threshold`. So the direction was **never spill**: on Windows
`StoragePolicy::Auto` always kept series on the heap unless `ram_limit()` was
set explicitly. Only two call sites exist (`DataLoader::load_stored`,
`Problem::set_data` via `route_series_storage`); both inherited this.

## Fix

- Windows branch using `GlobalMemoryStatusEx` / `ullAvailPhys`, clamped to
  `size_t`; `<windows.h>` included only under `#if defined(_WIN32)` with
  `WIN32_LEAN_AND_MEAN` + `NOMINMAX` guarded (same guards as `checkpoint.cpp`).
  Linux `sysconf` and macOS `sysctl` branches unchanged.
- `0` is now documented as **UNKNOWN**, not "no RAM", with the contract that
  callers must not auto-spill on unknown — which is what the code already did.
- Decision extracted into the pure helper
  `detail::choose_storage(estimated_bytes, available_bytes, limit_bytes)`
  returning `core::StoragePolicy`; `series_storage_threshold` now takes
  `available_bytes` instead of querying the OS itself. Same predicate, no
  behaviour change on Linux/macOS.
- Docs: `docs/content/guides/data-formats.md` now states the Auto rule
  (half of free RAM; heap when free RAM is unknowable). `api-contract-2.0.md`
  §6.3 only promises "best-effort threshold behavior" — unchanged, still true.
- `tests/mutation/f20_problem_storage_policy_mutations.ps1` M02/M03 needles
  re-pinned to the new predicate text (verified 1 match each via .NET regex).

Post-fix probe on this host: `available_ram_bytes = 50596110336`,
`threshold = 25298055168`, `choose_storage(288, avail, 0) = Heap`,
`choose_storage(1e12, avail, 0) = Mmap`.

## Tests (red first, confirmed)

Three cases appended to `tests/unit/test_storage_policy.cpp`:
`available_ram_bytes > 0` on supported platforms; `choose_storage` truth table
(threshold boundary, unknown RAM, override both directions); Auto keeps the 6×6
fixture on heap through the real `load_stored()` with no `ram_limit`.

Red check: with the Windows branch temporarily short-circuited to `return 0`,
`test_storage_policy.exe [ram]` gave `assertions: 10 | 9 passed | 1 failed`
(`REQUIRE( dtwc::detail::available_ram_bytes() > 0 )`, `0 > 0`). Probe removed,
rebuilt, green.

Green (`ctest -R "DataLoader|storage|mmap|Data"`):
- `build/highs-1151`: **7 tests, 6 passed, 0 failed, 1 skipped** (`test_metal_mmap`).
  `bin/test_storage_policy.exe`: 8330 assertions / 7 cases (`[ram]`: 10 / 3).
- `build/nollfio`: **7 tests, 4 passed, 0 failed, 3 skipped**
  (`unit_test_mmap_data_store`, `unit_test_mmap_distance_matrix`, `test_metal_mmap`).
- Full rebuild of both `build/highs-1151` and `build/nollfio` is clean — the new
  `<windows.h>` include reaches every TU that includes `dtwc.hpp`, including the
  llfio-OFF profile where it was not previously pulled in by llfio.

## Proposed CHANGELOG bullet (Unreleased)

- Fixed: `StoragePolicy::Auto` never spilled to the mmap-backed series store on
  Windows because the free-RAM query was unimplemented there; it now uses
  `GlobalMemoryStatusEx`. Unknown free RAM is documented as "keep on heap".

## Unresolved

- Test suites moved by +3 cases in `test_storage_policy`; AGENTS.md gate floors
  (130/130 etc.) need re-pinning by whoever runs the next full serial gate.
  Not run here (focused `ctest -R` only, per instruction).
- The F20 mutation campaign was not re-executed; only its needles were re-pinned
  to the refactored predicate.
- Behaviour change on Windows for very large datasets: Auto can now route to
  mmap, and mapped series are rejected by the CUDA/Metal upload paths
  (`data-formats.md`). Same exposure Linux/macOS already had; unexercised here.
- macOS still reports *total* physical memory as the "free" proxy (pre-existing).
