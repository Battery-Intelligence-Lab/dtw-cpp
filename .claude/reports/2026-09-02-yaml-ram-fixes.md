# YAML `--config` + free-RAM fixes (2026-09-02)

`build/nollfio` (clang Release, llfio OFF, YAML ON). All gates green.

## Per item — change + verification

1. **`<windows.h>` leak.** `detail::available_ram_bytes()` moved to new
   `dtwc/system_memory.cpp` (added to `dtwc/CMakeLists.txt`'s explicit source
   list); header keeps the declaration and the restored "deliberately NO
   `<windows.h>`" comment. Probe (`DataLoader.hpp` + `dtwc.hpp`, project include
   path, `#ifdef ERROR/GetMessage/min #error`) → exit 0; negative control with
   `windows.h` first → 2 errors, so it bites. TU 152,845 vs 199,947 lines (−47,102).
2. **YAML `null`/`~`.** `scalar_text` throws `CLI::ConfigError`. Real binary:
   `key 'band' is null; omit the key to use the default`, exit 110. New ctest case
   (g); pinned check count 21 → **23** (a case adds 2 counted checks).
3. **macOS free RAM.** Now `host_statistics64(HOST_VM_INFO64)` free+inactive
   instead of `HW_MEMSIZE` (total) — **[inferred]**, uncompilable here; rollback is
   the `HW_MEMSIZE` branch. Doc comment and `data-formats.md` name all three
   quantities and state they are not comparable.
4. **Auto + mmap + GPU.** CLI confirmed immune twice: `dtwc_cl.cpp:1241` pins
   `Heap`, `:1441` uses `dl.load()`. `data-formats.md` + `devices.md` carry the
   verbatim `DeviceError` and the Heap/`ram_limit` requirement.
5. **Minor.** `examples/cpp/config.{yaml,toml}` now run as shipped (`column`
   commented out; identical cost 611561). README gained a `DTWC_ENABLE_YAML | ON` row.
6. **Tier-1 GPU pins Heap.** Pure `detail::tier1_storage_policy()`, called in
   `api.cpp` before `set_data`. Unconditional test (3 assertions, run-verified).
7. **`Problem::set_ram_limit`/`ram_limit`** (12 lines), consumed by `set_data`
   (was hardcoded 0); loader ctor carries `loader.ram_limit()`. Test captures
   stderr and asserts threshold `1 B` reached `route_series_storage`. Mutation
   M01 regex updated. Nothing bound.
8. **`matrix_io.hpp` `stod` → `from_chars`** (full-token match; loop reproduces
   `getline` semantics incl. dropping a trailing empty field). New case prints
   `F14_LOCALE_NUMERIC locale=de-DE comma_decimal=yes ran` and still reads 1.5.
   Mechanism probe: `stod=1 from_chars=1.5` under `de-DE`.
9. **`parse_numeric_field`** per-field `lower_ascii` `std::string` replaced by
   allocation-free `equals_ascii_ci`; function body greps to `string_view` only.
   `lower_ascii` kept (`DataLoader.hpp:372`).

## Gates

- build → `ninja: no work to do`.
- `ctest -R "cli|config|storage|DataLoader|Data|supply"` → **10/10, 0 failed**.
- `ctest -R "distance_matrix|fileOperations|f14|conformance"` → **9/9**, 1 expected
  llfio skip. `ctest -R "conformance|csv|io|load|parse|text"` → **17/17**, 1
  expected Arrow-OFF skip.
- `check_docs_contract.py` passed; `generate_docs.py` updated (no extra diff).

## Proposed CHANGELOG (Unreleased)

- Fixed: a YAML `--config` key set to `null`/`~` is now a loud error instead of an
  empty value (`band: ~` silently meant band 0).
- Fixed: `<windows.h>` no longer reaches consumers through `<dtwc/dtwc.hpp>`.
- Fixed: macOS `StoragePolicy::Auto` measures free memory, not total RAM.
- Fixed: distance-matrix CSV reading no longer depends on the C numeric locale.
- Added: `Problem::set_ram_limit()` / `ram_limit()`, honoured by `set_data`.
- Changed: the Tier-1 `cluster(...)` GPU route pins `StoragePolicy::Heap`.
- Added: `DTWC_ENABLE_YAML` row in the README build-options table.

## Unresolved

- **Unknown-key error does not name the file.** CLI11 throws it inside
  `App::_parse_config`, not our `from_config`; wrapping needs replacing
  `CLI11_PARSE` (~8 lines, main parse path, double-names files for errors that
  already name them). Left reported per the ≤5-line condition.
- **macOS branch unbuilt here** — likeliest wrong claim is the Mach spelling
  (`HOST_VM_INFO64_COUNT`, `host_page_size`).
- `docs/api-contract-2.0.md` markers `DataLoader.hpp:291-297`/`:299-306` are stale
  line references (pre-existing); the checker only tests string presence.
