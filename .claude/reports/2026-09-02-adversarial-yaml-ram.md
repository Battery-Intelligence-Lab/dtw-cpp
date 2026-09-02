# Adversarial review — YAML `--config` + Windows free-RAM (2026-09-02)

Binaries: `build/nollfio` (YAML=ON), `build/yaml-off` (YAML=OFF), both fresh.
`ctest -R "test_cli_config_formats|storage"` in `build/nollfio`: 3/3 pass.

## Fix first

**1. CONFIRMED — `<windows.h>` leaks `ERROR` and `GetMessage` into the public umbrella header.**
`dtwc/dtwc.hpp:16` includes `DataLoader.hpp`, so every consumer now inherits them.
Probe: `#include "DataLoader.hpp"` + `#ifdef ERROR #error` → clang emits *"ERROR macro
leaked"* and *"GetMessage macro leaked"* (`interface`/`small` do not). `WIN32_LEAN_AND_MEAN`
does **not** imply `NOGDI`. Cost: `windows.h` alone preprocesses to 48,931 lines, ~25% of
the TU. The deleted comment ("deliberately NO `<windows.h>` in this widely-included
header") was right. Fix: move `available_ram_bytes()` into a `.cpp` (declare in the header).

**2. CONFIRMED — YAML `null`/`~` becomes `""`, which CLI11 reads as a *value*, not "unset".**
`scalar_text` returns `{}` for null. Observed with `--verbose` banner:
`band: ~` → `Band: 0` (near-diagonal DTW, different distances) instead of `Band: full`;
`verbose: ~` → verbose **on**; `name: null` → `Name:` empty → files named `_labels.csv`.
Same for any flag (`no-warm-start: ~`, `resume: ~`). `n-clusters: null`/`method: null` do
error (105). This is silent-wrong-answer class. Fix: throw on null, or skip the key
entirely (= "leave default"), which is the YAML-idiomatic reading.

**3. CONFIRMED (code path) — Windows `Auto` can now select Mmap, and mmap + GPU is a hard error.**
`Problem.cpp:931-940` throws `DeviceError` when `has_mmap_series_storage()` meets
CUDA/Metal. `StoragePolicy::Auto` is the default on both `DataLoader` (`DataLoader.hpp:308`)
and `Problem` (`Problem.hpp:189`), and **`Problem::set_data(Data)` itself routes through
`route_series_storage`** (`Problem.hpp:504-516`) — that, not `load_stored`, is the door most
callers use. The Python bindings take it (`_dtwcpp_core.cpp:1028-1034` → `set_data`;
`storage_policy` is an exposed property). So on Windows + LLFIO-ON + CUDA, a dataset over
50% of free RAM now fails where it previously ran on heap.
`dtwc_cl` **is** immune, but not for the reason one would guess: it pins
`prob.set_storage_policy(StoragePolicy::Heap)` unconditionally at `dtwc_cl.cpp:1241`.
Note also that `set_data` hardcodes `ram_limit_bytes = 0`, so on that path the free-RAM
figure is the *only* threshold source — there is no `ram_limit()` escape hatch, which
makes finding 4 sharper. Undocumented: `data-formats.md` gains the Auto sentence but not
this consequence. With LLFIO OFF the only new effect is a stderr warning
(`DataLoader.hpp:283-287`) where Windows was previously silent.

**4. CONFIRMED — the new doc sentence is false for macOS.** The three branches measure three
different quantities: Linux `_SC_AVPHYS_PAGES` = MemFree (excludes reclaimable cache),
Windows `ullAvailPhys` ≈ MemAvailable (includes standby), macOS `HW_MEMSIZE` = **total**
RAM (its own comment admits it). `data-formats.md` now claims "half of the free physical
RAM reported by the operating system (Linux, macOS and Windows)". Same config spills on
Linux and not on Windows/macOS.

## Precedence (observed, `-v` banner)

| Invocation | Name | k |
|---|---|---|
| defaults only | `dtwc` | 3 |
| `--config p.yaml` | FROM_YAML | 2 |
| `--config p.toml` | FROM_TOML | 3 |
| `--config p.yaml -k 7 --name FROM_CLI` | FROM_CLI | 7 |
| flags **before** `--config` | FROM_CLI | 7 |
| `--config p.yaml --config p.toml` | FROM_TOML | 3 |
| `--config p.toml --config p.yaml` | FROM_YAML | 2 |

CLI > file; last `--config` wins (silently, no "given twice" diagnostic); format-agnostic.

## Lower severity

- **PLAUSIBLE** — misclassification messages mislead. `"a:b" = 1` (TOML) sniffs YAML; in
  `yaml-off` it is refused with *"built without YAML support; use TOML"* — for a TOML file.
  Requires a quoted key containing `:`/`=`; always loud, never silent.
- **PLAUSIBLE** — unknown key errors as `INI was not able to parse not-an-option` (CLI11
  `Error.hpp:327`): no file name, says "INI". Malformed-YAML path does name the file.
- **PLAUSIBLE** — new `examples/cpp/config.yaml` cannot run as shipped (`column: Voltage`
  trips the applicability check on its own `input: data/dummy`); untested.
- **PLAUSIBLE** — `README.md` build-options table has no `DTWC_ENABLE_YAML` row.

## Refuted

- `allow_config_extras(error)`: no shipped/test TOML regresses. `conformance.toml` and
  `examples/cpp/config.toml` (minus the pre-existing `column` issue) run clean in both
  builds; deprecated `clusters`/`restart` still warn-and-accept. **No** `add_option` is
  under a capability `#if` (48/48 unconditional), so `column`/`gpu-precision`/`solver=gurobi`/
  `mmap-threshold` are accepted in the ARROW-OFF, CUDA-OFF build.
- Supply chain: two independent `curl` downloads are byte-identical and hash
  `75fa1ce3…5c4e`, matching the pin; CPM caches agree. Pattern matches Catch2/HiGHS/CLI11/
  Arrow. LICENSE.txt MIT, `FK_YAML_*_VERSION` = 0.4.4; only `single_include` (MIT) is
  exposed. `check_supply_chain_pins.py` PASS; `git ls-files "*CMakeLists.txt" "*.cmake"`
  = 30, so the 29→30 bump is right.
- Stream handling: whole text read once, fresh `istringstream` for TOML. `[table]` on
  line 1 and line 5 both parse.
- Scalars: `band: -1` → full; `1e-5` and `1.0e-5` both fine; `"2"` → 2; `1_0` → 10 (TOML
  agrees); duplicate keys, top-level sequences, nested maps and over-long sequences all
  produce loud non-zero exits (110/114). YAML+BOM parses; TOML+BOM was already broken.
- `choose_storage` tests do hit `estimated == threshold` (`4 GiB` vs `8 GiB` free → Heap);
  `series_footprint_bytes` has an explicit overflow guard.
- The new ctest pins per-flavour check counts and a PASS regex asserting the subject ran.

## Verdict

- **A (YAML/CLI11): fix first** — finding 2 (null semantics), then the message/example nits.
  The CLI11 delegation, precedence, tests and pin are sound.
- **B (free RAM): fix first** — findings 1 and 4; decide and document 3.
