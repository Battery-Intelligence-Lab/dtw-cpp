# Ancillary library survey (2026-09-02)

Read-only survey. Facts verified 2026-09-02 by four parallel agents (web) + repo
inspection. "unverified" is stated where it applies.

## The cost side of every row

A new dep is not just a `CPMAddPackage`. It must be registered in
`REGISTERED_ARCHIVE_IDENTITIES` (`scripts/check_supply_chain_pins.py:85-110`,
URL + SHA256), which two tests assert, plus `THIRD_PARTY_LICENSES.md`, a
CHANGELOG entry, and an OFF-build proof if optional. A swap therefore pays only
when it removes *risk*, not merely lines.

## 1. YAML config into CLI11

**CLI11 ships no YAML formatter.** `include/CLI/ConfigFwd.hpp` defines only
`Config`, `ConfigBase`, `using ConfigTOML = ConfigBase`, `ConfigINI` — verified
in source ([ConfigFwd.hpp](https://github.com/CLIUtils/CLI11/blob/main/include/CLI/ConfigFwd.hpp)).
The docs say plainly: *"CLI11 is not a full TOML parser as it just reads values
as strings"* ([book-config](https://cliutils.github.io/CLI11/book-config.html)).
The extension point is `virtual std::vector<ConfigItem> from_config(std::istream&) const`.
Note CLI11 is now **v2.7.2 (2026-08-02)**; we pin v2.6.2.

| Lib | Licence | Latest | Hdr-only | C++ | Maint. | Why / why not |
|---|---|---|---|---|---|---|
| [fkYAML](https://github.com/fktn-k/fkYAML) | [MIT](https://github.com/fktn-k/fkYAML/blob/develop/LICENSE.txt) | v0.4.4, 2026-08-20 | yes, single header | C++11 | commit 2026-09-02, 5 issues | **Pick.** Smallest integration cost for a `CLI::Config` subclass |
| [rapidyaml](https://github.com/biojppm/rapidyaml) | MIT | v0.16.0, 2026-07-22 | amalgamation | C++11 | 2026-08-31, 35 issues | Faster, exception-optional, but 14 MB w/ c4core |
| [yaml-cpp](https://github.com/jbeder/yaml-cpp) | MIT | 0.9.0, 2026-02-04 | no | C++11 | 183 open issues | We removed it yesterday; don't reinstate |
| [libfyaml](https://github.com/pantoniou/libfyaml) | MIT | v1.0.0-**beta1**, 2026-08-13 | no (C) | — | 2026-09-01 | Beta, C ABI, 15 MB |
| [mini-yaml](https://github.com/jimmiebergmann/mini-yaml) | MIT | no releases, 2025-10-26 | yes | C++11 | stale | YAML 1.0 only |

**Recommendation: do nothing now.** YAML was deliberately deleted on 2026-09-02
(`CHANGELOG.md:83-88`) because the hand-rolled loader overrode explicit CLI
flags — a *precedence* defect, not a parsing one. If YAML returns it must be
fkYAML behind a `CLI::Config` subclass; today the feature has no user.
**toml++** ([MIT](https://github.com/marzer/tomlplusplus)) parses TOML properly
but its last release is **v3.4.0, 2023-10-13** (master active to 2026-07-21) and
we use no typing beyond strings. Skip. One real gap either way: `dtwc_cl.cpp:770`
is the only `set_config` call and does **no extension check** — a `.yaml` file
passed to `--config` is silently parsed as TOML. A 5-line fix, not a dependency.

## 2. CSV/TSV parsing

Our parser already does the two things that matter: `std::from_chars` with
`chars_format::general` (`fileOperations.hpp:164-201`) and streaming
`getline` + `string_view` fields. ~300 lines total.

| Lib | Licence | Ragged? | `string_view`? | mmap/stream | Note |
|---|---|---|---|---|---|
| [csv-parser](https://github.com/vincentlaucsb/csv-parser) | MIT | **yes** (`VariableColumnPolicy`) | yes | mmap default | Best all-rounder; 2 open issues |
| [rapidcsv](https://github.com/d99kris/rapidcsv) | BSD-3 | unverified | no | **no** — whole file in RAM | Disqualified at GB scale |
| [fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser) | BSD-3 | **no** (column count is a template param) | no | stream | Disqualified: cannot express ragged |
| [zsv](https://github.com/liquidaty/zsv) | MIT | yes | yes | yes | Fastest, but C + autotools; poor CPM fit |
| [csv2](https://github.com/p-ranav/csv2) | MIT | undoc. | no | mmap | Stale (2023-12-23) |
| [lazycsv](https://github.com/ashtum/lazycsv) | MIT | undoc. | yes | mmap | Tiny, 96★ |
| [fast_float](https://github.com/fastfloat/fast_float) | Apache-2.0/MIT/BSL | n/a | n/a | n/a | 4–10× strtod; **canonical repo is `fastfloat/`, not `lemire/` (13★ fork)** |

**`std::from_chars` FP status:** libstdc++ since GCC 11, MSVC since 2018,
**libc++ only since LLVM 20** ([status](https://libcxx.llvm.org/Status/Cxx17.html)).
That is the whole fast_float argument: with Apple Clang / libc++ < 20 in the
matrix, `fileOperations.hpp:186` does not compile there. **Verify the floor
first.** Otherwise keep our parser — no candidate adds a capability we lack, and
rapidcsv (removed as unused) still materialises the whole file.

Two defects found in passing, both cheaper than any dependency:
`matrix_io.hpp:143` `read_csv` uses locale-dependent `std::stod` (inconsistent
with the loaders), and `lower_ascii` heap-allocates per numeric field just to
test for `"nan"`.

## 3. Hashing — keep all three

SHA-256 is frozen by FIPS 180-4, so correctness is decidable against NIST CAVP
vectors, and the digest is a format contract we control. [PicoSHA2](https://github.com/okdshin/PicoSHA2)
(MIT, 1 header, ~360 LOC, commit 2025-05-04) is a *lateral* move: no CMakeLists,
so it needs a shim target, and we would still own the vectors. Chocobo1/Hash is
GPL-3 — disqualified. OpenSSL pays only as an optional SHA-NI accelerator, and
checkpoint headers are not hot.

CRC32: **polynomial trap.** [google/crc32c](https://github.com/google/crc32c)
(BSD-3) is CRC-32**C** (0x1EDC6F41), not our zlib 0xEDB88320 — adopting it
breaks the on-disk format. [CRCpp](https://github.com/d-bahr/CRCpp) (BSD-3,
2026-08-31) is format-compatible but 2153 lines replacing 29. Keep; add a
256-entry table if throughput ever matters (same output, no dependency).

Non-crypto alternatives: **XXH3 alone has a written stability guarantee**
(["any future version will also generate the same hash value"](https://github.com/Cyan4973/xxHash/releases/tag/v0.8.0);
BSD-2 for the library, the xxhsum CLI is GPL-2). **wyhash** changed output
across final3/final4 and self-describes as ~62-bit; **rapidhash** (MIT, not
BSD-2) has *no* upstream stability statement and changed output v1→v3 in ~12
months — the guarantee circulating online belongs to the Rust port. Neither is
safe for a persisted digest.

## 4. Random — keep

`pcg-cpp` is dual Apache-2.0/MIT but **stale: last commit 2022-04-08, 53 open
issues** including MSVC build failures. xoshiro is CC0 reference C, not a
library. Decisive point: PCG removes **zero** of our 185 lines —
`std::uniform_int_distribution` and `std::shuffle` stay implementation-defined
whatever the engine, and that is what `portable_random.hpp` actually fixes.
`std::mt19937_64`'s sequence is fixed by the standard; that *is* the property
we need.

## 5. Formatting / logging

Repo uses **zero** `std::format`/`fmt::` and 47 `std::cout` sites in
`dtwc_cl.cpp`. `std::format` floors: GCC 13.1, MSVC 16.11, **libc++ 17**
(experimental before; `__cpp_lib_format` undefined until libc++ 19, so
feature-detect on `_LIBCPP_VERSION >= 170000`). At or above those, {fmt} (MIT,
v12.2.0 2026-06-16) is unnecessary. **spdlog: skip** — a second formatting stack
for progress and error lines. **indicators** (MIT) is the only maintained
progress-bar option and is weak: v2.3 dated **2023-02-15**, 49 open issues, and
`terminal_size.hpp` calls `ioctl(TIOCGWINSZ)`/`GetConsoleScreenBufferInfo` with
**no `isatty()` guard**, so redirected output garbles. We have no progress
display today; if we add one, gate it ourselves.

## 6. mmap / UTF-8 paths

**Keep the UTF-8 helpers.** They are 14 lines of *pure standard C++20*
(`path::u8string()` / `path(std::u8string)`), not platform code — simdutf (a
SIMD transcoder, for ~100-byte paths), utfcpp and Boost.Nowide replace nothing.
Nowide earns inclusion only for UTF-8 `argv`/`std::cout`/`fopen` on Windows, a
larger and different feature.

**mio is unmaintained** — zero tagged releases ever, last commit 2023-03-03,
52 open issues. llfio is active (2026-09-01) but has **no semver tags**, which
is exactly why `Dependencies.cmake` patches `QuickCppLibUtils.cmake`. One agent
argued for ~150 hand-rolled lines instead; that contradicts the maintainer's
"prefer maintained libraries over custom platform code" rule and
`.claude/LESSONS.md:209-212`, which requires an optional-dependency build proof
plus mapped-correctness, locking and failure-path gates before any swap. **No
change without those gates.**

## 7. Testing / bench

google-benchmark is genuinely used: `benchmark/benchmark.h` in **6 of 8** C++
benchmark TUs. Catch2 v3 has built-in `BENCHMARK` with bootstrap CIs
([docs](https://github.com/catchorg/Catch2/blob/devel/docs/benchmarks.md)) and
zero new deps. [nanobench](https://github.com/martinus/nanobench) (MIT, v4.6.0
2026-08-14, **0 open issues**, one header + one TU) adds Linux perf counters and
JSON/CSV/HTML output; google-benchmark has 174 open issues. Cost is 6 rewritten
files — a "later". (Catch2 is now v3.16.0; we pin v3.13.0.)

## 8. Other hand-rolled code worth noting

- `DataLoader.hpp:55-73` `available_ram_bytes()` **returns 0 on Windows** — no
  `GlobalMemoryStatusEx` — so `StoragePolicy::Auto` never spills there. A real
  functional gap, ~10 lines, no library needed.
- `DataLoader.hpp:132-151` temp-filename generation is TOCTOU-prone (no
  `mkstemp`/`GetTempFileName`).
- `checkpoint.cpp:81-138` hand-written leap-year + ISO-8601 date parser (~58
  lines) — C++20 `<chrono>` covers this where available.
- `dtwc/extern/nanoarrow/` (8,599 lines, Apache-2.0) is the only vendored code.

## Ranked verdict

**Do now (no new dependency in any of these):**
1. Reject `--config` files whose extension is not TOML (`dtwc_cl.cpp:770`).
2. Fix `matrix_io.hpp:143` `std::stod` → `std::from_chars`, matching the loaders.
3. Fix `available_ram_bytes()` on Windows (`GlobalMemoryStatusEx`).
4. Drop the per-field `lower_ascii` allocation in `parse_numeric_field`.

**Later:**
5. Determine the libc++ floor. If < LLVM 20, vendor **fast_float** (header-only,
   Apache-2.0) — this is the one CSV-adjacent dependency with a real argument.
6. If YAML is ever requested again: **fkYAML** behind a `CLI::Config` subclass.
7. Consider **nanobench** to replace google-benchmark (6 files to rewrite).

**Skip:** SHA-256, CRC32, PCG/xoshiro, spdlog, simdutf/utfcpp/Nowide,
rapidcsv/fast-cpp-csv-parser, toml++, mio, xxHash/wyhash/rapidhash as digest
replacements. llfio→anything is blocked on the LESSONS.md gates, not on this
survey.
