# Adversarial review — skip_rows parity, YAML removal, MEX mutex define

Baseline HEAD `6c66fdb`, uncommitted tree, 2026-09-02. Gate build `build/nollfio`
(clang++ 19, Release, ARROW/LLFIO/MATLAB OFF). Concurrent edits under `python/`,
`bindings/matlab/`, `dtwc/Problem.*`, `dtwc/checkpoint.*` ignored except where a
reviewed file is shared.

## Blockers

**S1 CONFIRMED — the docs-contract gate is RED.**
`--checkpoint-interval` (`dtwc/dtwc_cl.cpp:918-921`) is absent from both
`docs/content/getting-started/cli.md` and `.../configuration.md`.
Repro: `uv run python scripts/check_docs_contract.py --cli build/nollfio/bin/dtwc_cl.exe`
→ exit 1, `CLI reference drift: live but undocumented: ['--checkpoint-interval']`.
An independent recheck shows `configuration.md` misses it too. The flag arrives with
the concurrent checkpoint work but lands in `dtwc_cl.cpp`, so the YAML change cannot
merge past this gate. `--yaml-config` itself is clean in both pages.

**S2 CONFIRMED — `CHANGELOG.md` unmodified.** `git status` shows no change for A, B
or C. Non-negotiable #2. The YAML removal is a user-visible breaking removal.

## Change C — fix first

**S3 CONFIRMED — `matlab_suite` floor is too loose** (`tests/CMakeLists.txt:865`).
`sum([r.Passed]) >= 60` against an observed full run of 100 passed
(`.claude/reports/2026-09-02-mex-highs/ctest-matlab_suite.log`); a run with the whole
31-test `test_contract_parity` file excluded scored 71 and would still be green
(`R2025b-gate-excl-contract-parity.log`). An entire suite file can vanish silently.
Same line: the comment says "exactly one member is always filtered", but both
parallelisation names are on `allowed`, so *both* filtered also passes. Encode the
pair invariant; raise the floor to ~95 or make it per-file.

**S4 CONFIRMED — `matlab_suite` silently disappears when `matlab` is off PATH.**
`find_program(DTWC_MATLAB_EXECUTABLE NAMES matlab)` (`tests/CMakeLists.txt:860`) is
configure-time and PATH-only, with no `HINTS ${Matlab_ROOT_DIR}/bin`; on a miss the
`if()` at `:861` simply adds no test and ctest reports nothing missing. This is the
common case — `find_package(Matlab)` locates the install without `matlab.exe` on PATH.
Missing *at test time* does fail correctly. Related, PLAUSIBLE: the block is guarded
only by `if(DTWC_BUILD_MATLAB)` (`:859`), so if `matlab` is on PATH but
`find_package(Matlab COMPONENTS MX_LIBRARY)` fails, `bindings/matlab/CMakeLists.txt:8-11`
returns early, `dtwc_mex` never exists and `$<TARGET_FILE_DIR:dtwc_mex>` (`:865`) is a
generate-time hard error. Note `if(TARGET dtwc_mex)` is *not* a usable guard:
`add_subdirectory(tests)` is `CMakeLists.txt:289`, `bindings/matlab` is `:305`.

**S5 CONFIRMED (scope), REFUTED (ABI break) — `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`.**
`CMakeLists.txt:58-64`. Guard correctness is sound: the block sits after
`project()` (`:20`), so `MSVC`/`CMAKE_CXX_SIMULATE_ID` are set; the second disjunct is
load-bearing, not dead — GNU-frontend clang on Windows leaves `MSVC` false, the exact
case `bindings/matlab/CMakeLists.txt:41-44` already documents; MinGW (empty
`SIMULATE_ID`) and Linux/macOS are correctly excluded; the `STREQUAL` is CMP0054-safe.
The defect is scope and honesty: `add_compile_definitions` at directory scope precedes
`add_subdirectory(dtwc)` `:235`, `tests` `:289`, `python` `:301`, so with
`DTWC_BUILD_MATLAB=ON` the Python `.pyd`, `dtwc_cl` and every test change mutex init
too. The comment names only "HiGHS/llfio". It also overstates "every TU here": llfio's
`outcome` is built by a separate ExternalProject cmake invocation that does not inherit
the define.
ABI, verified directly in MSVC 14.50.35717 `include/mutex`: `:33-46` keeps the identical
`_Mtx_storage` member in both branches — **layout and size are unchanged**, only the
ctor's constexpr-ness. The sharper hazard is `:523-531`, where `condition_variable()`
is `= default` (trivial, no init call emitted) without the define and calls
`_Cnd_init_in_situ` with it. Mixing a defining and non-defining TU is an ODR violation
(IFNDR) on inline ctors, not a size/layout break. `dtwc/**/*.hpp` exposes no
`std::mutex`/`condition_variable`, and no headers are installed at all
(`CMakeLists.txt:265-278` installs only `dtwc_cl` + docs), so the consumer risk is
currently nil but unpinned.

## Change A — merge (advisory only)

**`lr_max_nodes` — genuine parity closure, doc row accurate.** At HEAD the field existed
only in C++ (`git grep -c lr_max_nodes 6c66fdb` → `dtwc/Problem.hpp:2`,
`dtwc/mip/lagrangian_root.cpp:4`) — live, validated (`Problem.hpp:87`) and consumed
(`lagrangian_root.cpp:720`), but with no Python and no MATLAB route. The diff adds
`.def_rw` + `repr` (`python/src/_dtwcpp_core.cpp:465,475`), the MEX get/set and field
list (`dtwc_mex.cpp:896,912,922`), and round-trip + non-integer-rejection tests in both
languages. The doc row (`docs/api-contract-2.0.md:286`) lists exactly the nine
`MIPSettings` members in struct order — accurate and complete. It is correctly **not** a
CLI flag (`grep -c lr_max_nodes dtwc/dtwc_cl.cpp` → 0), so no TOML/docs surface is owed.

**S11 CONFIRMED (LOW) — `lr_max_nodes` is `long`, so its public range is now
platform-dependent.** `dtwc/Problem.hpp:73` types it `long` while every sibling integer
(`time_limit_sec`, `numeric_focus`, `mip_focus`, `max_benders_iter`) is `int`. `long` is
32-bit on Windows (LLP64) and 64-bit on Linux (LP64), so `long` buys nothing on Windows
and the accepted range silently differs by platform. The typing is pre-existing, but
this change is what first exposes it as a public Python/MATLAB setter — a value above
2^31-1 will round-trip on Linux and raise on Windows. Both parity tests use `12345`, so
nothing catches it. Make it `int` (matching siblings) or `std::int64_t` (explicit).

**S12 CONFIRMED (LOW) — the MIP-settings field list is unenforced prose.**
`grep -n "lr_max_nodes\|max_benders_iter\|MIPSettings" scripts/check_docs_contract.py`
→ no hits. Unlike the Tier-1 signatures, which are pinned needle-for-needle
(`:563-570`), the §2.1 row can drift from the struct silently — exactly the drift this
change is fixing, one field late.

**S6 char→int trap — REFUTED in-repo, PLAUSIBLE forward.**
`git grep 'dtwc::load(' 6c66fdb` shows **zero** 3-arg `load(p, int, char)` call sites at
baseline. The 4-arg legacy forms (`load(csv, 0, ',', "quickstart")`) are *compile
errors* under the new signature — `const char*` will not convert to `char` — so the
compiler caught every existing site; all seven were updated. The residual is a new or
external `load(p, 1, ',')`, which binds `skip_rows=44, delimiter=0` silently.
Recommend closing it with `Dataset load(const std::filesystem::path&, int, char,
std::string_view = "") = delete;` plus the series twin — cheap and permanent.

**S7 CONFIRMED — undocumented third semantics for a DIRECTORY source.**
`api.cpp:151-153` hands `start_row` to `DataLoader`, which for a directory routes to
`load_folder` (`fileOperations.hpp:335-347`) and applies `start_row` **per file**, so
`skip_rows` drops leading lines of *every* series file rather than leading series. The
contract row (`docs/api-contract-2.0.md:126`) documents only the batch-file and
in-memory cases.

**S8 REFUTED — Parquet is not silently ignored.** `dtwc::load()` never routes to
Arrow/Parquet; a `.parquet` path reaches the text parser and throws `IOError`, the same
loud failure the header test pins. The CLI does guard explicitly
(`dtwc_cl.cpp:254-266`); Tier-1 has no equivalent because it supports no Parquet at all,
so the doc's "the `dtwc_cl --skip-rows` meaning" is loose but not wrong.

**REFUTED — `skip_rows >= N` inconsistency.** Both routes clamp to zero series and fail
loudly: CLI exit 1, `Error: fast_pam_seeded: Problem has no data points.`; Tier-1
`InvalidInput("cluster: dataset is empty.")` (`api.cpp:333`). Messages differ, kind does
not.

**REFUTED — `check_docs_contract.py` weakened.** The deleted block only checked YAML
`set_if_unset` key *names*; the retained `config_missing = live - config_flags` check
(`:2277-2286`) is strictly stronger. The signature pin at `:566` was updated correctly —
though only the path overload is pinned, not the series twin.

**Test coverage adequate.** `tests/unit/test_tier1_cpp_api.cpp:359-401` writes two
non-numeric header lines plus four data rows and pins `labels().size()==4`; `skip_rows`
of 1 or 3 would both fail it. Ran green (6 assertions).

## Change B — merge after S1/S2

All five attacks refuted. Validation loss: none — the removed block ran *before* the
retained `to_lower`/alias normalisation (`dtwc_cl.cpp:1029-1060`) and
`validate_cli_route_selectors`/`validate_cli_distance_configuration` (`:1067`, `:1072`),
all of which still cover `--config`. All 35 `set_if_unset` keys map to live CLI11 long
flags, boolean flags included (CLI11 2.6.2 routes zero-expected options through
`_add_flag_like_result`); the `clusters`/`restart` deprecation warnings still fire from
TOML. `check_supply_chain_pins.py` → exit 0,
`CMAKE_ARCHIVE_PIN_GATE verified=6 total=6`, `TRACKED_CMAKE_MANIFESTS total=28`
(`REGISTERED_CMAKE_MANIFEST_TOTAL` counts files, correctly unchanged). yaml-cpp is gone
from every tracked file; the only survivors are the intentional rejection test.
`ctest -R "test_cli_rejects_yaml_config|unit_test_cli_args|supply_chain"` → 3/3 passed.

**S9 CONFIRMED (LOW) — stale gates after removal.**
`.claude/skills/check-code-quality.md:433` greps for the now-deleted
`TODO: This always overrides CLI values if the YAML key exists` string, so that check is
a silent no-op. `.claude/skills/update-docs-skill.md:25` still calls
`configuration.md` "TOML/YAML config options".

**S10 CONFIRMED (LOW) — `test_cli_rejects_yaml_config` is exit-code-blind.**
`tests/CMakeLists.txt:748-757` uses `PASS_REGULAR_EXPRESSION`, which makes CTest ignore
the exit status entirely. The message is specific (so accept-and-ignore fails it) but is
coupled to the CLI11 version and to its plural form.

## Verdicts

| Change | Verdict |
| --- | --- |
| A skip_rows / lr_max_nodes parity | **MERGE.** Fix S1/S2. S6 deleted-overload and S7 doc line are advisory. |
| B YAML removal | **MERGE.** Clean, no feature loss. Fix S2, then S9/S10. |
| C MEX define + matlab_suite | **FIX FIRST.** S3 and S4 make the new gate able to pass while the suite shrinks or vanishes. S5's tree-wide scope must at least be documented. |
