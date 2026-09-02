# Checkpoint fixes — adversarial C1-1/C1-2/C1-5 + F15 nits (2026-09-02)

Gate: `build/highs-1151` (clang/Ninja/Release, HiGHS ON, llfio ON, Arrow OFF).

## 1. Metric tag (C1-1)

**Change.** New private `Problem::dense_cache_metric()` (`Problem.hpp:218-220`)
returns `dense_cache_configuration_.metric`, the field bound from
`distance_cache_configuration(core::MetricType::L1)` at `Problem.cpp:317`. Both
autosave sites (`Problem.cpp:843`, `:1109`) now pass it instead of the literal;
one comment at the first site states the invariant.
**Result.** No behaviour change (both routes evaluate to `L1` today);
`unit_test_checkpoint` still round-trips identity-checked generations.

## 2. Generation accumulation (C1-2)

**Change.** `save_checkpoint` (`checkpoint.cpp:579-591`) lists and removes every
other directory under `generations/` after `replace_current` succeeds — best
effort, `std::error_code`, failures ignored; victims are listed before removal
because deleting during an open `directory_iterator` is unspecified. Documented
in `checkpoint.hpp` (file header + `save_checkpoint`), api-contract §2.7, and
`checkpointing.md`.
**Test.** `unit_test_checkpoint`: "retains exactly one generation" now asserts
1 generation whose manifest `pairs_computed == N(N+1)/2`; the crash-resume test
keeps its mid-fill evidence through `manifest_pairs_computed(partial_dir) == 9`
(poisoned rows 0-1 of N=6) and asserts 1 generation with `pairs_computed == 21`
after the resumed fill; the Pruned-downgrade and explicit-overwrite tests assert
1 generation. New helper `manifest_pairs_computed`.
**Result.** Confirmed on the real CLI too: two runs over a 6-series fixture with
`--checkpoint-interval 2` leave exactly one generation (was 6, then 7).

## 3. Cost documentation

O(N^2) per save, O(N^3/save_interval) per fill, plus the sizing rule (block ≈
`save_interval·N` DTWs vs save ≈ N² number formats) added to `checkpoint.hpp`
(`save_interval`), api-contract §2.7, `checkpointing.md`, and `cli.md`.

## 4. Missing tests (C1-5)

(a) `enabled` + empty `directory` → `InvalidInput`, matrix untouched
(`count_computed() == 0`). (b) `directory` = an existing regular file → the
first autosave throws out of `fill_distance_matrix()`, `is_distance_matrix_filled()`
false, `count_computed() == 15` (diagonal 6 + poisoned row 0 + genuine row 1 of
N=6); a following `enabled=false` fill completes, leaves the poison bit-identical
and matches a brute-force reference elsewhere. (c) New
`tests/unit/unit_test_cli_checkpoint.cpp` drives the built `dtwc_cl`
(located via the OS executable path — Catch2 strips argv[0]'s directory) over a
temporary 6-series fixture: run 1 exits 0 and publishes one generation, run 2
prints `Resumed from checkpoint`, and `--checkpoint-interval` without
`--checkpoint` exits 1 with `--checkpoint-interval requires --checkpoint <dir>.`
Runtime 0.49 s.

## 5. F15 nits

`deterministic_series.hpp`: "under IEEE round-to-nearest" replaces the
unqualified FP-flag claim, and the x87 case is stated (10·k needs ≤ 56
significant bits ≤ 64, so the extended intermediate is exact and cannot
double-round). The source guard now rejects bare `uniform_real_distribution`
(catches CTAD) and `generate_canonical`; the header prose was reworded so the
guard does not match its own documentation.

## 6. Coordinator additions

- `--checkpoint-interval` documented in `cli.md` (table row + paragraph) and
  `configuration.md` (flag list + sentence). `configuration.md` was required by
  `assert_cli_reference`'s second half; it is outside the original file list.
- clang-tidy `checkpoint.cpp:661` (`numeric_limits<int>::digits >=
  numeric_limits<int32_t>::digits`, misc-redundant-expression) and
  `checkpoint.hpp:38` (unused `error.hpp`) are **pre-existing**: both are in
  `HEAD` (`git show HEAD:` confirms) and outside this change's diff.

## Verification

- `cmake --build build/highs-1151` — clean.
- `ctest -R "checkpoint|problem|distance_matrix|cli|deterministic"` — **20/20
  passed, 0 failed** (59.2 s), including the new `unit_test_cli_checkpoint`.
- Assertion counts: `unit_test_checkpoint` 265/16 (was 210/14),
  `unit_test_cli_checkpoint` 33/2 (new), `unit_test_deterministic_series` 178/6
  (was 177/6; CMake floor ≥177/≥6 still satisfied).
- `uv run python scripts/generate_docs.py` then
  `scripts/check_docs_contract.py` and `--cli build/highs-1151/bin/dtwc_cl.exe`
  — both "documentation contract checks passed".

## Unresolved

- Test count floors in `tests/CMakeLists.txt` gained one target
  (`unit_test_cli_checkpoint`); the AGENTS.md gate floor (128/128) must move to
  129 in the same session by whoever owns that file.
- The CLI test asserts `std::system` exit codes; a shell that mangles quoting on
  a path with spaces would fail loudly, not silently.
