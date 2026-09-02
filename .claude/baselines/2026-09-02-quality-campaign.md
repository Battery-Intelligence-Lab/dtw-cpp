# 2026-09-02 — code-quality / correctness campaign run-log

Base commit `47126d8` (branch `Claude`). Environment: Windows 11, clang 21 + Ninja
(`build/highs-1151` canonical, HiGHS ON, llfio ON, Arrow OFF), `build/nollfio`,
`build/arrow-pyarrow-23`, `build/cuda-verify` (MSVC host + nvcc 13.0, RTX 4000 Ada),
Python 3.13 venv via `uv`.

## Baseline (before any change)
`ctest --test-dir build/highs-1151` → 125/125, 6 skips (cuda×2, metal×3, io_readers×1).
Log: `build/highs-1151/baseline-2026-09-02-ctest.log`.

## Method
Four read-only review agents → `.claude/reports/2026-09-02-review-{core,algorithms,
io-cli,backends-bindings}.md`. Implementers per area (failing test first, focused
ctest only). Adversarial re-reviews → `2026-09-02-adversarial-batch1.md`,
`-batch2-algorithms.md`, `-batch2-mip-matlab.md`, `-batch3-fixers.md`; fixer rounds
for each. Two behaviour-neutral simplification passes (−96 and −46 lines).
CHANGELOG/LESSONS consolidated (71→57 bullets, 54→50 entries).

## Interim gate (after batch 1-2 fixers, before batch 3)
highs-1151 128/128 (6 skips) · nollfio 128/128 (9 skips) · arrow-pyarrow-23 130/130
(8 skips). Log: `build/gate-interim-2026-09-02.log`.

## PR #32 (Kasper Westman) port verification — hunk by hunk
| PR file | Disposition |
|---|---|
| `dtwc/core/llfio_include.hpp` | applied verbatim |
| `dtwc/Problem.hpp` (ctor pragma push/pop) | applied verbatim (deprecated fields still present) |
| `scripts/test_f22_cpp_deprecations.py` | applied verbatim; py_compile + --help exit 0 |
| `tests/unit/unit_test_deterministic_series.cpp` (`kLibcxx`) | applied verbatim; hashes are PR-measured data (Apple Clang + libc++), unverified here (host profile=relaxed) |
| `tests/CMakeLists.txt` (F15 regex `libcxx`) | applied verbatim |
| `tests/integration/test_distance_matrix_csv_contract.cmake` | ADAPTED: PR's `${matrix_path}` pin is POSIX-only (Windows emits an escaped `\` via `std::quoted`); now collapses the escaped pair first, correct on both platforms; passes on Windows |
| `tests/integration/test_fast_clara_parquet_parity.cmake` | applied verbatim; host observes `softdtw_cost=CC31540B20B024C0` (original encoding) |
| `tests/unit/algorithms/unit_test_nearest_medoid_assignment.cpp` | applied verbatim |
| `.claude/baselines/2026-09-01-f15-libcxx-profile.md` | added byte-identical |
| `CHANGELOG.md`, `.claude/LESSONS.md` | rewritten into today's consolidated entries (facts preserved) |
| `PLAN.md` | digest entry rewritten; the PR cites `2026-09-01-hpc-gcc-fp-and-deprecation.md`, which the PR does not contain — cited the file that exists instead |

## Final gate (all after the last code edit; serial)
| Matrix | Result | Skips |
|---|---|---|
| build/highs-1151 (clang, HiGHS ON, llfio ON, Arrow OFF) | 128/128, 0 failed | 6 (cuda×2, metal×3, io_readers) |
| build/nollfio | 128/128, 0 failed | 9 |
| build/arrow-pyarrow-23 | 130/130, 0 failed | 8 |
| build/cuda-verify `-R cuda` (MSVC+nvcc, RTX 4000 Ada) | 4/4 RAN: 7827/61, 532/5, 25/25, 688/8 assertions | 0 |
| Python `uv run pytest tests/python -q` | 1045 passed, 12 skipped, 0 failed | — |
| `scripts/check_docs_contract.py` | passed (after restoring D3-pinned clamp lines and repointing the tadpole marker) | — |
| `check_repo_hygiene.py` / `check_record_hygiene.py` | PASS / passed | — |
Log: `build/gate-final-2026-09-02.log` (native/Python), CUDA rebuilt via VS dev shell afterwards.
Diff size at gate: dtwc +1740/-1181, tests +1987/-253, python +245/-63, bindings +154/-73.
Two lessons re-learned this session: derivation-pinned production lines (D2/D3) are
contracts — run `check_docs_contract.py` before landing "dead arithmetic" removals.
