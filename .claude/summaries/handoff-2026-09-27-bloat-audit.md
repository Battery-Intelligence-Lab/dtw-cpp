# Handoff — 2026-09-27 — bloat audit and design review (Windows machine)

## Base

Branch `design-2.0`, HEAD `cd5d449` ("many things done.", Volkan, 2026-09-27 19:42 — it holds IF-2 S3 and the
09-24 record updates). No commits this session. Uncommitted: `.claude/CHARTER.md` (09-27 entry),
`.claude/plans/2026-09-27-design-review.md`, `.claude/plans/2026-09-27-audit/` (evidence), this handoff, the
PLAN §1 status line, one LESSONS entry, two old handoffs `git rm`-ed. The S4a / S4b / FX-2 agents that were running
on the Mac (snapshot `scratchpad/sync7/`) never reached git; the design review supersedes them.

## Done

- Whole-library bloat audit: 8 slices × (auditor + adversarial verifier), then a Fable-max panel (cut / keep /
  interface) and a Fable-max chair, then three Fable-max advisors → `.claude/plans/2026-09-27-design-review.md`
  (the document Volkan approves). Evidence: `.claude/plans/2026-09-27-audit/` (`digest.md` = verified findings by id,
  `chair.md` §4 = wave detail, `advisor_exec.md` §2–§3 = step schedule and Windows gate commands).
- CHARTER: Volkan's 09-27 instructions appended verbatim.

## Verified by me

- Build: `cmake --build build -j 20` exit 0 (LLVM clang + Ninja, Release, Arrow ON, HiGHS ON, Gurobi found, CUDA OFF).
  Project warnings: 4 × MSVC-CRT `getenv` deprecation.
- `ctest --test-dir build -C Release -j1`: 142 tests — 136 pass, 5 `MAY_SKIP` (2 CUDA, 3 Metal), **1 fail
  `test_config_spellings`** (16 CHECKs like `"cpu" == "cpu"`). Cause: the test reads the child's stdout in binary
  (`test_config_spellings.cpp:68-72`), `lines_of` splits on `\n` only (`:74-80`), `trimmed` strips only `' '`
  (`:194-198`), so each value keeps Windows' `\r`. Fix not yet applied or run.
- Gates: `check_repo_hygiene.py` PASS, `check_docs_contract.py --cli build/bin/dtwc_cl.exe` passed,
  `check_supply_chain_pins.py` PASS (after the new files were written).
- Opened for the headline cuts: pruned fill re-runs every abandoned pair in full (`pruned_distance_matrix.cpp:290-294`)
  and Auto picks it (`Problem.cpp:1189-1193`); MPI reached only by `bench_mpi_dtw.cpp` and `dtwc.hpp:60`; GPU
  1-vs-N / K-vs-N have no callers; v1 CLI flags (`git show v1.0.0:dtwc/dtwc_cl.cpp:43-54`) are all absent at HEAD;
  `load_checkpoint` turns a mismatch or any exception into `false` (`checkpoint.cpp:626-667`); the auto-spill builds
  the `.dtws` from an already heap-resident `Data` (`DataLoader.hpp:200-222`); `dtwcpp` was never on PyPI (404);
  every wheel / MEX / release archive builds `-DDTWC_ENABLE_LLFIO=OFF`; nvcc 13.0 rejects `_MSC_VER >= 1950`
  (`host_config.h:162`) and the only MSVC here is 14.50.35717 (VS 18).
- Baseline evidence FasterPAM vs FastPAM1: `.claude/baselines/2026-07-08-faster-pam-bench.md:54,82`.

## Reported by agents, unverified

- Every LOC estimate and the −108k total; the ~30 silent bugs in the design review §6 other than the ones above
  (the duplicate-medoid case is traced, not run; the Metal empty-series read is inferred).
- The workflow's resume re-ran four verifiers already cached (cache miss); results are consistent with the first run.

## Decisions

- Taken by Volkan this session: none beyond his instructions ("Do not create unnecessary things, extra defensive
  design…", "use parallel agents and workflows to continue design", "use Fable max as advisor" — CHARTER 09-27).
- 2026-09-28, Volkan: design review approved; agents commit locally; GPU CLARA assignment in 2.0; OneBatchPAM "first measure if they are equal"; mio "how is mio performance?" (open). Was awaiting (design review §7): Q1 commits by implementer agents on `design-2.0`; Q2 llfio → mio (overturns
  D-18); Q3 OneBatchPAM measured vs CLARA or kept regardless; Q4 GPU CLARA assignment in 2.0 or 2.1; plus the whole
  review, which overturns D-16, D-22 and design.md's compatibility table. His action: the PRIVATE Kasper report in
  pushed history.

## Next steps

1. On approval: replace `.claude/PLAN.md` with the seven phases A–G of the design review; fold `design.md` into
   MAP + DECISIONS (phase A).
2. Phase A first step: fix `lines_of` in `tests/unit/test_config_spellings.cpp` (strip a trailing `\r`), re-run
   `ctest -R test_config_spellings`; record HEAD conformance digits and the three benchmarks; prove
   `build/cuda-verify` (vcvars64 + `-allow-unsupported-compiler`, else the v143 toolset — owner installs).
3. Then phases B–G per `advisor_exec.md` §2, ≤ 4 agents, one per step, gates per step.

## Open questions

- Does nvcc 13.0 compile the project with MSVC 14.50 under `-allow-unsupported-compiler`? Not tried.
- Is `cpp_conformance` digit-identical between this Windows clang build and the Mac? Not compared.

## Status honesty

Windows only this session. Built and ran: the C++ suite (clang Release) and the three gate scripts. Not run: pytest,
the MATLAB suite, any CUDA build, anything on macOS/Metal. No library code was changed.
