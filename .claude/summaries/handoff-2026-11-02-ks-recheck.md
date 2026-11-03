# Handoff — 2026-11-03 (Windows) — RB1 a6c1ab2f and RB2 f9e731f9 committed; RB2 under review; RB3, RR next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's
  start 446d8b99 = origin. Tree clean.
- `pb/MM` f9e731f9 (RB2, tree d06ad68e) on a6c1ab2f (RB1, tree c711dbcc) on 1234030a (KS4; KS = 539d7d7e, 5422bb11, 4741d753,
  1234030a: superseded history); local only, worktree `C:/D/git/wt/MM`, clean. Trees there: `build` (clang), `build-arrow`,
  `build-nollfio`, `build-mex`, `build-cuda`, GCC in WSL. Work files `C:/D/git/wt/tmp/MM/rb1/`, `rb2/`, kit `kit/rb[12]_*`,
  logs `logs/rb[12]_summary.log`. Briefs ready: `brief_rb3.md` (fill `<RB2-TIP>`), `brief_rb2_review.md` (running).
- RR plan `rr_plan.md` (read by me 11-03; its code claims not yet checked). A reviewer for RB2 runs (agent `review_rb2/`, report
  `review_rb2/report.md`); after a restart check its transcript's mtime before relaunching. Disk until Volkan's yes:
  `C:/D/git/wt/tmp/MM/review_ks*` (about 2 GB), WSL `~/mm/ks3r2_*` and `~/conf_strict`, `C:/D/git/wt/tmp/conf_strict/`.

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4;
  `cpp_conformance` `strict` PASS on design-2.0 292a9bd7, GCC 13.3 and cl 19.50 (`baselines/2026-11-02-conformance-strict.md`).
- RB1 a6c1ab2f: kernels, `scores.cpp`, `soft_dtw.hpp`, `barycenter.cpp`, `Problem.cpp`, stream and readers are 819e8703's text but for
  accessor lines; `thread_scratch.cpp` holds the closed list (functions beside their users: ThinLTO keeps the thread_locals
  internal); `run()` holds `ScratchRelease`; `TeamCap` kept (my ruling D1).
- RB2 f9e731f9 (25 files, +927/-210): `run.cpp` makes the plan once (`thread_bytes`, `working_set`, `fit_threads`, `plan_memory`; a
  `TeamCap` from the plan to the end of `execute`), `core::memory_budget` (90 % of `available_memory()`, which sees a cgroup v2
  limit), `resolve_method(..., pam_matrix_holds)`, `IpcFile::longest_steps()`, tests 1-3 in `test_thread_scratch.cpp`, thresholds of
  seven tests read from the run's own text, docs (CHANGELOG, cli.md, README, MAP ...).

## Verified by me

- RB1: write-tree c711dbcc; kernel diffs read; clang `ctest -j1` 99/99; 26-run sweep equal to 819e8703's binary; probe: ~0 held after
  return (615 MB at 819e8703); docs gates pass; G6 logs opened: 245/245 identical-source objects byte-equal, classes (a)-(c) absent.
- RB2: write-tree d06ad68e; `run.cpp`, `run.hpp`, `arrow_ipc.*`, tests, docs diffs read; the rules are monotone on paper (every route
  compared at the requested threads' reserve); clang `ctest -j1` 99/99 (50 s); my scan `scratchpad/orch_rb2_scan.py` (refusal scan,
  bisection, grid; own cases): late6k least 43,248,595, long1 140,498,332, one cost = 819e8703's, streamed then in RAM (the rest of the
  cases: see the log there); an explicit limit with the default output dir: PAM at 1.5 MB runs on 3 of 24 threads, mapped matrix.
- G6 logs of RB2 opened: kernels, lanes, fill, swap, assignment loops text-identical to RB1 per TU and ThinLTO; GCC LTO growth only in
  `execute`, the walk, a cold clone of the fill, IO/CLI code; wheel: the CSV writer (+1 store).

## Reported by agents, unverified

- RB1/RB2 gates: Arrow 102, no-llfio 99, MEX 100 (`matlab_suite` 142/142), CUDA 98, GCC 98; pytest 915/19; RB2's sweeps 1,945 runs
  over 10 fixtures (equal to 819e8703's cost in all 705 of its runs), three mutations caught; class (d) (wheel Soft-DTW gradient
  +2 loads +2 stores, GCC inlining drift, source form cannot fix) carried to S10; D1's `TeamCap` read grows driver loops.
- Behaviour changes for Volkan (RB2's report): least limits rise at 24 threads (late6k 43.2 MB where 819e8703 streamed from 1.37 MB,
  long1 140 MB, every100 28.9 MB: the reserve); llfio vs no-llfio least limits differ about 10x (real longest series vs 2 x mean);
  the no-llfio matrix refusal is an `InvalidInput` (was `IOError`); no-llfio runs 15 `s200x100` limits 819e8703 refused.

## Decisions

- Volkan 11-02 (question tool): GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12 as the test stands; KS3's loops "Fix first
  (a KS5 unit)"; on the RAM limit a free text (DECISIONS §3, in full: relax it, about 10 % margin, "separate several GB for the algorithm,
  then just load the data", report a run that does not fit, "Nice log in plain text at the end"); "Approve RB and RR as written".
- Volkan 11-03 (question tool, DECISIONS §3) on RR: "cpuinfo, optional (Recommended)"; `--no-report` = "No text at all"; counters (swaps,
  start cost, pairs, nodes) "Later, own commit" (RR2). My calls: hostname printed; `Result::save` writes the report.
- My calls, not asked: D1 keep `TeamCap` (MAP §7: never `omp_set_num_threads`), D3 streams keep the requested threads, D4 `auto` changes
  only through a guard, D5 `std::bad_alloc` becomes `InvalidInput`; S8/S9 serial; class (d) accepted at RB1, re-measured at S10 (then
  Volkan: hoist in `soft_dtw_gradient`, or accept). Awaiting him: whether MM and HG ship in 2.0; the history purge; disk deletions.

## Next steps

1. PLAN MM RB: read `review_rb2/report.md` (open each cited line); a fix commit on RB2 if it finds a blocker, then RB3 from `brief_rb3.md`
   (`core::peak_memory()` = peak working set on Windows, as `ru_maxrss`: RR compares it with the budget; stage names = RR's). Then RR1
   (check `rr_plan.md`'s code claims; cpuinfo step zero on the six trees; the fill split must keep digits and `-v` lines), RR2 later.
2. Merge design-2.0 into pb/MM, fill `<KS-TIP>` in `brief_s5_draft.md` (tip = RB's), S5, S6, S7, S8, S9, S10 (final G6 vs MM base).
3. PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4, 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read at S10; the Mac needs its own G6 of the accessor TUs and a Metal build (memory
  `project-metal-on-mac-checklist`); RR's machine facts; whether the least-limit rise at 24 threads (the reserve) is what Volkan wants.

## Status honesty

- Run by me: clang `ctest -j1` at RB1 and RB2 and both KS commits, sweeps, the probe, the docs gates, G6 log reads, my RB2 scan. Not run by
  me: the Arrow, no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, any timing, mutations. Agents today: Fable audit, R1, R2, KS3,
  R1', R2', conformance, KS4, RB and RR planners, RB1 (905 k tokens), RB2 (814 k), the RB2 reviewer.
