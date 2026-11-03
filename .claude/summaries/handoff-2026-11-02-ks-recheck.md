# Handoff — 2026-11-03 (Windows) — RB1, RB2 and RB2b committed and verified; RB3 running; RR next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's
  start 446d8b99 = origin. Tree clean.
- `pb/MM` 014123c4 (RB2b, tree 3d72dc1b) on f9e731f9 (RB2, d06ad68e) on a6c1ab2f (RB1, c711dbcc) on 1234030a (KS4; KS = 539d7d7e,
  5422bb11, 4741d753, 1234030a: superseded history); local only, worktree `C:/D/git/wt/MM`. Trees there: `build` (clang), `build-arrow`,
  `build-nollfio`, `build-mex`, `build-cuda`, GCC in WSL. Work files `C:/D/git/wt/tmp/MM/rb1/`, `rb2/`, `rb2b/`, kit `kit/rb[12]*_*`, logs
  `logs/rb{1,2,2b}_summary.log`, RB2's review `review_rb2/` (log.md, sweeps/, tools/). Briefs: `brief_rb3.md` (a fresh agent runs it, log
  `logs/rb3_summary.log`), then `brief_rr1.md` (fill `<RB3-TIP>`). After a restart check the agent's transcript mtime before relaunching.
- RR plan `rr_plan.md` (read by me 11-03; its code claims not yet checked). Disk until Volkan's yes: `C:/D/git/wt/tmp/MM/review_ks*`
  (about 2 GB), WSL `~/mm/ks3r2_*` and `~/conf_strict`, `C:/D/git/wt/tmp/conf_strict/`.

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4;
  `cpp_conformance` `strict` PASS on design-2.0 292a9bd7, GCC 13.3 and cl 19.50 (`baselines/2026-11-02-conformance-strict.md`).
- RB1 a6c1ab2f: kernels, `scores.cpp`, `soft_dtw.hpp`, `barycenter.cpp`, `Problem.cpp`, stream and readers are 819e8703's text but for
  accessor lines; `thread_scratch.cpp` holds the closed list; `run()` holds `ScratchRelease`; `TeamCap` kept (my ruling D1).
- RB2 f9e731f9: `run.cpp` plans once (`thread_bytes`, `working_set`, `fit_threads`, `plan_memory`; a `TeamCap` from the plan to the end of
  `execute`), `core::memory_budget` (90 % of `available_memory()`, which sees a cgroup v2 limit), `resolve_method(..., pam_matrix_holds)`,
  `IpcFile::longest_steps()`, tests 1-3, seven tests' thresholds read from the run's own text, docs.
- RB2b 014123c4 (6 files): the stream refusal no longer claims a least limit and the stream's own refusals say their `--ram-limit` is the
  data cap; the methods' fixed terms priced (TADPole flags + cache, hierarchical dense copy, OneBatch table; MIP/LR-core not); auto's
  per-method threads documented; Tier-2 default 100 % in CHANGELOG; nits; tests.

## Verified by me

- RB1: write-tree c711dbcc; clang `ctest -j1` 99/99; 26-run sweep equal to 819e8703's binary; probe: ~0 held after return (615 MB at
  819e8703); docs gates pass; G6 logs opened: 245/245 identical-source objects byte-equal, classes (a)-(c) absent.
- RB2: diffs read; clang `ctest -j1` 99/99; my scan (`scratchpad/orch_rb2_scan.py`, 10 cases, bisection + grid): no violation, costs =
  819e8703's (it checks exit codes, not the "at least N" text, which is how B1 got past me). Fresh reviewer (Opus; text report, evidence
  `review_rb2/`): B1, S1 (fixed terms), S2, S3 + nits; I reproduced B1, S2, S3.
- RB2b: write-tree 3d72dc1b; diffs read (the OneBatch formula copy matches `one_batch_pam.cpp:32-38,253-258`); clang `ctest -j1` 99/99;
  late6k at 1 byte / 41,984,795 / 43,243,524 / 43,248,595 reads right (clause, then a run); peak commit at the plan's least limit (reviewer's
  `peak.py`, baseline 7.8 MB): TADPole 25.1 MB at 18.2, hierarchical 38.6 at 32.2, pam 7.6 at 0.16: within the budget once the baseline goes.
- G6 logs opened for RB2 and RB2b: kernel, lanes, fill pair, swap, assignment loops text-identical; GCC LTO changes are IO/CLI, `execute`,
  the fill's block driver (an out-of-line `vector<Verdict>` once a block), `fast_clara_chunked`'s outer loops (spills).

## Reported by agents, unverified

- RB1/RB2/RB2b gates: Arrow 102, no-llfio 99, MEX 100 (`matlab_suite` 142/142), CUDA 98, GCC 98; pytest 915/19; sweeps (RB2b: llfio 866,
  Parquet 218, no-llfio 861 runs), mutations caught; class (d) (wheel Soft-DTW gradient, GCC inlining) carried to S10; `TeamCap` read in driver loops.
- Behaviour changes for Volkan: least limits rise at 24 threads (late6k 43.2 MB where 819e8703 streamed from 1.37 MB: the reserve);
  llfio vs no-llfio least limits differ ~10x (real longest series vs 2 x mean); the no-llfio matrix refusal is an `InvalidInput` (was `IOError`).
- 64 MiB cgroup (RB2b): auto never refuses; auto with `-o` picks pam and maps the 100 MB matrix, whose swap thrashed (17 min, stopped);
  explicit pam without `-o` was OOM-killed. TADPole's envelopes are unpriced (RB3 leftover a); OneBatch's formula is copied (leftover b).

## Decisions

- Volkan 11-02 (question tool): GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12; KS3's loops "Fix first (a KS5 unit)"; on the RAM limit
  a free text (DECISIONS §3 in full: relax it, ~10 % margin, "separate several GB for the algorithm, then just load the data", report a run
  that does not fit, "Nice log in plain text at the end"); "Approve RB and RR as written".
- Volkan 11-03 (question tool, DECISIONS §3) on RR: "cpuinfo, optional (Recommended)"; `--no-report` = "No text at all"; counters (swaps,
  start cost, pairs, nodes) "Later, own commit" (RR2). My calls: hostname printed; `Result::save` writes the report.
- My calls, not asked: D1 keep `TeamCap`, D3 streams keep the requested threads, D4 `auto` changes only through a guard, D5 `std::bad_alloc`
  becomes `InvalidInput`; S8/S9 serial; class (d) accepted at RB1, re-measured at S10. Awaiting him: whether MM and HG ship in 2.0; the
  history purge; disk deletions.

## Next steps

1. PLAN MM RB3 (`brief_rb3.md`: `RunStage`, the `bad_alloc` catch outside the guards, `core::peak_memory()` = peak working set, test 4, the
   TADPole envelopes, the OneBatch pin): verify (write-tree, ctest, the OOM test on Windows and WSL), a fresh narrow reviewer. Then RR1
   (`brief_rr1.md`: cpuinfo step zero on the six trees; the fill split must keep digits and `-v` lines), RR2 later.
2. Merge design-2.0 into pb/MM, fill `<KS-TIP>` in `brief_s5_draft.md` (tip = RB's; its KS pricing rules become the plan's), S5, S6, S7, S8, S9, S10.
3. PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4, 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read at S10; the Mac needs its own G6 of the accessor TUs and a Metal build (memory
  `project-metal-on-mac-checklist`); whether the reserve's rise in least limits at 24 threads is what Volkan wants.

## Status honesty

- Run by me: clang `ctest -j1` at RB1, RB2, RB2b and both KS commits, sweeps, the probe, the docs gates, G6 log reads, my RB2 scan, B1 and the
  peaks. Not run by me: the Arrow, no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, any timing, mutations. Agents: Fable audit,
  R1, R2, KS3, R1', R2', conformance, KS4, RB and RR planners, RB1 (905 k tokens), RB2 (814 k + 233 k), its reviewer (309 k).
