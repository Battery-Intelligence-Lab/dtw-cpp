# Handoff — 2026-11-02 (Windows) — KS4 committed (1234030a); Volkan: simplify the RAM limit, add a run report

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's
  start 446d8b99 = origin. Tree clean.
- `pb/MM` 1234030a (KS4, tree c3ac7090, parent 4741d753 = KS3, 5422bb11 = KS2), local only, worktree `C:/D/git/wt/MM`, clean.
  No agent runs. Reviewer folders (disk, until Volkan's yes): `C:/D/git/wt/tmp/MM/review_ks3_r1/` 0.6 GB, `review_ks3_r2/`
  0.7 GB (+ WSL `~/mm/ks3r2_*` 0.4 GB), `review_ks1/fix` 0.08 GB, `review_ks2/rc` 0.65 GB, `C:/D/git/wt/tmp/conf_strict/`.

## Done

- Weekly-limit stall of 11-01 explained (memory `project-weekly-limit-stall`); status report v2 + Fable audit; PLAN, DECISIONS §3.
- KS3 4741d753 (release at phase end, storage and IPC routing on one thread, silhouette form fc, probe + 23-family gate).
- Fresh re-checks of KS3 (reports `review_ks3_r1/recheck_report.md`, `review_ks3_r2/recheck_report.md`): R1' one blocker (IPC
  routing priced ndim 1) + Parquet mean-length window (S1); R2' loops: four residual deviations (below).
- KS4 1234030a: `in_ram_thread_bytes(..., ndim)`, the routing passes `ipc->ndim()`; test loops `clara_256_lz4` (Standard) and
  `clara_256_ndim2` (DDTW), fails on 4741d753 (`logs/ks4_mut_route_ndim1.diff`); gate regex `(?![.Ll$])`. Log `logs/ks4_summary.log`.
- `cpp_conformance` `strict` PASS on design-2.0 292a9bd7: GCC 13.3 (WSL) and cl 19.50, 8 assertions each
  (`baselines/2026-11-02-conformance-strict.md`; PLAN updated, b985303a).

## Verified by me

- 4741d753 and 1234030a diffs read (stream, thread_scratch, run.cpp, arrow_ipc, scores, tests, docs, gate regex).
- Clang tree `C:/D/git/wt/MM/build`: `ctest -j1` 99/99 at 4741d753 (46 s) and at 1234030a (47 s), skips CUDA, Metal x2.
- Limit sweeps with my scripts (scratchpad `orch_*.sh`): at 4741d753 the four R1 refusals gone, one cost per fixture, no
  refusal above an accepted limit (llfio, no-llfio; the 5422bb11 binary shows 15 and 21); R1''s B1 reproduced (`late_nd2` DDTW
  refused 4.30-4.36M between streamed 4.28M and in-RAM 4.38M; `one_nd2` refused 5,506,137-5,772,440, 819e8703 runs from
  5,506,137). At 1234030a: `one_nd2` least 5,516,441; `late_nd2` DDTW in RAM from 4,378,108, nothing refused; `late150k_nd2`
  DDTW least 8,644,570, in RAM from 12,730,108. Conformance logs opened (Catch2 lines, FP model lines).

## Reported by agents, unverified

- KS3's and KS4's gates (six trees, pytest 915/19/0, docs, codegen `inner_loops=1616 families=23 missing=none calls=0
  libm=66`, 25 of 25 mutations at KS3), KS3's memory table. R2': four deviations, none new in KS3, all in 5422bb11 too: (a)
  clang silhouette post-pass +1 stack store a point; (b) clang ThinLTO ADTW and zero-cost handlers +1-2 stack loads a column;
  (c) GCC LTO Soft-DTW handlers +1-2 loads, +1 store in libm-bound cell loops; (d) wheel Soft-DTW gradient +2 loads, +2
  stores. No small source form (a: three tried; c, d register allocation). R2' gate notes N2-N4 (no code).
  R1': release sound (50 stream runs); fail-late at the pass; memory excess +3-4 MB at 24 threads (not 3.3).

## Decisions

- Volkan 11-02 (question tool): B1 "Release at phase end"; GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12 as the
  test stands; loops of KS3 "Fix first (a KS5 unit)"; and, answering the Parquet question with a free text on the RAM limit
  as a whole (DECISIONS §3, quoted in full): relax it, about 10 % safety margin, "separate several GB for the algorithm, then
  just load the data", a simpler loading interface that could serve an MCU with MBs of RAM (constrain the algorithm choice),
  report a run that does not fit, and "a log of the process afterwards ... Nice log in plain text at the end".
- My calls, not asked: S8/S9 serial; merge design-2.0 into pb/MM after KS closes; Parquet's window gets no code now.
- Awaiting Volkan, carried: whether MM and HG ship in 2.0, the history purge, disk deletions.

## Next steps

1. PLAN MM: spec (his approval first, then briefs) of a simpler memory model: one budget (`--ram-limit`, else ~90 % of
   available), a reserve for the algorithm, a loader that only chooses in RAM / mapped / streamed, the algorithm chosen by
   the budget, a typed report where a run does not fit; and what of KS goes (the exact per-thread pricing, KernelTeam/TeamCap,
   phase-end releases) so that KS5 (the loops) is solved by removal. Second spec: the plain-text end-of-run report.
2. Then merge design-2.0 into pb/MM, fill `<KS-TIP>` in `brief_s5_draft.md`, S5, S6, S7 (the planner, the report), S8, S9, S10.
3. PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4 and 7 if MM ships, FM's E1; conformance legs redone
   on the tag tree if MM merges first.

## Open questions

- Which of KS survives (release on return only?); where the run report lives (S7: `run` is the one path for every language);
  a GCC "no call in a DP loop" gate; `reserve_pair` cost in -fPIC objects; Parquet's mean-length window (carried to the new model).

## Status honesty

- Run by me: clang `ctest -j1` at both commits, the sweeps and scans above, the conformance log reads. Not run by me: the
  Arrow, no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, any timing, mutations.
- Agents this session: Fable audit, R1, R2, KS3 (763 k tokens), R1' (490 k), R2' (440 k), conformance (92 k), KS4 (412 k).
