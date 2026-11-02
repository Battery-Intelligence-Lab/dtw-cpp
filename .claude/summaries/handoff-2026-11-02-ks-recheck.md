# Handoff — 2026-11-02 (Windows) — RB1 committed (a6c1ab2f), verified; RB2 next; Volkan: simpler RAM limit, run report

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's
  start 446d8b99 = origin. Tree clean.
- `pb/MM` a6c1ab2f (RB1, tree c711dbcc), local only, worktree `C:/D/git/wt/MM`, clean. Below it 1234030a (KS4; KS = 539d7d7e,
  5422bb11, 4741d753, 1234030a, superseded history). Trees there: `build` (clang), `build-arrow`, `build-nollfio`, `build-mex`,
  `build-cuda`, GCC in WSL. RB1's work files: `C:/D/git/wt/tmp/MM/rb1/`, kit `kit/rb1_*`, log `logs/rb1_summary.log`.
- RR plan `rr_plan.md` (not yet reviewed by me). Disk until Volkan's yes: `C:/D/git/wt/tmp/MM/review_ks*` (about 2 GB), WSL
  `~/mm/ks3r2_*` and `~/conf_strict` (0.5 GB), `C:/D/git/wt/tmp/conf_strict/`.

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4.
- `cpp_conformance` `strict` PASS on design-2.0 292a9bd7, GCC 13.3 (WSL) and cl 19.50 (`baselines/2026-11-02-conformance-strict.md`).
- RB1 a6c1ab2f: 34 files, +849/-38 against 819e8703 (44 files, +449/-1,832 against 1234030a). Kernels, `scores.cpp`, `soft_dtw.hpp`,
  `barycenter.cpp`, `Problem.cpp`, the stream and readers are 819e8703's text but for accessor lines (`linear_short_buf`,
  `banded_col_buf`, `lanes_bufs`, `msm_rows`, `twe_rows`, `channel_bufs`, `derivative_bufs`, `wdtw_cache`, `interpolate_bufs`,
  `swap_losses`, `assign_table`, CUDA `workspace_of`). `thread_scratch.cpp` holds the closed list (4 functions, defined beside
  their users so ThinLTO keeps the thread_locals internal); `run()` holds `ScratchRelease`; `TeamCap` kept (my ruling D1).

## Verified by me

- RB1: `git write-tree` = c711dbcc; the kernel diffs against 819e8703 read (accessor lines only); `thread_local` grep: every
  buffer a run route grows is on the list; off it DBA's (no route calls it), the Soft-DTW gradient's, the silhouettes' sums
  (k x 64 B a thread).
- Clang `ctest -j1` 99/99 (38 s; skips CUDA, Metal x2); my sweep `orch_sweep.sh`, 26 runs, 819e8703's binary and RB1 equal in exit,
  cost, route; `check_docs`, `check_pins`, `generate_docs --check` pass; `rb1m_probe.exe file` (24 threads, 400 x 100,000):
  held after return -0.58 MB (CLARA, 300M), +0.3 MB (PAM, 400M); 819e8703 kept 615 MB (agent's log).
- G6 logs opened (`rb1/g6/cmp/`): 245/245 objects of identical-source TUs byte-equal; ThinLTO: dtw_dispatch's only loop change
  is -1 load in two WDTW handlers' outer part, lanes -3, silhouettes equal; GCC LTO: no kernel cell loop grows (grown cells are
  zstd and `lagrangian_root_exact`, via TeamCap); wheel: (d) present. The CUDA test case ran (its one skip is mmap).
- Earlier: 4741d753 and 1234030a read, their clang ctest 99/99, limit sweeps, R1''s B1 reproduced. Not run by me: any assembly.

## Reported by agents, unverified

- RB1 gates: Arrow 102, no-llfio 99, MEX 100 (`matlab_suite` 142/142), CUDA 98, GCC 98; pytest 915/19; `cpp_conformance`, codegen
  `families=23 calls=0`; 80 sweep runs equal; two mutations caught; the agent's per-TU clang: 8 driver loops grew (fast_pam(),
  fast_clara(), fill, ...), none when `TeamCap` is removed (D1's cost, as at KS3).
- Class (d): the wheel's Soft-DTW gradient lambda has +2 loads +2 stores a cell; text unchanged (GCC LTO calls the vector
  destructor out of line); absent in RB1's four other wheels. Source form cannot fix it; it flips with unrelated code size.
- Stale after RB1 (fix in RB2): `parallelisation.hpp` names `core::KernelTeam`; `tests/CMakeLists.txt` says "shrunk within
  --ram-limit"; `make_ipc_fixtures.py` points at removed cases (its fixtures stay: RB2's test 1 and 2 use them).

## Decisions

- Volkan 11-02 (question tool): GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12 as the test stands; KS3's loops
  "Fix first (a KS5 unit)"; on the RAM limit a free text (DECISIONS §3, in full): relax it, about 10 % margin, "separate several GB
  for the algorithm, then just load the data", a simpler loading interface (an MCU with MBs), report a run that does not fit, and
  "a log of the process afterwards ... Nice log in plain text at the end"; then "Approve RB and RR as written (Recommended)"
  (`C:/D/git/wt/tmp/MM/ram_budget_spec.md`): KS's pricing goes, sources back to 819e8703's text, "Release on return" stays.
- My calls, not asked: on `rb_plan.md`: D1 keep `TeamCap` (MAP §7: never `omp_set_num_threads`), D3 streams keep the requested
  threads, D4 `auto` changes only through a guard (N <= 5000 stays PAM, CLARA where PAM's matrix can neither stay in RAM nor be
  mapped), D5 `std::bad_alloc` becomes `InvalidInput`; S8/S9 serial; no code for Parquet's window. 11-02 night: class (d) is
  accepted at RB1 and carried to S10 (final tree, re-measured; then Volkan's call: fix by hoisting in `soft_dtw_gradient`, or accept).
- Awaiting Volkan: RR D1 cpuinfo as an optional dependency (hwinfo needs C++23, cpu_features has no core counts), D2
  `Result::save` writes the report, D3 `--no-report` drops the text too, D4 hostname by default, D5 RR2 (swaps, start cost);
  whether MM and HG ship in 2.0; the history purge; disk deletions.

## Next steps

1. PLAN MM RB2: brief from `rb_plan.md` 3.2-3.5 (no `RunStage`/`bad_alloc`), 3.6, 3.7, tests 1-3, threshold rewrites, docs, the stale
   lines above; my rulings; G6 on the changed TUs against RB1 and 819e8703 (Problem.cpp holds the fill loops); fresh narrow
   reviewer; I reproduce the oracles. Then RB3, then RR (review `rr_plan.md`, ask Volkan its D1-D5, plan is done, implement).
2. Merge design-2.0 into pb/MM, fill `<KS-TIP>` in `brief_s5_draft.md` (tip = RB's), S5, S6, S7, S8, S9, S10 (final G6 vs MM base).
3. PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4, 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read (D1) at S10; the Mac needs its own G6 of the accessor TUs (Apple's thread_locals
  cost a call an access) and a Metal build: memory `project-metal-on-mac-checklist`; RR's machine facts: a library or under 80 lines.

## Status honesty

- Run by me: clang `ctest -j1` at RB1 and both KS commits, sweeps, the probe, the docs gates, G6 log reads. Not run by me: the Arrow,
  no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, any timing, mutations. Agents today: Fable audit, R1, R2, KS3, R1', R2',
  conformance, KS4, RB planner, RR planner, RB1 (905 k tokens).
