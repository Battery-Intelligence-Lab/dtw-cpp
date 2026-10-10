# Handoff 2026-10-10 — SW timed and fixed, FM and MS designed (Windows, session 79bd8991)

## Base

- Branch `design-2.0` at `eeb9f8ce` before this file. Uncommitted and untouched: Volkan's own edits to `LICENSE`,
  `dtwc/dtwc.hpp`, `dtwc/Problem_IO.cpp`, `dtwc/dtwc_cl.cpp` (never stage, stash, restore or reset them).
- Worktrees: `C:/D/git/wt/SW` (pb/SW `af18d202`, fix pass done), `MV` (kept), `WM` (old). PE-L's was removed.
- Running since 15:30: `timing_sw.sh main` (SW re-time, about 5 h) then `timing_ms.sh R W1 D A P F W32 I report`
  (about 1.5 h); logs `C:/D/git/wt/tmp/timing_sw.log`, `timing_ms.log`.
- WSL: `~/sw`, `~/ms` (S64 34.5 GB, I4 4.3 GB), `~/pel`, `~/al`, `~/bf`, `~/cs`, `~/pe`; `ext4.vhdx` 92.8 GB on C:
  (196 GB free at 15:25). MATLAB R2024b was removed 10-10 about 13:20: `build/mex` no longer builds; use R2025b.

## Done

- PE-L merged `a30e6cf0` (glibc's kept batches returned by `malloc_trim(0)`); records `68316e6c`.
- SW on pb/SW: `fe848afc`, `68eb9342` (blocked swap, silhouette, Dunn); `2a63803f` (N_par 2000 from c0's registered
  1.5× rule: 1.26× at 1000); timed at 2a63803f, then the fix pass `e1bc7f97`, `af18d202` (Volkan "Fix it first").
- FM design approved with conditions (`C:/D/git/wt/tmp/FM/design.md`, `kit-plan.md`); implementation brief ready.
- MS design and decision rule registered (`C:/D/git/wt/tmp/MS/design.md`, `kit-plan.md`); kit built and smoked.
- Records `a6ea7767`, `a72d6d6d`, `eeb9f8ce` (DECISIONS §3, PLAN marks and candidates).

## Verified by me

- SW's code: runs evaluate [j0, j1) against the state at j0, the first accepted in index order (induction holds);
  silhouette and Dunn orders; fast_pam has no caller inside a parallel region (CLARA's samples run serially); the fix
  pass's 8-wide address is `tri_index` rewritten, its 1-wide loop base's.
- SW's kit binaries equal the commits (CR-stripped md5) at 2a63803f; c0 twice (both N_par 2000); the timed tables:
  `C:/D/git/wt/tmp/SW/kit/report.md` (I changed `driver.py`'s report to read every compiler's last full section: clang
  had been dropped), `report_second_attempts.md`, `e1_table_2a63803f.md`; GCC's WSL stalls from the per-pair reps.
- At 2a63803f: t1 14.5–16.7× PASS; t2 3.75–3.80×, t3 6.6–8.9×, t4 3.5–4.5×, m1 1.78× and e1 (misses 0.22 and 0.30 a
  lookup) FALSIFIED; t6b cl 0.966 then 0.961, GCC 0.970 then 0.976 FALSIFIED; p1–p4, scores PASS; results base = SW
  on every timed line.
- C: free space and the VHDX's size; R2024b's missing matlab.exe.

## Reported by agents, unverified

- SW identity at each step (I1 1758, I2 1752, I3 96, I4 100, S 36 on clang, GCC, cl; 0 mismatches); gates (clang 95,
  cl 94, GCC 94, matlab_suite 141 on R2025b, pytest 896/19/0, CLI 156 identical); the asm tables (`fix/path.py`); t2's
  refresh lookups 1.27e10 (27 % of SW's). FM's file:line claims. MS's smoke (zero-copy IPC, mincore 0 % after a drop).

## Decisions

- Volkan (question tool, 10-10): FM Q1–Q4 all (a): rows written in `.dtwm` v5, export at the stored width, v4 read,
  `matrix_dtype` one rule; SW "Fix it first" (GCC's stack reads; the hot-loop rule holds without exception); MS: space
  freed on C:, "WSL decides", "2^25 series, float64".
- Orchestrator: N_par by c0's rule; no second attempt where every pair lay below a band (t2, t3, t4, m1).

## Next steps (PLAN "Now")

1. SW: read the re-time's `report.md`; second attempts for narrow misses (`timing_sw.sh second`, ids adjusted); Volkan
   reruns `profile_sw_etw.cmd` (elevated); the merge question with the misses; integrator (`brief_int_sw.md` in the
   scratchpad: fill «HEAD», «FIGURES», «SUBJECT», write `C:/D/git/wt/tmp/int/sw_changelog_performance.md`); then MAP §7
   (the accepted swaps are the sequential sweep's; a compiler upgrade re-runs SW's identity kit), LESSONS, baselines.
2. MS: `C:/D/git/wt/tmp/MS/kit/report.md` applies the registered decision (F2 if it asks).
3. FM implementer after SW's merge (`brief_fm_impl.md`: fill «HEAD» and SW's reference folders).
4. Candidates: SW's incremental refresh and panel prefetch; BF's 8-column tasks; the rest of PLAN's list.

## Open questions (Volkan)

- GitHub history purge (baselines, CHARTER, Kasper folder); the Mac's restore after pulling; his four uncommitted
  edits; RA's notice wording; which other `.claude` files to untrack (all carried from 10-09).
- Compacting WSL's `ext4.vhdx` after MS (elevated: his).

## Status honesty

- Built and tested by agents (above); timed by me at the < 12 % gate: SW at 2a63803f, its second attempts, both c0
  runs; e1 by Volkan (11:17).
- Not run: SW's fix-pass timing and MS's parts (running), FM code, macOS (Metal, codegen) since RL, Linux release
  archives, ARC.
