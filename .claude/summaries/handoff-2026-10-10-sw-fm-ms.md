# Handoff 2026-10-10 — SW merged, MS measured, FM and MS's branch M building (Windows, session 79bd8991)

## Base

- Branch `design-2.0` at `dba7e809` before this file. Uncommitted and untouched: Volkan's own edits to `LICENSE`,
  `dtwc/dtwc.hpp`, `dtwc/Problem_IO.cpp`, `dtwc/dtwc_cl.cpp` (never stage, stash, restore or reset them).
- Worktrees: `C:/D/git/wt/FM` (pb/FM, FM's implementer, started 10-10 ~21:55), `MV` (kept), `WM` (old). SW's was
  removed after its merge (pb/SW kept). MS's branch-M agent works in `C:/D/git/wt/tmp/MS` and WSL `~/ms`.
- New tree `build/mex-r2025b` (MATLAB R2025b; R2024b was removed 10-10, so `build/mex` no longer builds).
- WSL: `~/sw`, `~/ms` (S64 34.5 GB, I4 4.3 GB), `~/fm`, older units' dirs; `ext4.vhdx` ≈ 93 GB on C:.

## Done

- SW re-timed at af18d202 (15:41–19:54, every part at total CPU < 12 %), second attempt 20:58, e1 by Volkan 21:17;
  record `C:/D/git/wt/tmp/SW/2026-10-09-sw-blocked-swap.md` (copied to baselines). Merged `557cd889`.
- MS measured 19:54–20:52: decoding binds (record `C:/D/git/wt/tmp/MS/2026-10-10-ms-measurement.md`, baselines).
- Records `33122a15` (Volkan's answers), `dba7e809` (MAP §7 swap invariant, LESSONS, PLAN ☑ SW).

## Verified by me

- SW at af18d202: t1 13.56× clang (13.47× without the ETW-overlapped rounds 14–18), 16.72× GCC; t2 3.57/3.72×, t3
  6.54/6.82/6.25×, t4 4.47/4.37× (FALSIFIED); t5 3.76× clang, 6.33× cl; t6a, t6b on clang and cl ≥ 0.985×; p1–p4 PASS
  (GCC p4 0.975× then 0.990×); scores PASS; m1 1.63× FALSIFIED; e1: refs 0.57×/0.76× base, misses 0.224/0.315 a
  lookup, FALSIFIED. Results base = SW on every line (`kit/report.md`, `sw_second.py`, `kit/e1_table_af18d202.md`).
- GCC's 24-thread t4–t6b: invalid twice (base's own medians 1.7–48× its 04:09 ones). Tested: Windows clang no drift
  (0.94–1.00×); `OMP_WAIT_POLICY=active` no help; 20 threads clean (t6a 0.39 ms twice), 23 mostly; host `sspservice`
  ≈ 0.9 core since 15:41 against ≈ 0 at the clean 06:31 run. Mechanism (a busy host core stalls WSL's barriers)
  inferred. WSL's apt upgrades 13:29–13:39 recorded, not tested as a cause.
- MS report arithmetic recomputed from its medians (W at 10^8 119.8 s, T_F 169.3 s, s ≈ 0.75, C_dec/C_team 0.165,
  bytes/A 3.28 GB/s); the rule's path: Decoding, no p ≤ 8 reaches 4.10 GB/s (8: 3.09), R_map24 7.34 ≥ 1.49 → branch M.
- SW's merge: from the integrator's logs and trees, clang 95 (3 MAY_SKIP), Arrow 98, CUDA 94 with
  test_cuda_correctness passed, `CODEGEN_NO_CALLS inner_loops=86 calls=0 verdict=PASS`, matlab_suite 141/141 (R2025b),
  pytest 896/19/0, `diff -r` of the CLI references (156, 65 files) and clara runs empty, CHANGELOG diff = the 6 lines.

## Reported by agents, unverified

- Integrator: unit_test_faster_pam's 1-and-4-threads case ran (14 assertions); Arrow summaries (F8 9/9, F13 4/4).

## Decisions

- Volkan (question tool, 10-10 ~21:20): e1 "Done: it finished"; SW "Merge now (Recommended)" with the misses
  recorded; MS "Accept as is" (the failed band-20 control; W agrees with the model at both bands): branch M next.
- Orchestrator: clang t1 not re-run (the overlap moves its median 0.09); GCC t4–t6b recorded not measurable after two
  attempts; validity rule for WSL shapes (base within 1.5× of its 04:09 median) registered before the second attempt.

## Next steps (PLAN "Now")

1. FM: the implementer's return (scratchpad `brief_fm_impl.md`); verify; timing per its kit on a quiet machine (WSL
   parts gated on base's drift); Volkan's ETW; merge question; integrator.
2. MS branch M: the agent registers bands in `C:/D/git/wt/tmp/MS/branch-m.md` before building; then timing (quiet,
   with a 23-thread check); the report decides whether mapped series go to production code.
3. Candidates (PLAN): SW's incremental refresh and panel prefetch; BF's 8-column tasks; the rest.

## Open questions (Volkan)

- GitHub history purge; the Mac's restore; his four uncommitted edits; RA's notice; which `.claude` files to untrack.
- Compacting WSL's `ext4.vhdx` (elevated: his) once MS's data is no longer needed.

## Status honesty

- Timed by me at the < 12 % gate: SW at 2a63803f and af18d202, MS's parts; e1 by Volkan (11:17, 21:17).
- Not run: FM and branch M (building), macOS (Metal, codegen) since RL, Linux release archives, ARC.
