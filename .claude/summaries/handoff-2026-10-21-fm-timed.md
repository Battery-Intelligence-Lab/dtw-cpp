# Handoff — 2026-10-21 (Windows) — FM timed, fixed and merged; pb/FM-metal merged; pushed

## Base

- `design-2.0` at the commit after 920a4fe4 (this file), pushed by Volkan's "please push" (10-21): FM's merge
  0548796b, pb/FM-metal's merge bc36375a. `pb/FM` 2ef3e075 and `pb/FM-metal` dcc847be are on origin and merged.
  Worktree `C:/D/git/wt/FM` (pb/FM) still exists; tree clean.

## Done

- Review for Volkan: the direction follows the 10-09 spec; throughput had stalled (nothing merged 10-10..10-21).
- PLAN: finished rows pruned, MS folded into MM (e2c34648); "Before the 2.0 tag" section (7857de03); candidate
  `import dtwcpp` loads scikit-learn and scipy's OpenBLAS pool, +870 MB (694341fa); FM ☑ (920a4fe4).
- Scratch: 16 finished units' folders and 6 venvs deleted (16 GB); SW's text evidence first to
  `baselines/2026-10-09-sw-blocked-swap/`; `wt/tmp/SW/kit` (empty) stays, locked by some handle.
- Processes: our stale 10-07 wait loop killed; Volkan's hung paper_test chain (pid 1140) killed on his OK.
- FM: timed 05:15-07:20 (45 of 46 PASS; FALSIFIED clang's float32 Lloyd 0.492); fix 2ef3e075 (`km` 1.226 PASS);
  merged 0548796b (integrator agent); pb/FM-metal merged bc36375a (me). Records: `baselines/2026-10-10-fm-float32-
  matrix.md` and its folder (report, raw files, D3 before and after, the merge's logs are `wt/tmp/int/fmp_*`, `fmm_*`).
- CHANGELOG: the runtime-DLL line moved from Fixed to Added (fe4cf8c2). Docs audit (agent, read-only):
  `baselines/2026-10-21-docs-audit.md`, 15 findings + 9 lower, not yet fixed.

## Verified by me

- `km` recomputed from the raw lines: 1.226 (fix), 0.492 (full run). D3: the `(69, 12, 6, 0)` loop gone; identity
  `RESULT L … 72 mismatches 0` and `L32 … 72 mismatches 0`.
- FM's merge (logs in `wt/tmp/int`): clang "100% … out of 95", Arrow "… out of 98" with `F8_PARITY … parity=9/9`,
  CUDA "… out of 94", `matlab_suite: 142 run, 142 passed`, pytest "899 passed, 19 skipped"; codegen 88 loops 0 calls;
  every gate log older than the merge commit (12:05:43). pb/FM-metal's merge: three builds exit 0, no warning; ctest
  clang 95, Arrow 98, CUDA 94, the Metal tests skipped (no Metal here).
- Pushes from this repo: 05:25 (694341fa) and 14:16 (0548796b) not mine (Volkan's, inferred); 11:55 and 14:5x mine.
- Corrections: the hung chain's MATLAB step had been done another way (10-13 05:11 files); "P1–P3, K1, K2 measured
  first" means first in MM's timing session (`MM/kit-plan.md` line 9).

## Reported by agents, unverified

- Fix agent: GCC and cl see nothing of the fix (guarded; not built). Integrator: the CLI sets 156 + 65 files identical,
  the GPU set's one difference is a source path in a CUDA error message. Docs audit: all but findings 1 and 2.

## Decisions

- Volkan (question tool): heavy work "Yes, all of it"; scratch "Yes, finished units"; the tag list "Yes, add the
  list"; pid 1140 "Stop its whole chain"; the Lloyd miss "Fix first"; E1 "Don't wait"; runtime DLLs "Keep bundling";
  RA's line "Move to Added"; "Please push" (today's FM work).
- Kept as approved: MM merges once, after its kit (S10); quiet-machine time is the bottleneck (my call, not asked).

## Next steps

1. Mac: prove the merged tree (Volkan's prompt): build, serial ctest with the Metal tests run, codegen, docs gates,
   pytest, matlab_suite R2026a, Apple clang's float32 `calculate_medoids` loop scalar. No re-timing.
2. PLAN MM: S1 (the reader) by a new implementer on `pb/MM` from design-2.0.
3. PLAN "Before the 2.0 tag": the docs audit's fixes (one docs unit); the [here] items in machine slots.
4. FM's E1 whenever Volkan runs `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` elevated; then ask about FM's scratch
   (29.6 GB) and remove `wt/FM`.

## Open questions

- Whether MM and HG ship in 2.0 or after it. From earlier: `ext4.vhdx`; the history purge; which `.claude` files to
  untrack.

## Status honesty

- Run today: FM's timing chain; the fix round (FM clang tree, D3, identity, ctest, `km`); FM's merge gates (four
  trees, CLI sets, pytest, matlab_suite); pb/FM-metal's builds and ctest. Not run: the fix under cl or GCC, E1, the
  Metal tests (the Mac's), the Linux wheel.
