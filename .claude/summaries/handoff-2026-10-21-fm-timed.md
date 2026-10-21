# Handoff — 2026-10-21 (Windows) — FM timed and fixed; merge gates next (interim, refreshed after the merge)

## Base

- `design-2.0` at 4d65c7a4, local only: 10 docs commits ahead of `origin/design-2.0` aabb2b89 (5 of 10-12, 5 today);
  tree clean. `pb/FM` at 2ef3e075 (local; worktree `C:/D/git/wt/FM`). `pb/FM-metal` dcc847be: on the Mac only.

## Done

- Review for Volkan: the direction follows the 10-09 spec; throughput stalled (nothing merged since SW, 10-10).
- PLAN: finished rows pruned, MS folded into MM (e2c34648); "Before the 2.0 tag" section (7857de03); candidate
  `import dtwcpp` loads scikit-learn and scipy's OpenBLAS pool, +870 MB (694341fa); FM's row (ff5e148b, 4d65c7a4).
- DECISIONS §3: the 10-21 rulings (7857de03, ff5e148b).
- Scratch: 16 finished units' folders in `C:/D/git/wt/tmp` and 6 venvs deleted (16 GB); SW's text evidence copied
  first to `baselines/2026-10-09-sw-blocked-swap/`; `wt/tmp/SW/kit` (empty) stays, locked by some handle.
- FM's record and evidence to `baselines/2026-10-10-fm-float32-matrix.md` and `.../2026-10-10-fm-float32-matrix/`.
- Processes: our stale 10-07 wait loop (bash 51532/30856) killed; Volkan's hung paper_test chain killed on his OK.
- `C:/D/git/wt/tmp/wait_load12.ps1` prints python's share of the CPU too (old copy `.bak`).
- FM's timing (`timing_fm.sh main`, 05:15-07:20, every part exit 0): 45 of 46 bands PASS; FALSIFIED clang's float32
  Lloyd `km` 0.492. Fix round (implementer agent): 2ef3e075, `km` 1.226 (0.988-1.484) PASS.

## Verified by me

- `kit/report.md` verdicts; `km` recomputed with my own script from the raw lines: 1.226 (fix), 0.492 (full run).
- D3 before/after (`kit/asm/3fe34165/d3f_clang.md`, `kit/asm/d3f_clang.md`): the `(69, 12, 6, 0) … MORE STACK OR
  CALLS` loop gone, the four float32 loops equal their float64 twins; `RESULT L clang cases 72 mismatches 0` and
  `RESULT L32 … 72 mismatches 0`; `logs/ctest_clang4.log` "100% tests passed, 0 tests failed out of 95".
- `git merge-tree --write-tree design-2.0 pb/FM`: no conflict (rc 0), at 3fe34165 and at 2ef3e075.
- pid 1140 (`python -`, paper_test venv): 0 reads, 18,734 writes, 187,340 other I/O calls in 5 s; its launching
  process 101252 had exited; after the kill no process started and `labels_cap1.m` kept its 10-13 05:09 time.
- Corrections: (1) I told Volkan that chain's MATLAB step "likely never started": `labels_cap1.log`/`.csv` were
  written 10-13 05:11, so that work was done another way (inferred). (2) The 10-12 handoff's "P1–P3, K1, K2 measured
  first" means first in MM's timing session, with MM's built code (`MM/kit-plan.md` line 9), not before MM's code.

## Reported by agents, unverified

- Fix agent: the clang build had no warnings; GCC and cl see nothing (the `__clang__` guard; not built); the float64
  D3 report equals 3fe34165's once addresses are normalised; the kit's trace and kill probe sources are stale.

## Decisions

- Volkan (question tool): heavy work "Yes, all of it"; scratch "Yes, finished units"; the tag list "Yes, add the
  list"; pid 1140 "Stop its whole chain"; the Lloyd miss "Fix first"; E1 "Don't wait".
- Proposed, awaiting him: MM merged in parts (S1–S2 and S7–S9 on identity gates, S3–S4 after the kit), against the
  design's one merge at S10; FM's scratch (29.6 GB) and `wt/FM` after the merge; whether MM and HG ship in 2.0.

## Next steps

1. PLAN FM: the merge gates on design-2.0 + pb/FM (integrator agent; launched after this handoff): clang 95, Arrow
   tree with `test_fast_clara_parquet_parity` run, CUDA 94, `matlab_suite` 142, pytest ≥ 899/19/0, docs gates, CLI.
2. After it: FM ☑ with the merge's counts; `pb/FM-metal` merges on the Mac onto the merged design-2.0 (no re-timing);
   ask about FM's scratch; remove `wt/FM` and `pb/FM` once merged.
3. FM's E1 whenever Volkan runs `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` elevated.
4. PLAN MM: S1 by a new implementer, after the merge-split question.
5. PLAN "Before the 2.0 tag": the [here] items in machine slots between units.

## Open questions

- From earlier: compacting `ext4.vhdx`; the GitHub history purge; RA's notice; which `.claude` files to untrack.

## Status honesty

- Run today: FM's timing chain (10-11's binaries); the fix agent rebuilt FM's clang tree and probe and ran D3, the
  Lloyd identity cases, ctest and `km`. Not run: the fix under cl or GCC (guarded out), the merge gates (pending),
  E1, the Arrow tree, Metal. Nothing pushed.
