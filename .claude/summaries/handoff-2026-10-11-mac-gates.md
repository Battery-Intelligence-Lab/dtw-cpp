# Handoff — 2026-10-11 (Mac) — synced after the Windows units: gates at 6ae7824f, pb/FM's Metal path proven

## Base

- `design-2.0` from 6ae7824f (Volkan pushed the Windows session; the Mac pulled it 22:45); HEAD = this file's commit,
  local only; tree clean. `origin/pb/FM` = 3fe34165; no local `pb/FM` branch.
- `.claude/worktrees/mac-FM/`: 6ae7824f with `origin/pb/FM` merged and NOT committed (`git merge --no-ff --no-commit`),
  with its own `build/` and `build-matlab/`. Remove it with `git worktree remove -f -f` once FM is merged.
- Kept `handoff-2026-10-10-sw-fm-ms.md` (Windows); deleted the 10-09 one.

## Done

- The pull deleted `.claude/CHARTER.md` and `.claude/baselines/` from the Mac (034d8994 untracked them). I restored
  them from `034d8994^` with `git archive … | tar -x -k` (577 files, ignored). They are the 10-09 versions. The
  Windows records after that (`tmp/FM`, `tmp/MS`, `tmp/MM`, the 10-09 to 10-11 baselines) are only in
  `C:/D/git/wt/tmp/handoff-mac/records-2026-10-11.zip`, which is not on the Mac.
- The Mac's gates at 6ae7824f and at 6ae7824f + pb/FM: record `baselines/2026-10-11-mac-gates-fm-metal.md` (on disk).
- PLAN marks: the macOS row ☑; FM's Mac evidence. LESSONS: untracking deletes files from other clones (two lines
  that only repeated CLAUDE.md went to make room).

## Verified by me (commands and verbatim lines in the record)

- 6ae7824f: build 0 warnings; `ctest -j1` 95 (CUDA the one MAY_SKIP); `test_codegen_no_calls` 61 inner loops 0 calls
  (RL's padding loop under Apple clang); Metal 2168 + 169 assertions; `check_docs` PASS (439 flags, 0 undocumented),
  `check_pins` 0 failures, generated docs current; conformance regen: D-19's silhouette ulp only; pytest from a fresh
  venv 904/11/0; matlab_suite 141/141/0/0 on R2026a.
- 6ae7824f + pb/FM: `git merge-tree` clean; build 0 warnings (`metal_dtw.mm` compiled for the first time); ctest 95;
  conformance D-19 only; pytest 907/11/0; matlab_suite 142/142/0/0 on R2026a.
- FM's Metal path, driven through the real CLI (400 series, lengths 64–512): the float32 Metal matrix equals the float64
  Metal matrix narrowed in all 160,000 entries; CPU float32 = CPU float64 rounded once in all entries; labels and
  medoids identical per device; a mapped float32 checkpoint is 64 + 80,200 × 4 bytes and resumes byte-identical; a
  float64 run on it is refused naming `matrix_dtype`. No test covers a GPU fill into a float32 matrix (CUDA or Metal).
- Correction to the 10-10 handoff ("the Metal tests never compiled since W4d, W13a, W10, GC"): all four merged before
  263e0e24, whose Mac ctest ran the Metal tests on 10-07; since then `dtwc/metal/` changed only in comments (AU).

## Reported by agents, unverified

- None this session (no agents ran).

## Decisions

- Taken by Volkan this session: none yet. His 10-11 rulings (MM, HG, FM "Accept all, time FM") are in DECISIONS §3.
- Awaiting him: where FM is timed (its kit and bands are the Windows machine's: `timing_fm.sh`, ≈ 3.5 h quiet; or
  bands re-registered for the Mac); MM's finding 7 (an explicit `ram_limit` in Python's and MATLAB's cluster() would
  go unhonoured for CSV and Parquet: route all of cluster() through C++ `run`?); copying the records zip to the Mac.

## Next steps

1. Volkan: copy `records-2026-10-11.zip` to the Mac (or run MM round 4 on Windows); then MM round 4, design only (PLAN
   MM): fix `tmp/MM/review-round3.md` (1 blocker, 12 should-fixes), put finding 7 to Volkan, and ask him to approve
   the design and bands. Code after FM's merge.
2. FM (PLAN FM): the timing (Windows, or Mac bands), Volkan's merge question, then an integrator; a GPU-into-float32
   test is a candidate for the merge (the CLI probe above is its oracle: float64 narrowed, bit for bit).
3. HG after MM. Candidates as in PLAN; the 10-10 handoff's `import dtwcpp` cost (sklearn, scipy, pandas).

## Open questions

- From 10-10: compacting `ext4.vhdx`; the GitHub history purge; RA's notice; which `.claude` files to untrack.
- The Mac's charter is the 10-09 copy; the Windows one may have later entries (DECISIONS §3 has the rulings).

## Status honesty

- Built and run on the Mac only: `build/`, `build-matlab/`, the FM worktree's two trees, two fresh-venv wheels. Not run:
  any timing (the machine was not quiet, load 2.6–3.0), Arrow (no Arrow tree here), `build-asan/`, Windows, Linux, CI.
