# Handoff — 2026-10-12 (Windows) — MM's design rounds 4–6 approved; branches and worktrees pruned

## Base

- `design-2.0` at cb701634, local only, ahead of `origin/design-2.0` aabb2b89 by four docs commits; tree clean.
  The Mac's five commits (1d96c83c..aabb2b89) were already on origin and fast-forwarded here.
- `pb/FM-metal` (dcc847be) is NOT on origin and not on this machine: it is on the Mac only.

## Done

- Worktrees: `C:/D/git/wt/WM` and `wt/MV` removed (clean; MV's ignored `build/` and `build-msvc/`, 2.2 GB, went
  with it, on Volkan's "reduce them whichever finished"). Left: `wt/FM` (pb/FM, open).
- Local branches: 75 `pb/*` merged into design-2.0 deleted (`git branch -d`); four superseded, unmerged ones deleted
  with `-D`: `int/X2-merge` 3ec3ec82, `pb/W4a` 7281f6a5 (patch in design-2.0), `pb/X3` 8c73029f (X3b, same subject:
  inferred), `worktree-agent-aff6e262…` 78af336f (Y2's llfio). Left: design-2.0, main, Claude, pb/FM, pb/MV.
- MM design rounds 4, 5, 6 (`C:/D/git/wt/tmp/MM/design.md`, `kit-plan.md`; reviews `review-round4/5/6.md`);
  F3's threshold 3.58 GB/s set by me per review 6, with its registered range on both branches.
- Rulings in DECISIONS §3 (9ca4d0e8, 0e986a9a, cb701634); PLAN's MM row (49cfa609, cb701634).

## Verified by me

- Remote heads (`git ls-remote`): 14; the only task branch is `pb/FM`; the others are Volkan's old ones.
- Review 4's blocker: `python/dtwcpp/__init__.py:241-247` routes MIP and kelley to highspy without HiGHS;
  `pyproject.toml:77` builds the wheel with HiGHS OFF; `mip_Highs.cpp:169-171` throws.
- `run.cpp:116-120` the half-of-what-the-series-leave rule; `matlab-mex.yml:37` HiGHS ON in the MEX;
  `lagrangian_root.cpp:831` dense N² D; `one_batch_pam.cpp:88` raw n·m.

## Reported by agents, unverified

- highspy wheels exist for cp39–cp314 on manylinux x86_64/aarch64, macOS arm64/x86_64, win_amd64 (uv index
  cache); highspy's win wheels ship no linkable libhighs. nanobind carries `python_error` across the hook; the
  `dtwc_run` copy is ODR-safe. Every band in kit-plan recomputed by the reviewer.

## Decisions

- Volkan: MM finding 7 "All through run"; FM "Windows, when idle"; BF "Candidate only"; remote "old branches …
  are fine"; highspy "should be dependency when people do uv install"; Q10 "account for it while loading … put some
  buffer"; MM round 6 "Approve as is" (its six unruled choices listed in DECISIONS §3).
- pb/MV not merged: FALSIFIED (cl lane packing 3–4× slower); kept per PLAN candidates.

## Next steps

1. Volkan: push design-2.0; push pb/FM-metal from the Mac.
2. When this machine is idle (Volkan's simulations done): FM's timing (PLAN FM), then merge pb/FM and pb/FM-metal;
   `test_fast_clara_parquet_parity` in the Arrow 23 tree (rebuild after 292defe3).
3. MM (PLAN MM): measure P1 (one-thread prefetch), P2 (glibc retention), P3 (headroom factor), K1, K2 first;
   code after FM's merge, steps in design.md §6.
4. HG after MM. Candidates unchanged (PLAN).

## Open questions

- From earlier: compacting `ext4.vhdx`; GitHub history purge; RA's notice; which `.claude` files to untrack.

## Status honesty

- Nothing built, run or tested this session (Volkan: no heavy runs while his simulations go). All MM work is
  desk work: designs and reviews, no code. The four `-D` commits leave no ref; gc may prune them in ~2 weeks.
