# Handoff 2026-10-10/11 — SW merged, MS's branch M confirmed, FM's fix round, MM designing (Windows, session 79bd8991)

## Base

- Branch `design-2.0` at `0d2ce1e9` before this file; the main checkout is clean. Volkan's five edits (`.gitignore`
  `/graphify-out`, `LICENSE` reflowed, the header blocks of `dtwc.hpp`, `Problem_IO.cpp`, `dtwc_cl.cpp`) were
  committed exactly at his request as `6490bc61` (10-11).
- Worktrees: `C:/D/git/wt/FM` (pb/FM at `c72adf3b` + the fix round's commits to come), `MV` (kept), `WM` (old).
- Trees: `build/mex-r2025b` for MEX gates (R2024b removed 10-10). WSL: `~/ms` (S64 34.5 GB, I4 4.3 GB), `~/fm`, `~/sw`.
- `graphify-out/` (Volkan's knowledge graph, 10-11, gitignored): misses 90 % of C++ callers; use `git grep`.

## Done

- SW merged `557cd889` (10-10); records `dba7e809`, `33122a15`.
- MS measured 10-10 (decoding binds); branch M timed 10-11 05:29–05:36: M1–M9 CONFIRMED (record
  `C:/D/git/wt/tmp/MS/2026-10-10-ms-measurement.md`, copied to baselines with `branch-m.md`).
- PLAN marks `e2882d97`; Volkan's rulings of 10-11 in DECISIONS §3 `0d2ce1e9`.

## Verified by me

- Branch M (`kit/report_M.md`, raw lines, load records): validity (a)–(f) PASS in all 20 runs (the stream's digest and
  cost in every run, 256/256 batches zero-copy); T_F(M) cold 5.104 s = 0.447 of the stream's 11.415 s (M2: 0.453).
  M3 straddled its edge in M and is scored on M2, inside. `report_M` scores every band on M2 once M2 exists: there M9
  warm is 1.127 < 1.15 (recorded, not scored; M ran beside msedgewebview2 at ≈ 2 cores, M2 did not).
- FM at `c72adf3b`: the logs show ctest clang 95, cl 94, CUDA 94 (test_cuda_correctness ran), GCC 94, no failure;
  pytest 899/19/0; matlab_suite 142/142 (rerun after a test-list fix); docs gates PASS; CLI diffs empty (156, 65).
- FM's D3 against its registered criterion (kit-plan: no load, call or spill added): FAILS in six loops by the
  record's own account (clang CH row sum and hierarchical's pass; cl the fill's gather; GCC CH between-cluster and the
  fill's tile setup; clang's float32 calculate_medoids), and my scan of `kit/asm/d3_*.md` finds more pairs with FM
  above base (some likely mispairings or cold paths).
- graphify (10-11): C++ cross-file caller recall 0.10 (119 of 1,198 files, 267 functions), precision 0.94.

## Reported by agents, unverified

- FM: D4 identity 0 mismatches on three compilers (I1 1758, I3 96, L 72, S 36, H 18, M 63, F5 288); F1, F3, F4, F6,
  F7, R1–R3 pass; M1 0.502 PASS, M2 PASS trivially, M3 0.791 FALSIFIED (OpenBLAS pools from `import dtwcpp` loading
  sklearn and scipy), its second attempt 0.549 PASS.

## Decisions

- Volkan (question tool, 10-10 ~21:20): SW "Merge now"; MS "Accept as is".
- Volkan (question tool, 10-11 ~05:45): branch M → "Design unit, with check" (unit MM: mapped Arrow IPC series as a
  view `Data` in every language, the per-series check block by block in parallel; design approved before code); his
  uncommitted edits → "You commit all five" (`6490bc61`).
- Orchestrator: FM goes back to its implementer before timing (Volkan's SW ruling "Fix it first" and FM's registered
  D3 criterion).

## Next steps (PLAN "Now")

1. FM: the fix round (scratchpad `msg_fm_fix.md`): every D3 FAIL fixed or reported after three arrangements, a verdict
   for every differing pair, D4/F1/R3 re-proved, full gates. Then timing on a quiet machine (`driver.py run clang
   msvc`, `run-wsl`, `mapped`, ≈ 3.5–4 h), Volkan's ETW (`profile_fm_etw.cmd`), the merge question, an integrator.
2. MM: the design agent (scratchpad `brief_mm_design.md`) writes `C:/D/git/wt/tmp/MM/design.md` and `kit-plan.md`;
   the orchestrator checks it and puts its questions to Volkan; code only after his approval, after FM's merge.
3. Candidates (PLAN): SW's incremental refresh and panel prefetch; BF's 8-column tasks; `import dtwcpp` loading
   scikit-learn, scipy and pandas (+870 MB of OpenBLAS pools at 24 threads, FM's M3).

## Open questions (Volkan)

- GitHub history purge; the Mac's restore; RA's notice; which `.claude` files to untrack.
- Compacting WSL's `ext4.vhdx` (elevated: his) once MS's data is no longer needed (MM's kit will reuse I4).

## Status honesty

- Timed by me at the < 12 % gate: SW (2a63803f, af18d202), MS's parts, branch M (M, M2). e1 by Volkan.
- Not run: FM's timing (blocked on D3), MM (design only), macOS (Metal, codegen) since RL, Linux archives, ARC.
