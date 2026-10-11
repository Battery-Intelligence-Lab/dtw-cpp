# Handoff 2026-10-10/11 — SW merged, branch M confirmed, FM accepted but untimed, MM designed; next on the Mac

Windows session 79bd8991, ended 2026-10-11 ~22:30.

## Base

- `design-2.0` is at this file's commit, LOCAL ONLY (`origin/design-2.0` is 263e0e24 of 10-07, about 120 commits
  behind); `pb/FM` at 3fe34165; the main checkout is clean. Pushing is Volkan's.
- **For the Mac**, `C:/D/git/wt/tmp/handoff-mac/`: `dtw-cpp-2026-10-11.bundle` (design-2.0 and pb/FM;
  `git fetch <bundle> design-2.0:design-2.0 pb/FM:pb/FM`) and `records-2026-10-11.zip` (the records outside git:
  `C:/D/git/wt/tmp/<X>/…` below is `tmp/<X>/…` in it, `.claude/baselines/` is `baselines/`).
- Windows: worktrees `wt/FM` (pb/FM), `wt/MV`, `wt/WM`; WSL `~/fm` (FM's kit, 29 GB) and `~/ms/data` (I4.arrow,
  P1.parquet, P1_f32.parquet; MM's kit needs I4 and P1); `ext4.vhdx` 142.5 GiB on C: (78 GB in use) until Volkan
  compacts it (elevated, WSL stopped).

## Done

- SW merged `557cd889` (10-10). MS measured 10-10; branch M timed 10-11: M1–M9 CONFIRMED, the mapped route 0.447 of
  the stream's cold time (0.453 in M2); record `tmp/MS/2026-10-10-ms-measurement.md`.
- FM: `c72adf3b`, the fix round `3fe34165` (Calinski–Harabasz fixed); Volkan accepted the six remaining assembly FAILs.
  NOT TIMED: the gated chain waited 19:34–22:14 (60–97 % CPU, Volkan's jobs), ran no part, and was stopped; record
  `tmp/FM/2026-10-10-fm-float32-matrix.md`.
- MM (mapped series): design rounds 1–3 and two adversarial reviews in `tmp/MM/` (design.md, kit-plan.md, hetero.md;
  review-round3.md: 1 BLOCKER and 12 should-fixes, open).
- Volkan's five edits committed at his request (`6490bc61`). Disk, approved: ≈ 63 GB deleted in WSL, 20 GB in `wt/tmp`.

## Verified by me

- Branch M: validity (a)–(f) in all 20 runs; M3 straddled and is scored on M2 (inside). `report_M` scores every band on
  M2 once M2 exists: there M9 warm is 1.127 < 1.15 (recorded, not scored).
- FM at 3fe34165 (logs): ctest clang 95, cl 94, CUDA 94; GCC's suite, identity, F1, F4 and M2 RESULT lines, exit 0;
  pytest 899/19/0; matlab_suite 142/142; docs PASS; CLI 156 and 65 files identical. 3fe34165 changes `scores.cpp` only.
- MM's premises: `fast_clara.cpp:242-248` (one check); `Dataset.m:11-12` (MATLAB refuses IPC); nanoarrow 0.8.0's
  `codecs.c` (ZSTD, LZ4F under their macros) and `nanoarrow_ipc.h` (SetEndianness required); GC's record line 30 (FP64
  L1 hashes equal on CPU and GPU); the core already fetches through CPM (`Dependencies.cmake:11-16`, `104-110`).
- graphify (`graphify-out/`, gitignored): C++ caller recall 0.10, precision 0.94; use `git grep`.

## Reported by agents, unverified

- FM's Windows identity runs (D4 0 mismatches, clang and cl; F and R parts): only GCC's RESULT lines opened by me.
- MM's round-3 review: its premises opened by me, not its 13 findings' line citations.

## Decisions (Volkan, question tool, 10-11; DECISIONS §3)

- MM "Design unit, with check": the reader "Vendor nanoarrow IPC"; compressed IPC in every language, piece by piece;
  the check fused into CLARA's pass "Inside MM"; codecs "vendor if version issues, else CPM" (the orchestrator: CPM);
  Q6 "Confirm" (beyond the budget only CLARA streams, in every language); Q9 "Keep, document IPC".
- HG "Own unit" after MM (Q8, hybrid determinism, open). FM "Accept all, time FM". His edits "You commit all five".

## Next steps

1. Sync the Mac: Volkan pushes, or fetches from the bundle; unzip the records.
2. macOS: clang-macos gates at design-2.0; the Metal tests (never compiled since W4d, W13a, W10, GC; MAP §3); then
   pb/FM's `metal_dtw.mm` change (never compiled).
3. MM round 4 (design only): fix review-round3.md. Its finding 7 reopens Q9 (an explicit `ram_limit` in Python's and
   MATLAB's cluster() would be silently unhonoured for CSV and Parquet): put it to Volkan, then ask him to approve the
   design and bands. Code after FM's merge.
4. FM's timing on the Windows machine (its bands are that machine's): `bash C:/D/git/wt/tmp/timing_fm.sh main`
   (≈ 3.5 h, quiet), `driver.py report`, Volkan's ETW (`profile_fm_etw.cmd`), the merge question, an integrator. Or
   Volkan re-registers the bands for the Mac.
5. HG after MM (`tmp/MM/hetero.md`). Candidates: `import dtwcpp` loads sklearn, scipy and pandas (+870 MB of OpenBLAS
   pools at 24 threads, FM's M3); SW's incremental refresh and panel prefetch; BF's 8-column tasks.

## Open questions (Volkan)

- FM: time on Windows, or re-register for the Mac? MM's finding 7: route all of cluster() through C++ `run`?
- Compacting `ext4.vhdx` (≈ 65 GiB back). Earlier: the GitHub history purge; the Mac's restore; RA's notice; which
  `.claude` files to untrack.

## Status honesty

- Timed 10-11: branch M (M, M2). FM not timed; MM design only, nothing built.
- Not run: macOS since RL, Linux release archives, ARC.
