# Handoff — 2026-11-02 (Windows) — KS3 committed (4741d753); its fresh re-checks decide whether KS closes

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's
  start 446d8b99 = origin. Tree clean.
- `pb/MM` 4741d753 (KS3, tree 61ec2e4f, parent 5422bb11), local only, worktree `C:/D/git/wt/MM`, clean. Nobody may rebuild
  `C:/D/git/wt/MM/build*` while the re-checks read them. The reviewers' own trees: `C:/D/git/wt/tmp/MM/review_ks3_r1/`
  (refusals, release; agent a6bef573) and `review_ks3_r2/` (loops, kit, post-LTO; agent a42b3bfb); claims for them:
  `C:/D/git/wt/tmp/MM/ks3_claims.md`. If a session restart killed them, check the transcripts' mtimes before relaunching.
- Disk trees stay until Volkan's yes: `review_ks1/fix` 0.08 GB, `review_ks2/rc` 0.65 GB (and the new review_ks3_*).

## Done

- Why 11-01 stalled: the weekly usage limit at 20:14Z (HTTP 429 `usage_limit_reached`, resets 07:00 London) killed the two
  KS re-check reviewers on their first call; nothing technical blocked (memory `project-weekly-limit-stall`).
- The status report corrected (v2) and audited by Fable 5.1 max (Volkan asked for it). PLAN, DECISIONS §3 (11-02 lines),
  the KS record `baselines/2026-11-01-ks-scratch.md` (disk). Fresh re-checks of 5422bb11: `review_ks1/recheck_report.md`,
  `review_ks2/recheck_report.md` (disk): R1's blockers only partly closed, silhouette loop reloads.
- KS3 (4741d753, implementer's report; its logs `C:/D/git/wt/tmp/MM/logs/ks3_summary.log`): `KernelTeam::release()` at the
  PAMs' end and at a re-plan, storage with one thread's scratch, IPC routing by the longest series' need (walk bound, then
  `IpcFile::longest()`), the PAMs' team beside their own term, silhouette form fc, probe 10 univariate kernels + 23-family gate.

## Verified by me

- 4741d753: parent 5422bb11, tree 61ec2e4f, 22 files +505/-213; I opened the diffs of `run.cpp`, `fast_clara_stream.hpp`,
  `thread_scratch.{hpp,cpp}` (release reaches every thread past a TeamCap: `team_of_every_thread()`), `arrow_ipc.{hpp,cpp}`
  (`longest_bound`, `longest`), `scores.cpp`, the tests, MAP §7, CHANGELOG, cli.md.
- My clang tree (`C:/D/git/wt/MM/build`) at 4741d753, 11-02 ~15:10Z: `ninja` no work, `ctest -j1` 99/99, skips CUDA and
  Metal x2, 46 s (`logs/ks3_orch_ctest.log`). `build-nollfio` also no work.
- R1's refusals, my own scripts (llfio and no-llfio after; 5422bb11 and 819e8703 for contrast): before shows them (late6k
  refused at 2.0M, 2.5M, in RAM at 5,668,107; long1 at 6,327,625 and in RAM at 12.58M, 16.6M; every100 at 2,500,124;
  s200x100 mapped / no-llfio exit 1). After, one cost per fixture: late6k 609392.5 (least 1,868,413, in RAM from 5,668,108),
  long1 1389997.75 (least 6,327,625, in RAM from 16,600,208), every100 5334612.625 (least 2,043,825; before 2,500,125),
  s200x100 3244.2483770000013 in RAM at 800K/1M, one2000 195712.5 (least 2,227,321 llfio, 3,042,210 no-llfio: equal to
  5422bb11's), long4 + `--dist-matrix` 45465.497809000066 at 10M and 100M. Grid scans: no refusal above an accepted limit
  (before: 15 on late6k, 21 on long1). Scripts: scratchpad `orch_sweep.sh`, `orch_mono.sh`, `orch_least.sh`.

## Reported by agents, unverified

- KS3's gates on 61ec2e4f (clang 99, Arrow 102, no-llfio 99, MEX 100, CUDA 98, GCC 98, pytest 915/19/0, docs, probe
  `inner_loops=1616 missing=none calls=0`, 25 of 25 mutations), its G6 reads (silhouette gather loop equal, clang post-pass
  +1 store), the memory table (`logs/ks3_measure.log`), and four deviations it left for Volkan: clang's post-pass +1
  store; clang ThinLTO `dtw_dispatch` column loops +1-3 loads a column in six functions (byte-equal from 5422bb11); GCC LTO
  Soft-DTW f32 handler cell loop +1 load +1 store; the wheel's `soft_dtw_gradient` cell loops +2 loads, +2 stack loads, +2 stores.
- KS3 deviated from brief item 3: the walk's bound alone failed on one single-batch file, so it routes by the exact longest.
- R1/R2's earlier reports (least limits by arithmetic, GCC per-TU table, kit blind spot): superseded by 4741d753's re-checks.

## Decisions

- Volkan 11-02 (question tool): B1 "Release at phase end (Recommended)"; GCC loops "Read post-LTO code first (Recommended)";
  cross-compiler check: "I never said 17 significant figures, ask me what it is" -> "1e-12 is fine I guess, as long as it
  doesn't change the clustering" (`cpp_conformance` as the test stands). In the chat: "check it with Fable 5.1 max" (one audit).
- My calls, not asked: release also at the stream's re-plan; S8 and S9 serial; merge design-2.0 into pb/MM after KS closes;
  fresh reviewers at default effort; accepting KS3's routing by the exact longest pending R1'. Awaiting Volkan, carried:
  whether MM and HG ship in 2.0, the history purge, disk deletions; the four deviations above once R2' classifies them.

## Next steps

1. PLAN KS: read R1' and R2' (open every cited line, reproduce each blocker); a blocker -> KS4 brief; none -> KS closes
   (the deviations, as R2' classifies them, to Volkan through the question tool).
2. Merge design-2.0 into pb/MM (`--no-ff --no-commit`, clang build and ctest), fill `<KS-TIP>` in `brief_s5_draft.md`, then
   PLAN MM S5 (CUDA tree here, Metal on the Mac), S6, S7 (reworks `run.cpp`'s planner, which KS changed), S8, S9 serial, S10
   (add a kernel band; G6 against the MM base with the fixed kit before the merge).
3. PLAN "Before the 2.0 tag": `cpp_conformance` strict legs (GCC in WSL, MSVC; the test as it stands), held until the
   re-checks stop building; L2b; Linux archives; F43; design.md §7 risks 4 and 7 if MM ships; FM's E1 (Volkan runs `profile_fm_etw.cmd`).

## Open questions

- Parquet's routing still prices the mean series length (`run.cpp` 339-349; R1' tests it); a GCC "no call in a DP loop" gate
  (unruled); `reserve_pair` costs 2.6-5.7 % of a pair at L = 8 in GCC -fPIC shared objects; `IpcFile::longest()` reads
  every record batch the bound cannot decide (R1' times it). `gh` unauthenticated: CI unread.

## Status honesty

- Run by me: clang `ctest -j1` (99/99) and the limit sweeps and scans above. Not run by me: the Arrow, no-llfio, MEX, CUDA and
  GCC gates, pytest, any assembly, any timing, the mutations, the `strict` conformance legs.
- Agents this session: the Fable audit (190 k tokens), R1 (349 k), R2 (530 k), the KS3 implementer (763 k), R1' and R2'
  (running); the advisor tool twice. Cost-state of 11-01: $519.49.
