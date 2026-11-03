# Handoff — 2026-11-03 (Windows) — RB1–RB3b committed; RB3b re-check running; RB3c (TADPole), RR1 next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's start 446d8b99 = origin. Tree clean.
- `pb/MM` 31f7158d (RB3b, tree 92c64d37) on 92c04967 (RB3, d26261d8) on 014123c4 (RB2b) on f9e731f9 (RB2) on a6c1ab2f (RB1) on 1234030a (KS4; KS =
  539d7d7e, 5422bb11, 4741d753, 1234030a: superseded history); local only, worktree `C:/D/git/wt/MM`. Trees there: `build` (clang), `build-arrow`,
  `build-nollfio`, `build-mex`, `build-cuda`, GCC in WSL `~/mm/rb3b`. Work files `C:/D/git/wt/tmp/MM/rb{1,2,2b,3,3b}/`, kit `kit/rb*_*`, logs
  `logs/rb{1,2,2b,3,3b}_summary.log`, `review_rb2/`, `review_rb3/` (reports, fixtures, cap tools `cap_run.py`, `cap_wsl.sh`), `review_rb3b/` (the re-check writes there).
  Briefs: `brief_rb3b_recheck.md` (running; touches nothing in the worktree), `brief_rb3c.md` (fill `<RB3b-TIP>`), `brief_rr1.md` (fill `<RB3c-TIP>`; its
  `RunStage` is RB3's), `brief_s5_draft.md`. After a restart check the agent's transcript mtime before relaunching.
- RR plan `rr_plan.md` (code claims not yet checked). Disk 11-03: Volkan approved, I deleted `review_ks1/2/3_r2`, most of `review_ks3_r1`, `conf_strict`, WSL
  `ks3r2_*` and `conf_strict` (about 3.9 GB); an empty `review_ks3_r1/src` stays (a process holds it).

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4; `cpp_conformance` `strict` PASS on
  design-2.0 292a9bd7, GCC 13.3 and cl 19.50 (`baselines/2026-11-02-conformance-strict.md`).
- RB1 a6c1ab2f kernels back to 819e8703's text + closed-list release; RB2 f9e731f9 the plan (budget, reserve, one loading decision); RB2b 014123c4 stream
  refusals and the methods' fixed terms priced; RB3 92c04967 `RunStage`, `run()` catches `std::bad_alloc` outside its scopes, `core::peak_memory()`,
  `test_run_out_of_memory`, TADPole envelopes priced.
- RB3b 31f7158d: a reader's and `--dist-matrix`'s `bad_alloc` pass through; nanoarrow/Arrow/llfio ENOMEM become `bad_alloc`; the ways out follow the run
  (`ways_out()`); "the plan holds H bytes"; a failed new `.dtwm` is removed; Linux peak = `VmHWM`; pins for `prunes`, envelopes, OneBatch's clamp.

## Verified by me

- RB1-RB2b: as earlier (write-trees, diffs, clang `ctest -j1` 99/99, sweeps, my scan, B1, peaks, G6 logs). RB3: tree d26261d8, diffs, `ctest` 100/100, real binary.
- RB3b: HEAD tree = 92c64d37 (clean); diffs read (run.cpp, io, core); clang `ctest -j1` 100/100 at 31f7158d (`scratchpad/orch_rb3b_ctest.log`). Real binary (WSL GCC
  `~/mm/rb3b`): the 127 MB CSV under `ulimit -v 200000` gives `read stage: out of memory; plan: none yet; ... ways out: --dtype float32, or the file converted to
  Arrow IPC (dtwc-convert) ...`; 30,000 one-step series PAM under `ulimit -v 2500000` gives the cluster report with "the plan holds 3600360000 bytes in RAM" and ways
  out `--matrix-dtype float32, --method clara, or a lower --ram-limit with an output directory (-o)`. `dtwc-convert` ships today (`pyproject.toml`).

## Reported by agents, unverified

- RB3b (Opus, 761k tokens): gates on the committed tree 92c64d37: clang 100, GCC 99, Arrow 103, no-llfio 100, MEX 101 (`matlab_suite` 142/142), CUDA 99, pytest 915/19,
  docs gates; G6 and 7 mutations on 67e6e8d0 (a test cast apart); E: a child's `ru_maxrss` 641 MB vs `VmHWM` 10.6 MB (parent held 600 MiB); every reviewer repro typed but LZ4F's
  refusal and `decode_batch`'s released-array path; ways tried: `--matrix-dtype float32` and `--method clara` ran, the mapped route ran on Windows. Residuals: the stream
  items say "a record batch" (a Parquet floor holds sampled series too: RR1's brief), the CSV test checks "plan: none yet", `ways_out` is 41 lines.
- RB3's reviews: code (Sonnet) no blocker; routes (Opus, 96 capped runs): the plan clause = `-v`'s in all 96; B1 TADPole `--dc` terminates (RB3c); S1-S3 fixed in RB3b.
- Earlier, open: 64 MiB cgroup (pam maps a 100 MB matrix that thrashed); least limits rise at 24 threads (the reserve); llfio vs no-llfio least limits differ ~10x;
  class (d) (wheel Soft-DTW gradient) carried to S10.

## Decisions

- Volkan 11-02: GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12; KS3's loops "Fix first (a KS5 unit)"; the RAM limit (DECISIONS §3: ~10 % margin, "separate several GB
  for the algorithm, then just load the data", report a run that does not fit, "Nice log in plain text at the end"); "Approve RB and RR as written". 11-03: "cpuinfo, optional";
  `--no-report` "No text at all"; counters "Later, own commit" (RR2). Later 11-03 (question tool): TADPole's regions "Fix it before the 2.0 tag" (RB3c); "MM and HG both in 2.0";
  deleting the KS scratch "Delete these".
- My calls, not asked: D1 keep `TeamCap`, D3 streams keep the requested threads, D4 `auto` changes only through a guard, D5 `std::bad_alloc` becomes `InvalidInput`; RB3b's
  pass-through (the helper `rethrow_naming_the_file` is 2.0-born); S8/S9 serial; class (d) accepted at RB1. Awaiting him: the history purge.

## Next steps

1. PLAN MM RB3b: the re-check's report (the routes table re-run, each advised way out followed, `map()` cleanup, `ways_out()` branches) -> a fix commit if a finding stands.
2. PLAN "Before the 2.0 tag" / MM RB3c (`brief_rb3c.md`): TADPole's two regions carry a failure out (the `run_openmp` slot pattern), G6 on TADPole's loops.
3. PLAN MM RR1 (`brief_rr1.md`: cpuinfo step zero on the six trees; the fill split keeps digits and `-v` lines; `test_run_out_of_memory`'s word may change cluster -> fill; RB3b's
   leftovers), RR2 later; then merge design-2.0 into pb/MM (`--no-ff --no-commit`), fill `<RR1-TIP>` in `brief_s5_draft.md`, S5-S9 serial, S10, then HG (both in 2.0).
4. PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4, 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read at S10; the Mac needs its own G6 and a Metal build (memory `project-metal-on-mac-checklist`); whether the reserve's rise at 24
  threads is what Volkan wants; whether `available_memory()` should see `RLIMIT_AS`; residuals: Gurobi's `catch (...)` (`mip_Gurobi.cpp:147`), `--dist-matrix`'s unpriced transient
  (98 MB for 16 MB), `api.cpp:141`; the history purge.

## Status honesty

- Run by me: clang `ctest -j1` at RB1, RB2, RB2b, RB3, RB3b and both KS commits, sweeps, the probe, the docs gates, G6 log reads, my RB2 scan, B1, the peaks, the RB3/RB3b real-binary
  runs above. Not run by me: the Arrow, no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, any timing, mutations. Agents so far: Fable audit, R1, R2, KS3, R1', R2', conformance,
  KS4, RB and RR planners, RB1 (905k tokens), RB2 (814k + 233k), its reviewer (309k), RB2b, RB3 (617k) and its reviewers (263k, 351k), RB3b (761k), its re-check running.
