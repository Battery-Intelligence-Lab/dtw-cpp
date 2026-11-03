# Handoff — 2026-11-03 (Windows) — RB1, RB2, RB2b, RB3 committed; RB3 reviewed; RB3b running; RR1 next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's
  start 446d8b99 = origin. Tree clean.
- `pb/MM` 92c04967 (RB3, tree d26261d8) on 014123c4 (RB2b, 3d72dc1b) on f9e731f9 (RB2) on a6c1ab2f (RB1) on 1234030a (KS4; KS =
  539d7d7e, 5422bb11, 4741d753, 1234030a: superseded history); local only, worktree `C:/D/git/wt/MM`. Trees there: `build` (clang),
  `build-arrow`, `build-nollfio`, `build-mex`, `build-cuda`, GCC in WSL `~/mm/rb3`. Work files `C:/D/git/wt/tmp/MM/rb{1,2,2b,3}/`, kit
  `kit/rb*_*`, logs `logs/rb{1,2,2b,3,3b}_summary.log`, `review_rb2/`, `review_rb3/` (reports `report_routes.md`, `report_code.md`; fixtures; cap tools). Briefs:
  `brief_rb3b.md` (its agent runs on pb/MM since 11-03, nothing else touches that worktree), then `brief_rr1.md` (fill `<RB3b-TIP>`; its `RunStage` is
  RB3's). After a restart check the agent's transcript mtime before relaunching.
- RR plan `rr_plan.md` (code claims not yet checked). Disk until Volkan's yes: `C:/D/git/wt/tmp/MM/review_ks*` (about 2 GB), WSL
  `~/mm/ks3r2_*` and `~/conf_strict`, `C:/D/git/wt/tmp/conf_strict/`.

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4;
  `cpp_conformance` `strict` PASS on design-2.0 292a9bd7, GCC 13.3 and cl 19.50 (`baselines/2026-11-02-conformance-strict.md`).
- RB1 a6c1ab2f: kernels back to 819e8703's text; `thread_scratch.cpp` holds the closed list; `run()` holds `ScratchRelease`; `TeamCap` kept (D1).
- RB2 f9e731f9: `run.cpp` plans once (`thread_bytes`, `working_set`, `fit_threads`, `plan_memory`), `core::memory_budget` (90 % of
  `available_memory()`, which sees a cgroup v2 limit), `resolve_method(..., pam_matrix_holds)`, `IpcFile::longest_steps()`, tests, docs.
- RB2b 014123c4: streamed refusals say what they need and that their `--ram-limit` is the data cap; the methods' fixed terms priced.
- RB3 92c04967: `RunStage`; `run()` (both overloads) catches `std::bad_alloc` outside `ScratchRelease` and `execute` and throws
  `InvalidInput("<stage> stage: out of memory; plan: ...; the process peaked at P bytes resident; ways out: ...")`; `core::peak_memory()`;
  `test_run_out_of_memory` (Job object / `RLIMIT_AS`); TADPole's envelopes priced; OneBatch's batch size pinned; docs. Psapi lesson in LESSONS.

## Verified by me

- RB1, RB2, RB2b: as in the earlier handoff (write-trees, diffs, clang `ctest -j1` 99/99, sweeps, my scan, B1 reproduced, peaks, G6 logs).
- RB3: HEAD tree = d26261d8, worktree clean; diffs of `run.cpp`, `distance_matrix.*`, both tests, docs read; clang `ctest -j1` 100/100 at 92c04967
  (`scratchpad/orch_rb3_ctest.log`); TADPole `prunes` = `bounds_valid` (`tadpole.cpp:68-80`, `prune = true` default). Real binary (WSL GCC, `ulimit -v
  2500000`, 30,000 one-step series, PAM, `--ram-limit 1T`, `-o ''`): the typed text at stage cluster, exit 1. Its ways out: `-m clara` runs, `--dtype float32`
  still fails, `--matrix-dtype float32` runs. A 127 MB CSV under `ulimit -v 200000`: `Error: load: failed to read '...': std::bad_alloc` (an `IOError`).

## Reported by agents, unverified

- RB1-RB3 gates: Arrow 103, no-llfio 100, MEX 101 (`matlab_suite` 142/142), CUDA 99, GCC 99; pytest 915/19; `cpp_conformance` in each; mutations M1-M3
  caught; placement probe 2.9 MB (committed placement) vs 111.8 MB (catch inside `execute`); G6: 239/239 identical-source objects equal, run/distance_matrix/
  arrow_ipc differ, post-ThinLTO text equal; GCC LTO: CLI11 x4, `vector<string>`, `main`, the fill's `vector<Verdict>` inlined again (RB2's shape); wheel: 3 inliner moves.
- The first RB3 gate crashed `test_arrow_ipc`: `psapi.h` before llfio renamed its inline `GetPerformanceInfo` (ODR); included after llfio now.
- RB3's two fresh reviews (read-only). Code (Sonnet): no blocker; `mip_Gurobi.cpp:147` and `run.cpp:717` rename a `bad_alloc` (opened: true); the envelope and
  OneBatch tests restate the code; `ru_maxrss` survives `exec`. Routes (Opus, 96 capped runs, Windows Job + WSL `ulimit -v`): the plan clause equals `-v`'s in all
  96; B1 TADPole with `--dc` terminates (`tadpole.cpp:210/:270`: 0xC0000409, rc 134); S1 stream and IO statuses (nanoarrow 12, Arrow OutOfMemory, ENOMEM) and the
  readers are `IOError`s, a failed `.dtwm` sizing leaves a stale zero file; S2 the ways out are one static list (`--dtype` frees no matrix, clara offered to
  clara); S3 the peak counts mapped pages (65-72 MB against a 20 MB budget). Their lines for S1, `mip_Gurobi`, `api.cpp:141` opened by me.
- Earlier, still open: 64 MiB cgroup (auto never refuses; pam maps a 100 MB matrix that thrashed); least limits rise at 24 threads (the reserve); llfio vs
  no-llfio least limits differ ~10x; class (d) (wheel Soft-DTW gradient) carried to S10.

## Decisions

- Volkan 11-02 (question tool): GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12; KS3's loops "Fix first (a KS5 unit)"; on the RAM limit a free text
  (DECISIONS §3: relax it, ~10 % margin, "separate several GB for the algorithm, then just load the data", report a run that does not fit, "Nice log in plain
  text at the end"); "Approve RB and RR as written". 11-03: "cpuinfo, optional"; `--no-report` "No text at all"; counters "Later, own commit" (RR2).
- My calls, not asked: D1 keep `TeamCap`, D3 streams keep the requested threads, D4 `auto` changes only through a guard, D5 `std::bad_alloc` becomes
  `InvalidInput`; S8/S9 serial; class (d) accepted at RB1. RB3b (brief `brief_rb3b.md`): a reader's and `--dist-matrix`'s `bad_alloc` passes through (the helper
  `rethrow_naming_the_file` is 2.0-born, absent at v1.0.0), library OOM statuses become `bad_alloc`, the ways out follow the run, pins, `VmHWM` if the floor is
  real; the checkpoint save stays deferred. NOT in RB3b, carried to Volkan: TADPole's two bare `omp parallel` regions (a few KB at the margin; algorithm loops = G6).
- Awaiting him: whether MM and HG ship in 2.0; the history purge; disk deletions.

## Next steps

1. PLAN MM RB3b (running): verify (write-tree, diffs, clang `ctest -j1`, the real-binary repros in `review_rb3/` on WSL and with the Job-object cap, the ways out
   each tried), a narrow re-check of the text; then ask Volkan about TADPole's regions with the other open questions.
2. PLAN MM RR1 (`brief_rr1.md`: cpuinfo step zero on the six trees; the fill split keeps digits and `-v` lines; `test_run_out_of_memory`'s word may
   change cluster -> fill), RR2 later; then merge design-2.0 into pb/MM (`--no-ff --no-commit`), fill `<RR1-TIP>` in `brief_s5_draft.md`, S5-S9 serial, S10.
3. PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4, 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read at S10; the Mac needs its own G6 and a Metal build (memory `project-metal-on-mac-checklist`); whether the
  reserve's rise in least limits at 24 threads is what Volkan wants; whether `available_memory()` should see `RLIMIT_AS`; TADPole's two regions (document
  the margin crash, or wrap them: G6); residuals: Gurobi's `catch (...)`, `--dist-matrix`'s unpriced transient (98 MB for a 16 MB matrix), `api.cpp:141`.

## Status honesty

- Run by me: clang `ctest -j1` at RB1, RB2, RB2b, RB3 and both KS commits, sweeps, the probe, the docs gates, G6 log reads, my RB2 scan, B1, the peaks,
  the RB3 real-binary runs above. Not run by me: the Arrow, no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, any timing, mutations. Agents so far:
  Fable audit, R1, R2, KS3, R1', R2', conformance, KS4, RB and RR planners, RB1 (905k tokens), RB2 (814k + 233k), its reviewer (309k), RB2b, RB3 (617k), its reviewers (263k, 351k), RB3b running.
