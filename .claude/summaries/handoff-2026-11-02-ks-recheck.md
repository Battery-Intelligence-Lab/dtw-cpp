# Handoff — 2026-11-03 (Windows) — RB1–RB3c committed and checked; RR1a running; RR1b, merge, S5–S10, HG next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's start 446d8b99 = origin. Tree clean.
- `pb/MM` dff4590c (RB3c, tree 63d0a1b7) on 31f7158d (RB3b) on 92c04967 (RB3) on 014123c4 (RB2b) on f9e731f9 (RB2) on a6c1ab2f (RB1) on 1234030a (KS4; KS =
  539d7d7e, 5422bb11, 4741d753, 1234030a: superseded history); local only, worktree `C:/D/git/wt/MM`. Trees there: `build` (clang), `build-arrow`,
  `build-nollfio`, `build-mex`, `build-cuda`, GCC in WSL `~/mm/rb3c`. Work files `C:/D/git/wt/tmp/MM/rb{1,2,2b,3,3b,3c}/`, kit `kit/rb*_*`, logs `logs/rb*_summary.log`,
  `review_rb2/`, `review_rb3/` (reports, fixtures, cap tools `cap_run.py`, `cap_wsl.sh`), `review_rb3b/`, `rr_claims_check.md`.
  Briefs: `brief_rr1a.md` (its agent runs on pb/MM since 11-03; nothing else touches the worktree), `brief_rr1b.md` (fill `<RR1a-TIP>`; its `RunStage` is RB3's),
  `brief_s5_draft.md` (fill `<RR1b-TIP>`). After a restart check the agent's transcript mtime before relaunching.
- Disk 11-03: Volkan approved, I deleted `review_ks1/2/3_r2`, most of `review_ks3_r1`, `conf_strict`, WSL `ks3r2_*` and `conf_strict` (about 3.9 GB); an empty `review_ks3_r1/src` stays.

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4; `cpp_conformance` `strict` PASS on
  design-2.0 292a9bd7, GCC 13.3 and cl 19.50 (`baselines/2026-11-02-conformance-strict.md`).
- RB1 a6c1ab2f kernels back to 819e8703's text + closed-list release; RB2 f9e731f9 the plan (budget, reserve, one loading decision); RB2b 014123c4 stream refusals and the
  methods' fixed terms priced; RB3 92c04967 `RunStage`, `run()` catches `std::bad_alloc` outside its scopes, `core::peak_memory()`, `test_run_out_of_memory`.
- RB3b 31f7158d: readers' and `--dist-matrix`'s `bad_alloc` pass through; nanoarrow/Arrow/llfio ENOMEM become `bad_alloc`; `ways_out()` follows the run; "the plan holds H bytes";
  a failed new `.dtwm` is removed; Linux peak = `VmHWM`. RB3c dff4590c: TADPole's two regions carry a failure out (per-thread slots, `run_openmp`'s pattern); MIP is told only
  `--method clara`; a checkpoint resume no float32; no-llfio no IPC item; Python's Parquet route raises `MemoryError`; the floor wording.

## Verified by me

- RB1-RB3b: write-trees, diffs, clang `ctest -j1` (99, then 100), sweeps, my scan, B1, peaks, G6 logs, real-binary runs of the CSV and 30,000-series cases (typed reports, true advice).
- RB3c: HEAD tree = 63d0a1b7 (clean); TADPole diff read; clang `ctest -j1` 100/100 at dff4590c (`scratchpad/orch_rb3c_ctest.log`); the MIP case under Job caps of 91 MB and 100 MB
  (Windows, RB3c's `dtwc_cl`): `... ways out: --method clara.`; the TADPole `--dc` repro under WSL `ulimit -v` 66000, 67000 and 68000: typed (RB3b: rc 134). Advisory timing, no builds
  or MATLAB running: `w2000x20` TADPole `--dc 4.0 --band 3`, 12 alternating runs, default threads: medians 160 ms (RB3b) and 153 ms (RB3c).

## Reported by agents, unverified

- RB3c (Opus, 641k tokens): gates on the committed tree 63d0a1b7: clang 100, GCC 99, Arrow 103, no-llfio 100, MEX 101 (`matlab_suite` 142/142), CUDA 99, pytest 916/19, docs gates; 256 of
  256 identical-source objects byte-equal; TADPole labels, medoids, cost and every `TADPoleStats` counter equal on 5 fixtures at 1, 4, 24 threads; mutations bite (try/catch removed ->
  0xC0000409; B4's handler removed -> `IOError`; B3 reverted -> the no-llfio test fails); Windows repro typed at 8 of 9 caps (a stack-page crash at the top cap: the margin case).
- RB3b (Opus 761k) and its reviews: as in the earlier handoff (all four re-check should-fixes closed by RB3c); residual: LZ4F's refusal and `decode_batch`'s released-array path stay `IOError`.
- Earlier, open: 64 MiB cgroup (pam maps a 100 MB matrix that thrashed); least limits rise at 24 threads (the reserve); llfio vs no-llfio least limits differ ~10x; class (d) (wheel Soft-DTW gradient).

## Decisions

- Volkan 11-02: GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12; KS3's loops "Fix first (a KS5 unit)"; the RAM limit (DECISIONS §3); "Approve RB and RR as written". 11-03: "cpuinfo,
  optional"; `--no-report` "No text at all"; counters "Later, own commit" (RR2); TADPole's regions "Fix it before the 2.0 tag"; "MM and HG both in 2.0"; deleting the KS scratch "Delete these".
- My calls, not asked: D1 keep `TeamCap`, D3 streams keep the requested threads, D4 `auto` changes only through a guard, D5 `std::bad_alloc` becomes `InvalidInput`; RB3b's pass-through;
  S8/S9 serial; class (d) accepted at RB1. **G6 on TADPole's pair loops (RB3c), open to Volkan's overrule:** FALSIFIED against my brief's strict band (clang/MSVC ABI: the catch funclet resumes at
  the loop latch, so about 6 more stack loads and 4 stores per pair in the density loops; GCC LTO and the wheel reshape the region loops; the inner LB/UB loops are unchanged). Accepted: PLAN's
  hot-loop rule names the kernel cell, matrix get/set, `dist_by_ind`, swap and assignment loops, not TADPole's pair loops; each pair is a DTW call or an O(L) bound; timing above. The row-function
  shape (fewer reloads and calls than the base, one stack store more in Stage 2) is on disk: `rb3c/variants/v3/dtwc/algorithms/tadpole.cpp`. Awaiting him: the history purge.

## Next steps

1. PLAN MM RR1a (running, `brief_rr1a.md`: cpuinfo step zero on the six trees + wheel + a cpuinfo-OFF tree; STOP and report if a download or a tree fails; `core/machine`, `print_settings`):
   verify (write-tree, diffs, clang `ctest -j1`, the `-v` header against an independent source, the pin's sha, G6).
2. PLAN MM RR1b (`brief_rr1b.md`; `rr_claims_check.md` found 11 of the plan's 104 code claims false: the fill split only for PAM, Lloyd, full-sample CLARA; `Result::save` writes the report last;
   the mean silhouette only under `-v`; `Problem.hpp` untouched); RR2 (counters) later; then merge design-2.0 into pb/MM (`--no-ff --no-commit`), S5-S9 serial, S10, then HG.
3. S9: map `std::bad_alloc` in the MEX gateway (`dtwc:internal` today). PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4, 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read at S10; the Mac needs its own G6 and a Metal build (memory `project-metal-on-mac-checklist`); whether the reserve's rise at 24 threads is what
  Volkan wants; whether `available_memory()` should see `RLIMIT_AS`; the TADPole G6 call above; residuals: Gurobi's `catch (...)` (`mip_Gurobi.cpp:147`), `--dist-matrix`'s unpriced transient
  (98 MB for 16 MB), `api.cpp:141`; the history purge.

## Status honesty

- Run by me: clang `ctest -j1` at RB1, RB2, RB2b, RB3, RB3b, RB3c and both KS commits, sweeps, the probe, the docs gates, G6 log reads, my RB2 scan, B1, the peaks, the real-binary runs above, one
  advisory timing. Not run by me: the Arrow, no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, mutations. Agents so far: Fable audit, R1, R2, KS3, R1', R2', conformance, KS4, RB and RR
  planners, RB1 (905k tokens), RB2 (814k + 233k) and its reviewer (309k), RB2b, RB3 (617k) and two reviewers (263k, 351k), RB3b (761k) and its re-check (362k), RB3c (641k), the RR claims check (439k).
