# Handoff — 2026-11-03 (Windows) — RB1–RR1a committed and checked; RR1b running; merge, S5–S10, HG next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's start 446d8b99 = origin. Tree clean.
- `pb/MM` 06f7d00d (RR1a, tree 6f5fc20f) on dff4590c (RB3c, tree 63d0a1b7) on 31f7158d (RB3b) on 92c04967 (RB3) on 014123c4 (RB2b) on f9e731f9 (RB2) on a6c1ab2f (RB1) on 1234030a
  (KS4; KS = 539d7d7e, 5422bb11, 4741d753, 1234030a: superseded history); local only, worktree `C:/D/git/wt/MM`. Trees there: `build` (clang), `build-arrow`, `build-nollfio`, `build-mex`,
  `build-cuda`, `build-nocpuinfo`, `build-mexci` (the CI MEX job's recipe), GCC in WSL `~/mm/rr1a`. Work files `C:/D/git/wt/tmp/MM/rb{1,2,2b,3,3b,3c}/`, `rr1a/`, kit `kit/rb*_*`,
  `kit/rr1a_*`, logs `logs/rb*_summary.log`, `logs/rr1a_summary.log` (+ `rr1a_mutations.log`), `review_rb2/`, `review_rb3/` (reports, fixtures, cap tools `cap_run.py`, `cap_wsl.sh`),
  `review_rb3b/`, `rr_claims_check.md`. Briefs: `brief_rr1b.md` (base 06f7d00d; its agent runs on pb/MM since 11-03, nothing else touches the worktree), `brief_s5_draft.md` (fill `<RR1b-TIP>`).
  After a restart check the agent's transcript mtime before relaunching.
- Disk 11-03: Volkan approved, I deleted `review_ks1/2/3_r2`, most of `review_ks3_r1`, `conf_strict`, WSL `ks3r2_*` and `conf_strict` (about 3.9 GB); an empty `review_ks3_r1/src` stays.

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4; `cpp_conformance` `strict` PASS on
  design-2.0 292a9bd7, GCC 13.3 and cl 19.50 (`baselines/2026-11-02-conformance-strict.md`).
- RB1 a6c1ab2f kernels back to 819e8703's text + closed-list release; RB2 f9e731f9 the plan (budget, reserve, one loading decision); RB2b 014123c4 stream refusals and the
  methods' fixed terms priced; RB3 92c04967 `RunStage`, `run()` catches `std::bad_alloc` outside its scopes, `core::peak_memory()`, `test_run_out_of_memory`.
- RB3b 31f7158d: readers' and `--dist-matrix`'s `bad_alloc` pass through; nanoarrow/Arrow/llfio ENOMEM become `bad_alloc`; `ways_out()` follows the run; "the plan holds H bytes";
  a failed new `.dtwm` is removed; Linux peak = `VmHWM`. RB3c dff4590c: TADPole's two regions carry a failure out (per-thread slots, `run_openmp`'s pattern); MIP is told only
  `--method clara`; a checkpoint resume no float32; no-llfio no IPC item; Python's Parquet route raises `MemoryError`; the floor wording.
- RR1a 06f7d00d: `dtwc_cl -v` names the machine (host, OS, CPU, cores, SIMD of CPU and build, RAM, a GPU run's GPU, compiler, libraries; `unknown` where not known); pytorch/cpuinfo
  optional (`DTWC_ENABLE_CPUINFO`, pinned e9320ce3 by archive SHA-256, static, PIC); `core/machine.{hpp,cpp}`, `total_memory()`, `test_machine`; the Linux CPU/VmRSS/VmPeak block is gone.

## Verified by me

- RB1-RB3b: write-trees, diffs, clang `ctest -j1` (99, then 100), sweeps, my scan, B1, peaks, G6 logs, real-binary runs of the CSV and 30,000-series cases (typed reports, true advice).
- RB3c: tree 63d0a1b7; TADPole diff read; clang `ctest -j1` 100/100; MIP under Job caps of 91 and 100 MB: `... ways out: --method clara.`; the TADPole `--dc` repro under WSL `ulimit -v`
  66000, 67000, 68000: typed (RB3b: rc 134). Advisory timing (no builds, no MATLAB), `w2000x20` TADPole `--dc 4.0 --band 3`, 12 alternating runs: medians 160 ms (RB3b), 153 ms (RB3c).
- RR1a: HEAD tree = 6f5fc20f, worktree clean; `machine.{hpp,cpp}`, the test, the TU, CMake, workflow and doc diffs read; clang `ctest -j1` 101/101 (`scratchpad/orch_rr1a_ctest.log`; the list
  = RB3c's + `test_machine`, same 3 skips); the real `dtwc_cl -v` header against independent sources: Windows (CIM: host, 24/24 cores, RAM 127.46 vs 127.5 GiB; registry: CPU; `ver`:
  build 26200), WSL (/proc, os-release, uname: all equal), the cpuinfo-OFF tree (CPU, Cores `unknown`), no-llfio (RAM `unknown`), CUDA (GPU line; nvidia-smi 20475 MiB vs 20474 MB); a fresh
  download of the pinned archive has the pinned SHA-256 and upstream `main` is still e9320ce3; the LICENSE text equals THIRD_PARTY_LICENSES's; `check_pins.py` cmake=21 failures=0,
  `check_docs.py --cli` PASS, `generate_docs.py --check` current; G6: `changed.txt` = `git diff --name-only dff4590c HEAD` (20 paths), `rr1a_g6_identical.py` rerun on the saved objects:
  cpuinfo OFF 258/258 byte-equal, ON 257/258 (`arrow_ipc.cpp.obj`; its inline-asm srcloc cause is the agent's IR dump, which I did not re-derive).

## Reported by agents, unverified

- RR1a (Opus, 643k tokens): gates on 6f5fc20f: GCC 100, Arrow 104, no-llfio 101, MEX 102 (`matlab_suite` 142/142; the CI MEX recipe 118/118), CUDA 100, cpuinfo-OFF 101, pytest 916/19;
  9 of 9 mutations caught; cpuinfo built in six trees, the wheel recipe and OFF (15 C files Windows, 20 Linux, no warnings); clang ThinLTO post-link function text equal with `.llvm.<hash>`
  masked; GCC LTO (cpuinfo ON or OFF) and the wheel move outer driver loops in unchanged functions (dtwBanded_mv 80 -> 79, fill block driver 296 -> 295, `reduced_cost_fixing` +1 load/store,
  `write_result_files` CSV loops) while every DTW cell loop, the lanes, the fill pair loop, swap and assignment are SAME; the wheel's `import` fails in WSL ("static TLS block") for RB3c's too.
- RB3c (Opus, 641k): clang 100, GCC 99, Arrow 103, no-llfio 100, MEX 101, CUDA 99, pytest 916/19; 256/256 identical-source objects byte-equal; TADPole results equal on 5 fixtures at 1, 4, 24
  threads; mutations bite; Windows repro typed at 8 of 9 caps (a stack-page crash at the top: the margin case). RB3b and its re-check: all four should-fixes closed by RB3c.
- Earlier, open: 64 MiB cgroup (pam maps a 100 MB matrix that thrashed); least limits rise at 24 threads (the reserve); llfio vs no-llfio least limits differ ~10x; class (d) (wheel Soft-DTW gradient).

## Decisions

- Volkan 11-02: GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12; KS3's loops "Fix first (a KS5 unit)"; the RAM limit (DECISIONS §3); "Approve RB and RR as written". 11-03: "cpuinfo,
  optional"; `--no-report` "No text at all"; counters "Later, own commit" (RR2); TADPole's regions "Fix it before the 2.0 tag"; "MM and HG both in 2.0"; deleting the KS scratch "Delete these".
- My calls, not asked: D1 keep `TeamCap`, D3 streams keep the requested threads, D4 `auto` changes only through a guard, D5 `std::bad_alloc` becomes `InvalidInput`; RB3b's pass-through;
  S8/S9 serial; class (d) accepted at RB1. **G6 on TADPole's pair loops (RB3c), open to Volkan's overrule:** FALSIFIED against my brief's strict band (clang/MSVC ABI: the catch funclet resumes
  at the loop latch, about 6 more stack loads and 4 stores per pair in the density loops); accepted: PLAN's hot-loop rule names the kernel cell, matrix get/set, `dist_by_ind`, swap and
  assignment loops, each pair is a DTW call or an O(L) bound, timing above; the row-function shape is on disk (`rb3c/variants/v3/dtwc/algorithms/tadpole.cpp`). **RR1a's GCC-LTO moves of
  outer driver loops** accepted on the same rule (the cause is the new code's partitioning; same class with cpuinfo OFF). Three RR1a leftovers went into RR1b's brief: cl's `build_simd`
  lacks FMA (cl defines no `__FMA__`), the `Build:` line lists `dtwc_core`'s libraries only (no Arrow, YAML), `const FARPROC` trips `misc-misplaced-const`. Awaiting Volkan: the history purge.

## Next steps

1. PLAN MM RR1b (running, `brief_rr1b.md`, base 06f7d00d): verify (write-tree, diffs, clang `ctest -j1`, real-binary reports: PAM on a CSV, streamed CLARA, a refused limit, a failing checkpoint
   save; `--no-report` prints no text; the other outputs byte for byte; the fill split's identity and its refusal cases; G6). `rr_claims_check.md` found 11 of the plan's 104 code claims false.
2. PLAN MM RR2 (counters) later; merge design-2.0 into pb/MM (`--no-ff --no-commit`, clang build + ctest); fill `<RR1b-TIP>` in `brief_s5_draft.md`; S5-S9 serial; S10; then HG.
3. S7/S8: THIRD_PARTY_LICENSES's cpuinfo row ("the module links none") changes when `report` reaches a binding. S9: map `std::bad_alloc` in the MEX gateway (`dtwc:internal` today).
   PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4, 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read at S10; the Mac needs its own G6, a Metal build (memory `project-metal-on-mac-checklist`) and now cpuinfo on Apple Silicon (the macOS CI jobs,
  wheel and archive have it ON; `test_machine` assumes a known CPU model); whether the reserve's rise at 24 threads is wanted; whether `available_memory()` should see `RLIMIT_AS`; the TADPole
  G6 call; residuals: Gurobi's `catch (...)` (`mip_Gurobi.cpp:147`), `--dist-matrix`'s unpriced transient (98 MB for 16 MB), `api.cpp:141`; the `OS:` line reads `Windows 10.0.26200` on Windows
  11 (the NT version, as `ver` prints it); the history purge.

## Status honesty

- Run by me: clang `ctest -j1` at RB1, RB2, RB2b, RB3, RB3b, RB3c, RR1a and both KS commits, sweeps, the probe, the docs gates, G6 log reads and the RR1a identical-objects rerun, the real-binary
  runs above, one advisory timing, RR1a's headers on five trees, its pin and licence checks. Not run by me: the Arrow, no-llfio, MEX, CUDA and GCC gates, pytest, any assembly, mutations. Agents so far:
  Fable audit, R1, R2, KS3, R1', R2', conformance, KS4, RB and RR planners, RB1 (905k tokens), RB2 (814k + 233k) and its reviewer (309k), RB2b, RB3 (617k) and two reviewers (263k, 351k), RB3b
  (761k) and its re-check (362k), RB3c (641k), the RR claims check (439k), RR1a (643k).
