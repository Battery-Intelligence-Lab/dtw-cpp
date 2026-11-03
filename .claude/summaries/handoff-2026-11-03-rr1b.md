# Handoff — 2026-11-03 (Windows) — RB1–RR1b3 committed and checked, design-2.0 synced; S5–S10, HG next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21); at the session's start 446d8b99 = origin. Tree clean.
- `pb/MM` ac25ffdc (RR1b3, tree 31185669) on 7e061212 (merge of design-2.0 cbd67d9e, tree bcf4d42c) on 755b74e9 (RR1b2, tree 6bda7737) on 43a4802b (RR1b, tree b9bd5cf2) on 06f7d00d
  (RR1a, tree 6f5fc20f) on dff4590c (RB3c) on 31f7158d (RB3b) on 92c04967 (RB3) on 014123c4 (RB2b) on f9e731f9 (RB2) on a6c1ab2f (RB1) on 1234030a (KS4; KS = 539d7d7e, 5422bb11,
  4741d753, 1234030a: superseded history); local only, worktree `C:/D/git/wt/MM`. Trees there: `build` (clang), `build-arrow`, `build-nollfio`, `build-mex`, `build-cuda`,
  `build-nocpuinfo`, `build-mexci`, GCC in WSL `~/mm/rr1b*`, `~/mm/sync`, the wheel `~/mm/sync_whl`. Work files `C:/D/git/wt/tmp/MM/rb*/`, `rr1a/`, `rr1b/` (g6, identity, reports, after,
  tools), `sync/` (the merge's WSL check), kit `kit/rb*_*`, `kit/rr1a_*`, `kit/rr1b_*`, logs `logs/rb*_summary.log`, `rr1a_summary.log`, `rr1b_summary.log`, `review_rb2/`, `review_rb3/`
  (fixtures, cap tools), `rr_claims_check.md`. S5 is RUNNING (an Opus implementer on pb/MM from `brief_s5.md`, base ac25ffdc, launched 11-03 ~16:10Z; its log
  `logs/s5_summary.log`; nothing else touches the worktree). After a restart check its transcript mtime and that log before relaunching.
- Disk 11-03: Volkan approved, I deleted `review_ks1/2/3_r2`, most of `review_ks3_r1`, `conf_strict`, WSL `ks3r2_*`, `conf_strict` (about 3.9 GB); an empty `review_ks3_r1/src` stays.

## Done

- 11-01's stall was the weekly usage limit (memory `project-weekly-limit-stall`); status report v2 + Fable audit; KS3, KS4; `cpp_conformance` `strict` PASS on design-2.0 292a9bd7, GCC 13.3, cl 19.50.
- RB1 a6c1ab2f kernels back to 819e8703's text + closed-list release; RB2 f9e731f9 the plan (budget, reserve, one loading decision); RB2b 014123c4; RB3 92c04967 `RunStage`, `run()` catches
  `std::bad_alloc` outside its scopes, `core::peak_memory()`; RB3b 31f7158d readers' and library OOM statuses pass as `bad_alloc`; RB3c dff4590c TADPole's two regions carry a failure out.
- RR1a 06f7d00d: `dtwc_cl -v` names the machine (`core/machine.{hpp,cpp}`, pytorch/cpuinfo optional `DTWC_ENABLE_CPUINFO`, pinned e9320ce3 by archive SHA-256).
- RR1b 43a4802b: every run writes `<name>_report.txt` last (`cli/report.{hpp,cpp}`, eight sections, ≤ 100 columns, ASCII); `fill` a stage of its own for PAM, Lloyd and full-sample CLARA only;
  `run_with` writes a failure report up to the failing stage (not for `check`, not without `-o`) and rethrows its own error; `Result::report()`, `save` writes it last; `--no-report` = no
  file, no `Report:` line (the 48th key); `core::cpu_seconds()`; `detail/result_files.hpp` (`WriteTimes`); `Problem.hpp` untouched; the RR1a leftovers (cl FMA, `const FARPROC`) fixed.
- RR1b2 755b74e9 (mine): the report's command row printed CLI11's TOML literals (`'x'`, `'''...'''` for over 100 characters with a backslash), which cmd.exe hands to dtwc_cl with the quotes;
  now double quotes. cli.md: a failure in `check` writes no report; the row wraps at 100 columns.
- Sync 7e061212 (mine): design-2.0 merged into pb/MM (571c40f9 the Linux wheel's TLS, 59202f68 `verify_results`, 671ae09f the user pages; one docs conflict, kept RB2's "data cap").
- RR1b3 ac25ffdc (mine, tests only): `Result::save` writes the report text last; a failure in `check` writes none; the fill stage exists for PAM, Lloyd and full-sample CLARA only.

## Verified by me

- RB1-RR1a as in the earlier handoff (write-trees, diffs, clang `ctest -j1` 99 to 101, real-binary runs, G6 log reads, headers on five trees, the cpuinfo pin and licence).
- RR1b: HEAD tree = b9bd5cf2, clean; report/run/result-files/Problem_IO/machine/api/config/CMake/test/doc diffs read; clang `ctest -j1` 103/103 at 43a4802b and again at 755b74e9 (3 skips:
  cuda, metal x2); the agent's `identity.py` rerun on my build against RR1a's saved binary: PASS, 5 fixtures (PAM in RAM and mapped, Lloyd, full CLARA: `-v` output and result files equal with
  and without `--no-report`; hierarchical N=2100 refuses with no fill); 10 failing invocations with and without `--no-report`: same exit code and stderr, a report only past `check`; 13
  invalid-parameter cases, RR1a's binary against mine: same exit and stderr; four real reports read (PAM CSV, streamed CLARA on LZ4 Arrow IPC under 20M, a 1-byte limit refused, a checkpoint
  save failing after the results); Arrow tree: 8 report/F8/F13/IPC/io_readers tests pass; `check_pins` cmake=21 failures=0, `check_docs` PASS, `generate_docs --check` current; key counts 48 in
  three places; no `skip` in any output. G6: `changed.txt` = `git diff --name-only 06f7d00d HEAD` (40 paths), the saved subject source = `git archive b9bd5cf2` (`diff -rq` clean),
  `rr1b_g6_identical.py` rerun: 268/268 identical-source objects byte-equal, `Problem.cpp`'s too; 13 post-ThinLTO modules differ, all text-equal or moved copies but run.cpp's `execute` (grew:
  the length scan and the size counts; not a hot loop). RR1b2: the old form fails in cmd.exe (exit 1, the file cannot be opened); the new one, from cmd.exe (a batch file), PowerShell and sh,
  repeats the run (four result files SHA-256-equal to a reference run); the new test fails with the requoting off.
- Sync merge: tree bcf4d42c gated before the commit: clang `ctest -j1` 103/103 (3 skips), `check_pins` failures=0, `check_docs` PASS (475 flags), `generate_docs --check` current; WSL GCC 13.3
  from `git archive`: build 0 warning lines, ctest 102/102; the Linux wheel built from it imports in WSL WITHOUT `GLIBC_TUNABLES` (the RR1a and RR1b smokes needed it) and clusters (labels 40).
  RR1b3: each of three reverted mutations (save's report block, the `check` exclusion, hierarchical in the fill-first list) fails `test_run_report`; clang 103/103, 146 assertions in 5 cases.

## Reported by agents, unverified

- RR1b (Opus, 838k tokens): gates on b9bd5cf2: GCC 102, Arrow 106, no-llfio 103, MEX 104 (`matlab_suite` 142/142), CUDA 102, cpuinfo-OFF 103, pytest 916/19; 13 mutations caught (M10 only in WSL);
  GCC LTO kernel list moved=0 grew=0; wheel module: 0 cpuinfo, `machine_facts`, `cpu_seconds` symbols (so THIRD_PARTY_LICENSES is unchanged); `cpu_seconds()` reaches `Problem_IO.cpp` by a
  function pointer so the bindings do not link `machine.cpp`; round 1 (25f262b5) replaced for a GCC false-positive warning and a "method auto" in a failure report.
- Earlier, open: 64 MiB cgroup (pam maps a 100 MB matrix that thrashed); least limits rise at 24 threads; llfio vs no-llfio least limits differ ~10x; class (d) (wheel Soft-DTW gradient).

## Decisions

- Volkan 11-02: GCC loops "Read post-LTO code first"; `cpp_conformance` 1e-12; KS3's loops "Fix first (a KS5 unit)"; the RAM limit (DECISIONS §3); "Approve RB and RR as written". 11-03: "cpuinfo,
  optional"; `--no-report` "No text at all"; counters "Later, own commit" (RR2); TADPole's regions "Fix it before the 2.0 tag"; "MM and HG both in 2.0"; deleting the KS scratch "Delete these".
- My calls, not asked: D1 keep `TeamCap`, D3 streams keep the requested threads, D4 `auto` changes only through a guard, D5 `std::bad_alloc` becomes `InvalidInput`; RB3b's pass-through;
  S8/S9 serial; class (d) accepted at RB1; **G6 on TADPole's pair loops (RB3c), open to Volkan's overrule** (the catch funclet resumes at the loop latch; row-function shape on disk,
  `rb3c/variants/v3/dtwc/algorithms/tadpole.cpp`); RR1a's GCC-LTO moves of outer driver loops accepted on the hot-loop rule; RR1b2 a new commit, not an amend (no history rewrite).
  Awaiting Volkan: the history purge.

## Next steps

1. S5 (GPU view interim; running; verify it when it hands back: write-tree, diff, clang `ctest -j1`, the CUDA tree's G8 equal-to-CPU runs and the report it writes, G6), then S6 CLI, S7 `run` for every `cluster()` (then THIRD_PARTY_LICENSES's cpuinfo row, "the module links none",
   changes if a binding reaches `machine_facts`), S8 Python, S9 MATLAB (map `std::bad_alloc` in the MEX gateway), S10 (G6 vs the MM base, class (d), Volkan's call), then HG. RR2 (counters)
   later, its own commit.
2. PLAN "Before the 2.0 tag": L2b, Linux archives, F43, design.md §7 risks 4 and 7, FM's E1; conformance redone if MM merges first.

## Open questions

- Class (d) and the driver loops' `TeamCap` read at S10; the Mac needs its own G6, a Metal build (memory `project-metal-on-mac-checklist`) and cpuinfo on Apple Silicon (`test_machine` assumes a
  known CPU model); whether the reserve's rise at 24 threads is wanted; whether `available_memory()` should see `RLIMIT_AS`; the TADPole G6 call; residuals: Gurobi's `catch (...)`
  (`mip_Gurobi.cpp:147`), `--dist-matrix`'s unpriced transient, `api.cpp:141`; the `OS:` line reads `Windows 10.0.26200` on Windows 11; `-v`'s `Build:` line lists `dtwc_core`'s libraries,
  the report's version row adds Arrow and YAML; the report's wrap may cut a UTF-8 sequence of a long non-ASCII path; a report that cannot be written after a successful run fails the run (IOError).

## Status honesty

- Run by me: clang `ctest -j1` at every MM commit to RR1b3, sweeps, the docs gates, G6 log reads and the RR1a and RR1b identical-objects reruns, real-binary runs, the RR1b identity rerun, the
  error-parity sweeps, the Arrow tree's 8 tests, RR1b2's shell checks and mutation, RR1b3's mutations, the merge's WSL GCC + wheel check. Not run by me: the no-llfio, MEX, CUDA and full Arrow
  gates, pytest, any assembly, the agents' mutations; RR1b2, the merge and RR1b3 were not re-run on those trees (the merge moves no Windows input; the rest is one block and tests). Agents so far: Fable audit, R1, R2, KS3, R1', R2', conformance, KS4, RB and RR planners, RB1, RB2 and its reviewer,
  RB2b, RB3 and two reviewers, RB3b and its re-check, RB3c, the RR claims check, RR1a (643k), RR1b (838k).
