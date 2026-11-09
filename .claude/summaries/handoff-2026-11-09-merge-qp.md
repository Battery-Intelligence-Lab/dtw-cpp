# Handoff — 2026-11-09 (Windows), interim — the finish plan, MM merged, the quality pass QP in flight

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it. Tree clean but for this file's commit.
  MM merged at c4bcefc2 (`--no-ff`, parents 53b9062a and pb/MM 0cc84a15; the merged code equals 0cc84a15's byte for byte).
- Worktrees: `C:/D/git/wt/MM` (pb/MM, merged; its eight build trees serve HG and the timing window), `wt/FM` and `wt/TL`
  (merged units, to remove on Volkan's yes), `wt/QP1..QP4` (the QP agents', branches `pb/QP1..4` from c4bcefc2).
  `C:/D/git/dtw-cpp/build/bin/dtwc_cl.exe` (clang, the merged tree) is QP's identity-gate base binary: do not rebuild it
  until QP's gates are done. WSL: `~/mm/merge_gcc` holds the merged code with trees `build` (GCC fast), `build-strict`,
  `build-arrow`, `build-release` (the last three were building when this was written).

## Done

- The finish plan `plans/2026-11-09-finish-2.0.md` (12314c49): the size trajectory, how MM grew from one unit into six,
  what verification cost, what the charter still lacks, the process cuts. PLAN.md rewritten to the remaining list.
- Volkan's four answers (53b9062a; DECISIONS §3 11-09): MM merges on correctness plus one sanity run; all of HG in 2.0;
  the bounded quality pass; the process cuts adopted.
- MM merged c4bcefc2 and pushed; PLAN item 1 ☑ (e7d9b94b).
- QP prepared and launched: briefs `C:/D/git/wt/tmp/QP/{qp_common,brief_qp1..5,brief_qp_review}.md`, the identity gate
  `C:/D/git/wt/tmp/QP/kit/cli_identity.sh` (27 CLI cases byte for byte, three of them with the run report and `-v` output
  compared with volatile rows dropped). QP1 (io, stream), QP2 (cli, api), QP3 (`Problem`), QP4 (bindings) running on
  Sonnet xhigh since about 03:55Z; QP5 (MM's tests under the 09-30 rule; clang-tidy) not launched (four-agent cap).

## Verified by me

- MM's gates on 0cc84a15, run by my chains (logs `orch_merge_*` in the session scratchpad): no-llfio 103/103, no-cpuinfo
  103/103, MEX 104/104 with `matlab_suite` (60 s), CUDA 102/102, WSL GCC 102/102 (`~/mm/merge_gcc`), fresh-venv pytest
  933 passed 19 skipped (dtwcpp 2.0.0rc1, highspy 1.15.1, pyarrow 26.0.1); clang 103/103 and Arrow 106/106 at S10b on the
  same tree. On the merged tree in the main checkout: clang 103/103, `check_docs` PASS, `check_pins` 0 failures,
  `generate_docs --check` current, Arrow 106/106 with `test_io_readers` listed; `git status` clean after the gates.
- The sanity run (Volkan's "one sanity run"): I served the kit's smoke of 01:06–01:08 on the same binaries (MM 0795c37f,
  base 4fa08e96 from aaeb7821), not a fresh run: single runs, no gate, no typeperf, the busy list (Volkan's MATLAB) noted;
  so "load recorded" is partial. MM/base wall ratios A: 0.37, 0.43, 0.37, 0.60; B: 0.90, 0.95, 0.96; no gross slowdown
  (`baselines/2026-11-09-mm-merge-sanity.md`, [inferred]).
- The identity kit: base against itself 111 files 0 differ; the merged main binary against the MM worktree's 131 files 0
  differ; the report and `-v` cases added after the advisor's note, 153 files over 30 cases, 0 differ.
- The release list on the merged code in WSL (`~/mm/merge_gcc`, GCC 13.3; my scripts `orch_strict_inner.sh`,
  `orch_linux_arrow_inner.sh`, `orch_linux_archive_inner.sh`): `cpp_conformance` strict 1/1; Arrow ON through the pyarrow
  23.0.1 shim 105/105 with `test_io_readers` listed, `dtwc_cl` linking libarrow, libparquet, libgomp; the Linux archive
  (`cpack`, 340 members, no libgomp, the notice's lines 41-51 say the system's runtime is used) and its smoke: FAILED on the
  merged code because `scripts/smoke_release_archive.py` forced the mapped matrix with `--ram-limit 1`, which the memory plan
  refuses; fixed to `--checkpoint <output>` (6de4a741, re-run OK). CI's three archive jobs would have failed the same way.
  Tree-wide clang-tidy (the repo's `.clang-tidy`, 31 own TUs): 4 warnings in all, handed to QP2 and QP3.
- Earlier today: see `handoff-2026-11-09-s10.md` (S9–S10b, the kit prep).

- QP1, QP2 and QP3 accepted (their reports, transcribed from the hand-backs because the Write tool refused the agents the
  path: `C:/D/git/wt/tmp/QP/report_qp1..3.md`): I read every code diff (the helpers keep each text byte for byte; QP3's
  `in_range` is the NaN-correct form of clang-tidy's De Morgan proposal) and ran the identity gate on each branch's binary,
  153 files over 30 cases, 0 differ; QP1's Arrow tree's five reader tests pass. Gathered on `pb/QP` (worktree
  `C:/D/git/wt/QP`, merges cf2acd99, 55bd2ce3, 4cda6d12, no conflict); its own build, ctest, identity and docs gates were
  running (`orch_qp_gather.log`). The cross-file items the agents reported are listed in `C:/D/git/wt/tmp/QP/gather_todo.md`
  for my commits on `pb/QP` after the gather: dead `row_group_rows` and `RunReport::notes`, the `check` parameter that
  shadows `check()`, one `made_name` for the seven `"series_" + to_string` sites, one text-file writer for the report's two
  sites, the "Metoid" misspelling, a `reserve` tidy flags.
- The merged tree's MSVC strict build FAILED: `dtwc/extern/nanoarrow/flatcc.c:3033` C2099 under `/fp:strict` (a float
  constant initialiser), because MM's `dtwc_third_party` links `project_options` and so took our FP flags. Fixed and merged:
  c4271bbf (`cmake/ProjectOptions.cmake`: the flags on `COMPILE_LANGUAGE:CXX` only; our code has no C), proven by cl strict
  `cpp_conformance` 1/1, clang fast 103/103 with the identity gate 0 differ against the merged binary, and WSL GCC rebuilt
  (112 steps) 102/102 (my chains `orch_rel_*`; the first WSL run logged into the repo root by a `cd` mistake, LESSONS).
- QP4 accepted (`report_qp4.md`; diffs read; its pytest 933/19, MEX 104 + matlab_suite 151, G10 324/324) and merged into
  `pb/QP` (eda1481b). My two commits there: ffe8939e, the cross-file findings (`made_name`, `write_text`, dead
  `row_group_rows` and `RunReport::notes`, the `piece` parameter, the MEX's one `dtwc:invalidInput`, "Medoid", a `reserve`;
  gates clang 103/103, identity 153 files 0 differ, check_docs PASS, MEX tree 104/104), and e0d4d63d, the MATLAB help fix QP4
  found (31 files' Doxygen headers moved below the help blocks; `tests/matlab/test_help_text.m` 3/3 on the fixed tree,
  3/3 FAIL on the base). `pb/QP` tree e135797a.

## Reported by agents, unverified

- QP5 (MM's tests under the 09-30 rule) and DOC (the docs' and CHANGELOG's final read, `brief_doc.md`, branch `pb/DOC`
  from 93ef5434) were running. Logs: `C:/D/git/wt/tmp/QP/logs/`. The agents' own gate counts in the four accepted reports
  (GCC, Arrow, CUDA were not theirs to run) are theirs until the QP merge's full gates.

## Decisions

- Volkan 11-09 (question tool): "Correctness gates + one sanity run"; "All of HG in 2.0" (against the finish plan's 2.1
  recommendation); "Bounded pass as written"; "Adopt the cuts". Recorded in DECISIONS §3.
- Mine: the sanity run served from the 01:06 smoke rather than a fresh run (the machine was no quieter); QP's identity gate
  extended to the report and `-v` text; wt/FM and wt/TL left in place (the deletion rule and rule 11 conflict).

## Next steps

1. QP (PLAN item 2): QP5 as QP1–4 (diff read, identity run, report transcribed), merged into `pb/QP`; the review
   (`brief_qp_review.md`, one Sonnet agent on `pb/QP`, with `gather_todo.md`'s done list); the full gates once on `pb/QP` in its
   worktree (clang, Arrow, no-llfio, no-cpuinfo, MEX + matlab_suite, CUDA, WSL GCC, fresh-venv pytest); merge `--no-ff` into
   design-2.0; push. Then `pb/DOC` the same way (docs gates only).
2. The release list (item 3): left are `cpp_conformance` under MSVC strict (`C:/D/git/wt/MM/build-cl-strict` was building;
   a waiter runs `ctest -R cpp_conformance` there and logs `orch_cl_strict_ctest.log`), F43 (the question batch) and the
   docs' and CHANGELOG's final read (one Sonnet agent after QP). HG's draft plan: `C:/D/git/wt/tmp/HG/hg_plan.md`.
3. The question batch at the QP merge (PLAN, after "Done before the tag"): HG on the design's prediction with advisory
   measurement, or a quiet window; the CI MEX job's llfio; the merged worktrees.
4. HG (item 5): draft its unit plan from `C:/D/git/wt/tmp/MM/hetero.md` myself (no planner agent).

## Open questions

- Those three for Volkan above. Still his: the history purge, the G6-on-TADPole acceptance, Metal on the Mac, Intel Macs.

## Status honesty

- Run by me today: everything under "Verified by me". Not run: any timing (Volkan's MATLAB sessions are open), Metal, the
  Linux wheel (last at S8b, the agent's), the MATLAB desktop. QP's agents, three WSL builds, cl's strict build and the
  tree-wide clang-tidy were in flight; a session restart kills agents silently (LESSONS): check `git log pb/QP1..4`,
  the summary logs and the transcript times before relaunching anything. LESSONS.md is at its 300-line cap: the lesson of
  this session (a unit that absorbs new rulings is re-asked "is this still one unit"; gates at a merge, not a commit) is in
  the finish plan §2 and §5.
