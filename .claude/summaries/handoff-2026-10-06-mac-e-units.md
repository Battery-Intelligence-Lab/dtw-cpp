# Handoff — 2026-10-06 (afternoon, Mac) — M1, W9e, W9c merged; L2b running; lrcore-without-HiGHS measured

## Base

Branch `design-2.0` on the Mac, base `33302535` (Volkan's push after the Windows session); HEAD = this file's commit.
Volkan pushes; the Windows machine then `git pull --no-rebase`. Read the Windows handoff `handoff-2026-10-06-w9b-and-
mac-merged.md` too (its next steps 1 TB and 5 follow-ups still stand). Tree clean after this commit.

## Done (each unit one `pb/<unit>` branch merged `--no-ff`, evidence on the main tree, worktree removed)

- Base check at `33302535` (`61903cc2`, `baselines/2026-10-06-macos-after-w9b.md`): build 189 steps zero warnings,
  ctest 95/95 + CUDA skip, conformance the one silhouette ulp (D-19), docs gates, pytest 970/12/0 (982 ids, as
  Windows), ASan+UBSan 94/94 no report (W9b's byte-mode readers under sanitizers for the first time).
- M1 `c0580948` (brief from Windows; Opus agent): one `dtwc::mip::build_p_median_model` (row-wise arrays; the triplets,
  the sort and solver_types.hpp go), linked HiGHS via `passModel`'s pointers, dtwc_cl's MIP outputs byte-identical;
  the wheel links no HiGHS (extension 5,034,320 → 1,203,728 bytes), `method="mip"` on it solves with highspy (`mip`
  extra, `highspy>=1.8`), SolverError names `"dtwcpp[mip]"` otherwise; the MEX links HiGHS statically (4.5 MB); CI
  MEX HiGHS ON + test_cluster_mip. Its adversarial review merged `bb35d337` (DTWC_REQUIRE_HIGHSPY fails a skipped
  highspy case in CI; the CI MEX job asserts test_cluster_mip ran; licences file; quoted extras; lrcore note).
- W9e `bf82dc1b` (brief from Windows; Opus): MATLAB on the core — `dtwc.cluster(data, k, Name, Value)` and
  `DTWClustering` set a dtwc::Config, apply() checks, Problem::cluster() runs; CamelCase keys are Python's words; the
  MEX links no run()/api (link map); text via the bound C++ reader, Parquet via parquetread, Arrow IPC refused;
  matlab_suite 140/139 → 142/141 by name; the 25 CLI runs byte-identical. ◐: Windows R2024b matlab_suite still to run
  (its brief stays in `plans/2026-10-06-briefs/`).
- W9c `834b7904` (brief written here `fed1f37d`; Opus): `hpc` crosses as one `job.toml` written by Python in dtwc_cl's
  config grammar after C++ apply() checked every value; `submit-job <rundir> [--gpu | --gpu-device <type>]`; one GPU
  table `_slurm/gpu_devices.txt` (a100 by gres type; a6000/l40s/h100 by `gpu_cc:` constraint [inferred]); `build
  --gpu-device <type>` (CUDA native, CPU portable); `smoke.slurm MODE=`; pytest 970/12 → 922/12 by id. The merge
  needed one fix (`f7c04420`: M1's test_mip used the helper W9c deleted); merged tree 926/11/0.
- Records: PLAN (M1 ☑, W9e ◐, W9c ☑, the Mac entry, the ARC and Windows rows under "Blocked"), DECISIONS §3 (six
  10-06 lines), LESSONS (one line: a wheel links nothing the user's packages ship), the contract's MATLAB method
  paragraph, `python-tests.yml` installs `[test,dev,io,mip]` + pandas (its pyarrow/pandas/sklearn cases skipped before).

## Running when this was written (the final version of this file says how they ended)

- L2b (brief `f196c57b`, Opus): `dtwc_core` / `dtwc_io` / the CLI; FILE_SET HEADERS per folder; bindings link the core.
  After its merge: the second short Mac pass (Metal tests, GPU CLI routes byte-identical).
- lrcore without HiGHS (Sonnet, measurement only): `baselines/2026-10-06-lrcore-root-without-highs-mac.md`.

## Verified by me (commands in the baselines named above)

Every merge: build zero warnings, `ctest -j1` 95/95 + 1 skip, codegen gate PASS, conformance the same ulp, the three
docs gates, pytest from a fresh venv (974/11/0 after M1 and W9e; 926/11/0 after W9c, 93 hpc ids removed by design),
matlab_suite 142/141/0/1 after W9e on the static-HiGHS MEX. Empty-series pairwise DTW returns max() (F48, documented;
confirmed through the binding; untouched).

## Reported by agents, unverified by me

- M1: model equivalence old/new for six (N, k) [its own third check]; the macOS crash of a user's highspy beside the
  HiGHS-linked extension (exit 139; cause inferred); highspy 1.5.3/1.7.x API gaps behind the `>=1.8` floor.
- W9e: Parquet series equal pyarrow's; the UTF-8 fix on Windows reasoned, not run; `squared_euclidean`+`gpu` fixed by
  construction. W9c: eight mutation checks bite; the ARC node tags `gpu_cc:8.6/8.9/9.0`; SIGILL risk of `-march=native`
  on mixed-CPU GPU nodes (hence `DTWC_NATIVE_CPU=OFF`).

## Decisions and open questions for Volkan

- M1 review finding 1: the wheel's `method="lrcore"` runs the subgradient root (Kelley needs linked HiGHS): accept
  (documented; numbers in the lrcore baseline) or keep HiGHS in the wheel? DECISIONS 10-06 M1-review line.
- W9c: `DTWC_NATIVE_CPU=OFF` for `--gpu-device` builds narrows the 09-30 "native CPU flags" ruling — confirm.
- `.claude/CLAUDE.md` line 71's Python gate should add `mip` to its extras (your file; not edited by agents or me).
- Trailers: agents' commits carry the model that wrote them (Opus/Sonnet); mine Fable (the harness attribution).
  DECISIONS 09-30 "no Fable" is about delegated tasks; not changed here.
- Still yours: the ARC leg (PLAN "Blocked"), TB on Windows, 20 stale September worktrees, `build-asan/`.

## Next steps

1. Windows: pull; re-run clang/CUDA trees, pytest (hpc transport and MIP route changed), matlab_suite R2024b for W9e
   (then delete its brief); TB when quiet. 2. Mac: L2b's merge + the second Mac pass (in progress here). 3. E: W9f
   (restores `read_distance_matrix`/`print_distance_matrix` contract rows). 4. G: W14a, WM, W14c.
5. Follow-ups noted by agents: Python names in-memory series after `skip_rows` by original ordinals (C++ renumbers);
   Python names Parquet rows by the first string column (C++ `series_<i>`); `load(Dataset, **options)` silently
   ignores options; `set_solver` prints before apply() raises; pre-staged folder/Parquet hpc input needs
   `parse_labels_csv` without 1..N names (fixed in W9c for files; folders untested); `test_conformance.m` not in
   matlab_suite; THIRD_PARTY_LICENSES has no MEX column (CI artefact only).

## Status honesty

Mac only (`build/`, `build-matlab/`, fresh-venv wheels; ASan at `33302535`). Not run here: CUDA, Linux, CI (five
workflow edits parsed only), Windows, ARC, Arrow-ON trees, Gurobi (mip_Gurobi.cpp untouched and uncompiled).
