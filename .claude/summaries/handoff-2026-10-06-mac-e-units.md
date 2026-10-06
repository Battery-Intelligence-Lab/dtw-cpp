# Handoff — 2026-10-06 (evening, Mac) — M1, W9e, W9c, L2b merged; lrcore-without-HiGHS measured; second Mac pass done

## Base

Branch `design-2.0` on the Mac, base `33302535` (Volkan's push after the Windows session); HEAD = this file's commit.
Volkan pushes; the Windows machine then `git pull --no-rebase`. Read the Windows handoff `handoff-2026-10-06-w9b-and-
mac-merged.md` too (its next steps 1 TB and 5 follow-ups still stand). Tree clean after this commit.

## Done (each unit one `pb/<unit>` branch merged `--no-ff`, evidence on the main tree, worktree removed)

- Base check at `33302535` (`61903cc2`, `baselines/2026-10-06-macos-after-w9b.md`): build zero warnings, ctest 95/95
  + CUDA skip, conformance the one silhouette ulp (D-19), docs gates, pytest 970/12/0 (982 ids, as Windows),
  ASan+UBSan 94/94 no report (W9b's byte-mode readers under sanitizers for the first time).
- M1 `c0580948` (brief from Windows; Opus): one `dtwc::mip::build_p_median_model` (row-wise arrays; triplets, sort and
  solver_types.hpp go), linked HiGHS via `passModel`'s pointers, dtwc_cl's MIP outputs byte-identical; the wheel links
  no HiGHS (extension 5,034,320 → 1,203,728 bytes), `method="mip"` solves with highspy (`mip` extra, `>=1.8`), a
  missing highspy is SolverError naming `"dtwcpp[mip]"`; the MEX links HiGHS statically; CI MEX HiGHS ON. Its review
  merged `bb35d337` (DTWC_REQUIRE_HIGHSPY fails a skipped highspy case in CI; the CI MEX job asserts test_cluster_mip
  ran; licences; quoted extras; the lrcore note).
- W9e `bf82dc1b` (brief from Windows; Opus): MATLAB on the core — `dtwc.cluster(data, k, Name, Value)` and
  `DTWClustering` set a dtwc::Config, apply() checks, Problem::cluster() runs; CamelCase keys are Python's words; the
  MEX links no run()/api; text via the bound C++ reader, Parquet via parquetread, Arrow IPC refused; matlab_suite
  140/139 → 142/141 by name; the 25 CLI runs byte-identical. ◐ until Windows R2024b runs matlab_suite (its brief stays).
- W9c `834b7904` (brief `fed1f37d`; Opus, 17 commits with its review): `hpc` crosses as one `job.toml` written by
  Python in dtwc_cl's config grammar after C++ apply() checked every value; `submit-job <rundir> [--gpu |
  --gpu-device <type>]`; one GPU table `_slurm/gpu_devices.txt` (a100 by gres type; a6000/l40s/h100 by `gpu_cc:`
  [inferred]); `build --gpu-device` (CUDA native, CPU portable); `smoke.slurm MODE=`; pytest 970/12 → 922/12 by id.
  The merge needed `f7c04420` (M1's test_mip used the helper W9c deleted); merged tree 926/11/0.
- L2b `15b289be` (brief `f196c57b`; Opus, 7 commits with its review): `dtwc_core` / `dtwc_cli` / `dtwc_io` (the last in a
  build with Arrow) behind an INTERFACE `dtwc++`; every folder lists its headers in a FILE_SET (40 unlisted ones
  listed); the core reads text only, `dtwc::run` dispatches Parquet/Arrow IPC to `io::read_arrow`; nanoarrow stays in
  the core; `fast_clara` refuses a stream request (review). Compile commands identical after the file sets; the MEX
  and the extension byte-identical after the split; with a pyarrow-25 Arrow shim the MEX and the wheel link no Arrow.
- lrcore without HiGHS measured (`3cb49bc8`, `baselines/2026-10-06-lrcore-root-without-highs-mac.md`, Sonnet): same
  answers everywhere; the subgradient root certifies more noisy-DTW roots than Kelley; Kelley wins only on a line
  metric (43–230×). Recommendation: accept the HiGHS-free wheel (DECISIONS); Volkan rules.
- Second short Mac pass after L2b (DECISIONS 10-05): ctest 95 with the Metal tests; the 12 GPU CLI routes 48/48 files
  byte-identical to the pre-L2b binary (L2b record).
- Records: PLAN (M1, W9c, L1/L2 ☑; W9e ◐; the Mac entry; ARC and Windows rows under "Blocked"), DECISIONS §3 (nine
  10-06 lines), LESSONS, MAP §3, the contract's MATLAB paragraph; CI installs `[test,dev,io,mip]` + pandas, the wheel smoke asserts `gpu_devices.txt`.

## Verified by me (commands in the baselines named above)

Every merge: build zero warnings, `ctest -j1` 95/95 + 1 skip, codegen gate PASS, conformance the same ulp, the three
docs gates, pytest from a fresh venv (974/11/0 after M1 and W9e; 926/11/0 after W9c and L2b, 93 hpc ids removed by
design, guard on), matlab_suite 142/141/0/1 after W9e and L2b. F48 (empty-series DTW returns max()) confirmed, untouched.

## Reported by agents, unverified by me

M1: model equivalence old/new for six (N, k); the macOS crash of a user's highspy beside the HiGHS-linked extension
(cause inferred). W9e: Parquet series equal pyarrow's; the UTF-8 fix on Windows reasoned, not run. W9c: eight
mutation checks bite; ARC node tags; SIGILL risk of `-march=native` on mixed-CPU GPU nodes. L2b: the Arrow-shim ctest
and its 16 Arrow CLI runs; `test_io_readers`'s streamed Result::save case fails at base in that shim tree (pyarrow
25; Windows's 23 passes). lrcore: 194 run pairs deterministic; several registered bands FALSIFIED.

## Decisions and open questions for Volkan

- Accept the HiGHS-free wheel's subgradient lrcore root (numbers above)? W9c's `DTWC_NATIVE_CPU=OFF` for
  `--gpu-device` builds narrows the 09-30 "native CPU flags" ruling — confirm. W9b's `_NotSeries` (Windows handoff).
- `.claude/CLAUDE.md:71`'s Python gate should add `mip` to its extras (your file; not edited by agents or me).
- Trailers: agents' commits name the model that wrote them (Opus/Sonnet), mine Fable (harness attribution); DECISIONS
  09-30 "no Fable" is about delegated tasks. Still yours: the ARC leg (PLAN "Blocked"), TB on Windows, 20 stale
  September worktrees, `build-asan/`, whether the MEX ships (THIRD_PARTY_LICENSES has no MEX column).

## Next steps

1. Windows: pull; re-run clang/CUDA/Arrow trees (`build/arrow-pyarrow-23`: the L2b Arrow-ON proof, commands in its
   record), pytest, matlab_suite R2024b for W9e (then delete its brief); TB when quiet. 2. E: W9f (restores the
   `read_distance_matrix` / `print_distance_matrix` contract rows). 3. G: W14a, WM, W14c. 4. Candidates: OpenMP on
   `mip-solvers` (the LR phase is serial; 3.6–5.6× at N ≥ 800 in a scratch build); `lr-core.md:231,245` and
   `lagrangian_root.cpp:585` overstate Kelley; Python names in-memory series after `skip_rows` by original ordinals
   (C++ renumbers) and Parquet rows by the first string column (C++ `series_<i>`); `load(Dataset, **options)` silently
   ignores options; `set_solver` prints before apply() raises; `test_conformance.m` is not in matlab_suite; HiGHS
   caches `BUILD_SHARED_LIBS=ON` so a re-configured tree builds Catch2 shared.

## Status honesty

Mac only (`build/`, `build-matlab/`, fresh-venv wheels; ASan at `33302535`; the Arrow shim tree lived in L2b's removed
worktree). Not run here: CUDA, Linux, CI (seven workflow edits parsed only), Windows, ARC, Gurobi (mip_Gurobi.cpp
untouched and uncompiled), an IPC-only Arrow build.
