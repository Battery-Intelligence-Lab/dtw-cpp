# DTWC++ 2.0 — PLAN

Branch `design-2.0`. Phases A–G of the design review (`plans/2026-09-27-design-review.md`, approved 2026-09-28) and the
units after them (DT, AL, BF, CS, PE, PE-L, SW, MS, AU, FM, TL) are done; their rows, with commits and results, are in
git (`git show 034d8994:.claude/PLAN.md`, `git show 11ef39c0:.claude/PLAN.md`; MM's step rows S1–S10b and the RB/RR
detour: `git show 17cc792f:.claude/PLAN.md`). What is left is below, in the order of `plans/2026-11-09-finish-2.0.md`
(the retrospective and the proposal; its four questions await Volkan, and until he answers the rulings of
`DECISIONS.md` §3 stand as written).

Marks: ☐ open · ◐ in progress · ☑ done (commit; the row goes at the next prune). One implementer per unit, never two on
the same files, at most four agents; each proven step is one local commit.

## Proof rule (every step)

Serial `ctest -j1`; `cpp_conformance` digit-identical to the phase base unless the step pre-registers a change;
pytest from a fresh `uv` venv when bindings, readers or defaults change; `matlab_suite` when the MEX changes; the
CUDA build (`build/cuda-verify-0928`, RTX 4000 Ada) for GPU steps, macOS CI for Metal; `git grep` proof for every
deleted name (and `git grep <name> v1.0.0` = 0). Gates: `scripts/check_docs.py --cli <dtwc_cl>`,
`scripts/check_pins.py`, gitleaks in CI. A new gate is shown to bite by a reverted mutation.
A step that changes a hot loop (kernel cell, matrix get/set, `dist_by_ind`, swap or assignment loop) shows the
inner loop's assembly before and after: no call, reload or spill added (Volkan 2026-09-29).

**Pre-registered result changes** against v1.0.0 (everything else stays digit-identical): `pam` results (FasterPAM
only); the LR-core seed (same cost, medoids may differ on ties); Soft-DTW summation order (1e-12 relative); the v1
one-argument `init::Kmeanspp` sequence; `Method::MIP` above N = 200 uses the selected solver; `dtwc_cl`'s and the
MEX's default `lrcore` root is the subgradient root (Volkan 10-07): the cost is unchanged, medoids and labels may
differ only where several optimal medoid sets exist, node counts and run times change.

## Remaining, in order ([here] this machine, WSL included · [CI] Volkan's push · [Volkan] his actions)

1. ◐ **MM merge** [here]. pb/MM 0cc84a15 (S1–S10b, 38 commits; worktree `C:/D/git/wt/MM`, work files `C:/D/git/wt/tmp/MM/`)
   into design-2.0 with `--no-ff`: clang, Arrow, MEX (+ `matlab_suite`), CUDA and WSL GCC trees `ctest -j1`, fresh-venv
   pytest, docs gates, results identical to the base (G10), conformance. A dry-run merge has no conflict (11-09). The
   11-09 merge rule (bands A, B, D, E2, F and K's ratio block; `s10_kit_plan.md` §2 + §7, kit `kit/s10/`) holds until
   Volkan answers Q1 of the finish plan; MM merges once.
2. ☐ **QP code-quality pass** [here]: scope and stopping rule in the finish plan §4.2 (the five largest own files, the
   repo's `.clang-tidy` over `dtwc/` without `extern/`, MM's new tests under the 09-30 rule); every commit
   behaviour-neutral (`ctest -j1`, conformance digit-identical, pytest); two rounds, then stop.
3. ☐ **Release list** [here]: L2b's open ends (a Windows build with Arrow ON; the Linux link order of `dtwc_core` /
   `dtwc_io`); F43 (the MEX maps no matrix yet; the llfio-ON MEX crash under R2024b did not reproduce 2026-09-30); the
   Linux release archives (`cpack` + `scripts/smoke_release_archive.py`; whether `libgomp` ships and the notice says so;
   Windows done: RA, CA); `cpp_conformance` strict and fast under GCC and MSVC redone on the merged tree (passed on
   292a9bd7 and in the pb/MM gate trees); the docs' final read (charter item 10) and CHANGELOG's.
4. ☐ **Timing, advisory** [here, a quiet machine; Volkan 11-04 "Later, I will tell you"]: MM's bands and FM's E1 (ETW
   counters, `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` run elevated by Volkan); numbers to `baselines/` tagged
   [confirmed]; a FALSIFIED hot-loop band reopens its loop, anything else is fixed or reworded after.
5. ☐ **HG** [decision]: the GPU on mapped series as a batch pipeline, and CPU + GPU together (`C:/D/git/wt/tmp/MM/hetero.md`,
   design only, nothing built; Volkan 10-11 "own unit after MM", 11-03 "MM and HG both in 2.0"; the finish plan's Q2
   recommends 2.1). S5's route ships either way: a view uploads an owned Float64 copy; a run beyond the budget is a
   typed error naming `clara` on `cpu`; a GPU run's host memory (~90 MiB for CUDA) is outside the plan.
6. [CI] design-2.0's tip green on Volkan's push: `python-wheels.yml` (cp39–cp314), `release-artifacts.yml` and
   `matlab-mex.yml` on Linux, macOS and Windows, the three unit jobs, `documentation.yml`; whether python-wheels' Linux
   jobs passed before TL (e3cba47d). The Mac: no Metal has been built since S5 (G8b lines: the 11-09 handoff). macOS
   wheels are arm64 only (x86_64 dropped 2026-07-06: no Intel runner libomp); charter item 5 names Intel Macs: his call.
7. [Volkan] ARC (rule 11): W9c's remote leg — `test`, the `sinfo` check of `gpu_cc:8.6/8.9/9.0` and the `a100` gres type,
   `upload`, `build htc-cpu`, `build htc-gpu`, `build htc-gpu --gpu-device a100`, three `dtwcpp.cluster` runs (`hpc`,
   `hpc:gpu`, `hpc:gpu` + `gpu_device="a100"`), the unknown-key job, `submit-smoke gpu`. GitHub history:
   `origin/design-2.0` holds 310 baseline files and the charter, and older pushed history the PRIVATE Kasper folder
   (09-28); removing them needs a history rewrite and a force push. The tag (`VERSION` is 2.0.0rc1), the release archives
   and the PyPI upload ("when we are hundred percent sure", charter item 4).

Done before the tag already: TADPole's bare `omp parallel` regions carry a refused allocation out (RB3c dff4590c on
pb/MM, Volkan 11-03 "Fix it before the 2.0 tag"); the Linux wheel's STATIC_TLS import failure (TL e3cba47d, 10-22); F42
does not reproduce (10-21); the docs audit's 15 findings (485dbeb7) and `verify_results.py`'s row offset (59202f68).

## Cut, deferred or not scheduled

- Cut by Volkan: S7b `--checkpoint` without `-o` maps (11-04); kit bands F2/F3, P1, P3, C1–C4, R3 (11-09). Deferred: RR2,
  the report's counters (swaps, start cost, pairs, nodes; 11-03 "Later, own commit").
- Candidates: BF's small fills under GCC (8-column tasks; GCC s1t 0.89–0.95×, the Mac's 18-thread fill 0.944×); CUDA's
  64 KB carveout above L = 2048 once the CUDA floor reaches 12.5; the Windows archive's HiGHS development install
  (277 headers) trimmed; MV's cl lane packing 3–4× slower in some processes (FALSIFIED 10-08, cause open, pb/MV kept);
  barycenter and Soft-DTW keep each thread's largest DP matrix after a call (W13e); CUDA `Auto` through `Problem` tested
  only for its refusals, and `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` covers MATLAB builds only; `test_mip.py`'s highspy
  test fails, not skips, on a `dtwc_cl` built without HiGHS; tracker prefixes in test names; `import dtwcpp` imports
  scikit-learn whenever installed (+870 MB at 24 threads, counted against `ram_limit`: a lazy import removes it); SW's
  O(N) nearest update after an accepted swap and a prefetch of the panel lines (t2, t3, m1 missed, 10-10); a batch-wise
  `dtwc-convert` (it holds every series in RAM and writes one uncompressed batch; a Parquet file beyond RAM cannot be
  converted by it, S10a).

## Records

Map `MAP.md` · rules and rulings `DECISIONS.md` · lessons `LESSONS.md` · session state `summaries/` (newest two, via
`session-handoff`) · literature `docs/CITATIONS.md` · LR-core maths `UNIMODULAR.md`. On disk only, out of git
(Volkan 2026-10-09): the charter `CHARTER.md` and the run records `baselines/`. Anything superseded is deleted; git
keeps it.
