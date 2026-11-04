# DTWC++ 2.0 — PLAN

Branch `design-2.0`. Phases A–G of the design review (`plans/2026-09-27-design-review.md`, approved 2026-09-28) and
the units after them are done; their rows, with commits and results, are in git (`git show 034d8994:.claude/PLAN.md`;
DT, AL, BF, CS, PE, PE-L, SW, MS and AU: `git show 11ef39c0:.claude/PLAN.md`). What is left is below.

Marks: ☐ open · ◐ in progress · ☑ done (commit; the row goes at the next prune). One implementer per step, never
two on the same files, at most four agents; each proven step is one local commit.

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

## Now: dtype at load and block-wise memory access

Spec `plans/2026-10-09-dtype-and-blocked-access.md` (Volkan's rulings of 10-08, approved 10-09), in this order:

- ☑ FM the opt-in float32 matrix (`matrix_dtype`; `.dtwm` v5 with the width and the rows written) (pb/FM c72adf3b,
  3fe34165, 2ef3e075; timed 10-21: 45 of 46 PASS, S2's mapped fill 1.60× and 1.14× more at float32, S4's mapped swap
  114×, swap and scores 1.1–2.1× at float32; clang's float32 Lloyd 0.49× FALSIFIED, fixed by a clang-only
  `vectorize(disable)` and re-timed 1.226; merged 0548796b: clang 95, Arrow 98 with the parquet parity test run, CUDA
  94, matlab_suite 142, pytest 899/19/0, CLI outputs identical; pb/FM-metal's Metal tests merged bc36375a; the Mac
  at 705ada87 10-21: clang-macos 95 with both Metal tests run, codegen 63 loops 0 calls, docs gates, pytest 907/11/0,
  matlab_suite 142; Apple clang's float32 sum scalar with or without the pragma (identical code); E1, the ETW
  counters, whenever Volkan runs `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` elevated)
- ◐ MM mapped Arrow IPC series as a view `Data` in every language, the series check fused into CLARA's pass, compressed
  IPC read piece by piece, Python's and MATLAB's `cluster()` through C++ `run` with highspy required, the load
  pricing the matrix (MS measured 10-10: at 10^8 series 75 % of the stream's cold time waits on its one decoder, so
  mapped series; Volkan 10-11 and 10-12, DECISIONS §3; design rounds 1–6 `C:/D/git/wt/tmp/MM/`, approved 10-12).
  On pb/MM: S1 the reader 70d512cd, hardened 4d1e0723 (nanoarrow 0.8.0 trusts a file's offsets and lengths; a
  crafted header crashed cl and GCC builds), GCC force-inline 84d84101 (Volkan 10-21); S2 codecs and decoding
  61b88767, 71d9d4b1, 25f65d34 (declared decoded sizes capped), fd8a7450 (its review: a V4 union, the footer); S3 the
  check, blocked and parallel, fused into CLARA's CPU pass, 14cf7b2b (its agent's reply lost at a session's end; gates
  read from its logs), dd7ba144 (its review: an error the PAMs or a block raise on accepted values, an overflowing
  distance, came before a refusal of a series not yet checked); six trees and matlab_suite green at each, pytest
  at each but dd7ba144 (no binding, reader or default); every review re-checked clean
  (`baselines/2026-10-21-mm-s1-reader.md`, `2026-10-22-mm-s2-codecs.md`, `2026-10-22-mm-s3-check.md`). S4 the stream
  62fab80d, its reviews' fixes 8dab1e78 (two blockers in the IPC reader's memory pricing) and 819e8703 (the walk's
  growth, prices measured both ways); six trees, matlab_suite, pytest green at each; closed 11-01
  (`2026-11-01-mm-s4-stream.md`). Then KS (below), then S5.
  - ☑ KS (superseded by RB below): the kernels' and algorithms' thread scratch priced on every route and released by
    `dtwc::run` on return (Volkan 11-01 "Kernel scratch first", "Release on return"; 11-02 "Release at phase end"). Four
    commits on pb/MM: 539d7d7e, 5422bb11, 4741d753, 1234030a (53 files, +2,345/-151 against 819e8703), each with fresh
    adversarial re-checks that found another window of `--ram-limit` values and then four residual loop deviations
    (record `baselines/2026-11-01-ks-scratch.md`; reviews `C:/D/git/wt/tmp/MM/review_ks*`). Volkan 11-02: "Fix first (a KS5
    unit)" for the loops, and on the RAM limit "Let's do an engineering solution. Relax it a little bit" (DECISIONS §3).
  - ◐ RB the simpler memory model (Volkan 11-02, "Approve RB and RR as written": `C:/D/git/wt/tmp/MM/ram_budget_spec.md`):
    one soft budget (`--ram-limit`, default 90 % of what the process may use), a reserve for the algorithm priced once
    (threads fitted once), one loading decision (in RAM, streamed, mapped) from budget minus reserve, `auto` by the same
    budget, a typed report where a run does not fit (`std::bad_alloc` at a stage boundary); KS's exact pricing removed,
    the CPU kernel and algorithm sources back to 819e8703's text (the KS5 loops by removal), "Release on return" kept.
    Plan `C:/D/git/wt/tmp/MM/rb_plan.md` (11-02, reviewed): ☑ RB1 restore the kernel sources + closed-list release
    (a6c1ab2f, 11-02; G6 absent for loop classes (a)-(c), class (d) the wheel's Soft-DTW gradient present, see the handoff),
    ☑ RB2 the plan (budget, reserve, one loading decision, `auto` guard; f9e731f9, 11-03, with its review's fixes 014123c4),
    ☑ RB3 the `bad_alloc` boundary (92c04967, 11-03), ☑ RB3b its fix commit (31f7158d: readers' and library OOM statuses
    become the one typed report, the ways out follow the run), ☑ RB3c (dff4590c, 11-03: TADPole's two regions carry a
    failure out, the re-check's four fixes; G6 on TADPole's pair loops falsified against the brief's strict band and
    accepted by me, see the handoff).
  - ◐ RR the end-of-run plain-text report in `Result` (same spec; Volkan 11-02: "Nice log in plain text at the end");
    planned (`rr_plan.md`; its code claims checked 11-03, `rr_claims_check.md`: 11 false, 37 moved), in two commits after
    RB3c: ☑ RR1a cpuinfo (optional) + machine facts + `print_settings` (06f7d00d, 11-03; `brief_rr1a.md`), ☑ RR1b the
    report, the fill split for PAM, Lloyd and full-sample CLARA only (43a4802b, 11-03; `brief_rr1b.md`), ☑ RR1b2 its
    command row's quoting (755b74e9, 11-03), ☑ RR1b3 three contracts pinned, tests only (ac25ffdc, 11-03); RR2 (counters)
    later. ☑ design-2.0 synced into pb/MM (7e061212, 11-03: the Linux wheel's TLS fix, verify_results, the user pages;
    one docs conflict; the wheel built from it imports in WSL without GLIBC_TUNABLES).
    S8's and S9's briefs carry the release on return into Python's and MATLAB's `cluster()`.
  Pulled forward: S8's read-only view `p_vec`, S9's MEX names.
  ☑ S5 a view on a GPU, interim: 99766bc1 + docs S5b d2abe94c on pb/MM (11-03; `brief_s5.md`; CUDA G8 verified, Metal lines
  await G8b on the Mac, listed in the handoff). ☑ S6 the CLI's reading: `convert_to_f32` gone, `-v` and the report name how
  an IPC file's series are held (04c793fc, 11-03; `brief_s6.md`; `format_of`/`read_input` stay: Parquet is `dtwc_io`'s).
  S7-S9 re-planned from today's code (11-03, `s7_plan.md`; the design's meter, pre-read pricing, `2·(M+E)` and P3 are
  superseded by RB): ☑ S7a the seam and the streamed `Result` (86e3467b, 11-03; `brief_s7a.md`), ☐ S7b `--checkpoint`
  without `-o` maps (cuttable), ☑ S8a Python plumbing (c3ba5458, 11-03; `brief_s8a.md`; the link: Volkan 11-03 "Link the CLI
  library", DECISIONS §3: the bindings link `dtwc_cli`, `_run`/`_Result`, cpuinfo hidden), ☑ S8b Python `cluster()` through
  `run` (1dbd03db, 11-03; `brief_s8b.md`; highspy required: PyPI's 1.15.1 has wheels for every tag and platform we build,
  checked twice 11-03) + ☑ S8b2 one set of words for a bad value whatever the input, stale `mip` extra lines (d1f0530f,
  11-03, mine), ☑ S9 MATLAB `cluster()` through `run`, `Result` wraps the run's, `Report`/`NoReport` (d04b61fd, 11-04;
  `brief_s9.md`; G10 660/660 in both MEX trees) + ☑ S9b two contracts pinned, tests only (6f199228, 11-04, mine).
  P1–P3, K1, K2 measured first in its timing session; MM merges once, after S10.
- ☐ HG the GPU on mapped series (a batch pipeline) and CPU + GPU together, measured first (Volkan 10-11: own unit after
  MM; `C:/D/git/wt/tmp/MM/hetero.md`)

## Before the 2.0 tag (Volkan 10-21)

Marked [here] (this Windows machine, WSL included, run in slots between timed units), [CI] (runs on Volkan's push;
`gh` is not authenticated here, so its results are read on GitHub) or [Volkan]. `VERSION` is 2.0.0rc1. MM and HG
both ship in 2.0 (Volkan 11-03).

- [done 11-03, RB3c dff4590c on pb/MM] TADPole's two bare `omp parallel` regions (`dtwc/algorithms/tadpole.cpp`): a
  refused allocation there (the first thread's DTW column with an explicit `--dc`) ended the process; it is carried out
  as RB3's typed report (Volkan 11-03: "Fix it before the 2.0 tag"). Repro and caps:
  `C:/D/git/wt/tmp/MM/review_rb3/runs/tad_region_*`.

- [done 11-02, redo on the tag tree if MM merges first] `cpp_conformance` under GCC and MSVC Release, `strict` and
  `fast`: the test as it stands (labels and medoids equal, scores within 1e-12; Volkan 11-02: "1e-12 is fine ... as long as
  it doesn't change the clustering"). `fast` passed in the pb/MM gate trees (GCC in WSL, cl in the CUDA tree); `strict`
  passed on design-2.0 292a9bd7: GCC 13.3 (WSL) and cl 19.50, 8 assertions in 1 test case each
  (`baselines/2026-11-02-conformance-strict.md`).
- [done 10-22] The Linux wheel could not be imported (glibc 2.39, since before MM): llfio's wg14_signals
  thread-local was `initial-exec`, so `_dtwcpp_core` was STATIC_TLS; TL 571c40f9, merged e3cba47d, sets llfio's
  own `WG14_SIGNALS_HAVE_ASYNC_SAFE_THREAD_LOCAL=0` on ELF: imports without the tunable, its Python suite
  901/17, GCC 94, Windows clang 95 (`baselines/2026-10-22-tl-static-tls.md`). [CI] whether python-wheels.yml's
  Linux jobs passed before it.
- [here] L2b's open ends: a Windows build with Arrow ON, and the Linux link order of `dtwc_core` / `dtwc_io` (WSL).
- [done 10-21] F42 does not reproduce: Auto, FP32, FP64 exit 0 through the CLI and a CUDA MEX on R2025b; July's
  crash was 926542f1's `static std::mutex` (MSVC >= 14.40's all-zero mutex under R2024b's 14.36 runtime), gone with
  e4a1d774, bcd5868d's macro and R2025b's runtime (`baselines/2026-10-21-f42-cuda-auto.md`).
- [here] The MEX maps no matrix yet (F43, an llfio-ON MEX crashing in `std::mutex` under R2024b, did not reproduce
  on 2026-09-30).
- [here] Release archives on Linux (WSL): `cpack` + `scripts/smoke_release_archive.py` (Windows done: RA, CA); the
  Linux wheel: whether `libgomp` ships, and the notice says so.
- [done 10-21] The docs audit (`baselines/2026-10-21-docs-audit.md`, 15 findings) fixed and merged 485dbeb7 (check_docs,
  check_pins, generate_docs green); `benchmarks/verify_results.py` named UCR rows from 0 where dtwc_cl names them from
  1 (every ARI compared shifted rows), fixed 59202f68. RA's notice: Volkan 10-21, "Keep bundling", the notice as
  written; its CHANGELOG line under Added. The site itself rebuilds in CI (`documentation.yml`).
- [CI] design-2.0's tip green: `python-wheels.yml` (cp39–cp314), `release-artifacts.yml` and `matlab-mex.yml` on
  Linux, macOS and Windows, the three unit jobs, `documentation.yml`.
- [Volkan] ARC (rule 11): W9c's remote leg — `test`, the `sinfo` check of `gpu_cc:8.6/8.9/9.0` and the `a100` gres
  type, `upload`, `build htc-cpu`, `build htc-gpu`, `build htc-gpu --gpu-device a100`, three `dtwcpp.cluster` runs
  (`hpc`, `hpc:gpu`, `hpc:gpu` + `gpu_device="a100"`), the unknown-key job, `submit-smoke gpu`.
- [Volkan] GitHub history: `origin/design-2.0` holds 310 baseline files and the charter, and older pushed history the
  PRIVATE Kasper folder (09-28); removing them needs a history rewrite and a force push.
- [Volkan] The tag, the release archives and the PyPI upload ("when we are hundred percent sure", charter item 4).

## Candidates (not scheduled)

- BF's small fills under GCC: the design's 8-column tasks (one cache line) wherever a slab has enough of them; GCC
  s1t (N 1000, 24 threads) ran 0.89–0.95×, perhaps neighbouring threads writing one line (untested; Volkan 10-09).
  On the Mac (6P + 12E) BF's fill of N 2000, L 500 at 18 threads ran 0.944× and 0.945× the 10-07 fill (one thread
  0.995×); 32 chunks per thread instead of 8 gave 1.038× (inconclusive): the last large chunks on efficiency cores
  (inferred, 10-12).
- CUDA: the 64 KB carveout above L = 2048 (−18 % at L 2049; passed its band) once the CUDA floor reaches 12.5 (it was
  reverted, b803e47, because the floor is 12.0 and ARC loads 12.4).
- The Windows archive ships HiGHS's development install (277 headers, a static `highs.lib`): trim it.
- MV: cl packing the lanes ran 3–4× slower in some processes (FALSIFIED 10-08; pb/MV kept): the cause is open.
- Barycenter and Soft-DTW keep each thread's largest DP matrix after a call (W13e): release or bound it.
- CUDA `Auto` through `Problem` is tested only for its refusals (`unit_test_variant_distmat.cpp:433`); the raw
  backend covers it (F42, 10-21). And `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` covers MATLAB builds only: HiGHS's and
  llfio's `std::mutex` could fault in a CLI or Python process that loaded an older msvcp140 first (untested).
- `test_mip.py::test_highspy_reaches_the_optimum_of_linked_highs` fails, not skips, when `DTWC_CL_PATH` names a
  `dtwc_cl` built without HiGHS (TL, 10-22): skip it there, or require HiGHS in that binary.
- Tracker prefixes in test names (`FX-1:`, F8/F13/F14 ids): name the contract instead.
- `import dtwcpp` imports scikit-learn whenever it is installed (`__init__.py:197` → `_clustering.py:16-18`), and
  through it scipy, pandas and joblib: +870 MB committed by scipy's OpenBLAS pool at 24 threads (FM's M3, 10-11), which
  MM's Python load would count against `ram_limit`. A lazy import of `DTWClustering` removes it.
- SW's swap after the fix pass: an O(N) update of the nearest and second-nearest after each accepted swap
  (FasterPAM's), the same values and ties as the full refresh; t2 (k 200) and the mapped m1 missed while every swap
  pays an O(N k) refresh (inferred, 10-10). And a prefetch of the panel lines a few rows ahead: SW's t3 missed
  0.22 LLC lines a lookup at IPC 0.53–0.56 (ETW 10-10, both commits), the packed rows' growing stride defeating the
  hardware prefetcher (inferred).

## Records

Map `MAP.md` · rules and rulings `DECISIONS.md` · lessons `LESSONS.md` · session state `summaries/` (newest two, via
`session-handoff`) · literature `docs/CITATIONS.md` · LR-core maths `UNIMODULAR.md`. On disk only, out of git
(Volkan 2026-10-09): the charter `CHARTER.md` and the run records `baselines/`. Anything superseded is deleted; git
keeps it.
