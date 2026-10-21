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

- ◐ FM the opt-in float32 matrix (`matrix_dtype`; `.dtwm` v5 with the width and the rows written; design
  `C:/D/git/wt/tmp/FM/design.md` approved 10-10 with conditions, Volkan's four answers in DECISIONS §3;
  c72adf3b and 3fe34165 on pb/FM 10-11: identity bit for bit on three compilers, gates pass; of the assembly check's
  FAILs Calinski-Harabasz's were fixed and six accepted by Volkan ("Accept all, time FM"); not timed: the quiet gate
  never passed 10-11 19:34–22:14; its bands are this Windows machine's; timed 10-21 05:15–07:20: 45 of 46 PASS
  (S2's mapped fill 1.60× and 1.14× more at float32, S4's mapped swap 114×, swap and scores 1.1–2.1× at float32),
  FALSIFIED clang's float32 Lloyd 0.49× (D3's accepted `calculate_medoids` loop, vectorised); Volkan 10-21: "Fix
  first" (a clang-only `vectorize(disable)`), E1 not awaited; Mac 10-11, merged into 6ae7824f uncommitted:
  `metal_dtw.mm` builds without warnings, ctest 95, pytest 907/11/0, matlab_suite 142/142 on R2026a, the CLI's Metal
  float32 matrix equals the float64 one narrowed in all 160,000 entries, its mapped checkpoint resumes)
- ◐ MM mapped Arrow IPC series as a view `Data` in every language, the series check fused into CLARA's pass, compressed
  IPC read piece by piece, Python's and MATLAB's `cluster()` through C++ `run` with highspy required, the load
  pricing the matrix (MS measured 10-10: at 10^8 series 75 % of the stream's cold time waits on its one decoder, so
  mapped series; Volkan 10-11 and 10-12, DECISIONS §3; design rounds 1–6 `C:/D/git/wt/tmp/MM/`, round 6's
  review: none left after F3's threshold; approved 10-12; code after FM's merge, P1–P3, K1, K2 measured first)
- ☐ HG the GPU on mapped series (a batch pipeline) and CPU + GPU together, measured first (Volkan 10-11: own unit after
  MM; `C:/D/git/wt/tmp/MM/hetero.md`)

## Before the 2.0 tag (Volkan 10-21)

Marked [here] (this Windows machine, WSL included, run in slots between timed units), [CI] (runs on Volkan's push;
`gh` is not authenticated here, so its results are read on GitHub) or [Volkan]. `VERSION` is 2.0.0rc1. Whether MM
and HG ship in 2.0 or after it is open (not asked).

- [here] `cpp_conformance` under GCC and MSVC Release, `strict` and `fast`: the same 17 significant figures.
- [here] L2b's open ends: a Windows build with Arrow ON, and the Linux link order of `dtwc_core` / `dtwc_io` (WSL).
- [here] F42: a crash of CUDA `Auto` precision through `Problem`, recorded on Windows and never re-run.
- [here] The MEX maps no matrix yet (F43, an llfio-ON MEX crashing in `std::mutex` under R2024b, did not reproduce
  on 2026-09-30).
- [here] Release archives on Linux (WSL): `cpack` + `scripts/smoke_release_archive.py` (Windows done: RA, CA); the
  Linux wheel: whether `libgomp` ships, and the notice says so.
- [here] The docs site (charter item 10): `docs/content` against the 2.0 surface, `scripts/generate_docs.py --check`
  and `check_docs.py` green; RA's notice: the wording of Microsoft's Distributable Code terms (indemnity, pass-through
  of protective terms) in `THIRD_PARTY_LICENSES.md`, and whether RA's CHANGELOG line is Fixed or Added (v1.0.0 shipped
  no wheels or archives), for Volkan to rule.
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
