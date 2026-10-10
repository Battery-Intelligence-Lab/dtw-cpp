# DTWC++ 2.0 — PLAN

Branch `design-2.0`. Phases A–G of the design review (`plans/2026-09-27-design-review.md`, approved 2026-09-28) and
the units after them are done; their rows, with commits and results, are in git (`git show 034d8994:.claude/PLAN.md`).
What is left is below.

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

- ☑ DT `dtype` where data is loaded, in C++, Python, MATLAB and the CLI (float32 one word away; G2: a double view
  under `--dtype float32` is no longer kept silently) (ddb9fc3b; merged 999457d1: clang 95, Arrow tree 98, CUDA 94,
  matlab_suite 141/141, pytest 895/19/0; float32 output as before; a float32 Parquet load peaks at 0.52× the memory)
- ☑ AL CLARA's assignment through the lanes kernel (added 10-09, "AL right after DT": per-pair today at 17.9 Gcell/s
  on 24 threads against the lanes fill's 105–131) (pb/AL 26d4af40; timed 10-09 06:16, PASS under clang, GCC and cl:
  clang 3.4–3.8× at 24 threads and 7.8× on one, GCC 4.8–5.7× and 10.0×, cl 1.29× under a band; merged 6359acba: clang
  95, Arrow 98, CUDA 94, matlab_suite 141, pytest 895/19/0; CLI and CLARA outputs identical)
- ☑ BF the blocked CPU fill (slabs and tiles; prefetch and range flush when mapped; checkpoints per slab; parallel NaN
  prefill) with the one automatic memory key `ram_limit` (pb/BF 1b072918; matrices equal base's, GCC squared within
  the route bound; timed 10-09: clang and cl 0.99–1.08×, mapped 22.6× at a 4 GiB cap and ≥ 37× at 2 GiB, checkpoints
  write 2 M against 80 M; GCC s1t 0.892× against ≥ 0.98, a registered second attempt; e3 FALSIFIED: LLC misses 0.405
  and 0.450 per entry against ≤ 0.4, base 0.87–1.18; c1's closing save 0.04 % over, recorded; GCC s1t FALSIFIED,
  0.951× the second time; Volkan 10-09: merge with the misses recorded; merged ec8d10e6: clang 95, Arrow 98, CUDA 94,
  matlab_suite 141, pytest 896/19/0; CLI outputs identical)
- ☑ CS CLARA reads its data once (all samples drawn first, their union read once, decoding overlapped) and its stream
  refuses each row the in-RAM run refuses (NaN, ±inf, an empty row, a band too narrow: base published a cost of
  1.8e308) (pb/CS afd022c7 on pb/AL; 360/360 runs identical; P1 decodes 7.28 → 1.93 passes; timed 10-09 08:46, PASS:
  streams 3.4–3.6× faster, in RAM 1.00–1.86×; merged 8af0336f: clang 95, Arrow 98, CUDA 94, matlab_suite 141, pytest
  896/19/0; a `--ram-limit` too small for the samples' labels and one row group is now refused, as designed)
- ☑ PE the Parquet stream holds its RAM limit (Volkan 10-09: next, before SW): every read decodes about 1 MiB at a
  time and the stream is priced by what it holds (pb/PE 4e5123ea; under `--ram-limit 1G` P1 peaks at 0.71 GB, was 1.47
  GB, against 1.16 GB allowed; 8/8 memory runs within the limit + 64 MiB; 360/360 identical; timed 10-09: 1.55–1.73×
  faster than before; merged cc45f4b4: clang 95, Arrow 98, CUDA 94, matlab_suite 141, pytest 896/19/0; small explicit
  limits rise: one decode is priced at about 10 MB)
- ☑ PE-L the stream's memory price on Linux (WSL, GCC 13.3, pyarrow 23's libraries; counters by the scope's
  memory.peak): the automatic limit under cgroups of 768M, 1G and 2G 12/12, never killed, results identical; explicit
  limits 7/8 at first (P1 f32 at 256M 0.8 MB over, beside its own labels file in the page cache), 8/8 after glibc's
  kept batches (320–680 MB) went back through `malloc_trim(0)`; merged a30e6cf0
- ◐ SW the blocked SWAP (accepted swaps identical to the sequential sweep's) and the scores' loop orders
  (design `C:/D/git/wt/tmp/SW/design.md` approved 10-09 with conditions; proven on pb/SW 2a63803f: every
  evaluation base's on clang, GCC and cl; c0 set the parallel threshold at 2000 points; timing 10-10)
- ◐ FM the opt-in float32 matrix (`matrix_dtype`; `.dtwm` v5 with the width and the rows written; design
  `C:/D/git/wt/tmp/FM/design.md` approved 10-10 with conditions, Volkan's four answers in DECISIONS §3;
  implementation after SW's merge)
- ◐ MS measure, after CS, whether CLARA's time at scale goes to decoding or to the disk; mapped series only if
  decoding dominates (plan `C:/D/git/wt/tmp/MS/design.md` with its decision rule registered 10-10; Volkan: WSL
  decides, 2^25 float64 series; kit building)
- ☑ AU `@author Volkan Kumtepeli` in every source file's header (Volkan 10-09; b77ba672: 160 files, comments only;
  merged 6521b01e: clang 95, CUDA 94, matlab_suite 140/140, pytest 893/19/0, CLI outputs byte-identical); the last four
  record citations in header blocks reworded after it

## Blocked on another machine or on Volkan

- macOS: run `test_codegen_no_calls` and the Metal tests on the Mac after RL (its padding loop stores one value at a
  128-byte stride, not Apple clang's `memset_pattern16` idiom, unproven there) and after DT/AL.
- ARC (Volkan, rule 11): W9c's remote leg — `test`, the `sinfo` check of `gpu_cc:8.6/8.9/9.0` and the `a100` gres
  type, `upload`, `build htc-cpu`, `build htc-gpu`, `build htc-gpu --gpu-device a100`, three `dtwcpp.cluster` runs
  (`hpc`, `hpc:gpu`, `hpc:gpu` + `gpu_device="a100"`), the unknown-key job, `submit-smoke gpu`.
- Release archives: `cpack` + `scripts/smoke_release_archive.py` on Linux (Windows done: RA, CA). Linux wheel:
  whether `libgomp` ships, and the notice says so.
- L2b's open ends: a Windows build with Arrow ON and the Linux link order of `dtwc_core` / `dtwc_io` are unproven.
- `cpp_conformance` under GCC and MSVC Release, `strict` and `fast`: the same 17 significant figures.
- GitHub history: `origin/design-2.0` holds 310 baseline files and the charter, and older pushed history the
  PRIVATE Kasper folder (09-28); removing them needs a history rewrite and a force push (Volkan's).
- RA's notice: the wording of Microsoft's Distributable Code terms (indemnity, pass-through of protective terms) in
  `THIRD_PARTY_LICENSES.md`, and whether RA's CHANGELOG line is Fixed or Added (v1.0.0 shipped no wheels or archives).
- F42: a crash of CUDA `Auto` precision through `Problem`, recorded on Windows and never re-run. No MEX route maps a
  matrix yet (F43, an llfio-ON MEX crashing in `std::mutex` under R2024b, did not reproduce on 2026-09-30).

## Candidates (not scheduled)

- BF's small fills under GCC: the design's 8-column tasks (one cache line) wherever a slab has enough of them; GCC
  s1t (N 1000, 24 threads) ran 0.89–0.95×, perhaps neighbouring threads writing one line (untested; Volkan 10-09).
- CUDA: the 64 KB carveout above L = 2048 (−18 % at L 2049; passed its band) once the CUDA floor reaches 12.5 (it was
  reverted, b803e47, because the floor is 12.0 and ARC loads 12.4).
- The Windows archive ships HiGHS's development install (277 headers, a static `highs.lib`): trim it.
- MV: cl packing the lanes ran 3–4× slower in some processes (FALSIFIED 10-08; pb/MV kept): the cause is open.
- Barycenter and Soft-DTW keep each thread's largest DP matrix after a call (W13e): release or bound it.
- Tracker prefixes in test names (`FX-1:`, F8/F13/F14 ids): name the contract instead.
- SW's swap after the fix pass: an O(N) update of the nearest and second-nearest after each accepted swap
  (FasterPAM's), the same values and ties as the full refresh; t2 (k 200) and the mapped m1 missed while every swap
  pays an O(N k) refresh (inferred, 10-10). And a prefetch of the panel lines a few rows ahead: SW's t3 missed
  0.22 LLC lines a lookup at IPC 0.56 (ETW 10-10), the packed rows' growing stride defeating the hardware prefetcher
  (inferred).

## Records

Map `MAP.md` · rules and rulings `DECISIONS.md` · lessons `LESSONS.md` · session state `summaries/` (newest two, via
`session-handoff`) · literature `docs/CITATIONS.md` · LR-core maths `UNIMODULAR.md`. On disk only, out of git
(Volkan 2026-10-09): the charter `CHARTER.md` and the run records `baselines/`. Anything superseded is deleted; git
keeps it.
