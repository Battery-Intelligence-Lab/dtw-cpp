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

1. ☑ **MM merged** c4bcefc2 (11-09; pb/MM 0cc84a15, S1–S10b, 38 commits, `--no-ff`; Volkan 11-09 "Correctness gates + one
   sanity run"). Gates on 0cc84a15: clang 103, Arrow 106, no-llfio 103, no-cpuinfo 103, MEX 104 with `matlab_suite`, CUDA
   102, WSL GCC 102, fresh-venv pytest 933/19, G10 660/660 (S9); on the merged tree: clang 103, Arrow 106, docs gates. The
   sanity run (the kit's A and B shapes once, under load, [inferred]: no gross slowdown, MM faster in all seven pairs) is on
   disk; the bands are advisory (item 4). The worktree `C:/D/git/wt/MM` and its eight build trees are now a merged unit's.
2. ☑ **QP code-quality pass** merged bb440d0e (11-09; `pb/QP`: five implementers on disjoint files, the orchestrator's
   gather and the MATLAB help fix, one review whose blocker and should-fixes closed in 7fc65957; briefs, reports and the
   identity gate in `C:/D/git/wt/tmp/QP/`). Behaviour-neutral but for "Medoid" in Lloyd's verbose lines (CHANGELOG):
   the identity gate's 30 CLI cases, report and `-v` text included, byte for byte the merge's; gates on the gathered branch
   clang 103, Arrow 106, no-llfio 103, no-cpuinfo 103, MEX 104 with matlab_suite 154 (the three help tests new), CUDA 102,
   WSL GCC 102, pytest 933/19, G10 324/324; the review's fix re-gated on clang and GCC. Under rule 19 none of MM's 99 test
   cases was trivial. Tree-wide clang-tidy: 0 own-code warnings left. Candidates it raised: `arrow_ipc.cpp` cut at its
   `IpcFile` banner, `NB_MODULE` into `bind_*` functions, a `Data::series<T>(i)`, the test files' shared helpers
   (`printed()`, `fixture()`, the "needs at least N bytes" parse), two error texts (`--n-init` without "got N"; two
   "needs approximately N bytes" wordings), `dtwc-convert`'s one batch.
3. ◐ **Release list** [here]. Done 11-09 on the merged code (WSL `~/mm/merge_gcc`, GCC 13.3): L2b's open ends (Arrow ON
   on Windows: the clang Arrow tree 106/106 at the merge; Arrow ON on Linux through the pyarrow 23.0.1 shim: 105/105 with
   `test_io_readers`, `dtwc_cl` linking libarrow, libparquet and libgomp); the Linux release archive (`cpack` 340 members,
   `smoke_release_archive.py` OK once its one-byte `--ram-limit`, a mapping threshold before the memory plan, became
   `--checkpoint`; `libgomp` is not shipped and `THIRD_PARTY_LICENSES.md` says so); `cpp_conformance` GCC strict 1/1 and
   fast (in the 102/102); `cpp_conformance` under MSVC fast (the CUDA tree's 102/102) and strict: the strict build of the
   merged tree FAILED in the vendored `flatcc.c` (C2099 under `/fp:strict`: MM's `dtwc_third_party` took our FP flags
   through `project_options`), fixed by c4271bbf (the flags on C++ only; cl strict 1/1, clang 103 + identity 0 differ,
   WSL GCC 102). F43: the CI MEX job builds with llfio since the commit after c3f26c47 (Volkan 11-09 "its R2024b run is
   the oracle": a crash there puts the flag back and becomes a tracked item). The docs' and CHANGELOG's final read (charter
   item 10) merged c3f26c47 (`pb/DOC`, 8 commits, 33 files; docs gates on the merged tree PASS; `check_site_links` runs in
   CI). It left ten code-side items (`C:/D/git/wt/tmp/QP/brief_dc.md`): unit DC, below.
3b. ☑ **DC, the code the docs read found wrong** merged 5b01d1fe (11-09; `pb/DC`, three commits, report
   `C:/D/git/wt/tmp/DC/report_dc.md`): Python's and MATLAB's `save_checkpoint` / `load_checkpoint` take the Problem's metric
   when none is named (rule 17; one pytest case, the MATLAB identity test extended); a `--checkpoint` run of a method that
   never allocates the matrix writes no all-NaN `.dtwm` (one CLI test that fails on the previous binary; CHANGELOG; two docs
   clauses); eight comments, docstrings and descriptions made true. Gates: clang 103, identity 0 differ, pytest 934/19, docs
   gates, the MEX tree 104/104 with the MATLAB suite (mine). Candidates it left: `save_dataset_csv(names=)` writes a header
   row `dtwc_cl` cannot read back; `help.md` and `visualize.md` still show `Method.Kmedoids`; TADPole on the mapped route
   leaves a never-filled `.dtwm`.
4. ☐ **Timing, advisory** [here, a quiet machine; Volkan 11-04 "Later, I will tell you"]: MM's bands, HG's baseline and
   FM's E1 (ETW counters, `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` run elevated by Volkan); numbers to `baselines/`
   tagged [confirmed]; a FALSIFIED hot-loop band reopens its loop, anything else is fixed or reworded after. Two of MM's
   design risks ride here (its design §7, risks 4 and 7): the mapped route is measured on a local NVMe and on WSL2 only
   (glibc's retention after a block's trim and llfio's prefetch unmeasured; a WSL2 result may not carry to native Linux),
   and a mapped file on ARC's file systems is an ARC run (item 7).
5. ◐ **HG** [here]: the GPU on mapped series as a batch pipeline, and CPU + GPU together (design
   `C:/D/git/wt/tmp/MM/hetero.md`; plan `C:/D/git/wt/tmp/HG/hg_plan.md`; Volkan 10-11 "own unit after MM", 11-03 "MM and HG
   both in 2.0", 11-09 "All of HG in 2.0", then "Build on the design's predictions; measure advisory", the device word
   "cpu+gpu", "Rounding errors are fine" for the hybrid). Steps on `pb/HG`, each merging on correctness (GPU FP64 results
   byte for byte the CPU's, the `cpu` route unchanged): ☑ HG0 the bit-equality test (908741d9) + ☑ HG1 the two-stream
   pipeline for CLARA's assignment (cb509789, Opus; `report_hg1.md`: kernel bodies identical to the base, G8 byte for byte
   on views of 256, 200k and 1.1M series in RAM and streamed, FP64 and FP32; the plan's new staging terms, a cap that shrinks
   the blocks under an explicit limit; one sanity run under load [inferred]: no slowdown), ☑ HG2 the CUDA fill reads a
   view's series through spans (42f6b12d + abe91544, Opus; `report_hg2.md`: its gate 0 reran HG1's G8 on real mappings with
   the CUDA tree llfio ON, every read line `mapped`, identical; the fill's matrix and the `.dtwm` byte for byte against the
   base in 15 comparisons, a checkpoint resume identical; one upload from spans, since the launches hold every series on the
   device: a tile-row pipeline would retile the kernels), ☑ HG3 the hybrid assignment (183a332a, a26c025e, 60b09c03,
   da6708ca, Opus; `report_hg3.md`: `Device::CPU_GPU` with rule-4 refusals everywhere but FastCLARA's in-RAM pass, one
   parallel region, `CLARAOptions::gpu_share`; FP64 L1 `cpu+gpu` = `cpu` = `gpu` by memcmp at four shares and by `cmp` on
   v200k and v1100k, FP32 0 near-ties; the sanity run [inferred]: FP64 0.72 of the GPU, FP32 1.76× the GPU — a gross
   slowdown, the serial 65,536-point CPU calibration block idling the GPU, which by hg_plan §3 stops the merge until),
   ☑ HG3e the calibration's cost (abdab042: the GPU's leading 65,536-point block, a 4,096-point trailing CPU block, both
   kept; the GPU share always [0, 65,536) then the rest, so a forced share repeats the bits; bands [inferred, loaded] FP32
   v1100k hybrid 1.12× gpu against ≤ 1.2 and FP64 0.75× against ≤ 0.8: both HOLD) + ☑ HG3f (1ccf05be: run.cpp's in-RAM
   CLARA through `fast_clara(prob, clara)`, so the plan's `gpu_staging_bytes` reach the run; a CLI cell that fails without
   the routing line; identity 0 differ, mine too), ☑ HG4 the hybrid fill (Opus; HG4a–d 3a2544bd, 5aca61e1, 220cff44, a9d64427:
   a CUDA fill session uploading once with row-range calls, whole tile-rows to either device, one region per slab,
   `Problem::set_gpu_share` the one knob, `auto` on cpu+gpu = pam at any N; identity 0 differ, mine too; G8 FP64 `cpu+gpu` =
   `cpu` = `gpu` by `cmp`; its FP32 sanity band FALSIFIED under the brief's floor-to-whole-tile-row split, 1.92× the GPU
   against ≤ 1.2) + ☑ HG4e (2c87f2d3; `report_hg4.md`'s HG4e section: per slab the tile-row bound least in
   max(GPU entries / s, CPU entries / (1 − s)), one s for the fill printed as itself, the calibrating slab found from the
   bounds so a forced run repeats a measured one, the recorded ratio by precision; my run of its four cases 2,206
   assertions; bands [inferred, 27 MATLAB processes]: FP32 0.90 and 0.89 × gpu against ≤ 1.2, FP64 0.72 and 0.68 × cpu
   against ≤ 0.8 (gpu FP64 reference not rerun): both HOLD; a forced share does not reach a slab before the calibrating
   one, kept and to be stated in the docs), ◐ HG5 the device word and `gpu_share` in every language (Volkan 11-09: "Key
   in every language"), the pinned buffer's stderr warning ("stderr warning always"), the docs HG1–HG4 made false
   (hg_plan.md §4b), CHANGELOG (Sonnet, launched 11-09 from 2c87f2d3, `brief_hg5.md`): its code half accepted 11-10
   (a6983659 on pb/HG, `report_hg5.md`; the diff read by me, its gates the agent's: CUDA 102/102, clang 103/103, identity
   1 differ = the `gpu-share` line, pytest 937, matlab_suite 156/156), its docs half 11-10 (68c99019; docs gates PASS,
   Hugo not installed so the pages' build unrun), `test_cuda.py`'s two failures on a CUDA wheel stale since before HG and
   fixed (80e656a1; 15 passed on the wheel);
   ☑ RD (11-10, 3eb5c01d on `pb/RD` from pb/HG a6983659, worktree `wt/RD`, logs `C:/D/git/wt/tmp/RD/`): confirmed before
   the fix, `for_each_data_line` over a stream buffer that throws after two lines returned 2 with no error (v1.0.0
   shipped the loop; with an allocator refusing 1 MiB, `readFile`, `load_batch_file` and the matrix's `read_csv` returned
   0 series / 0x0); `text_io_detail::read_line` puts badbit in the exception mask for the `getline`, `bad_alloc` leaves
   as itself, another failure is an `IOError` naming file and row; its test bites; clang 103/103, identity as pb/HG's
   (the agent's gates; the diff read by me); merges into pb/HG after the review hands the CUDA tree back; left: a
   failed `fread` mid-file is still end of file in the MSVC STL's and libc++'s filebuf (no badbit to see), pytest on
   the merged tree; to verify before the merge: the GPU read check in `test_run_out_of_memory` once RD is in (HG5b's GPU `ways_out` edits are unexercised), MATLAB's
   forced-share branch (no CUDA MEX tree); ◐ one review, launched 11-10 at 80e656a1 (`brief_hg_review.md`, `wt/HGR`); its six bands (hg_plan.md §3) in item 4's window. **HG merges once**: the steps accumulate on `pb/HG` (each implementer branches from its tip) and the docs stay
   true on design-2.0 (a view's series uploaded as an owned copy) until HG5's pages land with the code; design-2.0 is merged
   into `pb/HG` after DC lands (both touch `run.cpp`: DC the checkpoint save, HG1 the GPU memory terms), and the identity
   gate runs on the merged tree, not the branches alone. Until HG merges, S5's route ships.
6. [CI] design-2.0's tip green on Volkan's push: `python-wheels.yml` (cp39–cp314), `release-artifacts.yml` and
   `matlab-mex.yml` on Linux, macOS and Windows, the three unit jobs, `documentation.yml`; whether python-wheels' Linux
   jobs passed before TL (e3cba47d). The Mac: no Metal has been built since S5 (G8b lines: the 11-09 handoff). macOS
   wheels are arm64 only (x86_64 dropped 2026-07-06: no Intel runner libomp); charter item 5 names Intel Macs: his call.
7. [Volkan] ARC (rule 11): W9c's remote leg — `test`, the `sinfo` check of `gpu_cc:8.6/8.9/9.0` and the `a100` gres type,
   `upload`, `build htc-cpu`, `build htc-gpu`, `build htc-gpu --gpu-device a100`, three `dtwcpp.cluster` runs (`hpc`,
   `hpc:gpu`, `hpc:gpu` + `gpu_device="a100"`), the unknown-key job, `submit-smoke gpu`, and one `dtwc_cl` run on a mapped
   IPC file on ARC's file system (MM design §7 risk 4). GitHub history:
   `origin/design-2.0` holds 310 baseline files and the charter, and older pushed history the PRIVATE Kasper folder
   (09-28); removing them needs a history rewrite and a force push. The tag (`VERSION` is 2.0.0rc1), the release archives
   and the PyPI upload ("when we are hundred percent sure", charter item 4).

Done before the tag already: TADPole's bare `omp parallel` regions carry a refused allocation out (RB3c dff4590c on
pb/MM, Volkan 11-03 "Fix it before the 2.0 tag"); the Linux wheel's STATIC_TLS import failure (TL e3cba47d, 10-22); F42
does not reproduce (10-21); the docs audit's 15 findings (485dbeb7) and `verify_results.py`'s row offset (59202f68).

For Volkan's hand: the merged units' worktrees, clean, their branches merged (`C:/D/git/wt/FM` 4.6 GB, `wt/TL` 622 MB,
`wt/REL`, `wt/QP1`–`wt/QP5`, `wt/QPR`, and `wt/QP` and `wt/DOC` once HG no longer needs `wt/QP/build-cuda` as its base
binary): `git worktree remove --force <path>` and `git branch -d pb/<id>`; the permission classifier refuses the agent that
deletion (11-09). The CHANGELOG's Performance section compares 2.0 with earlier 2.0 (no v1.0.0 baseline exists for its
subjects): keep as the development record it is, or drop (DOC's open question).

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
