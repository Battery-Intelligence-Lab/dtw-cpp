# Handoff — 2026-11-10 (Windows) — HG merged with RD; one review dispositioned; the unit's gates run by me

## Base

- `design-2.0` at the commit holding this file, on the merge bfdb9c3a (pb/HG, `--no-ff`); tree clean. `pb/HG` 9cecafa3,
  `pb/RD` 3eb5c01d, `review/HG` 4d50f7e6 (probes only) stay local. Worktrees `C:/D/git/wt/HG` (trees `build/`,
  `build-cuda/`, `build-mex/`), `wt/RD`, `wt/HGR` are Volkan's to remove.
- Reports: `C:/D/git/wt/tmp/HG/report_hg5.md` (HG5's two halves, the review, my disposition), `report_hg4.md`; gate logs
  `C:/D/git/wt/tmp/HG/hgU_*`, WSL `~/mm/hgU/`; RD's `C:/D/git/wt/tmp/RD/`.

## Done

- HG5 (pb/HG 405f4d65..68c99019): `--gpu-share` / `gpu_share=` / `'GpuShare'` and the word `cpu+gpu` in every language,
  the refusal and `-v` texts, the pinned buffer's stderr line, the docs, CHANGELOG, MAP. The agent found two claims of
  my brief false and fixed them (dtwc_cl's own `CLARAOptions` never got the share; onebatch, tadpole and a stream
  dropped the key).
- RD (3eb5c01d, merged 418e1dec): the text reader's `while (getline)` took a failed stream for end of file, so a read
  that ran out of memory returned fewer series with no error (v1.0.0 shipped the loop). `text_io_detail::read_line`:
  `bad_alloc` leaves as itself, another failure is an `IOError` naming file and row.
- Two CUDA Python tests stale since before HG fixed (80e656a1).
- One adversarial review at 80e656a1: no race, no wrong result, kernels and the `gpu` route unchanged. Its blocker
  (d989585c: Tier-2 `one_batch_pam` / `tadpole` on a Problem refuse cpu+gpu and a forced share), its should-fix
  (6d036c8d: the fill's line names the ratio that split the slabs before the calibrating one; that geometry a fixture),
  nits N1, N3, N5, N6, and the GPU read check (447bfd52).
- HG merged into design-2.0 (bfdb9c3a); PLAN item 5 ☑.

## Verified by me

- Diffs read in full: HG5's code half, RD's commit; `devices.md`'s hybrid section against `on_both`.
- Mutations that bite (files restored by copy and `cmp`): the share check out of `one_batch_pam` fails 2 assertions
  (clang tree); the cpu+gpu check out of `tadpole` fails 3 and `s_here` taking a forced share in slab 0 fails 3 (CUDA).
- `test_run_out_of_memory` on the CUDA tree, three runs, 51 assertions each: the capped GPU read says "read stage: out
  of memory" and no `--dtype float32` (before RD the HG5 agent saw "dataset is empty" twice).
- Hierarchical on cpu+gpu, the real CUDA binary: the report's cluster row carries the fill's split.
- The unit's gates on 9cecafa3 (the merge's tree but for `.claude/PLAN.md`): CUDA ctest -j1 102/102
  (`test_cuda_correctness` Passed 179 s; device matrix gpu=cuda device=present cells=38/38; Metal's two skipped), clang
  103/103 (cells=30/30), MEX 104/104 with `matlab_suite: 156 run, 156 passed`, WSL GCC 102/102 (`test_io_error_contract`
  Passed there: RD's only libstdc++ run) and its Arrow tree 105/105, fresh-venv pytest 937 passed 20 skipped, a CUDA
  wheel of 9cecafa3 (`test.gpu()` hybrid True) `test_contract_parity` + `test_cuda` 159 passed, CLI identity 153 files
  1 differ (`print_config/config.toml`: `gpu-share = -1`), check_docs VERDICT=PASS (531 flags), check_pins failures=0,
  generated docs current.

## Reported by agents, unverified

- The review's probes ([hgr4] 108 forced fills × 5 repeats; [hgr3] a mapped resume, 0 wrong; the 2^14 chunk mutation;
  the `gpu` route cmp-identical to the pre-HG CUDA binary in 7 configurations; 10 kernel bodies 0 differ; tidy 0 in HG's
  lines). RD's standard-library reading (libstdc++, libc++ from source) and its allocator experiment.
- The review's model that 58–75 % of a large mapped fill's entries sit in one-tile-row slabs (not a measurement; my
  derivation from the geometry code is on `devices.md`: rows past min(1,024 L, N / 2) for 1 GiB slabs, at most three
  quarters of the entries; my first wording, "past about 1,024 L, most of a tall matrix", missed the mapped cap on h).

## Decisions

- Volkan: none new this session (11-09's two rulings stand, DECISIONS §3).
- Mine: the Tier-2 refusal covers cpu+gpu and a forced share only, `gpu` left as the documented gap; S2's cap on the
  tile-row height deferred to the timing window; [hgr3] not adopted (it writes through a `const_cast`); the review's
  zero-landing suspicion read and recorded, not a blocker (a kernel fault kills the context; a checkpoint and a mapped
  file's written mark follow a slab's success only).

## Next steps

1. PLAN item 3 (release list) and item 4 (timing, when Volkan says): HG's six bands, MM's, the one-tile-row-slab probe.
2. Before the next unit's identity gate: rebuild `build/bin/dtwc_cl.exe` (it pre-dates this merge).
3. PLAN item 6 (CI on Volkan's push), item 7 (his).
4. Small steps on offer, each its own: a Tier-2 Problem on `gpu` running onebatch / tadpole on the CPU (close the gap
   with a refusal); a failed `fread` mid-file reading as end of file in the MSVC STL's and libc++'s filebuf.

## Open questions

- None blocking. Volkan's: the merged worktrees and `pb/*` branches, the CHANGELOG's Performance section, Metal on the
  Mac (never built: HG's Metal paths, `hybrid_refused_message`'s Metal branch), a CUDA MEX tree (MATLAB's forced-share
  branch has not run), the history purge, ARC.

## Status honesty

- Built and run by me: every gate above, on a machine with Volkan's MATLAB sessions running; no timing claim of any
  kind is made, HG4e's sanity numbers stay [inferred].
- Not run: Metal, a CUDA MEX, the docs site's build (no Hugo here), cl's strict build, the release list's other trees.
- LESSONS not extended (the file is at its 300 lines); RD's rule is in `read_line`'s comment and the CHANGELOG.
