# Handoff — 2026-10-11/12 (Mac) — the Mac after the Windows units: gates, Arrow 26, sanitizers, FM's Metal path, timing

## Base

- `design-2.0` local only, ahead of `origin/design-2.0` 6ae7824f by this session's commits (records 1d96c83c, the
  parity limit 292defe3, the codegen gate 13d23225, this file's commit); tree clean.
- `pb/FM-metal` dcc847be (local) on `origin/pb/FM` 3fe34165: FM's Metal float32 test. Push both (Volkan's).
- Worktrees (`.claude/worktrees/`): `mac-FM` = 6ae7824f + pb/FM merged, NOT committed; `FM-metal` (pb/FM-metal). In the
  session scratchpad `mac1011/agents/E/`: `src-base` 263e0e24, `src-head` 6ae7824f (detached). Remove all four with
  `git worktree remove -f -f` once FM is merged and the timing recorded.
- Kept `handoff-2026-10-10-sw-fm-ms.md` (Windows); the 10-09 one deleted.

## Done

- The pull deleted `.claude/CHARTER.md` and `.claude/baselines/` (034d8994 untracked them): restored from `034d8994^`
  (the 10-09 versions). The Windows records after that (`tmp/FM`, `tmp/MS`, `tmp/MM`) are only in
  `C:/D/git/wt/tmp/handoff-mac/records-2026-10-11.zip`, not on the Mac.
- Records: `baselines/2026-10-11-mac-gates-fm-metal.md` (all results; the pyarrow shim kept beside it) and
  `2026-10-11-mac-bands.md` (bands registered 23:20, before any run). PLAN marks; MAP §4 counts and §7's gate note.
- 292defe3: the parity test's float32 stream limit fits libc++'s 24-byte `std::string` (4500, windows stated).
- 13d23225: `test_codegen_no_calls` is not registered under a sanitizer (its checks are calls in every loop).
- dcc847be (pb/FM-metal): a Metal fill into a float32 matrix holds the float64 fill narrowed, in RAM and mapped.

## Verified by me (verbatim lines in the record)

- 6ae7824f: build 0 warnings; ctest 95 (CUDA the MAY_SKIP); codegen 61 loops 0 calls; Metal 2168 + 169 assertions;
  docs gates PASS; conformance D-19's ulp only; pytest 904/11/0; matlab_suite 141/141 (R2026a). With pb/FM merged:
  0 warnings, ctest 95, pytest 907/11/0, matlab_suite 142/142; the CLI's Metal float32 matrix = float64 narrowed in
  all 160,000 entries; a mapped float32 checkpoint resumes; a float64 run on it is refused.
- After 292defe3 and 13d23225: Release ctest "100% tests passed out of 95"; the Arrow 26 tree "out of 98" (CUDA
  skipped), "F8_PARITY … route_markers=12/12 parity=9/9"; an ASan tree lists 94 tests without the codegen gate.
- dcc847be's two tests re-run: 21972 assertions in 17 cases, 10072 in 6. The reader maps its file
  (`parquet_chunk_reader.hpp`, `MemoryMappedFile::Open`): why max RSS exceeds a streamed run's limit.
- Correction to the 10-10 handoff: W4d, W13a, W10 and GC all merged before 263e0e24, whose Mac ctest ran the Metal
  tests on 10-07; since then `dtwc/metal/` changed only in comments.

## Reported by agents, unverified by me

- A (Arrow, pyarrow 26.0.0 shim): both trees 0 warnings; `test_io_readers` 67761 assertions; the libc++ windows of
  the parity fixture; A2 the stream's peak memory footprint under each limit (955 MB at 1G … 373 MB at 384M) — PASS
  4/4 against limit + 64 MiB; A3 `cmp` identical 8/8.
- B (ASan + UBSan on 6ae7824f + pb/FM): 0 reports in two serial runs (child logs too), the markers shown to bite.
- C: the mutations that fail dcc847be's tests. E: the kit's builds, flags and the smoke (all six workloads identical
  in labels, medoids, costs and matrices; ratios NOT evidence: E1 0.87 twice, below its 0.95 band — watch it).

## Decisions

- Volkan this session: "Do if there is anything you could do on a mac if not then handoff".
- Awaiting him: where FM is timed (Windows' kit, or new Mac bands); MM's finding 7 (route Python's and MATLAB's
  cluster() through C++ `run`?); the records zip on the Mac; the gate if this Mac's idle load never falls below 1.5.

## Next steps

1. The timed kit (E): started 2026-10-11 23:42 with `caffeinate -i zsh $E/run_kit.sh > $E/kit-console.log 2>&1`
   (resumable: finished runs are skipped), then `uv run --no-project python $E/analyse.py $E/runs`; record the
   verdicts in the record's §E. E6 is not measurable (no silhouette time in `-v`).
2. Windows: re-run `test_fast_clara_parquet_parity` in the Arrow 23 tree (MS STL: the float32 window is inferred);
   FM's timing, then merge pb/FM and pb/FM-metal; a CUDA counterpart of the float32 GPU test (none exists).
3. MM round 4 (PLAN MM), with the zip; HG after MM.
4. Candidates: one repo script for the pyarrow CMake shim (both machines hand-make one); CMake 4.4's CMP0219 warning
   in `test_cli_config_formats.cmake:131` (a macro's backslashes; passes either way); a silhouette time in `-v`.

## Open questions

- From 10-10: compacting `ext4.vhdx`; the GitHub history purge; RA's notice; which `.claude` files to untrack.
- The Mac's charter is the 10-09 copy; the Windows one may have later entries (DECISIONS §3 has the rulings).

## Status honesty

- Built and run on the Mac only: `build/`, `build-matlab/`, the FM trees, two fresh-venv wheels, the agents' Arrow 26,
  ASan and timing trees. Not run: Windows, Linux, CI, Arrow 23 with my parity change, CUDA, the timed kit's result.
