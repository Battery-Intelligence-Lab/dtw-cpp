# Handoff — 2026-10-02 — phase E: SW, VI, GC merged; W9b in revision (Windows)

## Base

Branch `design-2.0`, HEAD after this file's commit (on GC's PLAN mark `a018b77d`). Session 79bd8991.
Briefs, gates and the running record: session scratchpad `…/79bd8991-b0e9-441b-bc55-95584dcfaf55/scratchpad/`
(`brief_*.md`, `gates_*.md`, `pending_records_1001d.md`). W9b finished after this file's first commit: `pb/W9b` at
`974ca3cc` on base `558e09a6`, done and unmerged, not yet reviewed.

## Done (merge sha; one gated merge each)

- SW `6fd10ed1`: comments state reasons, not tracker ids (259 → 17 lines); tests keep `index_t`; `run.cpp` const read.
- VI `96547a5a`: `dtwc_cl.exe` carries a VERSIONINFO resource; two `int64_t` printf fixes (PRId64).
- GC `63415b4f`: FastCLARA assigns every series on the GPU on CUDA, in memory and streamed, on the fill's kernels
  (rectangle decode beside `decode_pair`); a `-v` line proves the GPU assignment ran (the CLI cell fails without it).
- Records `84e19714` and this commit: Volkan's 10-02 answers, unit lines, VI/WM rows, Metal noted for the Mac.
- Disk (Volkan-approved): 63 worktrees and the agents' loose logs, venvs and temp folders under `C:/D/git/wt` removed;
  kept `tmp/int*`, `venv/int`, W9b's worktree, `W9b-base` and venvs.

## Verified by me

- A session restart at 00:26 killed the SW integrator, W9b and GC mid-call (transcripts stopped, no processes); the
  post-resume "still running" notices were stale. W9b and GC resumed via SendMessage; a new integrator finished SW.
- SW's three CLI timeouts (180–633 s) were load: re-run alone on the same build, 3/3 passed in 2.43 s
  (`C:/D/git/wt/tmp/int/sw_cli_rerun.log`).
- GC review's main finding (nothing caught a silent CPU assignment) by reading the CLI cell and the FP32 case; GC's
  merge conflicts were comment-level (CHANGELOG, `test_metal_mmap.cpp`, `test_run_resolution.cpp`): resolutions mine.
- VI review fix `b63bc153` (mine): LegalCopyright = LICENSE lines 3–4; the node total prints 1885 via PRId64.
- v1.0.0's Python was never published: its README and CHANGELOG never mention Python; `pyproject.toml` has no name.
- Each merge commit and PLAN mark exists as listed (`git log`, PLAN rows opened).

## Reported by agents, unverified

- Integrators: SW — CUDA tree 94/0 (test_cuda_correctness ran), CLI 25 runs byte-identical, Arrow subset 5/5, pytest
  1094/19/0; VI — ctest 95 = 92 + 3 MAY_SKIP, CLI identical, VersionInfo identical under llvm-rc and rc.exe (.res
  byte-identical); GC — clang 95 = 92 + 3 MAY_SKIP, CUDA tree 94/0 (test_cuda_correctness ran), CLI 25 + 12 CPU runs
  byte-identical, GPU clara FP64 labels/medoids byte-identical to the CPU's, Arrow subset 5/5, pytest 1094/19/0.
- GC: FP64 labels, medoids and cost equal to the CPU's; fill SASS 12/12 identical; memcheck clean; band (quiet
  machine) entry/fill 1.050 FP32, 1.001 FP64; 6.6× / 1.6× the 24-thread CPU assignment. The streamed GPU route is
  checked only by GC's manual run: no gate tree has CUDA and Arrow together.
- W9b before revision: −1,926/+1,487 in 35 files; ctest 95; CLI 25 identical; pytest 1094 → 938 reconciled by name;
  `.pyd` 6,136,832 → 6,063,104 B; `cluster` timing equal.

## Decisions (Volkan 10-02, question tool; full quotes in DECISIONS §3)

- Wheels: measure MSVC against clang-cl first (PLAN WM). VERSIONINFO: added. ARC: Arrow ON, fail loudly. Headers:
  "whichever the best practices for modern CMake" → `FILE_SET HEADERS` per folder, in L2b. Cache rulings and a
  required `k` stand. `9056fcb9` was his. Metal: noted for his next Mac session (PLAN Blocked; W4e waits).
- v1 Python names: "nobody depends on them don't worry documenting the changes. We just need to have a proper
  documentation of the latest version for now."
- Readers: "Pritorise reading the same file in the same way in all languages if possible. So you can bind some reader
  …" → bindings bind the one C++ text reader and writer; Parquet/Arrow via pyarrow (MATLAB: parquetread, checked
  against C++); already-read data (numpy, lists, MATLAB matrices/cells) accepted everywhere.
- Clean-up: "Yes to both": a merged unit's worktree may be removed without asking.
- Put questions to him with the question tool, each self-explanatory (memory).

## Next steps (PLAN)

1. E: W9b (`974ca3cc`): the bound C++ reader and writers, Ctrl-Z refused (binary mode, `ctrl_z.csv`), already-read
   data (2-D array, ragged list, DataFrame); its report: ctest 95, CLI 25 identical, pytest 939/20/0 (pandas case
   skips without pandas), `.pyd` 6,156,288 B, `load()` as fast as base. → independent review → integrate (conflicts
   expected with SW's comments, GC's `run.cpp`, docs, CHANGELOG; devices.md / gpu-backends.md still name the deleted
   Python GPU refusal). Then remove `W9b-base`, `venv/W9b*` (approved).
2. E: W9e (`brief_w9e.md`, updated for the reader ruling) and M1 (`brief_m1.md`) after W9b; W9c; L2b after W9e (core vs
   io/CLI; bindings link core + the text reader/writer; `FILE_SET HEADERS`); W9f.
3. G: W14a — proper docs of 2.0 first (Volkan 10-02); WM on a quiet machine; W14c.
4. Follow-ups: C4244 at `unit_test_nonfinite_input.cpp:169`, `test_fill_request.cpp:189`, `run.cpp:97`,
   `test_cuda_correctness.cpp:2011`; tracker ids in
   tests/integration, conformance, benchmarks, examples; W4e after the Mac build; W13c–e after G.

## Open questions

- None waiting for Volkan. Remove `W9b-base`, `venv/W9b*` and the W9b worktree once W9b merges (approved).

## Status honesty

Windows only: clang tree, CUDA tree (MSVC + nvcc, RTX 4000 Ada), Arrow tree, wheel. Not run: macOS/Metal, Linux, CI,
MATLAB (no merged unit touched bindings/matlab). Timing under load is [inferred]; GC's band ran on a quiet machine.
