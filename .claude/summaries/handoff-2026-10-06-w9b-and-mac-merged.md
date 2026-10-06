# Handoff — 2026-10-06 — W9b merged; the Mac pass merged and verified on Windows

## Base

Branch `design-2.0`, HEAD = this file's commit (Windows, session 79bd8991) on `bdc47410`: origin's Mac pass
(`5d85fa6b`) merged into W9b's merge (`4265e3a6`, PLAN marks `03aad770`). Volkan pushes; the Mac then runs
`git pull --no-rebase`. Read the Mac's own record too: `handoff-2026-10-05-mac-pass-done.md`. Running when this was
written: TB, the x86 timing of the banded kernel's bounds change (next steps).

## Done

- W9b `4265e3a6`: Python's cluster(), DTWClustering, load() and set_data run on the C++ core (Config and apply() in
  the core, then Problem::cluster()); Python reads text through the bound C++ reader and writes through the C++ writer;
  text and distance-matrix files are read as bytes (a Ctrl-Z is refused, a CR ends a line only before its LF, ' '
  splits on spaces and tabs); reader messages name files in UTF-8; one conversion takes already-read data (2-D array,
  ragged list, DataFrame, Arrow array) for every entry; a run is named by C++'s rule.
- `bdc47410`: origin's Mac pass merged on Windows (Volkan 10-06: "You can pull and merge the macos changes").
- Records: this commit (DECISIONS §3, two 10-06 lines). Disk (approved 10-02): W9b's and W9b-base's worktrees, venvs
  and probe folders removed; the branch `pb/W9b` stays.

## Verified by me

- The W9b review's cited lines before the fix pass (set_data dropping names/ndim, two conversions of already-read data,
  the matrix reader in text mode, ' ' splitting on CR, `path.string()` in messages, a Python copy of `default_name`,
  `devices.md:22,66-67`, the `dtw-variants.md` example); after it, the merge ports (GC's message in apply(), no
  clara-plan refusal, no `std::as_const`) and `_NotSeries(TypeError, ValueError)`, by reading.
- Both merges dry-ran clean (`git merge-tree`); `bdc47410` has parents `03aad770` and `5d85fa6b`.

## Reported by agents, unverified

- W9b integrator, before merging (at bc9469fd): CLI clang 156, CUDA-CPU 65 and GPU 51 files byte-identical to GC's
  outputs at 63415b4f, so a332d671 changes no output bit on Windows. After `4265e3a6`: clang ctest 95 = 92 + 3 skips;
  CUDA 94 = 92 + 2 (test_cuda_correctness ran); conformance; docs gates; Arrow 5/5; matlab_suite 140/139/0/1; pytest
  962/20/0 reconciled by id (1113 → 982: 199 removed, 68 added); every CLI set equal before and after.
- `bdc47410`: `CODEGEN_NO_CALLS tool=CLANG_~1.EXE inner_loops=146 calls=0 verdict=PASS` (the any-loop gate's first
  Windows run; clang++ with GNU-style flags); clang ctest 95 = 92 + 3, same list; CUDA 94 = 92 + 2; pytest 962/20/0,
  the same 982 ids and statuses; Arrow 5/5; matlab_suite 140/139/0/1; docs gates; every CLI set byte-identical to
  W9b's merge (the GPU clara cells after normalising times and paths). The Mac's kernel commits change no bit here.
- W9b fixer: `load(X)` of a 20,000 × 1,000 array takes about 0.8× base's time [inferred, shared machine].

## Decisions

- Volkan 10-06 (chat): "You can pull and merge the macos changes".
- W9b's choices (DECISIONS §3 10-06; Volkan may veto): input that is not series raises `_NotSeries(TypeError,
  ValueError)` (scikit-learn's checks need ValueError); `refuse_gpu_method` is inlined into apply().
- Volkan 10-06 on the Mac (DECISIONS §3): `822225bc` (bounds per column) stays or goes by the x86 timing, TB.

## Next steps

1. Windows: TB (brief `brief_tb.md` in the Windows scratchpad): `822225bc` against `f5c58764` on x86, clang++ and MSVC;
   registered band: no cell slower than 1.05×; a FALSIFIED result goes to Volkan before any revert.
2. Mac: pull, rebuild, pytest again (W9b); the rest of the Mac's handoff stands.
3. E: W9e and M1 (briefs `brief_w9e.md`, `brief_m1.md` in the Windows scratchpad), W9c, L2b, the second Mac pass,
   W9f. W9f also restores the contract rows `Problem.read_distance_matrix` and `Problem.print_distance_matrix` (W9b
   deleted them with the methods and did not restore them when it bound the methods again), with a missing-file case.
4. G: W14a, WM, W14c.
5. Follow-ups: on Windows `dtwc_cl` gets argv in the ANSI code page, so `δ.csv` arrives as `d.csv` and can silently
   name another file (CLI11's `ensure_utf8`, or `wmain`); C++ `distance::dtw` returns 1.8e308 for an empty series;
   `DataLoader.hpp:106` throws on a non-ANSI extension; `std::as_const` leftovers in `api.cpp:119` and the binding;
   C4244 at `unit_test_nonfinite_input.cpp:169`, `test_fill_request.cpp:189`, `run.cpp:97`,
   `test_cuda_correctness.cpp:2011`; tracker ids in tests/integration, conformance, benchmarks, examples.

## Open questions

- Volkan may veto `_NotSeries(TypeError, ValueError)`. The Mac's 10-05 commits name Claude Fable 5.1 as co-author,
  while DECISIONS 09-30 says no Fable: his call whether that rule still stands.

## Status honesty

Windows, all at `bdc47410`: clang tree, CUDA tree (MSVC + nvcc, RTX 4000 Ada), Arrow tree, MEX (R2024b), wheel.
The Mac, per its handoff: `822225bc`, without W9b. Never run: Linux, CI. TB had not finished when this was written.
