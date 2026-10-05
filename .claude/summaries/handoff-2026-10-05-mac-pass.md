# Handoff — 2026-10-05 — the Mac pass is next; W9b finishes on Windows

## Base

Branch `design-2.0`, HEAD = this file's commit (on `89814d19`); clean tree. The newest code is GC's merge `63415b4f`;
later commits are records. Volkan pushes `design-2.0` before switching to the Mac. Still running on the Windows PC
when this was written: W9b's last pass (`pb/W9b`, local only, worktree `C:/D/git/wt/W9b`), then its gated merge into
the Windows `design-2.0`. Both machines now commit on `design-2.0`: whoever pushes second runs `git pull --no-rebase`
first (never rebase).

## Done (merge sha; one gated merge each)

- SW `6fd10ed1`: comments state reasons, not tracker ids; tests keep `index_t`; `run.cpp` const read.
- VI `96547a5a`: `dtwc_cl.exe` carries a VERSIONINFO resource; two `int64_t` printf fixes (PRId64).
- GC `63415b4f`: FastCLARA assigns every series on the GPU on CUDA, in memory and streamed; `-v` prints the pair count.
- Records `89814d19` and this commit: Volkan's 10-05 Mac answers; the PLAN's Metal line lists the Mac pass.

## Verified by me

- No macOS build on record after 2026-09-23 (`baselines/2026-09-21-macos-first-baseline.md` and the 09-22/23
  baselines are the last); since `e784e5c3` (09-22): 371 commits, 456 code files (+29,452/−65,313). CI never ran
  on `design-2.0`: its workflows trigger on `develop`, `Claude` and PRs to `main`. Linux is unbuilt too.
- Metal: 22 commits since 2026-09-28 touch `dtwc/metal` or the Metal tests (`git log --since=2026-09-28 --
  dtwc/metal tests/unit/test_metal_*`), `dtwc/metal/CMakeLists.txt` among them; none has been compiled.
- W9b review (independent agent): merge after fixes. I opened every cited line: `set_data(Data, names, ndim)` dropped
  names and ndim; already-read data was converted twice (`_api._series`, `_clustering._series_list`); the
  distance-matrix reader opened in text mode; ' ' split on CR; `path.string()` in reader messages; a Python copy of
  `default_name`; `devices.md:22,66-67` false after the merge; the `dtw-variants.md:285` example raised.

## Reported by agents, unverified

- GC integrator: clang 95 = 92 + 3 MAY_SKIP; CUDA 94/0 (test_cuda_correctness ran); CLI 25 + 12 CPU runs
  byte-identical; GPU clara FP64 labels/medoids equal the CPU's; Arrow subset 5/5; pytest 1094/19/0.
- W9b review: CRLF reads as LF on every route; a Ctrl-Z is refused on every series route (base truncated 9/9);
  `cluster()` bit-identical to base on 36 cases; `Result.save` bytes equal the CLI's; the wheel has no CLI11/fkYAML/
  Arrow; the GIL is released in long calls.

## Decisions (Volkan, question tool; DECISIONS §3)

- 10-05: "Merge W9b, then Mac (Recommended)"; then, W9b needing 2.5–4 more hours: "Go now; W9b finishes here
  (Recommended)". The Mac pass runs on `design-2.0` without W9b; pytest runs again on the Mac once W9b lands.
- 10-02 rulings stand (DECISIONS §3): one C++ reader and writer bound in every language, Parquet via pyarrow,
  already-read data everywhere; v1.0.0 Python names need no shims; `FILE_SET HEADERS` per folder in L2b; Metal noted.
- Working rules kept only in the Windows PC's Claude memory: at most 4 agents at once; never Fable (simple units
  Sonnet 5.5 xhigh, hard ones Opus); questions to Volkan through the question tool, each self-explanatory; speed
  claims only on a quiet machine; a merged unit's worktree may be removed without asking.

## Next steps

1. Mac (PLAN "Blocked on another machine", Metal and macOS), in order, one commit per proven fix:
   a. `git fetch && git checkout design-2.0 && git pull --no-rebase`.
   b. Build with the runbook's `clang-macos` commands (Metal is ON on Apple). Expect first-time failures in
      `dtwc/metal/*.mm`, its CMake and libc++ differences; fix each at its cause.
   c. Serial ctest: 0 failures; test_metal_correctness and test_metal_mmap RAN (assertions in their Catch2 summary,
      not a skip); record the counts and every Skipped name.
   d. `cpp_conformance` digit-identical; an epsilon-only difference goes to Volkan (clustering must not change).
   e. The docs gates and pytest from a fresh venv outside the repo (runbook); the MEX and matlab_suite if MATLAB is
      installed (`-DDTWC_BUILD_MATLAB=ON`).
   f. Numbers verbatim in `.claude/baselines/2026-10-xx-macos-design-2-0.md` ([confirmed]); then W4e (PLAN row W4e).
2. When W9b reaches origin: `git pull --no-rebase` on the Mac, rebuild, pytest again (W9b rewrote Python's reading,
   writing and data conversion).
3. Windows (E): W9e and M1 after W9b (briefs `brief_w9e.md`, `brief_m1.md` in the Windows session's scratchpad);
   W9c; L2b (core vs io/CLI, `FILE_SET HEADERS`); the second, short Mac pass; W9f.
4. G: W14a (docs of 2.0 first), WM (wheel compiler measurement, quiet machine), W14c.
5. Follow-ups: C4244 at `unit_test_nonfinite_input.cpp:169`, `test_fill_request.cpp:189`, `run.cpp:97`,
   `test_cuda_correctness.cpp:2011`; tracker ids in tests/integration, conformance, benchmarks, examples.

## Open questions

- None waiting for Volkan. If the Windows session stopped before W9b merged: `pb/W9b` keeps its commits; resume from
  `brief_w9b_fix.md` (merge design-2.0 + findings F1–F10) and `brief_int_w9b.md` in that scratchpad.

## Status honesty

Windows only: clang tree, CUDA tree (MSVC + nvcc, RTX 4000 Ada), Arrow tree, wheel; the MEX was last built 10-01.
Never run on `design-2.0`: macOS/Metal, Linux, CI. W9b was not merged when this was written.
