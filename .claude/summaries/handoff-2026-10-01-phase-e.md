# Handoff — 2026-10-01 — phase E most of the way (Windows)

## Base

Branch `design-2.0`, HEAD after this file's commit (on `c115e837`). Session 79bd8991 (continued from 09-30). Agent briefs
and gates: session scratchpad `…/79bd8991-b0e9-441b-bc55-95584dcfaf55/scratchpad/` (`brief_*.md`, `gates_*.md`,
`pending_records_1001d.md`). In flight at writing: W8c (Sonnet, `pb/W8c` from `171b724c`); `pb/W9a` (`151f14e`) done,
not merged.

## Done (10-01; merge sha, one gated merge each)

- W7c `6a44e4d2` (one `core::validate`); W12b `65aadc75` + `a0d58242` (`tests/unit/adversarial/` gone); W7g `601748f8`
  (one `distance.dtw` per language); W7d `1ae4a301` (`core::orient` + `run_dtw`; each kernel copies its Cost).
- L2a `1110fe10` (each `dtwc/` folder lists its own files); W8b `862a08fe` (one writer; a streamed `Result::save` no
  longer reads an empty Data); W7ef `c37e2673` (Soft-DTW and `dtwFull` on the linear kernel; `writable_distance_matrix()`;
  out-of-range enums refused); W10 `73d73616` (`set_device` + `set_gpu_precision`; cache keyed by computed precision);
  W14b `9056fcb9` (an unhonourable `ON` stops the configure; Gurobi defaults OFF).
- Records: `f61af854` and this commit (DECISIONS §3 per unit, three LESSONS, PLAN marks by the integrators).

## Verified by me

- W12b review fix `6fb95f4`: with both oracles mutated to ignore the band the sweeps fail 1,456 / 1,776 / 31 assertions;
  `dtw_routes_agree(a, inf)` no longer passes; serial ctest in `wt/W12b` 98 = 95 + 3 MAY_SKIP, 0 failed.
- L2a review fix `15d87f2`: reconfigured, 487 compile commands, `dtw_lanes.cpp` keeps `-fno-lto`, ninja no work to do.
- W8b review fix `26b0409`: `std::as_const` on the matrix read; Python's save silent at k = 1 like C++; `wt/W8b` ctest
  101 / 0 failed, pytest 1182 / 19 / 0, the renamed case ran.
- `9056fcb9` carries git's default merge message and no Co-Authored-By trailer; the reflog shows it committed at 20:53:22
  while the W14b integrator was between gates, which says it did not commit it. Who did is unknown (VS Code's Commit on
  a staged merge produces that message) [inferred]. Its tree is the one the integrator resolved and gated.
- Integrator logs, last green at `c115e837`: clang ctest 95 = 92 + 3 MAY_SKIP; CUDA tree 94 / 0 (test_cuda_correctness
  ran); pytest 1095 / 19 / 0; `test_conformance.py` 2 passed; Arrow tree 98 / 0 (test_io_readers 913 assertions in 18
  cases); matlab_suite 142 run, 141 passed, 1 incomplete. CLI byte identity held at every merge (W7d/W7ef 31 runs, W8b 25,
  W10 12 CPU + 10 GPU).

## Reported by agents, unverified

- W7d: shipped `dtwc_cl` DP loops that reload a pointer per cell 63 → 0 (`ab72231`); WDTW band 0.967 / 0.927 under load.
- W7ef: Soft-DTW peak working set 3,430.9 → 13.9 MB (8 × 8,000 samples, 8 threads); Interpolate 2 → 0 allocations per
  pair (its timing band failed once under load, A/A 8 %); WDTW 0.872 / 0.841.
- W10: Metal compiled blind; its macOS CI list is in its report (scratchpad `pending_records_1001d.md`).
- W14b: the configure-error probes are in `.claude/baselines/2026-10-01-w14b-configure-errors.md`; the default MEX
  imports no `gurobi130.dll` (`build/mex` still caches Gurobi ON, so it was not re-measured there).

## Decisions

- Volkan 10-01 (CHARTER): the bindings ship the essentials, no file readers (L2b); Python solves the MIP with the user's
  highspy, the MEX keeps HiGHS (M1); each `dtwc/` folder gets its own CMakeLists (done, L2a).
- Orchestrator rulings in DECISIONS §3 (Volkan may veto): cache identity by computed precision, applying his FP ruling;
  Metal's Auto is FP32; `ab72231` kept; out-of-range enums refused; `k` required (v1.0.0 exited 0 having clustered
  nothing); Arrow without Parquet is an IPC-only build that says so; MATLAB without its executable stops a test build.
- Awaiting Volkan: MATLAB key naming at W9e (CamelCase vs the Python names); the 41 headers no folder file lists (L2b);
  `scripts/slurm/build-arc.sh` defaults Arrow ON (now a configure error if ARC's Arrow is unusable); clang-cl wheels;
  VERSIONINFO for `dtwc_cl.exe`; whether `9056fcb9` was his.

## Next steps (PLAN)

1. E: merge W9a (Problem::cluster() resolves `auto` through `distance_strategy_`, which W10 deleted: re-point it at the
   device surface); W8c running → merge.
2. E: L2b (`dtwc_core` vs io/CLI; the bindings' file I/O moves to Python and MATLAB); M1 (highspy); W9b → W9c → W9e → W9f
   (Python and MATLAB still default `method='pam'`; W9a made C++ `auto`).
3. G: W14a (tier pages; migration page: 2.0 writes `<name>_labels.csv`… into `./results`, v1 wrote `<name>_Nc_<k>.csv`
   into `.`; the hidden v1 spellings in prose), then W14c.
4. Sweep: tracker ids, tests that narrow `index_t`, `run.cpp`'s redundant `std::as_const`. HEAD baseline on a quiet
   machine; W4e Metal; GPU CLARA assignment; lead: 16 double lanes; after G: W13c–e (W13d belongs with M1).

## Open questions

- macOS CI is the only gate of the Metal edits (W4d, W13a, W10); Linux CI builds Debug without the arch flag.
- Disk left by agents for Volkan to remove: detached worktrees `C:/D/git/wt/{W7ef-base,W8b-base,W9a-base}`, build dirs
  inside `wt/W7ef`, `wt/W9a`, about 40 worktrees under `C:/D/git/wt/`.

## Status honesty

Windows only (clang; MSVC for the MEX and CUDA). Not run: macOS/Metal, native Linux, CI. Timing under load is [inferred];
counters decided. Every merge ran the full gates its unit reached; the W14b gates ran on the committed merge tree.
