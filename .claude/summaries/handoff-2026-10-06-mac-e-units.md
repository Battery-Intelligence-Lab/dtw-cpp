# Handoff — 2026-10-06 (Mac) — M1, W9e, W9c, L2b merged; lrcore measured; kernel assembly audit; wheel at -O3

## Base

Branch `design-2.0` on the Mac, base `33302535` (Volkan's push after the Windows session); HEAD = this file's commit,
57 commits ahead, tree clean after it. Volkan pushes; Windows then `git pull --no-rebase`. Read the Windows handoff
`handoff-2026-10-06-w9b-and-mac-merged.md` too (its next steps 1 TB and 5 follow-ups still stand).

## Done (each unit one `pb/<unit>` branch merged `--no-ff`, evidence on the main tree, worktree removed)

- Base check at `33302535` (`61903cc2`, `baselines/2026-10-06-macos-after-w9b.md`): build zero warnings, ctest 95/95
  + CUDA skip, conformance the one silhouette ulp (D-19), docs gates, pytest 970/12/0, ASan+UBSan 94/94 no report.
- M1 `c0580948` + review `bb35d337` (Opus): one `build_p_median_model` for linked HiGHS and highspy; the wheel links no
  HiGHS (5.03 → 1.20 MB), `method="mip"` needs the `mip` extra, else SolverError; the MEX links HiGHS statically.
- W9e `bf82dc1b` (Opus): MATLAB sets a dtwc::Config, apply() checks, Problem::cluster() runs; CamelCase keys; the MEX
  links no run()/api; matlab_suite 142/141. ◐ until Windows R2024b runs matlab_suite (its brief stays).
- W9c `834b7904` (Opus): `hpc` crosses as one `job.toml` after C++ apply() checked every value; `submit-job` with
  `--gpu-device`; one GPU table; the merge needed `f7c04420` (M1's test_mip used a helper W9c deleted).
- L2b `15b289be` (Opus): `dtwc_core` / `dtwc_cli` / `dtwc_io` behind an INTERFACE `dtwc++`, headers in FILE_SETs; the
  core reads text only, `dtwc::run` dispatches Arrow to `io::read_arrow`; MEX and extension byte-identical.
- lrcore without HiGHS measured (`3cb49bc8`, Sonnet): same answers; Kelley wins only on a line metric. The second Mac
  pass after L2b: ctest 95 with the Metal tests, the 12 GPU routes 48/48 files byte-identical.
- Kernel assembly audit (`5f8b8357`, `baselines/2026-10-06-mac-kernel-assembly.md`, Opus, read-only), for Volkan's
  "Have you checked the assembly created for macbook and it is the best performance it could be?": the CLI, the wheel
  and the MEX carry the same loops; `-march`/`-mcpu` change none; the lanes kernel (0.92 cycles/cell f64) sits on its
  7-cycle `fcmgt→bif→fadd` chain, the per-pair kernels on `fcmp→fcsel→fadd`; no call or trap in a DP loop.
- `659f889f` the wheel's binding file at `-O3` (`NOMINSIZE`): LTO had run nanobind's `-Os` copy of the per-pair
  kernel in the whole module; Python `dtw` 1.12–1.50×, ragged fill 1.52×, equal fill 0.99×, `.so` +4.1 %.
- Records: PLAN (M1, W9c, L1/L2, NOMINSIZE ☑; W9e ◐; two kernel rows ☐ under "After G"; ARC and Windows under
  "Blocked"), DECISIONS §3 (ten 10-06 lines), LESSONS (the LTO lesson names the `-Os` copy), MAP §3, contract.

## Verified by me

Every merge: build zero warnings, `ctest -j1` 95/95 + 1 skip, conformance the same ulp, the three docs gates, pytest
from a fresh venv (974/11/0 after M1 and W9e; 926/11/0 after W9c and L2b), matlab_suite 142/141/0/1 after W9e and
L2b. The audit: opened the lines its answer rests on (python/CMakeLists.txt:29,31; nanobind-config.cmake:332,610;
dtw_lanes.cpp:48-50; Problem.cpp:529, 697-712, 843; warping.hpp:14-21; `asm/lanes_shipped.s`); re-ran kbench (lanes
`fminnm` 1.29–1.45×, + 16 lanes 1.42–1.86×, per-pair two columns 1.49×) and the Python timings. The fix: a fresh
verbose wheel build (no `-Os` on the binding line), pytest 926/11/0 with `DTWC_REQUIRE_HIGHSPY=1`, docs gates PASS.

## Reported by agents, unverified

M1: model equivalence for six (N, k); the highspy crash beside a HiGHS-linked extension (cause inferred). W9e: Parquet
equals pyarrow; the Windows UTF-8 fix reasoned. W9c: eight mutation checks; ARC node tags. L2b: the Arrow-shim ctest
and 16 Arrow CLI runs. Audit: the bitwise sweep (every variant's hash equal), the M5 latency table (4 FP pipes;
`fcmgt`+`bif` 4 cycles, `fminnm`→`fadd` 5), the fill gains at 18 and 6 threads, the x86 `fmin` lowering
(cross-compiled only), two open anomalies (4.3 FP ops/cycle; banded per-pair placement), `bench_dtw_baseline` stale
(09-22).

## Decisions

Taken by Volkan this segment: none ("Okay I did the things on other machine, please continue here", then the assembly
question). Proposed and awaiting him: accept the HiGHS-free wheel's subgradient lrcore root; W9c's
`DTWC_NATIVE_CPU=OFF` for `--gpu-device` builds narrows the 09-30 ruling; W9b's `_NotSeries`; AArch64 lanes with
`fminnm` and 128-byte blocks (1.41–2.00× single thread, fill 1.48–1.72×) and two-column per-pair kernels (1.44–1.98×
unbanded), both bitwise and recommended.

## Next steps

1. Windows: pull; re-run the clang/CUDA/Arrow trees (`build/arrow-pyarrow-23`, the L2b Arrow-ON proof), pytest,
   matlab_suite R2024b for W9e (then delete its brief); TB when quiet. 2. If Volkan agrees: "After G"'s two kernel
   rows as units, the AArch64 lanes first (x86 untouched; conformance digit-identical; `test_codegen_no_calls`).
3. E: W9f (restores the `read_distance_matrix` / `print_distance_matrix` rows). 4. G: W14a, WM, W14c. 5. Candidates:
   OpenMP on `mip-solvers`; `lr-core.md:231,245` and `lagrangian_root.cpp:585` overstate Kelley; Python's names after
   `skip_rows` and of Parquet rows differ from C++; `load(Dataset, **options)` ignores options; `set_solver` prints
   before apply() raises; `test_conformance.m` not in matlab_suite; HiGHS caches `BUILD_SHARED_LIBS=ON`.

## Open questions

- `.claude/CLAUDE.md:71`'s Python gate should add `mip` to its extras (Volkan's file). Still his: the ARC leg, TB on
  Windows, 20 stale September worktrees, `build-asan/`, whether the MEX ships (no licence column).
- Do the Linux and Windows wheels also run the `-Os` copy? `NOMINSIZE` covers them either way (MSVC gets `/Os`).
  My trailers said Fable until the `/model` switch, Opus 5.5 after (harness attribution).

## Status honesty

Mac only (`build/`, `build-matlab/`, fresh-venv wheels). Not run here: CUDA, Linux, CI (seven workflow edits parsed
only), Windows, ARC, Gurobi, an IPC-only Arrow build, x86 timing of the audit's variants, the Metal kernels' code (no
GPU disassembler in the Command Line Tools). ctest was not re-run for `659f889f`: no C++ tree compiles the module.
