# Handoff — 2026-10-07 (Mac) — "run whatever is left": the After-G kernels, E and G, Parquet rows, one 2.0.0 section

## Base

`design-2.0` on the Mac from `33302535` (Volkan's last push); HEAD = this file's commit, about 140 ahead, tree clean.
Kept `handoff-2026-10-06-mac-e-units.md`; the Windows 10-06 handoff is deleted, its open items carried below.

## Done (each unit a `pb/<unit>` branch merged `--no-ff`, evidence on the main tree, worktree removed)

- check-docs-reverse `39249aa0`: every flag `dtwc_cl --help` prints has a table row in `getting-started/cli.md`.
- arm-lanes `e26d5680`: on `__aarch64__` the lanes' min is `fminnm` (`LanesCell`) and W is 128 bytes; bitwise; x86 `.s`
  identical; quiet run 1.41–2.00× single thread, 1.61–1.69× 18-thread fill (`c652c9ca`).
- W13c `b36fad43`: one `core::kmedoids_pp`; `fast_pam(…, seed = 42)` absorbs `fast_pam_seeded` in C++, Python, MATLAB;
  the v1 one-argument `init::*` sequences changed as pre-registered. W13d had been done by M1.
- W9f `0e0f3056`: Python tests −1,043 lines (each removal named with its cover), `test_test_api` in C++ only, the
  examples run under pytest and ctest, matlab_suite allows no incomplete.
- W13e `32b219c6`: barycenter scratch thread_local, series read in place, `BarycenterOptions` embedded,
  `Problem::copy_distance_settings_from`.
- W14a `e7f6153a`: hand-written tier-1, tier-2 and migration pages; `api-contract-2.0.md`, `docs/sources/` and the rc1
  page deleted; lr-core.md states Kelley as measured.
- PF `30284994`: Python names series as C++ does; `load(Dataset, opts)` refuses; `set_solver` prints nothing;
  test_conformance.m in matlab_suite; no cached BUILD_SHARED_LIBS; CI installs matplotlib.
- PQ `b2687171` (Volkan 10-07): several float columns → one series per row, the first string column names rows (list
  layout too), skips as for CSV, the stream included; a sliced Arrow list read from its offset; the Arrow-25 test fixed.
- pair-2col `0edc388b`: the unbanded per-pair kernel computes two columns per pass (f64 L1 1.42–1.98×, ragged fill
  1.42–1.50×, every placement); the banded form FALSIFIED (0.85×) and reverted.
- lr-omp `57ad9ad2`: `mip-solvers` gets OpenMP; the dual's first loop forks from N = 280; at N ≥ 800 the subgradient
  root (the wheel) 3.5–7.3×, Kelley 1.17–1.49×; bit-identical.
- W14c `ae2df68c`: CHANGELOG is one `2.0.0 (unreleased)` section for v1.0.0's users (1,814 → 255 lines), MAP redone.
- FU `8d969203`: per-pair `distance::*` refuse an empty series; a negative skip is refused in every format; per-language
  Parquet errors; `std::as_const`, C4244, tracker ids in comments, stale texts. Records: PLAN, DECISIONS, LESSONS.

## Verified by me

After every merge, on the main tree: build 0 warnings; serial ctest all pass (95 → 94: the barycenter allocation test
went); conformance = D-19's ulp only; the three docs gates; matlab_suite 142/141 → 136/136 → 137 → 138/138/0/0; pytest
from a fresh venv 923/11 → 890 → 891 → 895/11/0 (the last after FU, `8d969203`). I ran the
arm-lanes kit on the quiet machine (every band PASS); from pair-2col's data, 64-byte loop alignment is no cure (best
unaligned/aligned 0.73–1.03); split `fcmp`/`fcsel` pairs in the shipped `dtwc_cl`. Opened: `kmedoids_pp`, `LanesCell`,
kernel 1's two-column loop (each cell's neighbours and roles), `resolve_parquet_layout`, lr-omp's diff, W14c's text.

## Reported by agents, unverified

arm-lanes' 15.36 M lane outputs and x86 identity; pair-2col's 11.73 M outputs over every Cell and its phase-2 timing;
lr-omp's 330 + 54 comparisons and its sweeps; PQ's three-language parity and Arrow-tree ctest 97/97 (pyarrow-25 shim);
W13e's 399/400 hex-float probes; W14c's adversarial checks of each claim; FU's before/after errors per language.

## Decisions

Taken by Volkan: 10-06 "okay run whatever is left sure"; 10-07 on Parquet "read each row as series I think or have some
option right? It is probably not rare to have multiple time series in the same file", rows named by the "First string
column". Awaiting him: the 10-06 handoff's three (lrcore's subgradient root — now 3.5–7.3× faster —, W9c's
`DTWC_NATIVE_CPU`, W9b's `_NotSeries`); a v1.0.0 matrix CSV's `-1` entries (refuse, or read as uncomputed? the
migration page warns); deleting the audit folder.

## Next steps

1. Volkan: push. 2. Windows (G, WM; E, W9e): pull; rebuild the clang, MSVC, CUDA and Arrow trees — PQ's reader on
   Arrow 23 (`build/arrow-pyarrow-23`, test_io_readers listed), the bindings (pytest, matlab_suite R2024b for W9e),
   FU's C4244 casts; TB when quiet; WM. 3. x86 lanes: `fminnm`/W 16 is AArch64-only; x86 needs its own band (PLAN C).
4. Windows follow-ups carried: argv in the ANSI code page (`δ.csv` arrives as `d.csv`: CLI11 `ensure_utf8` or `wmain`);
   `DataLoader.hpp:106` throws on a non-ANSI extension. 5. Candidates: 87 F8/F13/F14 failure-message prefixes and
   their markers; `Problem::dtw_function()(x, empty)` and a series emptied through `p_vec` then a CPU fill return
   max(); barycenter keeps its O(L²) scratch per thread after a call (512 MB at L 8,000).

## Open questions

- `.claude/CLAUDE.md` (Volkan's): :78 names the deleted contract; step 3 points into the audit folder; :71's Python
  gate lacks the `mip` extra, matplotlib and pandas.
- Trailers: the Sonnet units (check-docs-reverse, lr-omp) say Claude Sonnet 5.5, their true model; the rest Opus 5.5.
- From 10-06: 20 stale September worktrees, `build-asan/`, whether the MEX ships (no licence column).
- On the M5 a per-pair loop's speed moves ±30 % with code placement; a single build's banded timing is not evidence.

## Status honesty

Mac only (`build/`, `build-matlab/`, fresh-venv wheels; Arrow only in agent trees through the pyarrow-25 shim). Not
run: Windows (MSVC, clang-cl, CUDA, R2024b), Linux and GCC (pair-2col's bitwise claim is Apple clang's), CI (the
matlab-mex.yml and python-tests.yml edits are parsed only), ARC, Gurobi, x86 timing of any kernel change. The kernel
timings ran on a quiet Mac with Sophos and Tanium using about one core: arm-lanes and pair-2col on battery (clock
4.5 GHz), lr-omp on AC.
