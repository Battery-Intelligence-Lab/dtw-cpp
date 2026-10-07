# 2026-10-06 — TB: the banded kernel's bounds as arithmetic, timed on x86 (Windows) — PASS on clang++ and MSVC (run 2026-10-07)

Question (Volkan 10-06, DECISIONS §3): on this x86 box, is `dtw_kernel_banded` of `822225bc` (bounds per column) slower
than that of its first parent `f5c58764` (`55911b2a`'s bounds vectors and diag guard)? Registered band (the Mac's, fixed
before any run): the change stays only if no cell is slower than 1.05× (new/old, ns per cell, median of 3 interleaved
rounds), for clang (`build/`) and MSVC (`build/cuda-verify-0928`).

**Result [confirmed], 2026-10-07 23:34–23:36 on a quiet machine: PASS for both compilers.** No cell is slower than
1.05×: new/old 0.79–1.00 with clang++, 0.85–1.02 with MSVC (the largest, 1.02, is MSVC's unequal row at band 20). The
new kernel is faster at small bands under both compilers (clang 0.79–0.95 at bands 1–12, MSVC 0.85–1.01), where the
Mac timed it 1.08–1.20× slower; bands 50 and 200 are equal within 2 %. By Volkan's 10-06 ruling, `822225bc` stays.
All 720 runs were checked bit for bit against the reference (`problems: none`); the shortest timed interval was
68.7 ms. Each version is one build per compiler: the M5's ±30 % code-placement term (DECISIONS 10-07) was not
varied here; both compilers, with different code and layout, agree in direction. HEAD's banded kernel adds
pair-2col's compile-time abandon test to `822225bc`'s; this is the registered historic pair, not HEAD.

Ratios new/old (full tables with ns per cell and abandon fractions: `C:/D/git/wt/tmp/TB/tables.md`, raw `raw.txt`):

### clang: new/old, median ns per cell over 3 rounds

| band | 1 | 2 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 50 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n=1000, no abandon | 0.89 | 0.89 | 0.91 | 0.88 | 0.94 | 0.89 | 0.90 | 0.95 | 0.92 | 0.95 | 0.98 | 0.99 |
| n=3000, no abandon | 0.88 | 0.89 | 0.89 | 0.88 | 0.92 | 0.91 | 0.91 | 0.95 | 0.92 | 0.95 | 0.99 | 0.98 |
| n=1000, abandon | 0.82 | 0.81 | 0.81 | 0.85 | 0.86 | 0.88 | 0.90 | 0.92 | 0.96 | 0.96 | 0.98 | 0.99 |
| n=3000, abandon | 0.81 | 0.80 | 0.79 | 0.84 | 0.84 | 0.89 | 0.91 | 0.92 | 0.98 | 0.95 | 1.00 | 0.98 |
| n=1000 vs 1000-min(b,200), no abandon | 0.90 | 0.91 | 0.93 | 0.91 | 0.92 | 0.89 | 0.90 | 0.92 | 0.89 | 0.95 | 0.99 | 0.98 |

### msvc: new/old, median ns per cell over 3 rounds

| band | 1 | 2 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 50 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n=1000, no abandon | 0.93 | 0.94 | 0.95 | 0.94 | 0.97 | 0.98 | 0.98 | 1.00 | 0.96 | 0.99 | 1.00 | 1.00 |
| n=3000, no abandon | 0.92 | 0.92 | 0.90 | 0.93 | 0.96 | 0.96 | 0.98 | 1.00 | 0.96 | 1.01 | 1.01 | 1.00 |
| n=1000, abandon | 0.88 | 0.89 | 0.89 | 0.90 | 0.94 | 0.93 | 0.97 | 0.97 | 0.97 | 0.98 | 0.99 | 0.99 |
| n=3000, abandon | 0.85 | 0.85 | 0.86 | 0.88 | 0.93 | 0.92 | 0.94 | 0.96 | 0.95 | 1.00 | 0.99 | 1.01 |
| n=1000 vs 1000-min(b,200), no abandon | 0.92 | 0.93 | 0.92 | 0.97 | 0.98 | 0.97 | 0.97 | 1.01 | 0.98 | 1.02 | 1.00 | 1.00 |

## Load (`load.txt`; the 10-06 rows are the first, loaded attempt)

| when | total CPU (`typeperf`, 10 × 1 s) | busiest processes, % of all 24 logical processors |
| --- | --- | --- |
| 01:28:51 (before the first round) | 87.3–99.8 % | 38 `matlab` instances together 87.4 %, each 4.16–4.85 %; all processes 93.3 %; COMSOL none |
| 02:18:41 / 02:19:20 (at the stop) | 6.5–51.6 % | `sspservice` (Sophos) 5.12 %, `mathworksservicehost` 4.01 %, `matlab` 3.69–4.50 % each; 3 s later 38 `matlab` instances 87.9 %, all 92.4 % |
| 2026-10-07 23:34:11 (before the first round) | 2.1–7.1 % | `system` 1.51 %, `spotify` 0.62 %, `serviceshell` 0.54 %, Tanium 0.56 %; 4 idle `matlab` processes; COMSOL none |
| 2026-10-07 23:36:04 (after the last round) | 2.6–4.8 % | `system` 2.15 %, `sspservice` 0.88 %, `spotify` 0.73 %, `sophosfilescanner` 0.67 %; COMSOL none |

At the launch (01:15, orchestrator) the total was 7–15 %. `Get-Process -Name matlab`: 28 processes, the oldest started
2026-10-05 00:52:08, three at 2026-10-06 01:21:15.

## Prepared (all under `C:/D/git/wt/tmp/TB/`)

- Core difference: `git diff --stat f5c58764 822225bc` lists only `dtwc/core/dtw_kernel.hpp` (+8/−25), repo-wide;
  HEAD (`bdc47410`) has the same kernel as `822225bc`. `old/` and `new/` hold `core/dtw_kernel.hpp`, `core/dtw_cost.hpp`,
  `base/missing_utils.hpp`, `base/error.hpp` from each commit (`git archive`); only `dtw_kernel.hpp` differs.
- `bench.cpp`: `run_dtw<SpanL1Cost>(…, StandardCell{}, ea)` (what `dtwBanded<double>` runs for L1) in a noinline
  function called through a volatile pointer; f64, 4 fixed-seed pairs of i.i.d. uniform series (53 random bits, exact,
  so every compiler sees the same doubles); thread pinned with `SetThreadAffinityMask` to logical CPU 12 (a P-core:
  efficiency class 1, scheduling class 1; P-cores are 0, 1, 10–13, 22, 23) at `THREAD_PRIORITY_HIGHEST`; 30 ms
  warm-up, then the repetitions for ~75 ms; cells counted by a cost functor that counts its calls (one per computed
  cell); every timed call compared bit for bit with the reference result.
- Early abandon: threshold = 0.5 × the pair's banded distance (same value in old and new: computed by each binary,
  identical by the hash). Smoke run: columns 493–514 of 1000 (band 5), 1566–1588 of 3000 (band 200).
- Unequal-length row: a pair whose gap exceeds the band has no path (`dtw_kernel.hpp`: `if (n_long - n_short >
  band_width) return maxValue;`, no cell computed), so n vs 0.8 n is empty below band 0.2 n. The row uses n_long =
  1000, n_short = 1000 − min(band, 200): the widest gap each band admits, 0.8 n at band 200.
- Rows: n = 1000 and 3000 without and with abandon (equal lengths), plus the unequal row; bands 1, 2, 4, 5, 6, 8, 10,
  12, 16, 20, 50, 200. `driver.py run`: per cell one process per version, old→new in rounds 0 and 2, new→old in round
  1, both compilers in each round; `driver.py report`: medians, ratios, bold > 1.05, identity checks → `tables.md`.

## Flags (dtw_dispatch.cpp's codegen flags; dtwc_cl.exe's link flags; the project -D/-I dropped, none of the four headers has a conditional)

- clang++ 21.1.8 (x86_64-pc-windows-msvc): `-O3 -DNDEBUG -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
  -flto=thin -march=native -fno-finite-math-only -Werror=switch -fno-math-errno -fno-trapping-math -freciprocal-math
  -fassociative-math -fno-signed-zeros -fno-rounding-math -fopenmp`; link `-O3 -DNDEBUG -D_DLL -D_MT -Xclang
  --dependent-lib=msvcrt -flto=thin -Xlinker /subsystem:console -fuse-ld=lld-link` (`build_clang.sh`).
- MSVC cl 19.50.35723 (toolset 14.50.35717, the tree's): `/nologo /TP /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG
  -std:c++20 -MD /GL /arch:AVX2 /openmp:experimental /we4062 /fp:precise /fp:contract /Gy`; link `/machine:x64
  /INCREMENTAL:NO /subsystem:console /LTCG` (`build_msvc.bat`, vcvars64 as `int/cuda_build.bat`).

## Bit identity (`check.txt`)

`bench check` (lengths 1–40 × 1–40 and {64, 100, 255, 256, 500}², bands −1, 0–6, 8, 10, 12, 16, 20, 50, 200, abandon
off and at 0, ¼, ½, ¾, 1, 2 × the distance): `calls=170625 hash=6e1a304e07df9f75` from all four binaries (old = new,
clang = MSVC). The check bites: `new/` with `T diag = maxValue;` for `col[first_row - 1]` (folder `broken/`, clang)
gives `hash=8f56b1491288be66`.

## Rerun (machine quiet: total CPU low, nothing else above ~5 % of all cores, no COMSOL)

```sh
cd /c/D/git/wt/tmp/TB && sh load.sh "before first round" && uv run --no-project python driver.py run \
  && sh load.sh "after last round" && uv run --no-project python driver.py report
```

About 720 harness runs of ~0.15 s (≈ 2–3 min). Raw lines go to `raw.txt`, tables to `tables.md`.

## History

10-06 01:28 and 03:07: two attempts under Volkan's MATLAB job; the second was stopped at 574 of 720 runs on his
request (`raw.interrupted-0308.txt`, not a result). 10-07 23:34: the rerun above, on a quiet machine.
