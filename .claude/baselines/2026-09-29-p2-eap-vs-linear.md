# 2026-09-29 — P2: EAPruned or linear for unbanded per-pair Standard DTW?

**Question:** after K1 made the linear kernel call-free, is the EAPruned kernel (Herrmann & Webb 2021) still the
faster exact per-pair kernel for unbanded Standard DTW (L1, f64) on real data?

**Answer: no — route to the linear kernel.** The linear kernel is faster on all 7 datasets, within-class and
across-class alike: the pooled median of t_EAP / t_linear is 1.48–3.57 on thread cycles (1.46–3.48 on the wall
clock). Over all 2800 pairs EAP spends 2.38× the linear kernel's cycles. Every result agrees bitwise
(2800 / 2800, plus 400 / 400 for Rock as shipped: EAP = linear = full matrix), so re-routing changes no digit.
EAP wins only a pair it can prune to under about a third of the L² cells: 105 of 2800 pairs (3.8 %), 80 of them
Mallat within-class. A visited cell costs EAP 3.1–3.9× what a cell costs the linear kernel.

## Machine

`uv run --no-project python scripts/machine_facts.py --build-dir build`, run in `C:/D/git/dtw-cpp`:

| Fact | Value |
| --- | --- |
| Host | (omitted) |
| OS | Windows 11 (AMD64) |
| CPU | Intel(R) Core(TM) Ultra 9 285 — 24 cores / 24 threads |
| Memory | 127.5 GiB |
| GPU | NVIDIA RTX 4000 Ada Generation (20475 MiB, compute 8.9, driver 596.72) |
| Compiler | Clang 21.1.8 |
| Build type | Release |
| Build flags | `-O3 -DNDEBUG` |
| Generator | Ninja |
| OMP_NUM_THREADS | unset |
| Repository | design-2.0 @ 59750c4 (clean), VERSION 2.0.0rc1 |

The probe is not the library build: it compiles the library headers with the flags of `dtwc/core/dtw.cpp` from
`build/compile_commands.json`, minus `-flto=thin` and minus the dynamic-CRT trio (`-D_DLL -D_MT -Xclang
--dependent-lib=msvcrt`): `-O3 -DNDEBUG -std=c++20 -march=native -fno-finite-math-only -fno-math-errno
-fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -fopenmp`, plus the
TU's defines and include paths (`build.sh`). One thread, pinned to logical CPU 22, a P-core (`/affinity 400000`;
the probe reads back `affinity mask 0x400000`).

The integration tree moved during the work. The timed build (21:00) read it just after the Y3 merge (`b260415`,
20:58). The kernel files are identical at `59750c4` and at HEAD `a7901df` (`git diff --quiet` on
`core/dtw_kernel.hpp`, `warping.hpp`, `core/dtw_dispatch.cpp`, `core/dtw_cost.hpp`, `core/z_normalize.hpp`,
`core/scratch_matrix.hpp`). The only headers on the probe's include path that changed are `base/settings.hpp`
(adds `index_t`) and `base/missing_utils.hpp` (drops `missing_rate`), neither on a kernel's path.

Load [confirmed, `GetSystemTimes` per round]: the machine was 59–100 % busy per round (78–100 % over each
dataset's timing phase), other processes taking 55–100 % of it (other agents' `lld-link` LTO steps, COMSOL,
MATLAB, Sophos). The probe's thread held CPU 22 for 16–95 % of a round's wall time. The wall clock is
therefore advisory. The thread-cycle ratio is the evidence: per round it moved by ≤ 1 % on every dataset, while
the wall ratio swung from 1.31 to 3.68 (ECG5000).

## Band — registered by the orchestrator before the run

Route unbanded per-pair Standard DTW to the linear kernel if linear is faster on ≥ 5 of the 7 datasets (median
over pairs, pinned P-core); keep EAP if it is faster on ≥ 5; otherwise report the crossover (the pruning rate /
length at which EAP starts to win).

Operationalisation, written here before the first timing run:

- Per pair: each kernel timed once per round, the two interleaved (order alternates with round + pair index),
  7 rounds; per-pair ratio = median over rounds of t_EAP / t_linear (paired within a round). Ratio > 1: linear
  faster.
- Per dataset: "faster" is decided by the median of the per-pair ratio over all 400 pairs (200 within-class + 200
  across-class). Within and across are also reported separately; the total-time ratio (Σ t_EAP / Σ t_linear over
  the pairs' median times) is reported too, and flagged if it points the other way.
- Rock is not z-normalised as UCR ships it (per-series means up to 62, sd 1.5–31; the other six have mean 0,
  sd 1). The brief's premise is z-normalised data, so Rock counts in the verdict after per-series
  `dtwc::core::z_normalize`; Rock as shipped is reported as an extra row that does not count.
- Bitwise agreement: EAP vs linear by `memcmp` on every pair; the full-matrix kernel (`dtwFull`) as a third
  computation; the instrumented EAP copy must equal the library EAP bitwise, or its pruning counts are void.
- Added before the pinned run, after an unpinned smoke run (ECG5000, 2 rounds) found every core at ~100 % load:
  a second clock, `QueryThreadCycleTime`, which counts only the cycles charged to the probe's thread, so time
  slices other processes take on CPU 22 drop out. The verdict is stated on both clocks; if they disagree, the
  thread-cycle clock decides.

## Probe

Sources in `2026-09-29-p2-eap-vs-linear/`: `p2_eap_vs_linear.cpp`, `build.sh`, `run_p2.bat`, `p2_analyse.py`, and `p2_analysis.txt`; the per-pair CSVs and run logs were not kept (the tables below are the record).

- **Timed:** `dtwc::dtwFull_eap<double>(x, L, y, L, metric)`, which `make_standard` binds for `band < 0`
  (`core/dtw_dispatch.cpp:139–140`), and `dtwc::dtwFull_L<double>(x, L, y, L, -1, metric)`, which `dtwBanded`
  forwards to for `band < 0`. Both sit behind a `noinline` wrapper with the metric (L1) read at run time, as
  `p.metric()` is.
- **Third computation:** `dtwc::dtwFull` (full-matrix kernel).
- **Pruning count:** a verbatim copy of `core::dtw_kernel_eap` (`dtw_kernel.hpp:334–425`) with `++visited` in
  its two cell loops, run untimed. Computed fraction = visited cells / L² (a visited cell is one the kernel
  evaluates, whether it survives the prune or not). The copy equals the library EAP bitwise on every pair.
- **Data:** `data/benchmark/UCRArchive_2018/<name>/<name>_TEST.tsv`, first column the label; all values finite,
  all series of equal length. All 7 datasets present; no substitution.
- **Pairs:** per dataset 200 within-class and 200 across-class pairs, uniform over the unordered pairs of each
  kind, drawn without replacement by `mt19937_64` seeded from the dataset name (FNV-1a ^ 20260929). Rock and
  Rock:z draw the same pairs.
- **Clocks:** QPC (10 MHz) and `QueryThreadCycleTime` (TSC reference cycles, 2.494 per ns here). One cycle-clock
  read costs about 2000–2250 cycles (0.9 µs), and it sits inside both kernels' intervals. That shrinks every
  ratio toward 1, in EAP's favour. For ECG5000 (27 µs per linear pair) it understates linear's lead by ≤ 3 % on
  cycles and ≤ 7 % on the wall clock; for the other datasets it is < 0.1 %.

## Commands

```sh
cd .claude/baselines/2026-09-29-p2-eap-vs-linear   # the probe sources
./build.sh                                   # the probe and its -S -masm=intel listing, flags as above
cmd //c 'C:\D\git\wt\tmp\P2\run_p2.bat'
#  start "" /b /wait /affinity 400000 p2_eap_vs_linear.exe C:\D\git\dtw-cpp\data\benchmark\UCRArchive_2018 ^
#        p2_pairs.csv 7 ECG5000 StarLightCurves Mallat InlineSkate HandOutlines CinCECGTorso Rock:z > p2_run.txt
#  start "" /b /wait /affinity 400000 p2_eap_vs_linear.exe ... p2_pairs_rock_raw.csv 7 Rock > p2_run_rock_raw.txt
uv run --no-project python p2_analyse.py p2_pairs.csv > p2_analysis.txt
uv run --no-project python p2_analyse.py p2_pairs_rock_raw.csv > p2_analysis_rock_raw.txt
```

Both runs exited 0 (21:00–21:14). Outputs beside this record: `p2_run.txt` and `p2_run_rock_raw.txt` (per-round
lines and summaries); `p2_pairs.csv` and `p2_pairs_rock_raw.csv` (one row per pair: both clocks, visited cells,
the DTW value, the three bitwise checks); `p2_analysis.txt` and `p2_analysis_rock_raw.txt` (the tables below, the
fit, the totals); `machine_facts.md`. `p2_eap_vs_linear.s` and `.exe` are build products.

## Results [confirmed]

µs per pair: the median over pairs of each pair's median over 7 rounds, wall clock under load (advisory).
Ratio: the median over pairs of the per-pair ratio t_EAP / t_linear (> 1: linear faster). Computed fraction:
visited cells / L², median [p10–p90]. EAP wins: pairs with cycle ratio < 1. Bitwise: EAP = linear = full
matrix, and counted copy = library EAP.

| dataset | L | pairs | EAP µs/pair | linear µs/pair | ratio, wall | ratio, cycles | computed fraction | EAP wins | bitwise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ECG5000 | 140 | within 200 | 83.3 | 27.5 | 3.04 | 3.09 | 0.799 [0.587–0.959] | 0 | 200/200 |
| | | across 200 | 100.2 | 27.9 | 3.61 | 3.71 | 0.971 [0.909–0.998] | 0 | 200/200 |
| | | **pooled 400** | 96.2 | 27.7 | **3.48** | **3.57** | 0.934 [0.648–0.995] | 0 | 400/400 |
| StarLightCurves | 1024 | within 200 | 4478 | 1843 | 2.19 | 2.18 | 0.677 [0.439–0.984] | 2 | 200/200 |
| | | across 200 | 5807 | 1842 | 2.99 | 2.98 | 0.939 [0.617–0.999] | 0 | 200/200 |
| | | **pooled 400** | 5371 | 1842 | **2.70** | **2.66** | 0.842 [0.470–0.994] | 2 | 400/400 |
| Mallat | 1024 | within 200 | 1808 | 1643 | 1.08 | 1.09 | 0.364 [0.250–0.507] | 80 | 200/200 |
| | | across 200 | 3119 | 1648 | 1.85 | 1.85 | 0.592 [0.470–0.653] | 2 | 200/200 |
| | | **pooled 400** | 2482 | 1645 | **1.46** | **1.48** | 0.481 [0.290–0.639] | 82 | 400/400 |
| InlineSkate | 1882 | within 200 | 15855 | 5752 | 2.59 | 2.51 | 0.805 [0.543–1.000] | 0 | 200/200 |
| | | across 200 | 17511 | 5830 | 2.83 | 2.73 | 0.864 [0.561–1.000] | 0 | 200/200 |
| | | **pooled 400** | 16697 | 5771 | **2.67** | **2.60** | 0.834 [0.548–1.000] | 0 | 400/400 |
| HandOutlines | 2709 | within 200 | 19461 | 12009 | 1.60 | 1.59 | 0.513 [0.336–0.684] | 18 | 200/200 |
| | | across 200 | 24499 | 12068 | 1.98 | 1.92 | 0.615 [0.412–0.770] | 2 | 200/200 |
| | | **pooled 400** | 22213 | 12038 | **1.80** | **1.77** | 0.568 [0.363–0.745] | 20 | 400/400 |
| CinCECGTorso | 1639 | within 200 | 12950 | 4066 | 3.19 | 3.18 | 1.000 [0.833–1.000] | 0 | 200/200 |
| | | across 200 | 13003 | 4065 | 3.20 | 3.18 | 1.000 [0.999–1.000] | 0 | 200/200 |
| | | **pooled 400** | 12977 | 4065 | **3.20** | **3.18** | 1.000 [0.993–1.000] | 0 | 400/400 |
| Rock, z-normalised | 2844 | within 200 | 34827 | 12278 | 2.81 | 2.81 | 0.892 [0.606–1.000] | 1 | 200/200 |
| | | across 200 | 35745 | 12267 | 2.93 | 2.90 | 0.923 [0.787–0.998] | 0 | 200/200 |
| | | **pooled 400** | 35477 | 12272 | **2.89** | **2.86** | 0.908 [0.697–0.999] | 1 | 400/400 |
| *Rock as shipped (not counted)* | 2844 | within 200 | 35127 | 11934 | 2.92 | 2.91 | 0.931 [0.704–1.000] | 1 | 200/200 |
| | | across 200 | 33606 | 11954 | 2.82 | 2.78 | 0.886 [0.785–1.000] | 0 | 200/200 |
| | | *pooled 400* | 34032 | 11938 | *2.85* | *2.84* | 0.905 [0.756–1.000] | 1 | 400/400 |

Total-time ratio (Σ over the pairs' median times, EAP / linear, pooled), cycles / wall: ECG5000 3.329 / 3.247,
StarLightCurves 2.492 / 3.185, Mallat 1.448 / 1.476, InlineSkate 2.567 / 2.789, HandOutlines 1.737 / 1.789,
CinCECGTorso 3.117 / 3.154, Rock:z 2.755 / 2.813 (Rock as shipped 2.788 / 2.815). None points the other way.
z-normalising Rock changes nothing that matters here (pooled 2.86 against 2.84 as shipped).

## Verdict

| subject | result | band | verdict |
| --- | --- | --- | --- |
| datasets where linear is faster (pooled median, cycles) | 7 of 7 | ≥ 5 → route to linear | **route to linear** |
| same, wall clock | 7 of 7 | — | agrees |
| bitwise agreement | 2800 / 2800 (+ 400 / 400 Rock as shipped) | a mismatch is a defect | no defect |

The closest dataset is Mallat, pooled 1.48; even its within-class half, the best case for pruning here, is
1.09. Re-routing is a pure speed change: every distance is bit-identical, as `test_eap_dtw.cpp`'s BAND-EXACT
already requires. The call sites are `core/dtw_dispatch.cpp:139–140` (`make_standard`, univariate) and
`warping.hpp:662, 684` (`dtw_independent_mv`, per channel).

## Crossover [confirmed]

Median cycle ratio against the computed fraction, all 2800 pairs of the 7 datasets:

| computed fraction | pairs | median t_EAP / t_linear | EAP wins |
| --- | --- | --- | --- |
| < 0.05 | 1 | 0.004 | 1 |
| 0.10–0.15 | 1 | 0.389 | 1 |
| 0.15–0.20 | 4 | 0.536 | 4 |
| 0.20–0.25 | 22 | 0.686 | 22 |
| 0.25–0.30 | 34 | 0.833 | 34 |
| 0.30–0.35 | 62 | 0.970 | 43 |
| 0.35–0.40 | 111 | 1.147 | 0 |
| 0.40–0.50 | 201 | 1.410 | 0 |
| 0.50–0.60 | 331 | 1.760 | 0 |
| 0.60–0.80 | 605 | 2.211 | 0 |
| 0.80–1.00 | 1428 | 3.178 | 0 |

**EAP starts to win when it visits fewer than about a third of the L² cells.** A least-squares line through the
327 pairs between 0.20 and 0.45 crosses a ratio of 1 at 0.327. The two kernels trade wins only between 0.314
(the lowest fraction linear wins) and 0.348 (the highest EAP wins); EAP wins every pair below 0.30 and none
above 0.35. The per-cell costs give the same threshold independently. Per dataset
(thread cycles with one clock read of 2246 cycles removed, median over pairs), f\* = linear cycles per cell ÷ EAP
cycles per visited cell:

| dataset | L | EAP cycles / visited cell | linear cycles / cell | f\* |
| --- | --- | --- | --- | --- |
| ECG5000 | 140 | 12.83 | 3.27 | 0.255 |
| StarLightCurves | 1024 | 13.06 | 4.07 | 0.312 |
| Mallat | 1024 | 11.91 | 3.89 | 0.327 |
| InlineSkate | 1882 | 12.43 | 3.94 | 0.317 |
| HandOutlines | 2709 | 12.13 | 3.87 | 0.320 |
| CinCECGTorso | 1639 | 11.97 | 3.76 | 0.314 |
| Rock, z-normalised | 2844 | 11.71 | 3.72 | 0.318 |
| *Rock as shipped* | 2844 | 11.43 | 3.62 | 0.317 |

(Cycles are TSC reference cycles at 2.494 per ns. The linear kernel's 3.3–4.1 cycles are 1.3–1.6 ns per cell
under this load, against 1.36 ns in the K1 record.)

**There is no crossover length.** The computed fraction depends on the data, not on L. The within-class median
is 0.36 for Mallat (L 1024), 0.51 for HandOutlines (2709), 0.68 for StarLightCurves (1024), 0.80 for ECG5000 (140)
and for InlineSkate (1882), 0.89 for Rock (2844), and 1.00 for CinCECGTorso (1639): on CinCECGTorso the diagonal
upper bound prunes nothing for the median pair. The pairs EAP wins are 80 Mallat within, 18 HandOutlines within,
2 each of Mallat across, HandOutlines across and StarLightCurves within, and 1 Rock within. That Rock pair is
(28, 24), DTW = 0: two identical series, as shipped and after z-normalisation. The bound is then 0, and EAP
visits 0.1 % of the cells (8540, about three per row) and runs 260× faster than linear.

## Why [inferred from the listing and the per-cell costs]

EAP's inner loop as the probe compiles it (`p2_eap_vs_linear.s`, `dtw_kernel_eap<double, …L1Dist…>`), one visited
cell. It makes no call; the kernel's 4 calls are outside its loops:

```asm
    vmovapd xmm4, xmm0              ; up = max
    cmp     r12, r11 / jb …          ; s >= prev_lo ?   (bounds branches: 2 for up, 3 for diag)
    cmp     r12, rcx / jae …         ; s <  prev_hi ?
    vmovsd  xmm4, [rax + 8*r12]      ; up = prev[s]
    …                                ; diag = prev[s-1] behind 3 more branches
    vminsd  xmm3, xmm3, xmm4         ; m = min(left, up)      <- the carried `left` enters here
    vminsd  xmm4, xmm5, xmm3         ; m = min(diag, m)
    vucomisd xmm4, xmm0 / jne / jnp  ; m == max ?
    mov     r13, [rdi]; mov rbp, [rdi+8]   ; cost lambda's captures, reloaded
    vmovsd / vsubsd / vandpd         ; |x - y|
    vaddsd  xmm3, xmm3, xmm4         ; d = m + cost
    vucomisd / cmovae ×2 / cmova     ; first_live / last_live bookkeeping
    vcmpnlesd xmm4, xmm3, xmm6       ; d > thr ?
    vblendvpd xmm3, xmm3, xmm1, xmm4 ; left = d > thr ? max : d  -> the next cell's `left`
    vmovlpd [r10 + 8*r12], xmm3      ; curr[s] = left
    cmovb / cmp / je / test / je     ; continue (s < prev_hi or d <= thr) or break
```

The carried value `left` passes through five dependent floating-point operations per cell (`vminsd`, `vminsd`,
`vaddsd`, `vcmpnlesd`, `vblendvpd`). The linear kernel's carried value passes through two (`vminsd`, `vaddsd`;
see the K1 record's listing), in a loop unrolled ×4 with no data-dependent branch. The measured 3.1–3.9× cost per
cell fits that chain plus the per-cell branches. When EAP was adopted (2026-07-08; LESSONS: "near-diagonal pairs
gain 6–12×, unrelated pairs ~1.5×"), the linear kernel called `__std_min_d` in every cell: 7.2–9.5 ns per cell
at K1's base. K1 made the linear kernel 7.0× faster, but the fill that runs EAP only 1.36× faster, which
reversed the ranking.

## Not covered

- f32, SquaredL2, unequal lengths and `dtw_independent_mv`'s per-channel calls were not measured. The per-cell
  costs should carry over [inferred]; the pruning will differ.
- Could EAP be rewritten to win? One way: split each row into its live segments (no per-cell bounds tests) and
  take the prune test off the carried chain. At the within-class computed fractions measured here (median
  0.36–1.00), such a kernel would need to cost ≤ 1/f times the linear kernel's cell, i.e. within 1.0–2.8× of it,
  to break even. Unmeasured: this is the next decisive test if pruning is to stay in the exact fill.
