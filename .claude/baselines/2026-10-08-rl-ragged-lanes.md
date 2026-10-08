# 2026-10-08 — RL: ragged lanes (without a band the CPU fill's lanes take series of different lengths)

Unit RL, from design-2.0 `e691e48c` on `pb/RL`: `9d2b9310` (lanes, and the first candidate's fill A), `bffc21c9`
(docs), `356bef7e` (the candidate B's fill, the one timed), `0b5cfe42` (after the timing: cl keeps the one-pair route
for blocks of unequal length; the timed figures in CHANGELOG and dtw.md), `f8c3d23a` (B3: no capture test in the
kernel's column loop, which cost clang 4–5 % on long equal-length series). Windows 11, Intel Core Ultra 9 285; clang++ 21.1.8 (configured as the
main `build/`), cl 19.50 (MSVC 14.50, as MV's `build-msvc`), GCC 13.3 in WSL Ubuntu 24.04 (Release,
`DTWC_ARCH_LEVEL=v3`, as XG). Prepared on a shared machine by the unit (Opus 5.5); timed on a quiet one by the
orchestrator (B at 20:03, B3 at 23:14). Kit, sources and evidence: `C:/D/git/wt/tmp/RL/`; its text evidence is copied beside this record into `2026-10-08-rl-ragged-lanes/` at the same relative paths (the ETW results in `etw/`), while binaries, objects and full listings stay in the kit.

**Question.** Today a block of W columns goes through the lanes only if all W series are as long as the row's; any
variable-length data runs the one-pair kernel, 2.5–4× slower per cell under clang and about 5× under GCC. If each lane
keeps its own length (unbanded), is every lane still the per-pair distance, does the inner loop keep its code, and is
the fill faster by the registered band?

**Registered band (the orchestrator's, fixed before any run, verbatim):** "The change is adopted only if, under clang++
(Windows, the main build/'s flags) and under GCC 13.3 (WSL, the Linux release flags), each separately: the unbanded f64
L1 ragged fill N 1000, L 90-110 is >= 2.0x base pinned on one thread and at 24 threads; the unbanded f64 L1 wide ragged
fill (lengths 100-1000; N 500 at 24 threads, N 150 pinned on one thread) is >= 1.5x base at both; the f32 L1 and the f64
squared variants of the N 1000 L 90-110 fill at 24 threads are >= 1.5x base; the equal-length fills (N 2000 L 100
unbanded; N 500 L 1000 band 100; 24 threads) and the banded ragged fill (N 1000, L 90-110, band 30: the one-pair route)
stay within 0.97-1.03x; the outputs as stated under PROOF. Under cl (MSVC tree flags; the release archive keeps a cl
build) no timed shape falls below 0.97x base. Otherwise FALSIFIED (for that compiler, if the failure is
compiler-specific; say so)." Added by the orchestrator's review, before any timing (verbatim): "Make the candidate
row-local instead (call it B) ... B is the candidate, decided now on locality grounds and fixed before the run." and
"Add one shape to the kit for base, A and B: an unbanded f64 L1 ragged fill with N 8000 and lengths 40-60 at 24 threads.
Its matrix is 256 MB, well beyond L3, so write locality shows up in wall time; report Gcell/s and each build's ratio to
base. Registered for B on that shape under clang++ and GCC (fixed now): ≥ 1.5× base. Every other band clause applies to
B unchanged."

**Status: B (`356bef7e`) TIMED 2026-10-08 20:03: clang PASS; GCC FALSIFIED by one control (0.969×); cl FALSIFIED on
ragged data (four shapes at 0.83–0.97×). Adopted by Volkan's ruling for clang and GCC, cl keeping the one-pair route for
blocks of unequal length (`0b5cfe42`). The orchestrator then found B 4–5 % slower than base under clang on equal-length
L 8000 series on one thread; cause found and fixed in `f8c3d23a` (B3; below). B3 TIMED 2026-10-08 23:14, 11 shapes:
clang PASS 11/11, GCC PASS 11/11; cl one shape below its floor in the 3-round run (f32 ragged at 24 threads, 0.940×; under
cl that shape takes base's route) and 1.059× over a registered 20-pair second attempt, so cl runs at base's speed.
Merged into design-2.0 as `8c818735`.**

## The change (`pb/RL`)

- `9d2b9310` `dtwc/core/dtw_kernel.hpp`: `dtw_kernel_lanes<T>(std::span<const T> xs, const std::span<const T> *ys, int
  band, Dist, Cell)` (was `(const T *x, const T *const *ys, std::size_t n, ...)`). Lane w has its own length `ny[w] >= 1`;
  `Y` is `[max ny][W]`, a lane past its length repeating its last sample (finite; never read). The column loop is base's,
  run to `max ny`; `take(c)` (after column 0, and after any column `j` with `j + 1 == end`) copies `s[nx - 1]` out for the
  lanes of `c` samples and returns the next length a lane ends at: one compare per column, outside the i and w loops;
  equal lengths take every lane after the last column, as base did. A band needs every `ys[w]` as long as `xs`.
  `dtw_band_bounds` returns every row for `band < 0` whatever the column (it returned none past `column_count`, which the
  ragged lanes reach). `dtw_lanes.cpp`'s block function passes its spans through; `dtw_dispatch.hpp` says what `ys` may
  hold. `one_batch_pam.cpp`: the batch table takes ragged blocks through the same block function — one line
  (`lanes = prob.band < 0 || ys[w].size() == x.size();`) and its comment; its columns stay in batch (random) order, so
  its blocks waste what the fill's index order would; no band covers it. The lanes test: one section, "lengths of their
  own, unbanded" (lengths drawn in [1, n], n 1–1000, L1 and squared, ties or not, f64 and f32, each lane against
  `dtwBanded(-1)` within the route bound); "an empty x" for the empty case; two fill cases renamed.
  `scripts/codegen_probe.cpp`: the new signature. That commit's fill was A (below).
- `356bef7e` (B, the candidate): `Problem::fillDistanceMatrix_BruteForce`'s rows are base's again, in index order, row i
  against the series after it, the per-pair loop unchanged; only the lanes' blocks change. Without a band (and with a
  lane function) a row takes its columns in one longest-first order (`order`: a stable sort by length, built once per
  fill, serially), passing over those up to its own; a block goes to the lanes unless a band is set and its lengths
  differ (or all its pairs are known). If the order is the index order (a band, no lane function, equal lengths), a row
  starts at i + 1 as base did. A thread's chunk of consecutive rows i.. then writes (j, i..), neighbouring slots of the
  packed matrix, as at base (`run_openmp`'s 8 is chunks per thread: a chunk is N/(8·threads) rows, 41 at N 8000 on 24
  threads; corrected 2026-10-09, this record said rows i..i+7); A (`9d2b9310`: rows and columns longest first, row a =
  series order[a]) gave a chunk unrelated rows, each write its own cache line.
- `0b5cfe42` (after the timing, Volkan's ruling): `core::dtw_lanes_ragged`, a `constexpr bool` beside `dtw_lanes` under
  the same `#if defined(__aarch64__) || defined(__x86_64__)`: true there (clang, clang-cl, GCC), false under cl and
  elsewhere; its comment: cl's lanes are scalar, and on blocks of unequal length they lost 9–17 % to the one-pair
  kernel. The fill's ragged permission (`ragged = core::dtw_lanes_ragged && distance_.band < 0`, which also gates the
  length sort) and OneBatchPAM's condition require it; where it is false the fill is base's (index order, no sort,
  equal-length blocks only). The kernel is unchanged. CHANGELOG and dtw.md carry the timed figures.
- `f8c3d23a` (B3, after the L 8000 report): the kernel's column loop has no test. Each column's last row goes to its
  `Y` row (`Y[j] = s[nx - 1]`; no later column reads `Y[j]`), and lane w's distance is `Y[ny[w] - 1].v[w]` after the
  loop; `take` and its per-column test are gone. The fill is unchanged.
- `bffc21c9`, with `356bef7e`'s wording: `docs/content/method/dtw.md` (which series the lanes take), a CHANGELOG
  Performance line without a figure (to add from the timed run). `bffc21c9`'s message corrects a misprint in
  `9d2b9310`'s (A's cells per useful cell are 1.015–1.124× in length order, not "1.5–1.24×").

## What changed from the brief's idea, and why [confirmed by the assembly below]

- **No lane sort, no run segments.** The first candidate ran the columns in segments ending at each distinct length (the
  brief's "e.g."). cl's loop was unchanged, but GCC's register allocation was not: its f32 L1 and squared loops went from
  37 / 33 instructions with no stack operand to 42 / 41 with 7 (the broadcast `x[i]` stored and reloaded four times, two
  `y` vectors reloaded), its f64 L1 loop gained a folded reload of a `y` vector. A flat column loop with the end test after
  each column gives base's loops under all three compilers.
- **`take` copies `s`.** Captured by reference (variant C), `s`'s address escapes and cl, which has no type-based alias
  analysis, reloads it after every store to the column: +2 instructions and +1 memory operand per row step. By value
  (variant F, shipped) cl's loops are base's instruction for instruction.
- **The columns' order: longest first.** By the brief's count (cells past a lane's length), index order wastes 8.4 % /
  37.5 % / 33.2 % of the lanes' cells on the three ragged shapes at W 16, failing the "about 10 %" test on both wide
  shapes. Longest first (A: rows too) wastes 0.15 % / 2.9 % / 8.9 %; shortest first had less padding but more repeated
  tail lanes (1.018 / 1.068 / 1.222 cells per useful cell against 1.015 / 1.036 / 1.124). B keeps the rows in index order
  (the orchestrator's ruling, on write locality) and costs a little waste on the wide shapes (table below). Banded fills
  keep index order and the one-pair route for mixed-length blocks, so the registered banded-ragged control stays one.
- **Empty series**: not handled in the kernel (an empty `ys[w]` would read before its start); the fill relies on
  `validate_fill_request` refusing one, which unit ES adds for the CPU. RL must merge after ES (or with it).

## Waste counts [confirmed; counts, not timing] (`rprobe waste`; a row's last block repeats its last column)

Cells the lanes compute per useful cell (padding = cells past a lane's length, duplicates = the repeated tail lanes, as
fractions of the lanes' cells); W 16 is f64 under clang and GCC (32 their f32; cl 8 and 16):

| shape | index order (base's grouping) | A: rows and columns longest first | B: rows in index order, columns longest first |
| --- | --- | --- | --- |
| N 1000, L 90–110, W 8 / 16 / 32 | 1.088 / 1.109 / 1.132 (padding 8.4 % at W 16) | 1.007 / 1.015 / 1.031 (0.15 %) | 1.008 / 1.017 / 1.035 (0.29 %) |
| N 500, L 100–1000 | 1.583 / 1.681 / 1.791 (37.5 %) | 1.017 / 1.036 / 1.075 (2.9 %) | 1.025 / 1.054 / 1.113 (4.2 %) |
| N 150, L 100–1000 | 1.600 / 1.740 / 1.972 (33.2 %) | 1.057 / 1.124 / 1.262 (8.9 %) | 1.081 / 1.178 / 1.385 (11.2 %) |
| N 8000, L 40–60 | 1.164 / 1.188 / 1.201 (15.7 %) | 1.001 / 1.002 / 1.004 (0.04 %) | 1.001 / 1.002 / 1.005 (0.07 %) |

OneBatchPAM's batch columns are in batch order, so its ragged blocks waste about the index-order column.

## Expectations [inferred before the run; kept as written]

- clang and GCC: one thread, the lanes run 0.085–0.09 ns/cell (XL's f64 L1, W 16) where the one-pair kernel runs
  0.63–0.72 (TP, HB), 7–8× per cell, so the pinned ragged fills should clear 2.0× and 1.5× by far; at 24 threads XL
  recorded 36 Gcell/s ragged against 101–124 in the lanes. On N 8000 L 40–60 A's longest-first rows should show as
  extra write traffic against B (the orchestrator's mechanism); how much is the run's question.
- **cl: the floor is at risk.** cl's lanes are scalar and no cheaper per cell than its one-pair kernel: f64 L1 lanes
  0.717–0.739 ns/cell (XL, W 8) against the pair kernel's 0.680–0.715 (TP), and XL's 24-thread lanes and per-pair fills
  ran at the same rate (29.5 Gcell/s). With B's 0.8 %–8.1 % more cells (W 8), cl's f64 ragged shapes predict about
  0.94–0.99× at N 1000 and 0.88–0.93× for N 150 pinned, below the 0.97 floor; f32 about 1.05–1.1× (lanes 0.64–0.66
  against 0.71) and f64 squared about 0.98–1.02×. The smoke rounds agreed in direction (tiny shapes, loaded: not a
  measurement). If cl fails, the failure is cl's alone: its lanes would need cl to pack them (MV, FALSIFIED) before
  ragged blocks pay.

## The timed run of B (the orchestrator's, quiet machine, 2026-10-08 20:03:22–20:16:09) [confirmed]

`uv run --no-project python driver.py run` in the kit: 3 rounds, each running clang, cl and GCC, base and B alternating
which goes first (base, B and A rotating on N 8000); 189 probe lines in `raw.txt`, every probe exit 0; 767 s. Load
(`load.txt`: `typeperf`, 10 × 1 s; the busiest processes in % of all 24 logical processors; WSL's side):

| when | total CPU | busiest processes | WSL |
| --- | --- | --- | --- |
| 20:03:26 (before) | 3.0–22.5 % | `system` 0.95 %, `serviceshell` 0.58 %, `vmmemwsl` 0.41 %, `wmiprvse` 0.19 %, `taniumcx` 0.17 %; COMSOL none | up 0 min, load 0.00, no systemd job, 100 % idle |
| 20:15:49 (after) | 2.4–6.3 % | `system` 1.23 %, `taniumclient` 1.06 %, `serviceshell` 0.86 %, `vmmemwsl` 0.45 %, `wmiprvse` 0.24 %; COMSOL none | up 1 min, load 5.60 / 1.91 / 0.68 (the last GCC probes), no systemd job |

WSL was up 1 minute at the end: its VM idles out between rounds (clang's and cl's sections take minutes), so each round's
GCC section started on a freshly booted VM [inferred from the uptimes; not measured]. Five of GCC's pinned probes moved
their clock by more than 3 % over their fills (report below).

### clang (`report_B.md` verbatim)

| fill | W | base ms | B ms | base Gcell/s | B Gcell/s | ratio (median) | range | band | matrix hash base / B | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 3075.78 | 456.63 | 1.615 | 10.881 | 6.765 | 6.735-6.839 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | 5.272 / 5.308 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 142.99 | 38.52 | 34.746 | 128.971 | 3.536 | 3.417-3.712 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 2653.18 | 387.38 | 1.425 | 9.763 | 6.920 | 6.811-6.937 | >= 1.5: pass | `d0c4ea00ede2475c` = B | 1.1780 (0.1120, 0.0391) | 5.324 / 5.298 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1096.51 | 322.50 | 37.842 | 128.663 | 3.387 | 3.343-3.422 | >= 1.5: pass | `cd125a5433bb9769` = B | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 146.92 | 21.60 | 33.817 | 229.985 | 6.727 | 6.185-6.837 | >= 1.5: pass | `d0429e2d299c55ed` = B | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 136.62 | 38.91 | 36.367 | 127.694 | 3.555 | 3.346-3.569 | >= 1.5: pass | `76e1c6ac3e6ce69d` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 160.02 | 159.97 | 124.921 | 124.958 | 0.997 | 0.967-1.096 | 0.97-1.03: pass | `fa485cbe1fa07d20` = B | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 180.79 | 180.03 | 131.725 | 132.284 | 0.998 | 0.997-1.016 | 0.97-1.03: pass | `79637ca9e991a308` = B | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 83.09 | 84.13 | 58.431 | 57.706 | 0.977 | 0.974-1.019 | 0.97-1.03: pass | `bed68cfaa32b7445` = B | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 2597.72 | 705.17 | 30.695 | 113.074 | 3.684 | 3.449-3.897 | >= 1.5: pass | `64d2a0d49855c763` = B | 1.0023 (0.0007, 0.0015) | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B Gcell/s | B ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 30.695 | 112.291 | 3.503 | 3.491-3.839 | 113.074 | 3.684 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B: yes; lanes vs per-pair bitwise mismatches base + B: 0; outside the route bound: 0 -> pass
- ragged sweep (B): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- CLI (w8b_cli_runs.sh + w8b_cli_extra.sh, data/dummy): byte-identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 1
  - round 0 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 4.7-5.264 GHz
- band: 10/10 shapes in band; matrix hashes base == B (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict clang (B): PASS**

### cl (`report_B.md` verbatim)

| fill | W | base ms | B ms | base Gcell/s | B Gcell/s | ratio (median) | range | band | matrix hash base / B | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 8 | 3355.55 | 3691.16 | 1.481 | 1.346 | 0.906 | 0.905-0.968 | >= 0.97: FAIL | `b28673ce59c47ce8` = B | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 8 | 165.16 | 159.49 | 30.083 | 31.151 | 1.041 | 1.015-1.060 | >= 0.97: pass | `b28673ce59c47ce8` = B | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 8 | 2657.55 | 3052.34 | 1.423 | 1.239 | 0.870 | 0.869-0.957 | >= 0.97: FAIL | `d0c4ea00ede2475c` = B | 1.0809 (0.0608, 0.0140) | n/a |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 8 | 1141.98 | 1380.68 | 36.335 | 30.053 | 0.829 | 0.825-0.833 | >= 0.97: FAIL | `cd125a5433bb9769` = B | 1.0246 (0.0207, 0.0033) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 178.10 | 151.93 | 27.896 | 32.701 | 1.149 | 1.145-1.172 | >= 0.97: pass | `d0429e2d299c55ed` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 8 | 164.29 | 159.31 | 30.241 | 31.187 | 1.031 | 1.029-1.115 | >= 0.97: pass | `76e1c6ac3e6ce69d` = B | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 8 | 677.51 | 672.30 | 29.505 | 29.734 | 1.005 | 1.002-1.038 | >= 0.97: pass | `fa485cbe1fa07d20` = B | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 8 | 775.24 | 782.03 | 30.719 | 30.453 | 0.996 | 0.989-1.001 | >= 0.97: pass | `79637ca9e991a308` = B | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 8 | 93.78 | 94.29 | 51.768 | 51.485 | 0.990 | 0.985-1.021 | >= 0.97: pass | `bed68cfaa32b7445` = B | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 8 | 2658.94 | 2753.50 | 29.988 | 28.958 | 0.966 | 0.923-1.039 | >= 0.97: FAIL | `64d2a0d49855c763` = B | 1.0011 (0.0003, 0.0007) | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B Gcell/s | B ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 29.988 | 29.672 | 0.969 | 0.953-1.028 | 28.958 | 0.966 | = base | 1.0009 (0.0002, 0.0007) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B: yes; lanes vs per-pair bitwise mismatches base + B: 0; outside the route bound: 0 -> pass
- ragged sweep (B): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- pinned probes whose clock moved by more than 3% over their fills: 0
- band: 6/10 shapes in band; matrix hashes base == B (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict msvc (B): FALSIFIED**

### GCC (`report_B.md` verbatim)

| fill | W | base ms | B ms | base Gcell/s | B Gcell/s | ratio (median) | range | band | matrix hash base / B | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 3932.29 | 480.59 | 1.263 | 10.338 | 8.216 | 8.088-8.239 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | 5.424 / 5.356 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 200.25 | 39.48 | 24.811 | 125.850 | 5.049 | 4.957-5.420 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 3345.42 | 384.63 | 1.130 | 9.833 | 8.743 | 8.627-8.822 | >= 1.5: pass | `d0c4ea00ede2475c` = B | 1.1780 (0.1120, 0.0391) | 5.322 / 5.283 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1802.25 | 322.79 | 23.023 | 128.549 | 5.576 | 5.543-5.661 | >= 1.5: pass | `cd125a5433bb9769` = B | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 123.60 | 24.20 | 40.198 | 205.288 | 5.107 | 4.808-5.265 | >= 1.5: pass | `d0429e2d299c55ed` = B | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 205.74 | 45.29 | 24.148 | 109.711 | 4.481 | 4.320-4.547 | >= 1.5: pass | `c640e49a859012ae` / `1618537f9c633a7b` (FMA caveat) | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 149.32 | 153.68 | 133.870 | 130.076 | 0.969 | 0.955-0.977 | 0.97-1.03: FAIL | `fa485cbe1fa07d20` = B | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 180.31 | 176.38 | 132.078 | 135.019 | 1.022 | 1.007-1.044 | 0.97-1.03: pass | `79637ca9e991a308` = B | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 84.06 | 82.81 | 57.755 | 58.623 | 1.015 | 1.006-1.030 | 0.97-1.03: pass | `bed68cfaa32b7445` = B | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 2972.98 | 763.00 | 26.820 | 104.503 | 3.873 | 3.850-3.930 | >= 1.5: pass | `64d2a0d49855c763` = B | 1.0023 (0.0007, 0.0015) | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B Gcell/s | B ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 26.820 | 106.466 | 3.958 | 3.921-3.970 | 104.503 | 3.873 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq no; per-pair hashes base == B: yes; lanes vs per-pair bitwise mismatches base + B: 2,692,955; outside the route bound: 0 -> pass
- ragged sweep (B): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 264,419; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0.4933, f64 L1 0, f64 Sq 0.4943 -> pass
- clustering (clust/, data/dummy uncut): clustering identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 5
  - round 2 B f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.877-5.224 GHz
  - round 0 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.172-5.393 GHz
  - round 1 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.239-5.522 GHz
  - round 2 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.094-5.544 GHz
  - round 1 B f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.086-5.527 GHz
- band: 9/10 shapes in band; matrix hashes base == B (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict gcc (B): FALSIFIED**

### Verdicts and ruling

- **clang: PASS** on all 10 shapes: ragged fills 6.8–6.9× on one thread and 3.4–3.7× at 24 threads (f32 6.7×), the
  three controls 0.977–0.998×, N 8000 3.68× (A 3.50×); every matrix hash as at base.
- **GCC: FALSIFIED by one control**: the equal-length N 2000 L 100 fill at 24 threads, 0.969× (rounds 0.955–0.977;
  B 153.3–156.4 ms a fill against base's 148.9–149.8 in every round, whichever went first). Everything else in band:
  ragged 8.2–8.7× on one thread and 4.5–5.6× at 24 threads, N 8000 3.87× (A 3.96×), the other controls 1.015× and
  1.022×. Why B's equal-length L 100 fill is 3 % slower under GCC is not measured; its main loop is base's instruction
  for instruction, so the candidates are the per-call work around it (the two packing loops, `take`, the per-column end
  test), largest at L 100 [inferred].
- **cl: FALSIFIED on ragged data**: four f64 ragged shapes below 0.97× (N 1000 pinned 0.906, N 150 pinned 0.870,
  N 500 0.829, N 8000 0.966), as the Expectations predicted; the N 1000 24-thread fill 1.041×, f32 1.149×, f64 squared
  1.031×, the controls 0.990–1.005×.
- **Volkan, 2026-10-08 (question tool): "Adopt, cl one-pair"**: merge for clang and GCC, record the band miss honestly,
  and keep cl on the one-pair route for blocks of unequal length (`0b5cfe42`, below).

## After the ruling: cl keeps the one-pair route (`0b5cfe42`) [confirmed]

- **Lane calls** (`count/lane_calls.txt`; counting builds `build_count.*`: the shipped `dtw_lanes.cpp` with
  `count/count_lanes.hpp` forced in, which counts each `dtw_kernel_lanes` call, linked before that version's
  `dtwc_core`; 2 fills a run, 8 threads, f64 L1). Under cl, B2's counts equal base's on every shape: ragged N 300
  L 40–60 and N 150 L 100–1000 **0** (B, the timed candidate: 11,476 and 2,926), N 1000 L 90–110 16 (B 125,750: the
  16 are tail blocks whose lanes all have the row's length, which base takes too), equal-length N 300 L 50 11,476 (the
  counter works), band 30 16. Under clang B and B2 are equal (ragged N 300: 5,890).
- **Matrices**: cl's base and B2 fill every timed shape to the same hash (`count/cl_hashes.txt`: the four f64 ragged
  shapes, f32, squared, the three controls and N 8000), the hashes the timed run recorded.
- **Unchanged under clang and GCC**: `dtw_lanes.cpp`'s listing is GCC's byte for byte and clang's but for the
  anonymous-namespace hash and the `.zero` padding of the names it lengthens; the loops equal B's (clang 12/12, cl 8/8).
- **Gates** (`0b5cfe42`): cl ctest 94 = 91 + 3 (the lanes test, the fill test, OneBatchPAM and `cpp_conformance`
  pass); clang ctest 95 = 92 + 3, codegen 88 / 0 PASS, lanes test 223 / 3; the three docs gates pass; no new warning.

## The L 8000 regression and its fix (`f8c3d23a`, B3) [confirmed but where marked]

**Report (the orchestrator's, quiet machine):** unbanded f64 L1, equal lengths, L 8000, one thread on CPU 12, clang:
B 4.2–4.8 % slower than base, at the same instructions per cell (4.14) and LLC misses, IPC 4.80 against 5.02.

**Diagnosis timings** (mine, with the orchestrator's permission: "short pinned A/B timing loops on CPU 12", I being the
only agent, never while building; for diagnosis, not a record). `tools/ab.py`: each run one probe process, `fill f64 L1
24 8000 8000 -1 1 1 12` (N 24, one untimed fill, one timed), the builds alternating in rotated order; every run in
`diag/ab.txt`. B is `bin/clang_B2` (B's clang kernel and fill; `0b5cfe42` changed only cl). ms of the timed fill, range
(median); one hypothesis at a time:

| time | test | result | shows |
| --- | --- | --- | --- |
| 21:39 | reproduce, 5 runs | base 2767–2781 (2773), B 2901–2921 (2905): 1.048 | reproduced |
| 21:46 | (b) code placement: each row loop moved 32 bytes by a pad object before `dtw_lanes` (base's loop 175 B at 0 mod 64, 3 lines; B's 193 B at 32, 4 lines) | base at 0 / at 32: 2788 / 2786 (0.999); B at 0 / at 32: 2894 / 2886 (1.038, 1.035) | not code placement |
| — | (a) data placement: `s` and `Y` offsets mod 4096 printed (`diag/make_print.py`) | the same in both builds | — |
| 21:54 | (a) the heap moved by an environment variable of 0, 700, 1500, 2300 characters, 3 runs each | base medians 2735–2773, B 2869–2897: B +3.5–4.5 % at every pad | not data placement |
| 21:58 | the f64 L1 row loops under llvm-mca (`diag/mca/`) | 7.03 cycles an iteration, both | a static model sees no difference |
| 22:02 | B's assembly with its GPRs renamed to base's (`diag/rename_regs.py`; loop 179 B) | base 2720, B reassembled 2869 (1.055), renamed 2840 (1.044) | not the address registers, not the loop's size |
| 22:06 | swap: B's fill with base's kernel object; base's fill with B's | 1.0007; 1.052 | the kernel |
| 22:12, 22:15 | (c) B's kernel without `take` and the per-column test (`notake`, equal lengths only); without the padding loop (`nopad`) | 7 runs: base 2763–2837 (2774), B 2896–2920 (2902): 1.046, notake 2683–3332 (2732): 0.985; 5 runs: nopad 1.027 (a noisy batch: base 2734–2913) | the per-column capture test |

**Cause.** B's per-column end test (`if (j + 1 == end) end = take(j + 1);`, which copied each lane's distance out after
its own last column) changed clang's vector register allocation in the row loop. Base's and B's f64 L1 row loops are
the same instructions in the same order (38–39), but base keeps one `Y` group as a stack operand (`vsubpd ymm2, ymm0,
[rsp+0x90]`) and the mask in ymm3, while B holds all four `Y` groups in registers under other assignments
(`diag/mca/clang_{base,B2}_f64L1.s`). Without the test, the loop is base's register for register
(`diag/B_notake.f64L1.txt`) and runs at base's time. Why B's allocation runs at lower IPC on this core from L 4000 up
is not measured [unknown]; placement of code and data, the GPRs and the loop's size are excluded above. Where it starts
(5 runs each, 22:21–22:23): L 1000 (N 160) 1.003, L 2000 (N 80) 0.998, L 4000 (N 40) 1.049 (base 1557–1567, B
1632–1644).

**Fix (`f8c3d23a`).** No test in the column loop: each column's last row goes to its `Y` row, and the lanes read their
distances from `Y` after the loop (The change). Its f64 L1 row loop is base's but for GPR names and the stack offset
(`diag/fix.f64L1.txt`); f32 L1 and both squared loops of the linked probe equal base's after normalisation (B's f64
squared differed and ran at base's time, 22:51 below). Timings, 5 runs each:

| time | shape (one thread, CPU 12) | base | B | fix |
| --- | --- | --- | --- | --- |
| 22:19 | f64 L1 N 24 L 8000 | 2708–2762 (2735) | 2842–2934 (2876): 1.052 | 2736–2765 (2746): 1.004 |
| 22:21 | f64 L1 N 1000 L 90–110 (ragged) | 3076–3097 (3083) | 454–460 (455) | 451–468 (455): base / fix 6.78× |
| 22:21–22:23 | f64 L1 L 1000, 2000, 4000 | — | 1.003, 0.998, 1.049 | 1.001, 0.998, 1.0045 (L 4000: 1561–1572) |
| 22:51 | f64 squared N 24 L 8000 | 2860–3100 (2921) | 2884–3106 (2924) | B3 2857–3071 (2921) |
| 22:52 | f64 L1 N 24 L 8000, the committed build (`bin/clang_B3`) | 2722–2747 (2728) | — | B3 2722–2756 (2731): 1.001 |
| 23:03 | f64 L1 N 8 L 8000 (the kit's new shape) | 633–639 (635) | 664–670 (669): 1.052 | B3 634–638 (635): 1.000 |

The 22:51 batch's first round (3071–3106 ms, clocks 4.6–4.9 GHz) overlapped the end of my GCC sweep in WSL (8 threads,
done 22:51:27); without it base 2860–2922, B 2884–2951, B3 2857–2922. `fix` (`diag/bin/fix`) is the kernel committed as
B3, built in the diagnosis tree.

**GCC** (WSL vCPU 2, `diag/ab_gcc.sh`, 22:25–22:27, 5 runs): GCC's row loops in B are base's, and no gap shows: L 8000
base 2789–2850 (2826), B 2790–2857 (2805), fix 2796–2828 (2817); N 2000 L 100 on one thread base 1884–1941 (1900),
B 1894–1931 (1903), fix 1906–1921 (1910). So GCC's 24-thread control (0.969×, Verdicts) is not this cause; it stays
unexplained (24 threads were not timed here).

**Gates at `f8c3d23a`:** clang tree: no new warning; ctest 95 = 92 + 3; `test_codegen_no_calls` inner_loops 86, calls 0,
PASS (88 → 86: two f32 copy loops of 42 instructions are gone, the per-column capture's [inferred: they go with it];
every DP loop remains, the f64 packed row step back to base's 39 instructions with one stack operand:
`gate/B3_loops.txt`); lanes test 223 / 3; OneBatchPAM 10,638 / 12; `test_fill_request` 57 / 6. cl tree
(`build-msvc`): no new warning; ctest 94 = 91 + 3; its lanes loops as before (f64 134 instructions and 82 memory
operands, f32 257 and 171). Sweeps (`check/*_B3*`): clang and cl equal-length hashes as at base and ragged hashes as
B's, 0 mismatches; GCC both sweeps as B's (L1 as at base; f64 squared the FMA caveat, 0 outside the route bound). CLI
(clang tree, `cli/compare.txt`): 156 files byte-identical to base's. GCC clustering (`clust/`, the WSL cand tree at
B3): identical to base (`compare.txt`; B's in `compare_B.txt`), every output file byte-identical to B's.

## The timed run of B3 (the orchestrator's, quiet machine, 2026-10-08 23:14:11–23:29:09) [confirmed]

`uv run --no-project python driver.py run` in the kit (`driver.CAND = "B3"`), started by `wait_quiet.ps1` (QUIET at
23:14:10: total 4.5 %, Tanium 0.00 % of all cores; no agent running): 3 rounds of the 11 shapes, each round running
clang, cl and GCC, base and B3 alternating which goes first (base, B3 and A rotating on N 8000); every probe exit 0;
898 s. Its report is `report.md` (B's run: `report_B.md`); its probe lines are appended to `raw.txt` (the report reads
the last run). The probes are the kit's `bin/<compiler>_B3`: `src/B3` is `f8c3d23a` (its `dtw_kernel.hpp` hashes as the
commit's once line endings are normalised), `lib/clang_B3` and `lib/msvc_B3` are byte for byte the RL trees' libraries,
built after the trees' last source change, and GCC's `~/rl/repo-cand` kernel hashes as the commit's (checked by the
orchestrator before the run). `driver.py run --cl-retime` was not run: this run timed cl's B3 on all 11 shapes, a
superset of `CL_FILLS`, under the cl clause. Load (`load.txt`):

| when | total CPU | busiest processes | WSL |
| --- | --- | --- | --- |
| 23:14:15 (before) | 2.1–12.8 % | `system` 1.21 %, `serviceshell` 0.71 %, `vmmemwsl` 0.50 %, `taskmgr` 0.43 %, `wmiprvse` 0.30 %; COMSOL none | up 0 min, load 0.07, no systemd job, 99.9 % idle |
| 23:28:49 (after) | 1.0–10.0 % | `system` 1.08 %, `taskmgr` 0.65 %, `serviceshell` 0.56 %, `sspservice` 0.45 %, `wmiprvse` 0.28 %; COMSOL none | up 1 min, load 7.94 / 2.89 / 1.03 (the last GCC probes), no systemd job |

### clang (`report.md` verbatim)

| fill | W | base ms | B3 ms | base Gcell/s | B3 Gcell/s | ratio (median) | range | band | matrix hash base / B3 | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 3103.37 | 458.87 | 1.601 | 10.827 | 6.785 | 6.711-6.833 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | 5.266 / 5.251 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 146.77 | 40.20 | 33.852 | 123.597 | 3.630 | 3.458-3.696 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 2673.88 | 388.91 | 1.414 | 9.725 | 6.875 | 6.798-6.966 | >= 1.5: pass | `d0c4ea00ede2475c` = B3 | 1.1780 (0.1120, 0.0391) | 5.287 / 5.277 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1097.92 | 336.38 | 37.793 | 123.355 | 3.264 | 3.258-3.354 | >= 1.5: pass | `cd125a5433bb9769` = B3 | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 159.59 | 21.98 | 31.133 | 226.004 | 7.095 | 6.794-7.261 | >= 1.5: pass | `d0429e2d299c55ed` = B3 | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 139.03 | 41.65 | 35.736 | 119.297 | 3.490 | 3.255-3.518 | >= 1.5: pass | `76e1c6ac3e6ce69d` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 172.95 | 172.97 | 115.585 | 115.571 | 1.000 | 0.967-1.051 | 0.97-1.03: pass | `fa485cbe1fa07d20` = B3 | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 182.05 | 184.14 | 130.812 | 129.332 | 0.982 | 0.969-0.994 | 0.97-1.03: pass | `79637ca9e991a308` = B3 | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 84.65 | 85.26 | 57.348 | 56.940 | 0.996 | 0.992-1.029 | 0.97-1.03: pass | `bed68cfaa32b7445` = B3 | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 2673.44 | 736.54 | 29.825 | 108.258 | 3.630 | 3.628-3.769 | >= 1.5: pass | `64d2a0d49855c763` = B3 | 1.0023 (0.0007, 0.0015) | n/a |
| f64 L1 N 8 L 8000 unbanded, 1 thread pinned | 16 | 649.71 | 653.47 | 2.758 | 2.742 | 0.994 | 0.992-0.994 | 0.97-1.03: pass | `88dd1bba61d35f0b` = B3 | - | 5.267 / 5.261 |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B3 Gcell/s | B3 ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 29.825 | 105.465 | 3.544 | 3.440-3.884 | 108.258 | 3.630 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B3: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B3: yes; lanes vs per-pair bitwise mismatches base + B3: 0; outside the route bound: 0 -> pass
- ragged sweep (B3): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- CLI (w8b_cli_runs.sh + w8b_cli_extra.sh, data/dummy): byte-identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 5
  - round 0 base f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.749-5.27 GHz
  - round 0 B3 f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.755-5.263 GHz
  - round 1 B3 f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.805-5.327 GHz
  - round 2 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 4.978-5.279 GHz
  - round 0 base f64 L1 N 8 L 8000 unbanded, 1 thread pinned: clock 4.803-5.342 GHz
- band: 11/11 shapes in band; matrix hashes base == B3 (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict clang (B3): PASS**

### cl (`report.md` verbatim)

| fill | W | base ms | B3 ms | base Gcell/s | B3 Gcell/s | ratio (median) | range | band | matrix hash base / B3 | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 8 | 3385.00 | 3372.65 | 1.468 | 1.473 | 1.000 | 0.999-1.004 | >= 0.97: pass | `b28673ce59c47ce8` = B3 | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 8 | 176.16 | 171.66 | 28.203 | 28.943 | 1.026 | 0.987-1.042 | >= 0.97: pass | `b28673ce59c47ce8` = B3 | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 8 | 2696.97 | 2686.70 | 1.402 | 1.408 | 0.997 | 0.984-1.006 | >= 0.97: pass | `d0c4ea00ede2475c` = B3 | 1.0809 (0.0608, 0.0140) | n/a |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 8 | 1288.72 | 1360.59 | 32.198 | 30.497 | 1.008 | 0.947-1.009 | >= 0.97: pass | `cd125a5433bb9769` = B3 | 1.0246 (0.0207, 0.0033) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 185.79 | 187.49 | 26.742 | 26.500 | 0.940 | 0.939-1.078 | >= 0.97: FAIL | `d0429e2d299c55ed` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 8 | 181.74 | 185.15 | 27.337 | 26.834 | 1.004 | 0.982-1.005 | >= 0.97: pass | `76e1c6ac3e6ce69d` = B3 | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 8 | 726.26 | 707.22 | 27.525 | 28.266 | 0.983 | 0.924-1.027 | >= 0.97: pass | `fa485cbe1fa07d20` = B3 | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 8 | 820.59 | 811.55 | 29.022 | 29.345 | 1.009 | 0.987-1.011 | >= 0.97: pass | `79637ca9e991a308` = B3 | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 8 | 100.26 | 102.43 | 48.420 | 47.398 | 0.979 | 0.979-0.997 | >= 0.97: pass | `bed68cfaa32b7445` = B3 | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 8 | 2819.16 | 2833.11 | 28.284 | 28.144 | 1.015 | 0.995-1.030 | >= 0.97: pass | `64d2a0d49855c763` = B3 | 1.0011 (0.0003, 0.0007) | n/a |
| f64 L1 N 8 L 8000 unbanded, 1 thread pinned | 8 | 2755.88 | 2760.28 | 0.650 | 0.649 | 0.995 | 0.970-1.005 | >= 0.97: pass | `88dd1bba61d35f0b` = B3 | - | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B3 Gcell/s | B3 ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 28.284 | 26.297 | 1.023 | 0.930-1.074 | 28.144 | 1.015 | = base | 1.0009 (0.0002, 0.0007) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B3: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B3: yes; lanes vs per-pair bitwise mismatches base + B3: 0; outside the route bound: 0 -> pass
- ragged sweep (B3): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- pinned probes whose clock moved by more than 3% over their fills: 0
- band: 10/11 shapes in band; matrix hashes base == B3 (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict msvc (B3): FALSIFIED**

### GCC (`report.md` verbatim)

| fill | W | base ms | B3 ms | base Gcell/s | B3 Gcell/s | ratio (median) | range | band | matrix hash base / B3 | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 4063.40 | 502.27 | 1.223 | 9.892 | 8.129 | 8.058-8.232 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | 5.209 / 5.207 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 214.29 | 42.59 | 23.185 | 116.668 | 4.868 | 4.659-5.504 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 3446.94 | 396.38 | 1.097 | 9.541 | 8.710 | 8.527-8.785 | >= 1.5: pass | `d0c4ea00ede2475c` = B3 | 1.1780 (0.1120, 0.0391) | 5.187 / 5.204 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1873.72 | 344.06 | 22.145 | 120.601 | 5.479 | 5.425-5.534 | >= 1.5: pass | `cd125a5433bb9769` = B3 | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 122.64 | 22.08 | 40.512 | 225.005 | 5.554 | 4.499-6.039 | >= 1.5: pass | `d0429e2d299c55ed` = B3 | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 209.00 | 45.26 | 23.772 | 109.768 | 4.601 | 4.573-4.884 | >= 1.5: pass | `c640e49a859012ae` / `1618537f9c633a7b` (FMA caveat) | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 160.79 | 154.34 | 124.326 | 129.519 | 1.011 | 0.944-1.049 | 0.97-1.03: pass | `fa485cbe1fa07d20` = B3 | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 217.02 | 198.76 | 109.736 | 119.818 | 0.973 | 0.820-1.092 | 0.97-1.03: pass | `79637ca9e991a308` = B3 | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 92.67 | 96.97 | 52.388 | 50.065 | 0.973 | 0.956-1.012 | 0.97-1.03: pass | `bed68cfaa32b7445` = B3 | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 3187.04 | 803.26 | 25.019 | 99.266 | 3.890 | 3.726-4.091 | >= 1.5: pass | `64d2a0d49855c763` = B3 | 1.0023 (0.0007, 0.0015) | n/a |
| f64 L1 N 8 L 8000 unbanded, 1 thread pinned | 16 | 689.13 | 687.35 | 2.600 | 2.607 | 1.008 | 0.998-1.009 | 0.97-1.03: pass | `88dd1bba61d35f0b` = B3 | - | 5.249 / 5.239 |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B3 Gcell/s | B3 ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 25.019 | 100.282 | 4.012 | 4.008-4.031 | 99.266 | 3.890 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B3: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq no; per-pair hashes base == B3: yes; lanes vs per-pair bitwise mismatches base + B3: 2,692,955; outside the route bound: 0 -> pass
- ragged sweep (B3): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 264,419; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0.4933, f64 L1 0, f64 Sq 0.4943 -> pass
- clustering (clust/, data/dummy uncut): clustering identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 3
  - round 0 B3 f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.864-5.24 GHz
  - round 0 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.008-5.191 GHz
  - round 2 B3 f64 L1 N 8 L 8000 unbanded, 1 thread pinned: clock 4.801-5.256 GHz
- band: 11/11 shapes in band; matrix hashes base == B3 (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict gcc (B3): PASS**

### cl's second attempt on its one shape outside the band (the orchestrator's, 23:34:13–23:34:47) [confirmed]

`orch/ab_cl_f32.sh`, registered in its header before the run (verbatim): "20 pairs, each two probe processes (base and
B3, the first alternating), each probe one untimed fill and 3 timed; ratio = base ms / B3 ms per pair. Median ratio >=
0.97: noise, the shape passes on this attempt; < 0.97: a real cl slowdown, FALSIFIED stands and the cause is looked
for." Started by `wait_quiet.ps1` (QUIET at 23:34:13: total 4.8 %, Tanium 0.13 %). cl, f32 L1 N 1000 L 90–110 unbanded,
24 threads (`orch/ab_cl_f32.txt`):

| pairs | median ratio | range | base ms (median of 20) | B3 ms (median of 20) | median ratio, base first / B3 first |
| --- | --- | --- | --- | --- | --- |
| 20 | 1.059 | 0.864–1.335 | 181.4 | 175.3 | 1.099 (10) / 1.007 (10) |

One process's fill took 164–218 ms under base and 155–198 ms under B3: the 24-thread spread of this box, the first
process of a pair the slower. In the 3-round run the shape's rounds were 1.078 (base first), 0.940 (B3 first) and 0.939
(base first; every cl 24-thread shape ran slower in that round). Under cl, B3's fill gives this shape base's blocks
(`ragged` is false: no length sort, the blocks of consecutive columns; the lane calls equal base's,
`count/lane_calls.txt`), so no mechanism predicts a difference. **The shape passes on the second attempt.**

### Verdicts (B3)

- **clang: PASS** 11/11. Ragged fills 6.79× and 6.88× pinned on one thread, 3.63× and 3.26× at 24 threads, f32 7.10×,
  f64 squared 3.49×; N 8000 L 40–60 3.63× (A 3.54×); the controls 1.000×, 0.982×, 0.996×; the L 8000 one-thread
  fill 0.994× (rounds 0.992–0.994). Every matrix hash as at base.
- **GCC 13.3: PASS** 11/11. 8.13× and 8.71× on one thread, 4.87× and 5.48× at 24 threads, f32 5.55×, squared 4.60×;
  N 8000 3.89× (A 4.01×); the controls 1.011× (B's 0.969× not seen again), 0.973× and 0.973× (rounds 0.820–1.092 and
  0.956–1.012: WSL's 24-thread spread); L 8000 1.008×. L1 hashes as at base, squared within the route bound;
  clustering identical.
- **cl: at base's speed.** 10/11 in the 3-round run (f32 ragged at 24 threads 0.940×), that shape 1.059× over the
  registered second attempt; every matrix hash as at base.
- **B3 against B** (the two runs, 3 hours apart): on one thread B3's ragged fills ran B's times (clang 458.9 against
  456.6 ms and 388.9 against 387.4; base 0.8–0.9 % slower in the second run). At 24 threads the second run was slower
  for base and candidate alike (clang's equal-length control 160.0 → 173.0 ms in both builds; A on N 8000 113.1 →
  105.5 Gcell/s), so 24-thread ratios move by several per cent between runs: clang's N 500 L 100–1000 went from 3.39×
  (B) to 3.26× (B3), its f32 from 6.73× to 7.10×.

## Hardware counters, B against base (Windows ETW; the orchestrator's, 2026-10-08 21:25–21:30) [confirmed]

VTune 2026.4 cannot sample on this machine even elevated with its drivers running (Hyper-V with VBS: `vtune` reports a
virtual machine); Windows' own counters can. `C:/D/git/wt/tmp/vtune/profile_rl_etw.cmd`, run by Volkan from an
Administrator terminal: per case `xperf -on PROC_THREAD+LOADER+CSWITCH -Pmc InstructionRetired,UnhaltedCoreCycles,
LLCMisses,LLCReference,CacheMisses,BranchMispredictions CSWITCH` around one probe run (two fills: one untimed, one timed;
single-thread shapes pinned to CPU 12), then `etw_pmc.py` sums the counters read at every context switch for the probe
process's threads, and `etw_table.py` divides by the run's cells. The probes are copies of the kit's (`vtune/bin`,
`SHA256SUMS.txt`). No L1 or L2 event is among Windows' built-in sources. One case (clang B, s2) is missing: the batch file
had Unix line endings and `cmd.exe` jumped over it. Results `C:/D/git/wt/tmp/vtune/rl_etw/`:

| shape | build | timed fill ms | instructions / cell | cycles / cell | IPC | LLC references / Mcell | LLC misses / Mcell | LLC miss / ref | branch mispredictions / Mcell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 f64 L1 N 1000 L 90-110, 1 thread | clang_B | 460.1 | 2.628 | 0.516 | 5.09 | 324 | 122 | 0.377 | 708 |
| s1 f64 L1 N 1000 L 90-110, 1 thread | clang_base | 3078.9 | 8.100 | 3.327 | 2.43 | 310 | 117 | 0.378 | 3,680 |
| s1 f64 L1 N 1000 L 90-110, 1 thread | msvc_B | 3823.0 | 16.917 | 3.997 | 4.23 | 701 | 276 | 0.394 | 1,517 |
| s1 f64 L1 N 1000 L 90-110, 1 thread | msvc_base | 3413.7 | 12.260 | 3.610 | 3.40 | 580 | 207 | 0.357 | 437 |
| s2 f64 L1 N 150 L 100-1000, 1 thread | clang_base | 2664.8 | 8.055 | 3.766 | 2.14 | 307 | 80 | 0.262 | 1,183 |
| s2 f64 L1 N 150 L 100-1000, 1 thread | msvc_B | 2882.5 | 18.123 | 3.921 | 4.62 | 350 | 99 | 0.283 | 331 |
| s2 f64 L1 N 150 L 100-1000, 1 thread | msvc_base | 2658.7 | 12.155 | 3.725 | 3.26 | 219 | 63 | 0.289 | 1,013 |
| s5 f64 L1 N 24 L 8000, 1 thread | clang_B | 2899.0 | 4.144 | 0.863 | 4.80 | 96 | 37 | 0.382 | 34 |
| s5 f64 L1 N 24 L 8000, 1 thread | clang_base | 2786.8 | 4.144 | 0.825 | 5.02 | 104 | 37 | 0.355 | 32 |
| s3 f64 L1 N 8000 L 40-60, 24 threads | clang_A | 741.0 | 2.776 | 0.812 | 3.42 | 3,580 | 592 | 0.165 | 278 |
| s3 f64 L1 N 8000 L 40-60, 24 threads | clang_B | 733.7 | 2.792 | 0.800 | 3.49 | 2,872 | 470 | 0.164 | 494 |
| s3 f64 L1 N 8000 L 40-60, 24 threads | clang_base | 2683.4 | 8.193 | 3.305 | 2.48 | 34,011 | 631 | 0.019 | 973 |

- **Ragged lanes are faster by doing less, not by missing less**: clang s1 runs 3.1× fewer instructions per cell at
  2.1× the IPC; LLC misses stay about 120 per million cells.
- **cl's ragged loss** (B, before `0b5cfe42`): its scalar lanes run 38 % more instructions per cell than its one-pair
  kernel (16.9 against 12.3), not more misses.
- **L 8000 (s5)**: B the same 4.144 instructions per cell as base at IPC 4.80 against 5.02, LLC traffic negligible: the
  register-allocation regression fixed in B3, not memory.
- **The 256 MB matrix (s3, 24 threads)**: B 21 % fewer LLC misses than A (470 against 592 per million cells, about 1.2
  per matrix entry against 1.5), the row-local order's locality, but only 1.5 % fewer cycles per cell; base has 9.5×
  A's LLC references and 11.8× B's (34,011 per million cells).
- B3's counters (`profile_rl_etw.cmd` now profiles base against B3, s1, s2, s5, cl's s6 = N 8 L 8000 and s3, into
  `rl_etw2/`) await an elevated run on a quiet machine.

## Assembly [confirmed] (`asm/`, `asm/summary.txt`; one loop = one DP row step of W lanes; `tools/loops.py`, `tools/loopdiff.py` with registers and labels normalised)

clang: `dtw_lanes.cpp` `-S -masm=intel` with the tree's own flags (`-O3 -march=native`, the FP subset, `-fno-lto`).
GCC: the same file with `dtwc_core`'s WSL flags (`-O3 -march=x86-64-v3`, the FP subset, `-fno-signaling-nans`), plus
objdump of the LTO-linked probes. cl: `/FAs` with the tree's flags minus `/GL`. B changes `Problem.cpp` only: its
`dtw_lanes.cpp` listings equal A's under all three compilers but for the anonymous-namespace hash in a few symbol
names (clang, cl) and byte for byte (GCC); the table holds for A and B alike.

| compiler, loop | base `e691e48c` | candidate |
| --- | --- | --- |
| clang f64 L1 (W 16) | ymm `vminpd` ×8, 38 instructions; 1 folded reload of a `y` vector (invariant) | 38, the same packed form; no stack operand |
| clang f64 squared | 38; 1 reload of a `y` vector (invariant) | 38; the same reload |
| clang f32 L1 (W 32) | ymm `vminps` ×8, 37; 2 folded reloads of `y` vectors | 37; the same 2 |
| clang f32 squared | 37; 1 folded reload of a `y` vector | 37; the same |
| GCC f64 L1 / f32 L1 | 37 / 37 ymm, no stack operand | the same instruction sequences |
| GCC f64 squared / f32 squared | 33 and 37 (two copies) / 33, no stack operand | 33 and 33 / 33: the same, 4 register moves fewer in one copy |
| cl f64 (W 8), L1 and squared | scalar `vminsd`, 134 instructions, 82 memory operands, 57 on the frame | the same sequence |
| cl f32 (W 16), L1 and squared | scalar `vminss`, 257, 171, 122 | the same sequence |

No lanes loop calls anything under any compiler; no spill or reload is added on `left`, `diag` or `up`. The loop forms
tried (`variants/`): A-segments (GCC spills), B (end test at the top of the column; GCC f32 squared +4 moves), C
(reference capture; cl reloads `s`), F (shipped until `f8c3d23a`). The candidate column above ("no stack operand")
was F's other register allocation, which ran 4–5 % slower on long series (the L 8000 section); B3's clang f64 L1 loop
is base's again, stack operand included.

**`test_codegen_no_calls` coverage** (`gate/`, `tools/gate_loops.py`: the probe compiled exactly as the gate compiles it
— the donor `dtw_dispatch.cpp`'s command, LTO off, `-S` — against each version's probe and headers; the gate's own
count reproduced: base `inner_loops=142`, A and B `88`, `calls=0` in all three). Every innermost loop of the lanes
probes, by role:

| probe function | base | candidate (A = B) |
| --- | --- | --- |
| `dtwc_probe_lanes_f64` (L1) | packed DP row step (39 instructions, 8 ymm `vminpd`, 1 stack operand); scalar DP loop (116, 16 `vminsd`: column 0); 66 copy loops | packed DP row step (38, 8 ymm `vminpd`, no stack operand); the scalar DP loop (116); 8 copy loops |
| `dtwc_probe_lanes_f32` (squared) | packed DP row step (37, 8 ymm `vminps`); 2 scalar DP loops (32 and 39); 5 copy loops | the same three DP loops; 9 copy loops |

The 54 innermost loops gone are all copy loops of the f64 probe (66 → 8; f32 5 → 9): at base clang unrolled the packing
loop `for w: for t < n: Y[t].v[w] = yw[t]` across the 16 f64 lanes (one shared trip count), each lane's copy in four
versions (6, 12, 20 and 36 instructions, 16 of each) [inferred from the 16-fold multiplicity]; with per-lane lengths the
packing is a w-loop over a copy and a padding loop with lane-dependent trip counts, which it keeps rolled. No DP loop is
gone: the gate still sees every DP loop of `dtw_kernel_lanes` in both precisions, none calls, and no probe change was
needed.

## Bitwise [confirmed] (`check/`)

The shipped block function (`resolve_dtw_block_fn`) against the shipped per-pair function (`resolve_dtw_fn`), each build
linking its own `dtwc_core`; f64/f32 × L1/squared; 4 data kinds (uniform, random walk, integers with ties, ±max/2..max
overflowing to inf), 3 seeds, 69 partners per x. B's sweeps (`check/*_B*.txt`) equal A's line for line (`*_A*.txt`).

| sweep | clang | cl | GCC |
| --- | --- | --- | --- |
| equal length (XL's: lengths 1–1000, bands −1, 0, 1, L/10, L; 16,560,000 lane outputs per build) | every hash base = B (combined `cb658d8ea5205872`, XL's); 0 differ from per-pair | every hash base = B (`0fdf8bb2e6b24725`, XL's); 0 differ | L1 and f32 squared hashes base = B; f64 squared `a38d65676974ef64` → `f97babf59f98a072` (below); per-pair hashes equal; 0 outside the route bound |
| ragged (lengths in [1, n] per lane, n 1–1000, unbanded; 828,000 lane outputs per type and cost: y longer 410,275, shorter 411,583) | 0 differ from per-pair | 0 differ | L1 0 differ; squared 131,054 (f64) / 133,365 (f32) differ in the last bits, 0 outside the route bound (worst 0.49 of (nx+ny−1)·eps·max; bound 2) |

GCC's f64 squared equal-length change: 67 of 20,000 (kind, n, band) cases of seed 0 differ base against candidate
(`diag/gcc_f64sq_differ.txt`), banded and unbanded. Cause: GCC computes each column's row 0 (`s[0] + d * d`) lane by lane;
in base it fused lane 0's (`vfmadd132sd`) and not lanes 1–15 (`vmulsd` + `vaddsd`), in the candidate it fuses none
(`asm/gcc_*_dtw_lanes.s`). The multiply-add caveat of Volkan's 10-01 ruling; the candidate's lanes differ from the
per-pair route in 77 fewer outputs. f32 squared, both L1 and the per-pair route are unchanged.

- **The fills**: on a 300-series ragged fill (L 40–60) base, A and B give the same matrix hash under each compiler
  (`b50103b29cef2b4f` under all three), and on N 8000 L 40–60 under clang (`64d2a0d49855c763`); the kit's report
  checks every timed shape.
- **CLI** (`C:/D/git/wt/tmp/int/w8b_cli_runs.sh` + `w8b_cli_extra.sh`, `data/dummy`'s 25 series of 5,148–9,405
  samples, every unbanded block ragged: pam, clara, onebatch, tadpole, hierarchical, kmedoids, lrcore, bands, WDTW,
  DDTW, ADTW, squared, zero-cost, float32, k 1, verbose): the clang tree's `dtwc_cl.exe` at base (`cli_base/`) and at B
  (`cli_B/`): 156 files byte-identical (`cli/compare.txt`); exit codes 24 × 0 and 1 × 1 in both (band 10 refused). A
  gave the same (`cli_A/`, `cli/compare_A.txt`).
- **Clustering under GCC** (`clust/`, XG's scripts with `data/dummy` uncut): 36 runs per build (float64/float32 ×
  l1/squared × k 3, 5 × pam, kmedoids, unbanded and band 4300; onebatch × 4), base against B: every exit 0, labels and
  medoids identical 36/36; L1 matrices identical; squared matrices within 1.9e-7 relative, total cost 1.0e-8,
  silhouettes 1.2e-7 (`clust/compare.txt`: "clustering identical: yes"; A the same, `clust/compare_A.txt`).
- **cpp_conformance** passes in all three trees.

## Gates in the worktree [confirmed] (B, `356bef7e`; `0b5cfe42`'s and `f8c3d23a`'s in their sections)

| tree | result |
| --- | --- |
| `build/` clang++ 21.1.8 (as the main `build/`) | no new warning (base: 3 CRT `getenv` deprecations of our code); ctest 95 = 92 + 3 MAY_SKIP (CUDA, Metal × 2); lanes test 223 assertions / 3 cases (base 183); `unit_test_one_batch_pam` 10,638 / 12; `test_fill_request` 57 / 6; codegen 88 / 0 PASS; `check_docs` 434 flags, 58 pages, 0 undocumented, PASS; `check_pins` 0 failures; `generate_docs --check` current |
| `build-msvc/` cl 19.50 (as MV's, under `vcvars64`) | no new warning (base: 2 STL4047 in llfio); ctest 94 = 91 + 3 |
| WSL `~/rl/repo-cand/build-cand` GCC 13.3, v3 (B) | 0 warning lines; ctest 94 = 91 + 3 (no codegen test under GCC) |

## Kit (`C:/D/git/wt/tmp/RL/`)

- `rprobe.cpp`: XL's probe with XG's and MV's route bound, one source for the three compilers. `check` (XL's sweep),
  `ragged` (the candidates only), `waste` (counts: index, A, B), `fill T D N lo hi band threads reps [cpu]`: N random
  walks (XL's generator: the N 1000 L 90-110 data are XL's ragged fill's) through the linked version's
  `Problem::fill_distance_matrix` itself; one untimed fill, then 3; Gcell/s on each pair's own cells; the matrix hash
  (FNV of the packed lower triangle); the calling thread's clock (register-add chain; none under cl); B's and A's lanes'
  cells per useful cell for the ragged shapes.
- Builds per compiler: base, A, B and B3 (and B2 for clang and cl). `build_clang.sh` / `build_msvc.bat
  base|A|B|B2|B3` (dtwc_cl.cpp's flags, linked as `dtwc_cl.exe` is, against `lib/<compiler>_<v>/dtwc_core.lib`, saved
  from the RL trees at each version; HiGHS and Gurobi from the trees); `wsl/build_gcc.sh base|A|B|B3` (against
  `~/rl/lib/gcc_<v>/libdtwc_core.a`, saved from `~/rl/repo-base` and `~/rl/repo-cand` as it held A, then B, then B3;
  `wsl/trees.sh` builds them, `src/rl.patch` = `e691e48c..f8c3d23a`). `src/base`, `src/A`, `src/B`, `src/B2`,
  `src/B3`: the headers each probe compiled against (`COMMIT.txt`). `asm_clang.sh`, `asm_msvc.bat`:
  the listings; `tools/gate_loops.py`: the gate's loops.
- After the ruling: `driver.py run --cl-retime` (and `smoke --cl-retime`): cl only, `msvc_base` against `msvc_B3`
  (`f8c3d23a`, `src/B3`, `lib/msvc_B3`; it was `msvc_B2` before the fix) on cl's four f64 L1 ragged shapes, N 8000,
  the three controls and the long equal-length shape, into `raw_cl.txt` and `report_cl.md`, with Windows-only load
  records. `build_count.bat` / `build_count.sh <v>` and `count/count_lanes.hpp`: the lane-call counting builds (kit
  only).
- `driver.py run [--rounds 3] [--compilers clang,msvc,gcc]` (also `smoke`, `check`, `report`): 3 rounds, each running
  every compiler; the ten registered shapes and, since the fix, an eleventh (f64 L1 N 8 L 8000 unbanded, one thread
  pinned, 0.97–1.03×: the long equal-length shape) with base and the candidate (`driver.CAND`, B3 since the fix; B in
  the orchestrator's timed run) alternating which goes first, A beside them (the three rotating) on N 8000 L 40–60;
  single thread pinned (logical CPU 12 on Windows; `taskset -c 2` in WSL, the probe checking its affinity), the rest at
  the OpenMP default (24); load records before and after (`load.sh`: TB's Windows record, XG's WSL side); WSL booted
  first and waited on until systemd runs no job; every probe line in `raw.txt`; `report.md` per compiler: medians, the
  candidate's ratios with ranges, the band per shape, matrix hashes base / candidate (L1 equal under every compiler,
  squared under clang; elsewhere a squared difference is shown as the FMA caveat, the sweeps holding it to the route
  bound), the cells per useful cell of B's grouping (B3's too), pinned clocks (flagged if they move over 3 %), A's
  Gcell/s and ratio on N 8000, the sweeps, the CLI and clustering results, and the verdict per compiler for the
  candidate. `diag/`: the L 8000 diagnosis (`tools/ab.py`, the timing log `diag/ab.txt`, the pad, swap, renamed and
  variant builds).
- Duration [estimated from the cell counts (4.97e9, 3.78e9, 4.15e10, 2.0e10, 2.4e10, 4.9e9 and 8.0e10 cells) and XL's,
  TP's, XG's per-cell rates]: clang about 65 s a round, cl about 115 s (its lanes and pairs both scalar), GCC about
  75 s; 3 rounds plus the load records, about 14 minutes. The smoke (one round of tiny shapes, all three compilers,
  with A on N 400 L 40–60) took 52 s (`raw_smoke.txt`, `report_smoke.md`; not a measurement). One N 8000 fill per
  clang build (a functional check under load, not a measurement) ran its 256 MB matrix in 0.7–2.8 s a fill, the same
  hash (`64d2a0d49855c763`) from base, A and B. The long equal-length shape adds about 1.5 minutes (cl's scalar lanes
  2.75 s a fill, clang's and GCC's 0.64–0.68 s; the same matrix hash `88dd1bba61d35f0b` from base and B3 under all
  three, a functional check). The smoke with B3 (one tiny round, N 4 L 1000 for that shape) took 52 s.

## Open

- The CHANGELOG line carries B's figures (`0b5cfe42`); B3's replace them after the merge.
- GCC's equal-length N 2000 L 100 control at 0.969× in B's run (3 % in every round) was 1.011× in B3's (rounds
  0.944–1.049); its cause was never measured. It is not clang's L 8000 cause: GCC's row loops in B are base's, and on one
  thread B and B3 ran at base's time.
- B3's hardware counters (`rl_etw2/`, above).
- Why B's register allocation ran at lower IPC from L 4000 up on this core (the L 8000 section) is not measured.
- OneBatchPAM's ragged batch table is untimed; its columns could be taken in length order as the fill's are (more than a
  line).
- Apple clang's `memset_pattern16` idiom (`a332d671`): the new padding loop stores one value at a 128-byte stride, which
  is not that idiom, but `test_codegen_no_calls` has not run on the Mac.
- A banded fill could take its columns in length order too (equal-length blocks for the banded lanes); its own band.
