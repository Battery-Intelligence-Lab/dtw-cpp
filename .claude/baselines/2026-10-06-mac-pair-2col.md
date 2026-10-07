# 2026-10-06 — Per-pair DTW kernels: two columns per pass (Mac)

Unit pair-2col (PLAN "After G": ☐ per-pair kernels two columns per pass), from design-2.0 e26d5680 on `pb/pair-2col`.
Apple M5 Pro (18 cores), macOS (Darwin 25.6.0), Apple clang 21.0.0 (clang-2100.3.34.2), `clang-macos` preset; base, k1
and head built in the same worktree `build/`, base first. Evidence behind the design: `2026-10-06-mac-kernel-assembly.md`
(variant `v_skew2`, recommendation 4; the wheel's `-Os` row minimum, recommendation 1). **Phase 1 has no timing**: other
agents were building (load average 33–52 on 18 cores during the checks); the speed is left to the kit below, run on a
quiet machine in phase 2.

**Question.** If the per-pair kernels (`dtw_kernel_linear`, kernel 1, unbanded; `dtw_kernel_banded`, kernel 2) compute
columns j and j + 1 per pass of the inner loop — two min-then-add chains where one column is one 5.1–5.3-cycle chain —
does the linked binary run the audit's `v_skew2` loop, is every result of every Cell bitwise unchanged, does the
early-abandon test stay out of the loop that has no threshold in every build, and how much faster is it?

**Answer (phase 1).** The linked `dtwc_cl` runs `v_skew2`'s loop in both kernels: f64 L1 21 instructions per two cells
(13 per cell at base), f64 squared 23 (14), the same on `s` registers for f32; no call or trap; per two cells 4 loads and
1 store against base's 6 and 2 [confirmed]. Every output is bitwise unchanged on this toolchain: the probe sweep's 72
hashes (ndim 1: Standard L1/squared, early abandon at four thresholds, ADTW, WDTW, zero-cost, AROW, Soft-DTW; ndim 2
and 3: Standard L1/squared/L2, early abandon, ADTW, WDTW, zero-cost, AROW; f64 and f32; 11.73 M outputs per build)
equal base, conformance and 60 `dtwc_cl` output files are byte-identical, serial ctest is 95 (94 + the CUDA skip),
fresh-venv pytest 921/16/0 [confirmed]. The first attempt was **FALSIFIED** for ADTW f32: on a one-row matrix the
library's `-fassociative-math` regrouped `left + penalty`, moving 15 of 270,000 outputs by an ulp; a one-row matrix now
stays on the one-column loop, and the sweep agrees [confirmed]. The early-abandon flag is a template argument of both
kernels: clang unswitched the runtime test only at `-O3`, so at `-O2` and `-Os` base ran the column minimum in every
cell; now no level does [confirmed]. **Phase 2** (quiet machine, on battery; probe clock median 4.52 GHz): kernel 1
passes its bands, unbanded single pair 1.14–1.98× in every placement (f64 L1 1.42–1.98×), the ragged fill 1.42–1.50×,
the lanes fills 0.98–1.00×; kernel 2 is **FALSIFIED** by its band (head at 100 × 100 band 10 down to 0.85×, though
1.11–2.52× elsewhere) and reverted (73519cd7) [confirmed]. The same instructions move by up to 40 % between
placements; every banded shape k1 runs below 0.97× is base's kernel 2 loop with an `fcmp`/`fcsel` pair across a 64-byte
boundary [confirmed].

## Registered before the runs

The orchestrator's bands (before any measurement): kernel 1 unbanded ≥ 1.3× at L 100 and L 1000 (f64 L1) in every
placement, no shape below 0.97× in any placement; the ragged unbanded fill ≥ 1.2×; kernel 2 kept only if banded shapes
are ≥ 1.15× in every placement and none below 0.97×; the equal-length (lanes) fill unchanged within 0.97–1.03×. My
reading, fixed in `kit/summary.py` before any measurement: "banded shapes ≥ 1.15×" are the band-L/10 shapes (100 × 100
band 10, 1000 × 1000 band 100) and the ragged pair at band 20; "none below 0.97×" covers every single-thread shape of
head, including a narrow 100 × 100 band 2, and the ragged band-10 fill at 18 threads; the speed-ups are wall clock, with
the same verdicts recomputed on cycles at the add-chain clock as a cross-check (if the two disagree on kernel 2, rerun).
Mine, for the code: every cell gets `cell.combine(diag, up, left, cost(i, j), i, j)` with the same arguments as base;
the pair loop is `v_skew2`'s (21 instructions per two cells, f64 L1); no `bl`/`blr`/`brk` in it; no load or store beyond
the two columns' own; no column minimum in a loop without a threshold at `-O3`, `-O2` or `-Os`; ctest as base (95: 94 +
CUDA skip, `test_codegen_no_calls` runs); conformance, the sweep and the CLI outputs equal to base bit for bit.

## The change (commits on `pb/pair-2col`)

- **463d2b52 (k1)**, `dtwc/core/dtw_kernel.hpp`: kernel 1's body is `detail::dtw_linear<Abandon>` (:232-308), behind the
  unchanged `dtw_kernel_linear` (:310-320), which picks `Abandon` from the threshold. The pass loop (:264-284) computes
  rows 0..n_short-1 of columns j and j + 1: cell (i, j) from (diag, `short_side[i]`, left), cell (i, j + 1) from
  (left, cell (i, j), next_left), as the one-column loop gives them; row 0 of each column as before. The one-column loop
  (:286-303, base's body) runs an odd last column and every column of a one-row matrix. A column's minimum is kept only
  under `if constexpr (Abandon)` and tested after the pair by `abandons` (:253), so the pair is abandoned iff some column's
  minimum exceeds the threshold, as before. Kernel 2 gets the same split (`detail::dtw_banded<Abandon>`, :327-459 at
  head), its loops unchanged.
- **00fb9c36 (head)**: kernel 2's pass loop (:374-418). Column j + 1's band is column j's moved up by at most one row at
  each end (`dtw_band_bounds`: low and high grow by 0 or 1 per column), so column j alone takes row `first_row` when the
  band has left row 0 (:387-393), row 0 of either column is computed as before, the shared rows run two cells per row
  (:400-411), and column j + 1 alone takes row `high` until the band reaches n_long (:412-416), reading `col[high]` as the
  one-column loop would. An odd last column runs the one-column loop (:420-443).
- **73519cd7**: reverts 00fb9c36 (phase 2: kernel 2 failed its band); the header is 463d2b52's plus 92a747d9's comment.
- **92a747d9**: comment only. The pass loop's comment says that equal arguments give equal results only while the
  compiler keeps the adds as written (the one-row case), and that elsewhere the results matched on Apple clang 21;
  `dtwc_cl`'s and `cpp_conformance`'s `__text` byte-identical to 00fb9c36's (`otool -t`, the path line aside). The kit
  times 00fb9c36, code-identical.

## Deviations from the design, and why

1. **The early-abandon split is in source.** Design item 1 allowed it if the compiler does not unswitch. Probe
   (`probes/unswitch`: `run_dtw<SpanL1Cost>` with a runtime threshold, the library's FP flags, no LTO; inner loops ≥ 10
   instructions; `evidence/unswitch_loops.txt`):

   | build | base e26d5680 | head 00fb9c36, no threshold | head, threshold |
   | --- | --- | --- | --- |
   | `-O3` | kernels 1, 2: 13 (no min) and 15 (with min) | pair 21 (4 `fcmp`), one-column 13 | pair 25 (6 `fcmp`), one-column 15 |
   | `-O2` | one loop each, 16 instructions, column minimum (`fcmp`+`fccmp`+`fcsel`) in every cell | same as `-O3` | same |
   | `-Os` | one loop each, 16, minimum in every cell | same as `-O3` | same |

   No shipped artifact builds at `-O2`/`-Os` today (the CLI is `-O3` [confirmed]; the wheel has `NOMINSIZE`,
   python/CMakeLists.txt:30-32, and the MEX is `-O3` per the audit), so the split changes no shipped loop; a
   `MinSizeRel` or `RelWithDebInfo` build, or a binding without `NOMINSIZE`, kept the minimum in every cell [confirmed
   for clang at `-Os` and `-O2`]. `if constexpr` and a lambda (`Abandon && column_min > early_abandon`) avoid a plain
   `if` on a constant (MSVC /W4 C4127) and keep every variable read in both instantiations [inferred: MSVC and GCC not
   built here].
2. **A one-row matrix stays on the one-column loop (FALSIFIED first attempt).** The first head swept equal to base except
   `f32 adtw L1 p0.5` (hash `defd3662512acd80` against `324063cd9a247fc0`). A side-by-side harness of the two headers
   (`probes/adtw_diff`) located all 15 differing outputs of 270,000 at n_short = 1 (f64: 0); e.g. nx 1, ny 390: base
   397.191162, head 397.191132 (`evidence/adtw_diff_falsified.txt`, rerun on the committed head minus the guard,
   `evidence/falsified_head.diff`). Clang unswitched the inner loop's entry test `1 < n_short` out of the pass loop; in
   the copy for n_short = 1, row 0's `left = min + c` had one use, `left + penalty` in column j + 1's row 0, and under the
   library's `-fassociative-math` it compiled to `fadd s4, s1, s4` (p + c) then `fadd s0, s0, s4`: `m + (p + c)` for the
   source's `(m + c) + p` (`evidence/prefix_k1_f32_adtw_dtw_linear.s`, line 125). In the pass loop `here` and `left` also
   feed the next row's phi, so they have two uses, and LLVM's Reassociate (like GCC's reassoc) linearises single-use
   chains only [inferred from the passes]; base never had a same-pass chain (every neighbour came from memory or a
   loop-carried phi). The pass loop now requires n_short > 1 (dtw_kernel.hpp:264); the harness then gives 0 of 270,000
   for both types (`evidence/adtw_diff_head.txt`), and the full sweep agrees [confirmed]. StandardCell, AROWCell and the
   cost functors add nothing to a neighbour, so only ADTW and Soft-DTW had such a chain; the multivariate costs' sums over
   a runtime `ndim` (also open to regrouping) sweep equal too.
3. **The kit times three builds**, base, k1 (463d2b52) and head (00fb9c36), so phase 2 decides kernel 2 from one run; if
   kernel 2 is dropped, `git revert 00fb9c36` restores k1's code exactly (92a747d9 touches only kernel 1's comment), and
   k1 has its own ctest, conformance, sweep and CLI evidence below.

## Assembly: the per-pair loops of the linked `build/bin/dtwc_cl` (ThinLTO; `objdump -d`)

Listings: `2026-10-06-mac-pair-2col/asm/perpair_base.s` (8 loops), `asm/perpair_head.s` (20 loops). The CLI never passes
a threshold, so LTO keeps only the `<false>` instantiations of the Standard kernels [confirmed].

| loop (Standard cell) | base: instructions per cell | head: pair loop, per 2 cells | head: one-column loop | loads / stores per 2 cells, base → head |
| --- | --- | --- | --- | --- |
| kernel 1 f64 L1 | 13 | 21 (10.5 per cell) | 13, base's loop (same code, registers renamed) | 6 / 2 → 4 / 1 |
| kernel 2 f64 L1 | 13 | 21 | 13, base's | 6 / 2 → 4 / 1 |
| kernels 1, 2 f64 squared | 14 | 23 (11.5) | 14, base's | 6 / 2 → 4 / 1 |
| kernels 1, 2 f32 L1 / squared | 13 / 14 | 21 / 23 | 13 / 14, base's | 6 / 2 → 4 / 1 |

Head kernel 1 f64 L1 (`1000ada58`; two chains: `left → fcmp → fcsel → fadd → here`, then `here → fcmp → fcsel → fadd`):

```
ldr d5, [x12] ; ldr d6, [x15], #8 ; ldr d7, [x19, x13, lsl #3] ; fabd d7, d6, d7        old_up, x[i], y[j], cost(i, j)
fcmp d5, d4 ; fcsel d4, d5, d4, mi ; fcmp d3, d4 ; fcsel d4, d3, d4, mi                min(min(diag, old_up), left)
ldr d16, [x19, x14, lsl #3] ; fadd d7, d4, d7 ; fabd d4, d6, d16                       here; cost(i, j + 1)
fcmp d7, d3 ; fcsel d3, d7, d3, mi ; fcmp d2, d3 ; fcsel d2, d2, d3, mi ; fadd d2, d2, d4   min(min(left, here), next_left) + cost
str d2, [x12], #8 ; mov.16b v4, v5 ; mov.16b v3, v7 ; subs x16, x16, #1 ; b.ne
```

This is the audit's `v_skew2` loop (`2026-10-06-mac-kernel-assembly/asm/variants.s`) instruction for instruction, register
numbers aside; `y[j]` and `y[j + 1]` are reloaded every row as base reloads `y[j]` (the store to `short_side` may alias
them). In kernel 2's listing the first loop of each type (25 / 27 instructions, one `fminnm`, from `std::min(max, x)`) is
the per-pass code (row 0 and the rows only one column has), not a per-cell loop; it is the whole pass only at band 0. No
`bl`/`blr`/`brk` in any listed loop. `test_codegen_no_calls` (per-TU, LTO off): `inner_loops=153 calls=0 verdict=PASS`
at head [confirmed] (141 at e26d5680, per the arm-lanes record).

## Bitwise [confirmed]

- **ctest** (`ctest --test-dir build -C Release -j1 --output-on-failure`; `evidence/ctest_summary.txt`): k1 and head each
  95 tests, 94 passed, 1 skipped (`test_cuda_correctness`, MAY_SKIP), `test_codegen_no_calls` ran and passed. k1's checks
  ran before a comment-only edit of its header; the rebuilt `dtwc_cl` and `cpp_conformance` have `__text` byte-identical
  to the binaries they ran (`otool -t`, the path line aside).
- **Conformance** (`DTWC_CONFORMANCE_REGEN=1 build/bin/cpp_conformance`, the tracked file restored after each;
  `evidence/conformance_cli_compare.txt`): base, k1 and head regenerated files byte-identical; each differs from the
  tracked reference by D-19's silhouette ulp (`0.9689497276433483` against `0.96894972764334841`). Its 27 series are
  equal-length, so its fill ran the lanes.
- **Sweep** (`kit/src/pprobe.cpp check 3`, each build with its own headers and `dtw_lanes.cpp`, ThinLTO, placement p0;
  output `kit/check_output_base_k1_head.txt`, the three builds' files byte-identical): through the public per-pair
  functions (`dtwBanded`, `adtwBanded`, `wdtwBanded(…, g)`, `dtwMissing_banded`, `dtwAROW_banded`, `soft_dtw`, the `_mv`
  forms, and `run_dtw<SpanMVAROW…>` as `dtw_dispatch.cpp` calls it); lengths 1..1000 each against one of the same
  length, one of a random length 1..1000 and one within ±10; bands −1, 0, 1, L/10, L, |nx − ny|; data kinds uniform,
  random walk, integers {0,1,2} (ties, zero costs, equal series in distinct storage), ±max/2..max (costs overflow to
  inf), random walk with one sample in six NaN; seeds 0–2 for ndim 1 (Soft-DTW seed 0, unbanded), seed 0 for ndim 2 and
  3 (lengths in time steps); thresholds 0, d/2, 3d/4, d of that pair's own distance. Base, k1 and head equal:

| variant | outputs per type | hash f64 | hash f32 | max() f64 / f32 | non-finite f64 / f32 | abandoned f64 / f32 |
| --- | --- | --- | --- | --- | --- | --- |
| dtw L1 | 270,000 | `10c35cac6dfd1967` | `232de09efed5b443` | 72,149 / 72,149 | 78,525 / 78,525 | 0 / 0 |
| dtw Sq | 270,000 | `18c32513aebd9fa2` | `b575924d8e8644cb` | 72,159 / 72,159 | 78,543 / 78,543 | 0 / 0 |
| dtw L1 abandon 0 | 270,000 | `ed45278776058f90` | `5ce3ef42d029e847` | 260,251 / 260,251 | 3,318 / 3,318 | 188,102 / 188,102 |
| dtw L1 abandon d/2 | 270,000 | `2626ec4f4cde6f72` | `64f887404ab0b494` | 173,089 / 173,089 | 78,525 / 78,525 | 100,940 / 100,940 |
| dtw L1 abandon 3d/4 | 270,000 | `7b44fd353e799283` | `3e13493d16284707` | 166,136 / 166,136 | 78,525 / 78,525 | 93,987 / 93,987 |
| dtw L1 abandon d | 270,000 | `10c35cac6dfd1967` | `232de09efed5b443` | 72,149 / 72,149 | 78,525 / 78,525 | 0 / 0 |
| dtw Sq abandon d/2 | 270,000 | `b5087b1195845276` | `e061d05a49d810f9` | 169,484 / 169,484 | 78,543 / 78,543 | 97,325 / 97,325 |
| adtw L1 p0.5 | 270,000 | `569dba4456860c91` | `324063cd9a247fc0` | 72,149 / 72,149 | 78,525 / 78,525 | 0 / 0 |
| adtw abandon d/2 | 270,000 | `270346cfc0278104` | `ea95bc19dedc45dc` | 173,929 / 173,929 | 78,525 / 78,525 | 101,780 / 101,780 |
| wdtw g0.1 | 270,000 | `6c7011450d3b6210` | `29c0c68622cce19e` | 75,151 / 80,106 | 63,965 / 59,010 | 0 / 0 |
| zero_cost L1 | 270,000 | `ee324427247d7d90` | `afd9c69e4b1aaf90` | 72,107 / 72,107 | 38,916 / 38,916 | 0 / 0 |
| zero_cost Sq | 270,000 | `8d0a5c1c8ad3ae39` | `cecd50aea2142e69` | 72,117 / 72,117 | 38,934 / 38,934 | 0 / 0 |
| zero_cost abandon d/2 | 270,000 | `1a9b5280e4fb07fb` | `8d1e9be8ba010248` | 192,021 / 192,021 | 38,916 / 38,916 | 119,914 / 119,914 |
| arow L1 | 270,000 | `c8e8afb2d77d460d` | `d36598189d615d57` | 110,997 / 110,997 | 26 / 26 | 0 / 0 |
| arow Sq | 270,000 | `9b79324203734d8b` | `0d1a66813d166412` | 111,021 / 111,021 | 30 / 30 | 0 / 0 |
| soft_dtw g1 (unbanded) | 15,000 | `277a0558d140bc40` | `5e9966da2f1f1370` | 5,986 / 5,986 | 10 / 10 | 0 / 0 |
| all ndim 1, combined | 8,130,000 in all | | | | | `7534663e7bf5f015` |

| multivariate variant | outputs per type and ndim | f64 ndim 2 | f64 ndim 3 | f32 ndim 2 | f32 ndim 3 |
| --- | --- | --- | --- | --- | --- |
| dtw_mv L1 | 90,000 | `f87af0db99445607` | `d4ec62d6fad06bcf` | `6c6f1913a2bd66b3` | `0294c93c5603bd7c` |
| dtw_mv Sq | 90,000 | `40714005a874d68b` | `bbefe095b1a0e036` | `a91f5a41cd4263c6` | `40241f4ab1ea631c` |
| dtw_mv L2 | 90,000 | `942e5c0bd04e3ec7` | `13473925340d97a6` | `d054101e4892f402` | `79065787863d485f` |
| dtw_mv L1 abandon d/2 | 90,000 | `ad6540438ac59081` | `c72a8f20eaf8cc81` | `2670e6c406e0edb6` | `75ab0b5e4bcb6359` |
| adtw_mv p0.5 | 90,000 | `5117ad4ba40abcbb` | `14ff375279ce8878` | `7732bbd5f22d442d` | `c91ba52a9423a867` |
| wdtw_mv g0.1 | 90,000 | `42faa84c45bb2da5` | `79a5b689e2a5ba81` | `0d765fe0255cf8ed` | `b08aba0ec20ab5ad` |
| zero_cost_mv L1 | 90,000 | `d2a22f1834079a52` | `b7373ef460900a30` | `4f8f1893dd3b6581` | `361eee7289edadf9` |
| zero_cost_mv L2 | 90,000 | `0d354847691b7ecc` | `ab4bcf939c3033c0` | `9bd902a58585054d` | `be0c6cfa37fb8e23` |
| arow_mv L1 | 90,000 | `13b2819d63f5177d` | `cb40eafc54171d2f` | `6812e85fc4bc62d8` | `da94c3522cf517ce` |
| arow_mv Sq | 90,000 | `3ad63cd8bda51e9c` | `bbc7236c6bb448fc` | `e427fefb0b6e26c8` | `76b7a13d62bd3e41` |
| all multivariate, combined | 3,600,000 in all | | | | `139f4c59f717903f` |

  The probe's per-pair loops equal the linked `dtwc_cl`'s (below), so the sweep runs the shipped code for the Standard
  cells; the other cells' loops are the probe's copies, compiled with the same flags.
- **CLI** (`dtwc_cl -k 3 -m pam --skip-rows 1 --skip-cols 1`, base, k1 and head binaries;
  `evidence/conformance_cli_compare.txt`): `data/dummy` (25 series, 5148–9405 samples, every pair unequal: the per-pair
  path) unbanded; `--band 10` (refused with the typed error "band = 10 is narrower than the length difference …",
  compared as a log); `--variant adtw --band 4300`; `--variant wdtw --wdtw-g 0.1`; `--missing-strategy zero_cost`;
  `--dtype float32`; `--band 4300`; `--metric squared_euclidean`; `--missing-strategy arow`; `--variant softdtw`;
  `--dtype float32 --band 4300 --metric squared_euclidean`; and, because band 10 cannot run on `data/dummy`,
  `dummy_ragged` (series k cut to its first 1000 + k mod 11 samples: lengths 1000–1010) at `--band 10`, `--band 10
  --metric squared_euclidean --dtype float32`, `--variant adtw --band 10`, `--variant wdtw --wdtw-g 0.1 --band 10`,
  unbanded. 16 runs, 60 output files (distance matrix, labels, medoids, silhouettes) and 16 logs (timing line aside):
  0 differing, k1 and head against base.
- **Python** (the runbook's fresh venv outside the repo, `.[test,dev,io]` + matplotlib, `DTWC_CL_PATH` the head
  `dtwc_cl`): 921 passed, 16 skipped, 0 failed (Python 3.12.14; no `mip` extra: the documented 16 skips; one warning,
  sklearn skipping its array-API check).
- **Docs gates** (k1 and head): `check_docs.py --cli build/bin/dtwc_cl` VERDICT=PASS (397 flags, 59 pages);
  `check_pins.py` failures=0; `generate_docs.py --check` current.
- **Lint** (the preset build has `DTWC_DEV_MODE` off; `evidence/lint_summary.txt`): `cmake/CompilerWarnings.cmake`'s
  clang set (`-Wall -Wextra -Wshadow -Wconversion -Wsign-conversion -Wpedantic …`) on `pprobe.cpp` (every per-pair Cell
  through the public functions) and `scripts/codegen_probe.cpp`, base and head headers: no warning in `dtw_kernel.hpp`
  at either; the other warnings (the probe's own, `twe.hpp:69`) are the same at both.
- **Review**: a separate agent read the diff and traced kernel 2's asymmetric rows by hand (3×3, 4×5, 6×6 at band 1;
  4×6, 5×7 at band 2) and found nothing that breaks bitwise equality; its kit findings are fixed below (the band-10
  fill and a narrow band judged, a failed probe withholds the verdicts, the clock cross-check, this evidence saved).

## Timing kit (for the quiet run; nothing timed here)

`/private/tmp/claude-504/-Users-engs2321-git-dtw-cpp/11a5974a-eee8-48c6-bc97-994bfa954411/scratchpad/pair-2col-timing/`,
one command: `./run.sh` (5 repeats; `./run.sh N` for N; `./run.sh dry` is the smoke run). Sources also in
`2026-10-06-mac-pair-2col/kit/`; to rebuild there: `./prepare_src.sh e26d5680 463d2b52 00fb9c36`, then `./build_all.sh`.

- **Probes.** `pprobe_<build>_<placement>`: `src/pprobe.cpp` with that build's headers and `dtw_lanes.cpp`, compiled with
  `dtw_dispatch.cpp`'s compile command (`-O3 -march=native`, the library's FP flags, ThinLTO). The per-pair call is the
  fill's (`make_standard`: a `std::function` over `normalize_public_distance(dtwBanded<T>(x, y, band, T(-1), metric))`);
  its loops equal the linked `dtwc_cl`'s instruction for instruction, registers renamed (p0: base 8 of 8, k1 12 of 12,
  head 20 of 20; `same_loops.py`) [confirmed].
- **Placements** (link time only, where ThinLTO generates the code): p0 the default layout behind a 4-byte `_kit_pad`
  that `-Wl,-order_file` puts first in `__text`; p1 adds `-Wl,-mllvm,-align-loops=64`; p2 is p0 behind an 8-byte
  `_kit_pad`. Checked on the linked binaries (`placement.py`, report `placement.txt`): functions here are 4-byte
  aligned and loops unaligned by default (head `dtwc_cl`: 2,074 function starts at 0 / 4 / 8 / 12 mod 16: 533 / 506 /
  530 / 505); in p1 every per-pair loop starts at 0 mod 64; in p2 every per-pair loop sits exactly 4 bytes after
  p0's; every per-cell loop's instructions are the same in all three (digests). E.g. kernel 1 f64 L1 starts at 52 / 0 /
  56 mod 64 (base, one-column loop), 12 / 0 / 16 (k1, pair loop), 52 / 0 / 56 (head, pair loop) [confirmed]. p1 is not
  pure placement: its alignment nops sit before each inner loop, run once per entry (per column at base, per pass at
  head); at head they grow kernel 2's per-pass code from 25 to 38 instructions (f64 L1), the only loops in which a p1
  probe differs from `dtwc_cl` (4 of 20) [confirmed].
- **`run.sh`**, per repeat, rotating the order of builds and of placements: single thread, one pair at a time, f64/f32 ×
  L1/squared × {100 × 100, 1000 × 1000} × {unbanded, band L/10}, 100 × 100 at band 2, and the ragged pair 90 × 110
  unbanded and at band 20 (the narrowest that fits); 40 M cells per measurement, 5 per invocation, ns/cell and cycles/cell
  at 4.59 GHz and at the add-chain clock measured around each measurement; checksums compared across builds. Fills at 18
  threads through Problem's fill loop and schedule, f64 L1, Gcell/s counting only cells a kernel computes: ragged N 1000
  L 90–110 unbanded and band 10 (pairs more than 10 apart have no path and cost nothing), equal N 2000 L 100 unbanded and
  N 500 L 1000 band 100 (the lanes: a placement control); matrix hashes compared. A probe that fails leaves a FAILED line.
  About 2–3 minutes per repeat on a quiet machine [inferred from the sizes].
- `check_all.sh` reruns the sweep on the p0 probes (outputs `check_<build>.txt`; the three must be byte-identical).
- **`summary.py`** prints per shape and placement the medians, k1 and head against base (median and range over repeats,
  wall clock and clock-normalised, a flag where a repeat's two clocks differ by more than 3 %), the bands above with the
  kernel 2 decision (KEEP or DROP) and the candidate's remaining bands, once on wall clock and once on cycles. It gives
  no verdict if a probe failed or any shape lacks a build, placement or repeat (checked on a doctored results file: two
  problems reported, every verdict marked NOT VALID).
- **Smoke run** (`./run.sh dry`, `kit/smoke_dry_run_not_a_measurement.txt` and its summary, on a loaded machine): every
  shape ran (252 single-thread lines and 36 fills, no FAILED) and every checksum and fill matrix matched across base, k1
  and head; its numbers (add-chain clock 2.5–2.9 GHz under a load average near 50; 50 shape/placement pairs flagged for
  clock spread) are not measurements.

## Phase 2: the quiet-machine run [confirmed]

2026-10-07 03:29–03:40 BST, `./run.sh` (5 repeats). Conditions: no other agent or build (the orchestrator); on battery
(81 %, discharging); Sophos and Tanium about one core; load average 4.70 / 4.98 / 5.24 before, 6.00 / 5.35 / 5.16 after;
no process above 8 % CPU before the run (`top`, two samples). Probe clock (add chain, around each measurement):
3.94–4.59 GHz, median 4.52; per-row medians 4.28–4.50 GHz in the f64 rows, 4.46–4.58 in the f32 rows; in 21 of 168
build/shape/placement comparisons a repeat's base and build clocks differed by more than 3 %, and the clock-normalised
verdicts equal the wall-clock ones. 1,260 single-thread lines, 180 fills, no FAILED; every checksum and fill matrix
equal across base, k1 and head. Raw: `2026-10-06-mac-pair-2col/phase2/results_20261007_032902.txt`; the summary
verbatim: `phase2/summary.txt`.

Speed-up against base by build and placement (median over repeats; single thread unless named; ranges over the
shapes of the class):

| build, placement | f64 L1 100 unb | f64 L1 1000 unb | unbanded all | band L/10 + ragged b20 | band 2 | ragged fill unbanded | ragged fill band 10 | lanes fills |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k1 p0 | 1.46 | 1.98 | 1.46-1.98 | 0.71-1.17 | 0.85-1.00 | 1.50 | 1.03 | 0.99-1.00 |
| k1 p1 | 1.51 | 1.97 | 1.15-1.97 | 0.99-1.01 | 1.00-1.01 | 1.49 | 1.02 | 0.98-0.99 |
| k1 p2 | 1.42 | 1.97 | 1.14-1.97 | 0.71-1.03 | 0.87-1.04 | 1.42 | 0.89 | 0.99-0.99 |
| head p0 | 1.46 | 1.98 | 1.15-1.98 | 0.90-2.49 | 1.27-1.89 | 1.50 | 1.31 | 1.00-1.02 |
| head p1 | 1.51 | 1.98 | 1.15-1.98 | 0.85-2.52 | 1.16-1.89 | 1.50 | 1.19 | 1.02-1.02 |
| head p2 | 1.45 | 1.98 | 1.09-1.98 | 0.90-2.37 | 1.11-1.97 | 1.49 | 1.26 | 1.00-1.01 |

Verdicts (`summary.py`, wall clock):

- kernel 1 unbanded f64 L1 at L 100 and L 1000 ≥ 1.3× in every placement: k1 min 1.417 → PASS (head 1.451 → PASS).
- kernel 2: banded shapes (band L/10, ragged band 20) ≥ 1.15× in every placement: min 0.845 → **FAIL**; head none below
  0.97×: min 0.845 → FAIL (the ragged band-10 fill 1.19–1.31 → PASS). **DROP**: 00fb9c36 reverted (73519cd7).
- k1 no single-thread shape below 0.97× in any placement: min 0.708 (f64 L1 1000 × 1000 band 100, p2) → **FAIL by the
  letter**; every shape below 0.97× is banded, run on base's kernel 2 instructions (same register-renamed digests at
  other offsets), and k1's ragged band-10 fill at p2 (0.89×) runs the same loop. Kernel 1's own (unbanded) shapes:
  min 1.14 → PASS.
- k1 ragged unbanded fill ≥ 1.2×: 1.496 / 1.493 / 1.419 → PASS. k1 equal-length lanes fills within 0.97–1.03×:
  0.980–0.998 → PASS.

Kernel 2 at 100 × 100 band 10, head against base, p0 / p1 / p2: f32 L1 0.92 / 0.95 / 1.28, f32 squared 0.90 / 0.85 /
1.06, f64 squared 1.25 / 0.98 / 0.90, f64 L1 1.26 / 1.24 / 1.28. In p1 every loop starts on a 64-byte boundary and the f32
L1 pair loop has no split pair, yet runs 0.95×: at 21 rows per column the pass's own rows and two `dtw_band_bounds`
calls take the gain [inferred]. Its other banded shapes: band 2 1.11–1.97, 1000 × 1000 band 100 1.18–1.70, the ragged
pair at band 20 1.45–2.52.

Placement (`phase2/split_pairs.txt`, from `kit/split_pairs.py`: for each probe's loops, whether an `fcmp`/`fcsel`
pair straddles a 64-byte boundary, beside the speed-up of the shapes the loop serves): every k1 banded loop with a
split pair is the slow one — p0 f64 squared (14 instructions at 32 mod 64, an `fcmp` at 60) 0.71–0.93×, p2 f64 L1 (at
36) 0.71–0.89×, p2 f32 L1 (at 44) 0.86–1.00× — and without one the same code runs 0.99–1.17× of base. Kernel 1's
squared pair loop: all 5 placements with a split pair run 1.14–1.41×; of the 7 without one, 4 run 1.46–1.93× and 3
(k1 p2 f64, head p0 f32, head p2 f64) 1.14–1.41× [unexplained]. The shipped binary's placement is its own; this
unit's claim is the kernel 1 speed-up, which holds in every placement measured.

## Not done

- Placement: the per-pair loops move by up to 40 % with their address (above); aligning them (`-mllvm
  -align-loops=64` for the kernels, or `[[clang::code_align(64)]]`) is untried; p1, which does it for the whole probe,
  ran k1's unchanged banded loops at 0.99–1.01× of base.
- GCC and MSVC are not built here (no GCC on this Mac): that their reassociation leaves the pass loop's ADTW, Soft-DTW and
  multivariate results equal to base rests on both passes linearising single-use chains only [inferred]; x86-64 and
  Linux-aarch64 CI are the first evidence. MSVC /W4 on the `if constexpr` and the lambda: not compiled [inferred clean].
- The wheel and the MEX link the same header; their loops were not disassembled for this record [inferred equal to the
  CLI's: all three are `-O3` since `NOMINSIZE`]. The CLI runs had no multivariate input; multivariate equality is the
  probe's.
