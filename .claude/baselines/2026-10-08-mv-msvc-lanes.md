# 2026-10-08 — MV: cl packs the lanes (a value-returning min, 128-byte lanes under cl) — FALSIFIED

**Status: FALSIFIED** (timed by the orchestrator on a quiet machine, 2026-10-08 13:54:56, 142 s; Result below):
one single-thread shape at 0.867× (f64 squared L 1000 band 100, where the candidate ran in a slow mode in two of
three rounds) and the ragged control at 0.947×. The bitwise clause fails under its literal reading and passes
under the 10-01 ruling's (L1 bit for bit, squared within the route bound, labels and medoids identical), the
reading the orchestrator registered before the run (`2026-10-08-mv-msvc-lanes/results/reading_registered.md`).
`pb/MV` (`75490857`, `dd85b28e`) is not merged. Volkan, the same day: the Windows wheel and release archive move
to clang-cl (DECISIONS 2026-10-08).

Unit MV, from design-2.0 `b3b3a57e` on `pb/MV` (worktree `C:/D/git/wt/MV`). Windows 11, Intel Core Ultra 9 285,
cl 19.50.35723 (MSVC 14.50), clang++ 21.1.8 (GNU-style driver, MSVC ABI, as the main `build/`), GCC 13.3 (WSL
Ubuntu 24.04, the Linux wheel's flags). Prepared on a shared machine by the unit (Opus 5.5). Filed beside this record in `2026-10-08-mv-msvc-lanes/`:
`kit/`, `results/`, `asm/`, `variants/` (the diffs and loop listings cited below), `vec/` (base's and the
candidate's vec-reports); the rest (`probe/`, the other `vec/` reports, `lto/`, `relink/`, `gccdump/`, the CLI
outputs) stays in the local kit `C:/D/git/wt/tmp/MV/`.

**Question.** Volkan (2026-10-08, on moving the wheel to clang-cl): "I like clang but … Can we first try to vectorise
MSVC?" Is there one loop form, with no `#ifdef _MSC_VER` branch of the kernel and no intrinsics, that cl's
auto-vectoriser packs, leaving clang unchanged and GCC unchanged or reported?

**Registered band (the orchestrator's, fixed before any run, verbatim):** "the change is adopted only if, under cl,
every single-thread lanes shape of the XL kit (f64/f32 × L1/squared × L 100 and 1000 × unbanded and band L/10) is
≥ 2.0× base and both 24-thread equal-length fills are ≥ 1.5× base, the ragged fill stays within 0.97–1.03×, the
outputs are bitwise equal, and clang is unchanged (identical assembly, or timed within 0.97–1.03× on every shape).
Otherwise FALSIFIED: the branch is not merged and the clang-cl wheel is the way forward (Volkan decides)."

## cl's diagnosis and the forms tried [confirmed]

`vecrep.bat` (`dtw_lanes.cpp` with the MSVC tree's flags minus `/GL`, plus `/Qvec-report:2`), summarised by
`vecsum.py` (`vec/<form>.txt`). Reason 1200: a loop-carried dependence; 1303: too few iterations; 1500: possible
aliasing on multi-dimensional arrays; 1300: no computation; 1106: outer loop.

| form | the row step's lane loop (the hot loop) | other lane loops |
| --- | --- | --- |
| base: `std::min(std::min(diag, up), left)` (StandardCell), W 8 / 16 | **1200**, all four (f64/f32 × L1/Sq) | column 0's lane loop 1200; the rest 1500/1300 (per column or per call) |
| base's `std::min` + `#pragma loop(ivdep)` (W 16 / 32) | 1200 | — |
| `std::min` on the lane arrays' elements (probe `probe/tform.cpp`) | 1200 | — |
| value min, std::min's own comparison by value (`up < diag ? up : diag`, then `left < m ? left : m`), W 8 / 16 | f32 (16): **vectorized**; f64 (8): **1303** | |
| trip-count probe (`probe/t1303*.cpp`, value min) | a constant trip count of 4–11: 1303, double or float; 12, 16, 32: vectorized | |
| value min + 128-byte lanes under cl (W 16 / 32): **the candidate** | **vectorized**, all four | column 0's 1500 |
| + `__restrict` on the row pointer | vectorized; without `/GL` the per-row alias test and its scalar copy go (`variants/fas_loops_restrict.txt`), but the `/GL`/LTCG code keeps both, as without it (`variants/ltcg_loops_restrict.txt`, `ltcg_loops_v6.txt`) | dropped: no effect where it ships |
| + `#pragma loop(ivdep)` | vectorized, and the LTCG code loses the alias test too (`variants/ltcg_loops_v5.txt`) | not taken: an MSVC pragma in the shared kernel (clang and GCC warn under `-Wall`) |
| + `#pragma fp_contract(off)` in `dtw_lanes.cpp`, after the includes | 1200: cl does not inline the costs (contraction off) into the kernel (on): 8–16 calls per row step | rejected |
| + `#pragma fp_contract(off)` first in `dtw_lanes.cpp` | vectorized, no FMA, bitwise in the kit's 3-object link (`check/msvc_cand_pragma.txt`); but in `dtwc_cl.exe`'s full LTCG link the L1 lanes call `std::abs` per lane per row (a COMDAT from a contraction-on file, which cl will not inline into contraction-off code: `variants/ltcg_dtwc_cl_pragma_loops.txt`) | rejected (P1 found the same hazard, DECISIONS 2026-09-29) |
| `/fp:precise` without `/fp:contract` (flags, not source; `fptest2.bat`) | vectorized, all four, 0 FMA | project-wide FP flags: Volkan's |

So cl packs the lanes only with a value-returning min and at least 12 lanes; under `/fp:contract` it then fuses
squared L2's `d * d + min` into `vfmadd231ps/pd`, where its scalar per-pair kernel keeps a multiply and an add
(`vec/disp_base.asm`: no FMA in `dtw_linear<…, SpanSquaredL2Cost, StandardCell>`).

## The change (`pb/MV` `75490857`, `dd85b28e`)

- `dtwc/core/dtw_kernel.hpp`: `LanesCell` off AArch64 is its own Cell (was `using LanesCell = StandardCell`), its min
  std::min's comparison returned by value: the same operand as `std::min` for every input, NaN and ±0 included.
  `dtw_lanes<T>` is 128 bytes also under `_M_X64` (cl; clang-cl already had it). Comments: the 1200/1303 reasons, and
  cl's FMA in the packed lanes next to GCC's.
- `dtwc/core/CMakeLists.txt`: under GCC on x86-64 (`dd85b28e`: not AArch64, whose fminnm lanes are unchanged),
  `dtw_lanes.cpp` gets `--param=max-completely-peel-times=8` (below).
- `docs/content/method/dtw.md`: the lane counts (MSVC builds, such as the Windows wheel, have clang's) and MSVC's
  FMA in its vectorised lanes. CHANGELOG and MAP wait for the verdict (as XL's).

## Assembly [confirmed]

cl: the kit's probes (`build_msvc.bat`: the MSVC tree's flags, `/GL`, linked `/LTCG` as `dtwc_cl.exe` is, with a
`/MAP`), disassembled with names (`tools/msvc_dis.py`, `tools/loops.py`); the candidate's lanes functions were
checked in `dtwc_cl.exe` itself, relinked with a `/MAP` from the tree's objects (`relink_dtwc_cl.bat`;
`variants/ltcg_dtwc_cl_cand_loops.txt`: the same loops, packed, no call). One row step = one `i` of the kernel's
inner loop. The L1 loops are in `asm/main_loops.txt`.

| loop | base | candidate |
| --- | --- | --- |
| cl f64 L1 | scalar `vminsd`, W 8: 134 instructions per row step (16.75/cell), 82 memory operands, 57 rsp-relative | packed ymm `vminpd`, W 16: about 78 per row step (4.9/cell): 8 of row setup, about 15 of run-time alias test (the row pointer against `left`, `diag`, `y`: lea/cmp/branch triples), 4 × 13 packed, 3 of loop; no call |
| cl f32 L1 | scalar `vminss`, W 16: 257 (16.1/cell), 171 memory operands, 122 rsp-relative | packed ymm `vminps`, W 32: about 78 (2.4/cell), the same shape; no call |
| cl squared | as L1 | about 74 per row step (4 × 12: `vfmadd231pd/ps` in place of `vmulpd` + `vaddpd`) |
| clang f64/f32, L1/squared | ymm (XL's 128-byte loops) | **identical**: `dtw_lanes.cpp -S` under `build/`'s compile command, 165 functions, 10,477 instructions, the same after naming (`asm/clang_check.txt`; anonymous-namespace hash and the Cell's name aside) |
| GCC 13.3 f32 L1 (LTO, the wheel's flags) | ymm, 21 per row step (16 lanes), no stack | identical up to register names (`tools/loop_iso.py`); f32 squared: the same 19 instructions, scheduled differently |
| GCC f64 L1 | scalar (W 8), 72–75 per row step, 16 `vminsd`, 14–22 stack operands | scalar, 76–79, 16 `vminsd`, 16–20 stack operands |
| GCC f32 without the peel cap | — | scalar: the 16-lane loop unrolled completely (cunrolli: 19 statements, an estimated 160 ≤ 200), whose selects GCC's basic-block vectoriser does not pack: 169–172 instructions per row step, 56–66 stack operands |

Carried values under cl: in the packed loop every lane's `left` and `diag` are stored to the stack and reloaded the
next row step (cl's loop vectoriser keeps the lane arrays in memory); base kept every lane's `diag` and the `left`
of 4 of 8 (f64) or 5 of 16 (f32) lanes there (XL's record). So a store and reload is added on `left`'s chain for the other lanes;
cl offers no loop form that keeps a loop's arrays in registers. Reported, as the brief asks; the band decides.

## Bitwise [confirmed]

- **Sweep** (`xprobe check 3`, OpenMP 8 threads; `check/msvc_base.txt`, `check/msvc_cand.txt`): XL's sweep, the
  shipped block function against the shipped per-pair route, each probe linking its own `dtw_lanes.cpp` and
  `dtw_dispatch.cpp`; MV adds a count against the route bound of `tests/support/dtw_route_bound.hpp`.

| T, cost | W base/cand | lane outputs | hash base | hash cand | lanes ≠ per-pair (cand) | outside the route bound | max deviation (bound 2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 | 8/16 | 4,140,000 | `24e36e73a3691ea6` | the same | 0 | 0 | 0 |
| f32 L1 | 16/32 | 4,140,000 | `46691aba0dcc9185` | the same | 0 | 0 | 0 |
| f64 squared | 8/16 | 4,140,000 | `5a8bc2b4490f5d5e` | `5e25241a6f326bdc` | 672,624 | 0 | 0.309 |
| f32 squared | 16/32 | 4,140,000 | `9ac1ed55cd566b15` | `b515b0b120c479bc` | 670,401 | 0 | 0.328 |
| per-pair, f64 / f32 L1 and squared | — | 216,000 each | `e474310748f9a49f` / `c8eed1c8b1e628e0` / `b4b08fdef3da6fd8` / `34c544640bb6b6f5` | the same | — | — | — |

  Base: 0 mismatches, combined `0fdf8bb2e6b24725`, every hash equal to XL's cl base. Max deviation: the largest
  |a − b| / ((2n − 1) eps(T) max(|a|, |b|)) over the mismatches.
- **CLI** (`cli/cli_runs.sh`, XL's 16 runs, 64 files per build): clang++ 64 of 64 byte-identical to base. cl 58 of 64:
  the 6 others are the squared-L2 runs on `data/dummy` (25 × 1000): the four distance matrices (about 1 entry in 3
  differs, by at most 1.8 eps(float) and 2.5 eps(double) relative) and the two float32 silhouette files (the last printed
  digit); labels and medoids are identical in every run, and stdout differs only in the two float32 squared runs' total cost
  (96725.26697 → 96725.26868 unbanded, 113268.73120 → 113268.73279 band 100).
- **cpp_conformance** (`DTWC_CONFORMANCE_REGEN=1`, the reference restored after each; L1, band 3): byte-identical to
  base under both compilers, and equal to the tracked reference (line endings aside).
- **The timed fills** are f64 L1: their matrix hashes are compared in the report.

## Gates in the worktree [confirmed]

| tree | base `b3b3a57e` | candidate `75490857` (`dd85b28e` changes no clang or cl compile: both reconfigure with no work) |
| --- | --- | --- |
| `build/` clang++ (as the main `build/`: Ninja Release, Gurobi, HiGHS, llfio, YAML and Arrow OFF, benchmarks; deps from `C:/D/cpm-cache`) | ctest 95 = 92 + 3 MAY_SKIP (CUDA, Metal × 2); codegen `inner_loops=142 calls=0 PASS`; lanes 183 assertions / 3 cases | no new warning; ctest 95 = 92 + 3; codegen 142 / 0 PASS; lanes 183 / 3; OneBatchPAM 10,638 / 12; conformance 8 |
| `build-msvc/` cl (as WU8's: Ninja, Release, tests, Gurobi, YAML OFF; under `vcvars64`) | 2 warnings (STL4047, llfio's `codecvt`); ctest 94 = 91 + 3 | no new warning; ctest 94 = 91 + 3; lanes 183 / 3 (W 16; 175 at W 8 per XL [inferred]) |
| docs gates | — | `check_docs` 434 flags, 58 pages, 0 undocumented, PASS; `check_pins` 0 failures; `generate_docs --check` current |

GCC: `cmake` in WSL (`wsl-gcc/`, configure only) puts `--param=max-completely-peel-times=8` on `dtw_lanes.cpp` alone;
the assembly above is the LTO probe (`lto/lto.sh`, `-flto=auto`, as the wheel builds): the param survives LTO.

## Kit (`C:/D/git/wt/tmp/MV/`)

`xprobe.cpp` (XL's, plus the route-bound count), `build_msvc.bat base|cand` (sources in `src/base`, `src/cand` =
`dd85b28e`'s `dtwc/`; `make_cand.py` rebuilds `src/cand` from base), `build_clk.sh` (`bin/clk.exe`: the clock of CPU 12
for cl's cycles per row step), `driver.py run | smoke | report`, `load.sh`, `cli/cli_runs.sh`, `tools/`. Built:
`bin/msvc_base`, `bin/msvc_cand`, `bin/clk.exe`. clang is not timed: its assembly is identical (the band's first
alternative).

**One command, on a quiet machine:** `cd C:/D/git/wt/tmp/MV && uv run --no-project python driver.py run` — 3
interleaved rounds (each reads the clock, then alternates which build goes first), 16 single-thread shapes pinned to
logical CPU 12 and the three 24-thread fills, load records before and after (`load.txt`), every probe line in
`raw.txt`, the report in `report.md`. About 3 minutes (the smoke took 40 s; XL's full run of both compilers took 4).
The report prints the registered band's verdict, and, as facts outside the band, whether the speed clauses pass,
whether the L1 lanes are bitwise, and the squared lanes' count outside the route bound.

Smoke run (04:10, one tiny round, `raw_smoke.txt`, `report_smoke.md`): 22 probe lines, every exit 0; the clock read;
the load records written; fill matrices identical. Its numbers go in no table.

## Result (the orchestrator's run, 2026-10-08 13:54:56, quiet machine) [confirmed]

Load (`results/load.txt`): Windows total CPU 1.5–8.9 % before and 2.8–15.3 % after (10 one-second samples each),
the busiest process at most 1.1 % of all cores, no COMSOL. Clock of CPU 12: 5.271, 5.276, 5.260 GHz.

### msvc: candidate (cand) against base, median over rounds; ratio = base ns / cand ns (single thread) or cand Gcell/s / base Gcell/s (fill), per round

| shape | W base/cand | base ns/cell | cand ns/cell | ratio (median) | range | cycles/row step base/cand |
| --- | --- | --- | --- | --- | --- | --- |
| f32 L1 L 100 unbanded | 16/32 | 0.65157 | 0.11305 | 5.559 | 1.620-5.764 | 54.84 / 19.03 |
| f32 L1 L 100 band 10 | 16/32 | 0.65683 | 0.11562 | 5.681 | 5.538-5.762 | 55.28 / 19.46 |
| f32 L1 L 1000 unbanded | 16/32 | 0.65193 | 0.09918 | 6.586 | 5.919-6.610 | 54.98 / 16.74 |
| f32 L1 L 1000 band 100 | 16/32 | 0.64010 | 0.10330 | 6.323 | 5.909-6.360 | 54.03 / 17.39 |
| f32 Sq L 100 unbanded | 16/32 | 0.63831 | 0.11676 | 5.526 | 5.439-5.577 | 53.72 / 19.71 |
| f32 Sq L 100 band 10 | 16/32 | 0.66975 | 0.11751 | 5.699 | 1.747-5.700 | 56.37 / 19.78 |
| f32 Sq L 1000 unbanded | 16/32 | 0.64418 | 0.10800 | 5.920 | 1.794-6.015 | 54.33 / 18.23 |
| f32 Sq L 1000 band 100 | 16/32 | 0.63937 | 0.11542 | 5.540 | 5.467-5.633 | 53.97 / 19.43 |
| f64 L1 L 100 unbanded | 8/16 | 0.70871 | 0.21381 | 3.402 | 1.030-3.500 | 29.88 / 18.05 |
| f64 L1 L 100 band 10 | 8/16 | 0.76284 | 0.24817 | 2.976 | 2.868-3.226 | 32.17 / 20.95 |
| f64 L1 L 1000 unbanded | 8/16 | 0.73469 | 0.21778 | 3.381 | 3.371-3.502 | 30.92 / 18.34 |
| f64 L1 L 1000 band 100 | 8/16 | 0.72587 | 0.21200 | 3.424 | 3.035-3.531 | 30.64 / 17.90 |
| f64 Sq L 100 unbanded | 8/16 | 0.68342 | 0.24144 | 2.838 | 2.788-2.844 | 28.82 / 20.38 |
| f64 Sq L 100 band 10 | 8/16 | 0.68471 | 0.27531 | 2.570 | 2.315-2.687 | 28.81 / 23.24 |
| f64 Sq L 1000 unbanded | 8/16 | 0.71018 | 0.24487 | 2.889 | 2.721-2.900 | 29.95 / 20.65 |
| f64 Sq L 1000 band 100 | 8/16 | 0.68968 | 0.80188 | 0.867 | 0.692-2.914 | 29.11 / 67.49 |

| fill (f64 L1, OpenMP default threads) | W base/cand | base Gcell/s | cand Gcell/s | ratio (median) | range | matrices |
| --- | --- | --- | --- | --- | --- | --- |
| N 2000 L 100 unbanded, 24 threads | 8/16 | 29.86 | 78.51 | 2.629 | 2.615-2.643 | identical |
| N 500 L 1000 band 100, 24 threads | 8/16 | 30.46 | 76.60 | 2.514 | 2.484-2.546 | identical |
| N 1000 L 100 +-10 ragged unbanded, 24 threads | 8/16 | 30.40 | 30.08 | 0.947 | 0.866-1.133 | identical |

- sweep: 16,560,000 lane outputs per build, hashes base == cand: NO, lanes vs per-pair mismatches (base + cand): 1343025; fill matrices equal: yes
-   cand f32 L1 W 32: hash = base, mismatches against per-pair 0 of 4,140,000, outside the route bound 0, max dev 0.000 (bound 2)
-   cand f32 Sq W 32: hash differs from base, mismatches against per-pair 670,401 of 4,140,000, outside the route bound 0, max dev 0.328 (bound 2)
-   cand f64 L1 W 16: hash = base, mismatches against per-pair 0 of 4,140,000, outside the route bound 0, max dev 0.000 (bound 2)
-   cand f64 Sq W 16: hash differs from base, mismatches against per-pair 672,624 of 4,140,000, outside the route bound 0, max dev 0.309 (bound 2)
- band: every single-thread shape >= 2.0x: min 0.867 (16 shapes) -> fail; both equal-length fills >= 1.5x: 2.629, 2.514 -> pass; ragged fill in 0.97-1.03x: 0.947 -> fail; bitwise -> fail
- not the band, for the record: speed clauses -> fail; L1 lanes bitwise (= per-pair, hashes = base) -> yes; squared lanes outside the route bound -> 0

- clang: dtw_lanes.cpp assembly identical to base: yes (asm/clang_check.txt; identical assembly needs no timing under the band)
- **verdict: FALSIFIED**

**The candidate runs at two speeds, per process.** Each probe's five measurements agree within a few per cent,
but 6 of the candidate's 48 single-thread probes ran 3.3–3.9× slower than the same shape's other rounds, while
base's 48 vary by at most about 15 %. Candidate ns per cell, rounds 0 / 1 / 2 (`results/raw.txt`): f64 squared
L 1000 band 100 0.901 / 0.237 / 0.802; f64 L1 L 100 unbanded 0.688 / 0.214 / 0.208; f32 L1 L 100 unbanded
0.435 / 0.110 / 0.113; f32 squared L 100 band 10 0.117 / 0.384 / 0.118; f32 squared L 1000 unbanded 0.359 / 0.108
/ 0.108. Only f64 squared L 1000 band 100 has its median in the slow mode, which fails the band; every other
single-thread median is 2.57–6.59×. The 24-thread equal-length fills show no such spread (2.48–2.64×). The
ragged control runs the per-pair route, whose code the candidate does not change; its rounds span 0.866–1.133,
so its median of 0.947 is noise in this kit's 24-thread fill [inferred]. FALSIFIED stands on the slow mode
alone. Its cause is not established (Open).

## Open

- Settled by Volkan on 2026-10-08: the Windows wheel moves to clang-cl (unit CW), and the release archive too, with
  the cl build beside it as `dtwc_cl_msvc.exe` (DECISIONS). MSVC builds keep scalar lanes.
- The slow mode (Result): two mechanisms fit, neither tested. cl's run-time alias test may send some processes'
  row steps to its scalar copy (the unit read that copy as never running: a reading of the code, not a
  measurement), or the lane arrays cl keeps on the stack may 4K-alias the heap rows (their distance mod 4096
  changes per process). Decisive test: log those addresses mod 4096, and which branch the alias test takes, over
  about 20 launches of f64 squared L 1000 band 100.
- cl's run-time alias test costs about 15 instructions per row step under `/GL` (the scalar copy it guards never runs);
  `#pragma loop(ivdep)` removes it but is an MSVC pragma in the shared kernel.
- GCC's peel cap is a GCC heuristic's knob; GCC 14 or later may unroll differently. XG times GCC's lane width: with
  the cap, GCC's 128-byte value-min loops match the reference-min ones in instruction count (33/37, `asm/gcc_g128*`).
