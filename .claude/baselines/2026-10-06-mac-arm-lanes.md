# 2026-10-06 — AArch64 lanes: the min as `fminnm`, 128-byte blocks (Mac)

Unit arm-lanes (PLAN "After G": ☐ AArch64 lanes; phase C's lead "16 double lanes to hide the min-then-add latency"),
from design-2.0 254ecd3b on `pb/arm-lanes`. Apple M5 Pro (18 OpenMP threads), macOS (Darwin 25.6.0), Apple clang 21.0.0
(clang-2100.3.34.2), `clang-macos` preset; base and head built in the same worktree `build/`, base first. Evidence
behind the design: `2026-10-06-mac-kernel-assembly.md` (variant `v_fmin_w16`, recommendations 2 and 3). **No timing in
this record**: other agents were building on the machine, so the speed is left to the kit below, run on a quiet machine.

**Question.** With `dtw_lanes<T>` = 128 / sizeof(T) and an `fminnm` min, both on AArch64 only and in the lanes only,
does the linked binary ship the audit's `v_fmin_w16` loop, is every result bitwise unchanged, and is x86-64 untouched?

**Answer.** Yes on all three [confirmed]. Head's four lanes loops in the linked `dtwc_cl` take every min as one
`fminnm.2d` / `fminnm.4s`, all on q-registers, with no call or trap and one stack reload per row; the f64 L1 loop is
the audit's `v_fmin_w16` loop instruction for instruction (GPR numbers and the stack offset aside). Per cell: f64 L1
4.50 → 3.50 instructions, f64 squared 5.00 → 4.00, f32 L1 2.25 → 1.66, f32 squared 2.50 → 1.91. On x86-64 the three
translation units that instantiate or size the lanes compile to byte-identical assembly at base and head, with
`-march=x86-64-v3` and without. Serial ctest 95 = 94 passed + 1 skip (CUDA), as base; conformance output byte-identical
to base; the sweep's hashes equal (15.36 M lanes outputs per build, each bitwise equal to the per-pair kernel); `dtwc_cl`
64 output files byte-identical over 16 runs.

## Registered before the runs

The orchestrator's bands for the timing kit (before any measurement): single thread every shape ≥ 1.15× and f64 L1
≥ 1.3× at all four shapes, none below 0.97×; 18-thread equal fill ≥ 1.3×; ragged fill 0.97–1.03×. Mine, for this
record: the min is `fminnm` and nothing else; no `bl`/`blr`/`brk` in a lanes loop; no spill or reload beyond
`v_fmin_w16`'s one; x86 instruction-identical; ctest as base (95: 94 + CUDA skip, `test_codegen_no_calls` and the lanes
tests run); conformance, sweep and CLI outputs equal to base bit for bit.

## The change (23c88336 on `pb/arm-lanes`)

- `dtwc/core/dtw_kernel.hpp`: `LanesCell` (beside the lanes kernel, not in the Cell section, so the MSVC note on nested
  two-argument `std::min` there stays true of every cell it covers): on `__aarch64__` a `StandardCell` whose `combine` is
  `std::fmin(std::fmin(diag, up), left) + cost`; elsewhere `using LanesCell = StandardCell` (x86 instantiates the very
  same template). `dtw_lanes<T>` = 128 / sizeof(T) under `__aarch64__`, else 64 / sizeof(T). Comments state the reason
  (enough chains that the FP pipes, not the min-then-add latency, set the pace; Apple silicon's line is 128 bytes).
  `alignas(64) Row`, `kUnreachable`, the stack arrays and the tail blocks are W-generic (checked; the CLI runs pad
  partial blocks at W 16 and 32).
- `dtwc/core/dtw_lanes.cpp`: the block function passes `LanesCell{}`. `dtw_dispatch.hpp`: "bit for bit" holds on the
  series the fill admits.
- `scripts/codegen_probe.cpp`, `tests/unit/core/unit_test_dtw_kernel_lanes.cpp`: the lanes run the shipped cell; the
  fill test derives its sizes from W (2W + 5 series: two blocks and a tail at any W; row 0's whole first block known),
  identical to before at W 8 / 16. `unit_test_one_batch_pam.cpp`, `docs/content/method/dtw.md`: W stated per target.
  `CHANGELOG.md`: one line at the top of Unreleased, with the audit's speed until the quiet run.

## Exactness: why `fminnm` changes no admitted result (lines opened)

`std::min(a, b)` is `b < a ? b : a`; `fminnm` can differ from it only when an operand is NaN (`fminnm` returns the other
operand) or the operands are +0 and −0 (`fminnm` returns −0). So the lanes are unchanged unless a NaN or a −0 reaches a min.

- Only admitted input reaches the lanes. `resolve_dtw_block_fn` returns a lane function only for Standard DTW,
  `MissingStrategy::Error`, ndim 1 (dtw_lanes.cpp:43-56). Its callers: `Problem::rebind_dtw_fn` (Problem.cpp:430, 435),
  whose functions run only in `fillDistanceMatrix_BruteForce`'s `fill_lanes` (Problem.cpp:798, 803), a private member
  (Problem.hpp:134, 195) with one caller, `fill_distance_matrix` (Problem.cpp:856), after `validate_fill_request`
  (Problem.cpp:843); and OneBatchPAM's `FixedBatchDistances` (one_batch_pam.cpp:101-102, run at :118), whose member
  initialisers first call `problem.dtw_function()` / `dtw_function_f32()` (one_batch_pam.cpp:84-85), which run
  `validate_fill_request` (Problem.cpp:529, 537). That check (Problem.cpp:697-712) runs `detail::require_finite` over
  every series with `nan_is_missing = missing != Error`; under Error, the only strategy with lanes, NaN and ±inf both
  throw (warping.hpp:58-76). `dtw_kernel_lanes` is otherwise called only by `unit_test_dtw_kernel_lanes.cpp` (random
  walks and small integers: finite) and instantiated by `codegen_probe.cpp` (compiled, never run): no test feeds NaN or
  ±inf into the unchecked layer.
- No −0 and no NaN arise inside. The costs are `std::abs(a - b)` and `d * d` (dtw_lanes.cpp:52-56): `fabs` clears the
  sign and a square is +0 for ±0, so no cost is −0, and an input −0 gives +0. The seed is the cost; every other value is
  a min of such values or `max()`, plus a cost, and a sum of non-negative values is never −0. Finite input never makes a
  NaN: `a - b` is finite or ±inf, `|±inf|` and `(±inf)²` are +inf, and nothing subtracts DP values. A cost that
  overflows is +inf, which `fminnm` and `<` order alike; the sweep's kind 3 (values ±max/2..max) makes 942,507 +inf
  outputs in the f64 L1 sweep, each equal at base and head and to the per-pair kernel.
- Input the fill refuses (NaN, ±inf straight into `dtw_kernel_lanes`) can now give a different result from the
  per-pair kernel on AArch64; `LanesCell`'s comment says so. The audit's `probes/kbench/nanprobe.cpp` shows how.

## Assembly: the lanes inner loops of the linked `build/bin/dtwc_cl` (ThinLTO; `objdump -d`, the audit's `lane_loops.py`)

Before/after loops in full: `2026-10-06-mac-arm-lanes/asm/lanes_base.s`, `asm/lanes_head.s`. My base build's loops
equal the main tree's `build/bin/dtwc_cl` (the one the audit read, at `1000ecd8c`) instruction for instruction [confirmed].

| loop (one DP row) | base: W, insns/row (per cell) | head: W, insns/row (per cell) | min | stack |
| --- | --- | --- | --- | --- |
| f64 L1 | 8, 36 (4.50) | 16, 56 (3.50) | 8 `fcmgt.2d` + 8 `bif`/`bit`/`bsl` → 16 `fminnm.2d` | none → `ldr q30, [sp, #0x80]` |
| f64 squared | 8, 40 (5.00) | 16, 64 (4.00) | same | none → `ldr q30, [sp, #0x80]` |
| f32 L1 | 16, 36 (2.25) | 32, 53 (1.66) | 8 `fcmgt.4s` + 8 selects → 16 `fminnm.4s` | none → `ldr q28, [sp, #0x110]` |
| f32 squared | 16, 40 (2.50) | 32, 61 (1.91) | same | none → `ldr q28, [sp, #0x110]` |

Every loop: no `bl`/`blr`/`brk`, no store to the stack; one reload per row, as `v_fmin_w16` (`ldr q30, [sp, #0x30]` in
`2026-10-06-mac-kernel-assembly/asm/variants.s`). Head f64 L1 against `v_fmin_w16`, GPRs, stack offset and branch target
normalised: IDENTICAL, 56 = 56 instructions [confirmed]. Head f64 L1 row, the chain `fminnm → fminnm → fadd` per q-vector:

```
ldur q9, [x1, #-0x40] ; ld1r.2d { v12 }, [x0], #8 ; fabd.2d v10, v12, v30
fminnm.2d v19, v19, v9 ; fminnm.2d v17, v19, v17 ; fadd.2d v17, v17, v10   (x 8 q-vectors, 16 lanes)
... stp q17, q18 ... ; add x1, x1, #0x80 ; 10 mov.16b ; ldr q30, [sp, #0x80] ; subs ; b.ne
```

Outside the loops, in the enclosing block functions [confirmed]: the calls are the thread_local accessors (`blr`),
`vector<Row>::resize` and `__tlv_atexit`, as at base; head's two f32 functions add one `bl ___stack_chk_fail`, the
stack-protector canary compared once per call before `ret`, outside every loop.
`test_codegen_no_calls` (per-TU, LTO off): base `inner_loops=108 calls=0`, head `inner_loops=141 calls=0` (the pack
loop over W unrolls into W loops), both PASS [confirmed].

## x86-64: unchanged [confirmed]

`dtwc/core/dtw_lanes.cpp` (the only library TU that instantiates the lanes), `Problem.cpp` and `algorithms/one_batch_pam.cpp`
(the two that size W), each with its compile command from `compile_commands.json`, `-arch arm64` → `-target
x86_64-apple-macos13` (13.3 for the latter two: `std::to_chars`), `-flto=thin` → `-fno-lto`, `-S`, at base (254ecd3b
export) and head, with `-march=x86-64-v3` and without: all six `.s` pairs byte-identical (10,719 / 8,719 / 27,734 /
27,235 / 9,819 / 9,127 lines). The lanes min there stays one `vminpd` (no `vcmpunordpd`/`vblendvpd`).

## Bitwise [confirmed]

- **ctest** (`ctest --test-dir build -C Release -j1 --output-on-failure`): 95 tests, 94 passed, 1 skipped
  (`test_cuda_correctness`, MAY_SKIP), before the last comment edits (45.0 s) and again at 23c88336 (44.9 s).
  `test_codegen_no_calls` ran (verdict line above); `unit_test_dtw_kernel_lanes` 183 assertions in 3 cases (W 16 / 32:
  37- and 69-series fills); `unit_test_one_batch_pam '[lanes]'` 84 assertions. The rebuilt `dtwc_cl` and
  `cpp_conformance` at 23c88336 have `__text` byte-identical to the binaries every check below ran (`otool -t`).
- **Conformance** (`DTWC_CONFORMANCE_REGEN=1 build/bin/cpp_conformance`, reference restored after each): base and head
  regenerated files byte-identical; both differ from the tracked reference by the registered silhouette ulp (D-19:
  `0.9689497276433483` against `0.96894972764334841`). The 27 conformance series are equal-length, so its fill ran the lanes.
- **Sweep** (`kit/src/lprobe.cpp check 3`: the shipped block function through `resolve_dtw_block_fn`, each build linking
  its own `dtw_lanes.cpp`; f64/f32 × L1/squared; lengths 1..1000; bands −1, 0, 1, L/10, L; 4 data kinds — uniform, random
  walk, integers {0,1,2} with every fourth series equal to x, ±max/2..max overflowing to inf; 3 seeds, seed 0 the audit's
  data): hashes below, base = head, and every lane bitwise equal to the per-pair kernel in both.

| kernel | T | cost | W base / head | outputs per build | hash (base = head) | lanes ≠ per-pair (base, head) | non-finite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lanes | f64 | L1 | 8 / 16 | 3,840,000 | `9028e017684e2607` | 0, 0 | 942,507 |
| lanes | f32 | L1 | 16 / 32 | 3,840,000 | `b460be733cdc060c` | 0, 0 | 942,507 |
| lanes | f64 | squared | 8 / 16 | 3,840,000 | `ec936b4c11439a03` | 0, 0 | 943,485 |
| lanes | f32 | squared | 16 / 32 | 3,840,000 | `40b08d92a3ebe51b` | 0, 0 | 943,485 |
| per-pair (3 partner lengths, 6 bands) | f64 / f32 | L1 | — | 216,000 each | `1e7a1fe45449fde2` / `c8eed1c8b1e628e0` | — | — |
| per-pair | f64 / f32 | squared | — | 216,000 each | `b4b08fdef3da6fd8` / `34c544640bb6b6f5` | — | — |
| all, combined | | | | | `3efe380b36d3e4bf` | | |

Raw output: `scratchpad/arm/check_base.txt`, `check_head.txt` (the session scratchpad named under the timing kit).

- **CLI** (`dtwc_cl -k 3 -m pam`, base and head binaries): `tests/conformance/data/conformance_series.csv` (27 × 16, the
  repository's only equal-length series set: integer values, so ties and zero costs in every min) and `data/dummy`'s 25
  series cut to their first 1000 samples (real-valued; 24 columns per row: a full block and a padded tail at W 16, one
  padded block at W 32); float64/float32 × l1/squared_euclidean × band −1 / band 3 (conformance) or 100 (dummy): 16 runs,
  64 files (distance matrix at shortest round-trip digits, labels, medoids, silhouettes), 0 differing.
- **Python** (the runbook's fresh venv outside the repo, `.[test,dev,io]` + matplotlib, Python 3.12.14, `DTWC_CL_PATH` the
  head `dtwc_cl`): 921 passed, 16 skipped, 0 failed (no `mip` extra: the documented 16 skips).
- **Docs gates**: `check_docs.py --cli build/bin/dtwc_cl` VERDICT=PASS (397 flags, 59 pages); `check_pins.py`
  failures=0; `generate_docs.py --check` current.

## Timing kit (for the quiet run; nothing timed here)

`/private/tmp/claude-504/-Users-engs2321-git-dtw-cpp/11a5974a-eee8-48c6-bc97-994bfa954411/scratchpad/arm-lanes-timing/`,
one command: `./run.sh` (5 repeats; `./run.sh N` for N; `./run.sh dry` is the smoke run). `lprobe_base` / `lprobe_head`
link each version's own `dtw_lanes.cpp` (`build.sh`: its compile command, ThinLTO); their four lanes loops equal the
linked `dtwc_cl`'s, base and head, instruction for instruction [confirmed]. Each repeat alternates which build runs first:
single thread (the lane function alone; f64/f32 × L1/squared × L 100, 1000 × unbanded, band L/10; 5 measurements of
120 M cells per invocation; ns/cell, cycles/cell at 4.59 GHz and at the add-chain clock) and three fills at 18 threads (f64 N 2000 L 100
unbanded; N 500 L 1000 band 100; ragged N 1000 L 90–110) through Problem's fill loop and schedule, Gcell/s, with a
matrix hash per build. `summary.py` prints medians, head/base speed-ups with their range over repeats, matrix equality
and the bands above. The smoke run (`./run.sh dry`, `smoke_dry_run_not_a_measurement.txt`, on a shared machine) ran every
shape and matched the three fill matrices; its numbers are not measurements. Its sources are
also in `2026-10-06-mac-arm-lanes/kit/`; to rebuild there, `mkdir -p src/base src/head`, then `git archive 254ecd3b dtwc
| tar -x -C src/base --strip-components=1` (and the head commit into `src/head`), `./build.sh base`, `./build.sh head`.

## The quiet run (orchestrator, 2026-10-07 03:25–03:27, after every other unit had stopped)

`./run.sh 5` in the kit, base and head interleaved, alternating which goes first [confirmed;
`kit/results_20261007_032537.txt`, `kit/quiet_run_2026-10-07.log`]. Conditions: 1-minute load 1.49 at the start; on
battery power; Sophos and Tanium used about one core; the probes' measured clock 4.50–4.57 GHz, so no low-power
throttling. Single thread, median over 5 repeats, speed-up = base ns / head ns:

| shape (f64 / f32) | L1 L100 | L1 L100 b10 | L1 L1000 | L1 L1000 b100 | Sq L100 | Sq L100 b10 | Sq L1000 | Sq L1000 b100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 (W 8 → 16) | 1.888 | 1.647 | 2.002 | 1.949 | 1.555 | 1.521 | 1.599 | 1.563 |
| f32 (W 16 → 32) | 1.798 | 1.419 | 1.949 | 1.865 | 1.579 | 1.411 | 1.627 | 1.572 |

f64 L1 head 0.49–0.55 cycles/cell (the 0.5 floor of 4 FP ops on 4 pipes, 2 lanes); base 0.90–0.99. Fills at 18
threads, Gcell/s, matrices identical: N 2000 L 100 unbanded 66.80 → 111.85 (1.694×, range 1.637–1.717); N 500 L 1000
band 100 64.09 → 101.51 (1.614×, 1.566–1.652); the ragged per-pair fill N 1000 L 90–110 21.73 → 21.64 (1.003×,
0.989–1.009).

Bands: every shape ≥ 1.15× (min 1.411) PASS; f64 L1 ≥ 1.3× at all four shapes (min 1.647) PASS; none below 0.97×
PASS; 18-thread equal fill ≥ 1.3× (1.614, 1.694) PASS; ragged fill 0.97–1.03× (1.003) PASS.

## Not done

- GCC on AArch64 (the Linux-aarch64 wheel, `ubuntu-24.04-arm`) is not built here: that GCC lowers `std::fmin` to
  `fminnm` and vectorises it is [inferred], first exercised by that CI job. MSVC on ARM64 defines no `__aarch64__` and
  keeps W 8 and `std::min` [inferred from the macro]; clang-cl on ARM64 would take the change [inferred].
- The lanes' thread_local scratch doubles on AArch64: 2 · n · 128 bytes per thread (was 2 · n · 64) [inferred from the
  code].
- No speed claim: the CHANGELOG carries the audit's "about 1.4–2.0× single thread" until the quiet run replaces it.
