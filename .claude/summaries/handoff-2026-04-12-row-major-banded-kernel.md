---
name: handoff-2026-04-12-row-major-banded-kernel
description: Session handoff — CHANGELOG v2.0.0 merged into Unreleased; new row-major banded Metal kernel (1.7× CPU, 1.4× Metal wavefront) on tight-band 75×10000 band=100.
type: project
---

# Session Handoff — 2026-04-12 (afternoon) — Row-major Banded Metal Kernel

Branch: **Claude**. Working tree: uncommitted changes on top of the earlier Metal-port session (handoff-2026-04-12-metal-gpu-port.md).

## Goal

User asked to (1) clean up `CHANGELOG.md` — v2.0.0 was prematurely headed, hasn't been released, collapse into `Unreleased` — and (2) continue with next items from the prior handoff. "Ultrathink." The next item was **row-major banded Metal kernel** for tight-band speedup on long series.

## Accomplishments

### 1. CHANGELOG cleanup ([CHANGELOG.md](../../CHANGELOG.md))
- Removed the `# DTWC v2.0.0` header; all prior-v2.0.0 content now lives under `# Unreleased`. No git tag existed for v2.0.0 anyway.

### 2. Row-major banded Metal kernel ([dtwc/metal/metal_dtw.mm](../../dtwc/metal/metal_dtw.mm))
- **New kernel `dtw_banded_row`**: one thread per pair, row-major iteration, no intra-threadgroup barriers. Rolling two row-strips of length W = 2·band+1 in device-memory scratch.
- **Coalesced device-memory layout** (critical): `scratch[half_offset + r*stride + gid]` with `stride = total threads dispatched`. All 32 threads in a SIMD group reading `prev[r]` hit one 128-byte cache line. Without coalescing the kernel ran 2.4 s; with coalescing 1.77 s.
- **Register-rotated prev window**: `prev_r` / `prev_rp1` held in registers and rotated each j-iteration, dropping per-cell device-memory reads from 3 (cdiag/cup/cleft) to 1 (prev_rp1 load for next j). `cleft` comes from `last_cur` register. Dropped runtime from 1.77 s → 1.01 s.
- **Boundary-only INF-fill**: track `prev_r_lo` / `prev_r_hi` per row and only INF-fill cur cells in the delta between adjacent rows' bands (usually 0–2 cells per row), instead of all W cells. Avoids W·La extra device writes per pair.
- **Dispatcher** ([Problem.cpp untouched]; dispatch is in [metal_dtw.mm:593-604](../../dtwc/metal/metal_dtw.mm#L593-L604)): fires `dtw_banded_row` when `band > 0 && band * 20 < max_L && band ≤ 512`. Empirical crossover: at band/L ≈ 5 % the wavefront's 2·L barriers dominate → row-major wins; wider bands put too much sequential work on a single thread and lose to the wavefront's within-pair parallelism. Hard cap at 512 bounds per-thread scratch at ≤ 4 KB.
- **Result metadata**: added `MetalDistMatResult::kernel_used` (`"wavefront" | "wavefront_global" | "banded_row"`) so tests and diagnostic code can assert which path ran. See [metal_dtw.hpp:41-46](../../dtwc/metal/metal_dtw.hpp#L41-L46).

### 3. Tests ([tests/unit/test_metal_correctness.cpp](../../tests/unit/test_metal_correctness.cpp))
- Three new test cases (22 assertions): (a) band=5 / L=400 correctness vs `dtwBanded` CPU reference; (b) band=50 / L=5000 long-series tight-band correctness; (c) wide-band (band=100 / L=200) routes back to `wavefront` kernel. Plus a `cpu_banded_matrix` helper that uses `dtwc::dtwBanded<double>`.
- **Full test suite: 69/69 pass** (ctest), Metal subsuite 2/2 pass (91 assertions across 9 test cases).

### 4. Benchmarks and docs
- **[benchmarks/results/mac_m2max/metal_banded_row_20260412.json](../../benchmarks/results/mac_m2max/metal_banded_row_20260412.json)**: full sweep JSON with the new kernel.
- **[benchmarks/mac_metal_benchmarks.md](../../benchmarks/mac_metal_benchmarks.md)**: new section 5 "Row-major banded kernel (tight-band)" with the 1.01 s / 1.4× / 1.7× numbers, memory-bound analysis, and the dispatcher threshold explanation. Section 5.5 updated the "user-target workload" table with the new kernel selection.
- **[CHANGELOG.md](../../CHANGELOG.md)**: `Added (Metal GPU backend — Apple Silicon)` section updated — three kernel variants now listed (was two) with the tight-band speedup table.

## Headline numbers (Apple M2 Max, 38-core GPU)

**Clean before/after, same machine, 3 repetitions each.** Before numbers taken by temporarily forcing `use_banded_row = false` and rebuilding; after uses the default dispatcher. CV < 0.1% on both Metal runs — the speedup is not measurement noise.

| 75 × 10 000, band = 100 | Mean | stddev | CV | vs CPU | vs wavefront |
|---|---|---|---|---|---|
| CPU 12-thread baseline | 1 760 ms | 5.6 ms | 0.32 % | — | — |
| Metal wavefront (before) | 1 396 ms | 0.2 ms | 0.02 % | 1.26× | 1.00× |
| **Metal `dtw_banded_row` (after)** | **1 011 ms** | 0.5 ms | 0.05 % | **1.74×** | **1.38×** |

Wider-band regression check (dispatcher keeps these on the wavefront — numbers match pre-change baseline):

| Workload | Mean | Path |
|---|---|---|
| 75 × 10 000 unbanded | 5 800 ms | `wavefront_global` (unchanged) |
| 75 × 10 000 band=1 000 | 2 010 ms | `wavefront_global` (unchanged) |

## Decisions (with rationale)

1. **One thread per pair over threadgroup cooperation.** The motivating problem was wavefront-barrier overhead at tight bands. One-thread-per-pair removes all intra-threadgroup barriers. Trade-off: lower GPU occupancy (2775 threads on a 38 K-thread-capacity GPU), so the kernel is memory-bound rather than compute-bound. Acceptable for the target regime (tight bands → low compute per cell anyway).

2. **Coalesced scratch layout with cross-thread stride.** `scratch[r*stride + gid]` over naive `scratch[gid*W + r]`. The naive layout has 32 SIMD threads reading 32 different cache lines per load — saw 2.4 s before coalescing, 1.77 s after. Mandatory for SIMD-level memory efficiency on Apple GPUs.

3. **Register rotation of the prev window.** prev[r] and prev[r+1] are loaded once and shifted each j. cleft becomes the previous iteration's `last_cur` register. Cuts per-cell device reads from 3 to 1. 1.77 s → 1.01 s; consistent with expected memory-bound ~2× speedup.

4. **Boundary-only INF-fill.** Only INF-fill cells outside `[r_lo, r_hi]` that were inside `[prev_r_lo, prev_r_hi]` — i.e., the band-width delta between adjacent rows (≤ 2 cells). Avoids W·La fills per pair (≈ 2 M writes per pair at band=100/L=10000). Correctness relies on the invariant that out-of-band cells in prev row are either never written (INF from init) or have just been INF-filled by the previous row's boundary write.

5. **Threshold `band * 20 < max_L && band ≤ 512`.** Empirically calibrated: at band=100/L=10000 (band/L = 1 %) row-major wins 1.4×; at band=1000/L=10000 (band/L = 10 %) row-major **loses** 30× because per-thread work becomes 10 M cells. Crossover is ≈ band/L = 5 %. The 512 hard cap bounds per-thread scratch at 4 KB — purely defensive for very wide bands that somehow pass the 5 % test.

6. **No Problem.cpp changes.** All dispatch logic is inside `compute_distance_matrix_metal`; the `Problem::fillDistanceMatrix` → Metal path picks up the new kernel automatically. No API changes, no new enum values.

## Open questions / known gaps

- **Metal tight-band is still memory-bound at ~7 % GPU occupancy.** 5.58 Gcells in 1.01 s = 5.5 Gcells/s, one thread per pair. M2 Max has 38 cores × 1024 resident threads = 38 K capacity, we use 2 775 threads. A register-tile / `simd_shuffle` kernel (per the CUDA `dtw_regtile_kernel`) could lift this to near-peak by cooperating across 32 threads per pair without device-memory scratch. Non-trivial MSL work.

- **Wider bands (band * 20 ≥ max_L) still go through the wavefront.** Row-major loses 30× at band=1000/L=10000. A middle-regime kernel (coarse-grained within-pair parallelism, no per-cell barriers) is possible but not urgent — the wavefront is already fast at wide bands (cells-per-diagonal is large enough to amortize the barrier).

- **FP64 is still silent-fallback to FP32** on Metal — unchanged from prior session.

- **No LB_Keogh Metal** — carryover.
- **No 1-vs-all / k-vs-all Metal** — carryover.

## Next steps (if continuing)

1. **Commit** (suggested grouping):
   - Commit A: `CHANGELOG.md` v2.0.0 → Unreleased collapse (small, purely editorial).
   - Commit B: Row-major banded kernel — `dtwc/metal/metal_dtw.mm`, `dtwc/metal/metal_dtw.hpp`, `tests/unit/test_metal_correctness.cpp`, `benchmarks/mac_metal_benchmarks.md`, `benchmarks/results/mac_m2max/metal_banded_row_20260412.json`, and the CHANGELOG kernel-count bump.

2. **`dtw_regtile`-style Metal kernel** for short/medium series (max_L ≤ 256). Each SIMD group of 32 threads cooperates on one pair; each thread holds a register-tile of TILE_W columns (4 or 8); inter-thread communication via `simd_shuffle`. Complexity: high (register pressure, warp-shuffle patterns).

3. **1-vs-all / k-vs-all Metal kernels**. Port `compute_dtw_one_vs_all` and `compute_dtw_k_vs_all` from CUDA. Medium complexity — reuse the wavefront algorithm with a different 2D grid geometry (targets × queries). Enables GPU-accelerated k-medoids cluster-assignment loop on Apple Silicon.

4. **LB_Keogh Metal kernel** + pruning-threshold option in `MetalDistMatOptions`. Medium complexity. Most useful for nearest-neighbor workflows and FastPAM swap-phase pruning.

5. **v3.0.0 (or v2.0.0) tag**. The Unreleased section is very substantial. Breaking-change items (std::span API, C++20 minimum) warrant a major bump. Version string is the user's call — the earlier handoff suggested v3.0.0; the user's intent (see this session's prompt) is that v2.0.0 is still available.

## Files changed (this session only)

```
Modified:
  CHANGELOG.md                                   # v2.0.0 -> Unreleased; kernel-variants section updated
  benchmarks/mac_metal_benchmarks.md             # new section 5 (row-major banded), section 5.5 updated
  dtwc/metal/metal_dtw.hpp                       # MetalDistMatResult::kernel_used
  dtwc/metal/metal_dtw.mm                        # dtw_banded_row MSL kernel; pipeline; dispatcher
  tests/unit/test_metal_correctness.cpp          # +3 test cases, 22 new assertions

New:
  benchmarks/results/mac_m2max/metal_banded_row_20260412.json
  .claude/summaries/handoff-2026-04-12-row-major-banded-kernel.md  (this file)
```

## Reproducing

```sh
cmake --build build --config Release --target test_metal_correctness bench_metal_dtw
./build/bin/test_metal_correctness                                 # 91 assertions, 9 test cases
./build/bin/bench_metal_dtw --benchmark_filter="tightband|b100"    # 1.01 s vs 1.76 s CPU
ctest --test-dir build -C Release -j                                # 69/69 pass
```
