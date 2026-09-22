# X-27 — registered band for dropping Eigen

**Registered 2026-09-22, before any measurement.** Machine: Mac (Apple M5 Pro, 18 cores, 64 GiB),
AppleClang 21.0.0, arm64, Release, `DTWC_FP_MODEL=fast`, `DTWC_ENABLE_IPO=ON`.

## What is changing

Eigen is the project's only copyleft dependency (MPL-2.0) and the only MPL obligation in the wheel.
It has exactly two real uses and one dead link:

| site | use | replacement |
| --- | --- | --- |
| `core/scratch_matrix.hpp:28` | `ScratchMatrix<T>` privately inherits `Eigen::Matrix` | own column-major buffer |
| `core/matrix_io.hpp:182` | `to_full_matrix` returns `Eigen::MatrixXd` | return `std::vector<double>` |
| `mip/CMakeLists.txt:31` | links `Eigen3::Eigen` | nothing includes it — verified 0 hits in `dtwc/mip/` |

`to_full_matrix` is not a risk: both binding callers (`_dtwcpp_core.cpp:629`, `:866`) already
allocate a `std::vector<double>` and `memcpy` the Eigen matrix into it, so returning the vector
directly removes an N×N copy rather than adding one.

The risk is entirely in `ScratchMatrix`, because `core/dtw_kernel.hpp:220-221` holds it
`thread_local` and calls `resize(n_short, n_long)` on **every** DTW evaluation.

## The band

**Subject:** `BM_dtwFull/*` and `BM_dtwFull_L/*` in `bench_dtw_baseline` — the kernels that drive
`ScratchMatrix`.

**Prediction:** no-op within noise. **Band: median wall-clock within ±5 % of baseline.**

**Reasoning, stated before the numbers exist:**

1. `Eigen::Matrix::resize` leaves contents uninitialised. `std::vector::resize` value-initialises,
   which on a `thread_local` buffer resized per call would add an O(n_short × n_long) zero-fill to
   every DTW evaluation. The replacement must therefore be **grow-only and uninitialised** —
   `std::make_unique_for_overwrite<T[]>` (C++20, and not a naked `new`), reallocating only when the
   requested size exceeds the capacity already held. That is the row's own gate condition.
2. The header justifies Eigen as "aligned SIMD-ready allocation". X-04 measured that claim and it
   does not hold: **none of the six DTW kernel loops vectorise**
   (`.claude/baselines/2026-09-22-x04-codegen-report.md`), the recurrence being reported as *unsafe
   dependent memory operations*. Alignment that no vector instruction consumes cannot be paying for
   itself, so removing it should cost nothing.

**FALSIFIED if:** the median of `BM_dtwFull/*` or `BM_dtwFull_L/*` regresses by more than 5 %. That
is a result, not a failure, and would be recorded here rather than worked around — the licence
benefit does not entitle the change to a free pass on the hot path.

**Method:** `scripts/run_bench.sh build/bin/bench_dtw_baseline --benchmark_filter='BM_dtwFull'`,
three repetitions each side, median reported with spread. Wall-clock only — this machine has no
PMU (X-24/V-5), so the numbers are advisory by the project's own rule and are reported as such.

---

## Results — **BAND FALSIFIED** `[confirmed]`

The change is correct and Eigen is gone, but the hot path is consistently **~5 % slower**, which is
outside the ±5 % band on two of four sizes and at its edge on the other two. Recorded as a result,
not worked around.

Three repetitions each, `--benchmark_min_time=0.2s`, medians in µs. Within-set spread was ~0.15 %,
so this is not noise.

| benchmark | Eigen (baseline) | attempt 1 | attempt 2 | Δ (attempt 2) |
| --- | --- | --- | --- | --- |
| `BM_dtwFull/100` | 9.543 | 10.098 | 10.055 | **+5.36 %** |
| `BM_dtwFull/500` | 353.089 | 372.339 | 372.354 | **+5.46 %** |
| `BM_dtwFull/1000` | 1488.674 | 1558.244 | 1557.648 | **+4.63 %** |
| `BM_dtwFull/4000` | 24542.671 | 25767.523 | 25788.241 | **+5.08 %** |

**Attempt 1** — `std::unique_ptr<T[]>` via `make_unique_for_overwrite`, grow-only, unsigned extents
with `static_cast` on each access. Both preserved properties held: resize does not initialise and
does not shrink.

**Attempt 2 — hypothesis refuted.** The guess was that mixing `int` indices from the kernels with
`std::size_t` extents forced a zero-extend on every element access, so everything became signed
`Index` like Eigen's. It made no difference at all (+5.36 vs +5.80 on the smallest case, the rest
within noise of attempt 1). **The cause of the 5 % is not identified.** Two attempts is the limit
the project sets, so it is recorded rather than chased further.

What has been ruled out: value-initialisation (the buffer is uninitialised by construction),
repeated allocation (grow-only; the `thread_local` instance allocates once), and index signedness
(attempt 2). What has *not* been examined: allocation alignment (Eigen aligns dynamic matrices
itself; `new T[]` gives 16 bytes on this platform), and whether Eigen's `resize` fast path differs
in a way that matters at these sizes. A PMU run would answer it and this machine has no PMU (V-5).

## The trade, stated plainly

This is a licence-versus-performance decision and it is Volkan's, not mine:

- **Keeping the change** removes the project's only copyleft dependency, the only MPL-2.0 obligation
  in the wheel — no more MPL §3.2 source offer, one fewer pinned tarball, and `THIRD_PARTY_LICENSES.md`
  loses its largest section. Cost: ~5 % on `BM_dtwFull`, wall-clock, on one machine.
- **Reverting** keeps today's speed and keeps the MPL obligation, which standing rule 18 ("avoid
  copyleft") exists to avoid.

Both branches are cheap: the change is self-contained and the working tree holds it uncommitted.
Note also that the 5 % is measured *wall-clock on a laptop with no PMU*, which the project's own
rule calls advisory — the number deserves a bare-metal Linux confirmation (V-5) before it decides
anything permanent.
