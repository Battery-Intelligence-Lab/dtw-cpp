# 2026-09-29 — PF-5: DTW pairs in SIMD lanes, plain C++

**Question:** does computing W independent DTW pairs in SIMD lanes, in plain C++ (no intrinsics, no SIMD
library), beat the fixed scalar kernel by ≥ 1.5× in cells per second on this CPU for the brute-force fill?
**Answer: PASS.** f64 lanes are 3.7–4.2× (W = 4) and 5.2–7.9× (W = 8) the scalar K1-form kernel, full and
banded, every lane bitwise equal to the scalar result. The kill criterion in `DECISIONS.md` §1 is not met.

## Band (registered before the first run)

PASS if lanes / scalar ≥ 1.5× cells/s for f64 at L ∈ {100, 1000, 4000}, full and banded (band = L/10);
FALSIFIED otherwise. Statistic: the median over ≥ 11 interleaved rounds of the paired per-round ratio
t_scalar / t_lanes, both timed on the same W pairs, with the O(L·W) pack inside the lanes time. Evaluated for
W = 4 and W = 8 separately; PASS needs one W that clears all six (L, band) points. Bit-identity (every lane ==
the scalar reference, bitwise) is a precondition: a mismatch voids the timing.

## Machine (`scripts/machine_facts.py --build-dir build`)

| Fact | Value |
| --- | --- |
| OS | Windows 11 (AMD64) |
| CPU | Intel(R) Core(TM) Ultra 9 285 — 24 cores / 24 threads (8 P + 16 E), AVX2, no AVX-512 |
| Memory | 127.5 GiB |
| GPU | NVIDIA RTX 4000 Ada Generation (20475 MiB, compute 8.9, driver 596.72) |
| Compiler | Clang 21.1.8 (x86_64-pc-windows-msvc, MSVC STL 14.50; defines `_MSC_VER` 1950) |
| Build type / flags | Release, `-O3 -DNDEBUG` |
| Repository | design-2.0 @ ff53782 (clean), VERSION 2.0.0rc1 (headers read only) |

Load [confirmed, `GetSystemTimes` minus this process's `GetProcessTimes`, per configuration]: other processes
used 5–49 % of the machine during the single-thread run (median 18.5 % over the 32 configurations), 3–9 %
during the fill check. A `typeperf` snapshot just before: total 18–33 %, CPU 22 itself 5–27 %; top cumulative
CPU: MATLAB, Spotify, Slack, Chrome, plus other agents' builds.

## Probe

Sources in `2026-09-29-pf5-simd-lanes/`: `pf5_lanes.cpp` (kernels, verification, timing, fill),
`run_single.bat`, `vec_probe.cpp` + `vec_report.py` (vectorisation survey). The raw outputs (`verify.txt`,
`single.txt`, `fill.txt`, assembly, remarks) were not kept; the tables below are the record.

- **Scalar reference:** `linear_carry` + `NestedCell` copied verbatim from
  `2026-09-29-windows-kernel/min_carry_probe.cpp` (the K1 form: `std::min(std::min(diag, up), left)`,
  dp[i−1, j] carried, row pointer hoisted); `banded_carry` is the same loop restricted to
  `dtw_band_bounds(band, j, n)` rows per column (equal lengths). Cost `std::abs(x[i] − y[j])`.
- **Third computation:** the library's own `dtw_kernel_linear` / `dtw_kernel_banded` with `StandardCell`.
- **Lanes kernel** (W pairs sharing x; the W y-series interleaved `[L][W]`; per lane the reference's arithmetic
  in the reference's order):

```cpp
template <typename T, std::size_t W>
void lanes_dtw(const T *x, const T *Y, std::size_t n, int band, T *s, T *out) {
  constexpr T maxValue = std::numeric_limits<T>::max();
  std::fill(s, s + n * W, maxValue);
  const std::size_t hi0 = dtw_band_bounds(band, 0, n).second;
  for (std::size_t w = 0; w < W; ++w) s[w] = std::abs(x[0] - Y[w]);
  for (std::size_t i = 1; i < hi0; ++i)
    for (std::size_t w = 0; w < W; ++w)
      s[i * W + w] = std::min(std::min(maxValue, s[(i - 1) * W + w]), maxValue) + std::abs(x[i] - Y[w]);
  for (std::size_t j = 1; j < n; ++j) {
    const auto [lo, hi] = dtw_band_bounds(band, j, n);
    T y[W], diag[W], left[W]; // one or two registers each; y copied so no store can alias it
    for (std::size_t w = 0; w < W; ++w) y[w] = Y[j * W + w];
    std::size_t i = lo;
    if (lo == 0) {
      for (std::size_t w = 0; w < W; ++w) {
        diag[w] = s[w];
        left[w] = std::min(std::min(maxValue, maxValue), s[w]) + std::abs(x[0] - y[w]);
        s[w] = left[w];
      }
      i = 1;
    } else {
      for (std::size_t w = 0; w < W; ++w) { diag[w] = s[(lo - 1) * W + w]; left[w] = maxValue; }
    }
    for (; i < hi; ++i) {
      const T xi = x[i];
      T *si = s + i * W;
      for (std::size_t w = 0; w < W; ++w) { // the lane loop: contiguous, must become ymm ops
        const T up = si[w];
        left[w] = std::min(std::min(diag[w], up), left[w]) + std::abs(xi - y[w]);
        diag[w] = up;
        si[w] = left[w];
      }
    }
  }
  for (std::size_t w = 0; w < W; ++w) out[w] = s[(n - 1) * W + w];
}
```

`lanes_block<T, W>` packs the W columns into a 64-byte-aligned `thread_local` `[L][W]` buffer (grows, never
shrinks) and calls it; that is the entry a fill would use, and what was timed.

## Commands

```sh
CXX="C:/Program Files/LLVM/bin/clang++.exe"
FLAGS="-IC:/D/git/dtw-cpp/dtwc/. -O3 -DNDEBUG -std=c++20 -march=native -fno-finite-math-only -fno-math-errno \
  -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -fopenmp"
# = the dtwc/core/dtw.cpp command minus -flto=thin and minus the dynamic-CRT trio (static CRT)
"$CXX" $FLAGS -fno-color-diagnostics pf5_lanes.cpp -o pf5_lanes.exe
"$CXX" $FLAGS -S -masm=intel pf5_lanes.cpp -o pf5_lanes.s
./pf5_lanes.exe verify > verify.txt                       # exit 0
sed 's/left\[w\] = std::min(std::min(diag\[w\], up), left\[w\]) + std::abs(xi - y\[w\]);/left[w] = std::min(diag[w], up) + std::abs(xi - y[w]);/' \
  pf5_lanes.cpp > mutant_drop_left.cpp                    # mutant: drop `left` from the lane min
"$CXX" $FLAGS mutant_drop_left.cpp -o mutant_drop_left.exe && ./mutant_drop_left.exe verify > mutant_verify.txt  # exit 1
cmd //c 'C:\D\git\wt\tmp\PF5\run_single.bat'              # start "" /b /wait /affinity 400000 pf5_lanes.exe single 15
./pf5_lanes.exe fill > fill.txt                           # no affinity: 24 OpenMP threads
uv run --no-project python vec_report.py dtwc/algorithms/fast_pam.cpp dtwc/algorithms/fast_clara.cpp
```

## Bit-identity [confirmed]

`verify`: 224 configurations — (f64 W = 4, f64 W = 8, f32 W = 8, f32 W = 16) × L ∈ {100, 500, 1000, 4000} ×
band ∈ {full, L/10} × data ∈ {N(0,1) noise, random walk}, plus edge cases L ∈ {1, 2, 3, 7, 33} × band ∈ {full,
0, 1, 2}. Every lane is bitwise equal to the scalar reference (max |diff| = 0), and the scalar reference is
bitwise equal to the library's `dtw_kernel_linear` / `dtw_kernel_banded`: `VERIFY PASS mismatches=0`. The
check bites: the mutant gives `VERIFY FAIL mismatches=925`. Inside the timing run the accumulated scalar and
lanes sums are also equal in all 32 configurations (`sums_equal=1`). The identity is by construction:
`std::min(a, b)` is `(b < a) ? b : a`, which `vminpd b, a` computes exactly, NaN included; nothing is reassociated.

## Single thread, CPU 22, 15 interleaved rounds (order alternated), median [min–max] [confirmed]

ns per DP cell (banded: cells inside the band); ratio = t_scalar / t_lanes paired per round.

| T, W | L | band | scalar | lanes | ratio |
| --- | --- | --- | --- | --- | --- |
| f64, 4 | 100 | full | 0.884 | 0.239 | **3.70** [3.64–3.74] |
| f64, 4 | 100 | 10 | 0.587 | 0.154 | **3.82** [3.73–3.86] |
| f64, 4 | 500 | full / 50 | 1.241 / 0.944 | 0.316 / 0.225 | 3.92 / 4.19 |
| f64, 4 | 1000 | full | 1.288 | 0.325 | **3.98** [3.89–4.15] |
| f64, 4 | 1000 | 100 | 1.142 | 0.279 | **4.09** [4.06–4.12] |
| f64, 4 | 4000 | full | 1.325 | 0.331 | **4.00** [2.50–4.16] |
| f64, 4 | 4000 | 400 | 1.296 | 0.322 | **4.01** [2.36–4.25] |
| f64, 8 | 100 | full | 0.883 | 0.150 | **5.86** [5.72–6.00] |
| f64, 8 | 100 | 10 | 0.586 | 0.113 | **5.17** [4.44–6.13] |
| f64, 8 | 500 | full / 50 | 1.253 / 0.944 | 0.168 / 0.145 | 7.50 / 6.52 |
| f64, 8 | 1000 | full | 1.287 | 0.167 | **7.73** [7.57–9.80] |
| f64, 8 | 1000 | 100 | 1.168 | 0.156 | **7.51** [7.37–8.94] |
| f64, 8 | 4000 | full | 1.324 | 0.167 | **7.93** [7.58–8.65] |
| f64, 8 | 4000 | 400 | 1.288 | 0.169 | **7.73** [7.41–9.13] |
| f32, 8 | 100 / 500 / 1000 / 4000 | full | 0.910 / 1.244 / 1.312 / 1.353 | 0.125 / 0.159 / 0.168 / 0.167 | 7.25 / 7.82 / 7.70 / 8.07 |
| f32, 8 | same | L/10 | 0.592 / 0.972 / 1.140 / 1.301 | 0.083 / 0.114 / 0.140 / 0.161 | 7.18 / 8.55 / 8.14 / 8.03 |
| f32, 16 | same | full | 0.885 / 1.251 / 1.289 / 1.355 | 0.080 / 0.083 / 0.085 / 0.084 | 11.04 / 15.00 / 15.27 / 16.21 |
| f32, 16 | same | L/10 | 0.586 / 0.945 / 1.148 / 1.294 | 0.063 / 0.074 / 0.079 / 0.083 | 9.28 / 12.77 / 14.50 / 15.56 |

The scalar reference reproduces the K1 record's 0.88 / 1.28 / 1.31 ns/cell at L = 100 / 1000 / 4000 within
1 % (0.884 / 1.288 / 1.325) [confirmed].
The two low minima at L = 4000 (2.50, 2.36) are single rounds taken while other processes used ~30 % of the
machine; the medians are unaffected.

**Verdict: PASS** — both W = 4 (lowest median 3.70) and W = 8 (lowest median 5.17) clear 1.5× at all six band
points; so does every single round (worst 2.36).

Interpretation [inferred, no PMU counters]: every kernel here is latency-bound on the loop-carried
`vminpd → vaddpd` chain (~1.3 ns per step; ≈ 7 cycles if the core runs near its 5.5 GHz boost, which was not
measured). The scalar kernel retires one cell per step,
the lanes W per step, so ns/cell halves exactly when W doubles at L ≥ 500 (f64 0.325 → 0.167; f32 0.168 → 0.085)
and the ratio tracks W. Short L runs faster for both because the out-of-order core overlaps adjacent columns.
Wider W (f64 W = 16, four ymm chains) may gain further until register pressure (16 ymm) bites — untested.

## Multi-thread fill check, advisory (N = 256 random walks, L = 1000, f64, full; 24 threads, dynamic rows)

| | round 0 / 1 / 2 (s) | median | Gcell/s | vs scalar |
| --- | --- | --- | --- | --- |
| scalar (K1 form) | 1.558 / 1.515 / 1.475 | 1.515 s | 21.5 | — |
| lanes W = 4 | 0.428 / 0.421 / 0.410 | 0.421 s | 77.5 | ×3.60 |
| lanes W = 8 | 0.317 / 0.321 / 0.305 | 0.317 s | 102.9 | ×4.78 |

[confirmed] Packed matrices bitwise identical (W = 4 and W = 8 against scalar). Lanes include the per-block pack
and a padded tail (the last block of a row repeats a column; its extra results are discarded). The ratio is below
the single-thread one, most likely because 16 of the 24 threads run on E-cores [inferred, not isolated]. For
scale only: the library's current fill measured 3.5 Gcell/s aggregate (`BM_fillDistanceMatrix/100/1000/-1`,
`2026-09-29-windows-kernel-msvc-stl-min.md`) before K1.

## Assembly of the lane kernel's inner loop [confirmed, `pf5_lanes.s`, `pf5_lanes_f64_w4`]

clang unrolls the w-loop, SLP-vectorises it and keeps `diag`, `left`, `y` in ymm registers (no spill in the
loop). One i-step of the 4× unrolled body (the three others are the same), and the loop control:

```asm
.LBB1_32:
    vmovupd      ymm5, ymmword ptr [r11 - 96]    ; up = s[i][0..3]
    ...
    vminpd       ymm9, ymm5, ymm6                ; min(diag, up)
    vminpd       ymm4, ymm4, ymm9                ; min(·, left)  -- ymm4 = left, carried
    vbroadcastsd ymm9, qword ptr [rcx + 8*rbp]   ; x[i]
    vsubpd       ymm9, ymm9, ymm3                ; x[i] - y[j][0..3]
    vbroadcastsd ymm10, qword ptr [rip + __real@7fffffffffffffff]
    vandpd       ymm9, ymm9, ymm10               ; |.|
    vaddpd       ymm4, ymm9, ymm4                ; left = min + cost
    vmovupd      ymmword ptr [r11 - 96], ymm4    ; s[i][0..3] = left
    ...                                           ; i+1, i+2, i+3
    add rbp, 4 / sub r11, -128 / cmp r8, rbp / jne .LBB1_32
```

W = 8 f64 (`pf5_lanes_f64_w8`) runs two ymm chains per step; f32 W = 8 / 16 are the same with
`vminps` / `vaddps` / `vbroadcastss`. The scalar reference's loop is the same sequence in `xmm` with
`vminsd` / `vaddsd`. One wart: the abs mask is re-broadcast inside the loop (rematerialised, not on the chain).

## Vectorisation survey [confirmed, clang remarks; `vec_report.py`, flags of the TU, `-flto` dropped]

| site | state | remark / blocker | hot? |
| --- | --- | --- | --- |
| `core/lower_bound_impl.hpp:182` `lb_keogh` | **vectorised** (f64 w4×i4, f32 w8×i4) | `#pragma omp simd` is compiled out (clang here defines `_MSC_VER`); `-fassociative-math` carries the reduction | per pair (TADPole) |
| `core/z_normalize.hpp:50, 63, 72` | **vectorised** (w4/w8 ×i4) | same pragma note | once per series |
| `core/lower_bound_impl.hpp:93, 94, 109, 110` `compute_envelopes` | not vectorised | `Cannot vectorize early exit loop` — Lemire monotone deque, sequential by nature; full band calls `__std_max_element_d` / `__std_min_element_d`; 4 heap allocations per call | once per series |
| `core/distance_matrix.hpp:81` Dense `max()`; `core/mmap_distance_matrix.hpp:735` Mmap `max()` | **not vectorised** | `value that could not be identified as reduction is used outside the loop` (NaN-skipping max without no-NaNs) | once per MIP solve |
| Dense `all_computed()` (`std::none_of`, MSVC `<algorithm>:1597`); `mmap_distance_matrix.hpp:756` | **not vectorised** | `Cannot vectorize potentially faulting early exit loop` | every `is_distance_matrix_filled()`, incl. once per Lloyd assignment (`Problem.cpp:1352`): an O(N²) scalar scan per iteration |
| Dense `count_computed()` (`std::count_if`, `<algorithm>:822`); `mmap_distance_matrix.hpp:747` | vectorised (w4×i4) | — | checkpoint |
| `algorithms/fast_pam.cpp:290` FasterPAM `find_best_swap` O(N) loop | **not vectorised** | `early exit loop with reductions or recurrences`, `instruction cannot be vectorized`, `:292 unsupported terminator`: an out-of-line `dist_by_ind` call per element (not inlined, X-04), the throwing finite check, and the conflicting scatter `ploss[nearest[o]] +=` | the swap phase: O(N²) per sweep |
| `algorithms/fast_pam.cpp:118` `compute_nearest_and_second` | not vectorised | `more than one early exit`, `:120 unsupported terminator` (call + throw per element) | O(N·k) per accepted swap |
| `algorithms/fast_pam.cpp:472` k = 1 path; `:589` seeding | not vectorised | call per element | O(N²) / O(N·k), rare |
| `algorithms/fast_clara.cpp:171` assignment | not vectorised | a DTW call per (point, medoid); the cost is the DTW itself — the fix is lanes over medoids, not this loop | O(N·k) DTW |
| `core/medoid_assignment_policy.hpp:109` ordered objective | not vectorised | ordered sum with finite check — deliberately sequential | O(N) |

Trap met while doing this: two `-Rpass=` flags do not add up — the last replaces the first, so
`-Rpass=loop-vectorize -Rpass=slp-vectorizer` silently prints no loop-vectorize successes. Use one regex,
`-Rpass=loop-vectorize|slp-vectorizer`. (`scripts/codegen_report.py` passes a single `-Rpass`, so it is unaffected.)

## What integrating lanes into the fill would take (not done)

1. `core/dtw_kernel.hpp`: one `dtw_kernel_lanes<T, W, Cell>` beside `_linear` / `_banded`, calling `cell.combine`
   per lane (clang vectorises the inlined scalar combine); needs K1's nested min; Cost/Cell stay templates (§7).
2. Scratch: two `thread_local` `[L][W]` buffers (pack, rolling row), 64-byte aligned, grow-never-shrink (§7).
3. Orientation (§7 `n_short ≤ n_long`): the W pairs share x as the inner axis, so the W y must share one length;
   unequal lengths group columns by length or keep the per-pair kernel.
4. `Problem.cpp::fillDistanceMatrix_BruteForce` + `core/dtw_dispatch.cpp::resolve_dtw_fn`: bind once, beside
   `dtw_fn_` / `dtw_fn_f32_`, a block function (x, W columns → W distances); the row loop steps j by W, skips
   fully `is_computed` blocks (checkpoint), pads the tail; one writer per slot, no lock (§7).
5. Tests: per-lane bitwise equality with the scalar kernel over (T, W, L, band); lanes fill == per-pair fill.

## Future scope, one line each

- **Unequal lengths:** group the W columns by length, or add per-lane end masks; orientation is per lane.
- **Missing data:** `AROWCell`'s `if (isnan(cost))` must become a select to vectorise; ZeroCost is a select on the
  cost; Interpolate is preprocessing and needs nothing.
- **WDTW:** the weight `w[|i − j|]` is the same for every lane: one broadcast multiply, vectorises as is.
- **ADTW:** the penalty add is lane-uniform; vectorises once `ADTWCell` uses the nested min (K1).
- **DDTW:** derivative preprocessing per series, then the standard kernel: as is.
- **Early abandon:** the per-lane row minimum becomes a vector min; a block can stop only when all W lanes
  exceed their thresholds (or lanes are repacked), so abandoning saves less as W grows.
