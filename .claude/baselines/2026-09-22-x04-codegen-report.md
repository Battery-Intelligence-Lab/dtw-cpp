# X-04 — codegen report: what the compiler actually does to the DTW kernels

**Date:** 2026-09-22 · **Machine:** Mac (Apple M5 Pro, 18 cores, 64 GiB), AppleClang 21.0.0, arm64,
`clang-macos` preset, Release, `DTWC_FP_MODEL=fast` · **Rows:** X-04 (W0), feeds D-17 and A-06

Two tools, one question each. Both are report-only: they produce evidence, they do not decide.

## 1. Does IPO inline `Problem::dist_by_ind` into the SWAP kernels? (A-06 / core map Q10)

`scripts/check_ipo_inlining.py` disassembles the linked CLI, walks the caller blocks and counts
surviving calls. Registered as `check_ipo_inlining` (non-Windows; needs symbols and an objdump).

| build | callers | `dist_by_ind` calls |
| --- | --- | --- |
| `DTWC_ENABLE_IPO=ON` (ThinLTO, `-flto=thin` confirmed in 192 compile commands) | 23 | **18** |
| `DTWC_ENABLE_IPO=OFF` | 28 | **20** |

**Answer: IPO does not inline it.** ThinLTO removes 2 of 20 call sites and merges five caller
blocks; 18 survive, in the places that matter — `compute_nearest_and_second` (3), `tadpole` (3+2),
`assign_clusters` (2), `fast_pam_swap` (1), `fastpam1_swap_impl` (1), plus `silhouette`,
`calculate_medoids`, `kmeanspp` and `distanceInClusters`. A-06's "double preflight per pair" is
therefore paid at every call, LTO or not.

Two corrections made while producing this. The first reading was labelled `ipo=off` by guesswork;
the cache said `DTWC_ENABLE_IPO=ON`, so the label was wrong and the finding stronger than stated.
The second: the spec's caller regex matches bare `swap`, which also catches
`std::__function::__value_func::swap` — 2 of the original 25 "callers" were unrelated. The default
now also requires the project namespace, which is why the table says 23.

## 2. Which loops does the compiler vectorise? (X-04 proper)

`scripts/codegen_report.py` + `scripts/codegen_probe.cpp` + `tests/codegen_expectations.json`.

```sh
uv run --no-project python scripts/codegen_report.py            # check against the table
uv run --no-project python scripts/codegen_report.py --record   # re-seed it deliberately
```

**Result: none of the six kernel loops vectorise.** `vectorized=0 missed=6`.

| `dtwc/core/dtw_kernel.hpp` | why clang refused |
| --- | --- |
| 224, 227 | seed row and column — `loop not vectorized` |
| 232 | the main recurrence — `unsafe dependent memory operations` |
| 260, 266 | early-abandon variant — `Cannot vectorize early exit loop with writes to memory` |
| 267, 275 | `value that could not be identified as reduction is used outside the loop` |

This is not the compiler leaving easy wins on the table. Each cell reads `C(i-1,j-1)`, `C(i-1,j)`
and `C(i,j-1)`, so the recurrence carries a genuine dependency along both axes, and the
early-abandon variants add an early exit that writes memory. **For D-17 that means any SIMD gain has
to come from restructuring the traversal — an anti-diagonal or wavefront sweep — not from flags or
a library; and the early-abandon route would have to give up abandoning to get it.** Whether that
trade is worth making is a measurement question the PMU artefact (X-24, V-5) is meant to answer.

Drift detection proven by flipping one recorded entry to `true`:
`dtwc/core/dtw_kernel.hpp:224:3: vectorized -> not vectorized`, `drift=1 verdict=FAIL`; restoring
the table returns `drift=0 verdict=PASS`.

## Three things that made this report say nothing until they were fixed

Each produced an empty report, and in each case the script refused to print a pass — which is the
only reason they were caught rather than recorded as "no loops found".

1. **`dtwc/core/dtw.cpp` contains zero loops.** The kernels are function templates in
   `warping.hpp`; a template nobody instantiates generates no code, so no library translation unit
   reports on them. That is exactly why the row specifies a *probe TU*.
2. **An explicit instantiation with an internal-linkage template argument is dead code.** `AbsDiff`
   lives in an anonymous namespace, so the instantiations were internal, unreferenced, and dropped
   at `-O3` before the vectoriser ran — `nm` showed zero kernel symbols in a 35 KB object. The probe
   now exports `extern "C"` wrappers, which cannot be eliminated.
3. **`-Rpass` reports nothing under ThinLTO.** With `-flto=thin` the `-c` step emits bitcode and the
   optimisation passes run at *link* time, so there is nothing for the remark flags to report. The
   probe compile strips `-flto*`. Consequence for how to read this report: it describes per-TU
   codegen of these kernels under the project's real flags, not the final post-link code of an LTO
   build.

A fourth, smaller: `StandardProjectSettings.cmake` adds `-fcolor-diagnostics`, which wraps every
remark in ANSI escapes; the probe compile appends `-fno-color-diagnostics`.

## Scope of these numbers

AppleClang 21 on arm64 only. `tests/codegen_expectations.json` is therefore compiler- and
architecture-specific in the same way `tests/floors.cmake` is platform-specific. GCC does not
implement `-Rpass` at all (it uses `-fopt-info-vec`), so the report is clang-only today. That is why
`codegen_report.py` is **not** registered as a CTest gate: a gate that cannot run on half the
supported toolchains, pinned to a table recorded on one machine, is the unowned-red-gate shape that
X-33 had just finished cleaning up. It is a script you run, with its output recorded here. `check_ipo_inlining`
*is* registered, because a disassembler exists everywhere except Windows and its marker is
compiler-agnostic.
