# Core compute layer review — 2026-09-02

Read-only audit of `dtwc/core/`, `dtwc/warping*.hpp`, `soft_dtw.hpp`, `distance.hpp`,
`parallelisation.hpp`, `detail/decode_pair.hpp`, `settings.hpp`, `types/`. Nothing modified,
built, or run. Four reviewers (this pass + three isolated subagents: lower bounds, mmap,
pruned/parallel).

---

## A. Bugs

**A1 [High, Observed] MV + `MissingStrategy::Interpolate` silently runs a *univariate* DTW over
the interleaved channel stream.** `dtw_dispatch.cpp:58-67` — `make_interpolate` has **no
`ndim > 1` branch**, unlike its siblings `make_zero_cost` (`:47`) and `make_arow` (`:82`):
```cpp
auto xi = has_missing(x) ? interpolate_linear(x) : std::vector<T>(x.begin(), x.end());
return normalize_public_distance(dtwBanded<T>(xi, yi, p.band));
```
`interpolate_linear` (`missing_utils.hpp:74`) is flat/univariate. With `ndim=2` and
`x = [1,10, NaN,20, 3,30]`, the NaN is filled by interpolating between `10` and `20` — values
from the *other* channel — and `p.band` then counts flat elements, not timesteps. Wrong number,
no diagnostic. `distance_semantics.hpp:84-91` rejects `ndim>1` only for MSM/TWE.

**A2 [High, Observed] MV + `SoftDTW` flattens the same way.** `dtw_dispatch.cpp:253-262` has no
MV branch and passes `a.size()` (= `ndim*steps`) as a series length. The comment at `:250-252`
calls this "flat-vector MV treatment preserved", but MSM/TWE *throw* for the same input
(`:281`, `:293`) — two variants reject MV, one returns a meaningless number. Smallest fix:
`throw InvalidInput` in `make_soft_dtw` plus a guard mirroring `distance_semantics.hpp:84-91`.

**A3 [High, Observed path / Inferred consequence] `interpolate_linear` can throw *inside* the
OpenMP region.** `missing_utils.hpp:90` `throw std::runtime_error("interpolate_linear: all
values are NaN")`, reached from the per-pair lambda at `dtw_dispatch.cpp:63`, invoked from
`run_openmp` (`parallelisation.hpp:89+`). LESSONS records that an exception in an OpenMP region
is UB — MSM/TWE were deliberately made to reject at *bind* time for exactly this reason.
Trigger: any dataset with one all-NaN series under `Interpolate`. Fix: extend the existing
serial pre-scan (`Problem.cpp:836-848`).

**A4 [Medium, Observed] `MissingStrategy::Error` is unenforced on the pairwise API.** The
"throw if NaN" contract (`dtw_options.hpp:54`) is implemented **only** in
`Problem::fill_distance_matrix` (`Problem.cpp:836-848`). `dtwc::distance::dtw(...,
MissingStrategy::Error)` (`distance.hpp:117`) and `core::dtw_runtime` (`dtw.cpp:49`) run the
recurrence on NaN and return NaN. Since NaN is *also* the "uncomputed" sentinel
(`distance_matrix.hpp:66`, assert-only under NDEBUG), such a result is indistinguishable from an
unfilled entry and `all_computed()` never becomes true.

**A5 [Medium, Observed] `lb_enhanced`/`lb_webb` return an *inadmissible* bound when `band < 0`.**
`lower_bound_impl.hpp:667` `const int w = std::min(std::max(band,0), ni-1);` collapses the
elastic arms to the diagonal cell, so `{(i,i)}` no longer cuts an unbanded path. Counterexample:
`A=[0,5,0,0]`, `B=[0,0,5,0]`, `band=-1`, `V=2` → LB = 10, true full L1 DTW = 0. Same clamp at
`:819-820`. Unreachable today only because both call sites gate on `band >= 0`
(`pruned_distance_matrix.cpp:135-137`). Fix: `if (band < 0) return T(0);`.

**A6 [Medium, Observed] WDTW weight-array length unvalidated at the public boundary.**
`dtw_cost.hpp:125-128` indexes `weights[|row-col|]`, max `max(nx,ny)-1`. None of the
weights-taking overloads (`warping_wdtw.hpp:67, 93, 212, 244`) checks `weights.size()`:
`wdtwBanded(x, y, span{w_of_size_2}, 5)` with `|x|=|y|=100` reads 98 elements out of bounds
(F46 applied to weights). Related: `wdtw_weights_cache_` is keyed on `max_dev` only, not `g`
(`Problem.cpp:283, 294`), while `variant_params` is a **public** member (`Problem.hpp:267`) —
writing `prob.variant_params.wdtw_g = 0.5` without `set_variant` leaves stale weights (F49).

**A7 [Medium, Observed] The pruned fill wipes a restored checkpoint.**
`pruned_distance_matrix.cpp:80` `dm.resize(N)` unconditionally re-fills with NaN
(`distance_matrix.hpp:53`); the brute path avoids this (`Problem.cpp:822, 777, 786`). `Auto`
resolves to `Pruned`, so restore-then-fill discards the checkpoint.

**A8 [Medium, Observed] `run()` permanently mutates process-wide OpenMP state.**
`parallelisation.hpp:171` `omp_set_num_threads(requestedThreads);` is never restored — a
k-means++ init at 32 workers pins every later fill to 32 threads, and
`pruned_distance_matrix.cpp:178-179` derives `block_count` from `get_max_threads()`, so
`PruningStats` depend on call order. Fix: a `num_threads(...)` clause on the pragma.

**A9 [Medium, Observed] `.dtws` store: `ndim` unvalidated, no crash-consistency.**
`mmap_data_store.hpp:234` accepts `ndim` unchecked → `Data.hpp:98` `series_flat_size(i) % ndim`
divides by zero (SIGFPE); CRC32 is unkeyed (`crc32.hpp:14`). `create()` (`:178-197`) has no
publication byte or barrier, so a kill mid-write leaves a file whose header CRC validates and
whose every series reports length 0 — silently empty. `MmapDistanceMatrix` already solves this
(`mmap_distance_matrix.hpp:630, 650-653`).

**A10 [Low, Inferred] `dtw_independent_mv` sums no-path sentinels.** `warping.hpp:626-628` —
two channels returning `numeric_limits<double>::max()` give `+inf`, not the documented `max()`
sentinel (`:593-595`).

**A11 [Low, Observed] Stale comment asserts the wrong FP model.** `dtw_kernel.hpp:340` says
"under -ffast-math (this build)"; `cmake/StandardProjectSettings.cmake:59-70` supplies an
explicit subset *without* `-ffinite-math-only`. The factor 16 slack is regression-tested, not
derived — the comment overstates it.

**A12 [Low, Observed] `z_normalize` uses an absolute stddev floor.** `z_normalize.hpp:70`
`if (stddev > 1e-10)` zeroes genuinely-varying series in small units (e.g. pA). Make it
relative, or document the unit assumption.

**A13 [Low, Observed] `Index` advertises a concept it does not model.** `types/Index.hpp:25`
declares `random_access_iterator_tag` but has no `operator++(int)`, `--(int)`, `+=`, `-=`, nor
free `n + Index`; `pointer = size_t` makes `it->` ill-formed.

Carried forward from subagents (their reports hold full traces): moved-from `MmapDataStore`
keeps dangling pointers (`:139-140`); `compute_file_size` overflow (`:76`); locale-dependent
`std::stod` (`matrix_io.hpp:145`) against a locale-free `to_chars` writer (`:83`); ragged CSV
silently truncated (`:152-155`); `#pragma omp simd reduction` compiled only off-MSVC
(`lower_bound_impl.hpp:186-191`) — harmless in the pruned fill, but decisive at TADPole's
`lb >= dc` test (`algorithms/tadpole.cpp:180-182`).

---

## B. Hot-path / lock-free findings

The kernels themselves are clean: `thread_local` scratch throughout, flat buffers, no locks, no
virtual dispatch, and the parallel fills write non-overlapping regions. The exceptions:

**H1 [Medium, Observed] `dtw_kernel_banded` does three O(n) setup passes per pair before the DP
starts.** `dtw_kernel.hpp:439-447`:
```cpp
col.assign(n_long, maxValue);
low_bounds.resize(n_short); high_bounds.resize(n_short);
for (row…) { auto [lo,hi] = dtw_band_bounds(band,row,n_long); low_bounds[row]=lo; … }
```
At `n=1000, band=5` the DP visits ~11k cells while setup costs 1000 (`assign`) + 2000 (bounds)
writes. For narrow bands the setup can rival the recurrence. Fix without touching the inner
loop: compute `lo`/`hi` inline per row (two comparisons — `dtw_band_bounds` is already
`noexcept` and branch-only), and clear only the band-adjacent `col` cells rather than all
`n_long`. Removes two `thread_local` vectors. Measure before landing.

**H2 [Low-Medium, Observed] A hash lookup per pair on the WDTW path.**
`dtw_dispatch.cpp:168, 188` `p.wdtw_weights_cache().find(max_dev)` — an `unordered_map` probe
per distance call. Read-only during the fill, so lock-free and race-free, but for equal-length
datasets (the common case) it resolves to one key every time. A cached
`{last_max_dev, const vector<data_t>*}` pair in the closure would remove it; the cache-miss
fallback at `:172, 191` additionally **heap-allocates a weight vector per pair**.

**H3 [Medium, Observed] `make_interpolate` heap-allocates two vectors per pair**
(`dtw_dispatch.cpp:63-64`) inside the parallel fill, against "buffer > thread_local >> heap".
`make_ddtw` (`:144`) already uses `thread_local` scratch — apply the same pattern.

**H4 [Low, Observed] Two `atomic_min_double` CAS loops run per pair even when the LB is
disabled.** `pruned_distance_matrix.cpp:431-432` with `use_lb` false at `:333` (SquaredL2) —
pure contended-atomic overhead for no benefit. Hoist under `if (use_lb)`.

**H5 [Low, Observed] `volatile double total_` in `OrderedMedoidObjective`**
(`medoid_assignment_policy.hpp:70, 116`) forces a memory round-trip per `add`, defeating
register accumulation. It is a deliberate determinism barrier against `-fassociative-math` —
keep it, but it belongs in a documented "determinism costs" list, not silently in an
accumulation loop.

**H6 [Medium, Observed] Global shared mutable RNG.** `settings.hpp:46`
`inline std::mt19937 randGenerator(29);` is consumed by `std::shuffle`
(`initialisation.cpp:140`) and distributions (`:172, :184`). Any concurrent use is a data race
and non-reproducible; `core/portable_random.hpp` exists precisely to replace it. Plus A8's
process-wide `omp_set_num_threads`.

**H7 [Accepted, Observed] `std::function` per pair.** `Problem::dtw_fn_` is
`std::function<double(span,span)>`, so every distance call is an indirect call through a type
erasure boundary. This is a *per-pair*, not per-cell, cost and `design.md:16` accepts it
explicitly. Flagged for completeness — do not "fix" it without a benchmark; the fix would mean
templating the fill on the variant and multiplying code size.

---

## C. Duplication

1. **Four cost functors exist twice; the `core::` copies are unused.** `core::L1Dist /
   SquaredL2Dist / MVL1Dist / MVSquaredL2Dist` (`dtw_cost.hpp:45-77`) are byte-identical to
   `dtwc::detail::L1Dist / …` (`warping.hpp:211-244`). Repo-wide grep: **zero** uses of the
   `core::` versions; the live dispatchers (`warping.hpp:262-289`) select the `detail::` ones.
   `dtw_cost.hpp:79-91` documents that a duplicate *dispatcher* here diverged and made the
   "L2 is Euclidean" fix inert — the duplicate *functors* were left behind.
2. **Five `detail::*_impl` shims repeat the same preamble** (empty check, `x==y` shortcut,
   orientation swap, cost lambda): `warping.hpp:61-74, 89-101, 118-130, 138-150, 167-179,
   187-199`. One `inline` `orient(...)` helper returning a small aggregate removes ~40 lines.
   **Constraint:** the `DistFn`/`Cost` parameters must stay **template** parameters — never
   `std::function` — or the cost functor stops inlining into the recurrence.
3. **Band-feasibility preamble in three places:** `warping.hpp:397-405`, `warping.hpp:551-558`,
   `dtw_kernel.hpp:426-436`. The kernel copy is authoritative.
4. **Three routes for unbanded Standard DTW:** `distance.hpp:38-42` → `dtwFull_L`;
   `dtw.cpp:100-103` → `dtwFull_L`; `dtw_dispatch.cpp:131-133` → `dtwFull_eap`. Claimed
   digit-identical, asserted only by `test_eap_dtw.cpp`.
5. Subagent-reported with both locations cited: Keogh accumulation ×4
   (`lower_bound_impl.hpp:195-202 / 561-569 / 517-527 / 600-611`), Lemire envelope ×4
   (`:97-127 / 465-494`), CSV emission ×4 (`matrix_io.hpp:105-120, 186-198, 206-217`,
   `Problem_IO.cpp:167-187`), pruned fill body ×2 (`pruned_distance_matrix.cpp:214-290` vs
   `:371-432` — already drifted; the standalone copy lacks Enhanced/Webb/ADTW), pair decode ×4
   (`pruned_distance_matrix.cpp:203-212` re-implements the *single-`if`* form that
   `detail/decode_pair.hpp:66-70` documents as broken and replaced with a `while`).

---

## D. Simplifications / error-prone constructs

- **Inconsistent lambda capture in the dispatcher.** `make_msm` (`:283`), `make_twe`
  (`:295-296`), `make_wdtw_f32` (`:207`) snapshot parameters **by value** at bind time;
  `make_adtw` (`:230`) and `make_wdtw_f64` (`:171`) read `p.variant_params` at call time.
  LESSONS names by-value capture as a stale-parameter hazard. Read-through `p` costs one
  dependent load per pair — negligible next to the DP.
- **Dead conditions** `else if (i + 1 >= 1)` / `else if (j + 1 >= 1)` (`soft_dtw.hpp:227, 239`)
  are always true for `int` loop variables.
- **`SoftCell` stores gamma twice** (`dtw_kernel.hpp:129-130` plus `scale.gamma`).
- **Asymmetric early-abandon.** `dtw_kernel_banded` tests only `col[0]` for the first row
  (`:460`) where `dtw_kernel_linear` computes an explicit `row_min` (`:273-284`). The banded
  form is correct *only* because row 0 is a monotone accumulation of non-negative costs — that
  precondition deserves a one-line comment, not a code change.
- **`cpp-style.md:32` ("No `std::min({a,b,c})`") is contradicted by the code**:
  `dtw_kernel.hpp:60, 77, 176`, `msm.hpp:94`, `twe.hpp:94`. Per LESSONS the nested-min win was
  kernel-specific. Fix the **doc**, not the code — rewriting the recurrence on a stale style
  rule risks a regression with no evidence behind it.
- **Magic numbers:** `inf * 0.5` as the abandon test (`pruned_distance_matrix.cpp:271, 409`)
  where the kernel's actual sentinel is `numeric_limits<double>::max()`; block multiplier `8`
  (`:179, 298, 435`); `V = 5` (`lower_bound_impl.hpp:661, 700, 713`); CRC offsets `60`/`28`
  (`mmap_distance_matrix.hpp:321`, `mmap_data_store.hpp:95`); `1e-10` (`z_normalize.hpp:70`).
- **Wide interfaces:** non-const `MmapDistanceMatrix::raw()` (`:747`) is documented to corrupt
  the cache; `lb_keogh_symmetric` (`lower_bound_impl.hpp:372`) silently truncates on unequal
  lengths; the `Envelope` aggregate (`:331`) lets `upper`/`lower` disagree in size while
  `lb_keogh`/`lb_enhanced`/`lb_webb` validate only `upper` (`:359, 703, 818`) before indexing
  `lower`, `ul`, `lu`.
- `run(task, n, 0)` means "parallel" (`parallelisation.hpp:162`); zero workers should be an
  error or serial.

---

## E. Dead / obsolete code

- `core::L1Dist`, `core::SquaredL2Dist`, `core::MVL1Dist`, `core::MVSquaredL2Dist`
  (`dtw_cost.hpp:45-77`) — zero call sites.
- `core::SpanSquaredL2Cost` (`dtw_cost.hpp:109`) and `core::SpanMVSquaredL2Cost` (`:148`) —
  zero call sites; SquaredL2 reaches the kernel through the `detail::` functors and a lambda.
- **`MetricType::L2` is unreachable from any string entry point.** `parse_metric_token`
  (`distance_semantics.hpp:22-30`) accepts only `l1`, `squared_euclidean`, `sqeuclidean` — so
  the one metric whose MV behaviour actually differs (`MVL2Dist`, Euclidean,
  `warping.hpp:249`) cannot be selected from the CLI or bindings.
- Zero-production-caller lower-bound entry points: `compute_envelopes_mv` (`:420`),
  `lb_keogh_mv` (`:511`), `lb_keogh_squared` (`:552`), `lb_keogh_mv_squared` (`:594`),
  `lb_kim(const T*,…)` (`:251`), `lb_kim(vector,vector)` (`:283`),
  `compute_envelopes(vector,…)` (`:143`). The `lb_keogh_valid`/`lb_kim_valid` matrix
  (`lower_bounds.hpp:25-54`) is consulted only by `static_assert`s in
  `tests/unit/core/unit_test_dtw_api.cpp:30-38`; `lb_kim_valid<SquaredL2Metric> = true`
  (`:51`) is unreferenced *and* an incorrect claim (F47).
- `LowerBoundStrategy::None` in the pruned fill (`pruned_distance_matrix.cpp:92-93`) is
  unreachable — `Problem.cpp:931-935` short-circuits it to BruteForce.
- `MmapDistanceMatrix::sync()` (`:753`) and `MmapDataStore::sync()` (`:306`) have no production
  caller; the v1/v2 migration strings (`:353-360`) cannot match any file this codebase writes.
- Unused includes: `parallelisation.hpp:23` `<iostream>` (its own comment at `:36` asks the file
  to stay light) and `:17` `types/Range.hpp`; `matrix_io.hpp:37` `<iostream>`;
  `portable_random.hpp:8` `<algorithm>`; `lower_bound_impl.hpp:30` `<type_traits>`;
  `pruned_distance_matrix.hpp:45-48`. `settings.hpp:20` pulls `<iostream>` into nearly every TU.
- No `#if 0` blocks and no stale TODO/FIXME comments found anywhere in scope.

---

## F. Missing / forgotten

- A1/A2 are the two "enum value with no handler" cases: `Dependent`+`Interpolate` and
  `Dependent`+`SoftDTW` reach a univariate implementation. **No test covers either** —
  `grep -n "Interpolate|SoftDTW"` over `unit_test_mv_*.cpp` and `unit_test_multivariate_dtw.cpp`
  returns nothing.
- **`compute_distance_matrix_pruned` (`pruned_distance_matrix.hpp:92-96`) takes no
  `LowerBoundStrategy`** and hardcodes Kim+Keogh (`:333, 345`), so `Enhanced`, `Webb`,
  `Kim`-only and `Keogh`-only — all documented in `LowerBoundStrategy.hpp:11-19` — are
  unreachable from the Python route this function serves. It also lacks the ADTW support the
  `Problem` path has (`:252-258`).
- **`Precision::Float32` + `StoragePolicy::Mmap` declared but unimplemented**
  (`storage.hpp:25-28` vs `DataLoader.hpp:171-175`).
- **`MmapDistanceMatrix` has no `write_csv`/`read_csv`/`to_full_matrix`** (`matrix_io.hpp:98,
  125, 163` are Dense-only); `Problem_IO.cpp:166` open-codes the gap, `:222` throws.
- **`Envelope`/`WebbEnvelope` carry no source length or resolved radius** (F46, acknowledged at
  `lower_bound_impl.hpp:329-330`), so no consumer can check the radius ≥ window precondition
  A5 depends on. Two fields (`int radius; size_t n;`) make A5 and the `.lower` gaps assertable
  at zero runtime cost.
- **`design.md:19` says Soft-DTW "lives in the same kernel architecture"** — true for the
  forward pass, but `soft_dtw_gradient` (`soft_dtw.hpp:171-256`) keeps its own forward
  recurrence and **two** `thread_local` full `mx × my` matrices. At the stated 8K-sample target
  that is ~512 MB each per thread; `msm.hpp:26-29` explicitly rejected that footprint. The
  `dtw_kernel_full` SoftDTW dispatch (`dtw_dispatch.cpp:260`) carries the same per-thread O(n·m)
  cost inside an N² fill.
- **`ScratchMatrix::resize` narrows to `int`** (`dtw_kernel.hpp:221`), capping the full-matrix
  kernel at `INT_MAX` cells with no diagnostic.

---

## G. Top 5 recommended actions

| # | Action | Size | Perf risk |
|---|--------|------|-----------|
| 1 | **A1 + A2** — add `ndim > 1` handling, or an explicit `InvalidInput` at bind time mirroring `make_msm`, for `Interpolate` and `SoftDTW`; matching guard in `validate_problem_distance_semantics`; two regression tests. The only class of finding here that returns a *wrong number silently*. | S | None — bind time, outside every loop |
| 2 | **A3 + A4** — move all-NaN rejection into the existing serial pre-scan (`Problem.cpp:836-848`) and enforce the `Error` NaN contract at `distance::dtw`/`dtw_runtime`. Removes an OpenMP-UB path and stops NaN colliding with the "uncomputed" sentinel. | S/M | **Negative cost** — deletes per-pair work |
| 3 | **C1 + E** — delete the four unused `core::*Dist` functors, the two unused `Span*SquaredL2Cost`, and the unused includes. Retires the second-dispatcher hazard `dtw_cost.hpp:79-91` warns about. | S | None |
| 4 | **A5 + A6 + `.lower` size gaps** — `if (band < 0) return T(0);` in `lb_enhanced`/`lb_webb`; one `sizes_ok(env, n)` helper for all three bound entry points; a `weights.size()` precondition on the four WDTW overloads. | S | None — all per-call (≤3 comparisons), never per-cell |
| 5 | **A7 + A8 + H4 + decode SSOT** — pruned fill honours existing entries; `num_threads(...)` clause instead of `omp_set_num_threads`; hoist the two `atomic_min_double` calls under `if (use_lb)`; call `dtwc::detail::decode_pair`. | M | **Reduces** atomics; decode adds one loop test per pair |

**Candidate 6 (measure first): H1** — remove the two `low_bounds`/`high_bounds` vectors and the
full-width `col.assign` from `dtw_kernel_banded` (`:439-447`). Plausibly material for narrow
bands, but it touches the one file where an unmeasured change is most dangerous. Benchmark
`BM_dtwBanded/1000/50` before and after; do not land on reasoning alone.

**Explicitly rejected as perf-risky.** Do **not** fold the `*_impl` shims (C2) or the four
Keogh loops (C5) behind `std::function` or any runtime callable — they must remain template
parameters so the cost functor inlines into the recurrence. Do not add a mutex anywhere in the
fill: the current decomposition is lock-free by construction and the only synchronisation
(`atomic_min_double`) should shrink, not grow. Do not "fix" H7 (`std::function` per pair)
without a benchmark — it is a per-pair indirect call that `design.md:16` accepts deliberately.
Do not rewrite the recurrences to satisfy the stale `std::min({a,b,c})` style rule (D).

**Not recommended:** restructuring the kernels. `dtw_kernel_banded`, `dtw_kernel_linear` and
`dtw_kernel_eap` were traced cell-by-cell for band bounds, rolling-buffer aliasing, sentinel
guards and the EAP window invariant; no defect was found, and the `Cost`/`Cell` policy split
does what `design.md:15` claims. The duplication findings sit in the wrappers, not the loops.

**Unknown:** whether the EAP relaxation factor 16 (`dtw_kernel.hpp:344`) is a bound or a fitted
constant (R2-D4 open); whether `dtwFull_eap` and `dtwFull_L` stay digit-identical outside the
cases in `test_eap_dtw.cpp`; whether `MetricType::L2` was intended to be token-selectable.
