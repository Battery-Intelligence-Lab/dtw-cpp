# Adversarial review — batch 1 (uncommitted diff, 2026-09-02)

Read-only. No build, no tests run. All line numbers are the NEW tree.
**Observed** = read in the diff/tree. **Inferred** = reasoned from observed code.

## Defects, ranked

### 1. (High) GIL released on Python methods that mutate the `Problem`
`python/src/_dtwcpp_core.cpp:930` (`dist_by_ind`), `:1019` (`find_total_cost`),
`:999` (`read_distance_matrix`), `:1035` (`write_clusters`), `:1042`
(`write_silhouettes`).
**Observed:** each now wraps the call in `nb::gil_scoped_release`.
`Problem::dist_by_ind` runs `ensure_dense_cache_configuration_current_preflighted()`
and writes `distMat` (`Problem.cpp:683-732`); `write_clusters` calls
`find_total_cost()` (`Problem_IO.cpp:125`), which reaches the same lazy path;
`read_distance_matrix` replaces the whole matrix.
**Inferred:** before this change the GIL serialised these; now two Python threads
sharing one `Problem` race on `distMat`/`dtw_fn_` — silent corruption, not an
exception. Only `dist_by_ind`/`find_total_cost` carry the warning docstring;
`write_clusters` and `read_distance_matrix` carry none, and the CHANGELOG's
threading caveat names only the first two.
**Fix:** either keep the GIL for the mutating entry points, or extend the
warning to `write_clusters`/`read_distance_matrix` and say plainly that these
are not thread-safe.

### 2. (High) New `Data::series()` precision throw is reachable on a default route
`dtwc/Data.hpp:55`; `dtwc/Problem.cpp:743-754`;
`dtwc/core/pruned_distance_matrix.cpp:130,147,235,265-269`.
**Observed:** `pruned_strategy_applicable()` gates on variant, missing strategy,
dense storage, `band >= 0`, `size() >= 64` — **not** on precision.
`fill_distance_matrix_pruned` calls `prob.series(i)` unconditionally.
**Inferred:** Float32 data + Standard/ADTW + `band >= 0` + N ≥ 64 + dense now
resolves `Auto → Pruned` and throws `Data::series: data is stored as Float32`
from inside `run_openmp`; an explicit `DistanceMatrixStrategy::Pruned` bypasses
`pruned_strategy_applicable` entirely (`Problem.cpp:881`) and so throws at any
band. (`settings::DEFAULT_BAND == -1`, so plain `Auto` + f32 is unaffected.)
The old behaviour was UB (`p_vec[i]` on an empty vector), so the guard is right
— but nothing routes f32 away from Pruned, and the CHANGELOG presents A12 as a
pure hardening bullet.
**Fix:** `&& !prob.data().is_f32()` in `pruned_strategy_applicable`.

### 3. (Medium) Checkpoint metric fingerprint fix does not reach Python
`python/src/_dtwcpp_core.cpp:1274` and `:1284` call
`dtwc::save_checkpoint(prob, path)` / `load_checkpoint(prob, path)` — i.e. the
`core::MetricType::L1` default (`checkpoint.hpp:64,82`).
**Observed:** only `dtwc_cl.cpp:1676,1914` pass `cache_metric`.
**Inferred:** every Python user keeps the A5 bug — a SquaredL2 matrix is still
accepted by a later L1 run. `Problem` stores no metric member, so the default
cannot be made correct implicitly.
**Fix:** add an optional `metric` argument to both bindings.

### 4. (Medium) `decode_pair` down-correction is paid per pair on the device
`dtwc/detail/decode_pair.hpp:61,69`.
**Observed:** the function is `DTWC_DECODE_PAIR_HD` and is the on-device pair
decoder (`cuda/cuda_dtw.cu:1471` comment: "pairs are decoded on-device via
`decode_pair()`"). The new `while (row > 0 && row * (2*N - row - 1) / 2 > k) --row;`
adds a 64-bit multiply **and divide** per pair; the header itself states an
overestimate "is not constructible here today", so the cost is unconditional and
the benefit hypothetical. `row_start` is then recomputed with the same expression.
Separately, `if (row > N - 2) row = N - 2;` runs *after* `if (row < 0) row = 0;`,
so for `N < 2` `row` ends up negative — the pre-existing low clamp is weakened.
**Fix:** hoist `row_start` and test it in the loop; clamp low last.

### 5. (Medium) `test_io_readers` gate pins an assertion/case count band
`tests/CMakeLists.txt:626-636`: `PASS_REGULAR_EXPRESSION` requires
"All tests passed (300–9999 assertions in 9–99 test cases)".
**Inferred:** this pins implementation, not behaviour. A legitimate 100th test
case, or a refactor that merges assertions below 300, fails the gate for a
non-defect; conversely any count inside the band passes even if the intended
subject was deleted. The `FAIL_REGULAR_EXPRESSION` skip check is the part that
actually enforces F9.
**Fix:** keep the skip rejection; drop the upper bounds (or assert a named
`[arrow]` tag ran) rather than a count window.

### 6. (Low–Medium) `query_gpu_config` can hand out an unpublished slot
`dtwc/cuda/gpu_config.cuh:66-67`: on `cudaGetDeviceProperties` failure it
returns `configs[device_id]` — a reference to the still-unpublished cache slot —
and callers hold it (`cuda_dtw.cu:1254`, `const auto &gpu_cfg = ...`).
**Inferred:** if a later call on another thread succeeds and fills that slot
under the lock, the earlier holder reads `device_name` (a `std::string`) while it
is being written: a real data race. The docstring's "exactly one thread ever
writes a slot" is true but does not cover readers of a failed query.
**Fix:** return a `static const GPUConfig kUnavailable{}` on the failure path.

### 7. (Low) Per-thread exception slots are sized outside the region and unchecked
`python/src/_dtwcpp_core.cpp:1082-1089` sizes `errors` from
`omp_get_max_threads()` captured before the release/region; `:1103` indexes it
with a bare `omp_get_thread_num()`.
**Inferred:** safe today (the diff removed the only in-tree `omp_set_num_threads`
call, in `parallelisation.hpp:176-193`), but a user or third-party library
raising the limit between the two points, or a nested team, writes out of bounds.
**Fix:** add `num_threads(n_error_slots)` to the pragma.

### 8. (Low) mmap CSV write now scans the matrix for non-finites twice
`dtwc/Problem_IO.cpp:186` hoists `preflight_distance_matrix_csv(m)`, and
`core/matrix_io.hpp:204` (`operator<<`) runs it again — an extra O(N²) pass on
exactly the matrices large enough to be mmapped. The code comment is honest; the
CHANGELOG bullet says only "bytes unchanged".

### 9. (Low) Temp-path fix reduces, does not remove, collision
`dtwc/DataLoader.hpp:128-155` builds a name from `random_device` ⊕ clock ⊕ an
atomic counter and returns it; the file is never created exclusively.
CHANGELOG says "Fixed the mapped-series temp path" — it is now improbable, not
impossible.

### 10. (Low) Pruned-checkpoint test does not exercise the scenario it names
`tests/unit/core/unit_test_pruned_distance_matrix.cpp` calls
`fill_distance_matrix_pruned(prob, -1, Auto)` with N=4. `pruned_strategy_applicable`
requires `band >= 0` and `size() >= 64`, so `Auto` would never select Pruned for
this input. The test does pin the function contract (and does fail without the
fix, since the old unconditional `dm.resize` NaN-fills the sentinel) — but the
CHANGELOG's "(Auto resolves to Pruned for Standard DTW)" is unproven by it.

## Confirmed sound (traced, no defect found)

- `lb_enhanced`/`lb_webb` `band < 0 → 0`: the counterexample is correct
  (unbanded DTW of A=[0,5,0,0], B=[0,0,5,0] is 0 via (0,0)→(0,1)→(1,2)→(2,3)→(3,3)).
  No production perf loss: both pruned fills already gate envelope bounds on
  `band >= 0` (`pruned_distance_matrix.cpp:141,355`). Tests are non-vacuous.
- `envelope_sizes_ok` / `lb_keogh` returning 0 on a size mismatch: the pruned
  route computes envelopes from the same series and gates on `equal_len`
  (`pruned_distance_matrix.cpp:235`), so no false positive and no lost pruning.
- `require_wdtw_weight_span`: every in-tree caller sizes weights as
  `max_dev + 1 = max(nx, ny)` (`warping_wdtw.hpp:57,169,187`;
  `dtw_dispatch.cpp:174-201`), so the bound is exactly met — no false positive,
  and the band-sized-table concern does not arise.
- `run_openmp` `num_threads(...)` + `omp_chunk_size_for`: no process-wide state
  left behind; `run()`'s serial (`numMaxParallelWorkers == 1`) and unlimited
  (`0`) cases behave as before. `unit_test_run_thread_scope.cpp` fails against
  the old code on `omp_get_max_threads() == before`.
- `require_input_format_is_built` sits outside `#ifdef DTWC_HAS_PARQUET/ARROW`
  and classifies by filesystem only (`dtwc_cl.cpp:1363-1375`) — correct per F9.
- `validate_cli_route_selectors` whitelist is exactly the `--method`
  CheckedTransformer map (`dtwc_cl.cpp:830-839`) after alias folding, and the
  dispatch chain covers all eight non-`auto` methods with `auto` resolved first
  at `:1555`. The terminal `else throw` is unreachable for valid input.
- `read_distance_matrix` now throwing: the CLI already wraps it
  (`dtwc_cl.cpp:1663-1672`), so the CHANGELOG claim holds.
- CSV write delegation to `operator<<` is byte-identical (same
  `distance_matrix_csv_token`, same `,`/`\n` emission — `matrix_io.hpp:202-218`).
- `adopt_as_ndarray`: no leak window (capsule constructed while `unique_ptr`
  owns), no nested `gil_scoped_acquire`, `reserve(1)` gives a non-null pointer
  for the empty case; `nb::ndarray` parameter outlives the release scope in
  `pdlp_lp_bound`.
- Parquet extractor template: `raw_values()` already includes the array offset,
  list `value_offset` indexes the un-sliced child, and the f32/f64 destinations
  produce the same values as the six removed bodies. Null rejection is one
  `null_count()` per array, never per element.
- CUDA `require_pair_count_fits` / `require_cuda_device` / `wavefront_buffer_count` /
  `scan_series_lengths` (incl. the `initial_max = query.size()` overload) are
  behaviour-preserving; `require_shared_mem_fits` reads the **opt-in** cap
  (`gpu_config.cuh:80-85`), so it does not false-positive on legal >48 KB launches.
- `compute_distance_matrix_pruned`'s `if (use_lb)` around the `nn_dist` update is
  behaviour-neutral: `threshold` is read only inside `if (use_lb && ...)`
  (`pruned_distance_matrix.cpp:409`).
- `validate_ndata` / `ndata_wants_more` / `sorted_directory_files`: consistent
  across `count()`, folder, batch and metadata loads; `Ndata == -1` (the default)
  is unaffected.
- Removed public API (`readCSV`, `readTimeSeriesCSV`, `readCSVColumn`,
  `ParquetChunkReader::estimated_total_bytes/estimated_bytes_per_series/read_row_group`,
  `core::SpanSquaredL2Cost` and siblings) has zero remaining callers repo-wide.
- `dtw_runtime`/`distance::dtw` NaN pre-scan is at the public boundary only —
  neither is used by the per-pair matrix fill (`dtw_fn_` is), so the O(n+m) scan
  is not a hot-path cost.
- `settings.hpp` / `parallelisation.hpp` keeping `<iostream>` / `types/Range.hpp`
  with an explicit reason: honest, not a defect.
