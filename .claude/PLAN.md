# DTWC++ 2.0 — PLAN

Branch `design-2.0`. **Rewritten 2026-09-23 after a YAGNI pass** (`CHARTER.md`, entry of that date); the
previous plan is `git show e784e5c:.claude/PLAN.md`. Ledger IDs (`C-`, `O-`, `A-`, `B-`, `T-`, `G-`) are rows of
`specs/2026-09-07-diff-ledger.md`, now a frozen reference rather than the work list; `S-`, `X-`, `D-`, `V-`
IDs are the previous plan's. **A ledger row not named below is dropped**, unless someone touching its file
finds it silently wrong — then it comes back as an `FX` item.

## 0. The test every item passes

An item is here only if it does at least one of:

- **K1 interface** — keeps the user-facing surface stable, and the same in C++, Python and MATLAB, including
  `device = cpu | gpu | hpc`;
- **K2 speed** — SIMD, memory layout and allocation, GPU / HPC fit, dispatch kept out of hot loops;
- **K3 correct** — fixes an answer that is silently wrong, or behaviour that is unsound, on input a user can give;
- **K4 helper** — deletes at least two real copies.

Two rules decide *how*: **a mature, portable, permissively licensed library beats in-house code** (MIT, BSD,
Apache-2.0, BSL-1.0, zlib — each addition names the code it deletes), and **types beat checks** (§1.1).

**Who does what.** The main session designs, dispatches and reviews. One implementer (a subagent) per item,
never two on the same files: a test that pins the item's contract against an independent oracle → the change →
the §2 gates → one commit. An item's record is its status and commit hash on its line here; `baselines/` gets
a file only for a performance claim.

## 1. Where we are (2026-09-23; updated 2026-09-24)

- 2026-09-24: Volkan committed `899bb65` ("test improvement", 302 files) — the GT, FX, IF-1 and IF-2 S1 / S2 work of 09-23 and 09-24.
  On top, uncommitted: IF-2 S3. Tests on this Mac: ctest 142 / 142 (2 CUDA skips), pytest 1,245 / 0, `matlab_suite` green.

- Released: **v1.0.0**. `VERSION` = `2.0.0rc1`, untagged. Users are on the 1.x shape.
- 2026-09-22, eleven commits (`9c08074..e784e5c`): two release blockers fixed — CLI archives had no rpath
  (X-29) and were compiled `-march=native` (X-31); FP relaxations no longer leak into Catch2 / HiGHS / llfio,
  and `DTWC_FP_MODEL=strict|fast` exists (S-03, X-15); the conformance gate no longer passes when its
  reference file is missing (X-32); two red gates repaired (X-30, X-33); third-party notices match what ships
  (X-23); Eigen removed (X-27; its recorded ~5 % slowdown did not reproduce under interleaved A/B — PF-6); foundation headers moved to
  `dtwc/base/` (C-11, X-05, X-12) and the MIP solver helpers to `dtwc/mip/` (C-12); the codegen report exists (X-04: none of the six DTW kernel loops
  vectorise — each cell reads the one written just before it); PMU option (X-24), machine facts (X-26),
  typed-throw count (X-16), maintainer options renamed `DTWC_*` (B-14).
- macOS: 128 of 133 tests passed at the start of 2026-09-23 (the five failures were Windows-measured assertion
  floors); after GT-1 / GT-3, 129 of 131 pass with 2 CUDA `MAY_SKIP` skips.
- 2026-09-23: an uncommitted load-time series-count guard (A-10) was reverted, and §5 lists what else was dropped.

### 1.1 Integers (decided 2026-09-23)

Counts of series and cluster labels are **`int`** — the v1.0.0 surface (`clusters_ind`, `centroids_ind`),
MATLAB `int32`, numpy `int`. **No runtime guard for a count, anywhere**: 2³¹ series is out of reach, and whoever
approaches it knows what to do. **Products of counts** — pair index, packed offset, N·L, N², cells, bytes — are
`size_t` / `int64_t` **by type**, which already holds at the sites opened (`tri_index`, `decode_pair`,
`upper_triangle_pairs`, Metal's `pair_offset`, `.dtws` offsets, `ScratchMatrix::Index`). Three known products
are still narrowed, and each is fixed by type, not guarded: MPI's per-rank loop bound
`static_cast<int>(local_count)` (`mpi_distance_matrix.cpp:131, 134` — above `INT_MAX` pairs on a rank it
silently computes nothing; PF-4), CUDA's per-launch pair count (PF-4), and CLARANS's neighbour count
`static_cast<int>(0.0125·k·(N−k))` (`clarans.cpp:62`; FX-3). A check stays only where
a third-party API forces a narrower type than ours — `mip/index_guard.hpp`, because HiGHS and Gurobi index
columns with 32 bits and N² passes that at N = 46,341 (a truncated model would be silently wrong). Where we own
the loop, we chunk instead of refusing (PF-4).

## 2. Build and gates

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp && cmake --build --preset clang-macos
ctest --test-dir build -C Release -j1 --output-on-failure      # serial is the evidence run
python3 scripts/check_docs_contract.py && python3 scripts/check_supply_chain_pins.py
```

A gate exists for five things only: a no-op's conformance output is digit-identical on one machine; a test
cannot pass by skipping; the docs name only what the code has (`check_docs_contract.py`); every fetched
dependency is pinned (`check_supply_chain_pins.py`); the tracked tree holds no secrets, banned paths or stray
artefacts (`check_repo_hygiene.py` — it scans for credentials, which matters with `.env` in play).

### 2.1 What a machine needs

| Need | macOS (this machine) | Linux | Windows | Required? |
| --- | --- | --- | --- | --- |
| C++20 compiler, CMake ≥ 3.26, Ninja | Apple clang 21 + Homebrew | GCC 11/12, Clang 14–17 | MSVC 19.3x, clang-cl | yes |
| OpenMP | `brew install libomp`, `-DOpenMP_ROOT=/opt/homebrew/opt/libomp` | bundled | `/openmp:experimental` | yes, unless `-DDTWC_ALLOW_SEQUENTIAL=ON` |
| Python via `uv` | present | — | — | tests, scripts, wheels |
| CUDA | unavailable | CI / RTX box / ARC | RTX box | optional — CUDA items are proven on the RTX box (§4) |
| MATLAB | R2026a at `/Applications/MATLAB_R2026a.app` (not on `PATH`; found 2026-09-24 — the MEX builds with `-DDTWC_BUILD_MATLAB=ON`) | CI | CI + Windows box | optional |
| MPI, Gurobi, Arrow, Hugo | not installed | CI | CI | optional |

## 3. Work, in order

★ = before the 2.0 tag. Each line: **ID** — what — (rows it absorbs).

### 3.1 Gates that tell the truth — first, because a red or vacuous gate proves nothing

- ☑ ★ **GT-1** *(done 2026-09-23, uncommitted: 129 / 131 pass, 2 CUDA skips; five harness mutations — a failure, a partial
  skip, a full skip, zero cases, zero assertions — each rejected. Left: the same rule for
  `.github/scripts/assert-arrow-suite.sh`'s 348-assertion floor (the MATLAB floor went with IF-2 S3, 2026-09-24), on
  machines with Arrow / MATLAB; the stress script's `bc … || echo 1` passes when `bc` is missing)* A test passes when it ran at least one case, failed none and did not skip; a skip fails unless
  the test is registered `MAY_SKIP`, which prints why. The per-test assertion floors, their numbers in
  `tests/floors.cmake` and the four-platform re-measure go. Fixes the five macOS failures. `tests/integration/stress_test_cli.sh` records skips
  (`:34`) and still prints "All tests passed." — it fails on a skip too (T-20). (W0-R1, W0-R3, V-6, T-16's
  floor half)
- ☑ ★ **GT-2** *(done 2026-09-23, uncommitted: `check_docs_contract.py` 2,391 → 1,854 lines, `check_record_hygiene.py`
  and the 223 KB plan archive deleted, CHANGELOG's 127 KB development history cut; three kept checks shown to
  bite by mutation)* `check_docs_contract.py` keeps the checks that compare user docs with code (Tier-1 signatures,
  method catalogue, CLI reference) and loses every pin on `LESSONS.md`, `CHANGELOG.md`, handoffs, plan
  archives, baselines, reports and line numbers; `check_record_hygiene.py` is deleted. Records are then
  compacted by plain deletion. (W0-R2, H-1…H-4, D-4's gate edit)
- ☑ ★ **GT-3** *(done 2026-09-23, uncommitted: 4,369 lines deleted; `test_deprecated_shims_warn` compiles seven sampled shims
  and fails when one stops warning; the GCC / MSVC branches are unrun here. Left: `unit_test_deterministic_series.cpp:300-316`
  still counts calls in five benchmark files — same smell, same fix)* The F22 apparatus goes — three mutation scripts, the launcher, the probe targets, 3,804 lines —
  and one compile probe proves a deprecated shim still compiles and warns (B-19, F22). T-01's assertion of call
  counts *inside five sibling test files* is deleted.
- ☑ **GT-7** *(done 2026-09-23: 0 warnings, `__text` byte-identical, 131 / 131)* `cmake/StandardProjectSettings.cmake:100` passes GCC's `-fno-signaling-nans` to Clang, which ignores it with one
  warning per translation unit (124 in a full build), burying real warnings. Guard it by compiler; codegen is unchanged.
- ◐ ★ **GT-4** *(GT-4b done 2026-09-24, merged, uncommitted: checkpoint-path and llfio failures raise `IOError`, a cache for other data and a
  too-wide file `skip_cols` raise `InvalidInput`, the unreachable timestamp / atomic checks are `logic_error`, PDLP `use_gpu` without the backend
  raises `DeviceError`; 140 / 140, pytest 1,236 / 0. Left: `Problem.cpp` ~796 `fs::exists` and `cli/config_file.hpp` ~144 (with S3), the §5 table
  text (with S3's contract pass), `fileOperations.hpp` ~399's `directory_iterator` (dangling symlinks need care), stale PDLP "warns" text in
  `bindings/matlab/+dtwc/pdlp_lp_bound.m` ~22, `docs/content/math/lr-core.md` ~261, `docs/sources/lr-core-outcome.md` ~49, `dtwc/mip/CMakeLists.txt`
  ~69 and the pdlp docstring in `_dtwcpp_core.cpp`; [inferred] HiGHS 1.15.1 may report `gpu_used = false` for a GPU PDLP solve — check on a
  CUDA machine. GT-4b's list, from a second review of 2026-09-24 — 26 / 26 new `logic_error` sites unreachable, catch sites clean, but:
  `save_checkpoint` (`checkpoint.cpp` ~497, 500, 574), `Problem.cpp` ~756 and the mmap creators (`mmap_distance_matrix.hpp` ~624, `mmap_data_store.hpp`
  ~171, llfio `.value()`) still leak `filesystem_error` / llfio errors — Python `RuntimeError`, MATLAB `dtwc:runtime` → error-code overloads and
  `IOError`; a cache or checkpoint made for other data is `IOError` (`mmap_distance_matrix.hpp` ~511) while a wrong-size CSV matrix is
  `InvalidInput` → both `InvalidInput`, and §5's table row "checkpoint mismatch" moves; `skip_cols` wider than a row is `IOError` from a path
  (`fileOperations.hpp` ~257, 283) but `InvalidInput` in memory and in §1.2 → `InvalidInput`; the timestamp and lock-free-atomic checks
  (`checkpoint.cpp` ~90-101, `mmap_distance_matrix.hpp` ~274, 642) → `std::logic_error`; a GPU PDLP request on a build without CUPDLP_GPU
  warns and runs on the CPU (`mip/pdlp_lp.cpp` ~177-180) → a typed error; a YAML config on a build without YAML raises `CLI::ConfigError`
  (`cli/config_file.hpp` ~144) → `IOError`. Kept: a feature this build lacks takes its subsystem's
  type — `IOError` for a format, as `DeviceError` / `SolverError` for a GPU / Gurobi — the review suggested `InvalidInput`. First pass done
  2026-09-24, merged, uncommitted: of 245 bare `throw std::` sites in `dtwc/`, 190 now raise their §5 type (66
  `InvalidInput`, 106 `IOError`, 18 `DeviceError`) and 55 are `std::logic_error` for programming errors (26 newly, each with a reason);
  `check_typed_throws.py`, its CTest registration and the tests' Python requirement deleted; `test_error_taxonomy` reaches one live site per
  converted file (all 17 untyped on the base); a direct Benders call can no longer terminate; the contract's §5 states the file rule.
  CUDA and the Win32 branch unbuilt, Arrow syntax-checked. First half, 2026-09-23: manifest count gone, `check_ipo_inlining` unregistered)* `check_supply_chain_pins.py` keeps the SHA pins and drops `REGISTERED_CMAKE_MANIFEST_TOTAL`;
  `check_ipo_inlining` leaves CTest (its pass pattern accepts any count) and stays a manual tool beside the
  codegen report; one sweep turns the user-input `throw std::…` sites into typed errors, so Python sees
  `ValueError` rather than `RuntimeError`, and `check_typed_throws.py` is deleted (X-16).
- ☐ ★ **GT-5** *(2026-09-24: `.github/workflows/matlab-mex.yml` puts `build/mex-ci/bin` on MATLAB's path and uploads from it, but on macOS
  and Linux the MEX lands in `build/mex-ci/bindings/matlab/` [inferred — check on the first CI run])* CI on this branch — a draft PR to `main`, for CI only (W0-R5, D-8) — plus one leg that builds
  with every optional dependency OFF (B-15, first clause only).
- ☐ **GT-6** Five canonical benchmarks recorded once, with algorithmic counters (pairs, cells, swaps), before
  any PF item: dense N = 1000 × 512; banded 10 %; PAM N = 2000; OneBatchPAM 50k; barycenter k-means k = 3 (R5).

### 3.2 Correctness — answers that are silently wrong today

- ◐ ★ **FX-1** *(2026-09-24, merged, uncommitted: `Problem::validate_fill_request` runs once per fill and where the lazy
  `dist_by_ind` path starts; every GPU axis — variant, missing strategy, ndim, Float32 / view / mmap series, Metal FP64 and
  Metal index ≠ 0 — raises `DeviceError` (Metal WDTW had returned standard DTW, 7 against 3.28), a band narrower than longest −
  shortest raises `InvalidInput`, a squared-L2 cache now fills on the GPU (it was refused, not computed as L1); `test_fill_request`,
  `test_problem_set_device`, pytest `TestProblemDevice` fail on the base; 133 / 133. Three older tests had compared matrices whose
  unequal-length pairs were all 1.8e308 — now cut to a common length. Then the `dtw_function()` accessors validate once too, so
  OneBatchPAM and FastCLARA's assignment reject an infeasible band, and a complete dense matrix skips the check (nothing is computed).
  Left: the CLI → IF-2; Python Tier-1 → IF-3; Parquet-chunked FastCLARA, whose template holds no series, leaves chunks unchecked (Arrow,
  blind); CUDA blind → V-11)* One `validate_fill_request()` before every fill: GPU with a variant, missing-data strategy or
  `ndim` it does not implement ⇒ typed error naming the axis (G-04); a band narrower than a pair's length
  difference ⇒ typed error naming the pair and the smallest feasible band, widening only as an explicit token
  (S-15, D-12); GPU with an mmap series store ⇒ typed error; a precision the device cannot honour ⇒ typed error — Metal
  runs FP32 when FP64 is asked and says so only under `verbose` (`metal_dtw.mm:1263, 1994`), and `Problem`'s
  Metal route passes no precision at all (`Problem.cpp:1079-1082`) (F30). (X-02)
- ☐ ★ **FX-2** Soft-DTW self-distance: two duplicate series give `d(i,j) < d(i,i)` today. Derivation D7 decides
  between the Soft-DTW divergence (recommended) and the true diagonal; both land together (S-01, D-5).
- ☑ ★ **FX-3** *(done 2026-09-24, merged, uncommitted: O-06, the `--dist-matrix` size check, `[[nodiscard]]` with `--solver gurobi` on a
  Gurobi-less build now an error, and the Tier-1 / `Problem` writers checked after close landed with the `Problem` agent; `test_cli_loud_failures`
  15 / 15. Earlier the same day: S-04, B-05's CLI half, A-05, A-07 and S-10 — `test_cli_loud_failures`
  drives the real binary through six failures, EFBIG included; 132 / 132. Left for the `Problem` / `api` item: O-06, the
  `--dist-matrix` N mismatch, the Tier-1 writers, `[[nodiscard]]`)* Failures that exit 0 or report success: a failed checkpoint save or `--dist-matrix` load (S-04);
  `set_max_iter(0)` / `set_n_repetitions(0)` (O-06); Benders on invalid input (A-07); a failed output write, because the CLI and Tier-1 writers check only
  `is_open()` and a full disk truncates labels and medoids with exit 0 (`dtwc_cl.cpp:645-697`,
  `api.cpp:269-282`; B-05); CLARANS with `num_local ≤ 0`, which writes an empty clustering of cost `DBL_MAX`
  into `Problem` (`clarans.cpp:72, 244-248`; A-05), its neighbour count computed in 64 bits; a loaded
  `--dist-matrix` whose N differs from the series count, which is discarded silently and recomputed
  (`Problem.cpp:694-706, 873-878`). Also `-k`'s error names
  the deprecated `--clusters` (S-10); `[[nodiscard]]` on the bool-returning `set_solver` / `load_checkpoint`.
- ☑ ★ **FX-4** *(done 2026-09-24, merged, uncommitted: S-13 pinned by a `static_assert`; F25 — `get_name` / `p_vec` raise on storage they do
  not own. Earlier: S-06, S-07, C-08 and B-08 (LargeUtf8 names accepted too); B-08 proven by compiling `test_io_readers`
  against pyarrow 25.0.1's libarrow by hand — the CTest target still needs an Arrow CMake package. Left: S-13, F25)* `.dtws` open rejects `ndim = 0` instead of dividing by it, and opens a read-only input read-only
  (S-06, S-07); `Problem`'s move assignment drops the `noexcept` its `unordered_map` member cannot promise
  (S-13); the umbrella header refuses `-ffinite-math-only` with an `#error` (C-08); `Problem::get_name` and
  `p_vec` guard view mode only with `assert`, so a Release build reads out of bounds (`Problem.hpp:369-393` →
  typed error; F25); the Arrow IPC reader casts the `name` column to `StringArray` without a type check — UB on
  a non-UTF-8 column — and parses `ndim` with an unguarded `stoul` (`arrow_ipc_reader.hpp:148, 156`; B-08).
- ☑ ★ **FX-5** *(done 2026-09-24, merged, uncommitted: the wrapper and `cluster_generic.slurm` moved to `python/dtwcpp/_slurm/`
  (found with `importlib.resources`); `scripts/slurm/slurm_remote.sh` forwards; bash-3.2-safe seed; `test_hpc` 126 / 126; an
  installed wheel builds the job from a directory outside the repo, with every remote command sent to a fake ssh. Left: a real ARC run)* `device="hpc"` works from an installed wheel. `_hpc.py:391-393, 565` looks for
  `scripts/slurm/slurm_remote.sh` under `DTWC_REPO_ROOT` or the working directory, while the wheel ships only
  `python/dtwcpp` — so outside a checkout the feature cannot work. Ship the script as package data and find
  it with `importlib.resources` (non-negotiable 1). While there: `cluster_generic.slurm:105` expands
  `SEED_ARGS[@]` under `set -u`, which macOS's bash 3.2 rejects ("unbound variable"), failing four
  `tests/python/test_hpc.py` cases here.
- ☑ ★ **FX-6** *(done 2026-09-23 and merged, uncommitted: fast_float 8.3.0 vendored (SHA-256 `f23d93a4…9b90`),
  `minos 13.3` shown by `otool`, reader 491 → 775 MB/s with bit-identical values, merged tree 131 / 131. Left: the
  Arrow-side B-11 copies (`parquet_reader.hpp:36, 148`, `arrow_ipc_reader.hpp:49`, `parquet_chunk_reader.hpp:42,
  210`, `dtwc_cl.cpp:1269, 1304`) need an Arrow build; `--skip-cols 1` without `--skip-rows 1` still reads a pandas
  `,0` header as a leading 0 — documented, not detectable; the bundled libomp is FX-14)* **Text readers: keep the hand-written row splitter, vendor fast_float for the numbers, fix the
  defects** (X-28, D-B). Floating-point `std::from_chars` is *unavailable below macOS 26.0* (Tahoe, 2025) with the current SDK
  (verified: it compiles only at `-mmacosx-version-min=26.0`), so today's CLI binary runs only on macOS 26
  (`minos 26.0`) and the wheel job's `MACOSX_DEPLOYMENT_TARGET=11.0` (Big Sur, 2020; `python-wheels.yml:77`)
  cannot compile —
  a release blocker, and the portability failure `DECISIONS.md` named as the condition for fast_float
  (Apache-2.0 / MIT / BSL-1.0, one header; the audit measured it bit-identical to libc++ on 9,995,196 parses
  and faster). One `.cpp` helper replaces the three `from_chars` sites, so fast_float stays out of installed
  headers; one shared line loop replaces four copies; the macOS wheel target rises to 13.3 (Ventura 13.3, March 2023), the first release whose libc++ has
  floating-point `to_chars` (verified by the same compile test).
  Defects it fixes, each run by the audit: a trailing blank line becomes an empty series, which either clusters with
  `DBL_MAX` distances (exit 0) or fails later with a misleading message (`fileOperations.hpp:223`) — trailing blank lines are now ignored, a blank line
  followed by data is an error, and `set_data` rejects an empty series from any source; a folder of two-column
  files read without `--skip-cols` silently clusters the index column; an empty line in a folder file drops a
  value and shifts the rest; the legacy-header skip drops a leading NaN; dot-files such as `.gitkeep` become
  series; a pipe without a BOM reads 0 series; `+-1` parses as −1; `std::isspace` reads `LC_CTYPE`.
  The other io copies fold into one helper each: Arrow status checks ×3, open / close ×3, extension scan
  ×2, `saturating_add` ×2 (B-11, K4).
  Prototype and its input matrix: `plans/2026-09-23-fx6-reader-prototype/` (+117 / −94 lines; 190 MB read in
  0.25 s instead of 0.38 s; two existing tests encode the old blank-line contract). Every library measured
  worse on our input: Armadillo zero-pads ragged rows, reads an empty cell as 0 and a BOM as a leading 0;
  rapidcsv parses through `std::stod` (locale); csv-parser drops ragged rows and misrounds; Arrow's reader
  rejects ragged rows.
- ☐ ★ **FX-9** MATLAB `dtwc.load` reads CSV with `readmatrix` (`+dtwc/Dataset.m:82`), which NaN-pads ragged rows
  and, in the audit's R2026a run, lost the first series. Route it through the C++ reader with a MEX `read_data`
  entry mirroring Python's `_read_data` (~20 lines) and delete the second reader.
- ☑ ★ **FX-10** *(done 2026-09-23, merged; both new pytests fail on `e784e5c` and pass now)* Python: `io.load_dataset_csv` takes a BOM-prefixed first row for a header and drops the first
  series (`io.py:71-79` → `encoding="utf-8-sig"`); the `hpc` path writes values with `:.10g` (`_hpc.py:279`),
  so an HPC run clusters different numbers from a local one (→ `repr`).
- ☑ **FX-11** *(done 2026-09-23, merged)* The `--dist-matrix` reader accepts a 2×3 file as 2×2, lets the last write win on an asymmetric
  pair, and leaves short rows silently uncomputed (`core/matrix_io.hpp:125-172`) — typed errors instead.
- ☐ **FX-12** On Linux, `available_ram_bytes()` reads `MemFree`, which ignores reclaimable cache and a SLURM
  job's cgroup `memory.max`, so `StoragePolicy::Auto` can keep in RAM what the job may not hold. Read
  `MemAvailable` and cap by `memory.max` (~25 lines) — confirm the effect on a node first `[inferred]`.
- ☑ ★ **FX-13** *(done 2026-09-23 and merged, uncommitted; the merged tree passes 131 / 131 with 2 CUDA skips: CPU and Metal verified — before the fix 115 / 1,200 CPU bounds
  exceeded the exact distance and Metal pruned 15 / 240 pairs within threshold, after it 0; F12's Metal half ran on the
  device for the first time and matches the CPU; suite unchanged at 128 / 133; CUDA mirrored, unverified → V-10.
  Still open: F47 — `lb_kim_valid<SquaredL2Metric>` is true while LB_Kim returns L1 units; nothing uses it today)*
  Every lower bound admissible for its cost and band. The GPU LB_Keogh sums L1 excess even under
  squared L2 (`metal_dtw.mm:946-953`, `cuda_dtw.cu:810`; F27); Metal's envelope for full DTW defaults to
  `max_L/10` (`metal_dtw.mm:1512-1514`; F28); the CPU `compute_envelopes(band < 0)` builds a radius-0 envelope
  (`lower_bound_impl.hpp:44, 62`; F46) — each can exceed the true distance, and pruned pairs are written as
  `FLT_MAX`, a finite poison (design §9), rather than NaN. Reachable from Python
  `compute_distance_matrix_{metal,cuda}(…, use_lb_keogh=True)` with the default `band=-1`. Lands before PF-3's
  cascade, which relies on the same bounds; Metal's band geometry and sentinel are checked against the CPU
  oracle on this Mac (F12, Metal half).
- ☐ **FX-7** Pin with a test or prove unreachable: `dtw_kernel_eap` can return a stale cell after `break`
  (S-12); in hierarchical medoid selection a NaN candidate cost fails both comparisons and leaves
  `best_idx = mem[0]` (`hierarchical.cpp:246-258`; A-09).
- ☐ ★ **FX-8** Conformance under GCC and MSVC Release, once, with `DTWC_FP_MODEL=strict` (V-7).

- ☑ ★ **FX-15** *(done 2026-09-24, merged, uncommitted; the `Problem` half — ±inf always, NaN under `Error`, on the fill, the lazy path and
  the kernel accessor, through the same `require_finite` — landed with the `Problem` agent, replacing the fill's NaN-only scan: `dtwc::detail::require_finite` (`warping.hpp`) is the one check;
  `distance::*`, `dtw_runtime` and `soft_dtw_gradient` are the checked boundary, `warping*.hpp` stays the unchecked per-pair
  layer (documented), and the fill's objects are byte-identical; Python's distance functions and matrix entry points and MATLAB's
  `dtwc.distance.*` go through it. Premise corrected: `distance::*` checked nothing before and nothing rejected ±inf. 65 / 138 C++ and
  16 / 27 pytest cases fail on the base; MATLAB passed here, V-14 ☑. Left: MATLAB messages give 0-based positions)*
  **NaN and ±inf on the raw DTW entry points are silently wrong.** Python `dtw_distance` / `ddtw_distance`
  / `wdtw_distance` / `adtw_distance` (`_dtwcpp_core.cpp:653-690`), the C++ `warping.hpp` wrappers and
  `Problem::dtw_function()` under `Error` pass non-finite input straight to the kernels; of 6,252 NaN-input results
  the record measured 2,451 NaN, 3,681 the unreachable `max` and 120 finite numbers (`±inf` in both series gives
  `inf − inf` = NaN too). Reject non-finite input with a typed error naming the series and position unless a
  missing-data strategy is set — once, at each entry point, as `fill_distance_matrix`, `distance::dtw` and
  `dtw_runtime` already do. This is input validation, not an overflow guard.
- ☑ ★ **FX-14** *(done 2026-09-24, merged, uncommitted: `scripts/build_libomp_macos.sh` builds LLVM 23.1.1's libomp (tarball SHA-256
  pinned, equal to Homebrew's) for 13.3 through the runtimes build, since LLVM 22+ ships no per-project tarballs; both workflows call it;
  `smoke_release_archive.py` fails on any Mach-O above 13.3; delocate 0.13 enforces `MACOSX_DEPLOYMENT_TARGET` on the wheel. Every
  Mach-O in a local archive and wheel is 13.3; Homebrew-built artefacts fail both checks. Left: the workflows are unrun (GT-5))*
  **The bundled OpenMP runtime is built for macOS 26.** Homebrew's `libomp.dylib` carries `minos 26.0`,
  and both the CLI archive and the wheel (through delocate) bundle it, so after FX-6 the shipped artefacts still
  refuse to start below macOS 26. Bundle a runtime built for 13.3 — for example conda-forge's `llvm-openmp`, or
  LLVM's `openmp` runtime built in the release job with `MACOSX_DEPLOYMENT_TARGET=13.3` (Apache-2.0 WITH
  LLVM-exception either way). Proof: `otool -l` shows `minos` ≤ 13.3 on every dylib inside the archive and the
  wheel.

- ☑ ★ **FX-16** *(done 2026-09-24, merged, uncommitted: HiGHS builds static when `DTWC_BUILD_PYTHON` is ON (`cmake/Dependencies.cmake`,
  one condition); delocate's repair, run as cibuildwheel runs it, went from "Could not find all dependencies" to exit 0 with only
  libomp bundled (wheel 2.90 → 2.71 MB), both Mach-O files minos 13.3; a fresh venv outside the repo runs the wheel smoke (18 OpenMP
  threads, HiGHS MIP cost 48.0); the CLI archive still ships `lib/libhighs.*`. Left: Linux / Windows legs unrun (GT-5); on Linux,
  auditwheel probably bundles `libgomp`, which `THIRD_PARTY_LICENSES.md` says we never redistribute `[inferred]` → V-13)*
  **The wheels cannot be repaired.** `_dtwcpp_core` links `@rpath/libhighs.1.dylib`, `wheel.exclude` drops `lib/`, and
  the module records no `LC_RPATH`, so delocate, run as cibuildwheel runs it, stops with "Could not find all dependencies"; the FX-5 / FX-14
  wheel proofs got past it only with `DYLD_LIBRARY_PATH` (found 2026-09-24; answers RL-3's open question). Linux (`auditwheel`) and Windows
  are likely the same `[inferred]`. Link HiGHS (MIT) statically into the extension, or install `libhighs` inside the package with an
  `@loader_path` / `$ORIGIN` rpath — whichever deletes more. Proof: the cibuildwheel repair command succeeds with no library path set, and
  the repaired wheel imports and solves a MIP in a fresh venv outside the repo.

- ☑ ★ **FX-17** *(done 2026-09-24, merged, uncommitted: all four, plus the GT-4 review's lost `load:` prefix and one "format this build
  cannot read" type (`IOError`); also found and fixed: hpc applied an in-memory dataset's `skip_cols` twice. ctest 137 / 137, pytest 1,231 / 0.
  MATLAB run here → V-15 ☑; CUDA blind → V-16)* Silent failures found by the IF-2 design, each made loud now because IF-2 rewrites the code only later: MATLAB
  `Problem.set_method('pam'|'auto')` runs Lloyd k-medoids (`dtwc_mex.cpp` ~498-503) → `dtwc:invalidArgument` naming `dtwc.fast_pam`
  (blind → V-row); Python's hpc route drops `delimiter` silently (`_api.py` ~537-545) → `InvalidInput` until S4 carries it; the CLI's
  `--dtype float32 --device cuda` hands the CUDA fill an empty `p_vec` (`dtwc_cl.cpp` ~1693) and the CPU computes [inferred — confirm
  first] → `DeviceError`; `cli.md` documents the `--method` default as `pam` (it is `auto`).

- ☑ **FX-18** *(done 2026-09-24 with IF-2 S2, merged, uncommitted)* Auto (N ≥ 64, band ≥ 0) and `Pruned` read multivariate series as one
  interleaved series, so every pair was wrong (2,016 of 2,016 in the probe); they now use the multivariate kernel, and the pruned builder
  refuses multivariate or non-L1 input.
- ☑ ★ **FX-19** *(done 2026-09-24, merged, uncommitted: `detail::require_metric_supported` in `distance.hpp` is the one copy of the rule; WDTW,
  ADTW, Soft-DTW, MSM and TWE refuse a non-L1 metric; DDTW and the missing-data strategies keep it (they use it: DDTW 4 → 8 under squared L2),
  so `Problem::set_metric`'s narrower rule (standard only) is the stricter of two — widening it is additive. Examples fixed and
  `example_new_features` registered. FX-19b: `core::dtw_runtime` and Python's `distance.dtw` refuse instead of dropping it. Left for IF-2
  S4: Python's `ddtw_distance` binding takes no `metric`, so Python DDTW under squared L2 raises where C++ computes it)*
  `distance::dtw(x, y, params, band, metric)` drops `metric` for WDTW, ADTW, Soft-DTW, MSM and TWE (`distance.hpp` ~164-178),
  so a caller asking for squared L2 silently gets L1 → `InvalidInput`, the rule `Problem::set_metric` already applies. Also open from
  S2: `examples/cpp/example_new_features.cpp` ~93 does not compile and no target builds it (D-6); `examples/example_project/main.cpp` ~24
  discards a `[[nodiscard]]` result.
- ☑ ★ **DOC-1** *(done with IF-2 S3, 2026-09-24, merged; GT-4b's §5 rows included)* Contract and docs-gate text for IF-2 S1 / S2: `docs/api-contract-2.0.md` rows 37 / 38 (`settings::paths` removed pre-tag;
  `Problem::set_output_folder`, default `./results/`), §2.3 (the path-setter sentence goes), §2.1 (a `metric` row; the output folder's default),
  §2.2 ~315 and §2.7 ~498 (the one- and two-argument mmap / checkpoint forms), ~696 ("33 C++ diagnostic entities" → 29, and the pinned counts in
  `scripts/check_docs_contract.py` ~358, 509, 1596-1597, 1627, 1632 → `(28, 29, 12, 13, 15)`), `design.md` §2; regenerate `tier-2.md` and
  `migration.md` with `uv run --no-project python scripts/generate_docs.py`.

### 3.3 Interface — one surface in every language

- ☑ ★ **IF-1** *(done 2026-09-24, merged, uncommitted: `detail::parse_device` is the one grammar (`Env`, Python, MATLAB);
  `configure_device` deleted; Python `Problem(name="", *, device="cpu")` / `set_device`; MATLAB `'Device'` / `set_device` and
  `DTWClustering` use it — MATLAB passed here, V-12 ☑)* `Problem::set_device(Device, index = 0)`: `api.cpp::configure_device` (`:94-118`) moves into the
  session; `Env` stays the process default; Python and MATLAB gain `Problem(device=…)`. Nothing else is added:
  `DistanceMatrixStrategy` stays, and `CUDA` / `Metal` remain spellings of `gpu` (X-01).
- ◐ ★ **IF-2** *(S3 done 2026-09-24, merged, uncommitted: `run(Config)` / `run(Config, Data)` in `cli/run.*`; `dtwc_cl.cpp` 1,996 → 133 lines;
  Tier-1 `cluster()` a 20-line wrapper; `validate_gpu_request()` holds the data-free GPU rules the fill also calls; `Result` gains `method()`,
  `iterations()`, `converged()`; DOC-1 done; `test_run_resolution` and `test_cli_device_matrix` (25 cells, `--print-config` against the golden
  file); old vs new CLI byte-identical on 56 / 56 runs; mutations of six rules each caught. Blind → V-17. Left: S4 (bindings, hpc by config
  file). S2 done 2026-09-24, merged, uncommitted: `Problem::set_metric` in the CPU fill (the kernels' own metric argument), the
  GPU routes, the dense / mmap / checkpoint identities and autosave; `settings::paths` out (examples take folders from argv; the shim probe
  samples v1.0.0's `fillDistanceMatrix`); CLARA copies metric and GPU settings; one Benders guard; `test_problem_metric` (12 cases, CPU squared
  L2 equal to `distance::dtw` exactly). S3 needs: `set_metric`, `use_mmap_distance_matrix(path)`, the two-argument checkpoint forms; the
  three-argument ones mix metrics and lose their caller with S3 — refuse a mismatch or drop them pre-tag. S4 needs: Python / MATLAB
  `metric` and checkpoint bindings defaulting to the Problem's metric (a squared-L2 Problem must not save an L1 tag); `_clustering.py`
  ~375-389's detour and five "Problem is L1 only" comments go. S1 done 2026-09-24, merged, uncommitted: `base/names.hpp` + 13 tables beside their enums; `cli/config.{hpp,cpp}` — `Config`,
  `cli::bind`, `to_config_text`, `parse_config`, reading `--config` through the CLI's existing reader; `test_names` (324 assertions) and
  `test_config_spellings` (7,403; golden `tests/conformance/config_all_fields.{toml,yaml}`, 49 keys, every value non-default) — a mutation
  bit 4 times; 138 / 138. Deviations: files in `dtwc/cli/` (a root file breaks `repo_map.py`); CLI11 linked PUBLIC because `bind` takes a
  `CLI::App&` (2.1's installable package must export or hide it); `dtwc_cl --help` shows no defaults, so 16 of 49 are pinned against the
  binary now and the rest by S3's `--print-config`. Left for S3/S4: `core::parse_metric_token`'s own list, a `DistanceMatrixStrategy` table,
  the wheel column of `THIRD_PARTY_LICENSES.md` once the bindings compile CLI11 / fkYAML in. FX-1 found: the CLI has its own device parser — cpu / cuda only, no gpu or Metal — and its own CUDA fill
  (`dtwc_cl.cpp` ~337, ~1620): both go, for `detail::parse_device` + `Problem::set_device` + the Problem's fill. `run(Config)`
  resolves method and device together: CLARA, CLARANS, OneBatchPAM and TADPole compute on the CPU under `gpu` today, and CLARA's
  sub-Problems drop `cuda_settings` (`fast_clara.cpp` ~371, ~520) — so `gpu` with a matrix-free method is a typed error naming the
  methods that use the GPU, and `auto` under `gpu` picks a matrix method. **Designed 2026-09-24:** `plans/2026-09-24-if2-config-design.md`
  — `Config` keyed by the CLI long names, one `Name<E>` table per enum (`base/names.hpp`), `cli::bind(App&, Config&)` as the only key
  table, `run(Config)` as the one pipeline, hpc submits a config file; steps S1 (tables, Config, bind) ∥ S2 (`Problem::set_metric`,
  `settings::paths` out, CLARA keeps GPU settings) → S3 (`run`, CLI ~1985 → ~150 lines, `api`) → S4 (bindings, hpc))*
  `dtwc::Config` + `run(Config) → Result`: an aggregate of the existing option structs, with
  defaults as member initialisers that CLI11 reads and one string↔enum table per enum. The CLI, a config file,
  Python keywords and MATLAB name-value pairs are four spellings of it, and `hpc` submits it. The
  `settings::paths` globals become `Config` values. No `schema` key, no alias table beyond what CLI11 needs
  (X-03, B-01, B-02, O-11, O-21, A-01).
- ◐ ★ **IF-3** *(grammar half done 2026-09-24, merged, uncommitted: `_dtwcpp_core.parse_device` over `detail::parse_device`; Python's copy
  (−57 lines) and its cuda-only branch deleted; `test_device.py` re-pinned, `test_cuda.py`'s pin waits for GT-4's merge. Left: `run(Config)`,
  the FX-1 bypasses, `_hpc.py`'s cpu / cuda grammar (IF-2 S4), Python's `Result.device` saying `"metal"` where C++ says `"gpu"`. K1, decided by the contract: `api-contract-2.0.md` §6.1 makes `cuda` an alias of `gpu`, so Python follows it through a
  new `parse_device` binding over `detail::parse_device` — deleting Python's own grammar and `_resolve_device`'s cuda branch — and the three
  pytest cases that pin `cuda` as CUDA-only change. FX-1 found: Python Tier-1's GPU path (`_api.py` ~591), `DTWClustering` (`_clustering.py` ~376) and direct
  `core::fill_distance_matrix_pruned` calls bypass `validate_fill_request`; Python's device canonicaliser rejects `cuda` on a
  Metal build while C++ reads it as `gpu` — two pytest cases fail on the base)* Python and MATLAB Tier-1 are one call to `run(Config)`: this deletes the dispatch that `_api.py`
  re-implements (602 lines) and the second device store `_HPC_SELECTED` (S-18, F37, F24).
- ☐ ★ **IF-4** Tier-1 `Result` copies its labels, medoids, cost and `RunStats` at construction and keeps the
  session only for the lazy matrix path, behind a `std::once_flag`, so copies stop sharing a mutable `Problem`
  without waiting for PF-1 (X-19, S-17). `RunStats` (pairs, cells, prunes, iterations, device actually used) comes back from
  every gateway; tests assert it to prove a route ran (X-07). Python gets a progress / cancel callback polled
  per block with `PyErr_CheckSignals`, so Ctrl-C works during a GIL-released fill (X-09, X-09a).
- ☐ ★ **IF-5** Pre-tag surface slice (D-11): 2.0-only Python aliases go, each checked against `git show
  v1.0.0:python/py_main.cpp`; one estimator (D-14); `IOError` and `test` leave `__all__` (S-19); `str | Enum`
  setters and `.pyi` stubs from the C++ tables; `distance.msm` / `distance.twe` in Python and MATLAB (S-14);
  MATLAB's 32 unchecked `static_cast<int>` go through the existing `get_exact_int` (S-05).
- ☐ **IF-6** One artefact writer used by the Tier-1 API, the CLI and `Problem::write_*` — four schemas today,
  one of them `Problem_IO`'s 1.x filenames, which stay (O-09; Volkan, 2026-09-23: "output names as it is
  fine").
- ☐ **IF-7** `distance::dtw_path` from the barycenter backtrack (`barycenter.cpp:184-224`), which also becomes the
  tests' reference path (X-17, C-26, A-20).
- ☐ **IF-8** Zero-copy input for 2-D float64 / float32 arrays and lists of 1-D arrays (F26).

### 3.4 Performance

- ☐ **PF-1** `PackedOracle{span, n}` + `Problem::oracle()`, which checks "filled" once (O-13). Matrix-based
  algorithms take it and return a `ClusteringResult` instead of writing `Problem` members; every one of them
  runs on a matrix-only `Problem` (a test). The matrix-free algorithms — CLARA, CLARANS, OneBatchPAM, TADPole and
  barycenter k-means — compute from the series and keep `Problem&`, documented; there is no oracle concept (A12, A-02, A-03, A-12, A-19, A-21, S-11, O-03, O-16, X-11).
  Heuristics are checked against the exact solver on small fixtures, within a registered gap (X-08).
- ☐ **PF-2** `dist_by_ind` in O(1): snapshot compare → packed lookup → compute on a miss, with validation at the
  gateways — IPO does not inline it (X-04), so today the double preflight is paid per lookup. The distance-cache
  identity becomes the distance semantics only, so changing device keeps the matrix (O-01, O-02, O-08, A-23,
  C-01, C-22, S-16, A11).
- ☐ **PF-3** Scores take the oracle: the O(N·k) medoid silhouette and inertia come from the assignment, and the
  full silhouette on a matrix-free result is a typed error (X-20, S-20, D18). Lower-bound cascade with early
  abandon in the nearest-medoid assignment of CLARA, OneBatchPAM and barycenter k-means — admissible, because an
  argmin needs only the winner exactly (X-18). `Auto` becomes brute force for exact matrices; `Pruned` stays,
  documented as a diagnostic (C-04, C-05, C-25, D-16).
- ☐ **PF-4** One backend `fill()` translation unit holds the `#ifdef`s that `Problem.cpp` and `api.cpp` carry
  today. GPU fillers write the packed slice of their pair range, dropping the N×N buffer and its copy (A13,
  G-01…G-03). CUDA launches are chunked by a 64-bit pair offset as Metal's already are, which removes the
  N ≥ 65,537 refusal at `cuda/launch_prep.hpp:47`. `CUDAPrecision` and `MetalPrecision` become one
  `GpuPrecision` (X-14).
- ☐ **PF-5** **SIMD across pairs** for the exact-matrix fill — this reopens the SIMD kill (`DECISIONS.md` §1).
  Series *i* runs against W series in lockstep; lanes come from a length-sorted block of its pair range, and
  leftovers go to the scalar kernel. Each lane performs the scalar `min(diag, up, left) + cost` in the same
  order, so the result is digit-identical under `strict`, and under `fast` for L1. For SqL2 under `fast` — the
  default build — lane and scalar code may contract `d·d + min` to FMA differently; step 1 decides between
  compiling that cost without contraction in both and pinning SqL2 lanes under `strict` only. The fill is latency-bound, not memory-bound — L² dependent cells
  against 2L reads per pair — so independent lanes pay: expect ~2× NEON f64, ~4× AVX2 f64 or NEON f32, ~8×
  AVX-512 f64. **Plain C++, no library, no runtime dispatch** (Volkan, 2026-09-23: Highway "wasn't worth the effort"). The
  March–April 2026 attempt (`416acbd`…`d670143`) failed for two reasons that do not apply here — its dispatched
  route gathered pairs (28 operations per cell against 9) and ignored bands and variants (`923f723`), and
  Highway's dispatch made the small reduction loops 3–4× slower than auto-vectorised code (`55af20c`) — while
  its equal-length structure-of-arrays batch measured ~2.8× against four sequential calls (`LESSONS.md`). So:
  one lane loop over a `[t][W]` block with `#pragma omp simd` on the lane index, for StandardCell / ADTWCell ×
  L1 / SqL2 × f32 / f64, measured on this Mac against the scalar fill at equal threads; each build uses its
  baseline ISA (NEON here, SSE2 in x86 wheels). **Kill criterion:** below 1.5× cells per second on GT-6's
  dense benchmark, the item closes FALSIFIED and the scalar fill stays. The rolling row
  becomes `[n_short][W]` — at L = 512, W = 8, f64 that is 2 × 32 KB and leaves L1, so tile columns or prefer f32
  lanes. Cost and Cell policies get `concept`s here, because lanes give each policy a second implementation
  (C-15), and the fill moves to pair blocks with them (C-07). Gate: conformance against the scalar kernel per
  (variant, cost, precision), a pairs-per-block counter, one benchmark band registered first. May land after
  the tag.
- ☑ **PF-6** *(2026-09-23, `baselines/2026-09-23-x27-eigen-gap.md`)* The build without Eigen is **not** slower: interleaved
  A/B of fresh builds agree within noise (−0.43 / −0.03 / −0.14 / −0.09 % at 100 / 500 / 1000 / 4000), and both
  compile the inner loop to the same 17 instructions per cell. Alignment and accessor codegen FALSIFIED; X-27's 5 %
  was sequential measurement under drifting load [inferred]. The quiet re-run (13:56–14:10) agrees: within ±1 % at every size
  (−0.52 / +0.00 / +0.04 / +0.06 %), the spread band missed only at n = 100 through drift common to both builds —
  recorded after two attempts. Closed. What the dig found instead is PF-7.
- ◐ **PF-7** *(2026-09-23: 64-byte loop alignment FALSIFIED both ways — targeted `[[clang::code_align(64)]]` and
  build-wide `-align-loops=64` each left 20+ of 72 benchmarks slower, because each instantiation's own `fcmp`/`fcsel`
  offsets decide, not the loop start; neither applied. **The tree is on the cliff today:** `BM_dtwBanded` and the
  banded fills run up to 30 % slower than they could, and `dtwc_cl` has 10 split kernel loops. PF-7b then tried to remove the
  pair and was FALSIFIED twice (record §11): a NaN-exact `fminnm` min costs extra operations — two mins on the
  carried chain made the linear kernel 32–37 % slower; one min plus NaN repair was faster on 30 of 72 cells but
  slower on 20 — and 11 split pairs sit outside the Cells anyway. Nothing applied. The premise changes only after
  FX-15: once the raw paths reject non-finite input, plain `fmin` is exact on every accepted input and gets a new
  band)* **The placement cliff.** The DTW inner loop runs ~30 % slower whenever an `fcmp`/`fcsel` pair of the
  min-of-three straddles a 64-byte boundary — 8 of 8 such placements slow, 0 of 56 others; relinking the real
  benchmark with 4 bytes of padding costs 28–46 %. So any build can lose a third of its DTW speed to where the
  linker happens to put the loop, and an A/B between two builds says nothing until placement is pinned. Remedy,
  measured, not applied: 64-byte loop alignment — under ThinLTO it must go to the link step as
  `-Wl,-mllvm,-align-loops=64` (`-falign-loops=64` at compile time is accepted and ignored); with it 32 of 32
  placements were fast. Build-wide, so register a band first (every GT-6 benchmark no slower, spread ≤ 0.5 %),
  measure on a quiet machine, and check GCC / MSVC equivalents on the other machines. Layout does not change
  arithmetic, so conformance stays digit-identical by construction.

### 3.5 Release

- ☐ ★ **RL-1** Pre-tag deletions: 2.0-only symbols (C-09, by release history — D-3); the forwarding headers of
  the five headers born in 2.0 (`error`, `missing_utils`, `env`, `system_memory`, `random_engine`; `settings`,
  `parallelisation` and `timing` shipped in v1.0.0 and keep theirs for one release); the six `dtwc::solver`
  declarations with no consumer (D-21); the hand-written series-count guards (`fast_pam.cpp:60`,
  `fast_clara.cpp:71, 201, 482`, `one_batch_pam.cpp:56`, `mip/solution_transaction.cpp:93`), per §1.1. The
  `dtwc_*` option spellings keep warning until the release after 2.0, as adopted on 2026-09-07 (B-14).
- ◐ ★ **RL-2** *(contract half done 2026-09-24, merged: §1.1, §2.1, §2.2, §2.6, §2.7, §5, §6, §6.1, §6.4 now record the batch, tier pages
  regenerated — was: the contract lacked `set_max_iter` /
  `set_n_repetitions` reject n < 1; `[[nodiscard]]` `set_solver` / `load_checkpoint`; the `read_distance_matrix` size check; writers checked
  after close; the `dtw_function()` validation; its F25 sentence was false. Open: §2.2 says the writers raise `IOError`, which holds for
  `write_distance_matrix` only once GT-4 converts `Problem_IO.cpp` ~223, 227 and `core/matrix_io.hpp` ~105, 122)* Docs: README truths (HDF5 is Python-only; `dtwc_main` takes no arguments; the `develop` badge;
  `/convert`); an interop conventions page (L1, integer-cell band and no final square root, against
  dtaidistance, tslearn and aeon); ADTW credited to Herrmann & Webb; the Pages upload's missing `path:`; the break
  register as the migration guide. (H-3, the CHANGELOG cut, landed with GT-2.)
- ☐ ★ **RL-3** Release gate: CI matrices green, full pytest, the MATLAB suite, the archive smoke test, one
  llfio-ON wheel, one adversarial review of the diff since v1.0.0; a built wheel imports without
  `DYLD_LIBRARY_PATH` (the reader audit's local build needed it for libhighs, because `wheel.exclude` drops
  `lib/**` [inferred — check on the wheel CI artefact]. The tag, PyPI and ARC are Volkan's (R6).

### 3.6 After the tag (2.1)

Sharded fill `--shard i/n` / `--merge` for SLURM array jobs, and MPI as a filler (X-10, G-07); an installable
CMake package (S-02, with S-08); the `euclidean` / `window_fraction` interop tokens (A17); `load(…, label_col=)`
with ARI / NMI in Tier-1 (X-22); SBD / k-Shape, NN-chain hierarchical, AMI; the WASM playground (R7). C++ Tier-1
`device=hpc` stays CLI / Python only (D-10).

### 3.7 Break register

Every user-visible change the items above make, with its reason from `design.md` §2. RL-2 turns this table
into the migration guide.

| Item | What a user would notice | Reason | Mitigation |
| --- | --- | --- | --- |
| FX-1 | GPU with a variant, missing-data strategy, `ndim` or precision it does not implement, a band narrower than a pair's length difference, GPU with an mmap / view / Float32 store, Metal `gpu:N` with N ≠ 0 and `MetalPrecision::FP64` now throw; a squared-L2 cache fills on the GPU | R1: silently ignored, silently FP32, or a sum of 1.8e308 | the error names the axis; widening only as an explicit token (D-12) |
| FX-2 | the Soft-DTW self-distance changes | R1 | decided with D7 |
| FX-3 | a failed checkpoint save, matrix load or output write, `set_max_iter(0)` / `--max-iter 0`, a `--dist-matrix` of the wrong size, `--solver gurobi` on a build without Gurobi, Benders on invalid input and CLARANS with `num_local ≤ 0` now fail; writer errors are `IOError` | R1: silent success | typed error / non-zero exit |
| FX-4 | `get_name` / `p_vec` on a view or mmap store, `p_vec` on Float32 or metadata-only series (use `series(i)`), a `.dtws` with `ndim = 0`, a non-string Arrow `name` column and a non-integer Arrow `ndim` now throw; `Problem`'s move assignment is no longer `noexcept`; including `dtwc.hpp` under `-ffinite-math-only` / `-ffast-math` is a compile error | R2: UB, division by zero, false `noexcept`; R1: NaN tests folded away | typed error naming the file; the `#error` names `-fno-finite-math-only` |
| FX-5 | the SLURM wrapper and `cluster_generic.slurm` moved to `python/dtwcpp/_slurm/`; `scripts/slurm/slurm_remote.sh` forwards and always uses its checkout (`.env`, sources, job files); the packaged wrapper, which Python's `device="hpc"` runs, reads `.env` from `$DTWC_REPO_ROOT`, else the working directory; `upload` and `submit-*` refuse to run outside a checkout | R3: `device="hpc"` cannot work from a wheel otherwise | every command still works from a checkout |
| FX-15 | NaN or ±inf passed to `dtwc::distance::*`, `core::dtw_runtime`, `soft_dtw_gradient`, Python's distance functions and `compute_distance_matrix`, or MATLAB `dtwc.distance.*` now raises `InvalidInput`; the missing-data distances still accept NaN but reject ±inf | R1: NaN, the unreachable 1.8e308 or an ordinary-looking number | the message names x / y (or the series), the position and the fix |
| FX-6 | interior blank lines, multi-field folder lines (`data/dummy` without `--skip-cols 1`), `+-1`, a no-break space and an empty first value after `--skip-rows` now fail; trailing blank lines and dot-files are ignored; empty series are rejected | R1 | the message names the row and the fix (`--skip-cols`, `nan`) |
| FX-6 | the macOS wheel and CLI require macOS 13.3 | PRE-TAG: an 11.0 build never compiled | — |
| FX-9, FX-10 | MATLAB reads ragged rows as ragged series; a BOM file keeps its first series in Python; HPC runs get full-precision values | R1 | — |
| FX-11 | a non-square, short or asymmetric `--dist-matrix` file raises `InvalidInput` | R1 | triangle files still load |
| FX-13 | GPU LB_Keogh prunes fewer pairs (squared L2, full DTW, `INT_MAX` band); Metal at band 0 prunes more (radius 0, not `max_L/10`); a pruned pair is NaN, not `DBL_MAX`; a Metal `lb_envelope_band` narrower than the DTW window throws; `compute_envelopes(band < 0)` is the global envelope | R1: inadmissible bound, finite poison | `isnan` finds pruned entries; the error names the window and says to pass −1 |
| PF-3 | `Auto` fills by brute force instead of pruned | none needed: identical matrix, less work | — |
| FX-17 | MATLAB `set_method('pam'/'auto')` raises `dtwc:invalidArgument`; hpc: a file `Dataset` with `delimiter` raises `InvalidInput` and an in-memory `skip_cols` is applied once; `dtwc_cl --dtype float32 --device cuda` raises `DeviceError` | R1: it ran Lloyd / ignored or doubled the option / uploaded no series; PRE-TAG | `dtwc.fast_pam`; omit `delimiter` or use `cpu` / `gpu`; `--dtype float64` |
| IF-2 | `settings::paths` and its setters are gone; a Problem's default output is `./results/` | PRE-TAG (2.0-born) | `set_output_folder` |
| IF-2 | the two-argument checkpoint forms take `prob.metric()`; taking their address is ambiguous | R3 | pass the metric |
| FX-19 | `distance::dtw` with WDTW / ADTW / Soft-DTW / MSM / TWE and a metric other than L1 raises `InvalidInput` | R1: it returned the L1 distance | pass `MetricType::L1`, which they compute |
| FX-18 | multivariate Auto / Pruned fills are now correct; a direct pruned fill refuses multivariate or non-L1 data | R1 | — |
| IF-2 | CLARA sub-samples inherit metric, GPU index and precision (in-memory samples on a GPU raise `DeviceError`); TADPole under a non-L1 metric no longer prunes | R1 | `device="cpu"` for CLARA's in-memory samples |
| IF-2 | CLI `--device gpu` / `gpu:N` accepted, `cuda` runs on Metal, `hpc` raises `DeviceError`; on `gpu` `auto` runs pam above 5,000; onebatch, tadpole and a smaller CLARA sample on `gpu` raise `DeviceError` (CLARA was `InvalidInput`); CPU squared L2 runs | R3: the CLI contradicted §6.1 and Tier-1 | `--device cpu`; `slurm_remote.sh submit-cluster` |
| IF-2 | CLI: MIP settings checked for every method; Tier-1's texts for k > N and reader errors; `--checkpoint-interval 0` saves at the end; series storage Auto on `cpu`; loader lines only with `-v`; YAML without fkYAML is `IOError` | R3; PRE-TAG | — |
| IF-2 | C++ `cluster(device="hpc")` raises D-10's error without reading `.env`; `detail/tier1_method_resolution.hpp` is gone | PRE-TAG | `dtwc::run` |
| IF-3 | Python `device="cuda"` / `"cuda:N"` runs on Metal on a Metal build and `device()` returns `"gpu"`, where it raised `DeviceError`; with no GPU it still raises | R3: Python contradicted §6.1, which C++, MATLAB and Python's `Problem(device=)` follow; PRE-TAG (v1.0.0 Python had no device argument) | `device="cpu"`; test `dtwcpp.CUDA_AVAILABLE` to require CUDA |
| GT-4b | checkpoint-path and mmap-creation failures are `IOError`, not `RuntimeError` / `dtwc:runtime`; a cache for other data and a too-wide file `skip_cols` are `InvalidInput`, not `IOError`; PDLP `use_gpu` without the GPU backend raises `DeviceError` | R3 (§5); non-negotiable 3 | `except dtwcpp.DtwcError`; `use_gpu=False` |
| GT-4 | ~190 failures raise their §5 type, not `std::runtime_error` / `invalid_argument`; Python `except RuntimeError` misses them (they are `ValueError` / `OSError` subclasses), MATLAB sees `dtwc:invalidArgument` / `dtwc:ioError` | R3: the contract's §5 taxonomy, which the bindings translate | messages kept; `except dtwcpp.DtwcError` and `catch (const std::runtime_error &)` still catch all |
| IF-5 | `from dtwcpp import *` no longer rebinds `IOError`; 2.0-only aliases go; one estimator | PRE-TAG (never released) | attributes stay; contract §10.5 |
| (2.0 dev, unrecorded until 2026-09-24) | v1.0.0's `settings::resultsPath`, `dataPath`, `dtwc_dataPath` (and their `root_folder` / `dtwc_folder`) are gone | R2: they were the build machine's source tree, baked in at compile time (non-negotiable 1) | `Problem::set_output_folder`; pass data paths explicitly |
| RL-1 | 2.0-only symbols, five 2.0-born forwarding headers and six unused solver declarations disappear; v1.0.0 symbols in C-09 are deprecated for one release | R4 / PRE-TAG | `[[deprecated]]` for anything v1.0.0 shipped |

## 4. Pending verification on another machine `[BLOCKED-ENV]`

| # | Item | What proves it |
| --- | --- | --- |
| V-1 | X-26 | `uv run --no-project python scripts/machine_facts.py --build-dir build --with-dtwcpp` on the RTX box prints the card and its compute capability, and `test.gpu()` reports `validated` |
| V-2 | X-29 | Linux and Windows: `cpack -C Release`, then `python scripts/smoke_release_archive.py build/release` runs the unpacked CLI with HiGHS resolved from the archive |
| V-3 | X-29 | `smoke_release_archive.py::dependency_paths` returns `[]` on Windows — add a `dumpbin /dependents` leg before trusting that gate there |
| V-4 | X-25 | whether the quickcpplib patch is still needed: a wheel built with `DTWC_ENABLE_LLFIO=ON` |
| V-7 | FX-8 | `ctest -R '^cpp_conformance$'` under GCC and MSVC Release, `strict` and `fast`: the same 17 significant figures |
| V-8 | PF-4 | CUDA chunked launches at N ≥ 65,537 match the CPU oracle, and `compute-sanitizer` is clean |
| V-10 | FX-13 | the CUDA half of the lower-bound fix, compiled blind here: `cmake -S . -B build/cuda-verify -G Ninja -DCMAKE_BUILD_TYPE=Release -DDTWC_ENABLE_CUDA=ON`, build `test_cuda_lb_keogh` and `test_cuda_kernel_override`, run `test_cuda_lb_keogh.exe` and `test_cuda_kernel_override.exe "[m50]"`. Proof: all pass with no skip; the FX-13 property case reports 240 checked, 0 pruned, in FP64 and FP32; the F27 pair returns 0.25 and the `INT_MAX` pair 0.0, unpruned; pruned pairs are NaN |
| V-11 | FX-1 | the CUDA half of the fill validation, compiled blind: `cmake --build build/cuda-verify --target test_cuda_correctness test_fill_request test_problem_set_device`, then `test_cuda_correctness.exe "[fx1]"` — it must run unskipped and match the CPU within 1e-9 — and the other two executables |
| V-12 ☑ | IF-1 | *(passed 2026-09-24 on this Mac, R2026a + Metal: `IF1_OK`)* MATLAB: after a `-DDTWC_BUILD_MATLAB=ON` build, `matlab -batch "addpath('bindings/matlab','build/bin','build/bindings/matlab'); p=dtwc.Problem('x','Device','gpu'); p.set_data(randn(6,40)); p.fill_distance_matrix(); p.set_variant('wdtw',0.1); try, p.fill_distance_matrix(); error('t:t','x'); catch e, assert(strcmp(e.identifier,'dtwc:deviceError')); end; disp('IF1_OK')"`, then `ctest -R matlab_suite` |
| V-13 | FX-16 | a Linux wheel from the CI artefact: `unzip -l` / `auditwheel show` — if `libgomp` is inside, either build the Linux wheels against LLVM's libomp or correct the libgomp sentence in `THIRD_PARTY_LICENSES.md` (GCC's runtime exception allows the redistribution; the notice must say what ships) |
| V-14 ☑ | FX-15 | *(passed 2026-09-24 here: `FX15_OK`)* MATLAB: after a `-DDTWC_BUILD_MATLAB=ON` build, `matlab -batch "addpath('bindings/matlab','build/bin','build/bindings/matlab'); c={@() dtwc.distance.standard([0 NaN 1],[0 1 2]),'x[1] is NaN'; @() dtwc.distance.wdtw([0 1 2],[0 1 Inf]),'y[2] is +inf'; @() dtwc.distance.missing([0 1 2],[-Inf 1 2]),'y[0] is -inf'; @() dtwc_mex('soft_dtw_gradient',[0 NaN],[0 1],1),'x[1] is NaN'}; for k=1:4, ok=false; try, c{k,1}(); catch e, ok=strcmp(e.identifier,'dtwc:invalidArgument')&&contains(e.message,c{k,2}); end, assert(ok,'case %d',k); end, assert(isfinite(dtwc.distance.missing([0 NaN 2],[0 1 2]))); disp('FX15_OK')"` |
| V-15 ☑ | FX-17, IF-1, FX-15 | *(passed 2026-09-24: `test_contract_parity` 35 / 35; `matlab_suite` 125 run, 124 passed, 0 failed, 1 allowed; it first
  needed the MEX to share MATLAB's libomp — two runtimes aborted MATLAB, "OMP: Error #15", since before today)* MATLAB: `/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('bindings/matlab','build/bin','build/bindings/matlab'); r=runtests('tests/matlab/test_contract_parity.m'); assertSuccess(r)"` after a `-DDTWC_BUILD_MATLAB=ON` build, then `ctest -R matlab_suite` (V-12 and V-14's commands too) |
| V-16 | FX-17 | CUDA: with `four.csv` rows `0,1,2,3` / `1,2,3,4` / `5,6,7,8` / `6,7,8,9`, `dtwc_cl -i four.csv -k 2 -o o1 --dtype float32 --device cuda` exits 1 naming `precision = Float32`; the same with `--mmap-threshold 0 -o o2` too; `--dtype float64 --device cuda -o o3` exits 0 |
| V-17 | IF-2 S3 | CUDA: `ctest --test-dir build/cuda-verify -j1 -R "test_run_resolution|test_cli_device_matrix"` → `gpu=cuda device=present cells=24/24`; Arrow: `ctest -R "fast_clara_parquet|fast_clara_assignment|io_readers"` markers unchanged; a build with `-DDTWC_ENABLE_METAL=OFF -DDTWC_ENABLE_YAML=OFF`: `ctest -R "device_matrix|config_formats|config_spellings|run_resolution"`; Windows: `test_tier1_cpp_api "[unicode]"` |
| V-9 | F42, F43 | two crashes recorded on the Windows box and never re-run: the default `CUDAPrecision::Auto` through the public `Problem` + CUDA route access-violates (F42); an llfio-ON MEX under MATLAB R2024b crashes in `std::mutex` (F43). Reproduce each; a reproduction becomes an `FX` item `[inferred — from the 2026-09-21 archive]` |

## 5. Dropped on 2026-09-23 — reopen only with a user-visible reason

| What | Why |
| --- | --- |
| A-10's load-time guard; D-22 (widening the index type in three languages — proposed in an uncommitted draft, never committed); W1's `narrow<int>` / `checked_cast` | unreachable limits — §1.1 |
| The oracle *concept*, `SeriesSource`, X-13 `Data<T>` | one implementation each; a templated `Data` would push the element type into the frozen `Problem`, so the 24 `is_f32()` branches stay |
| The virtual filler Strategy, its factory, a "fill plan" value, an `ExecutionTarget` type, `capabilities` reading a factory | one `fill()` holding the `#ifdef`s does the same (PF-4) |
| `Config`'s `schema` key and alias table; one `validate()` per option struct as doctrine | there is no second schema; CLI11 maps the flags |
| The C ABI | no consumer; R / Julia bindings are a killed idea |
| R2 derivations as gates (D4–D6, D8–D19) | D7 stays because it decides FX-2; the rest become docs pages if and when the docs need them |
| R3's 21 review lenses, the F-number bookkeeping, the ledger as a work list | one adversarial review at the release gate (RL-3) |
| The W7 test campaign (T-02…T-21, except T-20, which is in GT-1) | X-07 and X-08 — tests that prove a route ran and a heuristic is near-optimal — ride with IF-4 and PF-1; duplicates are deleted when their file is touched anyway |
| The configure-time layer manifest, strict-per-layer mode, foreign-type-in-signature analysis | `repo_map.py layers` stays, as a report |
| X-21 (`env()` grep gate, nothrow-move `static_assert`) | FX-4 fixes S-13 directly; IF-2 removes the `settings::paths` globals |
| The per-item ritual: registered band, FALSIFIED record, run-log, essay rows | bands stay for performance claims only |
| The capacity model (± 20 % RSS at N ≥ 500k) in the release gate; R5's PMU artefact (V-5) | the libpfm4 option stays built; nothing waits on it |
| `test_api.hpp` → `capabilities.hpp` (O-18); B-15's tidy / format / warnings jobs | "cleaner" is not a reason |
| `DECISIONS.md` rule 14 and D-17 ("no further library without ledger evidence") | contradict rule 18: prefer a mature permissive library |

## 6. Decisions for Volkan

| # | Decision | Recommendation |
| --- | --- | --- |
| D-B | *(implemented 2026-09-23 as recommended)* FX-6 text readers: vendor fast_float and fix the hand-written reader; raise the macOS wheel target to 13.3; ignore trailing blank lines, reject an interior one, and reject empty series at `set_data` — which ends the old contract that a blank line round-trips an empty series (no clustering can use one) | yes to all three — the audit ran every library against our input and each was worse; fast_float is the one library that fixes the portability failure |
| D-E | FX-13: a Metal `lb_envelope_band` narrower than the DTW window throws on every route, including regtile / banded-row where LB never runs, so the same call does not succeed or fail by series length | keep — an argument that makes the bound unsound is an error wherever it is given |
| D-11 | Tag after RL-3, with every ★ item in | yes |
| D-13 | Python floor `>= 3.9` or `>= 3.10` | 3.10 (numpy 2 needs it) |
| D-14 | One estimator: `DTWClustering`, folding `DTWCKMedoids` | yes |
| D-4 | `.claude/reports/test_kasper_analysis/REPORT.md` is headed "PRIVATE — do not push to GitHub" yet is tracked and pushed on `origin/Claude` and `origin/design-2.0` | yours alone; no gate reads it since GT-2, so removing it is a plain `git rm` — but that does not remove it from pushed history |
| D-15 | MATLAB on macOS: the MEX now shares MATLAB's own libomp (two runtimes aborted MATLAB), so `maxNumCompThreads(n)` also caps DTWC++'s threads | keep — MATLAB's thread setting governing a MEX is what MATLAB users expect; the alternative crashed |
| D-6 | Medium-confidence deletions (list in `git show e784e5c:.claude/PLAN.md`, §7 D-6) | delete all but `examples/cpp/example_new_features.cpp` (register it) |

Decided 2026-09-24 by the main session, overturnable: the ten IF-2 design questions (`plans/2026-09-24-if2-config-design.md` §10 —
a new `ClusterMethod` type, `Problem::set_metric`, keywords parsed by CLI11 in the library, the CLI's storage following the device,
no hpc submission from `dtwc_cl`, `settings::paths` removed pre-tag, version skew accepted); `cuda` ≡ `gpu` in Python per the
contract. Decided 2026-09-23 by Volkan: no Highway (was D-A — PF-5 is plain C++); the 1.x output filenames stay (was D-D);
the Eigen gap is investigated rather than accepted (was D-C — PF-6). Decided earlier and still standing: D-10 (C++ `hpc` is CLI / Python only), D-12 (infeasible band is an error by
default), D-16 (`Pruned` is a diagnostic), D-18 (llfio stays), D-19 (tolerance, not a pinned libm), D-21
(delete the unused solver types), D-22 (**not** widening — §1.1; the row existed only in an uncommitted draft).

## 7. Records

| What | Where |
| --- | --- |
| Charter, map, design, plan, decisions | `.claude/{CHARTER,MAP,design,PLAN,DECISIONS}.md` |
| Row detail (frozen reference) | `specs/2026-09-07-diff-ledger.md` |
| Performance measurements | `baselines/YYYY-MM-DD-<topic>.md` |
| Session state | `summaries/handoff-YYYY-MM-DD-<topic>.md` via the `session-handoff` skill; keep the last few |
| Lessons, citations | `LESSONS.md`, `CITATIONS.md` (append; no longer gate-pinned) |
| Anything superseded | delete it; git keeps it |
