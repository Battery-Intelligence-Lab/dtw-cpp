# Tier-1 side effects, `Result::distance_matrix`, `lr_max_nodes` width, contract clauses

Date: 2026-09-02 · Branch `Claude` · Gate build `build/highs-1151`

## 1. What wrote where (traced cause)

`Problem::cluster()` → `cluster_by_kmedoids_lloyd()` → `cluster_by_kmedoids_lloyd_impl(true)`.
`true` = "persist artifacts", so every Tier-1 `kmedoids` call ran:

- `cluster_by_kMedoidsLloyd_single()` → `writeMedoids()` → `output_folder_ / (name_ + "medoids_rep_<r>.csv")`
- end of the repetition loop → `writeBestRep()` → `output_folder_ / (name_ + "_bestRepetition_Nc_<k>.csv")`

`output_folder_` defaults to `settings::paths::results` = `"./results/"` (`Problem.hpp:191`),
CWD-relative, and no writer created it — `open_output`/`writeMedoids` only checked
`ofstream::good()`. **[confirmed]** by `unit_test_Problem_phase0.cpp:54-74`, whose
pre-existing band was exactly "`cluster_by_kmedoids_lloyd()` throws `std::runtime_error`
when `output_folder_` cannot be opened".

**[confirmed]** by artifact: the repo-root `results/` (gitignored) contains
`dataset_bestRepetition_Nc_2.csv`, `datasetmedoids_rep_0.csv`
(`dataset` = the default Tier-1 `Dataset` name) alongside `benders_test*`, `fast_pam_test*` —
unit tests running with the source root as CWD had been writing there.
After the fix, `ctest -R unit_test_clustering_algorithms` adds **no new files** to `results/`.

Same route also printed unconditionally to stdout: the Lloyd loop
(`Problem.cpp:1322/1325/1334/1336/1345/1354/1380-1391/1410`, including `print_clusters()`)
and Benders (`benders.cpp:85/159/327/428`). `mip_Highs.cpp` / `mip_Gurobi.cpp` were already
gated on `mip_settings.verbose_solver || prob.verbose()`; `lagrangian_root.cpp` (LRCore) and
`algorithms/tadpole.cpp` print nothing and write nothing. Automatic checkpointing is
opt-in (`checkpoint.enabled = false`), so it is not a `cluster()` side effect.

## 2. Diff by file

**`dtwc/Problem.hpp`** — new private `bool persist_run_artifacts_{ false }`;
`MIPSettings::lr_max_nodes` `long` → `std::int64_t` (default `2000000` unchanged);
`<cstdint>` comment.

**`dtwc/Problem.cpp`** — `cluster_by_kmedoids_lloyd()` forwards `persist_run_artifacts_`
instead of hard-coded `true`; `cluster_and_process()` sets it for the duration of its
`cluster()` call through a scope guard, so it keeps writing exactly what it wrote before.
All eight Lloyd progress prints (content unchanged) gated on `verbose_`, including
`print_clusters()` inside the iteration loop.

**`dtwc/Problem_IO.cpp`** — new `ensure_output_directory()` (`fs::create_directories`,
throws `std::runtime_error` if the directory genuinely cannot be made) called from
`open_output()` (covers `write_clusters`, `write_silhouettes`, `write_medoid_members`,
`writeBestRep`), from `writeMedoids`, and from `write_distance_matrix` (both the
`io::write_csv` and the mmap branch).

**`dtwc/mip/benders.cpp`** — one `verbose_log = mip_settings.verbose_solver || prob.verbose()`;
the three progress lines (warm start, converged, complete) and the per-iteration line use it.
The `invalid problem size` guard is an *error* path, so it moved to `std::cerr` rather than
behind `verbose()` — gating it would have made a non-clustering early return silent.

**`dtwc/api.hpp` / `dtwc/api.cpp`** — `std::vector<double> Result::distance_matrix() const`:
dense row-major N×N, `fill_distance_matrix()` first exactly as `score()` does via
`scores::*`. Smallest surface consistent with Python's `Result.distance_matrix`
(dense N×N); no `problem()` accessor added.

**`dtwc/mip/lagrangian_root.{hpp,cpp}`** — `LagrangianParams::max_nodes` → `std::int64_t`
(agrees with `lr_max_nodes`); B&B counter `long nodes` → `std::int64_t`; the node-cap
`fprintf` `%ld` → `%lld` with a `long long` cast.
`bindings/matlab/dtwc_mex.cpp` needed **no** edit: `exact_int_from_double` returns `int`,
which widens to `int64_t` implicitly (MATLAB-settable range is unchanged and still
loudly range-checked).

**`dtwc/fileOperations.hpp` / `dtwc/DataLoader.hpp`** — new `path_to_utf8()`;
`load_folder` (`:368`) and `load_metadata_folder` (`DataLoader.hpp:553`) now emit UTF-8
series names instead of `path::string()`'s native narrow encoding.
`api.cpp::derive_name` was deliberately **left** on `path::string()` — see §5.

**`docs/api-contract-2.0.md`** (+ regenerated `docs/content/api/tier-1.md`) — §1.2 `skip_rows`
C++ cell gained the directory clause; §1.4 gained a `distance_matrix` row.

**Tests** — `tests/unit/test_tier1_cpp_api.cpp` (+3 cases), `unit_test_fileOperations.cpp`
(+1 case), `unit_test_mip.cpp` and `unit_test_Problem_phase0.cpp` retargeted to
`cluster_and_process()` (the entry point that now owns the artifacts).

## 3. Tests and results

New:

1. `[api][tier1][sideeffects]` — `ScopedWorkingDirectory` RAII moves the CWD to a fresh
   temp dir, runs `dtwc::cluster(load(series), 2, m)` for `kmedoids`, `lrcore`, `tadpole`,
   asserts each returns 4 labels / 2 medoids and `fs::is_empty(sandbox)` both inside and
   after the scope.
2. `[api][tier1][sideeffects][verbose]` — `std::cout.rdbuf` redirect (RAII) around a
   non-verbose `Problem::cluster()` for `Kmedoids`, `LRCore`, `TADPole`: captured string
   empty; the verbose `Kmedoids` run non-empty; medoids and labels identical either way.
3. `[api][tier1][result]` — `Result::distance_matrix()` size `n*n`, zero diagonal,
   symmetric, and equal element-by-element to an independently constructed
   `Problem::dist_by_ind`.
4. `[fileOperations][unicode]` — a `café.csv` in a temp folder; `series_name(0)` and
   `load_metadata()`'s name must equal the bytes `63 61 66 C3 A9`.

Red-then-green **[confirmed]**: reverting only `fileOperations.hpp:368` to `.string()` and
rebuilding gives `"caf?" == "café" FAILED`; restoring makes it pass. A standalone probe on
this host prints `native(4): 63 61 66 E9` vs `u8(5): 63 61 66 C3 A9`, so the ACP here is
**not** UTF-8 and the test is not a false green.

Gates (all after `cmake --build build/highs-1151`, "no work to do" on the final tree):

| filter | result |
|---|---|
| `-R "tier1\|problem\|kmedoids\|mip\|lrcore\|tadpole\|io\|cluster\|Problem"` | **25/25 passed, 0 failed**, 1 skip (`test_io_readers`, expected Arrow-OFF) |
| `-R "mip\|lrcore\|guards"` | **3/3 passed** (`test_mip_backend_guards`, `test_cuda_launch_guards`, `unit_test_mip`) |
| `-R "...\|DataLoader\|Data\|fileOperations\|conformance\|cli"` (superset) | **32/32 passed, 0 failed**, same 1 skip |
| `-R "cpp_conformance\|unit_test_cli\|test_cli\|test_error_taxonomy\|test_runtime_loudness\|unit_test_benders\|unit_test_clustering_algorithms\|unit_test_fast_pam\|unit_test_tadpole\|test_lagrangian_root\|unit_test_checkpoint"` | **17/17 passed** |
| `uv run python scripts/check_docs_contract.py` (after `generate_docs.py`) | `documentation contract checks passed` |

Subjects asserted to have RUN, not skipped: `test_tier1_cpp_api.exe` alone reports
**124 assertions in 14 test cases** (was 11 cases); `[sideeffects]` 18 assertions / 2 cases,
`[result]` 37 / 1, `[unicode]` 5 / 1.

## 4. Proposed CHANGELOG bullets (Unreleased)

- `Problem::cluster()` and every algorithm it dispatches to are now side-effect free: no file
  is written and nothing is printed unless `Problem::verbose()` is set. `cluster_and_process()`
  is unchanged and still writes the per-repetition medoids and best-repetition records.
- Every `Problem` output writer creates its output directory before opening a file, so the
  default CWD-relative `./results/` is created rather than assumed.
- Benders progress output is gated on `MIPSettings::verbose_solver` / `Problem::verbose()`;
  its invalid-problem-size guard now reports on stderr.
- Added `dtwc::Result::distance_matrix()` — dense row-major N×N distances, filled on demand
  like `score()` — matching Python's `Result.distance_matrix`.
- `MIPSettings::lr_max_nodes` and `mip::LagrangianParams::max_nodes` are `std::int64_t`, so
  the public node-cap range no longer differs between Windows and Linux.
- Series names loaded from a directory source are UTF-8 on every platform, so non-ASCII file
  stems no longer break the Python binding or the CLI's CSV output.
- Documented that `skip_rows` applies per file for a directory source (contract §1.2) and
  added `Result::distance_matrix` to contract §1.4.

## 5. Unresolved / residual

- **Dataset name is still native-encoded.** `api.cpp::derive_name` deliberately keeps
  `path::stem().string()`, because that name becomes a **filename component**
  (`Problem_IO` writers, `Result::save`) and `fs::path` built from a UTF-8 `std::string`
  is re-decoded as the ACP on Windows — converting it would trade a decode error for
  mojibake output filenames. A non-ASCII *file* (not folder) source therefore still yields
  a name Python cannot decode. Fixing it properly needs a `utf8_to_path` on the writer side;
  out of this task's scope, not attempted. **[open]**
- `write_medoid_members` / `writeBestRep` keep their own unconditional `std::cout`
  ("Best repetition: N") — they are only reachable from `cluster_and_process()` and the
  explicit public writers now, never from `cluster()`.
- The claim I most expect to be wrong: that no *other* consumer depended on
  `cluster_by_kmedoids_lloyd()` writing `medoids_rep_*.csv`. I found and updated the three
  that did (`unit_test_mip`, `unit_test_Problem_phase0`, `test_tier1_cpp_api`); MATLAB and
  Python suites are owned by other agents this session and were not run.
- Only `build/highs-1151` was exercised. `build/nollfio` and `build/arrow-pyarrow-23` were
  not rebuilt.

---

## 6. Follow-up (same session): UTF-8 names end to end — residual §5 CLOSED

The §5 residual ("dataset name is still native-encoded, because converting it would
trade a decode error for a mojibake output filename") is closed by fixing **both**
directions instead of neither.

### Changes

- **`dtwc/fileOperations.hpp`** — new `utf8_to_path(std::string_view)`, the inverse of
  `path_to_utf8`: `fs::path(std::u8string(...))`, so `path_to_utf8(utf8_to_path(s)) == s`
  and POSIX bytes pass through. It falls back to the native narrow interpretation when the
  input is **not** valid UTF-8, because MSVC's `char8_t` conversion *throws*
  (`std::system_error: No mapping for the Unicode character exists in the target
  multi-byte code page` — **[confirmed]** by a standalone probe). The only producer of such
  a name is a string that never went through a loader, i.e. a native-encoded `--name` from
  `argv` on Windows; without the fallback that would have become a hard failure on every
  CLI write.
- **`dtwc/api.cpp`** — `derive_name` now uses `path_to_utf8(path.stem())`, so a non-ASCII
  **file** source (not just a folder) yields a UTF-8 dataset name. `Result::save` builds its
  four output paths as `directory / utf8_to_path(base + "_labels.csv")` etc., replacing
  `std::filesystem::path(base.string() + ...)`.
- **`dtwc/Problem_IO.cpp`** — every place a name becomes a path component goes through
  `utf8_to_path`: `writeMedoids`, `write_clusters`, `write_silhouettes`,
  `write_medoid_members`, `write_distance_matrix`, `writeBestRep`.
- The CLI (`dtwc_cl.cpp`) was **not** changed: its `prob_name` comes from `--name` in `argv`,
  a different provenance (native narrow on Windows), and the `utf8_to_path` fallback keeps
  that case working unchanged.

### Tests

- `test_tier1_cpp_api.cpp` — new `[api][tier1][unicode]` with two sections:
  (a) a `café.csv` **file** source → `Dataset::name()` equals the bytes `caf C3 A9`;
  `Result::save` into a temp dir produces a file whose `path::u8string()` contains `café`,
  `<name>_labels.csv` exists under the UTF-8 name, and no entry carries the ACP-round-trip
  mojibake prefix `caf C3 83`;
  (b) a **directory** source (`café.csv` + `zeta.csv`) → the saved `folder_labels.csv`
  *content* contains the UTF-8 bytes, i.e. the CLI-visible name inside the CSV is UTF-8.
- `unit_test_fileOperations.cpp` — `utf8_to_path` round trip, ASCII identity, and the
  invalid-UTF-8 native fallback (`"caf\xe9.csv"` in, same bytes out).

Red-then-green **[confirmed]**, one revert at a time:

| reverted | failing assertion |
|---|---|
| `Result::save` labels path → `directory / (base + "_labels.csv")` | `fs::exists(out_dir / utf8_to_path(cafe + "_labels.csv"))` → `false` (plus the mojibake check) |
| `derive_name` → `path.stem().string()` | `dataset.name() == cafe` → `"caf\xE9" == "café"` |

Both restored; suite green again.

### Gates (rerun after the follow-up)

| filter | result |
|---|---|
| `-R "tier1\|problem\|kmedoids\|mip\|lrcore\|tadpole\|io\|cluster\|Problem\|DataLoader\|Data\|fileOperations\|conformance\|cli"` | **32/32 passed, 0 failed**, 1 skip (`test_io_readers`, expected Arrow-OFF) |
| `-R "mip\|lrcore\|guards"` | **3/3 passed** |
| `uv run python scripts/check_docs_contract.py` | `generated documentation is current` / `documentation contract checks passed` |

Subjects asserted to have RUN: `test_tier1_cpp_api.exe` alone reports **136 assertions in
15 test cases** (was 124/14 before the follow-up); `unit_test_fileOperations.exe [unicode]`
**8 assertions in 1 case**.

### Additional CHANGELOG bullet

- Non-ASCII dataset and series names are now UTF-8 end to end: loaders emit UTF-8 and every
  writer converts back with `utf8_to_path`, so `Result::save` and the `Problem` writers
  produce correctly named files instead of mojibake on Windows.

### Remaining

- `dtwc_cl.cpp`'s `--name` is still whatever `argv` supplied (ACP on Windows). Making it
  UTF-8 needs a `wmain`/`GetCommandLineW` boundary conversion — separate change, not
  attempted; the `utf8_to_path` fallback means the current behaviour is unchanged.
- Source literals in the two new tests use `\u00e9` / `\xc3\xa9` escapes, so they do not
  depend on the test file's own encoding.
