# DTWC++ audit — orchestration, data, IO, CLI, environment

2026-09-02 · branch `Claude` · read-only. Nothing modified, built, or run; all findings traced in source. ✔ = a sub-audit claim I re-verified directly.

## (A) Bugs

**A1 · High · `Problem_IO.cpp:207-222` — `read_distance_matrix` swallows every exception, so `--dist-matrix` reports false success.** Observed. Body is `try { … } catch (...) { std::cout << "Distance matrix could not be read! Continuing without matrix!"; }` — never rethrows. So the CLI guard at `dtwc_cl.cpp:1552-1559` is dead code and line 1554 still prints `"Loaded distance matrix from …"`. `dtwc_cl --dist-matrix typo.csv -v` prints a failure on **stdout**, then a success, then silently recomputes the whole O(N²) matrix. A truncated CSV also leaves the matrix partially populated. `stress_test_cli.sh:275-286` covers only the happy path and compares labels, which match either way. *Fix: propagate; the CLI already has the handler.*

**A2 · High · `dtwc_cl.cpp:1384, 1404` — Parquet/Arrow inputs parsed as CSV when the optional dep is OFF.** Observed ✔ (found independently twice). Both handlers sit *inside* `#ifdef DTWC_HAS_ARROW`/`DTWC_HAS_PARQUET` with no rejection outside them, so on the canonical `DTWC_ENABLE_ARROW=OFF` gate `dtwc_cl -i x.parquet` falls through to the CSV `DataLoader` (1412-1419) and dies on a numeric-parse error with no hint the build lacks Parquet. Same for a `.parquet` directory, which `require_ram_limit_is_applicable` (1269) additionally accepts as a `--ram-limit` target nothing can honour. This is LESSONS F9; the `.dtws` branch (1350-1354) shows the correct pattern.

**A3 · High · `io/parquet_reader.hpp:91-123`, `parquet_chunk_reader.hpp:51-76, 149-176` — Parquet nulls read as garbage.** Observed ✔ (`null_count`/`IsNull` appear zero times in both readers; `arrow_c_data.cpp:87-97` does check). A null cell yields a wrong series; a null element yields raw buffer bytes, straight into DTW distances. Untested.

**A4 · High · `checkpoint.cpp:883-884` — binary checkpoint save is non-atomic.** Observed ✔. Opened with `std::ios::trunc`; a crash mid-write truncates it and `--resume` then hard-fails (`dtwc_cl.cpp:1465`) with the previous good result already gone. The dense path *is* atomic (`replace_current`, `:432`).

**A5 · Medium · `Problem.cpp:581` — dense checkpoint identity hardcodes `MetricType::L1`.** Observed ✔. The CLI computes `cache_metric` (`dtwc_cl.cpp:1519`) but `configure_cli_distance_storage` discards it for the dense path (`(void)cache_metric;`, 425). A SquaredL2 run below `--mmap-threshold` writes the same fingerprint as an L1 run, and a later L1 run accepts the wrong matrix. Every other axis is fingerprinted; the 15-axis test at `unit_test_checkpoint_robustness.cpp:599` has no metric case.

**A6 · Medium · `dtwc_cl.cpp:892` — `--benders` is unvalidated.** Observed. The only string option with no `CheckedTransformer`. `Problem.cpp:1064` reads `benders == "on" || (benders == "auto" && N > 200)`, so `--benders ON`, `true`, or the typo `of` all silently mean *off* on a large MIP job. Documented as `auto|on|off` (`docs/…/cli.md:167`).

**A7 · Medium · `dtwc_cl.cpp:1645-1774` — the method dispatch chain has no terminal `else`.** Observed. An unhandled `method` — reachable from YAML, which bypasses CLI11's transformer (e.g. the CLI-valid alias `method: obp`; only `hclust` is re-mapped at 1044) — falls through every branch leaving `result` default-constructed. Lines 1777-1778 then **write a binary checkpoint of an empty result** before `write_labels_csv` throws. `Problem::cluster()` (`Problem.cpp:1036`) does it correctly with `default: throw`. Same class: unknown YAML `solver` silently keeps HiGHS (1545), unknown `linkage` silently becomes Average (1762).

**A8 · Medium · `dtwc_cl.cpp:715, 842-843` — `--column`, `--skip-rows`, `--skip-cols` accepted and silently ignored off their own format.** Observed. `parquet_column` is used only on Parquet paths, the skips only in the CSV branch (1415). The LESSONS-recorded class; `require_ram_limit_is_applicable` (216-226) is the in-repo template.

**A9 · Medium · `fileOperations.hpp:500-511` — locale-dependent `std::stod` with a silent skip.** Observed ✔. `std::stod(row[j])` inside `catch (const std::exception &) { /* Skip non-numeric values */ }`: under a comma-decimal locale `"1.5"` parses as `1`, `"1.5abc"` is accepted, and a bad field is dropped without diagnostic, silently shortening the series — contradicting the strict `from_chars` policy at `:152-161` in the same header. Cheapest fix: `readCSV`/`readTimeSeriesCSV`/`readCSVColumn` have **zero callers repo-wide** ✔; delete them and rapidcsv with them.

**A10 · Medium · directory iteration is unsorted and unfiltered** (`fileOperations.hpp:311`, `DataLoader.hpp:448, 519`). Series order — hence names, labels, medoids, the whole matrix — is filesystem-dependent, so results are not reproducible across machines; stray files are fed to `readFile`. `parquet_reader.hpp:185` and `dtwc_cl.cpp:1261` sort correctly.

**A11 · Medium · three inconsistent `Ndata` semantics** (`DataLoader.hpp:450` vs `fileOperations.hpp:322` vs `:373`). Observed. For `Ndata == 0`: `count()` returns 1, folder load returns **all** series, batch load returns 0. `Ndata < -1` is never rejected.

**A12 · Medium · `Data.hpp:52-57, 98` — precision and `ndim` preconditions unchecked.** Observed. `series()`/`series_f32()` return `p_vec[i]`/`p_spans_[i]` with no `is_f32()` guard although `size()` (`:44`) does branch, so `series()` on Float32 view data indexes an empty vector. No constructor rejects `ndim == 0`, making `series_flat_size(i) % ndim` a division by zero.

**A13 · Low-Med · `DataLoader.hpp:332-335` — extension match is case-sensitive.** `data.TSV` keeps the default `delim = ','` (`:245`). `path()` also unconditionally overwrites an explicit delimiter, so `loader.delimiter('\t').path("a.csv")` silently yields `','`.

**A14 · Low-Med · `DataLoader.hpp:52-70` — `StoragePolicy::Auto` is a no-op on Windows.** `available_ram_bytes()` returns 0 → threshold `SIZE_MAX` → mmap spill never triggers. macOS uses *total* RAM as a "free" proxy, so Auto under-spills.

**A15 · Latent · `mpi/mpi_distance_matrix.cpp:129` — `for (int idx = 0; idx < static_cast<int>(local_count); …)`** overflows above INT_MAX local pairs (N ≈ 65 k on one rank), silently yielding a zero matrix — the defect this file already fixed for `MPI_Allreduce`. Lines 135-139 also call `dtwBanded`/`dtwFull_L` directly, ignoring `--variant` and `missing_strategy` (LESSONS "silent dispatch"). **Latent:** `compute_distance_matrix_mpi` has no production caller ✔.

**A16 · Low · `main.cpp:20-21` violates non-negotiable #1** — `fs::path("data") / "dummy"` with "Run this from the project root directory"; README:188 also shows `dtwc_main ...` though `main()` takes no arguments. `env.cpp:86` names the wrong `.env` directory (actually `$DTWC_REPO_ROOT` **or the CWD**) and ignores `fs::current_path`'s `error_code`.

Lower value, all Observed: `checkpoint.cpp:615` `catch (...) { return false; }` makes "corrupt" indistinguishable from "absent"; generations are never pruned (`:396-415`); `save_checkpoint`/`load_checkpoint` take `std::string` not `fs::path` (`checkpoint.hpp:51,66`) unlike the binary API (`:91`), mangling non-ASCII Windows paths; `arrow_ipc_reader.hpp:173,179` use `assert` as the only bounds guard (gone under NDEBUG) and `:147` casts to `StringArray` untyped; `arrow_c_data.cpp:154` leaks the `ArrowArray` on a `get_next` failure and `:172` discards resolved names; `Problem_IO.cpp:89,122,141,199` and `dtwc_cl.cpp:1219` never check stream/`create_directories` results; `api.hpp:41-52` has no `skip_rows`, so Tier-1 C++ cannot read a headered CSV the CLI reads fine; `fileOperations.hpp:305,325` print unconditionally, so `api.cpp:145`'s `verbosity(0)` does not silence the loader.

## (B) Concurrency, locks and shared mutable state

The compute path is genuinely lock-free; the defects are in IO and instrumentation.

- **`Problem.cpp:686-703` — the only lock in matrix-fill orchestration** is `#pragma omp critical(distByInd_init)` guarding the lazy dense resize in `dist_by_ind`. A lock-free `needs_init` pre-check means steady-state lookups never enter it, but the pair is **double-checked locking with no atomic or fence**: the read at 686 races with the write at 695. Safety rests on a documented precondition (`Problem.cpp:670-672`), not an enforced invariant. Observed.
- **`Problem.cpp:301` `rebind_dtw_fn() const` mutates shared `mutable` state** (`dtw_fn_`, `dtw_fn_f32_`, `dtw_binding_owner_`) and is called inside that critical section; any parallel consumer that triggers a rebind races. Same status. Observed.
- **Pattern to preserve:** `Problem.cpp:265-296` precomputes `wdtw_weights_cache_` serially for every unique length so "the parallel DTW lambda never mutates the cache… no insertion after this point." That is the standard the two above do not meet.
- **`DataLoader.hpp:127-130` — non-atomic shared counter in the IO path.** Observed ✔. `static std::size_t counter = 0` with `counter++` in `default_series_cache_path()`: a race across concurrent `load_stored()` calls. Worse, the "unique" component is `reinterpret_cast<uintptr_t>(&counter)` — constant per process, so two processes collide on the same temp `.dtws` path.
- **`DataLoader.hpp:254, 477` — `static inline std::size_t s_bulk_read_invocations`**, incremented non-atomically from `load_heap`: test instrumentation in production code, racy under parallel loads.
- **Benign and correctly placed:** `checkpoint.cpp:377` `static std::atomic<uint64_t> sequence` (once per checkpoint); `env.cpp:264` `std::call_once`; `mpi_distance_matrix.cpp:120-141`, where each thread writes only its own index.

## (C) Duplication

- **Three schemas for the same four output artefacts:** `Problem_IO.cpp:34,83,110`, `dtwc_cl.cpp:561,584,611`, and `api.cpp:220-276` — different filenames and headers; only `api.cpp:223` creates the directory.
- **The distance-matrix CSV loop three times:** `core/matrix_io.hpp:100-115`, `:203-212`, `Problem_IO.cpp:161-180`.
- **Six near-identical Parquet extractors:** `parquet_reader.hpp:91-123,140-157`; `parquet_chunk_reader.hpp:51-76,113-129,149-176,184-199` — the `_f32` pair differs only in element type. **Three identical Arrow status checkers:** `parquet_reader.hpp:37`, `parquet_chunk_reader.hpp:43`, `arrow_ipc_reader.hpp:49`.
- **Two contradictory CSV number parsers in one header:** `fileOperations.hpp:131-168` vs `:499-512` (A9). **The row-skip/parse loop five times:** `fileOperations.hpp:250-263, 373-392`; `DataLoader.hpp:462-466, 501-509, 536-546` — `DataLoader.hpp:486`'s "mirrors load_batch_file's extraction EXACTLY" is a manual invariant that A11 shows has drifted.
- **YAML re-implements CLI11's transformers by hand** (`dtwc_cl.cpp:1033-1056` vs 721-905) and is out of sync (A7). `Env::threads()` (`env.cpp:361`) is byte-identical to `effective_max_threads()` (`:251`).

## (D) Simplifications / error-prone constructs

- **`run_cli_main` is ~1170 lines** (`dtwc_cl.cpp:691-1860`) mixing declaration, YAML loading, normalisation, validation, routing, dispatch and output — the structural cause of A6, A7 and F3, and why the YAML path has **no test at all**.
- **String-typed options that should be enums:** `method`, `solver`, `linkage`, `benders`, `dtype_str`, `gpu_precision`, `missing_strategy` stay `std::string` from parse to dispatch; `MIPSettings::benders` (`Problem.hpp:77`) reaches the library. `api.cpp:72-75` validates method strings, then `:331-359` re-parses them into `Method`.
- **Inconsistent validation:** `--n-clusters`/`--n-init` get `CLI::PositiveNumber`; `--max-iter`, `--dc`, `--numeric-focus`, `--mip-focus` (documented 0-3) get nothing; `--seed`'s `CLI::Range(0u, UINT_MAX)` is a no-op.
- `Problem_IO.cpp:158` — the `name_` parameter shadows the member. `load_folder(Tpath &)`/`load_batch_file(fs::path &)` (`fileOperations.hpp:303,353`) take non-const lvalue refs, so a temporary path will not compile; `DataLoader.hpp:267-276` accessors are non-`const`, which is why `Problem.hpp:284` takes `DataLoader &`. `MAX_NUMERIC_TOKEN_SIZE = 32` bounds the checkpoint reader (`checkpoint.cpp:57`) while the writer uses an unchecked 64-byte buffer (`:527`).

## (E) Dead / obsolete code

- **22 `[[deprecated]]` camelCase aliases in `Problem.hpp`** (34 across headers) with **zero non-test callers repo-wide** ✔; `design.md:51-52` says such helpers should not be preserved.
- **Zero callers ✔:** `readCSV`/`readTimeSeriesCSV`/`readCSVColumn` (`fileOperations.hpp:423,461,529`) — the sole reason for the rapidcsv include and the home of A9; `ParquetChunkReader::estimated_total_bytes()` (`:342`), `estimated_bytes_per_series()` (`:310`), `read_row_group(int)` (`:348`) — the first's "used by existing callers" comment describes a caller since removed from `fast_clara.cpp`, yet `estimated_bytes_per_series_` is still computed on every open.
- **Unreachable:** `dtwc_cl.cpp:1552-1559` (A1); both catches in `checkpoint.cpp:913-919`; `checkpoint.cpp:253` (`n > numeric_limits<size_t>::max()` on a `uint64_t`); `api.cpp:63-64, 107-109`. `CheckpointOptions` (`checkpoint.hpp:35-39`) has no C++ consumer. Unused includes: `fileOperations.hpp` `<cassert>`, `<chrono>`, `<sstream>`, `<string>` twice; `<span>` in both Parquet readers; `DataLoader.hpp:26`; `Data.hpp:16`. Only two TODOs remain in scope (`dtwc_cl.cpp:940`, `mpi_distance_matrix.cpp:96`); both accurate.

## (F) Missing / forgotten

- **F1 · The advertised checkpoint feature does not exist.** `checkpoint.hpp:5-8` promises resuming a partial matrix mid-`fill_distance_matrix()`; the only dense save is *after* clustering finishes (`dtwc_cl.cpp:1794`) and `save_interval` is inert. A crash during a multi-hour fill loses everything.
- **F2 · `design.md:62-64` "config-file loading is still a gap outside the CLI" is still open.** TOML/YAML live entirely in `dtwc_cl.cpp`; no shared settings representation for Python/MATLAB. Format autodetection is likewise CLI-only — `dtwc::load("x.parquet")` (`api.cpp:143-153`) always uses the text reader.
- **F3 · YAML gaps** (`dtwc_cl.cpp:950-992`): `column`, `skip-rows`, `skip-cols`, `dc`, `benders`, `batch-size`, `batch-weighting`, `checkpoint`, `dist-matrix`, `mmap-threshold` unmapped, and YAML unconditionally overrides explicit CLI flags (TODO at 940, contradicting the comment at 934). Both are documented at `configuration.md:74-95` — known debt; the consequent *validation* bypass (A7) is not.
- **F4 · Dictionary-encoded Parquet columns have no handler** (`parquet_schema.hpp:50-56`): auto-detect skips one and reports "No scalar/list Float32 or Float64 column found" for a file that plainly contains doubles.
- **F5 · No opt-out for the O(N²) distance-matrix CSV** (`dtwc_cl.cpp:1820-1825`) — at the default `--mmap-threshold 50000` a multi-GB unrequested file. **No NaN policy at load:** `parse_numeric_field` admits `"nan"`, but neither `Data` nor `DataLoader` records or validates a missing rate and neither calls `missing_utils.hpp`.
- **F6 · Coverage.** Nothing tests the YAML path, `--benders`, method fallthrough, Parquet nulls, non-double physical types, zero-row Parquet, dense `--checkpoint` CLI resume, checkpoint atomicity or generation growth, directory-order determinism, extension case, `ndim == 0`, cross-precision `series()`, or `.env` duplicate keys. `tests/unit/test_io_readers.cpp:33-38` degrades to a single `SKIP` under `ARROW=OFF` and is the only Arrow-dependent target with **no `FAIL_REGULAR_EXPRESSION`** (`tests/CMakeLists.txt:110-126` vs `:149-159`) — ctest scores it green while it asserts nothing.

## (G) Top 5 actions

1. **(S)** Stop the two false-success paths: let `read_distance_matrix` propagate (A1); add a terminal `else { throw }` plus rejection of unknown `solver`/`linkage` (A7). No perf impact — outside any parallel region.
2. **(S)** Move format and flag applicability **outside** the `#ifdef`s (A2, A8) on the `require_ram_limit_is_applicable` / `.dtws` pattern, and add the `FAIL_REGULAR_EXPRESSION` skip-guard to `test_io_readers` (F6) so the gate can see it.
3. **(S)** Fix the two racy statics and validate `--benders` (B, A6): `std::atomic` plus real entropy for `default_series_cache_path`; delete `s_bulk_read_invocations` from production; one `CheckedTransformer`. One relaxed atomic per *load call*, not per series. Also thread `cache_metric` into `distance_checkpoint_identity` (A5) with a 16th case at `unit_test_checkpoint_robustness.cpp:599`.
4. **(M)** Template the Parquet extractors and add the null check once (A3, C), then delete `readCSV`/`readTimeSeriesCSV`/`readCSVColumn` and rapidcsv (A9, E). Net negative lines. **Perf constraint: one `chunk->null_count()` read per chunk, never per element.**
5. **(M)** Split `run_cli_main` into parse → load → run → write over a `CliConfig` with enum-typed options (D). Single-threaded orchestration; no locks, no serialisation of compute.

**Perf-risky — flag before attempting.** *(a)* **F1's mid-fill interval checkpoint is the one genuinely risky item:** snapshotting a consistent N×N matrix from inside the parallel fill needs a barrier or a hot-path lock plus an fsync per interval. If implemented, do it between row-blocks with a lock-free double-buffer or an explicit serial phase — never a mutex inside the pair loop. *(b)* Hardening the `dist_by_ind` double-checked lock must not add an atomic load per lookup; enforcing the existing "prime serially" precondition is cheaper than adding synchronisation. *(c)* A4's temp-file-plus-rename and fsync are fine — checkpointing is already off the compute path — but must not migrate into the fill loop. *(d)* Sorting directory listings (A10) is one O(n log n) pass before the parallel load, not per series. *(e)* A12's precision guard must stay a predictable branch in a hot accessor, not a lock or virtual call.

Deferred (L): atomic + fsync checkpoint writes and generation pruning (A4); implementing or retracting the mid-fill checkpoint (F1); a shared config representation for the bindings (F2).
