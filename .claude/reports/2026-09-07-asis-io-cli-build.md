# AS-IS map — IO, CLI, and build system

Repo `C:\D\git\dtw-cpp`, branch `Claude`, HEAD `a31956e`. Read-only pass; nothing built or run.
Every claim is `[confirmed <path>:<line>]` (read directly) or `[inferred]`. Prior review verified and
extended: `.claude/reports/2026-09-02-review-io-cli.md` (92 lines, all sections).

Verdict on the prior review's IO/CLI items is carried in §7/§8 with fixed / open / changed status.
The CMake layer had no prior review and is mapped fresh.

---

## 1. Module map

### 1.1 IO

| File | Lines | Responsibility | Public entry points | Includes / dependencies |
|---|---|---|---|---|
| `dtwc/io/parquet_schema.hpp` | 157 | Shared Parquet column selection, null/range validation, one typed value copier. Whole file inside `#ifdef DTWC_HAS_PARQUET` [confirmed dtwc/io/parquet_schema.hpp:8,157] | `find_parquet_series_column`, `parquet_series_value_type`, `is_parquet_series_column`, `parquet_leaf_count`, `require_no_nulls`, `require_list_range`, `copy_arrow_numeric<T>` | `<arrow/api.h>`; std `<cstddef> <cstdint> <memory> <limits> <stdexcept> <string>` |
| `dtwc/io/parquet_reader.hpp` | 175 | Eager whole-file / whole-directory Parquet load into owning `Data`. `#ifdef DTWC_HAS_PARQUET` [confirmed :15,175] | `dtwc::io::load_parquet_file`, `load_parquet_directory` | `../Data.hpp`, `../settings.hpp`, `parquet_schema.hpp`, arrow api/io, `parquet/arrow/reader.h` |
| `dtwc/io/parquet_chunk_reader.hpp` | 564 | Row-group streaming reader + RAM-budget arithmetic for FastCLARA. `#ifdef DTWC_HAS_PARQUET` [confirmed :15,564] | class `ParquetChunkReader` (`num_row_groups`, `total_rows`, `is_list_layout`, `logical_series_count`, `row_group_rows`, `estimated_resident_bytes`, `estimated_materialization_peak_bytes`, `read_row_groups`, `read_row_groups_f32`, `read_rows`, `read_rows_f32`, `row_groups_per_batch`); detail `extract_series_from_column{,_f32,_as<T>}`, `extract_list_element`, `check_arrow_chunk` | as above + `parquet/file_reader.h`, `<numeric> <type_traits> <limits>` |
| `dtwc/io/arrow_ipc_reader.hpp` | 205 | mmap'd Arrow IPC (Feather v2) zero-copy span source. `#ifdef DTWC_HAS_ARROW` [confirmed :16,205] | class `ArrowIPCDataSource` (`open`, `size`, `ndim`, `series`, `series_flat_size`, `series_length`, `name`, `all_names`) | `../settings.hpp`, arrow api/io/ipc, `<cassert> <span>` |
| `dtwc/io/arrow_c_data.hpp` | 85 | Public surface for Arrow C Data / PyCapsule ingest. **No `#ifdef`** — always compiled | `dtwc::io::data_from_arrow`, `data_from_arrow_stream`, `release_arrow` | `../Data.hpp`; forward-declares `ArrowSchema/ArrowArray/ArrowArrayStream` [confirmed :40-44] |
| `dtwc/io/arrow_c_data.cpp` | 188 | Implementation over vendored nanoarrow | (same three) | `../error.hpp`, `nanoarrow/nanoarrow.h` |
| `dtwc/extern/nanoarrow/nanoarrow.h` | 4482 | Vendored Apache nanoarrow 0.8.0 amalgamation, header | — | Apache-2.0 [confirmed dtwc/extern/nanoarrow/nanoarrow.h:1-5]; `#define NANOARROW_NAMESPACE DtwcNanoarrow` [confirmed :30]; version [confirmed :24] |
| `dtwc/extern/nanoarrow/nanoarrow.c` | 4117 | Vendored implementation, compiled into `dtwc++` [confirmed dtwc/CMakeLists.txt:32] | — | Apache-2.0; include dir is `PRIVATE` [confirmed dtwc/CMakeLists.txt:92] |
| `dtwc/fileOperations.hpp` | 491 | CSV/TSV text parsing primitives, deterministic directory listing, folder/batch loaders, UTF-8 path round-trip | `path_to_utf8`, `utf8_to_path`, `ignoreBOM`, `sorted_directory_files`, `validate_ndata`, `ndata_wants_more`, `readFile<T>`, `load_folder<T>`, `load_batch_file<T>`, `LoadOptions`; detail `trim_ascii`, `split_fields`, `lower_ascii`, `equals_ascii_ci`, `parse_numeric_field<T>`, `parse_numeric_row<T>`, `parse_series_value_row<T>` | `settings.hpp` + 17 std headers |
| `dtwc/core/matrix_io.hpp` | 243 | Distance-matrix CSV write/read, Eigen expansion, ADL `operator<<` | `dtwc::io::write_csv`, `read_csv`, `to_full_matrix`; `dtwc::core::operator<<` (Dense + Mmap); detail `preflight_distance_matrix_csv`, `distance_matrix_csv_token`, `write_distance_matrix_csv_preflighted` | `../error.hpp`, `distance_matrix.hpp`, `mmap_distance_matrix.hpp`, `<Eigen/Core>` + 14 std |

There is **no HDF5 reader or writer in C++** [confirmed: `grep -rn "hdf5\|HDF5\|H5" dtwc/` returns
nothing; HDF5 lives only in `python/dtwcpp/convert.py` and `python/dtwcpp/io.py`]. The `.dtws` store is
`dtwc/core/mmap_data_store.hpp` (outside the line-by-line scope; its layout is mapped in §2).

### 1.2 CLI

| File | Lines | Responsibility | Entry points | Includes |
|---|---|---|---|---|
| `dtwc/dtwc_cl.cpp` | 1921 | The whole `dtwc_cl` binary: option declaration, validation, format routing, dispatch, output | `main` (:1910), `run_cli_main` (:764-1908); 14 file-static helpers; `cli_renames`/`format_deprecation_warning`/`canonical_flag_for` are exported for tests via `DTWC_CL_NO_MAIN` [confirmed :37-44,763,1921] | `dtwc.hpp`, `env.hpp`, `error.hpp`, `core/variant_validation.hpp`, conditional `core/mmap_data_store.hpp` / `io/arrow_ipc_reader.hpp` / `io/parquet_*.hpp`, `algorithms/detail/fast_clara_plan.hpp`, `CLI/CLI.hpp`, `cli/config_file.hpp` |
| `dtwc/cli/config_file.hpp` | 151 | `CLI::ConfigBase` subclass: sniffs TOML vs YAML, translates YAML into `CLI::ConfigItem`s | class `dtwc::cli::ConfigFile` (`from_config`) | `CLI/CLI.hpp`, `#ifdef DTWC_HAS_YAML` → `fkYAML/node.hpp` |
| `dtwc/main.cpp` | 41 | `dtwc_main` demo driver | `main()` — **takes no arguments** [confirmed :12] | `dtwc.hpp` |
| `examples/cpp/config.toml` | 118 | Example TOML config | — | — |
| `examples/cpp/config.yaml` | 92 | Example YAML config | — | — |

### 1.3 Build

| File | Lines | Responsibility |
|---|---|---|
| `CMakeLists.txt` | 419 | Version parsing from `VERSION`, 15 `option()`s, CUDA/Metal language enable, executables, install/CPack, configuration summary |
| `CMakePresets.json` | 147 | 6 configure / 3 build / 2 test presets |
| `cmake/ProjectOptions.cmake` | 118 | 13 `dtwc_*` developer options + `dtwc_setup_options/global_options/local_options` |
| `cmake/StandardProjectSettings.cmake` | 158 | Build-type default, FP model, arch tuning, reproducible-build flags, runtime output dirs |
| `cmake/Dependencies.cmake` | 453 | `dtwc_setup_dependencies()`: CPM fetch of CPMLicenses, Catch2, HiGHS, CLI11, fkYAML, Eigen, benchmark, llfio (+quickcpplib patch), Arrow/Parquet; MPI probe |
| `cmake/CPM.cmake` | 23 | Downloads CPM 0.42.1 at configure time with a pinned SHA256 |
| `cmake/CompilerWarnings.cmake` | 127 | MSVC / Clang / GCC / CUDA warning sets |
| `cmake/Sanitizers.cmake` | 87 | `-fsanitize=` / `/fsanitize=` wiring onto `dtwc_options` |
| `cmake/StaticAnalyzers.cmake` | 108 | cppcheck, clang-tidy, include-what-you-use macros |
| `cmake/Coverage.cmake` | 43 | `DTWC_ENABLE_COVERAGE` + `add_executable_with_coverage_and_test` (the sole test-registration macro) |
| `cmake/InterproceduralOptimization.cmake` | 9 | `check_ipo_supported` → `CMAKE_INTERPROCEDURAL_OPTIMIZATION ON`, else `SEND_ERROR` |
| `cmake/Cache.cmake` | 32 | ccache/sccache launcher |
| `cmake/FindGUROBI.cmake` | 172 | Gurobi discovery + `Gurobi::GurobiC/GurobiCXX` imported targets |
| `cmake/PreventInSourceBuilds.cmake` | 18 | Fatal on in-source build |
| `cmake/Utilities.cmake` | 139 | vcvarsall string helpers, `get_all_targets` |
| `cmake/VCEnvironment.cmake` | 71 | `run_vcvarsall()` |
| `dtwc/CMakeLists.txt` | 257 | `dtwc++` static lib, header-lint glob, OpenMP policy, optional-dep linkage |
| `tests/CMakeLists.txt` | 945 | Recursive `*.cpp` glob → one CTest per file, plus ~13 per-test gate blocks and 6 script-driven integration tests |
| `benchmarks/CMakeLists.txt` | 63 | 8 benchmark executables |
| `python/CMakeLists.txt` | 45 | nanobind discovery + `_dtwcpp_core` module |
| `.clang-tidy` | 138 | 84-check conservative set, `WarningsAsErrors: ''`, `HeaderFilterRegex '^(dtwc|tests)/.*\.(hpp|h)$'` |
| `.clang-format` | 96 | Custom brace style, 2-space, `AllowShortIfStatementsOnASingleLine: true` |
| `docs/Doxyfile` | 2769 | Doxygen settings (see §4.8) |

---

## 2. Data flow

### 2.1 Inputs — file to `Data` / `Problem`

Format is classified by **extension and `fs::is_directory` only**, before any reader is touched
[confirmed dtwc/dtwc_cl.cpp:1243-1289]. Extensions are lower-cased first [confirmed :1245-1246], so
`.CSV`/`.PARQUET` classify correctly (this covers the CLI half of prior-review A13; `DataLoader`'s own
extension match is a separate file, outside this scope).

| Format | Detected by | Parser | Number parser | Row/col skipping | Delimiter | Precision | Error behaviour |
|---|---|---|---|---|---|---|---|
| **CSV / TSV file or folder** (default fallthrough) | `text_input = !parquet && !arrow_ipc && !dtws` [confirmed :1289] | `dtwc::DataLoader` (`dl.start_column(skip_cols).start_row(skip_rows)`) [confirmed :1439-1441] → `fileOperations.hpp` `parse_numeric_row`/`parse_series_value_row` | `std::from_chars`, `chars_format::general`; leading `+` stripped; case-insensitive `"nan"` → quiet NaN; any other non-finite rejected [confirmed dtwc/fileOperations.hpp:176-212] | `start_row` skip loop [confirmed :294-295]; `start_col` as the first field index [confirmed :225-234] | `LoadOptions::delimiter`, default `','`; `' '` means "any whitespace run" [confirmed :113-126,368] | float64 (`data_t`), converted to f32 afterwards if `--dtype float32` [confirmed dtwc_cl.cpp:1460-1465] | `std::runtime_error` naming file, row, column and a 64-char-truncated token [confirmed fileOperations.hpp:163-174] |
| **Parquet single file** | `.parquet`/`.pq`, not a directory [confirmed :1279-1280] | `io::load_parquet_file` (eager) [confirmed :1431] or `ParquetChunkReader` (streaming) [confirmed :1311,1336-1341] | Arrow decode; `copy_arrow_numeric<T>` casts Float64/Float32 → `T` [confirmed parquet_schema.hpp:136-153] | n/a — `--skip-rows`/`--skip-cols` are **rejected** for Parquet [confirmed dtwc_cl.cpp:266-269] | n/a | source Float32 or Float64 only; destination `data_t` or `float` | `std::runtime_error` on nulls, out-of-range list offsets, wrong value type, negative metadata, row-count mismatch [confirmed parquet_schema.hpp:107-127; parquet_chunk_reader.hpp:182-217] |
| **Parquet directory** | any `.parquet`/`.pq` entry in the directory [confirmed :1268-1282] | `io::load_parquet_directory` → per-file `load_parquet_file`, concatenated [confirmed parquet_reader.hpp:142-171] | as above | n/a | n/a | as above | `runtime_error("No .parquet files found in …")` [confirmed parquet_reader.hpp:157] |
| **Arrow IPC** | `.arrow`/`.ipc`/`.feather` [confirmed :1284-1285] | `io::ArrowIPCDataSource::open` (mmap) then copied into `Data` [confirmed :1411-1423] | direct `double*` read from the mapped values buffer [confirmed arrow_ipc_reader.hpp:125,171-176] | n/a | n/a | **Float64 only** — Float32 rejected by name [confirmed arrow_ipc_reader.hpp:119-122] | `runtime_error` for missing `data` column, >1 chunk, wrong list type, non-Float64 values, non-monotone/out-of-bounds offsets, `ndim == 0` [confirmed :86-162] |
| **`.dtws` mmap store** | `.dtws` [confirmed :1286] | `core::MmapDataStore::open`, then **copied** into `Data` [confirmed :1382-1391] | raw `double` from the mapping | n/a | n/a | Float64 only (`ELEM_SIZE = 8`) [confirmed dtwc/core/mmap_data_store.hpp:63,119-121] | `runtime_error` on short file, bad magic, version, endian marker, elem size, header CRC32 [confirmed :103-127] |
| **`.dtws` names sidecar** | `<input>.names`, one name per line | plain `std::getline` loop [confirmed dtwc_cl.cpp:1394-1401] | — | — | newline | — | **none** — a short or absent-line file silently leaves later names empty (see §7 P8) |
| **Arrow C Data / PyCapsule** | not CLI-reachable; binding-only | `io::data_from_arrow` / `data_from_arrow_stream` over nanoarrow [confirmed arrow_c_data.cpp:38,129] | `ArrowArrayViewGetDoubleUnsafe` **per element** [confirmed :100] | n/a | n/a | float32 or float64 source → `data_t` | `dtwc::InvalidInput` on null pointers, unsupported schema, null cell/element, names-count mismatch [confirmed :41-124] |
| **Distance-matrix CSV** (`--dist-matrix`) | flag value | `Problem::read_distance_matrix` → `io::read_csv` [confirmed Problem_IO.cpp:261-272; matrix_io.hpp:124-178] | `std::from_chars`, full-field consumption required [confirmed matrix_io.hpp:153-159]; empty field = uncomputed | strips a trailing `\r`; skips empty lines [confirmed :135-136] | `,` fixed | float64 | throws on unopenable file or a partially-parsed field; **the CLI catches it, warns and continues** [confirmed dtwc_cl.cpp:1584-1587] |
| **Directory checkpoint** (`--checkpoint`) | flag value | `dtwc::load_checkpoint(prob, dir, cache_metric)` [confirmed dtwc_cl.cpp:1605] | (checkpoint.cpp, outside scope) | — | — | — | returns bool; "no valid checkpoint" printed only under `-v` [confirmed :1608-1610] |
| **Binary result checkpoint** (`--resume`) | `<output>/<name>_checkpoint.bin` [confirmed :1484-1485] | `dtwc::load_binary_checkpoint` + `validate_cli_resume_result` [confirmed :1489-1499] | — | — | — | — | hard `InvalidInput` on unreadable file or any shape/label/medoid/iteration/cost violation [confirmed :546-594] |
| **`--config` TOML** | first non-blank non-`#` line starts `[`, or has `=` before `:` [confirmed config_file.hpp:55-67] | CLI11's own `ConfigBase::from_config` [confirmed :44-47] | CLI11 | — | — | — | unmapped key → error (`allow_config_extras(error)`) [confirmed dtwc_cl.cpp:780] |
| **`--config` YAML** | anything else | fkYAML `deserialize_docs` → `CLI::ConfigItem`s [confirmed config_file.hpp:125-146] | `scalar_text` renders floats at `max_digits10` [confirmed :87-92] | — | — | — | `CLI::ConfigError` for invalid YAML, ≠1 document, non-mapping root, a `null`/`~` value, or a non-scalar leaf; **and for the whole file when built without fkYAML** [confirmed :84-93,132-144] |

**`.h5` is not a CLI format.** README advertises HDF5 in the I/O bullet [confirmed README.md:41] but an
`.h5` path satisfies `text_input` [confirmed dtwc_cl.cpp:1289] and is fed to the CSV `DataLoader`
(:1439) — the same failure mode LESSONS F9 describes and that `require_input_format_is_built` fixed
for Parquet/Arrow. `docs/content/guides/data-formats.md:16` states the correct position (Python
conversion only).

### 2.2 The Parquet two-phase plan (metadata before payload)

1. Classify by filesystem alone, outside every `#ifdef` [confirmed :1264-1295].
2. If `--ram-limit > 0` or the method is `auto`/`clara`, open `ParquetChunkReader` for **metadata
   only** and sum `logical_series_count()` and `estimated_materialization_peak_bytes()`
   [confirmed :1301-1328].
3. `resolve_parquet_cli_plan` decides materialize vs stream, or throws a typed error naming the byte
   figures [confirmed :275-315].
4. Only then is the payload read (`load_parquet_file` / `load_parquet_directory` / streaming CLARA)
   [confirmed :1368-1435].

### 2.3 Outputs — artefact inventory

| Artefact | Writer | Path | Header / schema | Conditions |
|---|---|---|---|---|
| labels | `write_labels_csv` [confirmed dtwc_cl.cpp:634-654] | `<output>/<name>_labels.csv` [confirmed :1854] | `name,cluster` | always |
| medoids | `write_medoids_csv` [confirmed :657-681] | `<output>/<name>_medoids.csv` [confirmed :1862] | `cluster,medoid_index,medoid_name` | always |
| distance matrix | `Problem::write_distance_matrix` → `io::write_csv` or the preflighted emitter [confirmed Problem_IO.cpp:210-236] | `<output_folder>/<name>_distance_matrix.csv` [confirmed dtwc_cl.cpp:1869-1870] | headerless full N×N, `,`-separated, one `\n` per row, `to_chars` general at `max_digits10`, uncomputed = empty field [confirmed matrix_io.hpp:9-14,97-120] | only when `is_distance_matrix_filled()` [confirmed :1868] |
| silhouettes | `write_silhouettes_csv` [confirmed :684-698] | `<output>/<name>_silhouettes.csv` [confirmed :1879] | `name,cluster,silhouette` (`setprecision(8)`) | matrix filled **and** `n_clusters > 1` [confirmed :1876] |
| binary result checkpoint | `dtwc::save_binary_checkpoint` [confirmed :1826] | `<output>/<name>_checkpoint.bin` [confirmed :1484-1485] | frozen binary v1 | **always**, unless replaying — no flag required, not mentioned in `--help` |
| directory checkpoint | `dtwc::save_checkpoint` [confirmed :1842] | `<--checkpoint>/…` | (checkpoint.cpp) | only with `--checkpoint`; failure downgraded to a warning [confirmed :1845-1847] |
| mmap distance cache | `Problem::use_mmap_distance_matrix` [confirmed :530] | `<output>/<name>_distmat.cache` [confirmed :1547-1548] | fingerprinted mmap matrix | method not clara/onebatch and `prob.size() >= --mmap-threshold` |
| Tier-1 API artefacts | `Result::save` [confirmed dtwc/api.cpp:251-320] | `<dir>/<base>_{labels,medoids,distance_matrix,silhouettes}.csv` [confirmed :263-266] | identical four headers to the CLI | Tier-1 only |
| legacy `Problem` artefacts | `writeMedoids`, `write_clusters`, `write_silhouettes`, `write_medoid_members`, `writeBestRep` [confirmed Problem_IO.cpp:72,124,158,187,242] | `<output_folder>/<name>medoids_rep_N.csv`, `<name>_Nc_K.csv`, `<name>_silhouettes_Nc_K.csv`, `medoidMembers_Nc_K_rep_R_iter_I.csv`, `<name>_bestRepetition_Nc_K.csv` | **different schemas** — `write_clusters` emits `Cluster centroids:`, `Data,its cluster`, and a trailing prose line `Procedure is completed with cost:` [confirmed :130-144] | library / `dtwc_main` path |
| config echo | none | — | — | the CLI never writes the resolved configuration back; `-v` prints it to stdout [confirmed :1165-1231] |

**Same artefact, more than one writer:** labels / medoids / silhouettes / distance-matrix have **two
byte-compatible writers** (`dtwc_cl.cpp:634-698` and `api.cpp:268-319`) and **one incompatible legacy
family** (`Problem_IO.cpp:72-236`). The distance-matrix CSV emit body exists in **three** places
(`matrix_io.hpp:106-116`, `matrix_io.hpp:213-222`, and — via `operator<<` — `api.cpp:294`).

---

## 3. CLI anatomy

### 3.1 `run_cli_main` phases

`run_cli_main` spans `dtwc_cl.cpp:764-1908` — **1145 lines in one function**.

| Phase | Lines | What happens |
|---|---|---|
| App + config declaration | 766-780 | `CLI::App`, `--version`, `set_config("--config")`, `config_formatter(ConfigFile)`, `allow_config_extras(error)` |
| Option declaration | 782-999 | 48 `add_option`/`add_flag` calls in 11 comment-delimited groups |
| Zero-arg help | 1001-1006 | prints help, returns `EXIT_SUCCESS` |
| Parse (config file merged by CLI11) | 1008 | `CLI11_PARSE` |
| Deprecation handling | 1010-1022 | `--clusters`, `--restart` → stderr warning; canonical flag wins |
| Post-parse validation | 1024-1036 | `--input` required; `n_clusters >= 1`; `n_init >= 1` |
| Normalisation | 1038-1065 | lower-case 6 selectors + 2 precision strings, then re-apply the alias maps by hand |
| Device / route / distance validation | 1067-1136 | `parse_device`, `validate_cli_route_selectors`, `validate_cli_distance_configuration`, `validate_variant_params`, `parse_ram_limit` |
| CLARA option build | 1139-1148 | `CLARAOptions`, `validate_clara_controls` |
| Env device | 1150-1160 | `dtwc::env().set_device(device)` |
| Verbose banner + diagnostics | 1162-1231 | config echo, OpenMP/SLURM/`/proc` probes |
| Output dir + `Problem` | 1233-1241 | `create_directories`, `StoragePolicy::Heap` |
| Format classification + applicability guards | 1243-1295 | see §2.1 |
| Parquet metadata plan | 1297-1366 | inside `#ifdef DTWC_HAS_PARQUET` |
| Payload load (the `#ifdef`-spliced if/else chain) | 1368-1444 | Parquet dir / `.dtws` / Arrow IPC / Parquet file / CSV |
| Precision conversion | 1459-1465 | `convert_to_f32` |
| Auto method | 1470-1482 | `resolve_cli_auto_method`; matrix-free classification |
| Resume | 1484-1502 | binary checkpoint load + validate |
| Problem configuration | 1504-1539 | band, iterations, seed, MIP settings, GPU, missing strategy, variant |
| Distance storage | 1541-1576 | `configure_cli_distance_storage`, then the solver enum with a terminal `else throw` |
| `--dist-matrix` import | 1578-1588 | catch → warning → continue |
| Checkpoint wiring | 1590-1611 | interval + directory load |
| CUDA matrix | 1613-1672 | rejects matrix-free + CUDA; fills via `compute_distance_matrix_cuda` |
| Dispatch | 1674-1822 | 8 branches + terminal `else throw` |
| Output | 1824-1907 | binary checkpoint, scoring state, directory checkpoint, 4 artefacts, summary |

### 3.2 Option inventory

50 registrations total: `--version`, `--config`, and 48 `add_option`/`add_flag` calls (2 of which are
deprecated hidden aliases). Column "S→E" marks a **string-typed option later re-parsed into an enum or
an if/else chain**.

| Flag(s) | Type | Validator / transformer | Default | Consumer | S→E |
|---|---|---|---|---|---|
| `--version` (:767) | flag | — | — | CLI11 | |
| `--config` (:776) | path | `ConfigFile` formatter; extras = error | "" | CLI11 | |
| `-i,--input` (:787) | string | **none** | "" | required check :1025; classification :1243 | |
| `-o,--output` (:788) | string | none | `./results` | `create_directories` :1234 | |
| `--name` (:789) | string | none | `dtwc` | output filenames | |
| `--column` (:790) | string | none | "" | Parquet readers; rejected off-Parquet :262-265 | |
| `--dtype,--data-precision,--data-type` (:793) | string | `CheckedTransformer`, 8 keys, ignore_case | `float64` | `convert_to_f32` :1460 | ✔ |
| `--ram-limit` (:803) | string | `parse_ram_limit` at :1132 | "" | Parquet plan | ✔ |
| `-k,--n-clusters` (:815) | int | `CLI::PositiveNumber` | 3 | everywhere | |
| `--clusters` (:822) | int | none | -1 | **deprecated**, hidden (`group("")`) | |
| `-m,--method` (:825) | string | `CheckedTransformer`, 12 keys | `auto` | `validate_cli_route_selectors` :433; dispatch :1686-1822 | ✔ |
| `-b,--band` (:835) | int | **none** | -1 | `prob.set_band` :1505 | |
| `--metric` (:836) | string | `CheckedTransformer`, 4 keys | `l1` | `cache_metric` :1544; CUDA :1632 | ✔ |
| `--variant` (:842) | string | `CheckedTransformer`, 8 keys | `standard` | `DTWVariantParams` :1093-1111 | ✔ |
| `--max-iter` (:849) | int | **none** | 100 | `prob.set_max_iter` :1506 | |
| `--n-init` (:850) | int | `PositiveNumber` (rechecked :1033) | 1 | `run_cli_pam` :460 | |
| `--dc` (:853) | double | **none** | -1.0 | `set_tadpole_dc` :1780 | |
| `--wdtw-g` (:862) | double | none (`validate_variant_params` :1122) | 0.05 | vparams | |
| `--adtw-penalty` (:863) | double | as above | 1.0 | vparams | |
| `--sdtw-gamma` (:864) | double | as above | 1.0 | vparams | |
| `--msm-c` (:865) | double | as above | 1.0 | vparams | |
| `--twe-nu` (:866) | double | as above | 0.001 | vparams | |
| `--twe-lambda` (:867) | double | as above | 1.0 | vparams | |
| `--mv-mode` (:869) | string | `CLI::IsMember`, 2 values | `dependent` | `MVMode` :1119 | ✔ |
| `--missing-strategy` (:872) | string | `CheckedTransformer`, 6 keys | `error` | `MissingStrategy` :1529-1536 | ✔ |
| `--sample-size` (:886) | int | none (`validate_clara_controls` :1146) | -1 | CLARA | |
| `--n-samples` (:887) | int | as above | 5 | CLARA | |
| `--seed` (:888) | unsigned | `CLI::Range(0u, UINT_MAX)` — **a no-op over the full unsigned range** | `DEFAULT_RANDOM_SEED` | seeds + MIP warm start | |
| `--batch-size` (:895) | int | none | -1 | OneBatchPAM | |
| `--batch-weighting` (:897) | string | `CheckedTransformer`, 4 keys | `nniw` | `OneBatchWeighting` :1706-1711 | ✔ |
| `--linkage` (:908) | string | `CheckedTransformer`, 3 keys | `average` | `validate_cli_route_selectors`; `Linkage` :1799-1806 | ✔ |
| `--skip-rows` (:917) | int | **none** (no non-negative check) | 0 | DataLoader :1440; rejected off-text :266-269 | |
| `--skip-cols` (:918) | int | **none** | 0 | as above | |
| `--dist-matrix` (:922) | string | none | "" | `read_distance_matrix` :1581 | |
| `--checkpoint` (:926) | string | none | "" | `load_checkpoint`/`save_checkpoint` | |
| `--checkpoint-interval` (:928) | int | **none**; requires `--checkpoint` at :1593-1597 | 0 | `prob.checkpoint.save_interval` | |
| `--resume` (:936) | flag | — | false | binary replay :1487 | |
| `--restart` (:942) | flag | — | false | **deprecated**, hidden | |
| `--mmap-threshold` (:945) | size_t | `CLI::NonNegativeNumber` | 50000 | `configure_cli_distance_storage` | |
| `--solver` (:950) | string | `CheckedTransformer`, 2 keys | `highs` | `Solver` enum :1571-1576 | ✔ |
| `--mip-gap` (:964) | double | **none** | 1e-5 | `mip_settings` | |
| `--time-limit` (:965) | int | **none** | -1 | `mip_settings` | |
| `--no-warm-start` (:966) | flag | — | false | `mip_settings.warm_start` | |
| `--numeric-focus` (:967) | int | **none**, documented 0-3 | 1 | `mip_settings` | |
| `--mip-focus` (:968) | int | **none**, documented 0-3 | 2 | `mip_settings` | |
| `--verbose-solver` (:969) | flag | — | false | `mip_settings` | |
| `--benders` (:975) | string | `CheckedTransformer`, 9 keys | `auto` | `mip_settings.benders` — reaches the library **as a string** [confirmed :1519] | ✔ |
| `-d,--device` (:987) | string | **none** at declaration; `parse_device` at :1071 | `cpu` | `DeviceSpec`, `env().set_device` | ✔ |
| `--gpu-precision,--gpu-dtype` (:988) | string | `CheckedTransformer`, 9 keys | `auto` | `cuda_settings.precision` :1525-1526 | ✔ |
| `-v,--verbose` (:999) | flag | — | false | everything | |

Twelve string-typed options are re-parsed downstream (✔). Nine options carry **no validator at all**
(`--input`, `--band`, `--max-iter`, `--dc`, `--mip-gap`, `--time-limit`, `--numeric-focus`,
`--mip-focus`, `--checkpoint-interval`, plus the two skip options) — `--numeric-focus`/`--mip-focus`
document a 0-3 range they do not enforce [confirmed :967-968].

### 3.3 Config-file precedence

`--config` is CLI11's own config mechanism, so **CLI11 owns precedence: a command-line value always
beats a file value** [confirmed dtwc_cl.cpp:770-776; class docstring config_file.hpp:5-8]. Keys are
the canonical long flags without `--`, identical in TOML and YAML. Deprecated keys (`clusters`,
`restart`) map onto the same hidden options and emit the same stderr warning [confirmed :1010-1022].
An unmapped key is a hard error [confirmed :780]. Format sniffing uses the first non-blank non-`#`
line [confirmed config_file.hpp:52-67]; an empty file is treated as empty TOML [:66]. Without
`DTWC_HAS_YAML` a YAML file is refused with a typed `CLI::ConfigError` — and the guard lives
**outside** the `#ifdef DTWC_HAS_YAML` block that holds the parser [confirmed :125-145].

This is the recorded remedy for a real bug: a hand-rolled YAML loader silently overrode explicit
flags [confirmed .claude/LESSONS.md:1317]. **DELIBERATE.**

**Coverage.** `test_cli_config_formats` runs in both flavours with a per-flavour pinned check count
(23 with YAML, 7 without) [confirmed tests/CMakeLists.txt:787-817], so the prior review's "the YAML
path has no test at all" (§D) is **fixed**. No CI job configures `-DDTWC_ENABLE_YAML=OFF`, so the
7-check flavour is never exercised on CI. `test_cli_rejects_yaml_config` pins that the removed
`--yaml-config` flag stays rejected, asserting the non-zero exit **and** the flag name rather than
CLI11 wording [confirmed :764-781].

### 3.4 Exit codes and error conventions

- `main` (:1910-1920) catches `std::exception` → `"Error: " << what()` on stderr → `EXIT_FAILURE`;
  `catch (...)` → `"Error: unknown non-standard exception"` → `EXIT_FAILURE`.
- Two styles coexist inside `run_cli_main`: **return `EXIT_FAILURE` after printing** (validation:
  :1026-1027, :1030, :1034, :1073, :1079, :1085, :1124, :1134, :1158, :1566, :1595, :1618, :1623,
  :1668) and **throw** (`InvalidInput` / `runtime_error`: :1254, :1345, :1350, :1354, :1490, :1498,
  :1576, :1806, :1820, plus everything inside the readers). Both end at `EXIT_FAILURE`; only the
  message prefix differs.
- Three failures are downgraded to **stderr warnings with the run continuing**: `--dist-matrix` load
  (:1584-1587), directory checkpoint save (:1845-1847), silhouette computation (:1889-1891).
- There is **no distinct exit code** for "bad usage" vs "runtime failure" vs "capability missing".

---

## 4. Build system anatomy

### 4.1 Option inventory (32 `option()` + 3 project cache STRINGs)

| Option | Default | Effect | Targets touched |
|---|---|---|---|
| `DTWC_BUILD_EXAMPLES` (CMakeLists.txt:28) | OFF | `add_subdirectory(examples/cpp)` :304 | examples |
| `DTWC_BUILD_TESTING` (:29) | OFF | `add_subdirectory(tests)` :300 + Catch2 fetch (Dependencies:18) | all test targets |
| `DTWC_BUILD_BENCHMARK` (:30) | OFF | benchmark fetch (Dependencies:128) + subdir :308 | 8 bench targets |
| `DTWC_BUILD_PYTHON` (:31) | OFF | `add_subdirectory(python)` :312; **also disables arch tuning** (StandardProjectSettings:91) | `_dtwcpp_core` |
| `DTWC_BUILD_MATLAB` (:32) | OFF | MEX subdir :316; **directory-scope `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` on MSVC** :68-70 | every target in the tree |
| `DTWC_DEV_MODE` (:33) | OFF | drives the defaults of warnings / WAE / clang-tidy / cppcheck / ccache | `dtwc_warnings`, `dtwc_options` |
| `DTWC_ENABLE_GUROBI` (:35) | **ON** | summary + warnings at root; the actual link is in `dtwc/mip` | mip-solvers |
| `DTWC_ENABLE_HIGHS` (:36) | **ON** | CPM HiGHS (Dependencies:32-66) | mip-solvers |
| `DTWC_HIGHS_GPU` (:42) | OFF | forces the `CUPDLP_GPU` cache var (Dependencies:37-41) | highs |
| `DTWC_ENABLE_MPI` (:43) | OFF | `find_package(MPI)`; adds `mpi/mpi_distance_matrix.cpp` + `DTWC_HAS_MPI` (dtwc/CMakeLists:205-210) | `dtwc++` |
| `DTWC_ENABLE_CUDA` (:44) | OFF | `enable_language(CUDA)` :165; `cuda/cuda_dtw.cu` + `DTWC_HAS_CUDA` (dtwc/CMakeLists:238-244) | `dtwc++` |
| `DTWC_ENABLE_METAL` (:47) | **ON** | Apple only; `metal/metal_dtw.mm` + `DTWC_HAS_METAL` (dtwc/CMakeLists:247-256) | `dtwc++` |
| `DTWC_ENABLE_ARROW` (:48) | OFF | Arrow/Parquet `find_package` or CPM; `DTWC_HAS_ARROW`/`DTWC_HAS_PARQUET` (dtwc/CMakeLists:220-235) | `dtwc++` |
| `DTWC_ENABLE_YAML` (:49) | **ON** | fkYAML fetch; `DTWC_HAS_YAML` on `dtwc_cl` only (CMakeLists:271-274) | `dtwc_cl` |
| `DTWC_ALLOW_SEQUENTIAL` (:57) | OFF | turns the missing-OpenMP `FATAL_ERROR` into a warning + `DTWC_SEQUENTIAL_BUILD` (dtwc/CMakeLists:174-194) | `dtwc++` |
| `DTWC_ENABLE_LLFIO` (Dependencies:152) | **ON** | llfio + quickcpplib superbuild; `DTWC_HAS_MMAP` (dtwc/CMakeLists:213-217, mip/CMakeLists:57) | `dtwc++`, `mip-solvers` |
| `DTWC_ENABLE_NATIVE_ARCH` (StandardProjectSettings:86) | ON | `-march=native` / `/arch:AVX2` on Release + RelWithDebInfo | all (directory scope) |
| `DTWC_REPRODUCIBLE_BUILD` (:131) | OFF | `-ffile-prefix-map` (Clang/GCC only) | all (directory scope) |
| `DTWC_ENABLE_COVERAGE` (Coverage.cmake:1) | OFF | `--coverage -O0` PUBLIC on every test target :34-42 | test targets |
| `dtwc_ENABLE_IPO` (ProjectOptions:21) | `${PROJECT_IS_TOP_LEVEL}` | `CMAKE_INTERPROCEDURAL_OPTIMIZATION ON`, else `SEND_ERROR` | global |
| `dtwc_ENABLE_COMPILER_WARNINGS` (:22) | `${DTWC_DEV_MODE}` | populates `dtwc_warnings` | `dtwc_warnings` |
| `dtwc_WARNINGS_AS_ERRORS` (:23) | `${DTWC_DEV_MODE}` | `-Werror` / `/WX` / tidy `-warnings-as-errors=*` | `dtwc_warnings`, tidy, cppcheck |
| `dtwc_ENABLE_SANITIZER_{ADDRESS,LEAK,UNDEFINED,THREAD,MEMORY}` (:24-28) | OFF ×5 | `-fsanitize=` on the `dtwc_options` INTERFACE | everything linking `project_options` |
| `dtwc_ENABLE_UNITY_BUILD` (:29) | OFF | `UNITY_BUILD` property on `dtwc_options` | — |
| `dtwc_ENABLE_CLANG_TIDY` (:30) | `${DTWC_DEV_MODE}` | `CMAKE_CXX_CLANG_TIDY` | global |
| `dtwc_ENABLE_CPPCHECK` (:31) | `${DTWC_DEV_MODE}` | `CMAKE_CXX_CPPCHECK` | global |
| `dtwc_ENABLE_PCH` (:32) | OFF | 3 PCH headers on `dtwc_options` | — |
| `dtwc_ENABLE_CACHE` (:33) | `${DTWC_DEV_MODE}` | ccache/sccache launcher | global |
| cache `DTWC_CUDA_ARCH_LIST` (CMakeLists:158) | `60;70;75;80;86;89;90` | seeds `CMAKE_CUDA_ARCHITECTURES` | CUDA |
| cache `DTWC_ARCH_LEVEL` (StandardProjectSettings:87) | `""` | `""`/`v3`/`v4` → native / `x86-64-v3` / `x86-64-v4` | all |
| cache `DTWC_MATLAB_SUITE_MIN_PASSED` (tests:922) | 121 | MATLAB pass floor | `matlab_suite` |

Thirteen options use the lower-case `dtwc_` prefix and nineteen the upper-case `DTWC_` prefix — an
inconsistency a user meets directly on the command line.

### 4.2 Dependency acquisition

All via CPM. `cmake/CPM.cmake` downloads CPM 0.42.1 at configure time with
`EXPECTED_HASH SHA256=f3a6dcc6…` [confirmed cmake/CPM.cmake:5-21] — so **configure requires network
access on a cold cache**.

| Package | Version / pin | Hash pinned | Licence | Header-only | Line |
|---|---|---|---|---|---|
| CPMLicenses.cmake | `VERSION 0.0.7` via GITHUB_REPOSITORY | **no** | MIT | n/a (CMake) | Dependencies:12-16 |
| Catch2 | v3.13.0 tarball | SHA256 ✔ | BSL-1.0 | no | :19-28 |
| HiGHS | v1.15.1 tarball | SHA256 ✔ | MIT | no | :42-57 |
| CLI11 | v2.6.2 tarball, `DOWNLOAD_ONLY` | SHA256 ✔ | BSD-3 | **yes** | :69-81 |
| fkYAML | v0.4.4 tarball, `DOWNLOAD_ONLY` | SHA256 ✔ | MIT | **yes** | :88-95 |
| Eigen | 5.0.1 tarball, `DOWNLOAD_ONLY` | SHA256 ✔ | MPL-2.0 | **yes** | :114-121 |
| google/benchmark | `VERSION 1.9.5` via GITHUB_REPOSITORY | **no** | Apache-2.0 | no | :130-138 |
| llfio | `GIT_TAG b17613fb2149…` (commit) | commit pin, no digest | Apache-2.0 / BSL-1.0 | no | :154-166 |
| quickcpplib | `git clone` + `checkout 3c1d8cb5…` | commit pin | BSL-1.0 | no | :193-236 |
| Apache Arrow + Parquet | apache-arrow-19.0.1 tarball | SHA256 ✔ | Apache-2.0 | no | :356-400 |
| nanobind | `VERSION 2.4.0` via GITHUB_REPOSITORY | **no** | BSD-3 | no | python/CMakeLists:18-23 |
| nanoarrow 0.8.0 | **vendored in-tree** | n/a | Apache-2.0 | no (`.c` compiled) | dtwc/extern/nanoarrow |

Three source-only pins (CPMLicenses, benchmark, nanobind) carry no digest.
`scripts/check_supply_chain_pins.py` (885 lines) enforces `URL_HASH SHA256=<64 hex>` only for
**archive URLs**, plus 40-hex SHA pins on every workflow `uses:`, an archive identity inventory, and a
tracked-CMake-manifest count of exactly 30. Two workflows run it (`documentation.yml:32`,
`python-tests.yml:15`).

**Absence surfacing.** Every optional dep except OpenMP degrades with a `message(WARNING)` plus an
explicit `set(<OPT> OFF PARENT_SCOPE)`: Arrow on Windows+Clang (Dependencies:346-353), Arrow CPM build
failure (:408-411), MPI not found (:442-447), llfio not found (:300-302), CUDA nvcc not found
(CMakeLists:209-211). Metal on non-Apple is **silently** disabled, documented as deliberate to avoid CI
noise [confirmed CMakeLists:45-46,222-225]. OpenMP is the one **hard requirement**:
`message(FATAL_ERROR)` unless `-DDTWC_ALLOW_SEQUENTIAL=ON` [confirmed dtwc/CMakeLists:184-194] —
recorded as an explicit no-silent-fallback decision [confirmed CMakeLists:51-57].

### 4.3 Compiler flags per configuration

- **Standard**: `CMAKE_CXX_STANDARD 20` globally [confirmed StandardProjectSettings:26] plus
  `target_compile_features(dtwc_options INTERFACE cxx_std_20)` [ProjectOptions:63];
  `CMAKE_CXX_EXTENSIONS OFF` [:23]; `CMAKE_EXPORT_COMPILE_COMMANDS ON` [:19].
- **FP model (Release + RelWithDebInfo only)** — explicitly **not** `-ffast-math`
  [confirmed StandardProjectSettings:45-72]:
  - GCC/Clang: `-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math
    -fno-signed-zeros -fno-rounding-math -fno-signaling-nans`; `-ffinite-math-only` deliberately
    omitted so `std::isnan()` stays valid [:63].
  - MSVC: `/fp:precise /fp:contract /Gy`.
  - `dtwc++` additionally carries PRIVATE `-fno-finite-math-only` to defend against a parent project
    using `-ffast-math` [confirmed dtwc/CMakeLists:196-200].
- **Arch tuning**: `-march=native` / `/arch:AVX2` (or `x86-64-v3`/`v4`, `/arch:AVX512`) on
  Release+RelWithDebInfo, disabled for Python wheels and for sub-project consumption
  [confirmed StandardProjectSettings:74-112].
- **Warnings**: MSVC `/W4 /permissive-` plus 17 specific `/w1…`; Clang 16 flags including
  `-Wconversion -Wsign-conversion -Wold-style-cast`; GCC = the Clang set + 5 more; CUDA via
  `-Xcompiler` [confirmed CompilerWarnings.cmake:13-100]. Attached as an INTERFACE on `dtwc_warnings`,
  **only when `dtwc_ENABLE_COMPILER_WARNINGS` is ON, i.e. only in `DTWC_DEV_MODE`**
  [confirmed ProjectOptions:65-74].
- **IPO/LTO**: on by default at top level; `check_ipo_supported` failure is `SEND_ERROR`, not a
  warning [confirmed InterproceduralOptimization.cmake:3-8].
- **Sanitizers**: `-fsanitize=<list>` compile+link on the `dtwc_options` INTERFACE (MSVC:
  `/fsanitize=address /Zi` + `_DISABLE_VECTOR_ANNOTATION`, requires the VS environment)
  [Sanitizers.cmake:64-85].
- **Coverage**: `--coverage -O0` PUBLIC on each test target; a non-GCC/Clang compiler is
  `FATAL_ERROR` [Coverage.cmake:34-42].

### 4.4 Test registration

One macro registers everything: `add_executable_with_coverage_and_test(TARGET_PATH [ARGN...])`
[confirmed cmake/Coverage.cmake:3-43]. It:
- derives the target name from the file stem and links `dtwc++ Catch2::Catch2WithMain project_options`
  — **not** `project_warnings` [:6];
- defines `DTWC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"` [:8];
- runs every test with `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` [:21,28,31] — tests are cwd-repo-root
  by construction;
- sets `SKIP_RETURN_CODE 4` on **every** test [:33] — CTest scores return code 4 as a pass.

`tests/CMakeLists.txt:1` globs `*.cpp` recursively with `CONFIGURE_DEPENDS` and turns each into one
CTest entry [confirmed :125-141]. Integration fixture programs deliberately use a `.cc` suffix to stay
out of that glob [confirmed :673-674, :723-724].

Because `SKIP_RETURN_CODE 4` is blanket, thirteen per-test blocks then **clear it** and add a
`FAIL_REGULAR_EXPRESSION` matching Catch2's skip line plus a `PASS_REGULAR_EXPRESSION` pinning an exact
subject marker and an assertion floor: `unit_test_checkpoint_binary` (:146-175),
`test_lb_keogh_derivation` (:181-193), `test_lb_enhanced_webb_derivation` (:198-212),
`test_lb_webb_intmax` (:217-230), `unit_test_DataLoader` (:236-246), `test_problem_api_2_0` (:251-277),
`unit_test_nearest_medoid_assignment` (:282-294), `unit_test_distance_matrix_csv` (:301-325),
`unit_test_problem_encapsulation` (:330-344), `unit_test_problem_storage_policy` (:349-378),
`unit_test_deterministic_series` (:381-397), `test_supply_chain_pinning` (:619-630), `test_io_readers`
(:643-654). Seven script-driven or external tests use `add_test` directly:
`test_distance_matrix_csv_contract` (:699-719, `LABELS integration;f14`), `test_cli_resume_state`
(:735-756, `integration;f17`), `test_cli_rejects_yaml_config` (:765-780, `cli`),
`test_cli_config_formats` (:797-816, `integration;cli`), `test_fast_clara_parquet_parity` (:833-849,
`integration;arrow;f8`), `test_fast_clara_assignment_contract` (:851-868, `integration;arrow;f13`),
`matlab_suite` (:934-940, `matlab`). Only those seven carry `LABELS`; the globbed unit tests carry
none. `TIMEOUT` appears on 9 tests (30-1800 s); `RUN_SERIAL TRUE` on 12.

Capability guards read the **target's** compile definitions rather than a CMake variable:
`get_target_property(... dtwc++ INTERFACE_COMPILE_DEFINITIONS)` then `IN_LIST` for `DTWC_HAS_ARROW`
(:642-643), `DTWC_HAS_PARQUET` (:822-823), `DTWC_HAS_MMAP` (:313, :363, :689); and `dtwc_cl`'s own
`COMPILE_DEFINITIONS` for `DTWC_HAS_YAML` (:788-789). The F8/F13 Parquet tests are **not registered at
all** without Parquet, so no skip can score green [confirmed :819-823]. On Windows the Arrow block
appends the Arrow/Parquet/pyarrow DLL directories to `PATH` for **every** test in the directory
[confirmed :899-907] — the LESSONS-recorded fix for `0xC0000135` before `main`.

Three F22 `OBJECT EXCLUDE_FROM_ALL` probe libraries (`f22_cpp_legacy_werror`,
`f22_cpp_legacy_suppressed`, `f22_cpp_canonical_werror`) are compiled with deliberately different
deprecation-diagnostic flags per compiler and driven by
`scripts/test_f22_cpp_deprecations.py` attached to the `test_problem_api_2_0` CTest entry
[confirmed :20-141].

### 4.5 Presets

`CMakePresets.json` v6, `cmakeMinimumRequired 3.26`. One hidden base (`default`: `binaryDir
${sourceDir}/build`, `DTWC_BUILD_TESTING=ON`) and 5 visible configure presets — `clang-win`,
`clang-win-debug`, `msvc`, `gcc-linux`, `clang-macos` — each gated on `hostSystemName`. 3 build presets
(`clang-win`, `clang-win-debug`, `clang-macos`) and 2 test presets (`clang-win`, `clang-macos`).
**No preset sets any `DTWC_ENABLE_*` value**, so preset builds inherit the defaults (HiGHS ON, Gurobi
ON, llfio ON, YAML ON, Arrow OFF). `gcc-linux` and `msvc` have no matching build or test preset.
`tests/CMakeLists.txt:399-617` is a configure-time assertion suite over this exact file (6 configure
presets, schema 6, floor 3.26.0, `clang-win` compiler `clang++`, parent `default`, `clang-win-debug`
reference, no Windows absolute paths).

### 4.6 CI matrix

10 workflows, 804 lines, **no `concurrency:` block and no scheduled trigger anywhere**.

| Workflow / job | Runner(s) | Build type | Notable flags | Tests |
|---|---|---|---|---|
| `ubuntu-unit.yml` `build-and-test` | ubuntu-latest × {gcc-11, gcc-12, clang-14/15/16/17, gcc-12 + ASan/UBSan} (7 legs) | Debug | sanitizer leg via raw `CMAKE_CXX_FLAGS="-fsanitize=address,undefined"` (:61), **not** the `dtwc_ENABLE_SANITIZER_*` options | `ctest -j2 -C Debug --output-on-failure` |
| `ubuntu-unit.yml` `arrow-and-test` | ubuntu-24.04 | Debug | `ARROW=ON`, `HIGHS=OFF`, `GUROBI=OFF`, `LLFIO=OFF` | full ctest, then `-R '^test_io_readers$'` + `assert-arrow-suite.sh`, then `-R '^test_fast_clara_parquet_parity$'` + an exact `F8_PARITY` grep |
| `windows-unit.yml` | windows-latest | Debug | defaults | `ctest -j1 -C Debug --verbose` |
| `macos-unit.yml` | macos-latest | Release | `--preset clang-macos`, `HIGHS=ON` | `ctest --test-dir build -j -C Release` |
| `cuda-mpi-detect.yml` (5 jobs) | ubuntu / macos / windows | Release or unset | `CUDA=ON` or `MPI=ON`, `TESTING=OFF` | mostly configure/build only; one `mpiexec -n 2 ./build/bin/unit_test_mpi` |
| `documentation.yml` `build` | ubuntu-latest, gcc-13 | Debug | `COVERAGE=TRUE`, `HIGHS=OFF`, `GUROBI=OFF`, `LLFIO=OFF`, `EXAMPLES=ON` | `ctest -j2 -C Debug`; also `check_docs_contract.py` (twice, one with `--cli`) and `check_supply_chain_pins.py` |
| `matlab-mex.yml` | ubuntu / macos / windows | Release | `MATLAB=ON`, HIGHS/GUROBI/LLFIO OFF, `TESTING=OFF` | MATLAB `runtests`, no ctest |
| `python-tests.yml` | 3 OS × 6 Python (18 legs) | — | built via `uv pip install` | `pytest tests/python/`; a separate `supply-chain` job runs the pin checker |
| `python-wheels.yml` | 4 platform legs | Release | `PYTHON=ON`, `HIGHS=ON`, `GUROBI=OFF`, `LLFIO=OFF` | `_wheel_smoke.run()` |
| `release-artifacts.yml` | 3 OS | Release | `HIGHS=ON`, `GUROBI=OFF`, `LLFIO=OFF`, `TESTING=OFF` | `cpack` + `smoke_release_archive.py` |

**No job builds with all optional deps OFF.** The nearest miss (`documentation.yml`) leaves
`DTWC_ENABLE_YAML` at its ON default and adds coverage. `DTWC_ENABLE_YAML=OFF`,
`DTWC_ENABLE_METAL=OFF` and `DTWC_ALLOW_SEQUENTIAL=ON` are set by **no** workflow. No workflow runs
clang-tidy, clang-format, cppcheck, codespell, or `scripts/check_repo_hygiene.py` (414 lines, which
enforces banned tracked paths, zero-byte blobs, `.gitignore` contents, a 10-pattern secret scan and
CHANGELOG structure).

### 4.7 Platform-specific handling

- **MSVC/CUDA**: scans `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*`, injects missing
  `CUDA_PATH_Vxx_y` env vars, and writes a `Directory.Build.props` pinning `CudaToolkitCustomDir`
  [confirmed CMakeLists:81-135,187-207].
- **MSVC/OpenMP**: `OpenMP_RUNTIME_MSVC experimental`; `/openmp:experimental` is attached **PUBLIC on
  `dtwc++` itself** as well as on `dtwc_options`, because a consumer linking only `dtwc++` (the
  nanobind module) previously serialised silently [confirmed dtwc/CMakeLists:158-168; the same
  rationale is restated at python/CMakeLists:34-43].
- **Clang-on-Windows/OpenMP**: probes the LLVM install for `libomp` and sets the three FindOpenMP hint
  cache variables [confirmed dtwc/CMakeLists:136-152].
- **Apple**: Metal default-ON with `enable_language(OBJCXX)`; `metal_dtw.mm` compiled `-fno-objc-arc`;
  `libomp` installed beside the binary with `INSTALL_RPATH @loader_path/../lib`
  [confirmed CMakeLists:216-226,290-293; dtwc/CMakeLists:247-256].
- **MATLAB/MSVC**: directory-scope `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` so every TU — including
  fetched HiGHS/llfio — agrees on the `std::mutex` constructor, since mixing is IFNDR
  [confirmed CMakeLists:59-70]. **DELIBERATE**, documented in place.
- **GCC/LTO**: the header lint at `dtwc/CMakeLists:94-115` fatals on a named `#pragma omp critical(x)`
  in any header, because the COMMON `.gomp_critical_user_*` symbol breaks LTO links against
  `libdtwc++.a` (binutils PR ld/32083, GCC PR lto/116361). **DELIBERATE**, and the only
  `CONFIGURE_DEPENDS` glob whose purpose is linting rather than source collection.
- **Windows/Arrow tests**: PATH injection, §4.4.
- **MSVC/F22 probes**: `_SILENCE_ALL_CXX20_DEPRECATION_WARNINGS` plus per-compiler
  deprecation-error/suppression/error-limit flag triples, because llfio's `path_view.ipp` instantiates
  a deprecated `std::codecvt` and killed the probe before any dtwc diagnostic
  [confirmed tests/CMakeLists.txt:69-110].

### 4.8 Doxygen

Not wired into CMake at all — no `doxygen`/`Doxyfile` reference exists in `CMakeLists.txt`,
`cmake/*.cmake` or `dtwc/CMakeLists.txt`; it is invoked only by the `mattnotmitt/doxygen-action` step
in `documentation.yml:58-61`. Key settings: `PROJECT_NAME DTWC++` (:45), **`PROJECT_NUMBER 1.0.0`**
(:51) — stale against the `VERSION` file that drives `DTWC_VERSION_STRING`;
`INPUT = ./dtwc ./examples ./python ./bindings/matlab ./CHANGELOG.md ./README.md` (:911);
`OUTPUT_DIRECTORY build/Doxygen` (:71); `RECURSIVE YES` (:1002); `EXCLUDE` empty (:1011);
`GENERATE_HTML YES`, `LATEX NO`, `XML NO`; **`WARN_AS_ERROR NO`** (:869) with all four `WARN_*`
categories ON; `EXTRACT_ALL YES` (:502); `PREDEFINED` **empty** (:2366) with `MACRO_EXPANSION NO`
(:2325) — so every `#ifdef DTWC_HAS_*` block is invisible to Doxygen and the Parquet/Arrow readers are
undocumented in the generated site; `HAVE_DOT YES` (:2457). Every path is **relative to the repo root,
not to the Doxyfile's own directory** (:64, :71, :911, :1125, :1323, :1358) — it only works with
cwd = repo root, which is how CI invokes it.

---

## 5. Performance-critical structures and their rationale — DO NOT BREAK

1. **Chunked Parquet decode with metadata-first planning.** `ParquetChunkReader`'s constructor reads
   only `parquet_reader()->metadata()` [confirmed parquet_chunk_reader.hpp:187-218]; the CLI uses that
   to decide stream-vs-materialize **before** any `ReadTable`/`ReadRowGroups`
   [confirmed dtwc_cl.cpp:1298-1300]. Rationale: "a streaming RAM cap must be decided before payload
   I/O and must model retained-plus-transient data" [confirmed .claude/LESSONS.md:342].
2. **One `null_count()` per array, never per element.** `require_no_nulls` is called once per chunk and
   once per values buffer, outside the copy loop [confirmed parquet_schema.hpp:105-106;
   parquet_reader.hpp:91-92; parquet_chunk_reader.hpp:81-83]. Rationale at `.claude/LESSONS.md:403`
   ("keeps the copy loop branch-free").
3. **One type dispatch per array in `copy_arrow_numeric<T>`** [confirmed parquet_schema.hpp:132-133,
   140-147] — the whole point of collapsing the six extractor bodies.
4. **RAM-limit arithmetic is saturating, never wrapping.** `saturating_add`, `saturating_multiply`,
   `saturating_from_i64` [confirmed parquet_chunk_reader.hpp:382-388, 500-512]; the estimator takes
   `max(encoded_bytes, num_values * source_width)` because `total_uncompressed_size` alone is not
   trustworthy [confirmed :514-527; rationale .claude/LESSONS.md:341]. Saturation routes to streaming,
   which is the safe direction.
5. **Row groups are indivisible; the budget is checked per row group and the cap throws rather than
   silently exceeding it** [confirmed parquet_chunk_reader.hpp:353-379, 442-486].
6. **Arrow field index ≠ Parquet leaf index.** `parquet_leaf_count` walks nested children so a
   preceding Struct does not shift the selected physical column [confirmed parquet_schema.hpp:28-40,
   76-91; rationale .claude/LESSONS.md:343].
7. **One shared schema selector for the eager, metadata, sparse and streaming paths**
   [confirmed parquet_reader.hpp:66, parquet_chunk_reader.hpp:170; rationale .claude/LESSONS.md:344].
8. **Arrow IPC offset validation is a single O(N) pass at open**, so `series()` needs no per-access
   bound check [confirmed arrow_ipc_reader.hpp:127-143].
9. **`.dtws` layout**: 64-byte header (`DTWS` magic, u16 version, u32 endian marker `0x01020304`, u8
   `elem_size = 8`, u64 n, u64 ndim, CRC32 over bytes 0-27), then an `(n+1)`-entry `uint64` offset
   table, then the contiguous `double` payload [confirmed dtwc/core/mmap_data_store.hpp:60-98,
   131-136]. Any layout change needs a version bump — the reader rejects `version != 1`.
10. **Distance-matrix CSV preflight before truncation (F14).** `preflight_distance_matrix_csv` runs the
    whole O(N²) scan **before** the destination file is opened on the mmap path, and
    `write_distance_matrix_csv_preflighted` exists precisely so the scan is not repeated
    [confirmed matrix_io.hpp:202-206; Problem_IO.cpp:220-230]. Frozen byte-for-byte by
    `test_distance_matrix_csv_contract`, which pins row/LF/CR counts [confirmed tests:692-696].
11. **`--mmap-threshold` default 50000, with OneBatchPAM exempt and TADPole deliberately NOT exempt**
    [confirmed dtwc_cl.cpp:485-490, 503-513].
12. **`default_series_cache_path` costs one relaxed `fetch_add` per *load*, never per series**
    [confirmed dtwc/DataLoader.hpp:134-135,153; rationale .claude/LESSONS.md:394].
13. **`-fassociative-math` is load-bearing for the EAPruned 16ε prune slack**, which is
    regression-tested rather than derived [confirmed .claude/LESSONS.md:95]. Changing the FP flag set
    invalidates that constant.
14. **IPO/LTO on by default + the named-omp-critical header lint** are coupled: dropping the lint
    re-opens the GCC LTO link break [confirmed dtwc/CMakeLists:94-115].
15. **`/openmp:experimental` PUBLIC on `dtwc++`** — moving it back to `project_options` alone silently
    serialises the Python module on MSVC [confirmed dtwc/CMakeLists:158-168].
16. **Arch tuning is disabled for wheels and for sub-project consumption** so wheel binaries stay
    portable [confirmed StandardProjectSettings:74-91].

---

## 6. Duplication

| # | Group | Sites | Class | What differs | Candidate action |
|---|---|---|---|---|---|
| D1 | **Arrow status checkers ×3** | `parquet_reader.hpp:36-40` `check_arrow`, `parquet_chunk_reader.hpp:42-46` `check_arrow_chunk`, `arrow_ipc_reader.hpp:49-53` `check_status` | byte-identical bodies | function name and the parameter name (`ctx` vs `context`) | one `detail::check_arrow` reachable from both the `DTWC_HAS_PARQUET` and `DTWC_HAS_ARROW` scopes |
| D2 | **Parquet extractors** — prior review counted six | now `copy_arrow_numeric<T>` (`parquet_schema.hpp:136-153`) + `extract_series_from_column_as<T>` (`parquet_chunk_reader.hpp:68-116`), with 2 thin wrappers at `:119-127` and `:130-138`, plus the eager `append_list_series` lambda at `parquet_reader.hpp:89-105` | **largely fixed**; residual = near | the eager reader re-implements the list-cell walk and derives the `series_` name from `vecs.size()` instead of `name_offset` | route `load_parquet_file` through `extract_series_from_column_as<data_t>` |
| D3 | **Distance-matrix CSV emit loop ×3** | `matrix_io.hpp:106-116` (inside `write_csv`), `matrix_io.hpp:213-222` (`write_distance_matrix_csv_preflighted`), and `api.cpp:294` via `operator<<` | near-identical | `ofstream` vs `ostream`; `write_csv` inlines the loop instead of calling the template defined a hundred lines below | have `write_csv` call `write_distance_matrix_csv_preflighted` after its own preflight |
| D4 | **Distance-matrix CSV open+close+error triple ×3** | `matrix_io.hpp:100-119`, `Problem_IO.cpp:226-233`, `api.cpp:291-297` | semantic | identical open flags (`out\|binary\|trunc`) and identical two messages, but three exception types (`runtime_error`, `runtime_error`, `IOError`) | one `open_matrix_csv(path)` helper |
| D5 | **Output-artefact writers ×2 compatible + 1 incompatible** | `dtwc_cl.cpp:634-698` vs `api.cpp:268-319` (same 4 headers, same order) vs `Problem_IO.cpp:72-203` (prose-mixed legacy schemas) | near / semantic | the CLI adds `output_series_name` streaming-index handling and pre-validates medoid indices; `api.cpp` creates the directory, the CLI does not | one artefact-writer module over a small record |
| D6 | **ofstream open/close/error helpers** | `Problem_IO.cpp:34-62` (`ensure_output_directory`/`open_output`/`close_output`) are correct — but `dtwc_cl.cpp:645-647, 669-671, 689-691` and `api.cpp` re-implement only the *open* half and never the close check | semantic | the CLI never checks the stream after writing | reuse the existing triple |
| D7 | **Extension lower-casing + Parquet directory scan ×2** | `parquet_reader.hpp:148-154` and `dtwc_cl.cpp:1268-1277` | near | both `transform`+`tolower`, both accept `.parquet`/`.pq`, both `std::sort`; the CLI uses a lambda with an explicit `static_cast<char>`, the reader uses `::tolower` directly | share one `is_parquet_extension` / `sorted_parquet_files` |
| D8 | **Alias normalisation duplicated against CLI11's own transformers** | `dtwc_cl.cpp:1040-1065` (to_lower + 8 hand-written remaps) vs the `CheckedTransformer` maps at `:795-800, 826-834, 837-841, 843-848, 874-879, 899-904, 909-912, 951-954, 976-982, 990-995` | semantic | the hand-written set covers only `hclust/obp/lr`, `sqeuclidean/l2sq`, `soft-dtw`, `zero-cost/zerocost`, `f32/fp32/float`, `f64/fp64/double` — a strict subset that must be kept in sync manually | now that YAML goes through CLI11, the post-parse remap is largely redundant; delete it or generate it from the same maps |
| D9 | **`saturating_add` defined twice** | `parquet_chunk_reader.hpp:500-505` (private static) and `dtwc_cl.cpp:1304-1307` (a local lambda) | semantic, identical logic | member vs lambda | expose the reader's helpers, or move them to a tiny header |
| D10 | **CMake: `foreach` over the same three F22 probe targets ×3** | `tests/CMakeLists.txt:59-67`, `:77-84`, plus three separate `target_compile_options` at `:111-122` | near | a different property set each time | one loop with a per-target option variable |
| D11 | **CMake: the per-test gate block ×13** | `tests/CMakeLists.txt:146-175, 181-193, 198-212, 217-230, 236-246, 251-277, 282-294, 301-325, 330-344, 349-378, 381-397, 619-630, 643-654` | near | each repeats `if(TARGET x)` + `set_property(TEST x PROPERTY SKIP_RETURN_CODE)` + `set_tests_properties(... FAIL_REGULAR_EXPRESSION <one of four skip regexes> PASS_REGULAR_EXPRESSION <marker>)` | one `dtwc_pin_test(NAME … MARKER …)` function with a single canonical skip regex |
| D12 | **CMake: `get_target_property(<var> dtwc++ INTERFACE_COMPILE_DEFINITIONS)` ×5** | `tests/CMakeLists.txt:313, 363, 642, 685-688, 822` | byte-identical calls | only the output variable name | query once at the top of the file |
| D13 | **CMake: `set(<OPT> OFF PARENT_SCOPE)` + `message(WARNING)` degradation ×4** | `Dependencies.cmake:346-353, 408-411, 442-447` and `CMakeLists.txt:209-211` | near | four different message shapes for one policy | one `dtwc_disable_optional(<name> <reason>)` |
| D14 | **CMake: `add_executable` + `target_link_libraries(… benchmark::benchmark …)` ×8** | `benchmarks/CMakeLists.txt:4-63` | near | three variants of the link set (with `benchmark_main`, without, and none at all) | a small `dtwc_add_benchmark()` |
| D15 | **CMake: 12 imperative preset assertions in one shape** | `tests/CMakeLists.txt:399-617` | semantic | each repeats read-parse-compare-`FATAL_ERROR` | table-driven |

---

## 7. Design problems

Severity: **H** = wrong answers or silent data loss; **M** = misleading behaviour or maintenance
hazard; **L** = hygiene.

| # | Problem | Evidence | Sev | Blast radius | Candidate action |
|---|---|---|---|---|---|
| P1 | **`run_cli_main` is 1145 lines** mixing declaration, validation, routing, dispatch and output. Prior review §D: **open** | dtwc_cl.cpp:764-1908 | M | every CLI change; the structural reason P2/P3/P4 persist | split into parse → plan → load → run → write over a `CliConfig` value object |
| P2 | **Twelve string-typed options are re-parsed into enums by hand**, and `MIPSettings::benders` reaches the library as `std::string` | dtwc_cl.cpp:1519; enum re-parses at :1093-1120, :1529-1536, :1571-1576, :1706-1711, :1799-1806; `api.cpp:326` re-normalises method strings a second time | M | library API, bindings, every new variant | typed enums parsed once at the CLI boundary |
| P3 | **Validation is inconsistent.** `--n-clusters`/`--n-init` get `PositiveNumber` and `--mmap-threshold` `NonNegativeNumber`, but `--band`, `--max-iter`, `--dc`, `--mip-gap`, `--time-limit`, `--numeric-focus`, `--mip-focus`, `--checkpoint-interval`, `--skip-rows`, `--skip-cols` get **nothing**; `--numeric-focus`/`--mip-focus` document 0-3 without enforcing it; `--seed`'s `CLI::Range(0u, UINT_MAX)` is a **no-op**. Prior review §D: **open, unchanged** | dtwc_cl.cpp:835, 849, 853, 890, 917-918, 928, 964-968 | M | invalid solver parameters reach HiGHS/Gurobi unchecked | attach `CLI::Range` wherever a range is documented; delete the no-op |
| P4 | **The payload-load if/else chain is spliced by the preprocessor**: the `else` at :1373 and the `#endif` at :1374 mean the chain's *head* differs between a Parquet and a non-Parquet build | dtwc_cl.cpp:1368-1444 | M | any edit to load order silently changes behaviour in one build flavour only | resolve an `enum class InputKind` first, then one `switch` in which every branch is present and the unbuilt ones throw |
| P5 | **`.h5` and every other unknown extension is fed to the CSV parser**, while README advertises HDF5 as a CLI I/O format. Same class as LESSONS F9, which `require_input_format_is_built` fixed for Parquet/Arrow | classification dtwc_cl.cpp:1289 → DataLoader :1439; README.md:41; `grep -rn "HDF5" dtwc/` = 0 hits | M | a user converting `.h5` gets a numeric-parse error with no hint | add `.h5`/`.hdf5` to `require_input_format_is_built` as "not supported by the C++ CLI; use `dtwc-convert`" |
| P6 | **Every CLI output stream is open-checked but never close-checked.** A full disk yields a truncated `_labels.csv` and exit 0. The correct helper triple already exists in this repo and is unused here | dtwc_cl.cpp:645-654, 669-681, 689-698 vs Problem_IO.cpp:45-62. Prior review flagged the `Problem_IO` half as fixed; the CLI half is **open** | H | all four CLI artefacts | reuse `open_output`/`close_output` |
| P7 | **`fs::create_directories(output_dir)` return and `error_code` unchecked.** Prior review (`dtwc_cl.cpp:1219`): **open at :1234** | dtwc_cl.cpp:1234 | M | an unwritable `-o` produces four "Cannot open output file" throws instead of one clear message | use the `std::error_code` overload |
| P8 | **`.dtws` `.names` sidecar is read with no length check.** `for (size_t i = 0; i < n && std::getline(nf, names[i]); ++i) {}` leaves the remaining names empty, which become empty `name` fields in `_labels.csv` | dtwc_cl.cpp:1394-1401 | M | label/medoid provenance for `.dtws` inputs | fill the tail with `series_<i>`, or reject a short sidecar |
| P9 | **`--dist-matrix` failure warns, then the CLI silently recomputes the whole O(N²) matrix.** The *reader* now propagates correctly (prior review A1 **fixed** at Problem_IO.cpp:255-272) and the CLI's decision to continue is explicit and loud — but a truncated CSV leaves the matrix **partially populated**, because `read_csv` accumulates rows and applies them only after the loop that can throw | dtwc_cl.cpp:1579-1588; matrix_io.hpp:167-177 | H (partial fill) | any run importing a truncated matrix | reset the matrix on failure, or make the import fatal |
| P10 | **`main.cpp` depends on a repo-relative path**, violating non-negotiable #1, with the comment acknowledging it. Prior review A16: **open** | dtwc/main.cpp:19-20 (`fs::path("data") / "dummy"`, "Run this from the project root directory") | M | `dtwc_main` is built and documented but installed nowhere | take `argv[1]`, or delete `dtwc_main` |
| P11 | **`README.md:188` shows `./build/bin/dtwc_main ...`** but `main()` takes no arguments. Prior review A16: **open** | README.md:188 vs main.cpp:12 | L | user confusion | fix the doc or the signature |
| P12 | **Both example configs hardcode `input = "data/dummy"`**, a repo-relative path; `config.toml`'s usage line names a path that does not exist | examples/cpp/config.toml:3 (`examples/config.toml`; the file is at `examples/cpp/config.toml`), :10; examples/cpp/config.yaml:13 (its usage line at :3 is correct) | L | copy-paste failures | absolute-path note + fix the usage line |
| P13 | **`arrow_ipc_reader.hpp` uses `assert` as the only bounds guard** on `series()`/`series_flat_size()`, compiled out under `NDEBUG` (i.e. in Release). Prior review: **open** | arrow_ipc_reader.hpp:173, 180 | M | an out-of-range index reads arbitrary mapped memory in a Release build | `if (i >= n_) throw` |
| P14 | **`name_col->chunk(0)` is `static_cast` to `StringArray` with no type check**, unlike the `data` column which is checked by name. A `LargeString` or `Int64` name column is undefined behaviour. Prior review: **open** | arrow_ipc_reader.hpp:147-149 (contrast the correct check at :119-123) | H | any Arrow IPC file whose `name` column is not exactly `utf8` | check `type_id() == arrow::Type::STRING` or reject |
| P15 | **`std::stoul` on the `ndim` metadata is unguarded** — a non-numeric or oversized value throws `std::invalid_argument`/`std::out_of_range` out of `open()` with no context | arrow_ipc_reader.hpp:156 | L | Arrow IPC only; `main` still catches it | `from_chars` + a typed message |
| P16 | **`data_from_arrow` probes nulls and dispatches type per element** (`ArrowArrayViewIsNull(values, j)` and `ArrowArrayViewGetDoubleUnsafe(values, j)` inside the copy loop), directly contradicting the per-chunk policy the Parquet path documents as a performance invariant | arrow_c_data.cpp:96-101 vs parquet_schema.hpp:104-106 | M | every Python/polars/DuckDB ingest | hoist the null check to the values view's null count; hoist the type switch out of the loop |
| P17 | **`data_from_arrow_stream` leaks the batch `ArrowArray` when `get_next` fails**, and **discards any resolved names**, always emitting `series_<i>`. Prior review: **both open** | arrow_c_data.cpp:154-157 (no `release_arrow` before `fail`), :172-175 | M | stream ingest from polars/pandas | release before throwing; carry names through |
| P18 | **`load_folder`/`load_batch_file` take non-const lvalue refs** (`Tpath &`, `fs::path &`), so a temporary path will not compile. Prior review §D: **open** | fileOperations.hpp:385, 434 (and the compat overloads :414, :484) | L | any caller writing `load_folder<double>(dir / "x")` | take `const fs::path &` |
| P19 | **`readFile`'s header-skip loop ignores `getline` failure**, so `--skip-rows 1000` on a 3-line file silently yields an empty series instead of an error | fileOperations.hpp:294-295 | L | CSV inputs with a mis-set `--skip-rows` | check the stream after the skip loop |
| P20 | **CMake: `option(DTWC_ENABLE_LLFIO …)` is declared inside a function**, so it does not sit beside the other 15 in the root file and exists only after `dtwc_setup_dependencies()` runs | cmake/Dependencies.cmake:152, inside `function(dtwc_setup_dependencies)` opened at :9 | M | `cmake -LH` discoverability; ordering hazards | move it next to the other `DTWC_ENABLE_*` options |
| P21 | **CMake: `set(DTWC_HAS_MMAP TRUE)` has no `PARENT_SCOPE`** and is therefore dead (see §8 O3) | cmake/Dependencies.cmake:299 | L | none today — the `TARGET llfio_hl` guard is what works | delete |
| P22 | **CMake: `DTWC_HAS_PARQUET_LIB` is exported twice and read nowhere** (see §8 O2) | cmake/Dependencies.cmake:328, 405 | L | none | delete both writes |
| P23 | **CMake: `include(cmake/FindGUROBI.cmake)` is unconditional**, so Gurobi is probed — including `file(GLOB "C:/gurobi*/win64")`, library-version regexes and a `GurobiCXX` static target — on **every** configure, including `-DDTWC_ENABLE_GUROBI=OFF` | CMakeLists.txt:229; FindGUROBI.cmake:14, 152-165 | M | configure time; a surprising `GurobiCXX` target in Gurobi-OFF trees | wrap in `if(DTWC_ENABLE_GUROBI)` |
| P24 | **CMake: warnings are off outside `DTWC_DEV_MODE`.** `dtwc_ENABLE_COMPILER_WARNINGS` defaults to `${DTWC_DEV_MODE}` (OFF) and no CI job sets `DTWC_DEV_MODE=ON`, so `project_warnings` is an **empty INTERFACE library in every CI build** and `-Wconversion`/`-Wsign-conversion` never run on CI | ProjectOptions.cmake:22, 65-74; no workflow sets `DTWC_DEV_MODE` | M | conversion/shadow defects reach main unflagged | one CI leg with `-DDTWC_DEV_MODE=ON` |
| P25 | **CMake: benchmark targets for CUDA / Metal / MPI are added unconditionally**, guarded only inside the `.cpp` | benchmarks/CMakeLists.txt:23, 33, 41 (no `if(DTWC_ENABLE_CUDA)` etc.); `bench_mpi_dtw` links neither `benchmark::benchmark` nor `project_options` | L | only with `DTWC_BUILD_BENCHMARK=ON`, which no CI job sets | guard the targets |
| P26 | **CMake: `dtwc++`'s PUBLIC header list is incomplete** — `error.hpp`, `missing_utils.hpp`, `warping_missing*.hpp`, `core/matrix_io.hpp`, `core/variant_validation.hpp`, `types/Range.hpp`, `cli/config_file.hpp`, `enums/*` and all of `io/*.hpp` except `arrow_c_data.hpp` are absent | dtwc/CMakeLists.txt:33-82 vs the actual tree | L | IDE listing; any future `FILE_SET HEADERS` install | complete the list or drop it |
| P27 | **CMake: CI's `-fsanitize` bypasses the project's own sanitizer plumbing.** `ubuntu-unit.yml:61` injects raw `CMAKE_CXX_FLAGS` instead of `-Ddtwc_ENABLE_SANITIZER_ADDRESS=ON`, so the five sanitizer options are never exercised | ubuntu-unit.yml:61; ProjectOptions.cmake:24-28 | L | the sanitizer wiring is untested | use the options |
| P28 | **CMake: IPO failure is `SEND_ERROR` with no named escape hatch.** With `dtwc_ENABLE_IPO` defaulting ON at top level, a toolchain without IPO fails configure without being told `-Ddtwc_ENABLE_IPO=OFF` exists | InterproceduralOptimization.cmake:7 | L | exotic toolchains | name the escape hatch in the message |
| P29 | **CMake: `SKIP_RETURN_CODE 4` is applied blanket then individually cleared 13 times.** The default is "a skip is a pass"; correctness depends on remembering to opt out | Coverage.cmake:33 vs the 13 `set_property(TEST … PROPERTY SKIP_RETURN_CODE)` clears | M | any new test that must never skip | invert the default: opt **in** to skip-tolerance |
| P30 | **CMake: four different skip regexes** across the gate blocks; the weakest (`"[Ss][Kk][Ii][Pp]"`, tests:290) matches any output containing the substring, the strongest is a 60-character anchored alternation (tests:171, 651) | tests/CMakeLists.txt:171, 189, 243, 290, 651 | L | false failures or missed skips | one shared variable |
| P31 | **No CI job builds with all optional deps OFF**, which is the actual gate for non-negotiable #3. The nearest miss leaves YAML ON and adds coverage | §4.6 | M | the core-builds-bare contract is not continuously verified | one `-DDTWC_ENABLE_{HIGHS,GUROBI,LLFIO,ARROW,YAML,METAL}=OFF -DDTWC_ALLOW_SEQUENTIAL=ON` leg |
| P32 | **`THIRD_PARTY_LICENSES.md` names only HiGHS**, while the binary statically contains vendored nanoarrow (Apache-2.0) plus CLI11, Eigen, fkYAML, Catch2 and optionally Arrow and llfio. `cpm_licenses_create_disclaimer_target` covers **CPM packages only**, so vendored nanoarrow appears in neither | THIRD_PARTY_LICENSES.md:3-6; CMakeLists.txt:319, 285-288 | M | redistribution compliance | add nanoarrow's NOTICE and the header-only deps |

**Not problems — DELIBERATE, with evidence:**

- Guards outside their own `#ifdef`: `require_input_format_is_built` under `#ifndef`,
  `require_ram_limit_is_applicable` before the `#ifdef`, the YAML refusal in the `#else`
  [confirmed dtwc_cl.cpp:227-251, 1291-1295; config_file.hpp:143-145; rationale LESSONS.md:345].
  Prior review **A2 fixed**, **A8 fixed**.
- Terminal `else { throw }` on every string dispatch chain [confirmed dtwc_cl.cpp:1575-1576,
  1805-1806, 1817-1822; rationale LESSONS.md:383]. Prior review **A7 fixed**.
- `--benders` now carries a `CheckedTransformer` [confirmed :976-982]. Prior review **A6 fixed**.
- `read_distance_matrix` propagates; the caller decides [confirmed Problem_IO.cpp:253-260; rationale
  LESSONS.md:381]. Prior review **A1 fixed** (residual is P9).
- `require_no_nulls` / `require_list_range` / `copy_arrow_numeric` [confirmed parquet_schema.hpp:98-153].
  Prior review **A3 fixed**.
- `sorted_directory_files` + `validate_ndata` / `ndata_wants_more` [confirmed fileOperations.hpp:313-349;
  rationale LESSONS.md:408, 421]. Prior review **A10, A11 fixed**.
- `readCSV` / `readTimeSeriesCSV` / `readCSVColumn` and rapidcsv deleted [confirmed CHANGELOG.md:433].
  Prior review **A9, §E fixed**.
- `default_series_cache_path` is atomic with real per-process entropy
  [confirmed DataLoader.hpp:137-156]. Prior review **§B fixed**.
- Both `List` and `LargeList` accepted [confirmed parquet_schema.hpp:45-49; rationale LESSONS.md:234].
- Metal silently OFF on non-Apple [confirmed CMakeLists.txt:45-46, 222-225].
- `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` at directory scope for MATLAB builds
  [confirmed CMakeLists.txt:59-70].
- The explicit FP subset without `-ffinite-math-only` [confirmed StandardProjectSettings.cmake:45-72].
- OpenMP as a hard requirement with an explicit opt-out [confirmed dtwc/CMakeLists.txt:184-194].
- `PASS_REGULAR_EXPRESSION` avoided for rejection tests in favour of a `.cmake` driver script that
  asserts the exit code and the flag name [confirmed tests/CMakeLists.txt:759-780; rationale
  LESSONS.md:1304].
- Test `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` + `DTWC_TEST_DATA_DIR` — repo-relative by design for
  *tests*, outside non-negotiable #1's runtime scope [confirmed Coverage.cmake:8, 21].
- `.cc` suffix on integration fixture programs to keep them out of the CTest glob
  [confirmed tests/CMakeLists.txt:673-674].

---

## 8. Obsolete and dead code, stale comments and docs

Each item names the command run and its result.

| # | Item | Proof | Candidate action |
|---|---|---|---|
| O1 | `macro(dtwc_enable_include_what_you_use)` — **zero callers** | `grep -rn "include_what_you_use" --include=CMakeLists.txt --include=*.cmake .` (build/_deps filtered) → only the definition at `cmake/StaticAnalyzers.cmake:101` | delete |
| O2 | `DTWC_HAS_PARQUET_LIB` — **set twice, read nowhere** | `grep -n "DTWC_HAS_PARQUET_LIB" cmake/Dependencies.cmake dtwc/CMakeLists.txt CMakeLists.txt tests/CMakeLists.txt python/CMakeLists.txt` → only `Dependencies.cmake:328` and `:405`, both writes | delete both |
| O3 | `set(DTWC_HAS_MMAP TRUE)` at `Dependencies.cmake:299` is **function-local** (no `PARENT_SCOPE`) and never read | `grep -n "DTWC_HAS_MMAP" cmake/*.cmake CMakeLists.txt dtwc/CMakeLists.txt dtwc/mip/CMakeLists.txt tests/CMakeLists.txt` → the only consumers are `TARGET llfio_hl` guards (`dtwc/CMakeLists.txt:213`, `dtwc/mip/CMakeLists.txt:57`) and `IN_LIST` checks against the *compile definition* (tests:313, 363, 689) | delete the write |
| O4 | `GUROBI_LIBRARIES` / `GUROBI_INCLUDE_DIRS` "legacy support" — **zero readers** | `grep -n "GUROBI_LIBRARIES\|GUROBI_INCLUDE_DIRS" -r cmake dtwc CMakeLists.txt` → only the definitions at `cmake/FindGUROBI.cmake:168-169` | delete |
| O5 | `[[maybe_unused]]` on `require_ram_limit_is_applicable` is vestigial — the function is now called unconditionally | `dtwc_cl.cpp:215` declares it `[[maybe_unused]]`; the call at `:1292` sits above the `#ifdef DTWC_HAS_PARQUET` at `:1297` | drop the attribute (keep it on `:201` and `:275`, which really are `#ifdef`-only) |
| O6 | `#include <sstream>` in `matrix_io.hpp` is unused | `grep -n "stringstream\|ostringstream\|istringstream\|sstream" dtwc/core/matrix_io.hpp` → only the include at `:39` | remove |
| O7 | `#include "settings.hpp" // for resultsPath` in `fileOperations.hpp` — **the comment is false**; `resultsPath` and `settings::` appear nowhere in the file (the `data_t` there is a template parameter, not the settings alias) | `grep -n "resultsPath\|settings::" dtwc/fileOperations.hpp` → the include line only | fix the comment; check transitive consumers before removing the include |
| O8 | `dtwc/CMakeLists.txt:212` comment reads **"llfio (required)"** — llfio is optional | `Dependencies.cmake:152` (`option(DTWC_ENABLE_LLFIO … ON)`) + the `if(TARGET llfio_hl)` guard at `dtwc/CMakeLists.txt:213` | fix the comment |
| O9 | `tests/CMakeLists.txt:759-760` comment states "**TOML via `--config` is the one CLI config mechanism**" — contradicted 24 lines later at `:783` ("`--config` takes TOML or YAML") and by `config_file.hpp` | both lines in the same file | rewrite: the `--yaml-config` *flag* is gone; YAML now arrives through `--config` |
| O10 | `README.md:41` lists **HDF5** among the CLI's I/O formats "auto-detected from extension" | `grep -rn "hdf5\|HDF5\|H5" dtwc/` → no hits; `docs/content/guides/data-formats.md:16` states the correct position (Python conversion only) | qualify the README bullet |
| O11 | `README.md:188` `./build/bin/dtwc_main ...` — `dtwc_main` accepts no arguments | `dtwc/main.cpp:12` `int main()` | fix |
| O12 | `examples/cpp/config.toml:3` usage line names `examples/config.toml`; the file is at `examples/cpp/config.toml` | the file's own path; the YAML twin at `config.yaml:3` is correct | fix |
| O13 | `docs/Doxyfile:51` `PROJECT_NUMBER = 1.0.0` while the build reads the version from the `VERSION` file into `DTWC_VERSION_STRING` | `CMakeLists.txt:13-14, 245` | drive it from `VERSION`, or drop it |
| O14 | `docs/Doxyfile` has `PREDEFINED` empty and `MACRO_EXPANSION NO`, so **all six `#ifdef DTWC_HAS_*` IO readers are excluded from the generated API docs** | Doxyfile:2325, 2366 vs the `#ifdef` at the top of every `io/*.hpp` | add `PREDEFINED = DTWC_HAS_ARROW DTWC_HAS_PARQUET DTWC_HAS_MMAP DTWC_HAS_YAML` |
| O15 | `CMakePresets.json`: `msvc` and `gcc-linux` have **no** matching build or test preset, unlike the other three | CMakePresets.json:42-71 vs :90-146 | add them, or document why not |
| O16 | Prior review §E items now **gone**: `ParquetChunkReader::estimated_total_bytes()`, `estimated_bytes_per_series()`, `read_row_group(int)`; `<span>` in both Parquet readers; `readCSV`/`readTimeSeriesCSV`/`readCSVColumn`; the unreachable CLI `--dist-matrix` guard | complete reads of `parquet_chunk_reader.hpp` (564 lines), `parquet_reader.hpp` (175), `fileOperations.hpp` (491) — none present | none |
| O17 | The two TODO comments the prior review cited in `dtwc_cl.cpp` are gone | `grep -n "TODO" dtwc/dtwc_cl.cpp` → no hits; `:940` is now the `--restart` deprecation comment | none |

---

## 9. Lock, atomic, static and shared-state inventory (this scope)

| Site | Kind | Hot / cold | Invariant | Status |
|---|---|---|---|---|
| `dtwc/DataLoader.hpp:139` `static std::atomic<std::size_t> counter` | function-local atomic, `fetch_add(relaxed)` | **cold** — one increment per `load()` call, explicitly not per series [confirmed :134-135] | uniqueness of the temp `.dtws` path across concurrent loads | correct; relaxed suffices because only distinctness matters |
| `dtwc/DataLoader.hpp:142` `static const std::string process_tag` | function-local static, lambda-initialised (thread-safe static init) | cold, once per process | uniqueness across *processes* — a static address is identical in every process of the same image [confirmed :135-136] | correct; mixes `random_device` with the wall clock [confirmed :140-147] |
| `dtwc/DataLoader.hpp:282` `static inline std::atomic<std::size_t> s_bulk_read_invocations` | class-level atomic | cold (once per bulk read) | test instrumentation only (`bulk_read_count` :307, `reset_bulk_read_count` :312) | now atomic — the prior review's race is fixed — but **still test instrumentation compiled into production**, and `reset_bulk_read_count` is inherently racy if two tests run in parallel |
| `dtwc/dtwc_cl.cpp:729` `static const std::vector<CliRename> table` | function-local static | cold, once | single source of truth for the rename table, shared with `unit_test_cli_args.cpp` | correct; immutable after init |
| `dtwc/dtwc_cl.cpp:1156` `dtwc::env().set_device(device)` | mutates the **process-wide** `Env` singleton | cold, once per run | one source of truth for device selection, so the no-silent-fallback rules apply in one place [confirmed :1150-1154] | correct for a CLI; the CLI is the only writer |
| `dtwc/io/*` | **none** | — | — | the readers hold no shared mutable state; `ParquetChunkReader` documents "NOT thread-safe … one reader per thread or serialize externally" [confirmed parquet_chunk_reader.hpp:148-150] and every mutating member is `private` and written only in the constructor |
| `dtwc/core/matrix_io.hpp` | **none** | — | — | all free functions; the 64-byte `std::array` number buffer is a local |
| `dtwc/cli/config_file.hpp` | **none** | — | — | all helpers are `static` free functions; `app_` is a `const CLI::App *` |
| `dtwc/fileOperations.hpp` | **none** | — | — | the loaders write to `std::cout` unconditionally under `verbose > 0` [confirmed :388, 405, 437, 475] — a shared stream, not shared state |
| `#pragma omp critical` / mutex / `thread_local` | **none in this scope** | — | — | `grep -n "omp critical\|std::mutex\|thread_local" dtwc/dtwc_cl.cpp dtwc/io/* dtwc/fileOperations.hpp dtwc/core/matrix_io.hpp dtwc/cli/config_file.hpp` → no hits. The header lint at `dtwc/CMakeLists.txt:101-114` enforces that any header-level critical is unnamed |

No locks of any kind exist in the IO, CLI or build scope. The three atomics are all once-per-call and
off every hot path.

---

## 10. Open questions for the designer

1. **What is the intended lifetime of `dtwc_main`?** It violates non-negotiable #1, is documented with
   an invocation its signature cannot accept, and is installed nowhere. Keep it as a tutorial with an
   `argv` path, or delete it?
2. **Should the CLI reject `.h5`/`.hdf5` explicitly** (P5), and should the README stop listing HDF5 as
   a CLI-readable format — or should a native reader exist?
3. **Where should the artefact writers live?** Three schemas exist (CLI, Tier-1 `Result::save`, legacy
   `Problem_IO`). The api-contract freezes the CLI/Tier-1 pair as byte-identical
   [confirmed docs/api-contract-2.0.md:827]. Is the legacy `Problem_IO` family still part of the
   contract, or deletable?
4. **Is the `#ifdef`-spliced load chain (P4) acceptable, or should it become a typed `InputKind`
   switch?** The latter also removes D7 and makes P5 a one-line addition.
5. **Should `--benders` stay a `std::string` on `MIPSettings`** — the only string-typed option that
   crosses into the library — or become an enum? Changing it is an API break for the bindings.
6. **What is the policy for a partially-read `--dist-matrix`** (P9)? Today a truncated CSV half-fills
   the matrix and the run continues after a warning.
7. **Should `DTWC_ENABLE_LLFIO` move to the root option block** (P20), and should the split `dtwc_*` /
   `DTWC_*` option prefixes be unified?
8. **Should CI gain (a) an all-optional-deps-OFF leg (P31), (b) a `DTWC_DEV_MODE=ON` leg so the warning
   set is actually applied (P24), and (c) a `DTWC_ENABLE_YAML=OFF` leg** so the 7-check config-format
   flavour is exercised?
9. **`SKIP_RETURN_CODE 4` is blanket-on with 13 opt-outs (P29).** Invert it, or keep the default and add
   a lint requiring every new test to either pin a marker or state why it may skip?
10. **Is a network-dependent configure acceptable long-term?** `cmake/CPM.cmake` downloads CPM itself at
    configure time, and llfio's bootstrap `git clone`s quickcpplib and then **patches** the downloaded
    file [confirmed Dependencies.cmake:249-277]. Should the llfio superbuild be replaced, or vendored
    the way nanoarrow was?
11. **Who owns third-party licence aggregation?** `cpm_licenses_create_disclaimer_target` covers CPM
    packages only, so vendored nanoarrow (Apache-2.0) is unattributed anywhere (P32).
12. **Should the nanoarrow ingest adopt the per-chunk null/type policy** the Parquet path documents as a
    performance invariant (P16), given both feed the same DTW kernels?
13. **Is `--checkpoint-interval` actually reaching the mid-fill save path?** The CLI wires
    `prob.checkpoint.save_interval` [confirmed dtwc_cl.cpp:1598-1600], which the prior review's F1
    reported as inert. Verifying that end-to-end is outside this scope and needs a direct test.
14. **Doxygen documents nothing about the IO layer** (O14) and is not a CMake target. Should the API
    site cover the optional readers, and should `doxygen` be buildable from CMake?
