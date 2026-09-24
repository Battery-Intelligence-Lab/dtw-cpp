# IF-2 design — `dtwc::Config` + `run(Config) → Result`

Produced 2026-09-24 by a read-only design agent against the working tree of that day (line numbers from it; they drift).
Reviewed by the main session; its decisions on the open questions are in §10 and in `DECISIONS.md` (2026-09-24). This file is
the context pack for the IF-2 implementers (S1–S4, §9). `[inferred]` = not opened or run by the designer.

## Findings while reading (each shapes the design)

- `Problem`'s dense fill computes L1 only (`Problem.cpp:387, 1112-1121`); `--metric squared_euclidean` works only through the
  CLI's own CUDA fill (`dtwc_cl.cpp:1679`), so deleting that fill needs `Problem::set_metric`.
- `--dtype float32 --device cuda`: the CLI fill reads `p_vec` (`dtwc_cl.cpp:1693`), empty for Float32 (`Data.hpp:33-34`); the CPU
  then computes every pair [inferred past the call] — a silent device fallback (→ FX-17).
- MATLAB `set_method('pam'|'auto')` selects Lloyd k-medoids, not PAM (`dtwc_mex.cpp:498-503`) (→ FX-17).
- Python's hpc path silently drops `delimiter` and refuses `skip_rows` (`_api.py:537-545`) (→ FX-17 for the drop).
- The CLI's `auto` ignores the device (`dtwc_cl.cpp:194-199`), so `--device cuda` fails above N = 5000 (`:1661-1666`); Tier-1
  picks pam there (`tier1_method_resolution.hpp:33-35`).
- `cli.md:55` documents the `--method` default as `pam`; the code's is `auto` (`dtwc_cl.cpp:826`) (→ FX-17).
- B-14 is the CMake option prefix (ledger row 118); the CLI's deprecated spellings are `--clusters` and `--restart`
  (`dtwc_cl.cpp:737-779`, contract §4).

## 1. `Config` (K1, K4)

A C++20 aggregate in `dtwc/config.hpp`; member initialisers are today's CLI defaults (`dtwc_cl.cpp:800-1017`); each key is the
CLI long name.

| Group | Fields (default) | Existing struct reused |
| --- | --- | --- |
| Input and storage | `input` (""), `column` (""), `skip_rows` 0, `skip_cols` 0, `char delimiter` 0 (new flag `--delimiter`), `core::Precision dtype` Float64, `size_t ram_limit` 0, `size_t mmap_threshold` 50000, `dist_matrix` ("") | Tier-1 `Dataset`'s options |
| Method | `ClusterMethod method` Auto, `k` 3, `max_iter` 100, `n_init` 1, `unsigned seed` 42, `sample_size` −1, `n_samples` 5, `batch_size` −1, `batch_weighting` NearestNeighbor, `linkage` Average, `tadpole_dc` −1 | the user fields of `CLARAOptions`, `OneBatchPAMOptions`, `HierarchicalOptions` |
| Distance | `band` −1, `core::MetricType metric` L1, `core::DTWVariantParams variant`, `core::MissingStrategy missing` Error | `DTWVariantParams`, whole |
| Device | `Device device` CPU, `CUDASettings gpu` (`device_id` from `gpu:N`; `precision` from `--gpu-precision`) | `CUDASettings`, whole |
| Solver | `Solver solver` HiGHS, `MIPSettings mip` (adds `--max-benders-iter`, `--lr-max-nodes`) | `MIPSettings`, whole |
| Checkpoint | `checkpoint` (""), `checkpoint_interval` 0 (= save at the end only), `resume` false | — |
| Output | `output` "./results" ("" = write nothing), `name` "dtwc", `verbose` false | replaces `settings::paths::results` |

Not reused whole: `CLARAOptions` (repeats k / max_iter / seed, carries streaming plumbing, `fast_clara.hpp:42-47`), `DTWOptions`
(its `constraint` field makes `band` inert unless set, `dtw.cpp:35`), `CheckpointOptions` (defaults differ from the CLI's).

One new type: `enum class ClusterMethod {Auto, PAM, OneBatch, CLARA, Kmedoids, MIP, LRCore, TADPole, Hierarchical}`. `Method`
stays the four values `Problem::cluster()` dispatches (`Problem.cpp:1331-1353`): extending it (ledger A-01) would make
`set_method` accept values `cluster()` must then refuse — a check where a type does the job. `detail::Tier1ExecutionTarget` goes.

## 2. String↔enum tables (K4)

`dtwc/base/names.hpp` (~40 lines): `Name<E>{string_view text; E value;}`; `parse_name(table, text, what)` — ASCII
case-insensitive, throws `InvalidInput("unknown <what> '<text>'. Valid: …")` (pytest's "unknown method" match survives);
`name_of(table, value)`. Each table sits beside its enum; the first entry per value is canonical, later ones are today's CLI11
aliases copied verbatim from `dtwc_cl.cpp:813-1013`: `enums/{Method,Solver,LowerBoundStrategy}.hpp`; `core/dtw_options.hpp`
(variant, metric, missing strategy, mv_mode); `core/storage.hpp` (dtype, storage policy); `algorithms/{hierarchical,
one_batch_pam}.hpp`; `Problem.hpp` (DistanceMatrixStrategy, GPU precision 0/1/2); `config.hpp` (ClusterMethod, with `obp`, `lr`,
`hclust`). Kept: `detail::parse_device` (`env.cpp:214-233`, a grammar because of `:N`); `core::parse_metric_token`
(`distance_semantics.hpp:24-32`, now calls the metric table); `MIPSettings::benders` stays a string with a CLI11 map.
CLI11 binds each enum with `CheckedTransformer(cli::choices(table), ignore_case)`, so config files read the same tables; the MEX's
seven hand-written parsers (`dtwc_mex.cpp:469-531`) become `parse_name` calls; Python and MATLAB keywords go through CLI11 (§3).

## 3. CLI11 and the config file

`void cli::bind(CLI::App&, Config&)` in `dtwc/config.cpp` — the only TU that includes CLI11 (already fetched every configure,
`Dependencies.cmake:76-89`; linked PRIVATE). It is the only key table: one line per key; nested fields keep flat kebab keys
(`wdtw-g` sets `cfg.variant.wdtw_g`); `always_capture_default()` shows `Config{}`'s defaults in `--help`. Precedence (file <
flags) and the unknown-key error (`dtwc_cl.cpp:798`) stay CLI11's. `--clusters` / `--restart` stay hidden options writing the same
field with today's warning; `cli_renames` goes. Built on `bind`:

- `parse_config(pairs)`: Python keywords and MATLAB name-value pairs (`max_iter` → `max-iter`, `k` → `n-clusters`) are rendered as
  TOML lines and read by CLI11's `parse_from_stream` (v2.6.2, `App.hpp:957`); CLI11 errors become `InvalidInput`.
- `to_config_text(Config)`: every bound key, enums by canonical name, doubles as shortest round-trip `to_chars`; generated from
  `bind`, so no second field list.
- `dtwc_cl --print-config` (new): the contract test drives the real binary with it, and CLI users get a file to submit.

Types delete checks: the seed `Range` (`:906-908`), the case / alias remap (`:1063-1090`), `validate_cli_route_selectors`
(`:433-451`). `parse_ram_limit` (`:72-170`) stays — CLI11's `AsSizeValue` truncates, rejecting the documented `1.5G`.

## 4. `run`

`Result run(const Config&)` reads `input`; `Result run(const Config&, Data)` takes data in memory (input-only fields must be at
their defaults, or the typed error of the rule at `:258-270`).

1. Before any I/O (moved from `:1043-1161`): k, max_iter, n_init ≥ 1; `validate_problem_distance_semantics`,
   `validate_mip_settings`, the CLARA controls; `gurobi` without Gurobi → `SolverError`.
2. Classify the input (`:1281-1333`); the Parquet metadata plan (`:1335-1404`).
3. Resolve method × device (N from Parquet metadata when available, else after loading):
   - `hpc` → `DeviceError`: "run: device 'hpc' submits a run to a SLURM cluster, which Python's dtwcpp.cluster(...,
     device='hpc') and slurm_remote.sh submit-cluster do; dtwc_cl and dtwc::run compute where they start. No local fallback was
     attempted."
   - `gpu` on a build without a GPU backend → the frozen §6.1 message.
   - `auto` → pam on `gpu`; on `cpu`, pam for N ≤ 5000, else clara.
   - `gpu` with onebatch, tadpole, or clara whose sample is smaller than N → `DeviceError`: "run: method '<m>' computes its
     distances on the CPU as it goes, so device 'gpu' would sit idle; the GPU fills the distance matrix that pam, kmedoids, mip,
     lrcore and hierarchical use (and clara when its sample covers every series). Choose one of those, or device 'cpu'. No CPU
     fallback was attempted."
   - A metric other than l1 on `cpu` → `InvalidInput` (today's `:383-384`), unless S2 makes the CPU fill honour it (§10.2).
   - On `gpu`: non-standard variant, missing strategy or float32 → FX-1's `DeviceError` texts, raised here before loading by one
     data-free helper that `validate_fill_request` (`Problem.cpp:969-1009`) also calls.
4. Load: series storage Heap on `gpu`, Auto on `cpu` (Tier-1's rule, `tier1_method_resolution.hpp:47-52`); the CLI's Heap pin
   (`:1279`) goes; `DataLoader::load_local()`, never `env()` (`DataLoader.hpp:408-413`).
5. Configure the Problem: `set_data`, band, max_iter, n_repetitions, seed, `set_variant`, `set_missing_strategy`, `set_metric`
   (new, additive), `mip_settings`, `set_solver`, `set_cuda_settings(gpu)`, `set_device(device, gpu.device_id)`.
6. Distance storage: mmap only when `output` is non-empty (`:481-541`); then `--dist-matrix` and the dense checkpoint
   (`:1621-1658`).
7. Matrix methods call `fill_distance_matrix()`; the CLI's CUDA block (`:1660-1719`) goes.
8. One dispatch switch replaces the CLI's chain (`:1721-1869`) and `api.cpp:364-392`.
9. Outputs (moved from `:1871-1956`), written only when `output` is non-empty.
10. Return `Result`, which gains the resolved method, iterations and converged (the CLI summary `:1958-1969` needs them; the first
    fields of IF-4's RunStats).

`cluster()` keeps its signature and becomes ~20 lines: a `Config` from its arguments, `output={}`, `name=ds.name()`, the device
from `env()` / `parse_device`; a path dataset → `run(c)`, an in-memory one → `run(c, ds.materialize_local())`. Every Tier-1
route already uses the CLI's defaults, so conformance stays digit-identical. Deletes `api.cpp:76-92`, `:313-392` and
`tier1_method_resolution.hpp`.

## 5. hpc: serialise to a config file

Python calls `_config.pairs(**kw)` → `_dtwcpp_core._config_text(pairs)` with `device="cpu"` (validated locally by the parser the
cluster uses), uploads the file beside the input, and `slurm_remote.sh submit-cluster <input> <config> <name> <upload> [device]`
runs `dtwc_cl --config … --input … --name … --output …` (flags beat the file, as `conformance.toml` already relies on). Deletes
~275 of `_hpc.py`'s 625 lines (its validators, `build_dtwc_command`, the 20-keyword signatures), ~110 of `slurm_remote.sh`
(`:442-579, 639-648`) and ~50 of `cluster_generic.slurm`. Passing arguments instead would keep all three validation layers
(`sbatch --export` cannot carry an arbitrary argv [inferred]). `skip_rows` and `delimiter` then reach the remote run.

## 6. `settings::paths`

`data` (`settings.hpp:69`) with `set_data_path` / `setDataPath` — used by four C++ examples, `benchmarks/UCR_dtwc.cpp`, five test
files and the shim probe (`tests/CMakeLists.txt:79-80`) → `Config::input`. `results` (`:74`) with `set_results_path` /
`setResultsPath` — only production user `Problem.hpp:203`, which then defaults to `"./results/"` → `Config::output`. All 2.0-born
(v1.0.0 had constants `resultsPath`, `dataPath`, `dtwc_dataPath` computed from the source tree, removed during 2.0 unrecorded).

## 7. Deleted (approximate)

CLI: ~360 lines outright (device parser, CUDA fill, three validators, remap, variant-parameter mapping, `env` setup, renames);
~1,100 move to `run.cpp` / `config.cpp`; `dtwc_cl.cpp` 1985 → ~150. api: ~75 + the 54-line resolution header. MEX: ~65.
settings: 48. Python and bash: ~435 (§5). IF-3 afterwards: ~230 of `_api.py:370-597` and ~110 of `__init__.py:128-251`.

## 8. Compatibility

CLI and config files: every flag, key and value still accepted with the same default. New: `--device gpu|gpu:N` (Metal on macOS),
`--delimiter`, `--max-benders-iter`, `--lr-max-nodes`, `--print-config`. `auto` on a GPU runs pam instead of failing above
N = 5000. On `cpu`, series larger than half the free RAM go to a temporary `.dtws` store (same results). `--checkpoint-interval 0`
means "save at the end only" (was an error at fill time). Tier-1: identical results; path datasets may be Parquet, Arrow or
`.dtws`; a CLARA run whose sample covers every series runs on the GPU; `device="hpc"` raises the D-10 error without reading
`.env`. Python: nothing but hpc.

| Item | What a user would notice | Reason | Mitigation |
| --- | --- | --- | --- |
| IF-2 | `dtwc_cl --dtype float32 --device cuda` raises `DeviceError` | R1: the CPU computed every pair [inferred] | load Float64, or use `cpu` |
| IF-2 | MATLAB `Problem.set_method('pam'/'auto')` raises `dtwc:invalidArgument` | R1: it ran Lloyd | `dtwc.fast_pam` |
| IF-2 | `settings::paths` and its eight setters are removed | PRE-TAG (2.0-born, D-3) | `Config::output` / `set_output_folder` |
| IF-2 | hpc option errors are C++ `InvalidInput`, not `_hpc`'s `TypeError` | PRE-TAG | — |

## 9. Implementation split

| Step | Owns | Tests it adds |
| --- | --- | --- |
| S1 tables, `Config`, `bind` (no behaviour change) | `base/names.hpp`, the enum headers of §2 except `Problem.hpp`, `config.{hpp,cpp}`, CMake | `test_names.cpp`; contract `test_config_spellings.cpp` with golden `tests/conformance/config_all_fields.toml` (every field non-default): CLI argv, TOML and YAML each render to the golden text, and `to_config_text(Config{})` differs on every line |
| S2 Problem and settings (parallel with S1) | `Problem.{hpp,cpp}`, `settings.hpp`, examples, `UCR_dtwc.cpp`, path-using tests, `fast_clara.cpp:367-371, 515-520` (copy `cuda_settings`) | `test_problem_metric.cpp`: dense SqL2 on Metal equals CPU `distance::dtw` within FP32 tolerance; CPU non-L1 as §10.2 decides; CUDA leg blind (V-row) |
| S3 `run`, CLI, `api` (after S1, S2) | `run.cpp`, `api.{hpp,cpp}`, `dtwc_cl.cpp`, `unit_test_cli_args.cpp`, `test_tier1_cpp_api.cpp`, `check_docs_contract.py` (`:701, 815-817, 553-557`), docs, examples' TOML / YAML, contract addendum | `test_run_resolution.cpp` (every method × device cell); `test_cli_device_matrix.cmake` driving the binary, `--print-config` equal to the golden text; existing `test_cli_*` unchanged; conformance digit-identical |
| S4 bindings and hpc (after S3) | `_dtwcpp_core.cpp` (`_config_text`), `_config.py`, `_hpc.py`, `_slurm/*`, `_api.py`'s hpc branch, `dtwc_mex.cpp` | pytest `test_config_spellings.py` (keywords render to the golden text); `test_hpc.py` rewritten so the generated file run by a local `dtwc_cl` is byte-identical to a local run; `test_config_spellings.m` (blind, V-row) |

## 10. Open questions — decided by the main session on 2026-09-24 (overturnable by Volkan)

1. New `ClusterMethod` enum rather than extending `Method` (A-01): **yes** — a type, not a runtime refusal.
2. `Problem::set_metric`: **yes**, additive. The CPU fill honours it too if that is the kernels' existing metric parameter passed
   through (the Python estimators already compute squared L2 on the CPU by a precomputed-matrix detour, `_clustering.py:384-389`,
   which IF-3 then deletes); otherwise CPU non-L1 stays `InvalidInput`.
3. Python and MATLAB keywords parsed by CLI11 inside the library: **yes** — one parser, one set of messages.
4. The CLI's series storage follows the device (Heap on `gpu`, Auto on `cpu`): **yes**, as Tier-1 already does — "set the problem,
   then it is decided how to load the data" (CHARTER 2026-09-23). No `--storage` flag.
5. `dtwc_cl` submitting to hpc itself: **no** (D-10 stands); `slurm_remote.sh submit-cluster` is the CLI user's route.
6. Remove `settings::paths`: **yes**, pre-tag (2.0-born, D-3); the unrecorded removal of v1.0.0's source-tree constants goes into
   the break register (R2: they pointed at the build machine's source tree — non-negotiable 1).
7. Version skew (an older `dtwc_cl` on the cluster rejects new keys, loudly): **accept**; document "rebuild on the cluster".
8. Blind CUDA and MATLAB parts get V-rows; CLI11's rendering of enum and double defaults is [inferred] — the golden test decides.
