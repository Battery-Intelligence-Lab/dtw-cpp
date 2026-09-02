# Handoff — 2026-09-02 parity / checkpoint / portability pass

Second pass of the day, after `handoff-2026-09-02-quality-campaign.md`.
Maintainer instructions: CasADi rule (same name everywhere), implement mid-fill
checkpoint, MATLAB is callable headless, portable F15 (or CMake-defined), ffmpeg
quality, clean up anything obsolete.

## Accomplished (evidence: `.claude/reports/2026-09-02-*.md`)

- **MEX HiGHS crash root-caused and fixed.** Same `.mexw64` crashed R2024b and
  passed R2025b; hosts differ only in private `MSVCP140.dll` 14.36 vs 14.40
  (constexpr `std::mutex`). `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` tree-wide for
  MSVC-ABI MATLAB builds, before `Dependencies.cmake`. MATLAB MIP + PDLP routes run
  on both releases. `tests/matlab` registered as CTest `matlab_suite`
  (`DTWC_MATLAB_SUITE_MIN_PASSED=121`, Incomplete fails outside an allow-list).
  `[BLOCKED-ENV]` retired. (`mex-highs/`, `fixes-s3-s10`)
- **Mid-fill checkpoint** (`Problem::checkpoint`, rows between saves, one
  generation retained, metric from the Problem's configuration, CLI
  `--checkpoint-interval`). Found and fixed: brute-force `resize(N)` wiped restored
  matrices, so resume never worked (15 s → 0.15 s). (`midfill-checkpoint`,
  `adversarial-checkpoint-f15`, `checkpoint-fixes`)
- **F15 portable**: `genrand_res53` integer conversion, one fingerprint per
  schedule, verified byte-identical across clang/MSVC-STL/libstdc++ and FP flags.
  (`f15-portable`)
- **YAML CLI config removed** (TOML only; loader overrode CLI flags). (`yaml-removal`)
- **Parity**: `skip_rows` in `load()` and `lr_max_nodes` in all three languages;
  Python Tier-1 re-routed through C++ (`device`, `_read_data` → `dtwc::Data`
  single copy, GIL released, `UndefinedScore` bound, byte-identical CSVs incl.
  UTF-8 names); MATLAB `cluster.m` = parsing + one MEX call into `dtwc::cluster`
  (10/10 methods, F18/F40 closed, ragged cell sources, checkpoint options,
  checkpoint `metric`). (`parity`, `python-parity`, `python-parity-2`,
  `python-fixes`, `matlab-parity`, `matlab-fixes`, `adversarial-python`,
  `adversarial-matlab`)
- **C++ Tier-1 side effects removed**: Lloyd persisted CSVs to CWD-relative
  `./results` and printed unconditionally; `Result::distance_matrix()` added;
  `lr_max_nodes` → `int64_t`; UTF-8 names end to end (`path_to_utf8`/
  `utf8_to_path`). (`tier1-side-effects`)
- **Obsolete sweep**: compiled tree clean; stale skills/TODO/style docs fixed;
  `.mailmap` for Kasper Westman. (`obsolete-sweep`)
- **YAML reinstated the right way** (maintainer asked): `--config` accepts TOML
  or YAML, sniffed from content and parsed into CLI11 `ConfigItem`s by
  `dtwc/cli/config_file.hpp` (fkYAML v0.4.4, MIT, `DTWC_ENABLE_YAML` default
  ON; caches from before today carry a stale OFF). Unknown config keys are now
  errors in both formats. (`yaml-cli11`, `adversarial-yaml-ram`)
- **Windows free-RAM bug**: `available_ram_bytes()` returned 0 so
  `StoragePolicy::Auto` never spilled; `GlobalMemoryStatusEx` added, decision
  extracted to `choose_storage`. (`available-ram`)
- **Library survey** (`ancillary-library-survey`): no other dependency
  justified; fast_float is the one candidate worth revisiting if the libc++
  floor is below LLVM 20 (floating `from_chars`).

## Final gates (serial, this tree)

| Gate | Result |
|---|---|
| `build/highs-1151` (YAML ON) | 131/131, 6 skips |
| `build/nollfio` (YAML ON) | 131/131, 9 skips |
| `build/arrow-pyarrow-23` (YAML ON) | 133/133, 8 skips |
| `build/cuda-verify` (`-R cuda`) | 4/4, none skipped |
| Python (`tests/python`) | 1112 passed, 16 skipped, 0 failed |
| MATLAB `matlab_suite` (R2024b, R2025b) | 124 run, 123 passed, 0 failed, 1 allow-listed incomplete |
| docs contract, repo/record hygiene, supply-chain pins (30 manifests, 7 archives), F22 C++ | PASS |

## Decisions

- `save_interval` = matrix rows between saves (deterministic, testable), not seconds.
- One generation per checkpoint directory (removed after `CURRENT` swap).
- Deleted `load(path, int, char, ...)` overloads close the char→int positional trap.
- YAML removed rather than fixed (maintainer: TOML only).
- Bindings call C++ Tier-1; Python keeps only HPC route, `elapsed_s`, `plot`.

## Open

- CLI `--name` from `argv` is still ACP-encoded on Windows (needs `wmain`).
- F22 Python mutation harness refuses a dirty tree; run after commit.
- Python GPU device-canonicalisation and MATLAB `gpu:N` forwarding are inferred
  (no CUDA Python module / CUDA MEX built here).
- `tests/python/test_hpc.py` binary discovery prefers `build/highs-1151`; the
  arrow binary needs pyarrow DLL dirs on PATH.
- `io.py::load/save_dataset_csv` and `Dataset.series_names()` are Python-only.
- `docs/public` generated site not regenerated.
