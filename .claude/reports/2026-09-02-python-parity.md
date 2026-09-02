# Python Tier-1 drift closure (items 1–7)

Date: 2026-09-02. Branch `Claude`. Build dir `build/cfg-gate-normal` (clang,
Release, `DTWC_BUILD_PYTHON=ON`); `.pyd` + all `python/dtwcpp/*.py` copied into
`.venv/Lib/site-packages/dtwcpp/` (non-editable install) before every run.

## Outcome

All seven drift items are closed, each with a test written red first.
**Baseline** (before any change, fresh `.pyd`): `uv run pytest tests/python -q`
→ **1053 passed, 12 skipped, 0 failed**.
**After**: **1086 passed, 16 skipped, 0 failed** (+33 passed, +4 skipped — the
four GPU-gated canonicalisation cases skip on this CUDA-OFF box).
`scripts/check_docs_contract.py` → `generated documentation is current` /
`documentation contract checks passed`. `scripts/generate_docs.py` →
`generated documentation updated`. No contract pin needed updating.

## Route decision

**Minimal correct fixes + bind the two C++ pieces Python was re-implementing**
(`dtwc::device()` and the file parser), NOT the full re-route to bound
`dtwc::load`/`dtwc::cluster`. Justification:

* Python `cluster()` owns four things C++ Tier-1 does not have: the SLURM/HPC
  route, `elapsed_s`/`summary()`, `plot()`, and `Result.distance_matrix` (a
  documented §1.4 attribute). C++ `Result` exposes none of these — it holds a
  private `shared_ptr<Problem>` with no accessor — so binding `dtwc::cluster`
  would require new public API in `dtwc/api.hpp`, which I do not own and other
  agents are editing live.
* Python `_api.py`/`_clustering.py` also carry behaviour C++ does not: strict
  `int` domain normalisation (`_normalize_tier1_int`, rejecting `bool`/`np.bool_`
  /float/str before any side effect), the sklearn estimator, and numpy
  conversion. Those are Python-language obligations, not drift.
* What *was* duplicated — the path parser and the device store — is now gone:
  `Dataset.as_series()` calls the C++ `DataLoader` and `device()` calls
  `dtwc::device()`. **No Python parser for path sources remains** (`np.loadtxt`
  and `np.genfromtxt` no longer appear anywhere in `python/dtwcpp/`).

Two new binding symbols in `python/src/_dtwcpp_core.cpp` (my file; no `dtwc/`
edit was needed):
`device(name)` / `device()` → `dtwc::device`, and private
`_read_series(source, skip_cols=0, skip_rows=0, delimiter="")` which drives
`dtwc::DataLoader` with the same configuration `Dataset::materialize_local()`
uses (`api.cpp`) and rewraps a non-`dtwc::Error` failure as `IOError`.

## Per item

| # | Drift | Fix | Test | Before → After |
|---|---|---|---|---|
| 1a | `device(name)` returned the caller's string | `device()` returns `_core_device(name)` = `dtwc::device(name)` | `test_device.py::TestCanonicalDeviceName::test_returns_the_cpp_canonical_name`, `test_canonical_name_is_a_fixed_point`, `test_gpu_aliases_canonicalise` (GPU-gated) | `device("cuda:0")` → `"cuda:0"` → `"gpu"` |
| 1b | module global `_DEFAULT_DEVICE` | deleted; the getter reads `dtwc::Env` through `dtwc::device()`. Only `_HPC_SELECTED` (a bool) remains | `test_cpu_getter_reads_the_shared_env` | two stores → one |
| 2 | `skip_cols` applied after `np.loadtxt`; ragged rows failed | path sources parse through `_read_series` → `DataLoader` | `test_api.py::TestLoad::test_path_source_parses_a_non_numeric_id_column`, `test_path_source_supports_ragged_rows` | `ValueError: could not convert string 'alpha'` / `number of columns changed from 3 to 2` → both load |
| 3 | in-memory `skip_cols` ignored | rows sliced after `skip_rows`, with C++'s `skip_cols > len(row)` rejection | `test_in_memory_source_honours_skip_cols`, `test_in_memory_skip_cols_beyond_series_length_is_rejected` | value silently kept → columns erased / `InvalidInput("load: skip_cols exceeds an in-memory series length.")` |
| 4 | `Result.score` unusable after a matrix-free run | `Result` retains the `Problem` (as C++ `Result` does) and `_scoring_problem()` calls `fill_distance_matrix()` on demand; `save()` then writes all four CSVs | `TestMatrixFreeScoring` (15 cases: 3 methods × 4 scores, plus save/unknown-score/hpc) | `InvalidInput` → score equals an independently computed `compute_distance_matrix` oracle to 1e-12 |
| 5 | `k <= N` guard missing | `cluster()` raises the exact C++ messages after materialisation | `TestClusterLocal::test_k_above_series_count_is_rejected`, `test_empty_dataset_is_rejected` | crashed downstream → `InvalidInput("cluster: k must not exceed the number of series.")` / `"cluster: dataset is empty."` |
| 6 | `Problem::checkpoint` unbound | `.def_rw("checkpoint", …, nb::rv_policy::reference_internal)`; `CheckpointOptions` docstrings rewritten (rows-between-saves, the three `InvalidInput` conditions, the Pruned downgrade) | new `tests/python/test_checkpoint_options.py`, 8 cases | `AttributeError` → in-place mutation, whole-struct assignment, defaults, end-to-end fill + resume, both validation errors |
| 7 | `get_device()` Python-only | kept as a **deprecated alias** (it is in `__all__`, so removal would breach §4) emitting `dtwcpp.get_device is deprecated; use dtwcpp.device`; registered in the F22 policy fixture; all four internal/test call sites repointed to `device()` | `test_deprecation_policy.py[get_device]` + the hygiene scan | silent → warns once, caller-attributed |

`F22_PYTHON_GATE alias_symbols=13 operations=14 primary_warn_once=14
canonical_silent=14 equivalent=14 class_routes=3 identity=2 ordinary_legacy=0
verdict=PASS` (was 12/13/13/13/13).

### Item 6 correction to the brief

The brief expected "exactly 4 generations under `directory/generations`" for
N=4, `save_interval=1`. The C++ reference does not do that: `save_checkpoint`
publishes a generation that *supersedes* the previous one, so a completed fill
leaves **exactly one** (`tests/unit/unit_test_checkpoint.cpp:368-394`, N=5,
interval=1, `REQUIRE(count_generations(ckpt_dir) == 1)`; contract §2.7 "A
directory holds exactly one generation after a successful save"). Measured here
for N=4 at interval 1 with strategy Auto, BruteForce and Pruned: 1, 1, 1
**[confirmed]**. The Python test therefore asserts the C++ end state (one
generation, `is_distance_matrix_filled()`, bit-identical restore into a fresh
`Problem`) plus a resume test pinning the "resize is a wipe" fix.
**Limitation [confirmed]:** because each publication supersedes the last, no
Python-visible artifact distinguishes N mid-fill saves from one final save;
that distinction is only gated in C++.

### Item 1b — the part I did NOT change, deliberately

`dtwcpp.device("hpc")` still leaves `Env` on CPU. `Env::set_device("hpc")` *is*
the eager `.env` + SSH auth probe (`env.cpp:331-359`); calling it would make
`device("hpc")` raise `DeviceError` on any box without credentials. Contract
§1.1 **explicitly documents** this as a Python difference ("HPC credentials
deferred to the wrapper"), which the brief's own rule exempts. Recorded, not
silently fixed.

## Extra item from the coordinator

`tests/python/test_hpc.py::TestLocalRoundTrip::test_required_input_message_names_toml_first`
pinned `"…(TOML; YAML if built with DTWC_ENABLE_YAML)\n"`; the source of truth
is now `dtwc/dtwc_cl.cpp:1016`
`"Error: --input is required via CLI or config file (TOML)\n"`. Pin updated to
that exact string. It is **not** skipped here: `_hpc.find_dtwc_binary` resolves
`build/highs-1151/bin/dtwc_cl.exe` and the class runs — `-k TestLocalRoundTrip`
→ **5 passed** against the real binary.

## Docs

`docs/api-contract-2.0.md`: Python column of §1.1 (Returns, Delegates to),
§1.2 (`skip_cols`), §1.3 (`k`), §1.4 (`score`), and the §2.7 options-struct row.
`scripts/generate_docs.py` regenerated `docs/content/**`; that run also picked
up **other agents' in-flight source edits** (`tier-2.md`, `cli.md`,
`configuration.md`, `contributing/*`, `math/lr-core.md`,
`method/missing-data.md`, `interface-parity.md`). I hand-edited none of those.

## Proposed CHANGELOG bullets (Unreleased) — not applied

```
- Python `dtwcpp.device()` now returns the canonical name `dtwc::device()`
  returns, so `device("cuda:0")` reports `"gpu"` as C++ and MATLAB do, and the
  module-level device copy is gone: `dtwc::Env` is the single store for every
  local selection (`hpc` stays Python-side because its credential check is
  deferred to submit time).
- Python `Dataset.as_series()` parses a path with the C++ `DataLoader` instead
  of `np.loadtxt`, so `skip_cols` drops leading FIELDS before numeric parsing (a
  text id column now loads) and variable-length rows are preserved. In-memory
  sources now honour `skip_cols`, rejecting a value larger than a series length
  with `InvalidInput`, matching the C++ in-memory `load()` overload.
- Python `cluster()` enforces the C++ guards `cluster: dataset is empty.` and
  `cluster: k must not exceed the number of series.`
- Python `Result` retains the clustered `Problem`, so `score()` and `save()`
  work after a matrix-free run (`onebatch`/`clara`/`tadpole`): the distance
  matrix is filled on demand exactly as C++ `Result::score()`/`save()` do, and
  `save()` writes all four CSVs instead of two.
- `Problem.checkpoint` is now bound in Python (`CheckpointOptions` view, so
  `prob.checkpoint.enabled = True` mutates the Problem), making automatic
  mid-fill checkpointing reachable from Python.
- `dtwcpp.get_device()` is deprecated; use `dtwcpp.device()`.
```

## Unresolved / residual

* `Result.save()` propagates the silhouette failure where C++ catches
  `UndefinedScore`, warns and skips the file (`api.cpp:275-289`). Python cannot
  replicate this: `UndefinedScore` derives from `InvalidInput` and is not a
  separate bound exception, and catching all `InvalidInput` would swallow real
  errors. Fixing it means adding a leaf to the frozen §6 taxonomy — a decision
  above this task. Reachable case: `save()` on a k=1 result.
* `python/dtwcpp/io.py::load_dataset_csv` / `save_dataset_csv` have **no C++ or
  MATLAB counterpart at all** (no `dtwc::load_dataset_csv`, absent from the
  contract) — a Python-only public surface, i.e. a CasADi-rule item nobody owns.
* Ragged **in-memory** sources still go through `np.asarray(..., dtype=float)`
  and fail; C++ accepts a ragged `vector<vector<double>>`. Not in the brief's
  list; left untouched to avoid changing the error type of malformed input.
* Series names: C++ `Result::save` writes `problem_->series_name(i)` (real
  names from the loader), Python writes `str(i)`. Item 3 of the earlier drift
  list; out of scope here and still open.
* The AGENTS.md Python floor (1033/1035 passed, 1 expected F39 red) is stale
  again: this session observed **1086 passed / 16 skipped / 0 failed** with no
  F39 red. Someone should reconcile it.

## The claim I most expect to be wrong

That the device canonicalisation is right for **GPU ordinals**. This box has no
GPU (`CUDA_AVAILABLE False`, `METAL_AVAILABLE False`), so
`test_gpu_aliases_canonicalise` **skipped** — the `"cuda:0" → "gpu"` and
`"gpu:1" → "gpu:1"` behaviour is **[inferred]** from `api.cpp:43-49`
(`canonical_device_name`) and `env.cpp:285-328`, confirmed only for `cpu`.
A GPU build running that one test would settle it.
