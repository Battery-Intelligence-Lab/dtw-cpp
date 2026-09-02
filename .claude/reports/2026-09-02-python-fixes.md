# Python adversarial fixes — F1–F10 (2026-09-02)

Base: uncommitted branch `Claude`. Rebuilt `build/cfg-gate-normal` → copied
`_dtwcpp_core.cp313-win_amd64.pyd` into `.venv`; verified `_read_data` present
and `_read_dataset` gone before any test run.

## Per item

**F2 GIL / item 1.** `_read_dataset` replaced by `_read_data` with
`nb::gil_scoped_release` around the whole `DataLoader` read
(`python/src/_dtwcpp_core.cpp:227`). Also released around
`m.def("device", name)` (the other site F2 named; `Env::set_device` probes GPU
/`.env`/`sinfo`). **Measured** (2000×500 CSV, spinner thread): ticks in a second
thread during the 0.202 s read = **6 736 673** (review measured **3** before).

**Item 2 — single-copy route.** `_read_data(path, skip_cols, skip_rows,
delimiter) -> dtwc::Data`; `Data` already exposes `p_vec`/`p_names`, so no new
accessor was needed. `Dataset` now caches the `Data` (`_as_data`/`as_data`);
`as_series()`/`series_names()` are lazy, uncached views off it. `cluster()`
calls `prob.set_data(Data)` and, on cpu, fills through
`prob.distance_matrix()` — the route the CLI uses — instead of
`compute_distance_matrix(list)`. GPU backends still get a transient `p_vec`.
`io.py:82` switched; `_read_dataset` deleted (no other users).

Measured, same binary, interleaved ×4, read + `set_data` on a 2000×500 CSV
(13.2 MB, 10⁶ doubles):

| route | wall | `tracemalloc` peak |
|---|---|---|
| via `as_series()` + `set_data(list, names)` | 0.393–0.406 s | 32.24 MB |
| via `as_data()` + `set_data(Data)` | 0.194–0.198 s | **0.00 MB** |

Pre-change `.pyd` on the same file measured 0.721–0.891 s / 32.24 MB (machine
is shared; the same-binary row is the decisive one). `tracemalloc` counts only
Python allocations — the 8 MB payload now lives solely in the C++ `Data`.
Arbiter for the new cpu matrix route: `res.distance_matrix` is digit-identical
to `compute_distance_matrix` at band −1 and band 5 (new test
`TestCpuMatrixRoute`).

**F1 non-ASCII / item 3.** The C++ UTF-8 change **has landed**: `dtwc_cl.exe`
in `build/highs-1151` writes `caf\xc3\xa9`, and `load()` returns `café`. No
xfail needed. `Result.save` now opens the three text files with
`encoding="utf-8"` (default newline, so `\n`→`\r\n` exactly as C++'s text-mode
`ofstream`). New fixture `TestNonAsciiSeriesNames`: `café/beta/gamma/delta`
folder, all four CSVs byte-identical to `dtwc_cl.exe`. Existing ASCII
byte-identity test still passes.

**F3 F22 pins / item 4.** `PASS_MARKER` → `alias_symbols=13 operations=14
primary_warn_once=14 canonical_silent=14 equivalent=14 …`; focused ledger
18→19; every mutant `(failed, passed)` pair re-pinned to sum to 19 (21 of 26
literal pairs bumped; the failing set is unchanged because no mutant touches
the new `get_device` alias). **The script cannot be run**: `assert_clean_targets`
(:575) runs `git diff --quiet HEAD -- python/src/_dtwcpp_core.cpp
python/dtwcpp/__init__.py python/dtwcpp/_api.py`, the whole campaign's Python
work is uncommitted, and the only flags are `--build-dir/--mode/
--confirm-exclusive-build-access` — no override. Verbatim:
`F22_PYTHON_MUTATION_HARNESS_ERROR mutation targets have staged or unstaged changes`.
I am forbidden to commit. Everything else was exercised directly against the
edited sources: `assert_inventory` 31/31 (every mutant needle still
materialises), `find_artifacts` OK, all pairs sum to 19, and the pinned marker
occurs exactly once in the live `pytest -q -s test_deprecation_policy.py`
stdout with `19 passed` → `F22 PIN CHECK verdict=PASS`.

**F4 pin drift / item 5.** Re-anchored all 11 Python pins in
`docs/api-contract-2.0.md` (+ mirrors in `check_docs_contract.py`):
`_api.py:281-301`→`413-433` (`_normalize_method`), `_api.py:102-107`→`174-180`
(`medoid_indices`), `__init__.py:306-326`→`319-339`, `__init__.py:213-238`→
`214-242`, `135-163`→`140-168`, `_api.py:142`→`160-172`, `143`→`153`,
`289-326`→`330-367`, `295-299`→`336-340`. **Guard strengthened**: new
`assert_python_line_pins()` reads each range *out of the contract* and asserts
the pinned window really contains the named identifier (8 pins), so document
and script cannot drift apart. Verified discriminating: restoring the stale
`102-107` makes it fail with
`contract pin python/dtwcpp/_api.py:102-107 no longer contains 'def medoid_indices'`.

**F5 / item 6.** `Result.distance_matrix` is now a property that fills through
the retained `Problem` on first read and caches, matching
`dtwc::Result::distance_matrix()`. `_scoring_problem`/`save` read the raw
`_distance_matrix` so laziness is unchanged; `plot()` now works after a
matrix-free cpu run. Tests re-anchored to `_distance_matrix is None` (nothing
materialised) plus new `test_distance_matrix_fills_on_demand_after_a_matrix_free_run`
and `test_plot_works_after_a_matrix_free_cpu_run`.

**F6/F8 / item 7.** `save()` no longer warns: it prints
`Warning: silhouettes skipped: <what()>` to `sys.stderr`, the exact C++ text
(`api.cpp:284-289`); the test now runs `save()` under
`warnings.simplefilter("error")`. NaN → empty field, ±inf → `InvalidInput`
before the matrix file is opened. Oracle is the **C++ writer itself**
(`Problem.write_distance_matrix`): bytes identical for NaN, and the inf message
string-equal (`distance-matrix CSV: computed non-finite value at row 0, column 1.`),
with `_labels.csv`/`_medoids.csv` already on disk as in C++.

**F9 / item 8.** Added `test_no_python_side_copy_of_the_local_device`. Honest
caveat: every *behavioural* discriminator (cuda/gpu:N → `gpu`) needs a GPU and
skips here, so the new assertion is **structural** — it fails on the pre-change
code because that code answered `device()` from the module global
`_DEFAULT_DEVICE` which the new code does not define.

**F10 / item 9.** Covered in the CHANGELOG bullets below.

## Gates

- `uv run python -m pytest tests/python -q` → **1112 passed, 16 skipped,
  0 failed** (was 1110/16 before the two added tests).
- `uv run python scripts/check_docs_contract.py` → `generated documentation is
  current` + `documentation contract checks passed`.
- `uv run python scripts/generate_docs.py --check` → current (I had to run
  `generate_docs.py` once: `tier-1.md` and `migration.md` were stale, the
  latter from my re-anchored pins).

## Proposed CHANGELOG (Unreleased)

- `dtwcpp.load()` now parses paths through the C++ `DataLoader` into an owning
  `dtwc::Data` handed straight to `Problem.set_data`, creating no Python
  floats: a 2000×500 CSV drops from 32.2 MB of `tracemalloc` peak to 0 and
  from ~0.40 s to ~0.20 s. A single-column CSV is consequently N series of one
  point each, matching the CLI, where `np.loadtxt` previously gave 1 series of
  N points.
- `Result.distance_matrix` fills on demand from the retained `Problem` after a
  matrix-free `onebatch`/`clara`/`tadpole` run, matching
  `dtwc::Result::distance_matrix()`; `plot()` therefore works for those runs.
- `Result.save` writes UTF-8, so non-ASCII series names are byte-identical to
  `dtwc_cl`; it reports a skipped silhouette on stderr like C++ instead of
  raising under `-W error`, writes an empty field for a NaN distance and
  raises `InvalidInput` for ±inf.
- The Tier-1 file read releases the GIL.

## Unresolved

1. The F22 mutation harness cannot execute until the Python work is committed
   (clean-tree guard, no override flag). Pins are verified by construction, not
   by a mutant run.
2. F9's new assertion is structural, not behavioural; item 1a of the device
   drift still has no executable coverage on a no-GPU box.
3. `tests/python/test_hpc.py::_local_binary` now prefers
   `build/highs-1151/bin/dtwc_cl.exe` like `test_api.py`. It previously took
   the newest build, and a concurrent agent's `build/arrow-pyarrow-23` rebuild
   (15:38) made 5 tests fail with `0xC0000135` (that binary needs the pyarrow
   DLL dirs on PATH; `--help` alone exits 127). Environmental, not a code
   regression — but worth a permanent fix in `_hpc.find_dtwc_binary`.
4. The claim I most expect to be wrong: that filling the cpu matrix through
   `Problem.distance_matrix()` is behaviour-neutral. It is digit-identical on
   the banded and unbanded fixtures and to `dtwc_cl` bytes, but it is a
   different C++ code path from `compute_distance_matrix` (mmap-capable,
   no `use_pruning` knob).
