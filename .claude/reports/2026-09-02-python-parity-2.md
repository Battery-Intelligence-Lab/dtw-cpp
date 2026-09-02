# Python Tier-1 residual drifts (items 1–3) — closed

Date: 2026-09-02. Branch `Claude`. Build `build/cfg-gate-normal` (`_dtwcpp_core`
target); `.pyd` + `python/dtwcpp/*.py` copied into
`.venv/Lib/site-packages/dtwcpp/` before every run. CLI evidence:
`build/highs-1151/bin/dtwc_cl.exe`.

## Outcome

All three residual drifts from `.claude/reports/2026-09-02-python-parity.md`
are closed, each with a test written red first (14 new nodes, all red on the
pre-change `.pyd`, all green after).

**Baseline** (fresh `.pyd`, before any change): `uv run pytest tests/python -q`
→ **1086 passed, 16 skipped, 0 failed**.
**After**: **1100 passed, 16 skipped, 0 failed** (+14), both with
`-p no:randomly` and with the default random ordering.
`scripts/check_docs_contract.py` → `generated documentation is current` /
`documentation contract checks passed`; `scripts/generate_docs.py` →
`generated documentation updated`.
`scripts/test_f22_cpp_deprecations.py --build-dir build/cfg-gate-normal` →
`F22_CPP_DIAGNOSTICS inventory=33/33 … verdict=PASS`.
`F22_PYTHON_GATE alias_symbols=13 operations=14 … verdict=PASS` (unchanged —
`UndefinedScore` is a new canonical symbol, not an alias).

## Per item

| # | Fix | Test | Before → After |
|---|---|---|---|
| 1 | `dtwcpp.UndefinedScore` bound as `PyErr_NewException("dtwcpp.UndefinedScore", g_exc_invalid, …)` (single base = the bound `InvalidInput`, so MRO is `UndefinedScore → InvalidInput → DtwcError → ValueError`); translator branch placed **before** `InvalidInput` (it derives from it). Exported in `__init__.py`/`__all__`. `Result.save` catches it, `warnings.warn(f"silhouettes skipped: {e}", RuntimeWarning)` (stderr) and skips the file, as `api.cpp:281-299` does | `test_api.py::TestSaveUndefinedSilhouette` (4 nodes) + `test_contract_parity.py::test_error_hierarchy` | `save()` on k=1 raised `InvalidInput: silhouette requires at least 2 non-empty clusters…` → writes labels/medoids/matrix, warns, skips silhouettes; `score("silhouette")` raises `UndefinedScore` |
| 2 | `_float_rows()` in `_api.py`: ndarray and rectangular sources keep the NumPy path, a ragged sequence (which `np.asarray(..., dtype=float)` rejects) is converted row by row. `skip_rows`/`skip_cols` unchanged, applied per series | `TestRaggedInMemorySource` (5 nodes) | `ValueError: setting an array element with a sequence…inhomogeneous shape` → `cluster()` runs; labels and medoids **digit-identical** to the C++ route (`Problem.set_data` + `compute_distance_matrix` + `fast_pam_seeded`) |
| 3 | Binding `_read_series` → `_read_dataset`, returning `(p_vec, p_names)` (needs `nanobind/stl/pair.h`). `Dataset._materialize()` caches both; new `Dataset.series_names()`; `cluster()` passes the loader names to `Problem.set_data` and `Result`. `Result.save` writes them, plus C++'s line endings (platform newline for the three text-mode files, LF for the binary-mode matrix), `:.8g` silhouettes (`setprecision(8)`) and `:.17g` matrix values (`to_chars(general, max_digits10)`) | `TestSeriesNames` (5 nodes), incl. a real-CLI byte-identity run | Python wrote `0,1,2,…` with LF, `%.18e` matrix and `repr(float)` silhouettes → all four CSVs **byte-identical** to `dtwc_cl -i … -k 2 -m pam` |

Loader names verified against the source and the real binary: batch file →
**1-based row number** (`fileOperations.hpp:422`), folder → **file stem**
(`fileOperations.hpp:353`), in-memory → **0-based ordinal**
(`api.cpp:178-179`). CLI output confirmed CRLF + 8-significant-digit
silhouettes by `od -c` on `named_labels.csv`/`named_silhouettes.csv`.

### Scope note (beyond the brief)

The brief asked for byte-identity on labels + medoids. `_silhouettes.csv` and
`_distance_matrix.csv` were also divergent (number format and newline), so both
were fixed and the test asserts all four files. This is the §1.4 "identical
bytes in every language" clause; it is the one change here that was not
explicitly requested.

## `io.py::load_dataset_csv` / `save_dataset_csv`

Still no C++/MATLAB counterpart (Python-only public surface, in `__all__` and
in `docs/content/getting-started/python.md`). They are *not* what `_read_dataset`
does: `save_dataset_csv` writes an `np.savetxt` matrix with an optional header
row; `load_dataset_csv` returns `(N,L) ndarray, header names)` and auto-detects
whether row 0 is a header — neither the header round-trip nor the name return
exists in `DataLoader`. Its **numeric parser was** duplicated, so it is deleted:
header detection stays in Python (one `csv.reader` row), the rows are now parsed
by `_dtwcpp_core._read_dataset(path, 0, skip_rows, ",")`. `tests/python/test_io.py`
(all CSV round-trips) and `convert.py`'s use of it stay green.

## Proposed CHANGELOG bullets (Unreleased) — not applied

```
- Python binds `dtwcpp.UndefinedScore` (a subclass of `dtwcpp.InvalidInput`, as
  `dtwc::UndefinedScore` is of `dtwc::InvalidInput`), and `Result.save()` now
  mirrors C++: with fewer than two realised clusters it warns and skips
  `<name>_silhouettes.csv` instead of failing a clustering that succeeded.
  `Result.score("silhouette")` still raises.
- Python `dtwcpp.load()` accepts ragged in-memory data — a list of 1-D
  sequences of different lengths — matching the C++ `load(series_type)`
  overload; rectangular arrays keep the NumPy fast path.
- Python `Result.save()` writes the dataset's own series names (file stem per
  file for a folder source, 1-based row number for a batch file) instead of
  ordinals, and matches the CLI's number formatting and line endings, so a
  Python run and a `dtwc_cl` run on the same file now produce byte-identical
  `_labels.csv`, `_medoids.csv`, `_silhouettes.csv` and `_distance_matrix.csv`.
- `dtwcpp.io.load_dataset_csv` parses its numeric rows with the C++ DataLoader
  instead of its own CSV parser (header detection unchanged).
```

## Unresolved / residual

* `Dataset.series_names()` has **no C++ counterpart on `Dataset`** — C++ exposes
  names only through `Problem::series_name(i)` after materialisation. The
  Python accessor is the minimum needed to carry them into `Result`; if strict
  CasADi symmetry is wanted, `dtwc::Dataset` would need the same accessor.
* The contract's `_api.py:NNN` line pins that my edits shifted were refreshed
  (`:42`, `:143`, `:289-326`, `:295-299`). Two pins are mirrored verbatim in
  `scripts/check_docs_contract.py` (`_api.py:102-107`, `_api.py:281-301`).
  Both are exact at `HEAD` and were already stale in the working tree I
  inherited (the previous uncommitted session shifted them); mine shifted them
  further, to 151 and 372. Updating them means editing the marker list in
  `scripts/check_docs_contract.py`, which the brief scopes out and another agent
  is currently holding, so I left both alone. They name code I did not change.
* `dtwcpp.io`'s CSV/HDF5/Parquet helpers remain a Python-only public surface
  (CasADi-rule item nobody owns).
* The AGENTS.md Python floor is stale again: **1100 passed / 16 skipped /
  0 failed**, no F39 red on this box.

## The claim I most expect to be wrong

That `f"{v:.17g}"` reproduces `std::to_chars(general, max_digits10)` for **every**
double, not just the ones in the fixture. They agree on the 144 values of the
byte-identity test **[confirmed]**; agreement on subnormals and on values that
choose scientific notation is **[inferred]** from both being shortest-round-trip
general formatting at 17 significant digits. A differential fuzz over random
doubles would settle it. Distances are non-negative and O(1) here, so the
exponent branch is rarely exercised.
