# Adversarial review — uncommitted Python parity work (2026-09-02)

Scope: `git diff -- python tests/python docs/api-contract-2.0.md` plus
`.claude/reports/2026-09-02-python-parity{,-2}.md`. Read-only; the `.pyd`
(14:45) is newer than `_dtwcpp_core.cpp` (14:44), so no rebuild.
`pytest tests/python/{test_api,test_device,test_contract_parity,test_checkpoint_options}.py`
→ **523 passed, 5 skipped**; `test_deprecation_policy.py` → 19 passed.

## Findings (ranked)

**F1 — HIGH / CONFIRMED. Non-ASCII file names crash `load()` for folder sources.**
`python/src/_dtwcpp_core.cpp:227-247` returns `p_names` as `std::vector<std::string>`;
nanobind decodes strictly as UTF-8, but `DataLoader` builds a folder-source name
from `path::stem().string()`, which on Windows is native ANSI. Repro: a folder
holding `café.csv`, `beta.csv`, `gamma.csv`, `delta.csv` →
`dtwcpp.load(dir).as_series()` raises
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 3` at
`python/dtwcpp/_api.py:84`, while `dtwc_cl -i <dir>` succeeds and writes
`caf\351,0` into `_labels.csv`. This is a **regression**: the old `np.loadtxt`
path named series `str(i)` and never touched the stem. Even after a decode fix,
`Result.save` opens the three text files with `open(..., "w")`
(`_api.py:243,252,283`), i.e. the locale encoding, so CLI byte-identity holds
only for ASCII names. Fix: emit UTF-8 names from C++ (or decode with
`surrogateescape`) and give the writers an explicit encoding.

**F2 — MEDIUM / CONFIRMED. `_read_dataset` never releases the GIL.**
`_dtwcpp_core.cpp:227` has no `nb::gil_scoped_release`, unlike the 50 other
sites — including the comparable I/O bindings `save_checkpoint`/`load_checkpoint`
(:1347/:1363). Measured on a 25.5 MB, 2000×500 CSV: during the 0.351 s read a
second Python thread advanced **3** ticks; during a 2.553 s
`fill_distance_matrix` (which releases) it advanced **1 043 528**. `device(name)`
(:216) likewise holds the GIL across `Env::set_device`'s GPU probe. No new lock
is added anywhere, so the lock-free rule is respected; `_HPC_SELECTED` and the
`Env` mutation are GIL-serialised (process-wide device state is semantically
racy by design, exactly as in C++).

**F3 — MEDIUM / CONFIRMED. The F22 Python mutation gate is broken by the new alias.**
`tests/python/test_deprecation_policy.py:417` now emits
`alias_symbols=13 operations=14 primary_warn_once=14 canonical_silent=14 equivalent=14`
(captured from the live run), but `scripts/test_f22_python_deprecation_mutations.py:26-30`
still pins `12/13/13/13/13` and requires `stdout.count(PASS_MARKER) == 1` (:797),
so the harness's baseline can never match. It refuses to run against a dirty
tree (`F22_PYTHON_MUTATION_HARNESS_ERROR`), so this is a source comparison, not
an execution. The policy behaviour itself is correct: `get_device()` emits one
caller-attributed `DeprecationWarning` (`stacklevel=2`).

**F4 — MEDIUM / CONFIRMED. Silent contract pin drift (brief item 9).**
`docs/api-contract-2.0.md:613` and `scripts/check_docs_contract.py:329` pin
`Result.medoid_indices` to `_api.py:102-107`; those lines now hold
`Dataset.as_series`/`series_names` (the property moved 102 → 151).
`docs:149` / `script:332` pin `_normalize_method` to `_api.py:281-301`; those
lines now hold the tail of `Result.save` (moved 281 → 372). The guard only
asserts the literal string is present, so it passes while misdirecting. Four
sibling pins (`:42`, `:143`, `:289-326`, `:295-299`) were refreshed and verified
correct, so the document is now internally inconsistent.

**F5 — MEDIUM / CONFIRMED. `Result.distance_matrix` stays `None` after the lazy fill.**
C++ has `Result::distance_matrix()` (`dtwc/api.hpp`: "Fills the matrix first if
it is not yet materialised, exactly as score() does"). Python fills the retained
`Problem` but never publishes it: after
`r = cluster(X, k=2, method='clara'); r.score('inertia')`, `r.distance_matrix is
None`, and `r.plot()` prints `no local distance matrix to plot (cpu run)` and
returns `None` (`_api.py:294-301`) — a wrong message for a cpu run and a
capability C++ now has.

**F6 — LOW-MEDIUM / CONFIRMED. `save()` raises under `-W error`.**
`_api.py:275` warns `RuntimeWarning`; C++ writes to `std::cerr` and returns
(`dtwc/api.cpp:284-289`). `python -W error -c "... r.save(d)"` →
`RuntimeWarning: silhouettes skipped: …` propagates and no file is written.
No `filterwarnings=error` exists in the repo config today, so no gate breaks —
but it is a divergence for warning-strict callers.

**F7 — JUDGEMENT / measured (brief item 2). Three full materialisations; not pathological.**
2000×500 (10⁶ elements, 25.5 MB CSV): `_read_dataset` **0.348–0.535 s**,
`Problem.set_data` back-conversion **0.005–0.007 s**; rebuilding the same 10⁶
`PyFloat`s costs **0.055 s**, so Python-object churn is ~15 % of the read and
parsing dominates. The real cost is memory and retention: ~32 MB of
PyFloat + list slots for an 8 MB payload (4×), cached on `Dataset._series` for
the handle's life, plus a third conversion inside
`compute_distance_matrix(series, …)` (`_api.py:539`). A single-copy route exists
and neither report considers it: `dtwc::Data` is already bound
(`_dtwcpp_core.cpp:754-782`) and `Problem.set_data(Data)` (:1026) moves it, so
`_read_dataset` could hand back a `Data` and create no Python floats at all.
Not merge-blocking at this size; it matters at the 100M-series target.

**F8 — LOW / CONFIRMED. Non-finite CSV divergence.** `_api.py:268-271` writes
`f"{v:.17g}"` unconditionally (NaN → `nan`, inf → `inf`). C++
`dtwc/core/matrix_io.hpp:67-89` writes an **empty field** for NaN (uncomputed)
and `preflight_distance_matrix_csv` (:30-64) throws `InvalidInput` on ±inf
before opening the file. Reachable only through a caller-supplied
`Result(distance_matrix=…)`; verified by writing a NaN matrix.

**F9 — LOW / CONFIRMED (brief item 10, spot-check 1). Non-discriminating device tests.**
Every assertion in `TestCanonicalDeviceName` (`tests/python/test_device.py:192-222`)
also passes on the pre-change code: old `device("cpu")` returned `"cpu"` and
`_sync_env` already wrote `Env`. All discriminating cases live in the GPU-gated
`test_gpu_aliases_canonicalise`, which **skips** on this box, so drift item 1a
has zero executable coverage here (report 1 disclosed the gap honestly).
Spot-check 2 — `test_saved_labels_carry_the_file_names` (`test_api.py:600`) —
is genuinely discriminating (pre-change names were 0-based ordinals).

**F10 — LOW / CONFIRMED. Undocumented behaviour change.** A single-column CSV
was 1 series of N points under `np.loadtxt`; it is now N series of 1 point each
(matching the CLI, so the direction is right) — absent from the proposed
CHANGELOG. Pre-existing Python-only leniency survives: `[[1, None]]` silently
becomes NaN, `[['1','2']]` parses, and a 1-D in-memory source raises a bare
`TypeError: 'numpy.float64' object is not iterable`; the C++ signature admits
none of the three.

## Refuted attacks

* **Byte identity (item 3) — REFUTED; the claim holds.** Independent fixture
  `data/test/AllGestureWiimoteX_dist_50.csv` (N=50, k=3, unused by the
  implementer): all four CSVs `cmp`-identical to
  `build/highs-1151/bin/dtwc_cl.exe`. A hostile fixture (0.1+0.2, 1e21, 1e100,
  5e-324, 2.2250738585072014e-308, 1e-7, −0) also identical, including `1e+21`
  and `1e+100`. Direct differential test (clang++ `to_chars(general,
  max_digits10)` vs Python `:.17g`, 19 values incl. subnormals, −0, exponents
  ≥ 100): identical for all; NaN/inf are the only gap (F8).
* **Error taxonomy (item 4) — REFUTED.** MRO is
  `UndefinedScore → InvalidInput → DtwcError → ValueError`; the translator
  branch precedes `InvalidInput`; `except InvalidInput`/`except ValueError`
  still catch it; `save` swallows only `UndefinedScore` (`score("nope")` still
  raises `InvalidInput`).
* **Lazy fill (item 5) — REFUTED.** Second `score()` is 4.6× cheaper
  (4.1 → 0.9 ms); the band is configured before `set_data`; an `hpc` `Result`
  raises the contract message and `save()` writes labels+medoids only.
* **Ragged / skips (item 6) — REFUTED.** Per-row `skip_cols` erase, the exact
  C++ message, `skip_rows` clamping and 0-based in-memory names all match
  `dtwc/api.cpp:161-181`.
* **`device()` canonicalisation (item 7) — REFUTED.** `gpu`, `cuda:0`, `gpu:1`
  raise `DeviceError` with the verbatim `Env` text; `CUDA:0`/`GPU`/`HPC`
  case-fold; the `hpc` deferral matches contract §1.1.

## Other

`scripts/check_docs_contract.py` currently **fails** at
`generate_docs.py --check` (stale `docs/content/api/tier-1.md`). Attribution is
uncertain — another agent is editing `dtwc/` live — but it must be regenerated
green before merge.

## Verdict

**Fix first, then merge.** Blocking: F1 (hard crash on legitimate input,
regression vs the old path), F2 (one-line GIL fix restoring the module
convention), F3 (broken gate constant), F4 (pin drift the guard cements).
Same-PR desirable: F5, F6, and the F10 CHANGELOG line. F7 is a design note for
the scale target; F8/F9 can be logged. The core parity claims — byte identity,
error taxonomy, lazy fill, ragged/skip semantics and device errors — all
survived adversarial retesting on inputs the implementer did not use.
