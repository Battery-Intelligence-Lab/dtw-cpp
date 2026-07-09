# Arrow C Data interface ingest + llfio-optional core build (Task 5.7)

**Date:** 2026-07-09 · **Machine:** dev box (Win11, clang 18, Ninja). **Base
commit:** c9a7882 (Task 5.6). **Baseline gates:** C++ 95/95; Python 391 passed /
10 skipped.

## Two deliverables (5.7 required the first as a prerequisite)

### A. Core builds WITHOUT llfio (fixes non-negotiable #3, unblocks the wheel)

**Symptom.** The Python wheel could not build on this box (documented "deferred"
for Tasks 5.5/5.6). Root cause was NOT the wheel: `-DDTWC_ENABLE_LLFIO=OFF` — the
sandbox escape hatch — was itself broken. `dtwc/core/mmap_distance_matrix.hpp:43`
did `#include <llfio/v2.0/llfio.hpp>` **unconditionally**; `mmap_data_store.hpp`
likewise. The R3/0.12 work that added `DTWC_ENABLE_LLFIO` only verified a
`configure`-only run (Dependencies.cmake:157) — it never *compiled* a TU that
includes the mmap headers, so the missing include-guard went unnoticed. Core
therefore could not build without the optional llfio dep — a direct violation of
runbook non-negotiable #3.

**Fix.** Guard the llfio-touching parts behind the existing `DTWC_HAS_MMAP`
compile define (set PUBLIC on dtwc++ only when `llfio_hl` links):
- `mmap_distance_matrix.hpp`: keep `MmapDistanceMatrix` a COMPLETE type in both
  configs (so `Problem::distMat_t = std::variant<Dense, Mmap>` and every
  std::visit/std::get site compile unchanged). Only the 4 members that touch a
  mapped file (the `llfio::mapped_file_handle`, the file-creating ctor, `open()`,
  `sync()`) are `#ifdef`'d out; their `#ifndef` replacements **throw** — no
  silent degradation.
- `mmap_data_store.hpp`: whole class compiled out when llfio absent (it has no
  non-mmap fallback; every include site is already guarded).
- `dtwc_cl.cpp`: guard the `.dtws` load block (throws a clear "rebuild with
  llfio" error when built without it).

**Verification.**
- **No-llfio build [HARD] → CONFIRMED.** Fresh `build/nollfio`
  (`-DDTWC_ENABLE_LLFIO=OFF`) compiles + links `dtwc++`, `dtwc_cl.exe`,
  `dtwc_main.exe` clean (exit 0); CLI `--help` runs.
- **With-llfio no-regression [HARD] → CONFIRMED.** `build/highs-1151` rebuilt;
  full `ctest` **95/95 → 96/96** (the +1 is the new Arrow test suite; every
  pre-existing test digit-identical). The guards are logic-transparent when
  `DTWC_HAS_MMAP` is defined.

**OPEN (documented, not silently shipped).** The *Windows wheel with llfio ON*
still fails — a backslash bug in the existing quickcpplib ninja-propagation patch
(Dependencies.cmake): `-DCMAKE_MAKE_PROGRAM=C:\Users\...\ninja` reaches
quickcpplib's `ExternalProject_Add` CMAKE_ARGS with backslashes → CMake syntax
error. The local gating wheel is therefore built with `DTWC_ENABLE_LLFIO=OFF`
(mmap distance-matrix → clear runtime throw in Python). `pyproject.toml` is left
UNCHANGED (Linux/CI wheels keep llfio, which builds fine there). Proper fix =
`file(TO_CMAKE_PATH ...)` on `CMAKE_MAKE_PROGRAM` inside the patch. Left for a
follow-up; it does not block 5.7.

### B. Task 5.7 — Arrow C Data interface ingest (zero-copy, no pyarrow)

Vendored **nanoarrow 0.8.0** (Apache-2.0), symbol-namespaced
`NANOARROW_NAMESPACE=DtwcNanoarrow` to avoid ODR clashes with any other Arrow
library in-process. Two files: `dtwc/extern/nanoarrow/nanoarrow.{h,c}`, compiled
into `dtwc++` (PRIVATE include — never leaks to consumers). No build-time fetch
(unlike a CPM dep, so a sandboxed wheel build cannot fail on it).

New seam `dtwc::io::data_from_arrow` (io/arrow_c_data.{hpp,cpp}) reads an
(ArrowSchema, ArrowArray) pair — a `list`/`large_list` of float32/float64 (each
element → one univariate series), or a struct containing one (+ optional utf8
name child). `data_from_arrow_stream` consumes the `__arrow_c_stream__` batch
protocol. Public surface uses forward-declared C structs, so nanoarrow stays
encapsulated; the Python binding reaches it only through `data_from_arrow` /
`release_arrow`. Values are copied once into owning `Data::p_vec`; borrowed Arrow
buffers are never aliased (caller may release immediately). "Zero-copy" = the
buffer-direct read that avoids both pyarrow and the old per-element
`list(row) for row in X` Python-object materialisation — not in-place aliasing.

Python: `_dtwcpp_core.data_from_arrow_c_array(obj)` extracts the PyCapsule(s) from
`__arrow_c_array__` (pyarrow/DuckDB single array) OR `__arrow_c_stream__`
(polars/pandas batch stream) and returns a `Data`. Exported as
`dtwcpp.data_from_arrow_c_array`. `DTWClustering._prepare_data` auto-detects
either dunder and ingests without pyarrow.

## Registered bands (fixed BEFORE the runs) + verdicts

**C++ gate** `tests/unit/io/test_arrow_c_data.cpp` (producer = nanoarrow builder,
an independent code path from the reader): **7 cases / 75 assertions → all PASS.**
- large_list<double> round-trips to Data exactly (p_vec ==, names series_i,
  ndim 1); list<float32> converts to double exactly (values exactly
  representable); caller names override; empty list → empty Data; null series
  REJECTED (InvalidInput); non-floating (int64) child REJECTED; top-level
  non-list REJECTED.

**Python polars gate** (`scratchpad/polars_gate.py`, INDEPENDENT producer;
pyarrow forcibly blocked via `sys.modules['pyarrow']=None`):
- **G1 [HARD] → CONFIRMED.** `data_from_arrow_c_array(pl.Series List<f64>)`
  p_vec == input series, exact.
- **G4 [HARD] → CONFIRMED.** polars `List<f32>` → exact float64 values.
- **G3 [HARD] → CONFIRMED.** `DTWClustering(k=2).fit_predict(polars Series)`
  converges and recovers the 2 well-separated groups (labels {1,1,1} / {0,0,0}).
- **G2 [HARD] → CONFIRMED.** pyarrow is NEVER importable/imported across the
  whole ingest+cluster run — proves the "without pyarrow" requirement (not merely
  "pyarrow absent" — it is installed and actively blocked).

## Gates

- C++ `ctest` (build/highs-1151, llfio ON): **96/96 passed, 0 failed** (baseline
  95 → +1 suite `test_arrow_c_data`; 6 documented skips: CUDA×2, Metal×3,
  io_readers). No regression.
- No-llfio build (build/nollfio): compiles + links clean.
- Python `pytest tests/python`: **391 passed, 10 skipped** (unchanged from
  baseline — the Arrow path is additive).
- Side benefit: rebuilding the wheel unblocked the previously-deferred **5.5/5.6
  Python runtime parity** — `MVMode`, `DTWVariant.MSM/TWE` now present at runtime;
  `mv_mode="independent"` verified end-to-end (DTW_I ≤ DTW_D holds in Python).

## Honesty notes

- nanoarrow bundler emitted a mangled include `#include "nanoarrownanoarrow.h"`
  (from `--header-namespace nanoarrow` without a separator); normalised to
  `#include "nanoarrow/nanoarrow.h"` in the vendored .c (include dir = extern/).
- The DTWClustering fit convenience path materialises the ingested Data to numpy
  (its predict/cluster_centers_ logic needs per-series arrays). The TRUE
  zero-Python-copy ingest is `data_from_arrow_c_array → Problem.set_data(Data)`,
  exposed for the 100M-series pipeline; not wired into sklearn-style fit.
- TC-DTW LB (arXiv:2101.07731, plan "if time") still deferred — a pruning bound,
  orthogonal to this ingest deliverable.
