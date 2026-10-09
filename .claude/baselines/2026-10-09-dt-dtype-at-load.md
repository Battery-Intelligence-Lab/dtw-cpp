# 2026-10-09 — DT: `dtype` where series are loaded; float32 read directly (peak 0.522× base, band ≤ 0.6×: PASS)

Unit DT of `plans/2026-10-09-dtype-and-blocked-access.md` (Volkan 10-08: "it should be easily changable like
DataLoader(d_type = f32) then data is loaded as f32"), from design-2.0 `7ab1f354` on `pb/DT`. Windows 11, Intel Core
Ultra 9 285, clang++ 21.1.8 (`build/` as the main checkout's; `build-arrow/` and the base `build-base-arrow/` as
`build/arrow-pyarrow-23`, Arrow 23.0.1 through the pyarrow shim), MATLAB R2024b (`build-mex/`). Shared machine:
memory and outputs only, nothing timed. Raw lines and the kit: `2026-10-09-dt-dtype-at-load/`.

**Questions.** Does every reader store float32 directly, so that a float32 load holds no double copy, with the
outputs of today's `--dtype float32`? Is float64 unchanged? Is G2 (a double view kept under float32) fixed?

**Registered band (the orchestrator's, before any run, verbatim):** "the peak working set of `dtwc_cl --dtype
float32` reading a Parquet file of at least 200 MB of float64 values is at most 0.6× base's."

## Results

- **Peak working set [confirmed]: 473.2 MB against base's 905.8 MB, 0.522× (4 runs each, within 0.03 %): PASS.**
  File: 240,337,908 bytes, 30,000 rows × 1,000 float64 values (240 MB), one row group (pyarrow 23 defaults). The
  run exits after the load (`-k 1000000`). Float64 loads fall too, 905.8 → 596.8 MB (0.659×). Base holds the mapped
  file, Arrow's decoded table of the whole file and the double series at once (peak private 990 MB); the candidate
  holds the mapped file (about 210 MB of its working set [inferred]: 473 − 262 MB private), the float32 series and
  one 16 MiB batch (peak private 262 MB; the process alone 10 MB). Rounding to float32 alone, with the whole file
  still decoded at once, would have kept the mapped file and the decoded table: about 0.83× [inferred], outside the
  band; hence the batches.
- **Float64 unchanged [confirmed]:** `w8b_cli_runs.sh` + `w8b_cli_extra.sh` against `rl_post_clang`: 156 of 156
  files identical (the four `--dtype float32` CSV-folder runs among them).
- **Float32 as today [confirmed]:** `f32_runs.sh`, base and pb/DT Arrow builds, 22 runs (CSV folder, batch CSV,
  Parquet list and rows layouts with and without skips, Arrow IPC; kmedoids, pam, clara `--seed 123`; float64
  controls): 154 files each, byte-identical but for the verbose line "Converted to float32", which a direct read no
  longer prints. The verbose memory line reports the stored width (~114 MB float32 against base's ~228 MB).
- **Parquet batches are invisible [confirmed]:** a throwaway build reading one row per batch passes test_io_readers
  (1544 assertions) and gives the outputs of the 16 MiB batches and of base (list and rows layouts, skip 0 and 7).
- **G2 [confirmed]:** a Float64 view passed to `dtwc::run` under Float32 is converted: its cost equals the owned
  float32 run's (3.45000004768371582); under base's condition it ran in double (3.45000000000000018).
- **Gates [confirmed]:** clang ctest 95 = 92 + 3 MAY_SKIP skips (no new warning: the three `getenv` warnings are
  base's); cpp_conformance passes; Arrow tree's five Arrow/Parquet/CLARA tests pass; pytest from a fresh venv
  895 / 19 skipped / 0 failed (893 + 2); matlab_suite 141 / 141 / 0 failed / 0 incomplete (140 + 1); check_docs,
  check_pins, generate_docs --check pass.

## How float32 output equals today's

Today's `--dtype float32` parses each value as double, then rounds it (`static_cast<float>`). Every reader now does
the same: text through `parse_numeric_field<float>`, which is the double parse rounded (so a text value is one float
in every language; the C++ test's witness 1 + 2^-24 + 1e-34 gives 1, where a float parse gives 1 + 2^-23); Parquet
through `copy_arrow_numeric<float>` and the C-Data ingest through `static_cast<float>` of the double (identity for
Float32 sources); pyarrow's float64→float32 cast is IEEE round to nearest (bitwise equal to NumPy's on 1,000,011
values of magnitude 1e-40 to 1e38, with FLT_MAX, the subnormals, -0, NaN and ±inf); MATLAB's `single()` likewise.

## Not honoured, and refused

- Arrow before 23: `read_all` takes `GetRecordBatchReader`'s Result form from Arrow 19 (the CPM fallback's version,
  where it appears and the `unique_ptr` Status form is deprecated) and that Status form before 19, as Arrow
  19.0.1's and 18.1.0's `parquet/arrow/reader.h` declare them [read, not compiled: only Arrow 23 here].
- A file that is one series (a scalar column) is decoded whole: its one series' double copy is held.
- Python and MATLAB still decode a Parquet file whole before the cast (pyarrow, `parquetread`): the double copy is
  one file's, not the dataset's.
- An unknown dtype is InvalidInput in every language (`dtwc:invalidArgument`); a non-string dtype is a Python
  TypeError; a dtype beside a Dataset (`load(ds, dtype=...)`, `cluster(ds, k, dtype=...)`, MATLAB likewise) is refused.
